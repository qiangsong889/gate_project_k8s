"""同步流式笔记助手。运行：python agent-gpt.py。

历史只保存在内存；工具中断不代表其副作用已撤销。
/retry 继续未完成轮次；/max N 调整输出上限；exit 退出。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import inspect
import re
from threading import Lock
from typing import cast

import anthropic
from anthropic.types import (
    Message, MessageParam, ToolParam, ToolResultBlockParam,
    ToolChoiceAutoParam, ToolChoiceNoneParam,
)
from dotenv import load_dotenv


class Conversation:
    def __init__(
        self,
        keep_last: int | None = 3,
        max_tokens: int = 4096,
        prompt_version: str = "v3",
        model: str = "claude-sonnet-5",
        *,
        max_tool_rounds: int = 10,
        client: anthropic.Anthropic | None = None,
        system: str | None = None,
        tool_definitions: list[ToolParam] | None = None,
        tool_funcs: Mapping[str, Callable[..., str]] | None = None,
        prices_per_million: tuple[float, float] | None = None,
    ) -> None:
        self._check_nonnegative("keep_last", keep_last, allow_none=True)
        self._check_nonnegative("max_tool_rounds", max_tool_rounds)
        self.keep_last = keep_last
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.model = model
        self.prompt_version = prompt_version
        # 延迟加载本地工具，允许测试时不加载模型、不连接数据库。
        if tool_definitions is None or tool_funcs is None:
            from tools import tools, TOOL_FUNCS
            tool_definitions = tools if tool_definitions is None else tool_definitions
            tool_funcs = TOOL_FUNCS if tool_funcs is None else tool_funcs
        if system is None:
            from prompts import get_prompt
            system = get_prompt("knowledge_search", prompt_version).template
        load_dotenv()
        self.client = client if client is not None else anthropic.Anthropic(
            timeout=60.0, max_retries=2,
        )
        self.system = system
        self.tools = tool_definitions
        self.tool_funcs = dict(tool_funcs)
        self.prices_per_million = prices_per_million
        self.message_turns: list[list[MessageParam]] = []
        self._pending: list[MessageParam] | None = None
        self._tool_rounds = 0
        self._lock = Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.unpriced_requests = 0
        self.interrupted_streams = 0

    @staticmethod
    def _check_nonnegative(name: str, value: int | None, *, allow_none: bool = False) -> None:
        if allow_none and value is None:
            return
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} 必须是非负整数")

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        if type(value) is not int or value <= 0:
            raise ValueError("max_tokens 必须是正整数")
        self._max_tokens = value

    def _join_messages(self) -> list[MessageParam]:
        self._check_nonnegative("keep_last", self.keep_last, allow_none=True)
        turns = deepcopy(self.message_turns)
        # 当前待完成轮次不参与历史轮次的 keep_last 计数，也不省略结果。
        completed_count = len(turns) - int(self._pending is not None)
        cutoff = 0 if self.keep_last is None else max(0, completed_count - self.keep_last)
        for turn in turns[:cutoff]:
            for message in turn:
                content = message["content"]
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            cast(ToolResultBlockParam, block)["content"] = "内容已省略，需要时请重新搜索"
        return [message for turn in turns for message in turn]

    def _record_usage(self, final: Message) -> None:
        usage = final.usage
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        # 缓存计价与普通 input 不同；没有完整价格配置时不伪装成精确账单。
        cached = (getattr(usage, "cache_creation_input_tokens", 0) or 0) + (
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )
        if self.prices_per_million is None or cached:
            self.unpriced_requests += 1
            cost_text = "费用未计价"
        else:
            input_rate, output_rate = self.prices_per_million
            cost = (usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1_000_000
            self.total_cost += cost
            cost_text = f"估算=${cost:.6f} | 已计价累计=${self.total_cost:.6f}"
        print(f"\nin={usage.input_tokens} out={usage.output_tokens} | {cost_text}")

    def _request(self, *, allow_tools: bool) -> Message:
        complete = False
        choice: ToolChoiceAutoParam | ToolChoiceNoneParam = (
            {"type": "auto"} if allow_tools else {"type": "none"}
        )
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                messages=self._join_messages(),
                tools=self.tools,
                tool_choice=choice,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                final = stream.get_final_message()
                complete = True
        finally:
            if not complete:
                # 未取得完整 usage，不将中断请求当成零费用。
                self.interrupted_streams += 1
            print()
        return final

    def _execute_tools(self, final: Message, current: list[MessageParam]) -> None:
        calls = [block for block in final.content if block.type == "tool_use"]
        if not calls or len({call.id for call in calls}) != len(calls):
            raise RuntimeError("工具调用为空或 ID 重复，未保存这次回复")
        results: list[ToolResultBlockParam] = []
        try:
            for call in calls:
                result: ToolResultBlockParam = {"type": "tool_result", "tool_use_id": call.id}
                try:
                    func = self.tool_funcs.get(call.name)
                    if func is None:
                        raise ValueError(f"未知工具：{call.name}")
                    if not isinstance(call.input, dict):
                        raise ValueError("工具参数必须是对象")
                    # 当前笔记工具参数全部为字符串；拒绝 None、数组等错误输入。
                    if any(not isinstance(value, str) for value in call.input.values()):
                        raise ValueError("笔记工具参数必须是字符串")
                    inspect.signature(func).bind(**call.input)
                    output = func(**call.input)
                    if not isinstance(output, str):
                        raise TypeError("笔记工具必须返回字符串")
                    result["content"] = output
                except Exception as exc:
                    result["content"] = f"工具执行失败：{type(exc).__name__}: {exc}"
                    result["is_error"] = True
                results.append(result)
        except KeyboardInterrupt:
            # 保留已完成结果；为当前和未执行调用补错误结果，保证历史可继续发送。
            for index, call in enumerate(calls[len(results):]):
                results.append({
                    "type": "tool_result", "tool_use_id": call.id, "is_error": True,
                    "content": "用户中断；当前调用是否完成未知，重试前请核实。" if index == 0
                    else "用户中断；此工具尚未执行。",
                })
            current.extend([
                {"role": "assistant", "content": final.content},
                {"role": "user", "content": results},
            ])
            raise
        current.extend([
            {"role": "assistant", "content": final.content},
            {"role": "user", "content": results},
        ])

    def _run_pending(self) -> str:
        current = self._pending
        if current is None:
            raise RuntimeError("没有待继续的轮次")
        while True:
            allow_tools = self._tool_rounds < self.max_tool_rounds
            final = self._request(allow_tools=allow_tools)
            # 内容处理前先记录 usage，截断和异常停止同样消耗 token。
            self._record_usage(final)
            if final.stop_reason == "end_turn":
                reply = "".join(block.text for block in final.content if block.type == "text")
                if not reply or any(block.type == "tool_use" for block in final.content):
                    raise RuntimeError("最终回复为空或仍含工具调用；可使用 retry() 重试")
                current.append({"role": "assistant", "content": final.content})
                self._pending = None
                return reply
            if final.stop_reason == "max_tokens":
                raise RuntimeError("回复被截断；提高 max_tokens 后调用 retry()，此前结果已保留")
            if final.stop_reason != "tool_use":
                raise RuntimeError(f"未处理的停止原因：{final.stop_reason}；此前结果已保留")
            if not allow_tools:
                raise RuntimeError("工具预算已耗尽，模型仍要求工具；此次调用未执行")
            self._tool_rounds += 1
            self._execute_tools(final, current)

    def send(self, user_text: str) -> str:
        """开始新一轮；有未完成轮次时先 retry() 或显式 finish_pending()。"""
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("问题不能为空")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("同一 Conversation 不支持并发发送")
        try:
            if self._pending is not None:
                raise RuntimeError("上一轮尚未完成：用 retry() 继续，或 finish_pending() 后问新问题")
            self._pending = [{"role": "user", "content": user_text}]
            self.message_turns.append(self._pending)  # 唯一的历史注册点，失败也不重复追加。
            self._tool_rounds = 0
            return self._run_pending()
        finally:
            self._lock.release()

    def retry(self) -> str:
        """继续同一轮，不重复用户问题，也不自动重放已完成的工具。"""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("当前会话正在执行")
        try:
            return self._run_pending()
        finally:
            self._lock.release()

    def finish_pending(self) -> None:
        """停止继续本轮但保留其完整记录，允许发送新问题。"""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("当前会话正在执行")
        try:
            self._pending = None
        finally:
            self._lock.release()

    def close(self) -> None:
        self.client.close()


def main() -> None:
    conversation = Conversation()
    print("/retry 继续；/finish 保留记录并结束待完成轮次；/max N；exit 退出")
    try:
        while True:
            try:
                text = input("你: ").strip()
                if text == "exit":
                    break
                if not text:
                    continue
                if text == "/retry":
                    conversation.retry()
                elif text == "/finish":
                    conversation.finish_pending()
                elif match := re.fullmatch(r"/max\s+(\d+)", text):
                    conversation.max_tokens = int(match.group(1))
                else:
                    conversation.send(text)
            except KeyboardInterrupt:
                print("\n已中断；记录已保留，可输入 /retry 或 /finish")
            except EOFError:
                break
            except (anthropic.APIError, RuntimeError, ValueError) as exc:
                print(f"\n{exc}")
    finally:
        conversation.close()


if __name__ == "__main__":
    main()
