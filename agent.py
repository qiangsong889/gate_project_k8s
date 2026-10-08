import anthropic
import psycopg2
import anthropic
from collections.abc import Generator
from typing import Any
import logging

from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from copy import deepcopy
from typing import ClassVar
from dotenv import load_dotenv
from prompts import get_prompt
from tools import tools, TOOL_FUNCS
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("kb_assistant")

class Conversation:
    MODEL_PRICING: ClassVar[dict[str, dict[str, float]]] = {
        "gpt-6-astra": {
            "input": 10.00,
            "output": 50.00,
        },
        "gpt-5.6-sol": {
            "input": 4.00,
            "output": 20.00,
        },
        "gpt-5.6-terra": {
            "input": 2.00,
            "output": 12.00,
        },
        "gpt-5.6-luna": {
            "input": 0.20,
            "output": 1.20,
        },
        "claude-sonnet-5": {"input": 2.0, "output": 10.0},   # $/1M tokens
        "claude-haiku-4-5-20251001": {"input": 0.8, "output": 4.0},
    }
    def __init__(
        self,
        keep_last:  int | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "v1",
        model: str = "claude-sonnet-5",
    ) -> None:
        load_dotenv()  # 自动从 ANTHROPIC_API_KEY 读 key
        self.model = model
        self.max_tokens = max_tokens
        self.prompt_version = prompt_version
        self.client = anthropic.Anthropic()
        self.total_cost: float = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_read_input_tokens: int = 0
        self.total_cache_creation_input_tokens: int = 0
        self.keep_last: int | None = keep_last
        self.message_turns: list[list[MessageParam]] = []
        self.messages_send_to_ai: list[MessageParam] = []
    
    def _omit_old_tool_results(
        self,
    ) -> list[list[MessageParam]]:
        if self.keep_last is None:
            return self.message_turns
        if not isinstance(self.keep_last, int):
            raise ValueError("keep_last mube be int")
        if self.keep_last < 0:
            raise ValueError("keep_last 不能小于 0")
        
        result = deepcopy(self.message_turns)
        cutoff = max(0, len(result) - self.keep_last)

        for turn in result[:cutoff]:
            for message in turn:
                content = message.get("content")

                if not isinstance(content, list):
                    continue

                for block in content:
                    if (isinstance(block, dict) and block["type"] == "tool_result"):
                        block["content"] = "内容已省略，需要时请重新搜索"

        return result
    
    
    def _join_messages(self, current_messages) -> list[MessageParam]:
        return [message for turn in self._omit_old_tool_results() for message in turn] + current_messages
        
    def _calcualte_cost(self, input_token: int, output_token: int, cache_read_input_tokens: int | None = None, cache_creation_input_tokens: int | None = None) -> float:
        model_price = self.MODEL_PRICING.get(self.model)
        if model_price is None:
            raise ValueError(f"没有找到模型价格 {self.model}")
        cache_cost = 0.0
        if cache_read_input_tokens is not None:
            cache_cost += cache_read_input_tokens * model_price['input'] * 0.1
        if cache_creation_input_tokens is not None:
            cache_cost += cache_creation_input_tokens * model_price['input'] * 1.25
            
        return (input_token * model_price['input'] + output_token * model_price['output'] + cache_cost)/1_000_000
    
    def _assistant_message(self, content) -> MessageParam:
        return {"role": "assistant", "content": content}
        
    def _user_message(self, content) -> MessageParam:
        return { "role": "user", "content": content }
    
    def _set_max_token(self, max_tokens):
        self.max_tokens = max_tokens
        print(f"max_tokens set to {max_tokens}\n")
        
    def close(self) -> None:
        self.client.close()
        
    def send_stream(
        self,
        user_text: str,
    ) -> Generator[dict[str, Any], None, None]:
        current_messages: list[MessageParam] = [
            self._user_message(user_text)
        ]
        for _ in range(10):
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=get_prompt(
                    "knowledge_search",
                    self.prompt_version,
                ).template,
                messages=self._join_messages(current_messages),
                tools=tools,
                cache_control={"type": "ephemeral"},
            ) as stream:
                for text in stream.text_stream:
                    yield {
                        "type": "text",
                        "delta": text,
                    }

                final = stream.get_final_message()

            # 原来的 _record_usage() 会打印，所以这里只更新统计。
            usage = final.usage
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens
            if usage.cache_read_input_tokens is not None:
                self.total_cache_read_input_tokens += usage.cache_read_input_tokens
            if usage.cache_creation_input_tokens is not None:
                self.total_cache_creation_input_tokens += usage.cache_creation_input_tokens

            try:
                cost = self._calcualte_cost(
                    input_token=usage.input_tokens,
                    output_token=usage.output_tokens,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                    cache_creation_input_tokens=usage.cache_creation_input_tokens
                )
            except ValueError:
                cost = None

            if cost is not None:
                self.total_cost += cost

            if final.stop_reason == "end_turn":
                current_messages.append(
                    self._assistant_message(final.content)
                )

                answer = "".join(
                    block.text
                    for block in final.content
                    if block.type == "text"
                )

                # 在发出 done 前保存，确保调用方收到时历史已更新。
                self.message_turns.append(current_messages)
                return_messages_saved = True

                yield {
                    "type": "usage",
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read": usage.cache_read_input_tokens,       # 这次从缓存读了多少
                    "cache_write": usage.cache_creation_input_tokens,  # 这次写进缓存多少
                    "cost": cost,
                    "total_cost": self.total_cost,
                }
                
                yield {
                    "type": "done",
                    "answer": answer,
                }
                return
            if final.stop_reason == "max_tokens":
                raise RuntimeError(
                    "回复被截断，请提高 max_tokens 后重试"
                )

            if final.stop_reason != "tool_use":
                raise RuntimeError(
                    f"未处理的停止原因：{final.stop_reason}"
                )

            calls = [
                block
                for block in final.content
                if block.type == "tool_use"
            ]

            if not calls:
                raise RuntimeError(
                    "模型要求调用工具，但没有返回工具调用"
                )

            results: list[ToolResultBlockParam] = []

            try:
                for block in calls:
                    yield {
                        "type": "tool",
                        "tool_use_id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }

                    result: ToolResultBlockParam = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                    }

                    func = TOOL_FUNCS.get(block.name)

                    try:
                        if func is None:
                            raise ValueError(
                                f"未知工具：{block.name}"
                            )

                        result["content"] = func(**block.input)

                    except Exception as exc:
                        result["content"] = f"检索服务暂时不可用"
                        result["is_error"] = True
                        logger.exception("数据库操作失败: %s", exc)
                        # raise RuntimeError("工具执行失败")
                    results.append(result)
                    yield {
                        "type": "tool_done",
                        "tool_use_id": block.id,
                        "name": block.name,
                        "is_error": bool(result.get("is_error", False)),
                    }
                    
            finally:
                current_messages.extend([
                    self._assistant_message(final.content),
                    self._user_message(results),
                ])

        raise RuntimeError(
            "达到最大调用次数，仍未生成最终回答"
        )

            
    def send(self, user_text: str) -> str:
        for event in self.send_stream(user_text=user_text):
            if event["type"] == "done":
                return event["answer"]
        raise RuntimeError("没有收到完整回答")

    # def send(self, user_text: str) -> str:
        
        # return
        # current_messages: list[MessageParam] = []
        # current_messages.append(self._user_message(user_text))
        # try:
        #     for i in range(10):
        #         print(f"请求次数: {i + 1}\n")
        #         self.messages_send_to_ai = self._join_messages(current_messages)
        #         with self.client.messages.stream(
        #             model=self.model,
        #             max_tokens=self.max_tokens,
        #             system=get_prompt("knowledge_search", self.prompt_version).template,
        #             messages=self._join_messages(current_messages),
        #             tools=tools
        #         ) as stream:
        #             for text in stream.text_stream:
        #                 print(text, end="", flush=True)   # flush=True每收到一点就打印一点 end=""同一行
        #             final = stream.get_final_message()

        #         print()
        #         try:
        #             self._record_usage(
        #                 input_tokens=final.usage.input_tokens,
        #                 output_tokens=final.usage.output_tokens,
        #             )
        #         except ValueError as e:
        #             print(e)
                
                
        #         if final.stop_reason == "end_turn":
        #             current_messages.append(self._assistant_message(final.content))
        #             self.message_turns.append(current_messages)
        #             return "".join(b.text for b in final.content if b.type == "text")

        #         if final.stop_reason == "max_tokens":
        #             # current_messages.append(self._assistant_message(final.content))
        #             self.message_turns.append(current_messages)
        #             raise RuntimeError("回复被截断，请提高 max_tokens 后重试")
                

        #         if final.stop_reason != "tool_use":
        #             # current_messages.append(self._assistant_message(final.content))
        #             self.message_turns.append(current_messages)
        #             raise RuntimeError(f"未处理的停止原因：{final.stop_reason}")
                
        #         results: list[ToolResultBlockParam] = []
        #         for b in final.content:
        #             if b.type == "tool_use":
        #                 print(f"REAL TOOL_USE: {b.name}, PARAMS: {b.input}\n")
        #                 tool_result: ToolResultBlockParam = {
        #                     "type": "tool_result",
        #                     "tool_use_id": b.id,
        #                 }
        #                 func = TOOL_FUNCS.get(b.name)
        #                 if func is None:
        #                     tool_result["is_error"] = True
        #                     tool_result["content"] = f"Unknown tool name: {b.name}"
        #                 else:
        #                     try:
        #                         tool_result["content"] = func(**b.input)
        #                     except Exception as e:
        #                         tool_result["content"] = f"工具执行失败：{e}"
        #                         tool_result["is_error"] = True
                        
        #                 results.append(tool_result)

        #         if not results:
        #             self.message_turns.append(current_messages)
        #             raise RuntimeError("模型要求调用工具，但没有返回工具调用")
                
        #         current_messages.append(self._assistant_message(final.content))
        #         current_messages.append(self._user_message(results))  
            
        #     self.message_turns.append(current_messages)
        #     raise RuntimeError("达到最大调用次数，仍未生成最终回答")
       
        # except (KeyboardInterrupt, anthropic.APIError):
        #     self.message_turns.append(current_messages)
        #     raise
        # finally:
        #     print()
            

if __name__ == "__main__":
    conversation = Conversation(prompt_version="v3", keep_last=3)
    # for event in conversation.send_stream("etcd 是干什么的"):
    #     print(event)
    try:
        while True:
            try:
                print(f'当前对话轮：{len(conversation.message_turns)} \n')
                user_text = input("你: ")
                if user_text == "exit":
                    break
                elif not user_text.strip():
                    continue
                elif match := re.fullmatch(r"/max\s+(\d+)", user_text.strip()):
                    if match is not None:
                        conversation._set_max_token(int(match.group(1)))
                        continue
                print("助手：", end="", flush=True)
                # conversation.send(user_text)
                for event in conversation.send_stream(user_text):
                    if event["type"] == "text":
                        print(event["delta"], end="", flush=True)
                    elif event["type"] == "tool":
                        print(f"\n[调用工具: {event['name']}]")
                print()
            except anthropic.APIConnectionError as e:
                print(f"\n请求失败 {e}")
            except anthropic.APIError as e:
                print(f"\n 消息有问题 {e}")
            except KeyboardInterrupt:
                print(f"用户中断")
                break
            except ValueError as e:
                print(f"数据格式有问题 {e}")
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
    finally:
        conversation.client.close()
