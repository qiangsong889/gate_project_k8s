"""离线测试：python -m unittest discover -s gate_project_k8s -p test_agent_gpt.py"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from anthropic.types import Message

spec = importlib.util.spec_from_file_location("agent_gpt", Path(__file__).with_name("agent-gpt.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Conversation = module.Conversation


def response(reason="end_turn", calls=()):
    content = [dict(type="tool_use", id=f"call_{i}", name=name, input=args)
               for i, (name, args) in enumerate(calls)] or [dict(type="text", text="answer")]
    return Message(id="msg", type="message", role="assistant", model="test",
                   content=content, stop_reason=reason,
                   usage={"input_tokens": 10, "output_tokens": 5})


class Stream:
    def __init__(self, item):
        self.item = item
        self.text_stream = iter(())
    def __enter__(self):
        if isinstance(self.item, BaseException):
            raise self.item
        return self
    def __exit__(self, *args):
        return False
    def get_final_message(self):
        return self.item


class Tests(unittest.TestCase):
    def make(self, responses, funcs=None, **kwargs):
        client = Mock()
        client.messages.stream.side_effect = [Stream(item) for item in responses]
        conv = Conversation(client=client, system="test", tool_definitions=[],
                            tool_funcs=funcs or {}, prices_per_million=(1, 2), **kwargs)
        return conv, client

    def test_tool_pair_and_final(self):
        func = Mock(return_value="note")
        conv, client = self.make([response("tool_use", [("read", {"source": "a"})]), response()], {"read": func})
        self.assertEqual(conv.send("question"), "answer")
        self.assertEqual(len(conv.message_turns), 1)
        self.assertEqual([m["role"] for m in conv.message_turns[0]], ["user", "assistant", "user", "assistant"])
        self.assertEqual(conv.total_input_tokens, 20)
        self.assertEqual(conv.total_output_tokens, 10)
        self.assertAlmostEqual(conv.total_cost, .00004)
        func.assert_called_once_with(source="a")

    def test_truncation_retry_does_not_duplicate_or_replay(self):
        func = Mock(return_value="note")
        conv, _ = self.make([response("tool_use", [("read", {})]), response("max_tokens"), response()], {"read": func})
        with self.assertRaisesRegex(RuntimeError, "截断"):
            conv.send("question")
        self.assertEqual(len(conv.message_turns[0]), 3)
        with self.assertRaisesRegex(RuntimeError, "上一轮"):
            conv.send("duplicate")
        conv.max_tokens = 8192
        conv.retry()
        self.assertEqual(len(conv.message_turns), 1)
        func.assert_called_once()

    def test_network_failure_retry(self):
        import anthropic
        import httpx
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.com"))
        conv, _ = self.make([error, response()])
        with self.assertRaises(anthropic.APIConnectionError):
            conv.send("question")
        conv.retry()
        self.assertEqual(len(conv.message_turns), 1)
        self.assertEqual(conv.interrupted_streams, 1)

    def test_interrupt_preserves_partial_batch_and_pairs_every_call(self):
        first = Mock(return_value="completed")
        second = Mock(side_effect=KeyboardInterrupt)
        third = Mock(return_value="never")
        conv, _ = self.make([response("tool_use", [("a", {}), ("b", {}), ("c", {})]), response()],
                            {"a": first, "b": second, "c": third})
        with self.assertRaises(KeyboardInterrupt):
            conv.send("question")
        results = conv.message_turns[0][-1]["content"]
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["content"], "completed")
        self.assertTrue(results[1]["is_error"])
        self.assertTrue(results[2]["is_error"])
        third.assert_not_called()
        conv.retry()
        first.assert_called_once()
        second.assert_called_once()

    def test_budget_reserves_final_request(self):
        conv, client = self.make([response("tool_use", [("missing", {})]), response()], max_tool_rounds=1)
        conv.send("question")
        self.assertEqual(client.messages.stream.call_args.kwargs["tool_choice"], {"type": "none"})
        self.assertTrue(conv.message_turns[0][2]["content"][0]["is_error"])

    def test_omit_only_old_results_without_mutating_history(self):
        conv, _ = self.make([], keep_last=0)
        conv.message_turns = [[{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "id", "content": "original"}]}]]
        joined = conv._join_messages()
        self.assertIn("省略", joined[0]["content"][0]["content"])
        self.assertEqual(conv.message_turns[0][0]["content"][0]["content"], "original")

    def test_bad_tool_arguments_are_error_results(self):
        func = Mock(return_value="no")
        conv, _ = self.make([response("tool_use", [("read", {"source": 12})]), response()], {"read": func})
        conv.send("question")
        func.assert_not_called()
        self.assertTrue(conv.message_turns[0][2]["content"][0]["is_error"])

    def test_finish_keeps_failed_turn(self):
        conv, _ = self.make([response("max_tokens"), response()])
        with self.assertRaises(RuntimeError):
            conv.send("old")
        conv.finish_pending()
        conv.send("new")
        self.assertEqual(len(conv.message_turns), 2)


if __name__ == "__main__":
    unittest.main()
