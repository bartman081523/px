"""
Layer A — budget / preservation unit tests for InfiniteContextManager.
Pure Python, no GPU, no torch.
"""
import unittest
from infinite_context import InfiniteContextManager, DEFAULT_ARCHIVE_NOTICE


class WordTokenizer:
    """Mock tokenizer: 1 word == 1 token. Accepts the optional add_generation_prompt
    kwarg so it works with the manager's call site."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in msgs:
            c = m["content"]
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            parts.append(str(c))
        return " ".join(parts)

    def encode(self, text):
        return text.split()


def _hist(n_pairs, words=5):
    msgs = [{"role": "system", "content": "sys prompt"}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": " ".join(["w"] * words)})
        msgs.append({"role": "assistant", "content": " ".join(["a"] * words)})
    return msgs


class TestLayerABudget(unittest.TestCase):
    def setUp(self):
        self.tok = WordTokenizer()

    def _tok_count(self, msgs):
        return len(self.tok.encode(self.tok.apply_chat_template(msgs)))

    def test_budget_guarantee_large_history(self):
        # 60 pairs -> ~600 tokens; cap at 120.
        msgs = _hist(60, words=5)
        mgr = InfiniteContextManager(max_tokens=120, max_history_messages=None)
        out = mgr.process_history(msgs, tokenizer=self.tok)
        self.assertLessEqual(self._tok_count(out), 120)
        # System preserved + archive notice present (we truncated).
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[0]["content"], "sys prompt")
        self.assertTrue(any("archived" in m["content"] for m in out if m["role"] == "system"))

    def test_headroom_subtracts(self):
        msgs = _hist(40, words=5)
        # max_tokens=200, headroom=50 -> effective budget 150
        mgr = InfiniteContextManager(max_tokens=200, max_history_messages=None, headroom=50)
        out = mgr.process_history(msgs, tokenizer=self.tok)
        self.assertLessEqual(self._tok_count(out), 200)
        # And it must respect the tighter effective budget (it stops at <=150).
        self.assertLessEqual(self._tok_count(out), 150)

    def test_system_always_preserved_even_when_huge(self):
        msgs = [{"role": "system", "content": " ".join(["s"] * 1000)}]
        msgs += [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        mgr = InfiniteContextManager(max_tokens=10, max_history_messages=None)
        out = mgr.process_history(msgs, tokenizer=self.tok)
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[0]["content"], " ".join(["s"] * 1000))

    def test_image_content_passthrough(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "image": "data:image/png;base64,XYZ"},
            ]},
        ]
        mgr = InfiniteContextManager(max_history_messages=10)
        out = mgr.process_history(msgs)  # no tokenizer -> message-count path
        # The multimodal content list must survive untouched.
        self.assertEqual(out[-1]["content"][1]["type"], "image")
        self.assertEqual(out[-1]["content"][1]["image"], "data:image/png;base64,XYZ")

    def test_custom_archive_notice(self):
        msgs = _hist(10, words=5)
        mgr = InfiniteContextManager(
            max_history_messages=2, archive_notice="[memory compressed]"
        )
        out = mgr.process_history(msgs)
        self.assertIn("[memory compressed]", [m["content"] for m in out if m["role"] == "system"])

    def test_no_truncation_when_within_budget(self):
        msgs = _hist(3, words=3)  # small
        mgr = InfiniteContextManager(max_tokens=10000, max_history_messages=None)
        out = mgr.process_history(msgs, tokenizer=self.tok)
        self.assertEqual(len(out), len(msgs))  # unchanged
        self.assertNotIn("archived", "".join(m["content"] for m in out))


if __name__ == "__main__":
    unittest.main()