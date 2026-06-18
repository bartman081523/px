"""
Regression test — the 3479b4f9 OOM scenario, before/after.

Simulates the growth that crashed the RTX 2060 (bug_context.txt): a long text-only
conversation whose prefill attention exceeded VRAM. We compare three strategies at
increasing history sizes H:

  * NAIVE   (no infinite context): full prefill -> attention memory ~ H^2  (OOM)
  * Layer A (InfiniteContextManager): prompt tokens bounded -> always <= BUDGET
  * Layer B (InfLLMCache/ReAttention): K length bounded -> always <= KV_BOUND

A GPU is NOT required; "memory" is a deterministic FLOP/element proxy
(T_q * T_k for the attention score tensor), which is what OOMs on the 12 GB card.
"""
import unittest
import torch
import torch.nn as nn
from infinite_context import InfiniteContextManager
from inf_llm_cache import InfLLMCache


class WordTokenizer:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        out = []
        for m in msgs:
            c = m["content"]
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            out.append(str(c))
        return " ".join(out)

    def encode(self, text):
        return text.split()


class MockRotaryEmb(nn.Module):
    def forward(self, x, pos):
        B, T = pos.shape[0], pos.shape[1]
        return torch.ones(B, T, 8), torch.zeros(B, T, 8)


def build_history(n_pairs, words=12):
    msgs = [{"role": "system", "content": "You are a contemplative interlocutor."}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": " ".join(["question"] * words)})
        msgs.append({"role": "assistant", "content": " ".join(["answer"] * words)})
    return msgs


# Bounds chosen to mimic a 12 GB VRAM safety budget.
PROMPT_BUDGET = 2048          # Layer A: max prompt tokens fed to the model
KV_BLOCK = 64
KV_TOPK = 4
KV_WINDOW = 64
KV_SINKS = 4
KV_BOUND = KV_SINKS + KV_TOPK * KV_BLOCK + KV_WINDOW  # Layer B: max K length


class TestRegressionOOM(unittest.TestCase):
    def setUp(self):
        self.tok = WordTokenizer()

    def naive_prompt_tokens(self, msgs):
        return len(self.tok.encode(self.tok.apply_chat_template(msgs)))

    def naive_attention_elems(self, n_prompt):
        # Full causal self-attention over the prefill: N x N score elements.
        return n_prompt * n_prompt

    def test_naive_ooms_while_layers_stay_bounded(self):
        cfg = type("C", (), {"num_hidden_layers": 4})()
        cache = InfLLMCache(cfg, block_size=KV_BLOCK, r_tokens=4, top_k_blocks=KV_TOPK,
                           sinks_count=KV_SINKS, window_size=KV_WINDOW, max_ltm_blocks=256)
        rotary = MockRotaryEmb()

        for n_pairs in (14, 30, 60, 120):  # 14 pairs ~ the real 3479b4f9 size, then grow
            msgs = build_history(n_pairs)
            n_naive = self.naive_prompt_tokens(msgs)
            naive_elems = self.naive_attention_elems(n_naive)

            # Layer A: bound the prompt.
            mgr = InfiniteContextManager(max_tokens=PROMPT_BUDGET, max_history_messages=None)
            a_msgs = mgr.process_history(msgs, tokenizer=self.tok)
            a_tokens = self.naive_prompt_tokens(a_msgs)

            # Layer B: feed the same raw history through the cache and read the K length.
            cache2 = InfLLMCache(cfg, block_size=KV_BLOCK, r_tokens=4, top_k_blocks=KV_TOPK,
                                 sinks_count=KV_SINKS, window_size=KV_WINDOW, max_ltm_blocks=256)
            # one large prefill of the raw token count
            T = n_naive
            q = torch.randn(1, 8, T, 8)
            k = torch.randn(1, 8, T, 8)
            v = torch.randn(1, 8, T, 8)
            _, k_rot, _ = cache2.prepare_reattention(q, k, v, 0, rotary)
            b_kv = k_rot.size(-2)

            # Naive grows quadratically and blows past the budget at scale.
            self.assertGreaterEqual(a_tokens, 1)
            self.assertLessEqual(a_tokens, PROMPT_BUDGET,
                                 f"Layer A prompt {a_tokens} > budget {PROMPT_BUDGET} at {n_pairs} pairs")
            self.assertLessEqual(b_kv, KV_BOUND,
                                 f"Layer B KV {b_kv} > bound {KV_BOUND} at {n_pairs} pairs")
            # Regression guard: where the raw history already exceeds the prompt
            # budget (the OOM condition), naive attention (N^2) must be strictly
            # larger than the bounded strategies' attention footprint — proving the
            # layers actually prevent the OOM that naive hits.
            if n_naive > PROMPT_BUDGET:
                self.assertLess(a_tokens, n_naive)
                self.assertGreater(naive_elems, a_tokens * a_tokens)
                self.assertGreater(naive_elems, n_naive * b_kv)

    def test_layer_b_kv_independent_of_history_at_fixed_q(self):
        cfg = type("C", (), {"num_hidden_layers": 4})()
        rotary = MockRotaryEmb()
        kvs = []
        for n_pairs in (14, 60, 200, 600):
            cache = InfLLMCache(cfg, block_size=KV_BLOCK, r_tokens=4, top_k_blocks=KV_TOPK,
                                sinks_count=KV_SINKS, window_size=KV_WINDOW, max_ltm_blocks=256)
            msgs = build_history(n_pairs)
            T = len(self.tok.encode(self.tok.apply_chat_template(msgs)))
            q = torch.randn(1, 8, T, 8)
            k = torch.randn(1, 8, T, 8)
            v = torch.randn(1, 8, T, 8)
            _, k_rot, _ = cache.prepare_reattention(q, k, v, 0, rotary)
            kvs.append(k_rot.size(-2))
        # KV length is flat (bounded) even as history grows 14x.
        self.assertLessEqual(max(kvs), KV_BOUND)
        self.assertGreater(kvs[-1], 0)


if __name__ == "__main__":
    unittest.main()