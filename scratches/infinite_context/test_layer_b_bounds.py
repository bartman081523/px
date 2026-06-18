"""
Layer B — bounded-context / invariant unit tests for InfLLMCache + ReAttention.
Uses a mock rotary; no real model / GPU required.
"""
import unittest
import torch
import torch.nn as nn
from inf_llm_cache import InfLLMCache, apply_reattention_patch, remove_reattention_patch


class MockConfig:
    def __init__(self, layers=4):
        self.num_hidden_layers = layers
        self.hidden_size = 64
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.head_dim = 8


class MockRotaryEmb(nn.Module):
    def forward(self, x, pos):
        B, T = pos.shape[0], pos.shape[1]
        return torch.ones(B, T, 8), torch.zeros(B, T, 8)


def _rand_kv(B, H, T, D, device="cpu"):
    return torch.randn(B, H, T, D), torch.randn(B, H, T, D), torch.randn(B, H, T, D)


class TestLayerBBounds(unittest.TestCase):
    def test_no_prefill_bypass_large_prefill_is_bounded(self):
        # The wip bug: prefill > 1024 returned full KV (N^2). Here it must be bounded.
        cfg = MockConfig()
        cache = InfLLMCache(cfg, block_size=64, r_tokens=4, top_k_blocks=4,
                           sinks_count=4, window_size=32, max_ltm_blocks=256)
        rotary = MockRotaryEmb()
        B, H, T, D = 1, 8, 2000, 8
        q, k, v = _rand_kv(B, H, T, D)
        q_rot, k_rot, v_out = cache.prepare_reattention(q, k, v, 0, rotary)
        bound = cache.sinks_count + cache.top_k_blocks * cache.block_size + cache.window_size
        self.assertLessEqual(k_rot.size(-2), bound)
        self.assertLess(k_rot.size(-2), T)  # NOT the full prefill
        self.assertEqual(q_rot.size(-2), T)  # queries preserved

    def test_usable_length_constant_across_history_growth(self):
        cfg = MockConfig()
        cache = InfLLMCache(cfg, block_size=64, r_tokens=4, top_k_blocks=4,
                           sinks_count=4, window_size=32, max_ltm_blocks=256)
        rotary = MockRotaryEmb()
        B, H, D = 1, 8, 8
        bound = cache.sinks_count + cache.top_k_blocks * cache.block_size + cache.window_size
        lengths = []
        for step in range(50):  # 50 incremental tokens
            q, k, v = _rand_kv(B, H, 1, D)
            cache.prepare_reattention(q, k, v, 0, rotary)
            lengths.append(cache.get_usable_length(0))
        # Usable length never exceeds the bound, regardless of 50-token history.
        self.assertTrue(all(l <= bound for l in lengths))
        # And it is far below the raw seen-token count.
        self.assertLess(max(lengths), cache.seen_tokens)

    def test_ltm_eviction_cap(self):
        cfg = MockConfig()
        cache = InfLLMCache(cfg, block_size=64, r_tokens=4, top_k_blocks=4,
                           sinks_count=4, window_size=32, max_ltm_blocks=4)
        rotary = MockRotaryEmb()
        B, H, D = 1, 8, 8
        for _ in range(10):  # 10 blocks -> eviction must cap at 4
            q, k, v = _rand_kv(B, H, 64, D)
            cache.prepare_reattention(q, k, v, 0, rotary)
        self.assertLessEqual(len(cache.ltm_k[0]), 4)

    def test_seq_length_tracks_real_total(self):
        cfg = MockConfig()
        cache = InfLLMCache(cfg, block_size=64, r_tokens=4, top_k_blocks=4,
                           sinks_count=4, window_size=32, max_ltm_blocks=256)
        rotary = MockRotaryEmb()
        B, H, D = 1, 8, 8
        for n in (10, 20, 30):
            q, k, v = _rand_kv(B, H, n, D)
            cache.prepare_reattention(q, k, v, 0, rotary)
        self.assertEqual(cache.get_seq_length(0), 60)  # real total, not bounded

    def test_representative_keys_on_cpu(self):
        cfg = MockConfig()
        cache = InfLLMCache(cfg, block_size=64, r_tokens=4, top_k_blocks=4,
                           sinks_count=4, window_size=32, max_ltm_blocks=256)
        rotary = MockRotaryEmb()
        q, k, v = _rand_kv(1, 8, 128, 8)
        cache.prepare_reattention(q, k, v, 0, rotary)
        self.assertEqual(cache.ltm_rk[0][0].device.type, "cpu")
        self.assertEqual(cache.ltm_k[0][0].device.type, "cpu")

    def test_apply_and_remove_patch(self):
        class FakeAttn(nn.Module):
            _layer_idx = 0
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)
                self.o_proj = nn.Linear(8, 8)
                self.q_norm = nn.Identity()
                self.k_norm = nn.Identity()
                self.rotary_emb = MockRotaryEmb()
                self.layer_idx = 0
                self.scaling = 1.0
                self.sliding_window = None
                self.attention_dropout = 0.0
                self.training = False
                self.head_dim = 8
                class _Cfg: _attn_implementation = "eager"
                self.config = _Cfg()
            def forward(self, *a, **k):
                return "orig"
        attn = FakeAttn()
        # Make named_modules pick it up by class name containing "Gemma3Attention"
        FakeAttn.__name__ = "Gemma3Attention"
        model = nn.Module()
        model.add_module("attn", attn)
        n = apply_reattention_patch(model)
        self.assertGreaterEqual(n, 1)
        self.assertTrue(getattr(attn, "_px_reattention_patched", False))
        remove_reattention_patch(model)
        self.assertFalse(getattr(attn, "_px_reattention_patched", False))


if __name__ == "__main__":
    unittest.main()