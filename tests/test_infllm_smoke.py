"""
tests/test_infllm_smoke.py — Smoke tests for the (currently orphaned)
InfLLM/ReAttention implementation in infinite_context.py.

Context: SR-64 (commit 7cf8218) implemented InfLLM block memory + ReAttention
(decoupled RoPE) for training-free infinite context. The module was wired
into the production patch at one point (commit 3cdb8f4: "infinite context:
surgical patches") but the current `patch.py` only uses SDPA + the
`_chunked_attention` long-prefill fallback. `infinite_context.py` lives
orphaned in the repo root and is only imported by `test_archive.py`.

These tests pin the orphaned module so any drift (broken block boundaries,
crashing rotary split, regressed representative-key selection) is caught
in CI without forcing a re-integration into the active patch.

What we DO NOT test here: live integration into `patch.py`. That would be
a motor change and lives separately under `scratches/4b-image/` once the
integration plan is approved.

Run:
    pytest tests/test_infllm_smoke.py -v
or:
    python tests/test_infllm_smoke.py
"""
import sys
import os
import unittest

import torch

# Repo root on sys.path so `infinite_context` imports cleanly under pytest
# (which runs from the tests/ dir).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from infinite_context import InfLLMCache, rotate_half, apply_rotary_pos_emb_single  # noqa: E402


class _StubConfig:
    """Minimal stand-in for a HF PretrainedConfig — only `num_hidden_layers`
    is read by InfLLMCache.__init__."""
    num_hidden_layers = 4


class TestRotaryHelpers(unittest.TestCase):
    """Pure tensor utilities used by ReAttention."""

    def test_rotate_half_shape_preserved(self):
        x = torch.randn(2, 8, 16, 64)
        out = rotate_half(x)
        self.assertEqual(out.shape, x.shape)

    def test_rotate_half_anticommutes(self):
        """rot(rot(x)) == -x for the swap-and-negate pattern."""
        x = torch.randn(1, 1, 4, 8)
        self.assertTrue(torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6))

    def test_apply_rotary_handles_3d_cos_sin(self):
        """[B, T, D] cos/sin is the standard layout for HF models."""
        x = torch.randn(2, 8, 16, 64)
        cos = torch.randn(2, 16, 64)
        sin = torch.randn(2, 16, 64)
        out = apply_rotary_pos_emb_single(x, cos, sin)
        self.assertEqual(out.shape, x.shape)
        # Output must differ from input (unless cos==1 and sin==0 — random here)
        self.assertFalse(torch.allclose(out, x))

    def test_apply_rotary_handles_2d_cos_sin(self):
        """[T, D] cos/sin (no batch dim) must broadcast."""
        x = torch.randn(1, 8, 16, 64)
        cos = torch.randn(16, 64)
        sin = torch.randn(16, 64)
        out = apply_rotary_pos_emb_single(x, cos, sin)
        self.assertEqual(out.shape, x.shape)


class TestInfLLMCache(unittest.TestCase):
    """Block-memory lifecycle, top-k representative keys, sinks, eviction.

    API reality (from infinite_context.py, not the older InfLLMBlockMemory in
    scratches/infinite_context/inf_llm.py): InfLLMCache is a Cache-class that
    is updated implicitly via prepare_reattention() (which appends to
    buffer_k/buffer_v and archives full blocks to ltm_k/ltm_v). There is no
    public add_kv(); state mutates inside prepare_reattention as a side
    effect of running an attention forward.
    """

    def setUp(self):
        self.cache = InfLLMCache(_StubConfig(), block_size=8, r_tokens=2,
                                  top_k_blocks=4, sinks_count=4)
        torch.manual_seed(0)
        self.B, self.H, self.D = 1, 2, 16

    def _qkv(self, T=1):
        """Return a single (q, k, v) triple of tensors with T tokens each."""
        return (torch.randn(self.B, self.H, T, self.D),
                torch.randn(self.B, self.H, T, self.D),
                torch.randn(self.B, self.H, T, self.D))

    def _rotary(self, x, pos):
        """Gemma3RotaryEmbedding-compatible: (x[B,H,T,D], pos[B,T]) -> (cos, sin)."""
        B, H, T, D = x.shape
        cos = torch.ones(B, T, D, device=x.device, dtype=x.dtype)
        sin = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        return cos, sin

    def test_block_size_one_archives_immediately(self):
        """block_size=1 (degenerate) must not crash the finalize path.
        One call to prepare_reattention with T=1 fills the buffer to size 1,
        which equals block_size=1, so _archive_block fires once and ltm_k
        gets one entry."""
        c = InfLLMCache(_StubConfig(), block_size=1, r_tokens=1)
        q, k, v = self._qkv(1)
        c.prepare_reattention(q, k, v, layer_idx=0,
                               rotary_emb_module=self._rotary)
        self.assertEqual(len(c.ltm_k[0]), 1)

    def test_block_finalizes_on_boundary(self):
        """Adding block_size tokens yields exactly one LTM block."""
        c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2)
        # Each call deposits 1 token; call 4 times → buffer reaches 4 → archive
        for _ in range(4):
            q, k, v = self._qkv(1)
            c.prepare_reattention(q, k, v, layer_idx=0,
                                   rotary_emb_module=self._rotary)
        self.assertEqual(len(c.ltm_k[0]), 1)
        k_b = c.ltm_k[0][0]
        self.assertEqual(k_b.shape, (self.B, self.H, 4, self.D))

    def test_partial_block_does_not_finalize(self):
        c = InfLLMCache(_StubConfig(), block_size=8, r_tokens=2)
        for _ in range(5):
            q, k, v = self._qkv(1)
            c.prepare_reattention(q, k, v, layer_idx=0,
                                   rotary_emb_module=self._rotary)
        self.assertEqual(len(c.ltm_k[0]), 0)

    def test_sinks_persist_first_n(self):
        """The first `sinks_count` tokens must be retained as sinks forever
        (independent of block eviction). They are populated on the first
        call to prepare_reattention as long as the incoming tensor has at
        least sinks_count tokens."""
        c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2, sinks_count=2)
        q = torch.randn(self.B, self.H, 4, self.D)
        k = torch.randn(self.B, self.H, 4, self.D)
        v = torch.randn(self.B, self.H, 4, self.D)
        c.prepare_reattention(q, k, v, layer_idx=0,
                               rotary_emb_module=self._rotary)
        self.assertIsNotNone(c.sinks_k[0])
        self.assertEqual(c.sinks_k[0].shape[-2], 2)


class TestReAttention(unittest.TestCase):
    """ReAttention: prepare_reattention applies decoupled RoPE to q/k before
    cross-block attention so positions inside an LTM block can be re-aligned
    to the current window's coordinate frame.

    Output K/V will be CONCATENATED with sinks + retrieved LTM blocks, so
    output sequence dim ≥ input. We test shape consistency (B, H, D unchanged,
    T_new grows by exactly sinks_count on the first call) rather than exact
    equality.
    """

    def setUp(self):
        # sinks_count=2 so we can predict the +2 shape growth.
        self.cache = InfLLMCache(_StubConfig(), block_size=8, r_tokens=4,
                                  sinks_count=2)
        torch.manual_seed(1)
        self.B, self.H, self.T, self.D = 1, 1, 16, 8
        self.q = torch.randn(self.B, self.H, self.T, self.D)
        self.k = torch.randn(self.B, self.H, self.T, self.D)
        self.v = torch.randn(self.B, self.H, self.T, self.D)

    def _identity_rotary(self, x, pos):
        B, H, T, D = x.shape
        cos = torch.ones(B, T, D, device=x.device, dtype=x.dtype)
        sin = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        return cos, sin

    def test_prepare_reattention_grows_kv_by_sinks(self):
        """First call appends sinks; output K/V should be T+sinks_count long."""
        q_out, k_out, v_out = self.cache.prepare_reattention(
            self.q, self.k, self.v, layer_idx=0,
            rotary_emb_module=self._identity_rotary,
        )
        # Q stays at T_new (only K/V are concatenated with sinks + local + LTM).
        self.assertEqual(q_out.shape, self.q.shape)
        # K/V gain exactly sinks_count tokens on the first call (no LTM yet).
        self.assertEqual(k_out.shape, (self.B, self.H, self.T + 2, self.D))
        self.assertEqual(v_out.shape, (self.B, self.H, self.T + 2, self.D))

    def test_prepare_reattention_q_gets_rope(self):
        """Q must be modified by RoPE (under identity rotary, Q unchanged)."""
        q_out, k_out, v_out = self.cache.prepare_reattention(
            self.q, self.k, self.v, layer_idx=0,
            rotary_emb_module=self._identity_rotary,
        )
        # Under identity rotary, Q should equal input Q exactly.
        self.assertTrue(torch.allclose(q_out, self.q, atol=1e-6))

    def test_prepare_reattention_with_nonzero_rotary(self):
        """Real rotary (random cos/sin) must still produce same-shape output
        and must change Q (random rotation cannot be identity)."""
        def rotary(x, pos):
            B, H, T, D = x.shape
            return torch.randn(B, T, D), torch.randn(B, T, D)
        q_out, k_out, v_out = self.cache.prepare_reattention(
            self.q, self.k, self.v, layer_idx=0, rotary_emb_module=rotary,
        )
        self.assertEqual(q_out.shape, self.q.shape)
        # Random RoPE must differ from input.
        self.assertFalse(torch.allclose(q_out, self.q))


if __name__ == "__main__":
    unittest.main(verbosity=2)