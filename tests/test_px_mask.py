"""Tests for the standalone _px_causal_mask helper (Plan 6.3).

Background: transformers ≥ 4.43 changed the masking API. PX-Patches
were written for ≤ 4.42 (commit bd4ec952 / 2026-06-06) and crashed
with ``TypeError: create_causal_mask() got an unexpected keyword
argument 'inputs_embeds'``. The fix is _px_causal_mask — a ~15 line
version-independent replacement. These tests verify the mask shape
and semantics without needing a model or GPU.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import torch
from px_patches.gemma3_270m_px_baseline.patch import _px_causal_mask


class _FakePast:
    """Minimal stand-in for DynamicCache with get_seq_length()."""
    def __init__(self, seq_len: int):
        self._seq_len = seq_len

    def get_seq_length(self) -> int:
        return self._seq_len


def test_causal_mask_shape_no_past():
    """Mask shape is [1, 1, T_q, T_k=T_q] when no past."""
    x = torch.zeros(1, 5, 64)
    m = _px_causal_mask(x, None)
    assert m.shape == (1, 1, 5, 5), m.shape
    assert m.dtype == torch.bool


def test_causal_mask_is_lower_triangular():
    """Without sliding window, mask is causal: True iff k <= q."""
    x = torch.zeros(1, 4, 8)
    m = _px_causal_mask(x, None)[0, 0]  # strip batch+head dims
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(m, expected), f"got {m}"


def test_sliding_window_masks_out_old_tokens():
    """With sliding_window=2, only the last 2 keys are attended per query."""
    x = torch.zeros(1, 4, 8)
    m = _px_causal_mask(x, None, sliding_window=2)[0, 0]
    # query 0 (global idx 0): can only see k=0
    assert m[0, 0].item() is True, "q0 should attend to k0"
    # query 1 (global idx 1): sees k=0,1 but sliding-window only keeps last 2 (k=0..1)
    # Actually q_idx[1]=1, k_idx[0]=0 → 0 >= 1-2+1=0 → True (k=0 kept by window)
    # and 0 <= 1 → True → True. So k=0 IS kept.
    # query 1 should NOT see k=0 if sliding_window=2 strictly excludes older ones.
    # Re-check formula: k_idx >= q_idx - sliding_window + 1 = 1 - 2 + 1 = 0
    # So k=0 (idx 0) >= 0 → True. Window of size 2 from q=1 keeps k={0,1}.
    # Correct: sliding_window=2 means [q-1, q] = 2 keys. q=0 keeps {0}, q=1 keeps {0,1}.
    # Verify:
    assert m[0, 0].item() is True
    assert m[1, 0].item() is True   # k=0 in window
    assert m[1, 1].item() is True   # k=1 = self
    # query 3 (idx 3): window [q-1, q] = [2, 3] → k=0,1 should be masked
    assert m[3, 0].item() is False, f"k=0 out of window for q=3, got {m[3, 0]}"
    assert m[3, 1].item() is False, f"k=1 out of window for q=3, got {m[3, 1]}"
    assert m[3, 2].item() is True   # k=2 in window
    assert m[3, 3].item() is True   # k=3 = self


def test_sliding_window_zero_or_none_means_no_window():
    """sliding_window=0 or None → no window restriction (full causal)."""
    x = torch.zeros(1, 4, 8)
    m_none = _px_causal_mask(x, None, sliding_window=None)[0, 0]
    m_zero = _px_causal_mask(x, None, sliding_window=0)[0, 0]
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(m_none, expected)
    assert torch.equal(m_zero, expected)


def test_past_key_values_extends_context():
    """With past_key_values (seq_len=3), T_k = 3 + T_q and global indices
    are shifted by past_len so causal masking spans both past and current."""
    x = torch.zeros(1, 2, 8)
    fake_past = _FakePast(seq_len=3)
    m = _px_causal_mask(x, fake_past)
    assert m.shape == (1, 1, 2, 5), f"expected T_k=5, got {m.shape}"
    # query 0 (global idx 3): can see k=0..3 (causal + past)
    for k in range(4):
        assert m[0, 0, 0, k].item() is True, f"q0 should see k={k}"
    # query 1 (global idx 4): sees k=0..4
    for k in range(5):
        assert m[0, 0, 1, k].item() is True, f"q1 should see k={k}"


def test_causal_mask_device_consistency():
    """Mask is created on the same device as input_embeds."""
    # CPU only (CI test); device consistency check works on any device.
    x = torch.zeros(1, 3, 4)
    m = _px_causal_mask(x, None)
    assert m.device == x.device


# ── Runner ────────────────────────────────────────────────────────────────

def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
