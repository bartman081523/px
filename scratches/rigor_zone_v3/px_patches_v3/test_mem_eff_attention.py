"""
test_mem_eff_attention.py — Regression tests for _chunked_attention GQA support
================================================================================

Background: gemma3 4b uses Grouped Query Attention (GQA) with Hq=8, Hkv=4 (ratio 2:1).
The current _chunked_attention does `torch.matmul(q, k.transpose(-1,-2))` which
treats Hq and Hkv as batch dims. PyTorch's matmul broadcasts only when one dim is 1,
so Hq=8 vs Hkv=4 crashes with "tensor a (8) must match tensor b (4)".

270m/1b use 4:1 GQA (Hq=4, Hkv=1) — works by accident because 1 broadcasts.

These tests compare _chunked_attention against the reference
`F.scaled_dot_product_attention(..., enable_gqa=True)` for the actual Gemma3 model
configs. cos_sim >= 0.999 required (the PX engine promises lossless long-prefill).

Run from repo root:
    pytest px_patches/gemma3_270m_px_baseline/test_mem_eff_attention.py -v
or:
    python px_patches/gemma3_270m_px_baseline/test_mem_eff_attention.py

Status (TDD red, before fix):
  - 270m/1b: should PASS (1-broadcast works)
  - 4b: should FAIL (8 vs 4 broadcast violation)
"""
import math
import os
import sys
import torch
import torch.nn.functional as F

# Allow running as `python path/to/test_mem_eff_attention.py` from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _reference_attention(q, k, v, scaling, sliding_window=None):
    """Ground truth: SDPA with built-in GQA. PyTorch >= 2.5 has enable_gqa."""
    is_causal = sliding_window is None
    return F.scaled_dot_product_attention(
        q, k, v,
        scale=scaling,
        is_causal=is_causal,
        enable_gqa=True,
    )


def _full_causal_mask(Tq, Tk, device):
    # True where attention is allowed
    qpos = torch.arange(Tq, device=device)
    kpos = torch.arange(Tk, device=device)
    return kpos[None, :] <= qpos[:, None]


def _full_mask_with_window(Tq, Tk, sliding_window, device):
    qpos = torch.arange(Tq, device=device)
    kpos = torch.arange(Tk, device=device)
    causal = kpos[None, :] <= qpos[:, None]
    window = kpos[None, :] >= (qpos[:, None] - sliding_window + 1)
    return causal & window


# Real Gemma3 model configs (from AutoConfig.from_pretrained)
GQA_CONFIGS = [
    # (name, Hq, Hkv, head_dim, sliding_window)
    ("gemma-3-270m-it", 4, 1, 256, None),
    ("gemma-3-1b-it",   4, 1, 256, 512),
    ("gemma-3-4b-it",   8, 4, 256, 1024),
]


def _cos_sim(a, b):
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _run_one(model_name, Hq, Hkv, head_dim, sliding_window, T, B=1, dtype=torch.float32):
    """Compare _chunked_attention vs SDPA-GQA reference for one config."""
    # Import the function under test from the patch module
    from px_patches.gemma3_270m_px_baseline.patch import _chunked_attention

    torch.manual_seed(0)
    device = "cpu"  # regression: pure-tensor test, no GPU needed

    q = torch.randn(B, Hq,  T, head_dim, device=device, dtype=dtype)
    k = torch.randn(B, Hkv, T, head_dim, device=device, dtype=dtype)
    v = torch.randn(B, Hkv, T, head_dim, device=device, dtype=dtype)
    scaling = 1.0 / math.sqrt(head_dim)

    # Reference: SDPA-GQA with explicit causal mask when sliding window is set
    if sliding_window is None:
        ref = _reference_attention(q, k, v, scaling)
    else:
        ref = F.scaled_dot_product_attention(
            q, k, v, scale=scaling, enable_gqa=True,
            attn_mask=_full_mask_with_window(T, T, sliding_window, device),
        )

    out = _chunked_attention(q, k, v, scaling, sliding_window=sliding_window)

    sim = _cos_sim(out, ref)
    return sim, out.shape, ref.shape


def test_gqa_270m():
    sim, sout, sref = _run_one("gemma-3-270m-it", Hq=4, Hkv=1, head_dim=256,
                                sliding_window=None, T=64)
    assert sref == sout, f"shape mismatch ref={sref} got={sout}"
    assert sim >= 0.999, f"270m cos_sim={sim} < 0.999"
    print(f"[OK] 270m  Hq=4 Hkv=1  cos_sim={sim:.6f}")


def test_gqa_1b():
    sim, sout, sref = _run_one("gemma-3-1b-it", Hq=4, Hkv=1, head_dim=256,
                                sliding_window=512, T=128)
    assert sref == sout, f"shape mismatch ref={sref} got={sout}"
    assert sim >= 0.999, f"1b cos_sim={sim} < 0.999"
    print(f"[OK] 1b    Hq=4 Hkv=1  sliding=512  cos_sim={sim:.6f}")


def test_gqa_4b():
    """THIS IS THE RED TEST: 4b uses Hq=8 Hkv=4 (2:1 GQA). Currently crashes."""
    sim, sout, sref = _run_one("gemma-3-4b-it", Hq=8, Hkv=4, head_dim=256,
                                sliding_window=1024, T=128)
    assert sref == sout, f"shape mismatch ref={sref} got={sout}"
    assert sim >= 0.999, f"4b cos_sim={sim} < 0.999"
    print(f"[OK] 4b    Hq=8 Hkv=4  sliding=1024  cos_sim={sim:.6f}")


def test_gqa_4b_long_prefill():
    """The actual OOM scenario: long T. Hits MEM_EFF_THRESHOLD chunked path."""
    sim, sout, sref = _run_one("gemma-3-4b-it", Hq=8, Hkv=4, head_dim=256,
                                sliding_window=1024, T=2048)
    assert sref == sout, f"shape mismatch ref={sref} got={sout}"
    assert sim >= 0.999, f"4b-long cos_sim={sim} < 0.999"
    print(f"[OK] 4b-long T=2048  Hq=8 Hkv=4  cos_sim={sim:.6f}")


if __name__ == "__main__":
    tests = [
        ("270m (4:1, no sliding)", test_gqa_270m),
        ("1b (4:1, sliding 512)", test_gqa_1b),
        ("4b (2:1, sliding 1024)", test_gqa_4b),
        ("4b-long (2:1, sliding 1024, T=2048)", test_gqa_4b_long_prefill),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)