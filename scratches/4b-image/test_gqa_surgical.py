"""
scratches/4b-image/test_gqa_surgical.py — TDD-red/green für GQA-Fix
==================================================================

Pinnt: chunked_attention_gqa (mit Fix) liefert für 270m/1b/4b
cos_sim >= 0.999 gegen F.scaled_dot_product_attention(enable_gqa=True).

TDD-Zyklus:
  Rot:  entferne expand_kv_for_gqa aus chunked_attention_gqa → 4b crasht
        (genau wie im Motor), 270m/1b passen (1-broadcast-Workaround).
  Grün: mit expand_kv_for_gqa → alle 4 Tests passen (2:1 GQA wie 4b, 4:1 wie
        270m/1b, lange Prefill, kurze Prefill).

Run:
    /path/to/venv/bin/python scratches/4b-image/test_gqa_surgical.py
"""
import math
import sys
import torch
import torch.nn.functional as F

# Standalone — keine Motor-Imports.
from gqa_repeat import expand_kv_for_gqa, chunked_attention_gqa  # noqa: E402


GQA_CONFIGS = [
    # (name, Hq, Hkv, head_dim, sliding_window)
    ("270m", 4, 1, 256, None),
    ("1b",   4, 1, 256, 512),
    ("4b",   8, 4, 256, 1024),
]


def _cos_sim(a, b):
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _full_mask_with_window(Tq, Tk, sliding_window, device):
    qpos = torch.arange(Tq, device=device)
    kpos = torch.arange(Tk, device=device)
    causal = kpos[None, :] <= qpos[:, None]
    window = kpos[None, :] >= (qpos[:, None] - sliding_window + 1)
    return causal & window


def _reference(q, k, v, scaling, sliding_window):
    if sliding_window is None:
        return F.scaled_dot_product_attention(
            q, k, v, scale=scaling, is_causal=True, enable_gqa=True,
        )
    return F.scaled_dot_product_attention(
        q, k, v, scale=scaling, enable_gqa=True,
        attn_mask=_full_mask_with_window(q.shape[-2], k.shape[-2], sliding_window, q.device),
    )


def _check(model_name, Hq, Hkv, head_dim, sliding_window, T, B=1, dtype=torch.float32):
    torch.manual_seed(0)
    q = torch.randn(B, Hq,  T, head_dim, dtype=dtype)
    k = torch.randn(B, Hkv, T, head_dim, dtype=dtype)
    v = torch.randn(B, Hkv, T, head_dim, dtype=dtype)
    scaling = 1.0 / math.sqrt(head_dim)

    ref = _reference(q, k, v, scaling, sliding_window)
    out = chunked_attention_gqa(q, k, v, scaling, sliding_window=sliding_window)

    assert out.shape == ref.shape, f"{model_name}: shape {out.shape} != ref {ref.shape}"
    sim = _cos_sim(out, ref)
    assert sim >= 0.999, f"{model_name} T={T}: cos_sim={sim} < 0.999"
    print(f"[OK] {model_name:5s}  Hq={Hq} Hkv={Hkv} T={T:5d}  cos_sim={sim:.6f}")


def test_expand_kv_no_op():
    """n_rep=1 muss idempotent sein (keine Allokation/Kopie)."""
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    k2, v2 = expand_kv_for_gqa(k, v, n_rep=1)
    assert k2 is k and v2 is v, "n_rep=1 sollte exakt dieselben Tensoren zurückgeben"


def test_expand_kv_2x():
    """n_rep=2 expandiert Hkv korrekt: 4 → 8 Heads, Werte sind interleaved-repeats."""
    k = torch.randn(1, 4, 2, 3)
    v = torch.randn(1, 4, 2, 3)
    k_e, v_e = expand_kv_for_gqa(k, v, n_rep=2)
    assert k_e.shape == (1, 8, 2, 3), f"erwartet (1,8,2,3), got {k_e.shape}"
    # repeat_interleave: [k0, k0, k1, k1, k2, k2, k3, k3]
    expected = torch.repeat_interleave(k, 2, dim=1)
    assert torch.equal(k_e, expected), "n_rep=2 falsch expandiert"


def test_gqa_all_configs():
    for name, Hq, Hkv, hd, sw in GQA_CONFIGS:
        _check(name, Hq, Hkv, hd, sw, T=64)


def test_gqa_4b_long_prefill():
    """Der eigentliche OOM-Auslöser: lange Prefill mit 4b-Config (T=2048)."""
    _check("4b-long", Hq=8, Hkv=4, head_dim=256, sliding_window=1024, T=2048)


if __name__ == "__main__":
    tests = [
        ("expand_kv no-op",          test_expand_kv_no_op),
        ("expand_kv 2x",             test_expand_kv_2x),
        ("GQA alle Konfigurationen", test_gqa_all_configs),
        ("4b lange Prefill (T=2048)", test_gqa_4b_long_prefill),
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
