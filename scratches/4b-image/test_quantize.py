"""
test_quantize.py — TDD-rot/green für Quantize-Helper (int8, per-channel-symmetric)
====================================================================================

Hintergrund: 4b-Modell in bf16 braucht ~8 GB Gewichte. Auf RTX 2060 12 GB kein
Headroom für Lang-Prefill (siehe GQA-Smoke vom 23. Jun, 202 MB MLP-Allokation
OOM trotz 215 MB freien VRAMs). int8-Quantisierung reduziert Gewichte auf ~4.5 GB.

Was diese Tests pinnen:
  - quantize_state_dict(weights, scheme="int8") liefert
      {"weight": int8 [out,in], "scale": float [out]}
  - dequantize_state_dict(quantized) liefert float-Tensoren mit
      ||dequant(quant(x)) - x|| / ||x|| < 0.05 (5% relativ Error)
  - storage_bytes(quantized) ≈ raw_storage_bytes / 2 (8-bit vs 16-bit)
  - per-channel-scales: jede Output-Row hat eigenen scale
  - dtype-Preserve: dequant von bf16-input ist bf16-output
  - scheme="none" ist no-op (idempotent für nicht-quantisierte Pfade)

TDD-Zyklus:
  Rot:  noch keine quantize.py → ImportError → alle Tests FAIL
  Grün: quantize.py implementiert → alle Tests PASS

Run:
    /path/to/venv/bin/python test_quantize.py
"""
import math
import sys
import numpy as np
import torch


# Scheme constants — referenced in quantize.py + tests for consistency.
SCHEME_NONE = "none"
SCHEME_INT8 = "int8"


# Reusable test fixtures ----------------------------------------------------

def _linear_weight(out=64, in_=128, seed=0):
    """A typical nn.Linear weight matrix [out_features, in_features]."""
    torch.manual_seed(seed)
    return torch.randn(out, in_, dtype=torch.float32) * 0.5


# Tests ---------------------------------------------------------------------

def test_quantize_int8_roundtrip_small_error():
    """Round-Trip-Fehler < 5% relativ. Standard-Erwartung für int8."""
    from quantize import quantize_state_dict, dequantize_state_dict

    w = _linear_weight(out=64, in_=128)
    q = quantize_state_dict({"weight": w}, scheme=SCHEME_INT8)
    w_back = dequantize_state_dict(q)["weight"]

    rel_err = (w_back - w).norm() / w.norm()
    assert rel_err < 0.05, f"int8 roundtrip rel_err={rel_err:.4f} > 0.05"
    print(f"[OK] int8 roundtrip rel_err={rel_err:.4f}")


def test_quantize_int8_compresses_2x():
    """int8-Speicher ≈ raw_bf16 / 2. Per-Channel scales kommen dazu, aber
    sind klein (1 float pro Output-Channel)."""
    from quantize import quantize_state_dict, storage_bytes

    w = _linear_weight(out=64, in_=128)
    raw_bytes = w.numel() * w.element_size()  # 64*128*4 = 32768 bytes (fp32)
    q = quantize_state_dict({"weight": w}, scheme=SCHEME_INT8)
    q_bytes = storage_bytes(q)

    # int8 weight: 64*128*1 = 8192 bytes. Scale: 64*4 = 256 bytes. Total = 8448.
    # 32768 / 8448 ≈ 3.88x reduction (vs 2x für fp32 → fp16).
    # Wir testen nur, dass q_bytes < raw_bytes / 2 (strenger als 3.88x).
    assert q_bytes < raw_bytes / 2, f"q_bytes={q_bytes} not < raw_bytes/2={raw_bytes/2}"
    print(f"[OK] int8 compresses 2x+ ({raw_bytes} → {q_bytes}, factor {raw_bytes/q_bytes:.2f}x)")


def test_quantize_per_channel_scales():
    """Jede Output-Row hat eigenen scale. weight.shape[0] == scale.shape[0].

    Scale-Key-Konvention: <original_name>_scale (Suffix-Pattern), damit
    dequantize_state_dict ohne separaten Index das Pendant findet.
    """
    from quantize import quantize_state_dict

    w = _linear_weight(out=64, in_=128)
    q = quantize_state_dict({"weight": w}, scheme=SCHEME_INT8)

    assert "weight" in q and "weight_scale" in q, f"missing keys in {list(q.keys())}"
    assert q["weight"].dtype == torch.int8, f"weight dtype {q['weight'].dtype} != int8"
    assert q["weight_scale"].shape == (64,), f"scale shape {q['weight_scale'].shape} != (64,)"
    assert q["weight_scale"].dtype == torch.float32, f"scale dtype {q['weight_scale'].dtype} != fp32"
    print(f"[OK] per-channel scales (key='weight_scale', shape={q['weight_scale'].shape}, dtype={q['weight_scale'].dtype})")


def test_dequant_preserves_dtype():
    """bf16-Input → dequant → bf16-Output. Verhindert stille dtype-Drift."""
    from quantize import quantize_state_dict, dequantize_state_dict

    w = torch.randn(32, 64, dtype=torch.bfloat16)
    q = quantize_state_dict({"weight": w}, scheme=SCHEME_INT8)
    w_back = dequantize_state_dict(q)["weight"]
    assert w_back.dtype == torch.bfloat16, f"dequant dtype {w_back.dtype} != bf16"
    print(f"[OK] dequant dtype preserved (bf16 → bf16)")


def test_quantize_none_is_noop():
    """scheme='none' muss exakt dieselben Tensoren zurückgeben (keine Kopie,
    keine Allokation). Für 270m-Pfad (Default-Quantization='none')."""
    from quantize import quantize_state_dict

    w = _linear_weight()
    q = quantize_state_dict({"weight": w}, scheme=SCHEME_NONE)
    # Bei no-op: quantisierter Tensor ist der Original-Tensor (keine Kopie).
    assert q["weight"] is w, f"scheme='none' returned different tensor (copy!)"
    assert "scale" not in q, f"scheme='none' should not produce scale, got keys {list(q.keys())}"
    print(f"[OK] scheme='none' is idempotent no-op")


def test_quantize_int8_handles_outliers():
    """Werte mit großen Ausreißern dürfen nicht saturieren ohne dass
    per-Channel scaling sie auffängt. Pro Output-Channel: scale = max_abs/127."""
    from quantize import quantize_state_dict, dequantize_state_dict

    # Eine Zeile hat Ausreißer (max_abs=10), andere klein (max_abs=0.1).
    w = torch.zeros(4, 8, dtype=torch.float32)
    w[0] = torch.randn(8) * 10.0  # huge
    w[1] = torch.randn(8) * 0.001  # tiny
    w[2] = torch.randn(8) * 1.0    # normal
    w[3] = torch.randn(8) * 1.0    # normal

    q = quantize_state_dict({"weight": w}, scheme=SCHEME_INT8)
    w_back = dequantize_state_dict(q)["weight"]

    # Per-channel means each row's max_abs separately. Reasonable error per row.
    for row in range(4):
        rel_err_row = (w_back[row] - w[row]).norm() / w[row].norm()
        # Per-channel int8 hat ~1/127 ≈ 0.8% Error pro Element, summiert
        # auf unter 5% relativ pro Row.
        assert rel_err_row < 0.05, f"row {row} rel_err={rel_err_row:.4f} > 0.05"
    print(f"[OK] int8 handles outliers via per-channel scales")


if __name__ == "__main__":
    tests = [
        ("int8 roundtrip < 5% error",  test_quantize_int8_roundtrip_small_error),
        ("int8 compresses 2x",         test_quantize_int8_compresses_2x),
        ("per-channel scales",         test_quantize_per_channel_scales),
        ("dequant preserves dtype",    test_dequant_preserves_dtype),
        ("scheme='none' is noop",      test_quantize_none_is_noop),
        ("int8 handles outliers",      test_quantize_int8_handles_outliers),
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
