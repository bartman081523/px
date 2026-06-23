"""
test_quantized_linear.py — TDD-rot/green für QuantizedLinear-Monkeypatch
=========================================================================

Hintergrund: Phase B von Plan 1. quantize.py quantisiert nur State-Dicts;
ein nn.Linear speichert seine Gewicht-Matrix in `self.weight` und macht
`F.linear(x, self.weight, self.bias)`. Wir brauchen einen Drop-in-Ersatz
für nn.Linear, der:
  1. int8-Gewicht + per-Channel-Scale als storage hält
  2. im forward() dequantisiert (oder matmul auf int8 mit Scale-Korrektur)
  3. gleiche Output-Shape liefert wie das Original
  4. via Monkeypatch in ein bestehendes Model eingehängt werden kann

Wichtig (epistemisches Mandat): alle Tests pinnen VERHALTEN, nicht
Implementierungs-Details. Wenn jemand die Dequantisierung durch einen
schnelleren Pfad ersetzt (z.B. cuBLAS int8 GEMM), müssen die Tests
stabil bleiben.

Tests:
  - QuantizedLinear als Drop-in: forward(x) hat gleiche Shape wie nn.Linear
  - cos_sim vs bf16-Referenz >= 0.95 (int8 ist nicht verlustfrei, aber nahe dran)
  - Storage: int8 weight + fp32 scale ist ~4x kleiner als bf16 equivalent
  - Monkeypatch via setattr(model, name, QuantizedLinear(...)) funktioniert
  - Mit Bias korrekt addiert
  - State-Dict-Round-Trip (save/load) funktioniert

Run:
    /path/to/venv/bin/python test_quantized_linear.py
"""
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper: bf16-Referenz-Linearschicht ---------------------------------------

def _reference_linear(in_=128, out=64, seed=0, bias=True, dtype=torch.bfloat16):
    """Erzeugt ein nn.Linear mit reproduzierbaren Gewichten."""
    torch.manual_seed(seed)
    layer = nn.Linear(in_, out, bias=bias)
    layer.weight.data = torch.randn_like(layer.weight.data) * 0.1
    if bias:
        layer.bias.data = torch.randn_like(layer.bias.data) * 0.05
    return layer.to(dtype)


# Tests ---------------------------------------------------------------------

def test_quantized_linear_output_shape_matches():
    """QuantizedLinear.forward(x) hat gleiche Shape wie nn.Linear.forward(x)."""
    from quantized_linear import quantize_linear

    layer = _reference_linear(in_=128, out=64, bias=True)
    q_layer = quantize_linear(layer)

    x = torch.randn(2, 5, 128, dtype=torch.bfloat16)
    y_ref = layer(x)
    y_q = q_layer(x)
    assert y_q.shape == y_ref.shape, f"shape mismatch {y_q.shape} != {y_ref.shape}"
    print(f"[OK] output shape matches ({y_q.shape})")


def test_quantized_linear_cos_sim_vs_bf16():
    """cos_sim zwischen quantisiertem und Original-Output >= 0.95.

    Warum 0.95: int8 mit per-Channel-symmetric hat ~1% Error pro Element.
    Auf 64 Output-Dims summiert sich das auf cos_sim > 0.99 in der Praxis.
    0.95 lässt Headroom für ungewöhnliche Input-Distributionen."""
    from quantized_linear import quantize_linear

    torch.manual_seed(0)
    layer = _reference_linear(in_=128, out=64, bias=True)
    q_layer = quantize_linear(layer)

    # Multiple inputs, deterministisch.
    sims = []
    for i in range(8):
        x = torch.randn(1, 16, 128, dtype=torch.bfloat16)
        y_ref = layer(x).float().flatten()
        y_q = q_layer(x).float().flatten()
        sim = F.cosine_similarity(y_ref.unsqueeze(0), y_q.unsqueeze(0)).item()
        sims.append(sim)
    mean_sim = sum(sims) / len(sims)
    assert mean_sim >= 0.95, f"cos_sim={mean_sim:.4f} < 0.95 (sims={sims})"
    print(f"[OK] cos_sim vs bf16: mean={mean_sim:.4f} (8 prompts, min={min(sims):.4f})")


def test_quantized_linear_uses_less_storage():
    """int8 + scale braucht weniger Speicher als bf16 equivalent.

    bf16: out*in*2 bytes
    int8 + scale (fp32): out*in*1 + out*4 bytes
    Für out=64, in=128:
      bf16: 64*128*2 = 16384 bytes
      int8+scale: 64*128*1 + 64*4 = 8192 + 256 = 8448 bytes
      Ratio: 16384/8448 ≈ 1.94x
    """
    from quantized_linear import quantize_linear, storage_bytes

    layer = _reference_linear(in_=128, out=64, bias=False)
    q_layer = quantize_linear(layer)

    bf16_bytes = layer.weight.numel() * layer.weight.element_size()
    q_bytes = storage_bytes(q_layer)
    assert q_bytes < bf16_bytes / 1.5, f"q_bytes={q_bytes} not < bf16_bytes/1.5={bf16_bytes/1.5}"
    print(f"[OK] storage {bf16_bytes} → {q_bytes} ({bf16_bytes/q_bytes:.2f}x reduction)")


def test_quantized_linear_monkeypatch_replaces_layer():
    """setattr(parent_module, name, QuantizedLinear) ersetzt das Layer.

    Simuliert den model_manager-Anwendungsfall: walk through a Module,
    replace every nn.Linear with a QuantizedLinear. Forward muss
    weiter laufen, Outputs müssen plausibel sein."""
    from quantized_linear import quantize_linear

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(64, 128, bias=True)
            self.l2 = nn.Linear(128, 32, bias=False)
        def forward(self, x):
            return self.l2(F.relu(self.l1(x)))

    torch.manual_seed(42)
    m = M().to(torch.bfloat16)
    # Replace.
    m.l1 = quantize_linear(m.l1)
    m.l2 = quantize_linear(m.l2)

    x = torch.randn(2, 64, dtype=torch.bfloat16)
    y = m(x)
    assert y.shape == (2, 32), f"forward output shape {y.shape} != (2, 32)"
    print(f"[OK] monkeypatch → forward shape {y.shape} preserved")


def test_quantized_linear_with_bias():
    """Bias wird korrekt zur dequantisierten Forward-Output addiert.

    Im quantize-Helper haben wir nur 2D-Gewichte quantisiert; bias bleibt
    separat. QuantizedLinear muss bias weiter unterstützen."""
    from quantized_linear import quantize_linear

    layer = _reference_linear(in_=64, out=32, bias=True)
    q_layer = quantize_linear(layer)
    assert q_layer.bias is not None, "bias was lost during quantize"
    assert q_layer.bias.shape == (32,), f"bias shape {q_layer.bias.shape} != (32,)"

    x = torch.randn(4, 64, dtype=torch.bfloat16)
    y_ref = layer(x)
    y_q = q_layer(x)
    rel_err = (y_q.float() - y_ref.float()).norm() / y_ref.float().norm()
    assert rel_err < 0.1, f"bias+quant rel_err={rel_err:.4f} > 0.1"
    print(f"[OK] bias preserved + rel_err={rel_err:.4f}")


def test_quantized_linear_state_dict_roundtrip():
    """QuantizedLinear hat state_dict mit weight (int8) + scale (fp32).
    Save → Load (neue Instanz) → Forward-Output identisch (cos_sim=1.0)."""
    from quantized_linear import quantize_linear, QuantizedLinear

    layer = _reference_linear(in_=64, out=32, bias=True)
    q_layer = quantize_linear(layer)

    # Save
    sd = q_layer.state_dict()
    assert "weight" in sd and "weight_scale" in sd, f"missing keys in {list(sd.keys())}"
    assert sd["weight"].dtype == torch.int8

    # Load in fresh QuantizedLinear
    fresh = QuantizedLinear(64, 32, bias=True)
    fresh.load_state_dict(sd)

    x = torch.randn(2, 64, dtype=torch.bfloat16)
    y1 = q_layer(x)
    y2 = fresh(x)
    assert torch.equal(y1, y2), "state_dict roundtrip changed outputs"
    print(f"[OK] state_dict roundtrip byte-identical")


if __name__ == "__main__":
    tests = [
        ("output shape matches",           test_quantized_linear_output_shape_matches),
        ("cos_sim vs bf16 >= 0.95",        test_quantized_linear_cos_sim_vs_bf16),
        ("storage < bf16 / 1.5",           test_quantized_linear_uses_less_storage),
        ("monkeypatch replaces layer",     test_quantized_linear_monkeypatch_replaces_layer),
        ("bias preserved",                 test_quantized_linear_with_bias),
        ("state_dict roundtrip",           test_quantized_linear_state_dict_roundtrip),
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
