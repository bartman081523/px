"""
test_pipeline_quantize.py — TDD-rot/green für End-to-End-Pipeline
==================================================================

Phase C von Plan 1. Was diese Tests pinnen:
  - Eine "quantize_all_linears(model)" walkt rekursiv durch ein nn.Module
    und ersetzt JEDES nn.Linear durch ein QuantizedLinear.
  - Output-Shape bleibt über die gesamte Pipeline erhalten.
  - cos_sim vs unquantisiertes Model >= 0.90 (für ein simples Model
    deutlich machbar; 4b-Cos-Sim wird in Phase D gemessen).
  - state_dict eines quantisierten Models hat int8-Gewichte (Speicher-Beweis).
  - Idempotent: zweimal quantize_all_linears ändert nichts (no-op beim 2. Mal).

Run:
    /path/to/venv/bin/python test_pipeline_quantize.py
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def _small_model():
    """Ein kleines MLP — typische 2-Layer + Hidden, ReLU dazwischen."""
    torch.manual_seed(123)
    return nn.Sequential(
        nn.Linear(64, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 10, bias=False),
    ).to(torch.bfloat16)


# Tests ---------------------------------------------------------------------

def test_pipeline_quantize_all_linears_replaces_each():
    """Jedes nn.Linear im Model wird zu QuantizedLinear."""
    from quantize_pipeline import quantize_all_linears, QuantizedLinear

    m = _small_model()
    n_linears_before = sum(1 for _ in m.modules() if isinstance(_, nn.Linear))
    m_q = quantize_all_linears(m)
    n_linears_after = sum(1 for _ in m_q.modules() if isinstance(_, nn.Linear))
    n_q_after = sum(1 for _ in m_q.modules() if isinstance(_, QuantizedLinear))

    assert n_linears_before == 3, f"unexpected number of Linears: {n_linears_before}"
    assert n_linears_after == 0, f"still {n_linears_after} Linears remain after quantize"
    assert n_q_after == 3, f"expected 3 QuantizedLinear, got {n_q_after}"
    print(f"[OK] replaced {n_linears_before} Linears → {n_q_after} QuantizedLinears")


def test_pipeline_output_shape_preserved():
    """Forward-Output-Shape identisch vor/nach Quantisierung."""
    from quantize_pipeline import quantize_all_linears

    m = _small_model()
    m_q = quantize_all_linears(m)

    x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    y_ref = m(x)
    y_q = m_q(x)
    assert y_q.shape == y_ref.shape, f"shape mismatch {y_q.shape} != {y_ref.shape}"
    print(f"[OK] forward shape {y_q.shape} preserved")


def test_pipeline_cos_sim_vs_bf16_high():
    """cos_sim >= 0.90 zwischen quantisiertem und Original-Output.

    Ein simples 3-Layer MLP mit zufälligen Gewichten: jeder int8-Round-Trip
    sammelt ~1% Fehler, 3 Layer in Serie summieren das auf < 10% relativ
    (cos_sim bleibt > 0.90)."""
    from quantize_pipeline import quantize_all_linears

    torch.manual_seed(0)
    m = _small_model()
    m_q = quantize_all_linears(m)

    sims = []
    for i in range(5):
        x = torch.randn(1, 8, 64, dtype=torch.bfloat16)
        y_ref = m(x).float().flatten()
        y_q = m_q(x).float().flatten()
        sim = F.cosine_similarity(y_ref.unsqueeze(0), y_q.unsqueeze(0)).item()
        sims.append(sim)
    mean_sim = sum(sims) / len(sims)
    assert mean_sim >= 0.90, f"cos_sim={mean_sim:.4f} < 0.90 (sims={sims})"
    print(f"[OK] pipeline cos_sim mean={mean_sim:.4f} (5 prompts, min={min(sims):.4f})")


def test_pipeline_storage_quantized():
    """state_dict des quantisierten Models hat int8-Gewichte (Beweis der
    tatsächlichen Quantisierung). Per-Channel scales sind fp32."""
    from quantize_pipeline import quantize_all_linears

    m = _small_model()
    m_q = quantize_all_linears(m)
    sd = m_q.state_dict()

    int8_keys = [k for k, v in sd.items() if v.dtype == torch.int8]
    assert len(int8_keys) == 3, f"expected 3 int8 weights, got {len(int8_keys)}: {int8_keys}"
    # Jeder int8-Key hat ein passendes _scale-Item.
    for k in int8_keys:
        scale_key = k + "_scale"
        assert scale_key in sd, f"missing scale for {k}"
        assert sd[scale_key].dtype == torch.float32
    print(f"[OK] state_dict has {len(int8_keys)} int8 weights + matching scales")


def test_pipeline_idempotent():
    """Zweimal quantize_all_linears auf ein bereits quantisiertes Model
    darf nicht crashen und nicht zusätzliche QuantizedLinear-Layer einfügen."""
    from quantize_pipeline import quantize_all_linears, QuantizedLinear

    m = _small_model()
    m_q1 = quantize_all_linears(m)
    n_q_after_first = sum(1 for _ in m_q1.modules() if isinstance(_, QuantizedLinear))

    m_q2 = quantize_all_linears(m_q1)
    n_q_after_second = sum(1 for _ in m_q2.modules() if isinstance(_, QuantizedLinear))

    assert n_q_after_first == n_q_after_second, (
        f"idempotency broken: {n_q_after_first} → {n_q_after_second}")
    print(f"[OK] idempotent ({n_q_after_first} → {n_q_after_second} QuantizedLinears)")


if __name__ == "__main__":
    tests = [
        ("replaces all Linears",       test_pipeline_quantize_all_linears_replaces_each),
        ("output shape preserved",     test_pipeline_output_shape_preserved),
        ("cos_sim vs bf16 >= 0.90",    test_pipeline_cos_sim_vs_bf16_high),
        ("state_dict has int8",        test_pipeline_storage_quantized),
        ("idempotent",                 test_pipeline_idempotent),
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
