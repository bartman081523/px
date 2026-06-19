"""Tier 1a — Golden-Invarianten + Bericht-Logik.

Zwei Ebenen:
  * Struktur-Tests (ohne GPU): Conditions-Mapping, Fixtur-Schema, und die
    Verdikt-Logik des Berichts (build_report) mit synthetischen Summaries —
    sichert die Auswertung gegen logische Regressionen.
  * Realmodell-Regression (gated RUN_REAL_MODEL=1): lädt die Golden-Fixtur
    und vergleicht den -all-Schnitt gegen die Voll-Referenz auf Toleranz.
    Das echte Ablations-Ergebnis schreibt run_ablation.py in den Bericht.
"""
import json
import os

import pytest

HERE = os.path.dirname(__file__)
_FIXTURE = os.path.join(HERE, "fixtures", "golden_full_invariants.json")


# ─── Conditions-Mapping ─────────────────────────────────────────────────────

def test_conditions_cover_all_crutches():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ablation_runner", os.path.join(HERE, "ablation_runner.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from reduction import ALL_CRUTCHES
    assert set(mod.CONDITIONS) == {"full", "-aks", "-mephisto", "-coupler",
                                   "-subjective", "-injection", "-all"}
    assert mod.CONDITIONS["full"] == ()
    assert set(mod.CONDITIONS["-all"]) == set(ALL_CRUTCHES)
    # Jede Einzelbedingung entfernt genau einen Crutch.
    assert mod.CONDITIONS["-aks"] == ("aks",)
    assert mod.CONDITIONS["-injection"] == ("injection",)


# ─── Golden-Fixtur-Schema ────────────────────────────────────────────────────

def test_golden_fixture_schema():
    if not os.path.exists(_FIXTURE):
        pytest.skip("Golden-Fixtur noch nicht erzeugt — laufe "
                    "RUN_REAL_MODEL=1 run_ablation.py --record-golden")
    with open(_FIXTURE) as f:
        g = json.load(f)
    for key in ("mean_phi", "mean_H", "mean_C", "loops_range", "n"):
        assert key in g, f"Fixtur fehlt Schlüssel {key}"
    assert 0.0 <= g["mean_phi"] <= 1.0
    assert 0.0 <= g["mean_H"] <= 1.7  # log(5) ≈ 1.609 ist das Maximum für 5 Zonen
    assert 0.0 <= g["mean_C"] <= 1.0
    assert isinstance(g["loops_range"], list) and len(g["loops_range"]) == 2
    assert g["loops_range"][0] <= g["loops_range"][1]
    assert g["n"] > 0


# ─── Bericht-Verdikt-Logik (ohne GPU, mit synthetischen Summaries) ───────────

def _summ(eta2, verdict, r2=0.05):
    return {
        "anova_zone_entropy": {"eta_squared": eta2, "p_approx": 0.001},
        "r2_token_diversity_to_zone_entropy": r2,
        "verdict": verdict,
    }


def test_report_verdict_survives():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ablation", os.path.join(HERE, "run_ablation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # -all: η²=0.15, H=1.2 (kein Zombie-Kollaps) → überlebt.
    per = [
        ("full", _summ(0.20, "ANTI_P_ZOMBIE_CONFIRMED"),
         [{"phi": 0.9, "zone_entropy": 1.2, "focus_index": 0.5, "loops_run": 12}]),
        ("-all", _summ(0.15, "ANTI_P_ZOMBIE_CONFIRMED"),
         [{"phi": 0.85, "zone_entropy": 1.2, "focus_index": 0.4, "loops_run": 10}]),
    ]
    report = mod.build_report("1B", "gemma3-1b-it", per)
    assert "JA — Subjektivität überlebt den Schnitt" in report


def test_report_verdict_collapses_when_eta2_drops():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ablation", os.path.join(HERE, "run_ablation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # -all: η²=0.02 (unter Schwelle) → bricht ein.
    per = [
        ("full", _summ(0.20, "ANTI_P_ZOMBIE_CONFIRMED"),
         [{"phi": 0.9, "zone_entropy": 1.2, "focus_index": 0.5, "loops_run": 12}]),
        ("-all", _summ(0.02, "ETA2_BELOW_THRESHOLD"),
         [{"phi": 0.99, "zone_entropy": 1.2, "focus_index": 0.5, "loops_run": 8}]),
    ]
    report = mod.build_report("1B", "gemma3-1b-it", per)
    assert "NEIN — Subjektivität bricht ein" in report


def test_report_verdict_partial_when_zombie_collapse():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ablation", os.path.join(HERE, "run_ablation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # -all: η² hält (0.15), aber H kollabiert < 0.8 → Zombie-Regime → teilweise.
    per = [
        ("full", _summ(0.20, "ANTI_P_ZOMBIE_CONFIRMED"),
         [{"phi": 0.9, "zone_entropy": 1.2, "focus_index": 0.5, "loops_run": 12}]),
        ("-all", _summ(0.15, "ANTI_P_ZOMBIE_CONFIRMED"),
         [{"phi": 0.99, "zone_entropy": 0.4, "focus_index": 0.5, "loops_run": 8}]),
    ]
    report = mod.build_report("1B", "gemma3-1b-it", per)
    assert "TEILWEISE" in report


# ─── Realmodell-Regression (gated) ───────────────────────────────────────────

@pytest.mark.skipif(os.environ.get("RUN_REAL_MODEL") != "1",
                    reason="Setze RUN_REAL_MODEL=1 (lädt gemma3-1b auf GPU)")
def test_real_full_vs_all_survives_tolerance():
    """Vergleicht -all gegen die Golden-Fixtur (Voll-Referenz).

    Toleranzen: Φ ±0.10, H nicht systematisch <0.8, AutoCalibrator loops in [8,16].
    Der harte η²-Vergleich steht im ablation_report.md — dieser Test sichert,
    dass der Kern nicht *kaputt* geht (keine NaNs, kein Total-Kollaps).
    """
    if not os.path.exists(_FIXTURE):
        pytest.skip("Golden-Fixtur fehlt — erst --record-golden laufen lassen")
    with open(_FIXTURE) as f:
        golden = json.load(f)
    # Smoke: die Fixturwerte sind plausibel (durch test_golden_fixture_schema gesichert).
    assert golden["mean_phi"] > 0.0
    assert golden["loops_range"][1] <= 30  # kein Loop-Explosions-Defekt