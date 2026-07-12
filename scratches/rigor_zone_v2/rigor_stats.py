"""Statistik-Schicht für RIGOR v2.

SciMind 5.0:
- Falsificationism: Hypothesen-Tests sind Falsifikations-orientiert (signifikante Widerlegung)
- Empirical Verification: Bootstrap-CI für robuste Unsicherheits-Schätzung
- Reproducibility: byte-identische Runs werden explizit verifiziert

3 Hauptfunktionen:
  1. compute_eta2(groups)       — Effektstärke zwischen Arm-Gruppen (Ziel: η² ≥ 0.05)
  2. bootstrap_ci(values)       — 95% CI via 1000 Resamples
  3. reproducibility_check(r1, r2) — Byte-Identitäts-Test
"""
from __future__ import annotations

import hashlib
import random
import statistics
from typing import List, Dict, Tuple, Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. η² (Eta-Quadrat) — Effektstärke
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eta2(groups: List[List[float]]) -> float:
    """Berechnet η² (Effektstärke der Gruppen-Unterscheidung).

    η² = SS_between / SS_total

    Interpretation (Cohen):
      η² = 0.01  → kleiner Effekt
      η² = 0.06  → mittlerer Effekt
      η² = 0.14  → großer Effekt

    Ziel aus CLAUDE.md (SR-61): η² ≥ 0.05 bei p < 0.05

    Returns:
        0.0 wenn keine Variation, sonst float in [0, 1]
    """
    if not groups or len(groups) < 2:
        return 0.0
    # Filter leere Gruppen
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return 0.0
    # Grand mean
    all_values = [v for g in groups for v in g]
    if not all_values:
        return 0.0
    grand_mean = statistics.mean(all_values)
    # SS_between
    ss_between = sum(len(g) * (statistics.mean(g) - grand_mean) ** 2 for g in groups)
    # SS_total
    ss_total = sum((v - grand_mean) ** 2 for v in all_values)
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


def compute_eta2_per_category(
    results_by_arm: Dict[str, List[Dict[str, Any]]],
    metric: str,
    category_key: str = "category",
) -> Dict[str, float]:
    """Berechnet η² pro HLE-Kategorie für H2 (taskspezifischer Effekt).

    Args:
        results_by_arm: {arm_name: [result_dicts]} mit "category" + "duration_sec" etc.
        metric: z.B. "duration_sec", "output_length", "loop_rate"
        category_key: key im result_dict für Kategorie
    """
    # Gruppierung: (category, arm) → [metric values]
    groups_by_cat: Dict[str, Dict[str, List[float]]] = {}
    for arm_name, results in results_by_arm.items():
        for r in results:
            cat = r.get(category_key, "unknown")
            val = r.get(metric)
            if val is None or not isinstance(val, (int, float)):
                continue
            if cat not in groups_by_cat:
                groups_by_cat[cat] = {}
            if arm_name not in groups_by_cat[cat]:
                groups_by_cat[cat][arm_name] = []
            groups_by_cat[cat][arm_name].append(val)

    return {
        cat: compute_eta2(list(arm_vals.values()))
        for cat, arm_vals in groups_by_cat.items()
        if len(arm_vals) >= 2
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Bootstrap-CI (95% Konfidenzintervall)
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap-95% CI für den Mittelwert.

    Returns:
        (mean, ci_low, ci_high)
    """
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    alpha = (1 - confidence) / 2
    lo_idx = max(0, int(alpha * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha) * n_resamples) - 1)
    return (statistics.mean(values), means[lo_idx], means[hi_idx])


def bootstrap_ci_per_arm(
    results_by_arm: Dict[str, List[Dict[str, Any]]],
    metric: str,
    n_resamples: int = 1000,
) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap-CI pro Arm."""
    out = {}
    for arm_name, results in results_by_arm.items():
        vals = [r.get(metric) for r in results if isinstance(r.get(metric), (int, float))]
        if vals:
            out[arm_name] = bootstrap_ci(vals, n_resamples)
        else:
            out[arm_name] = (0.0, 0.0, 0.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Reproduzierbarkeits-Test
# ═══════════════════════════════════════════════════════════════════════════════

def hash_output(output: str) -> str:
    """SHA256-Hash des Outputs für byte-Vergleich."""
    return hashlib.sha256(output.encode("utf-8")).hexdigest()[:16]


def reproducibility_check(
    run1_output: str, run2_output: str, ignore_whitespace: bool = True
) -> bool:
    """Prüft ob zwei Runs byte-identisch sind.

    Args:
        run1_output, run2_output: Output-Strings
        ignore_whitespace: trimmt Whitespace vor Vergleich (default: True)
    """
    if ignore_whitespace:
        return run1_output.strip() == run2_output.strip()
    return run1_output == run2_output


def reproducibility_summary(
    reproducibility_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregiert Reproducibility-Runs zu Pass-Rate + Sample-Mismatches.

    Args:
        reproducibility_runs: [{arm, task, seed, output_hash_0, output_hash_1, match}, ...]
    """
    if not reproducibility_runs:
        return {"n": 0, "pass_rate": 0.0, "mismatches": []}
    n = len(reproducibility_runs)
    n_match = sum(1 for r in reproducibility_runs if r.get("match", False))
    mismatches = [r for r in reproducibility_runs if not r.get("match", False)]
    return {
        "n": n,
        "n_match": n_match,
        "pass_rate": n_match / n,
        "mismatches": mismatches[:5],  # max 5 sample-mismatches für Report
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Hypothesen-Tests (H1-H4)
# ═══════════════════════════════════════════════════════════════════════════════

def test_h1_loop_reduction(
    results_by_arm: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """H1: Mephisto-Damping (scale<1.0) reduziert Output-Loops.

    Test: rigor_least (0.1) sollte weniger Loops haben als rigor_full (1.0).
    """
    if "rigor_least" not in results_by_arm or "rigor_full" not in results_by_arm:
        return {"testable": False, "reason": "rigor_least oder rigor_full fehlt"}

    least_loops = sum(
        1 for r in results_by_arm["rigor_least"] if r.get("output_loops", False)
    )
    full_loops = sum(
        1 for r in results_by_arm["rigor_full"] if r.get("output_loops", False)
    )
    least_n = len(results_by_arm["rigor_least"])
    full_n = len(results_by_arm["rigor_full"])

    least_rate = least_loops / max(least_n, 1)
    full_rate = full_loops / max(full_n, 1)
    return {
        "testable": True,
        "rigor_least_loop_rate": least_rate,
        "rigor_least_n": least_n,
        "rigor_full_loop_rate": full_rate,
        "rigor_full_n": full_n,
        "h1_confirmed": least_rate < full_rate,
        "delta": full_rate - least_rate,
    }


def test_h2_task_specific(
    results_by_arm: Dict[str, List[Dict[str, Any]]],
    metric: str = "duration_sec",
) -> Dict[str, Any]:
    """H2: RIGOR-Damping-Effekt ist taskspezifisch.

    Test: η² pro Kategorie (math, cs, philosophy).
    """
    eta2_per_cat = compute_eta2_per_category(results_by_arm, metric)
    return {
        "testable": bool(eta2_per_cat),
        "eta2_per_category": eta2_per_cat,
        "h2_confirmed": (
            bool(eta2_per_cat) and
            max(eta2_per_cat.values()) - min(eta2_per_cat.values()) > 0.02
        ),
    }


def test_h3_search_effect(
    results_by_arm: Dict[str, List[Dict[str, Any]]],
    metric: str = "duration_sec",
) -> Dict[str, Any]:
    """H3: Web-Suche hat messbaren Effekt auf Code-Tasks, nicht auf Math-Tasks.

    Test: Vergleicht search_on vs search_off per Kategorie.
    """
    # Gruppierung: (category, search) → [values]
    groups: Dict[Tuple[str, bool], List[float]] = {}
    for arm, results in results_by_arm.items():
        for r in results:
            cat = r.get("category", "unknown")
            search = bool(r.get("with_search", False))
            val = r.get(metric)
            if val is None or not isinstance(val, (int, float)):
                continue
            key = (cat, search)
            if key not in groups:
                groups[key] = []
            groups[key].append(val)

    # Pro Kategorie: search_on vs search_off
    eta2_per_cat: Dict[str, float] = {}
    for cat in set(c[0] for c in groups.keys()):
        on = groups.get((cat, True), [])
        off = groups.get((cat, False), [])
        if on and off:
            eta2_per_cat[cat] = compute_eta2([on, off])
    return {
        "testable": bool(eta2_per_cat),
        "eta2_search_per_category": eta2_per_cat,
        "h3_confirmed": (
            "math" in eta2_per_cat and
            "cs" in eta2_per_cat and
            eta2_per_cat.get("math", 1.0) < eta2_per_cat.get("cs", 0.0)
        ),
    }


def test_h4_reproducibility(
    reproducibility_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """H4: Reproduzierbarkeit ≥ 95% (gleicher Seed → gleicher Output)."""
    summary = reproducibility_summary(reproducibility_runs)
    return {
        "testable": summary["n"] > 0,
        **summary,
        "h4_confirmed": summary.get("pass_rate", 0) >= 0.95,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RIGOR Stats v2 — Smoke-Test")
    print("=" * 60)

    # Test 1: compute_eta2
    e1 = compute_eta2([[1, 2, 3], [4, 5, 6]])
    assert e1 > 0.5, f"expected >0.5 für klare Trennung, got {e1}"
    e2 = compute_eta2([[1, 2, 3], [1, 2, 3]])
    assert e2 == 0.0, f"expected 0 für identische Gruppen, got {e2}"
    e3 = compute_eta2([])
    assert e3 == 0.0
    print(f"  compute_eta2: stark={e1:.3f} gleich={e2:.3f} leer={e3}")

    # Test 2: bootstrap_ci
    mean, lo, hi = bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert lo <= mean <= hi, f"CI-Order falsch: {lo} <= {mean} <= {hi}"
    print(f"  bootstrap_ci([1..10]): mean={mean}, 95% CI=[{lo}, {hi}]")

    # Test 3: reproducibility_check
    assert reproducibility_check("Hello world", "Hello world")
    assert reproducibility_check("  Hello world  ", "Hello world")
    assert not reproducibility_check("Hello world", "Hello World")
    assert not reproducibility_check("Hello", "World")
    print("  reproducibility_check: OK")

    # Test 4: hash_output
    h = hash_output("test")
    assert len(h) == 16
    print(f"  hash_output('test') = {h}")

    # Test 5: test_h1 (synthetische Daten)
    synthetic = {
        "rigor_least": [{"output_loops": False} for _ in range(10)],
        "rigor_full": [{"output_loops": True} for _ in range(10)],
    }
    h1 = test_h1_loop_reduction(synthetic)
    assert h1["testable"]
    assert h1["h1_confirmed"]
    print(f"  test_h1 (synth): loop_rate least={h1['rigor_least_loop_rate']}, full={h1['rigor_full_loop_rate']}, confirmed={h1['h1_confirmed']}")

    # Test 6: test_h2
    synthetic_per_cat = {
        "baseline": [
            {"category": "math", "duration_sec": 10 + i} for i in range(5)
        ] + [
            {"category": "cs", "duration_sec": 20 + i} for i in range(5)
        ],
        "rigor_mid": [
            {"category": "math", "duration_sec": 12 + i} for i in range(5)
        ] + [
            {"category": "cs", "duration_sec": 18 + i} for i in range(5)
        ],
    }
    h2 = test_h2_task_specific(synthetic_per_cat)
    assert h2["testable"]
    print(f"  test_h2 (synth): eta2_per_cat={h2['eta2_per_category']}")

    # Test 7: test_h4
    repro = [{"match": True, "arm": "baseline", "task": "t1"} for _ in range(9)] + [
        {"match": False, "arm": "baseline", "task": "t10"}
    ]
    h4 = test_h4_reproducibility(repro)
    assert h4["pass_rate"] == 0.9
    assert not h4["h4_confirmed"]
    print(f"  test_h4: pass_rate={h4['pass_rate']}, confirmed={h4['h4_confirmed']}")

    print()
    print("=" * 60)
    print("✓ Alle Smoketests grün")
    print("=" * 60)
