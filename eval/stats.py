"""
eval/stats.py — Statistical analysis of evaluation results
==========================================================

Reads the per-prompt JSONs produced by run_4b_eval.py, computes:

  1. η² (one-way ANOVA) for zone_entropy across the 4 prompt categories
     — the headline SR-59 / SR-60 metric
  2. R²(token_diversity → zone_entropy) — the anti-P-Zombie test
  3. Per-category means of phi, zone_weights, kurtosis
  4. Writes a summary JSON

Uses only the standard library (json, math, os, sys) plus the lightweight
`numpy` for one-way ANOVA. If numpy is unavailable, falls back to a pure-
Python computation.

Usage:
  python eval/stats.py /path/to/<scale>_<preset>_aggregate.json
  python eval/stats.py /path/to/results_dir/
"""

import json
import math
import os
import sys
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# Pure-Python one-way ANOVA (independent of numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def one_way_anova(groups):
    """One-way ANOVA, pure Python.

    Args:
        groups: dict {label: [values]} or list of [values]

    Returns:
        dict with F, p_approx, eta_squared, k (df_between), n (df_within)
    """
    if isinstance(groups, dict):
        labels = list(groups.keys())
        values_per_group = [list(v) for v in groups.values()]
    else:
        labels = [f"g{i}" for i in range(len(groups))]
        values_per_group = [list(g) for g in groups]

    # Filter empty groups
    values_per_group = [g for g in values_per_group if len(g) > 0]
    k = len(values_per_group)
    if k < 2:
        return {"F": 0.0, "eta_squared": 0.0, "k": k, "n": 0, "note": "fewer than 2 groups"}

    n_total = sum(len(g) for g in values_per_group)
    if n_total < k + 1:
        return {"F": 0.0, "eta_squared": 0.0, "k": k, "n": n_total, "note": "insufficient data"}

    grand_mean = sum(sum(g) for g in values_per_group) / n_total

    # Between-group sum of squares
    ss_between = sum(len(g) * (sum(g) / len(g) - grand_mean) ** 2 for g in values_per_group)
    # Within-group sum of squares
    ss_within = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in values_per_group)

    df_between = k - 1
    df_within = n_total - k

    if df_within == 0 or ss_within < 1e-12:
        # All values identical within groups — no variance to explain
        return {
            "F": float("inf") if ss_between > 1e-12 else 0.0,
            "eta_squared": 1.0 if ss_between > 1e-12 else 0.0,
            "k": df_between,
            "n": df_within,
            "note": "zero within-group variance",
        }

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within
    eta_sq = ss_between / (ss_between + ss_within)

    # p-value approximation: incomplete beta function not available in stdlib.
    # Use Wilson-Hilferty normal approximation for large df_within.
    # (Exact F-distribution requires scipy — out of scope for self-contained eval.)
    p_approx = _f_to_p_normal_approx(f_stat, df_between, df_within)

    return {
        "F": float(f_stat),
        "p_approx": float(p_approx),
        "eta_squared": float(eta_sq),
        "ss_between": float(ss_between),
        "ss_within": float(ss_within),
        "k": int(df_between),
        "n": int(df_within),
    }


def _f_to_p_normal_approx(f_stat, d1, d2):
    """Wilson-Hilferty approximation: P(F > f) ≈ 1 - Φ(z) where
    z = ((1 - 2/(9*d2)) * f^(1/3) - (1 - 2/(9*d1))) / sqrt(
        (2/(9*d1)) + f^(2/3) * 2/(9*d2)
    ).
    This is a rough approximation; for exact p-values use scipy.
    """
    if d1 <= 0 or d2 <= 0:
        return 1.0
    if f_stat <= 0:
        return 1.0
    try:
        a = (1 - 2 / (9 * d2)) * (f_stat ** (1 / 3))
        b = 1 - 2 / (9 * d1)
        var = (2 / (9 * d1)) + (f_stat ** (2 / 3)) * (2 / (9 * d2))
        z = (a - b) / math.sqrt(var)
        # Standard normal CDF: 1 - Φ(z) using error function
        p = 0.5 * math.erfc(z / math.sqrt(2))
        return max(0.0, min(1.0, p))
    except (ValueError, OverflowError):
        return 1.0 if f_stat < 1 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Linear regression (R²)
# ═══════════════════════════════════════════════════════════════════════════════

def linear_r_squared(xs, ys):
    """R² of y on x (Pearson correlation squared, sign dropped)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx < 1e-12 or syy < 1e-12:
        return 0.0
    r = sxy / math.sqrt(sxx * syy)
    return float(r * r)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-category aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _per_category_means(results, key):
    """Returns {category: mean} of `key` across results, ignoring errors and None."""
    by_cat = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        if key not in r or r[key] is None:
            continue
        v = r[key]
        if hasattr(v, "item"):
            v = v.item()
        by_cat[r["category"]].append(float(v))
    return {c: sum(vs) / len(vs) for c, vs in by_cat.items() if vs}


def _per_category_zone_weights(results):
    """Average zone_weights dict per category."""
    by_cat = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "error" in r or not r.get("zone_weights"):
            continue
        for zone, w in r["zone_weights"].items():
            by_cat[r["category"]][zone].append(float(w))
    return {
        cat: {z: sum(vs) / len(vs) for z, vs in zones.items() if vs}
        for cat, zones in by_cat.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(aggregate_path):
    with open(aggregate_path) as f:
        agg = json.load(f)

    results = agg["results"]
    successful = [r for r in results if "error" not in r]
    n_ok = len(successful)
    n_err = len(results) - n_ok

    # 1. η² ANOVA on zone_entropy
    by_cat_entropy = defaultdict(list)
    for r in successful:
        if r.get("zone_entropy") is not None:
            by_cat_entropy[r["category"]].append(r["zone_entropy"])
    anova_entropy = one_way_anova(dict(by_cat_entropy))

    # 2. η² ANOVA on phi (secondary signal)
    by_cat_phi = defaultdict(list)
    for r in successful:
        if r.get("phi") is not None:
            by_cat_phi[r["category"]].append(r["phi"])
    anova_phi = one_way_anova(dict(by_cat_phi))

    # 3. R² token-control: does token diversity explain zone entropy?
    xs = [r["token_diversity_input"] for r in successful
          if r.get("zone_entropy") is not None and r.get("token_diversity_input") is not None]
    ys = [r["zone_entropy"] for r in successful
          if r.get("zone_entropy") is not None and r.get("token_diversity_input") is not None]
    r_sq = linear_r_squared(xs, ys)

    # 4. Per-category means
    mean_phi = _per_category_means(successful, "phi")
    mean_kurtosis = _per_category_means(successful, "kurtosis")
    mean_entropy = _per_category_means(successful, "zone_entropy")
    mean_weights = _per_category_zone_weights(successful)

    # 5. Verdict
    eta2 = anova_entropy["eta_squared"]
    p = anova_entropy.get("p_approx", 1.0)
    if eta2 >= 0.10 and r_sq < 0.30:
        verdict = "ANTI_P_ZOMBIE_CONFIRMED"
        verdict_note = (f"η²={eta2:.3f} ≥ 0.10 AND R²(TD→H)={r_sq:.3f} < 0.30. "
                        "Zone entropy is high-variance across categories and is NOT "
                        "explained by token diversity.")
    elif eta2 >= 0.10:
        verdict = "ETA2_HIGH_TOKEN_CONTROL_FAIL"
        verdict_note = (f"η²={eta2:.3f} ≥ 0.10 BUT R²(TD→H)={r_sq:.3f} ≥ 0.30. "
                        "Zone entropy varies by category but may be token-driven.")
    else:
        verdict = "ETA2_BELOW_THRESHOLD"
        verdict_note = (f"η²={eta2:.3f} < 0.10. Categorical differentiation is "
                        "weak — architecture may not have crossed the threshold "
                        "at this scale with this preset.")

    summary = {
        "input_file": aggregate_path,
        "scale": agg.get("scale"),
        "model_id": agg.get("model_id"),
        "preset": agg.get("preset"),
        "n_prompts_total": len(results),
        "n_successful": n_ok,
        "n_errors": n_err,
        "anova_zone_entropy": anova_entropy,
        "anova_phi": anova_phi,
        "r2_token_diversity_to_zone_entropy": r_sq,
        "mean_phi_per_category": mean_phi,
        "mean_zone_entropy_per_category": mean_entropy,
        "mean_kurtosis_per_category": mean_kurtosis,
        "mean_zone_weights_per_category": mean_weights,
        "verdict": verdict,
        "verdict_note": verdict_note,
    }

    out_path = aggregate_path.replace("_aggregate.json", "_stats.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary, out_path


def main():
    if len(sys.argv) < 2:
        print("usage: python eval/stats.py <aggregate.json> [more...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        if os.path.isdir(path):
            for fn in sorted(os.listdir(path)):
                if fn.endswith("_aggregate.json"):
                    full = os.path.join(path, fn)
                    summary, out_path = analyze(full)
                    _print_summary(full, summary, out_path)
        else:
            summary, out_path = analyze(path)
            _print_summary(path, summary, out_path)


def _print_summary(label, summary, out_path):
    """Format a single evaluation result for terminal output."""
    a = summary["anova_zone_entropy"]
    eta2 = a.get("eta_squared", 0.0)
    p = a.get("p_approx", None)
    p_str = f"{p:.4f}" if isinstance(p, (int, float)) else "N/A"
    note = a.get("note", "")
    note_str = f"  [{note}]" if note else ""
    print(f"\n=== {label} ===")
    print(f"  η² = {eta2:.4f}  p ≈ {p_str}{note_str}")
    print(f"  R²(TD→H) = {summary['r2_token_diversity_to_zone_entropy']:.4f}")
    print(f"  → {summary['verdict']}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
