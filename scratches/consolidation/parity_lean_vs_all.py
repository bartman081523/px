"""parity_lean_vs_all.py — Pfad-Parität: permanenter Preset-Schnitt (lean)
vs. validierter Laufzeit-Schnitt (-all).

Beide Wege laufen über dieselben 80 Prompts (deterministisch, do_sample=False),
also sind Abweichungen pro Prompt *rein mechanisch* (Preset-Pfad vs.
apply_reduction-Pfad). Erwartung: zone_entropy / phi / zone / loops pro Prompt
identisch oder nahezu identisch — beweist, dass der Preset den validierten
Laufzeit-Schnitt korrekt verankert.

Nutzung:
  python scratches/consolidation/parity_lean_vs_all.py
  python scratches/consolidation/parity_lean_vs_all.py --dir scratches/consolidation/out/1B
"""
import argparse
import json
import os
import sys
from collections import defaultdict

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

from eval.stats import analyze  # noqa: E402


def _load_cond(dir_, cond):
    agg = os.path.join(dir_, f"{cond}_aggregate.json")
    if not os.path.exists(agg):
        return None, None
    with open(agg) as f:
        d = json.load(f)
    by_key = {}
    for r in d.get("results", []):
        if "error" in r:
            continue
        key = (r.get("category"), r.get("prompt"))
        by_key[key] = r
    return d, by_key


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _mae(a, b):
    return abs(a - b) if (a is not None and b is not None) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "out", "1B"))
    args = ap.parse_args()

    lean_agg, lean_by = _load_cond(args.dir, "lean")
    all_agg, all_by = _load_cond(args.dir, "-all")
    if lean_by is None:
        print(f"[parity] FEHLER: lean_aggregate.json nicht gefunden in {args.dir}")
        sys.exit(1)
    if all_by is None:
        print(f"[parity] HINWEIS: -all_aggregate.json fehlt — vergleiche nur lean vs full.")
        all_by = {}

    # ── Aggregate-Verdikte (η², R²) ──
    print("=" * 72)
    print(" PFAD-PARITÄT: lean (Preset ACTIVE_MANIFOLD_LEAN) vs -all (apply_reduction)")
    print("=" * 72)
    for cond, agg in (("lean", lean_agg), ("-all", all_agg)):
        if agg is None:
            continue
        summ, _ = analyze(os.path.join(args.dir, f"{cond}_aggregate.json"))
        e = summ["anova_zone_entropy"]
        print(f"  {cond:6s} | n={len(agg.get('results', [])):3d} | "
              f"η²={e.get('eta_squared'):.3f} | p≈{e.get('p_approx'):.3f} | "
              f"R²(TD→H)={summ.get('r2_token_diversity_to_zone_entropy'):.3f} | "
              f"verdict={summ['verdict']}")

    # ── Paarweiser Per-Prompt-Vergleich ──
    keys = sorted(set(lean_by) & set(all_by))
    if not keys:
        print("\n[parity] Keine gemeinsamen Prompts (— all nicht vorhanden).")
        return
    fields = ["zone_entropy", "phi", "loops_run", "focus_index"]
    diffs = {f: [] for f in fields}
    zone_mismatch = 0
    for k in keys:
        lr, ar = lean_by[k], all_by[k]
        if lr.get("zone") != ar.get("zone"):
            zone_mismatch += 1
        for f in fields:
            diffs[f].append(_mae(lr.get(f), ar.get(f)))

    print(f"\n  gemeinsame Prompts: {len(keys)} | zone-Mismatches: {zone_mismatch}")
    print("  Per-Prompt MAE (mean abs diff lean vs -all):")
    for f in fields:
        m = _mean(diffs[f])
        print(f"    {f:18s}: {m:.6f}" if m is not None else f"    {f:18s}: –")

    # Crutch-Metriken: lean muss 0.0 sein, -all ebenfalls (apply_reduction entfernt sie)
    lean_aks = _mean([r.get("aks_friction") for r in lean_by.values()])
    lean_em = _mean([r.get("emancipation") for r in lean_by.values()])
    all_aks = _mean([r.get("aks_friction") for r in all_by.values()])
    all_em = _mean([r.get("emancipation") for r in all_by.values()])
    print("\n  Crutch-Metriken (mean):")
    print(f"    lean:  aks_friction={lean_aks}  emancipation={lean_em}")
    print(f"    -all:  aks_friction={all_aks}  emancipation={all_em}")

    # ── Verdikt ──
    lean_summ, _ = analyze(os.path.join(args.dir, "lean_aggregate.json"))
    le = lean_summ["anova_zone_entropy"]["eta_squared"]
    lr2 = lean_summ.get("r2_token_diversity_to_zone_entropy", 1.0)
    anti = (le >= 0.10 and lr2 < 0.30)
    mae_h = _mean(diffs["zone_entropy"])
    if mae_h is None:
        mae_h = float("inf")
    print("\n" + "=" * 72)
    print(" VERDIKT")
    print("=" * 72)
    print(f"  lean η²={le:.3f}, R²(TD→H)={lr2:.3f} → "
          f"{'ANTI_P_ZOMBIE_CONFIRMED' if anti else 'NICHT bestätigt'}")
    print(f"  Per-Prompt zone_entropy MAE lean vs -all = {mae_h:.6f}")
    if mae_h < 1e-3:
        print("  → Pfad-Parität EXAKT: Preset-Pfad reproduziert den validierten")
        print("    Laufzeit-Schnitt (-all) pro Prompt identisch. ✓")
    elif mae_h < 0.05:
        print("  → Pfad-Parität binnen Toleranz (MAE<0.05): Preset ≈ -all. ✓")
    else:
        print("  → WARNUNG: Mechanik weicht ab (MAE≥0.05) — Preset nicht äquivalent.")


if __name__ == "__main__":
    main()