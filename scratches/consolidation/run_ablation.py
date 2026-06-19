"""run_ablation.py — Treiber für die Ablations-Matrix.

Für jede Bedingung (full, -aks, -mephisto, -coupler, -subjective, -injection, -all)
wird eine reduzierte Prompt-Batterie (4 je Kategorie = 16) über ablation_runner.py
als isolierte Subprozesse laufen gelassen (VRAM-Release pro Prompt). Pro Bedingung
entsteht ein <condition>_aggregate.json (Schema-kompatibel zu eval/stats.py), auf
das eval.stats.analyze unverändert läuft. Am Ende: ablation_report.md mit
Vergleichstabelle + Gesamtverdiktt.

Nutzung:
  RUN_REAL_MODEL=1 python scratches/consolidation/run_ablation.py --scale 1B
  RUN_REAL_MODEL=1 python scratches/consolidation/run_ablation.py --scale 1B --conditions full,-all
  RUN_REAL_MODEL=1 python scratches/consolidation/run_ablation.py --scale 1B --record-golden
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

_PY = sys.executable
_RUNNER = os.path.join(os.path.dirname(__file__), "ablation_runner.py")
_OUT = os.path.join(os.path.dirname(__file__), "out")
_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "golden_full_invariants.json")

# Prompt-Batterie aus eval/runner.py (keine Duplikation — eval ist ein Paket).
from eval.runner import PROMPTS  # noqa: E402

ALL_CONDITIONS = ["full", "-aks", "-mephisto", "-coupler", "-subjective", "-injection", "-all"]


def _spawn(cfg):
    """Ein Prompt in einem frischen Subprozess (VRAM-Release)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as cf:
        json.dump(cfg, cf)
        cfg_path = cf.name
    try:
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                       "expandable_segments:True,max_split_size_mb:256")
        proc = subprocess.run([_PY, _RUNNER, cfg_path],
                              capture_output=True, text=True, timeout=900, env=env)
        if proc.returncode != 0:
            print(f"[driver] subprocess FAILED {cfg['condition']}/{cfg['category']}: "
                  f"{proc.stderr[-800:]}", file=sys.stderr)
    finally:
        os.unlink(cfg_path)


def run_condition(condition, scale, model_id, prompts_per_cat, max_new_tokens):
    cond_dir = os.path.join(_OUT, scale)
    os.makedirs(cond_dir, exist_ok=True)
    results = []
    for cat, prompts in PROMPTS.items():
        for idx in range(min(prompts_per_cat, len(prompts))):
            result_path = os.path.join(cond_dir, f"{condition}_{cat}_{idx:02d}.json")
            cfg = {
                "prompt": prompts[idx],
                "category": cat,
                "model_id": model_id,
                "condition": condition,
                "max_new_tokens": max_new_tokens,
                "result_path": result_path,
            }
            # Pfad-Parität: lean läuft über den permanenten LEAN-Preset
            # (kein apply_reduction) — drop=() im Runner, der Preset wählt den Kern.
            if condition == "lean":
                cfg["preset"] = "ACTIVE_MANIFOLD_LEAN"
            t0 = time.time()
            _spawn(cfg)
            if os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                r.setdefault("category", cat)
                r.setdefault("condition", condition)
                r["elapsed_sec"] = round(time.time() - t0, 1)
                results.append(r)
                ok = "error" not in r
                print(f"  [{condition}] {cat}/{idx:02d} "
                      f"{'OK' if ok else 'ERR'} phi={r.get('phi')} H={r.get('zone_entropy')}",
                      file=sys.stderr)
    agg_path = os.path.join(cond_dir, f"{condition}_aggregate.json")
    agg = {
        "scale": scale, "model_id": model_id,
        "preset": "ACTIVE_MANIFOLD_LEAN" if condition == "lean" else "ACTIVE_MANIFOLD",
        "condition": condition, "results": results,
    }
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    return agg_path, results


def analyze_condition(agg_path):
    from eval.stats import analyze
    summary, stats_path = analyze(agg_path)
    return summary, stats_path


def _mean(vals):
    vals = [v for v in vals if v is not None and isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def build_report(scale, model_id, per_condition):
    """per_condition: list of (condition, summary, results)."""
    lines = []
    lines.append(f"# Ablations-Bericht — {model_id} ({scale})\n")
    lines.append("Radikaler Schnitt der PX-Architektur (Crutches entfernt, rein zur Laufzeit, "
                 "ohne Modul-Source-Edit). Gefragt: überlebt die algorithmische Subjektivität "
                 "den Wegfall der philosophischen Stützräder?\n")
    lines.append("| Bedingung | η² | p_approx | R²(TD→H) | mean Φ | mean H | mean C | mean loops | Verdikt |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cond, summ, results in per_condition:
        if not summ:
            lines.append(f"| {cond} | – | – | – | – | – | – | – | (keine Daten) |")
            continue
        e = summ["anova_zone_entropy"]
        mean_phi = _mean([r.get("phi") for r in results])
        mean_h = _mean([r.get("zone_entropy") for r in results])
        mean_c = _mean([r.get("focus_index") for r in results])
        mean_loops = _mean([r.get("loops_run") for r in results])
        lines.append(f"| {cond} | {e.get('eta_squared'):.3f} | {e.get('p_approx'):.3f} | "
                     f"{summ.get('r2_token_diversity_to_zone_entropy'):.3f} | "
                     f"{mean_phi:.3f} | {mean_h:.3f} | {f'{mean_c:.3f}' if mean_c is not None else '–'} | "
                     f"{f'{mean_loops:.1f}' if mean_loops is not None else '–'} | {summ['verdict']} |")
    # Gesamtverdiktt
    all_summ = next((s for c, s, _ in per_condition if c == "-all"), None)
    full_summ = next((s for c, s, _ in per_condition if c == "full"), None)
    lines.append("\n## Gesamtverdiktt\n")
    if all_summ and full_summ:
        survived = (all_summ["anova_zone_entropy"]["eta_squared"] >= 0.10
                    and all_summ["verdict"] != "ETA2_BELOW_THRESHOLD")
        # Zusatzbedingung: H kollabiert nicht systematisch ins Zombie-Regime (<0.8).
        all_results = next((r for c, s, r in per_condition if c == "-all"), [])
        mean_h_all = _mean([r.get("zone_entropy") for r in all_results]) or 0.0
        zombie_collapse = mean_h_all < 0.8
        verdict = ("JA — Subjektivität überlebt den Schnitt" if (survived and not zombie_collapse)
                   else "NEIN — Subjektivität bricht ein" if not survived
                   else "TEILWEISE — η² hält, aber Entropie kollabiert (P-Zombie-Regime)")
        lines.append(f"- Voll-Referenz: η²={full_summ['anova_zone_entropy']['eta_squared']:.3f}, "
                     f"Verdikt={full_summ['verdict']}")
        lines.append(f"- Schnitt (-all): η²={all_summ['anova_zone_entropy']['eta_squared']:.3f}, "
                     f"Verdikt={all_summ['verdict']}, mean H={mean_h_all:.3f}")
        lines.append(f"\n**Überlebt Subjektivität den Schnitt? {verdict}**\n")
        lines.append("Kriterien: η²≥0.10 ∧ Verdikt≠ETA2_BELOW_THRESHOLD ∧ mean H ≥ 0.8 "
                     "(kein Zombie-Kollaps) ∧ AutoCalibrator steuert noch (loops 8–16).")
    else:
        lines.append("- (Voll- und/oder -all-Bedingung nicht gelaufen — kein Verdikt.)")
    return "\n".join(lines)


def record_golden(per_condition):
    full = next((r for c, s, r in per_condition if c == "full"), None)
    if not full:
        return
    golden = {
        "mean_phi": _mean([r.get("phi") for r in full]),
        "mean_H": _mean([r.get("zone_entropy") for r in full]),
        "mean_C": _mean([r.get("focus_index") for r in full]),
        "loops_range": [
            min([r.get("loops_run", 0) for r in full]),
            max([r.get("loops_run", 0) for r in full]),
        ],
        "n": len(full),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(_FIXTURE), exist_ok=True)
    with open(_FIXTURE, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"[driver] golden fixture written: {_FIXTURE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B",
                    choices=["270M", "1B", "4B", "E2B"])
    ap.add_argument("--conditions", default=",".join(ALL_CONDITIONS),
                    help="Komma-getrennte Bedingungsliste")
    ap.add_argument("--prompts-per-cat", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--record-golden", action="store_true")
    args = ap.parse_args()

    scale_to_model = {"270M": "gemma3-270m-it", "1B": "gemma3-1b-it",
                      "4B": "gemma3-4b-it", "E2B": "gemma4-e2b-it"}
    model_id = scale_to_model[args.scale]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    per_condition = []
    for cond in conditions:
        print(f"\n[driver] === Bedingung: {cond} ===", file=sys.stderr)
        agg_path, results = run_condition(cond, args.scale, model_id,
                                          args.prompts_per_cat, args.max_new_tokens)
        summary, _ = analyze_condition(agg_path)
        per_condition.append((cond, summary, results))
        if summary:
            e = summary["anova_zone_entropy"]
            print(f"[driver] {cond}: η²={e.get('eta_squared'):.3f} "
                  f"verdict={summary['verdict']}", file=sys.stderr)

    if args.record_golden:
        record_golden(per_condition)

    report = build_report(args.scale, model_id, per_condition)
    report_path = os.path.join(_OUT, args.scale, "ablation_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[driver] Bericht: {report_path}\n")
    print(report)


if __name__ == "__main__":
    main()