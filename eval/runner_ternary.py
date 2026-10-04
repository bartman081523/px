"""
eval/runner_ternary.py — η²-Eval für ternary-bonsai-27b
========================================================

Eigener Runner-Modus für das PTQ1_0-ternary-Modell: statt
Subprozess-pro-Prompt (gemma, Modell-Load ~10s) läuft hier EIN Prozess
pro Preset (Modell-Load ~3min dominiert; generate ist transient, kein
Leak). Ein Prozess pro Preset — nicht pro Prompt — isoliert VRAM und
CUDA-Kontext zwischen Presets.

Aufruf:
    python eval/runner_ternary.py ACTIVE_MANIFOLD --ntok 24
    python eval/runner_ternary.py BASELINE --ntok 24 --limit 5

Schreibt <outdir>/ternary-27b_<preset>_aggregate.json im
eval/stats.py-Format (η²-ANOVA über zone_entropy je Kategorie) und
ruft stats.analyze am Ende auf.

Telemetrie-Parität zu runner.py: get_px_metrics NACH generate (letzter
Decode-Forward); kurtosis aus cognitive_signature (Prefill-Wert — bei
Decode bleibt _task_kurtosis stehen). Recursion-Schritte sind im
Decoder-Telemetrie-Fenster unsichtbar (loops_run=0 am letzten Forward —
bekanntes SR-61b-Verhalten, dokumentiert in patch.py).
"""

import argparse
import gc
import json
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PP = os.path.join(_ROOT, "px_patches")
if _PP not in sys.path:
    sys.path.insert(0, _PP)

import torch
from transformers import AutoTokenizer

from eval.runner import PROMPTS, shannon_entropy, token_diversity
import ternary_bonsai_27b_px.runtime_qwen35_ptq as rt
import ternary_bonsai_27b_px.patch as px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=["BASELINE", "ACTIVE_MANIFOLD",
                                       "ACTIVE_MANIFOLD_RELAY",
                                       "ACTIVE_MANIFOLD_LEAN"])
    ap.add_argument("--ntok", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0,
                    help="n Prompts pro Kategorie (0 = alle 20)")
    ap.add_argument("--outdir", default=os.path.join(_ROOT, "eval", "results"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    signs, meta = rt.load_signs()
    fold = rt.FoldOps(signs, mode="signs_first")
    model = rt.build_and_load(fold, verbose=True)
    tok = AutoTokenizer.from_pretrained(rt.OUT_DIR, fix_mistral_regex=True)

    if args.preset != "BASELINE":
        px.apply_px_patch(model, config_preset=args.preset)
        print(f"[ternary-eval] gepatcht: {args.preset}", file=sys.stderr)
    else:
        print("[ternary-eval] BASELINE — unpatched", file=sys.stderr)

    gen_base = {
        "do_sample": False,
        "temperature": 1.0,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.eos_token_id,
    }
    if args.preset != "BASELINE":
        from generators import _px_gen_kwargs
        gen_base = _px_gen_kwargs(model, gen_base)

    results = []
    abort = False
    t_wall0 = time.time()
    for cat, plist in PROMPTS.items():
        if abort:
            break
        for p in (plist[:args.limit] if args.limit else plist):
            msgs = [{"role": "user", "content": p}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
                ids = tok(text, return_tensors="pt").input_ids.to(model.device)
                n_in = ids.shape[1]
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=args.ntok,
                                         **gen_base)
                dt = time.time() - t0
                new_tokens = out[0][n_in:]
                completion = tok.decode(new_tokens, skip_special_tokens=True)
                n_new = len(new_tokens)

                metrics = {}
                if args.preset != "BASELINE":
                    try:
                        metrics = px.get_px_metrics(model) or {}
                    except Exception as e:                      # noqa: BLE001
                        print(f"[ternary-eval] metrics failed: {e}",
                              file=sys.stderr)
                zw = metrics.get("zone_weights", {}) or {}
                phi = metrics.get("phi", 1.0)
                if hasattr(phi, "item"):
                    phi = phi.item()
                sig = metrics.get("cognitive_signature", {}) or {}
                k = sig.get("kurtosis", None)
                if hasattr(k, "item"):
                    k = k.item()

                results.append({
                    "category": cat,
                    "prompt": p,
                    "completion": completion,
                    "model_id": "ternary-bonsai-27b",
                    "preset": args.preset,
                    "completion_tokens": n_new,
                    "input_tokens": n_in,
                    "gen_time_sec": dt,
                    "sec_per_token": dt / max(n_new, 1),
                    "phi": float(phi),
                    "zone": metrics.get("zone", "UNKNOWN"),
                    "zone_weights": {k2: float(v) for k2, v in zw.items()}
                                     if zw else {},
                    "zone_entropy": shannon_entropy(zw),
                    "kurtosis": float(k) if k is not None else None,
                    "token_diversity_input": token_diversity(ids[0]),
                    "loops_run": metrics.get("steps", 0),
                    "entropy": metrics.get("entropy", 0.0),
                })
                print(f"[ternary-eval] {cat} | H={results[-1]['zone_entropy']:.3f}"
                      f" phi={phi:.3f} zone={results[-1]['zone']}"
                      f" {n_new}tok {dt:.1f}s", file=sys.stderr)
            except Exception as e:                                # noqa: BLE001
                print(f"[ternary-eval] {cat} ERROR: {e}", file=sys.stderr)
                results.append({"category": cat, "prompt": p,
                                "error": str(e), "model_id": "ternary-bonsai-27b",
                                "preset": args.preset})
                # CUDA-Kontext-Risiko: bei GPU-Error abbrechen
                if "CUDA" in str(e) or "cublas" in str(e):
                    print("[ternary-eval] CUDA-Fehler — Preset abbrechen",
                          file=sys.stderr)
                    abort = True
                    break

    agg = {
        "scale": "ternary-27b",
        "model_id": "ternary-bonsai-27b",
        "preset": args.preset,
        "results": results,
    }
    out_path = os.path.join(args.outdir,
                            f"ternary-27b_{args.preset}_aggregate.json")
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    vm = torch.cuda.max_memory_allocated() / 2**30
    print(f"[ternary-eval] {len(results)} Prompts in "
          f"{(time.time() - t_wall0)/60:.1f} min, VRAM {vm:.2f} GiB", file=sys.stderr)
    print(f"[ternary-eval] wrote {out_path}", file=sys.stderr)

    from eval.stats import analyze
    summary, _ = analyze(out_path)
    print(json.dumps(summary["anova_zone_entropy"], indent=2))
    print(f"Verdict: {summary['verdict']}")
    print(f"R²(TD→H) = {summary['r2_token_diversity_to_zone_entropy']:.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())