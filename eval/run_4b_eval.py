"""
eval/run_4b_eval.py — Main driver for 4B η² evaluation
======================================================

For each of 4 categories × N prompts:
  1. Build a JSON config
  2. Spawn runner.py as a fresh subprocess (VRAM isolated)
  3. Collect the result JSON
  4. (Optional) Pipe to stats.py at the end

Hardware-aware defaults for the RTX 2060 (12GB):
  - max_new_tokens = 30
  - bf16 weights
  - use_cache = False (in runner)
  - one prompt per subprocess (no accumulation)

Usage:
  python eval/run_4b_eval.py --scale 4B --prompts-per-cat 3
  python eval/run_4b_eval.py --scale 4B --prompts-per-cat 20 --preset ACTIVE_MANIFOLD
  python eval/run_4b_eval.py --scale 1B --prompts-per-cat 20   # sanity-check at smaller scale
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Project root on path so we can import eval.runner.PROMPTS
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.runner import PROMPTS  # noqa: E402

# Model registry shortcuts (kept in one place so adding a new scale is trivial)
SCALE_TO_MODEL = {
    "270M": "gemma3-270m-it",
    "1B":   "gemma3-1b-it",
    "4B":   "gemma3-4b-it",
    "E2B":  "gemma4-e2b-it",
}


def _spawn_runner(prompt_text, model_id, preset, max_new_tokens, result_path,
                  timeout=600):
    """Spawn a single prompt through runner.py. Returns the parsed JSON or
    raises if the subprocess failed. timeout=600s = 10 minutes per prompt,
    well above the 4B-generation time of ~30s for 30 tokens.
    """
    cfg = {
        "prompt": prompt_text,
        "model_id": model_id,
        "preset": preset,
        "max_new_tokens": max_new_tokens,
        "result_path": result_path,
    }
    cfg_path = result_path + ".cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    cmd = [
        sys.executable,
        os.path.join(_ROOT, "eval", "runner.py"),
        cfg_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "prompt": prompt_text}

    if proc.returncode != 0:
        return {
            "error": f"exit_code_{proc.returncode}",
            "prompt": prompt_text,
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
        }
    
    # SR-61 debug: always print stderr to see calibration status
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")

    try:
        with open(result_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"error": f"read_failed: {e}", "prompt": prompt_text}


def run_eval(scale, prompts_per_cat, preset, max_new_tokens, output_dir, dry_run):
    """Run prompts × categories. Saves per-prompt JSON, returns aggregated list."""
    if scale not in SCALE_TO_MODEL:
        raise ValueError(f"Unknown scale {scale!r}. Use one of: {list(SCALE_TO_MODEL)}")
    model_id = SCALE_TO_MODEL[scale]

    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    n_total = prompts_per_cat * len(PROMPTS)
    t_start = time.time()

    for cat_idx, (category, prompts) in enumerate(PROMPTS.items()):
        for p_idx, prompt in enumerate(prompts[:prompts_per_cat]):
            idx = cat_idx * prompts_per_cat + p_idx
            result_path = os.path.join(
                output_dir, f"{scale}_{preset}_{category}_{p_idx:02d}.json"
            )

            print(f"\n[{idx+1:3d}/{n_total}] {scale} | {preset} | {category} | "
                  f"prompt #{p_idx+1}", flush=True)
            print(f"  > {prompt[:80]}{'...' if len(prompt) > 80 else ''}",
                  flush=True)

            if dry_run:
                all_results.append({
                    "prompt": prompt, "category": category, "dry_run": True,
                })
                continue

            t_prompt = time.time()
            result = _spawn_runner(
                prompt, model_id, preset, max_new_tokens, result_path
            )
            dt = time.time() - t_prompt
            result["category"] = category
            result["elapsed_sec"] = dt
            all_results.append(result)

            if "error" in result:
                print(f"  ✗ {result['error']} ({dt:.1f}s)", flush=True)
            else:
                print(f"  ✓ phi={result.get('phi', 'NA'):.3f} "
                      f"H={result.get('zone_entropy', 'NA'):.3f} "
                      f"zone={result.get('zone', 'NA')} "
                      f"({dt:.1f}s, {result.get('completion_tokens', 0)}tok)",
                      flush=True)

    # Aggregate dump
    agg_path = os.path.join(
        output_dir, f"{scale}_{preset}_aggregate.json"
    )
    with open(agg_path, "w") as f:
        json.dump({
            "scale": scale,
            "model_id": model_id,
            "preset": preset,
            "prompts_per_cat": prompts_per_cat,
            "n_total": n_total,
            "total_time_sec": time.time() - t_start,
            "results": all_results,
        }, f, indent=2)
    print(f"\n[run_4b_eval] Wrote {agg_path}")
    return all_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", default="4B",
                   choices=list(SCALE_TO_MODEL.keys()),
                   help="Model scale (default: 4B)")
    p.add_argument("--prompts-per-cat", type=int, default=3,
                   help="Prompts per category (default: 3 for smoke test)")
    p.add_argument("--preset", default="ACTIVE_MANIFOLD",
                   help="PX preset (default: ACTIVE_MANIFOLD)")
    p.add_argument("--max-new-tokens", type=int, default=30,
                   help="Max generation length (default: 30 for VRAM safety)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: eval/results/<scale>_<preset>_<ts>)")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't actually run — just verify prompt loading")
    args = p.parse_args()

    out_dir = args.output_dir or os.path.join(
        _ROOT, "eval", "results",
        f"{args.scale}_{args.preset}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_eval(
        scale=args.scale,
        prompts_per_cat=args.prompts_per_cat,
        preset=args.preset,
        max_new_tokens=args.max_new_tokens,
        output_dir=out_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
