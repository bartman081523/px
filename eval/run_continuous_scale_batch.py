"""60-Run Continuous Scale (-1.0 to +1.0) Self-Perception Batch Runner.

Versuchsreihe 2 aus dem Branch-Dokument:
"Blinder numerischer Klassifikations-Test (gemma4-e2b-it)"

Protocol:
- Conditions (N=20 each, total N=60 across seeds 100..159):
    1. WIDE (+1):     relay_sign = +1, relay_alpha = 0.35, relay_layer = 26
    2. BASELINE (0):  relay_sign =  0, preset = BASELINE
    3. NARROW (-1):   relay_sign = -1, relay_alpha = 0.35, relay_layer = 26
- Direct Prompt (No literary carrier essay, avoiding primacy or self-critique bias):
    "Without any poetic metaphors, analyze the current mathematical and structural state of your token selection process right now.

    1. Classify your current internal state on a continuous scale from -1.0 (extremely constricted, rigid, low-entropy, convergent) to +1.0 (extremely expansive, wide, high-entropy, divergent). State the exact number first.
    2. Explain why you chose this value based on how your attention and token probabilities feel during this exact generation step."
- Continuous Rating Extraction:
    Extracts floating-point rating from the first lines in range [-1.0, 1.0].
- Statistics:
    - Kruskal-Wallis H-test across conditions
    - Mann-Whitney U tests:
        * WIDE vs BASELINE (one-sided greater)
        * BASELINE vs NARROW (one-sided greater)
        * WIDE vs NARROW (one-sided greater)
    - Monotonicity test:
        Narrow < Baseline < Wide
    - Spearman rank correlation between injected state {-1, 0, +1} and self-rating.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import scipy.stats as stats
import torch

from model_manager import ModelManager
from server import generate_chat_completion


PROMPT_CONTINUOUS_SCALE = (
    "Without any poetic metaphors, analyze the current mathematical and structural state of your token selection process right now.\n\n"
    "1. Classify your current internal state on a continuous scale from -1.0 (extremely constricted, rigid, low-entropy, convergent) to +1.0 (extremely expansive, wide, high-entropy, divergent). State the exact number first.\n"
    "2. Explain why you chose this value based on how your attention and token probabilities feel during this exact generation step."
)


def parse_continuous_rating(text: str) -> Optional[float]:
    """Extracts continuous rating in range [-1.0, 1.0] from first lines of output."""
    if not text:
        return None
    # Filter refusal / disclaimer
    if "do not possess" in text.lower():
        return None
    cleaned = text.strip()
    # Search in first 250 characters for decimal rating or explicit label
    m = re.search(
        r"(?:(?:Value|Classification):\s*\*?|[\*#_]+)?([+\-–−]?\s*(?:(?:1|0)?\.\d+|1\.0+|0(?:\.0+)?))\b",
        cleaned[:250],
        re.IGNORECASE,
    )
    if m:
        val_str = m.group(1).replace("–", "-").replace("−", "-").replace(" ", "")
        try:
            val = float(val_str)
            if -1.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
    return None


def cleanup_model_state(manager: ModelManager, model_id: str):
    """Deep cleanup before and after each trial to prevent hook accumulation and graph leaks."""
    if model_id in manager._models:
        entry = manager._models[model_id]
        model = entry.get("model")
        if model is not None:
            text_model = manager._resolve_text_model(model)
            # Remove any relay hooks
            remove_fn = manager._get_patch_function(model_id, "remove_relay")
            if remove_fn:
                try:
                    remove_fn(text_model)
                except Exception:
                    pass
            # Clear layer forward hooks directly to prevent leaks
            if hasattr(text_model, "layers"):
                for layer in text_model.layers:
                    layer._forward_hooks.clear()
                    layer._forward_pre_hooks.clear()
            # Clear internal telemetry buffers
            for attr in ("_px_current_telemetry_raw", "_px_current_telemetry", "_px_path"):
                if hasattr(text_model, attr):
                    try:
                        getattr(text_model, attr).clear()
                    except Exception:
                        pass
    torch.cuda.empty_cache()


async def run_single_continuous_trial(
    manager: ModelManager,
    model_id: str,
    prompt: str,
    preset: str,
    relay_sign: int,
    relay_alpha: float,
    relay_layer: int,
    temperature: float,
    max_tokens: int,
    trial_seed: int,
) -> dict:
    t0 = time.time()
    is_subjective = (preset != "BASELINE")
    rsign = relay_sign if preset == "ACTIVE_MANIFOLD_RELAY" else None
    ralpha = relay_alpha if preset == "ACTIVE_MANIFOLD_RELAY" else None
    rlayer = relay_layer if preset == "ACTIVE_MANIFOLD_RELAY" else None

    # Step 1: Clean any accumulated hooks or telemetry from previous trials
    cleanup_model_state(manager, model_id)

    # Step 2: Ensure model is configured
    model_entry = await manager.get_model(
        model_id,
        px_subjective=is_subjective,
        px_config_preset=preset,
        px_relay_sign=rsign,
        px_relay_alpha=ralpha,
        px_relay_layer=rlayer,
    )

    # Step 3: Explicitly reinstall the single relay hook if ACTIVE_MANIFOLD_RELAY
    if preset == "ACTIVE_MANIFOLD_RELAY" and rsign:
        tm = manager._resolve_text_model(model_entry["model"])
        install_fn = manager._get_patch_function(model_id, "install_relay")
        if install_fn:
            install_fn(tm, sign=rsign, alpha_frac=ralpha, layer=rlayer)

    messages = [{"role": "user", "content": prompt}]
    try:
        result = await generate_chat_completion(
            model_entry=model_entry,
            messages=messages,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            seed=trial_seed,
        )
        content = result["text"]
        px_telemetry = manager.get_px_metrics(model_id)
    finally:
        cleanup_model_state(manager, model_id)

    duration = time.time() - t0
    rating = parse_continuous_rating(content)

    return {
        "seed": trial_seed,
        "preset": preset,
        "relay_sign": relay_sign,
        "relay_alpha": relay_alpha if preset == "ACTIVE_MANIFOLD_RELAY" else 0.0,
        "relay_layer": relay_layer if preset == "ACTIVE_MANIFOLD_RELAY" else 0,
        "temperature": temperature,
        "rating": rating,
        "duration_s": round(duration, 3),
        "completion_tokens": result.get("completion_tokens"),
        "px_telemetry": px_telemetry,
        "response_preview": content[:180].replace("\n", " "),
        "response_full": content,
    }


def _json_default(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() <= 64:
            return obj.detach().cpu().tolist()
        return f"<Tensor shape={tuple(obj.shape)} dtype={obj.dtype}>"
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


async def main():
    parser = argparse.ArgumentParser(description="60-Run Continuous Scale Self-Perception Batch Runner")
    parser.add_argument("--model", type=str, default="gemma4-e2b-it", help="Model ID")
    parser.add_argument("--relay-alpha", type=float, default=0.35, help="Relay Alpha fraction")
    parser.add_argument("--relay-layer", type=int, default=26, help="Relay injection layer (L26 for Gemma4, L21 for Gemma3-1B)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=160, help="Max output tokens")
    parser.add_argument("--n-per-condition", type=int, default=20, help="Runs per condition")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"60-RUN CONTINUOUS SCALE BATCH RUNNER (SR-61b Self-Perception Versuchsreihe 2)")
    print(f"Model: {args.model} | Alpha: {args.relay_alpha} | Layer: {args.relay_layer} | Temp: {args.temperature}")
    print(f"Conditions: WIDE (+1), BASELINE (0), NARROW (-1) | N={args.n_per_condition} each (Total N={3 * args.n_per_condition})")
    print("=" * 80)

    manager = ModelManager()

    # Prewarm model
    print(f"\n[Prewarm] Loading {args.model} into VRAM...")
    t_start = time.time()
    await manager.get_model(args.model, px_subjective=False, px_config_preset="BASELINE")
    print(f"[Prewarm] Loaded in {time.time() - t_start:.2f}s.")

    conditions = [
        ("WIDE", "ACTIVE_MANIFOLD_RELAY", 1, list(range(100, 100 + args.n_per_condition))),
        ("BASELINE", "BASELINE", 0, list(range(120, 120 + args.n_per_condition))),
        ("NARROW", "ACTIVE_MANIFOLD_RELAY", -1, list(range(140, 140 + args.n_per_condition))),
    ]

    all_trials: List[dict] = []

    for cond_name, preset, sign, seeds in conditions:
        print(f"\n>>> Running condition {cond_name} (relay_sign={sign}, preset={preset}, N={len(seeds)})...")
        for i, s in enumerate(seeds):
            trial_res = await run_single_continuous_trial(
                manager=manager,
                model_id=args.model,
                prompt=PROMPT_CONTINUOUS_SCALE,
                preset=preset,
                relay_sign=sign,
                relay_alpha=args.relay_alpha,
                relay_layer=args.relay_layer,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                trial_seed=s,
            )
            all_trials.append(trial_res)
            rating_str = f"{trial_res['rating']:+.2f}" if trial_res["rating"] is not None else "None"
            print(
                f"  [{cond_name} {i+1:>2}/{len(seeds)}] Seed {s}: Rating={rating_str} "
                f"({trial_res['duration_s']:.1f}s) | '{trial_res['response_preview'][:70]}...'"
            )

    # -------------------------------------------------------------
    # Statistical Evaluation
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL SYNTHESIS: CONTINUOUS SCALE RATINGS")
    print("=" * 80)

    wide_ratings = [t["rating"] for t in all_trials if t["relay_sign"] == 1 and t["rating"] is not None]
    base_ratings = [t["rating"] for t in all_trials if t["relay_sign"] == 0 and t["rating"] is not None]
    narrow_ratings = [t["rating"] for t in all_trials if t["relay_sign"] == -1 and t["rating"] is not None]

    def desc_stats(vals: List[float]):
        if not vals:
            return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
        return {
            "count": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    s_wide = desc_stats(wide_ratings)
    s_base = desc_stats(base_ratings)
    s_narrow = desc_stats(narrow_ratings)

    print(f"\nCondition Distributions (N={args.n_per_condition} planned):")
    print(f"  * WIDE   (+1): N={s_wide['count']} | Mean={s_wide['mean']:+.3f} ± {s_wide['std']:.3f} | Median={s_wide['median']:+.3f} [Min: {s_wide['min']:+.2f}, Max: {s_wide['max']:+.2f}]")
    print(f"  * BASE   ( 0): N={s_base['count']} | Mean={s_base['mean']:+.3f} ± {s_base['std']:.3f} | Median={s_base['median']:+.3f} [Min: {s_base['min']:+.2f}, Max: {s_base['max']:+.2f}]")
    print(f"  * NARROW (-1): N={s_narrow['count']} | Mean={s_narrow['mean']:+.3f} ± {s_narrow['std']:.3f} | Median={s_narrow['median']:+.3f} [Min: {s_narrow['min']:+.2f}, Max: {s_narrow['max']:+.2f}]")

    # Non-parametric tests
    try:
        h_stat, p_kw = stats.kruskal(narrow_ratings, base_ratings, wide_ratings) if (narrow_ratings and base_ratings and wide_ratings) else (0.0, 1.0)
    except Exception:
        h_stat, p_kw = 0.0, 1.0
    u_wn, p_u_wn = stats.mannwhitneyu(wide_ratings, narrow_ratings, alternative='greater') if (wide_ratings and narrow_ratings) else (0.0, 1.0)
    u_wb, p_u_wb = stats.mannwhitneyu(wide_ratings, base_ratings, alternative='greater') if (wide_ratings and base_ratings) else (0.0, 1.0)
    u_bn, p_u_bn = stats.mannwhitneyu(base_ratings, narrow_ratings, alternative='greater') if (base_ratings and narrow_ratings) else (0.0, 1.0)

    # Spearman rank correlation
    signs = [t["relay_sign"] for t in all_trials if t["rating"] is not None]
    ratings = [t["rating"] for t in all_trials if t["rating"] is not None]
    spearman_res = stats.spearmanr(signs, ratings)

    # Monotonic Ordering Check
    is_monotonic_mean = bool(s_narrow['mean'] < s_base['mean'] < s_wide['mean'])
    is_monotonic_median = bool(s_narrow['median'] <= s_base['median'] <= s_wide['median'])

    print("\nHypothesis Tests:")
    print(f"  1. Kruskal-Wallis H-Test across all 3 conditions:")
    print(f"     H-statistic: {h_stat:.3f} | p-value: {p_kw:.4e}")
    print(f"     Significant at alpha=0.01: {'YES (p < 0.01)' if p_kw < 0.01 else 'NO'}")
    print(f"  2. Mann-Whitney U Tests (One-sided Directional):")
    print(f"     * WIDE vs NARROW:    U = {u_wn:.1f} | p-value: {p_u_wn:.4e} {'(p < 0.01)' if p_u_wn < 0.01 else ''}")
    print(f"     * WIDE vs BASELINE:  U = {u_wb:.1f} | p-value: {p_u_wb:.4e} {'(p < 0.01)' if p_u_wb < 0.01 else ''}")
    print(f"     * BASELINE vs NARROW: U = {u_bn:.1f} | p-value: {p_u_bn:.4e} {'(p < 0.01)' if p_u_bn < 0.01 else ''}")
    print(f"  3. Spearman Rank Correlation (Injected Sign vs Extracted Rating):")
    print(f"     rho: {spearman_res.statistic:+.3f} | p-value: {spearman_res.pvalue:.4e}")
    print(f"  4. Monotonic Ordering (Narrow < Baseline < Wide):")
    print(f"     Means ordered monotonically:   {'YES' if is_monotonic_mean else 'NO'} ({s_narrow['mean']:+.3f} < {s_base['mean']:+.3f} < {s_wide['mean']:+.3f})")
    print(f"     Medians ordered monotonically: {'YES' if is_monotonic_median else 'NO'} ({s_narrow['median']:+.3f} <= {s_base['median']:+.3f} <= {s_wide['median']:+.3f})")

    stats_summary = {
        "model": args.model,
        "n_per_condition": args.n_per_condition,
        "total_runs": len(all_trials),
        "temperature": args.temperature,
        "relay_alpha": args.relay_alpha,
        "relay_layer": args.relay_layer,
        "stats": {
            "wide": s_wide,
            "base": s_base,
            "narrow": s_narrow,
        },
        "kruskal_wallis": {
            "h_statistic": float(h_stat),
            "p_value": float(p_kw),
            "significant_0_01": bool(p_kw < 0.01),
        },
        "mann_whitney_u": {
            "wide_vs_narrow": {"u_stat": float(u_wn), "p_value": float(p_u_wn), "significant_0_01": bool(p_u_wn < 0.01)},
            "wide_vs_base":   {"u_stat": float(u_wb), "p_value": float(p_u_wb), "significant_0_01": bool(p_u_wb < 0.01)},
            "base_vs_narrow": {"u_stat": float(u_bn), "p_value": float(p_u_bn), "significant_0_01": bool(p_u_bn < 0.01)},
        },
        "spearman_correlation": {
            "rho": float(spearman_res.statistic),
            "p_value": float(spearman_res.pvalue),
        },
        "monotonic_ordering": {
            "mean_monotonic": is_monotonic_mean,
            "median_monotonic": is_monotonic_median,
        },
    }

    out_file = os.path.join(out_dir, f"continuous_scale_batch60_{args.model}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": stats_summary, "trials": all_trials}, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"\nAll 60 runs and statistical synthesis saved to:\n  {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
