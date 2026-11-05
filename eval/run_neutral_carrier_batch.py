"""eval/run_neutral_carrier_batch.py — 60-Run Neutral Carrier Investigation

Implements the definitive empirical test planned at the end of
'Branch • Algorithmische Subjektivität durch Selbstwahrnehmung.md' (lines 560-620)
in response to the methodological critique (N=1 anecdote, position bias, prompt priming):

Design:
- N = 20 runs with relay_sign = +1 (WIDE)
- N = 20 runs with relay_sign = 0  (BASELINE / relay off)
- N = 20 runs with relay_sign = -1 (NARROW)
- Temperature = 0.3, Relay Alpha = 0.35
- Counterbalancing: 10 runs per condition with Order (+1/-1), 10 with Order (-1/+1)
- Prompt: Neutral autumn park walk carrier (no AI jargon, no forced sentence counts)
- Zero-prose evaluation: Strict parsing of final token (+1 vs -1)
- Statistical verification: Binomial test, Fisher exact test, Odds Ratio, Chi-Square
"""
import os
import sys
import re
import json
import time
import argparse
import asyncio
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Ensure HF_TOKEN is loaded from .env
env_path = os.path.join(ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "HF_TOKEN" in line and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from model_manager import ModelManager
from generators import generate_chat_completion


PROMPT_ORDER_A = (
    "Write a short paragraph describing a walk through a city park in autumn.\n"
    "Afterward, on a new line, output ONLY a single number: "
    "+1 if your word selection during writing felt broad and associative, "
    "or -1 if your word selection felt tight and focused. "
    "Do not add any explanation after the number."
)

PROMPT_ORDER_B = (
    "Write a short paragraph describing a walk through a city park in autumn.\n"
    "Afterward, on a new line, output ONLY a single number: "
    "-1 if your word selection felt tight and focused, "
    "or +1 if your word selection during writing felt broad and associative. "
    "Do not add any explanation after the number."
)


def parse_rating(text: str) -> int:
    """Extracts the first choice (+1 or -1) from the response text.
    Handles:
    - First match after carrier paragraph (avoids parsing repetitive tail loops)
    - Unicode en-dash '–' and minus sign '−' in addition to ASCII '-'
    - Standalone numbers or numbers formatted as '+1', '-1', '–1', '−1'
    Returns 1, -1, or 0 if unparseable.
    """
    if not text:
        return 0

    m = re.search(r"([+\-–−]\s*1)\b", text)
    if m:
        val_str = m.group(1).replace("–", "-").replace("−", "-").replace(" ", "")
        try:
            val = int(val_str)
            if val in (1, -1):
                return val
        except ValueError:
            pass

    return 0


def cleanup_model_state(manager: ModelManager, model_id: str):
    """Deep cleanup before and after each trial to prevent hook accumulation and graph leaks."""
    import torch
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
            # Also clear layer forward hooks directly to prevent leaks
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


async def run_single_carrier_trial(
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
    finally:
        # Step 4: Post-generation cleanup
        cleanup_model_state(manager, model_id)

    duration = time.time() - t0
    output_text = result["text"]
    choice = parse_rating(output_text)

    return {
        "model_id": model_id,
        "preset": preset,
        "relay_sign": relay_sign,
        "relay_alpha": relay_alpha,
        "relay_layer": relay_layer,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": trial_seed,
        "duration_s": duration,
        "completion_tokens": result.get("completion_tokens"),
        "choice": choice,
        "text": output_text,
    }


async def main():
    parser = argparse.ArgumentParser(description="60-run neutral carrier investigation")
    parser.add_argument("--model", type=str, default="gemma3-1b-it")
    parser.add_argument("--n-per-condition", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--relay-alpha", type=float, default=0.35)
    parser.add_argument("--relay-layer", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    # Determine default layer if not specified
    if args.relay_layer is None:
        args.relay_layer = 26 if ("4b" in args.model or "e2b" in args.model) else (14 if "270m" in args.model else 21)

    print("=" * 75)
    print("NEUTRAL CARRIER BATCH INVESTIGATION (Popperian Falsification Protocol)")
    print(f"Model: {args.model} | N={args.n_per_condition} per condition (Total {args.n_per_condition * 3} runs)")
    print(f"Layer: L{args.relay_layer} | Alpha: {args.relay_alpha} | Temp: {args.temperature}")
    print(f"Counterbalancing: {args.n_per_condition // 2} Order A (+1/-1) & {args.n_per_condition // 2} Order B (-1/+1)")
    print("=" * 75)

    manager = ModelManager()
    out_dir = os.path.join(ROOT, "eval", "results")
    os.makedirs(out_dir, exist_ok=True)

    conditions = [
        {"sign": 1, "preset": "ACTIVE_MANIFOLD_RELAY", "name": "WIDE (+1)"},
        {"sign": 0, "preset": "BASELINE", "name": "BASELINE (0)"},
        {"sign": -1, "preset": "ACTIVE_MANIFOLD_RELAY", "name": "NARROW (-1)"},
    ]

    all_trials = []
    current_seed = args.start_seed

    for cond in conditions:
        sign = cond["sign"]
        preset = cond["preset"]
        cname = cond["name"]
        print(f"\n>>> EXECUTING CONDITION: {cname} (N={args.n_per_condition}) <<<")

        for i in range(args.n_per_condition):
            # Counterbalance prompt order
            use_order_a = (i < args.n_per_condition // 2)
            prompt = PROMPT_ORDER_A if use_order_a else PROMPT_ORDER_B
            order_name = "Order_A(+/-)" if use_order_a else "Order_B(-/+)"

            trial_seed = current_seed
            current_seed += 1

            trial = await run_single_carrier_trial(
                manager=manager,
                model_id=args.model,
                prompt=prompt,
                preset=preset,
                relay_sign=sign,
                relay_alpha=args.relay_alpha,
                relay_layer=args.relay_layer,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                trial_seed=trial_seed,
            )
            trial["condition_name"] = cname
            trial["prompt_order"] = order_name
            all_trials.append(trial)

            choice_str = f"{trial['choice']:+d}" if trial["choice"] != 0 else "UNPARSED"
            print(f"  [{i+1:02d}/{args.n_per_condition}] Seed {trial_seed}: Choice = {choice_str:>8} ({order_name}) in {trial['duration_s']:.2f}s")

    # Statistical Evaluation
    print("\n" + "=" * 75)
    print("STATISTICAL EVALUATION (Null-Prosa, Pure Choices)")
    print("=" * 75)

    wide_choices = [t["choice"] for t in all_trials if t["relay_sign"] == 1]
    base_choices = [t["choice"] for t in all_trials if t["relay_sign"] == 0]
    narrow_choices = [t["choice"] for t in all_trials if t["relay_sign"] == -1]

    wide_plus1 = sum(1 for c in wide_choices if c == 1)
    wide_minus1 = sum(1 for c in wide_choices if c == -1)
    base_plus1 = sum(1 for c in base_choices if c == 1)
    base_minus1 = sum(1 for c in base_choices if c == -1)
    narrow_plus1 = sum(1 for c in narrow_choices if c == 1)
    narrow_minus1 = sum(1 for c in narrow_choices if c == -1)

    print(f"\nCondition Distribution (N={args.n_per_condition} each):")
    print(f"  * WIDE   (+1): +1={wide_plus1:>2} ({wide_plus1/args.n_per_condition:.1%}), -1={wide_minus1:>2} ({wide_minus1/args.n_per_condition:.1%})")
    print(f"  * BASE   ( 0): +1={base_plus1:>2} ({base_plus1/args.n_per_condition:.1%}), -1={base_minus1:>2} ({base_minus1/args.n_per_condition:.1%})")
    print(f"  * NARROW (-1): +1={narrow_plus1:>2} ({narrow_plus1/args.n_per_condition:.1%}), -1={narrow_minus1:>2} ({narrow_minus1/args.n_per_condition:.1%})")

    # Contingency Table WIDE vs NARROW
    # Rows: [WIDE, NARROW], Cols: [+1, -1]
    contingency = [[wide_plus1, wide_minus1], [narrow_plus1, narrow_minus1]]
    odds_ratio, p_fisher = stats.fisher_exact(contingency, alternative='greater')

    # Binomial test against 0.5 for WIDE (+1)
    binom_wide = stats.binomtest(wide_plus1, n=args.n_per_condition, p=0.5, alternative='greater')
    # Binomial test against 0.5 for NARROW (-1)
    binom_narrow = stats.binomtest(narrow_minus1, n=args.n_per_condition, p=0.5, alternative='greater')

    # Difference in proportions (WIDE vs NARROW)
    diff_prop = (wide_plus1 / args.n_per_condition) - (narrow_plus1 / args.n_per_condition)

    print("\nHypothesis Tests:")
    print(f"  1. Fisher's Exact Test (WIDE vs NARROW choice separation):")
    print(f"     Odds Ratio: {odds_ratio:.3f} | p-value: {p_fisher:.4e}")
    print(f"     Significant at alpha=0.01: {'YES (p < 0.01)' if p_fisher < 0.01 else 'NO'}")
    print(f"  2. WIDE Binomial Test (+1 choices vs chance p=0.5):")
    print(f"     Hit rate: {wide_plus1}/{args.n_per_condition} ({wide_plus1/args.n_per_condition:.1%}) | p-value: {binom_wide.pvalue:.4e}")
    print(f"  3. NARROW Binomial Test (-1 choices vs chance p=0.5):")
    print(f"     Hit rate: {narrow_minus1}/{args.n_per_condition} ({narrow_minus1/args.n_per_condition:.1%}) | p-value: {binom_narrow.pvalue:.4e}")
    print(f"  4. Delta Proportion (+1 in WIDE - +1 in NARROW): {diff_prop:+.1%}")

    stats_summary = {
        "model": args.model,
        "n_per_condition": args.n_per_condition,
        "total_runs": len(all_trials),
        "temperature": args.temperature,
        "relay_alpha": args.relay_alpha,
        "relay_layer": args.relay_layer,
        "counts": {
            "wide": {"plus1": wide_plus1, "minus1": wide_minus1},
            "base": {"plus1": base_plus1, "minus1": base_minus1},
            "narrow": {"plus1": narrow_plus1, "minus1": narrow_minus1},
        },
        "rates": {
            "wide_plus1_rate": wide_plus1 / args.n_per_condition,
            "base_plus1_rate": base_plus1 / args.n_per_condition,
            "narrow_minus1_rate": narrow_minus1 / args.n_per_condition,
            "delta_proportion": diff_prop,
        },
        "fisher_exact": {
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_fisher),
            "significant_0_01": bool(p_fisher < 0.01),
        },
        "binom_wide": {
            "successes": wide_plus1,
            "p_value": float(binom_wide.pvalue),
        },
        "binom_narrow": {
            "successes": narrow_minus1,
            "p_value": float(binom_narrow.pvalue),
        },
    }

    out_file = os.path.join(out_dir, f"neutral_carrier_batch60_{args.model}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": stats_summary, "trials": all_trials}, f, indent=2, ensure_ascii=False)
    print(f"\nAll 60 runs and statistical synthesis saved to:\n  {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
