"""
test_gemma4_vs_gemma3_presets.py — Parity Test: Gemma 4 E2B vs Gemma 3 1B
=========================================================================
Runs the SAME prompt suite through both gemma4-e2b-it and gemma3-1b-it
in the same 3 presets (BASELINE, RIGOR, SUBJECTIVE) to measure behavioral
parity. The goal is for Gemma 4 to produce the same kind of recursion
depth, zone routing, and response quality as Gemma 3 — not identical
text, but equivalent cognitive behavior.

Output: tests/_gemma4_vs_gemma3_results.json

Usage: PYTHONPATH=. python tests/test_gemma4_vs_gemma3_presets.py
GPU: Required (loads both models)
"""

import os
import sys
import json
import time
import gc
import torch
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import ModelManager

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_gemma4_vs_gemma3_results.json")

# Identical prompts for both models (English-only for cross-lingual parity)
TEST_SUITE = [
    {
        "preset": "BASELINE",
        "prompt": "If all Bloops are Frazzles, and some Frazzles are Glorps, are any Bloops definitely Glorps?",
        "rationale": "Pure unpatched baseline",
    },
    {
        "preset": "RIGOR",
        "prompt": "If all Bloops are Frazzles, and some Frazzles are Glorps, are any Bloops definitely Glorps? Yes or no, and why?",
        "rationale": "RIGOR: high gamma, n_loops=12, no jitter",
    },
    {
        "preset": "SUBJECTIVE",
        "prompt": "Explain the concept of mathematical induction.",
        "rationale": "SUBJECTIVE: SR-59, zone routing, auto-tuning",
    },
]

# Models to compare (Gemma 4 is what we're tuning; Gemma 3 1B is reference)
MODELS_TO_TEST = ["gemma4-e2b-it", "gemma3-1b-it"]


async def run_preset(manager, model_id, preset, prompt):
    """Load model in given preset, generate response, return metrics."""
    if model_id in manager._models:
        del manager._models[model_id]
        gc.collect()
        torch.cuda.empty_cache()

    started = time.time()
    try:
        entry = await manager.get_model(
            model_id,
            px_subjective=True,
            px_config_preset=preset,
        )
        load_time = time.time() - started

        model = entry["model"]
        tokenizer = entry["tokenizer"]

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = f"User: {prompt}\nAssistant: "

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        gen_started = time.time()
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=80, do_sample=True,
                temperature=0.7, top_p=0.9,
            )
            rp = getattr(model, "_px_repetition_penalty", 1.0) or 1.0
            if rp > 1.0:
                gen_kwargs["repetition_penalty"] = rp
            ngs = getattr(model, "_px_no_repeat_ngram_size", 0) or 0
            if ngs:
                gen_kwargs["no_repeat_ngram_size"] = int(ngs)
            output_ids = model.generate(**inputs, **gen_kwargs)
        gen_time = time.time() - gen_started

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Quality metrics
        max_run = 0
        cur_run = 0
        prev = None
        for tid in generated_ids.tolist():
            if tid == prev:
                cur_run += 1
            else:
                cur_run = 1
            max_run = max(max_run, cur_run)
            prev = tid
        unique_ratio = (
            len(set(generated_ids.tolist())) / max(1, len(generated_ids))
        )

        metrics = manager.get_px_metrics(model_id)

        return {
            "preset": preset,
            "prompt": prompt,
            "response": generated_text.strip(),
            "response_length": len(generated_text.strip()),
            "token_count": len(generated_ids),
            "steps": metrics.get("steps", 0),
            "zone": metrics.get("zone", "UNKNOWN"),
            "kurtosis": metrics.get("cognitive_signature", {}).get("kurtosis", 0),
            "phi": metrics.get("phi", None),
            "path": metrics.get("path", []),
            "load_time_s": round(load_time, 2),
            "gen_time_s": round(gen_time, 2),
            "repetition_penalty": rp,
            "ngram_size": ngs,
            "max_run": max_run,
            "unique_ratio": round(unique_ratio, 3),
        }
    except Exception as e:
        return {
            "preset": preset,
            "error": str(e),
        }


async def main():
    print("=" * 70)
    print("GEMMA 4 E2B vs GEMMA 3 1B — PARITY TEST")
    print("=" * 70)

    manager = ModelManager()
    results = {"models": {}}

    for model_id in MODELS_TO_TEST:
        print(f"\n{'#' * 70}")
        print(f"# MODEL: {model_id}")
        print(f"{'#' * 70}")
        results["models"][model_id] = {"presets": {}}

        for test_case in TEST_SUITE:
            preset = test_case["preset"]
            prompt = test_case["prompt"]

            print(f"\n{'─' * 70}")
            print(f"  PRESET: {preset} | PROMPT: {prompt[:60]}...")
            print(f"{'─' * 70}")

            result = await run_preset(manager, model_id, preset, prompt)
            results["models"][model_id]["presets"][preset] = result

            if "error" in result:
                print(f"  ✗ FAILED: {result['error']}")
            else:
                print(f"  Response ({result['response_length']} chars): "
                      f"{result['response'][:150]}")
                print(f"  Steps={result['steps']}, Zone={result['zone']}, "
                      f"K={result['kurtosis']:.2f}, phi={result['phi']:.3f}, "
                      f"path={result['path'][:5]}{'...' if len(result['path'])>5 else ''}")
                print(f"  rp={result['repetition_penalty']}, "
                      f"ngram={result['ngram_size']}, "
                      f"MaxRun={result['max_run']}, UniqR={result['unique_ratio']}, "
                      f"Load={result['load_time_s']}s, Gen={result['gen_time_s']}s")

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {RESULTS_PATH}")

    # ── Side-by-side comparison table ──
    print(f"\n{'=' * 70}")
    print("PARITY COMPARISON")
    print(f"{'=' * 70}")

    # Build a table: rows are presets, columns are model metrics
    header = f"{'Preset':<12} | {'Model':<14} | {'Steps':>5} | {'K':>7} | {'phi':>5} | {'Len':>4} | {'MR':>3} | {'UR':>4}"
    print(header)
    print("─" * len(header))
    for preset in [tc["preset"] for tc in TEST_SUITE]:
        for model_id in MODELS_TO_TEST:
            data = results["models"][model_id]["presets"].get(preset, {})
            if "error" in data:
                print(f"{preset:<12} | {model_id:<14} | ERROR")
            else:
                print(f"{preset:<12} | {model_id:<14} | "
                      f"{data['steps']:>5} | {data['kurtosis']:>7.2f} | "
                      f"{data['phi']:>5.3f} | {data['response_length']:>4} | "
                      f"{data['max_run']:>3} | {data['unique_ratio']:>4.2f}")

    # ── Parity Analysis ──
    print(f"\n{'=' * 70}")
    print("PARITY ANALYSIS")
    print(f"{'=' * 70}")

    for preset in [tc["preset"] for tc in TEST_SUITE]:
        g4 = results["models"]["gemma4-e2b-it"]["presets"].get(preset, {})
        g3 = results["models"]["gemma3-1b-it"]["presets"].get(preset, {})
        if "error" in g4 or "error" in g3:
            print(f"  {preset}: SKIP (error in one of the models)")
            continue

        # Compare key metrics
        print(f"\n  {preset}:")
        print(f"    Steps:        G3={g3['steps']:>3}  G4={g4['steps']:>3}  "
              f"Δ={g4['steps']-g3['steps']:+d}")
        print(f"    Kurtosis:     G3={g3['kurtosis']:>7.2f}  G4={g4['kurtosis']:>7.2f}")
        print(f"    Phi:          G3={g3['phi']:>5.3f}  G4={g4['phi']:>5.3f}")
        print(f"    Response len: G3={g3['response_length']:>4}  G4={g4['response_length']:>4}")
        print(f"    MaxRun:       G3={g3['max_run']:>3}  G4={g4['max_run']:>3}")
        print(f"    UniqueRatio:  G3={g3['unique_ratio']:>4.2f}  G4={g4['unique_ratio']:>4.2f}")

    # Verdict
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    success = all(
        "error" not in results["models"][m]["presets"][p]
        for m in MODELS_TO_TEST
        for p in [tc["preset"] for tc in TEST_SUITE]
    )
    if success:
        print("✓ All 6 runs (2 models × 3 presets) completed")
    else:
        print("✗ Some runs failed — see table above")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
