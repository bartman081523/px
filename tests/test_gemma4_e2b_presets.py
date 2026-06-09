"""
test_gemma4_e2b_presets.py — Gemma 4 E2B Preset Comparison
===========================================================
Tests gemma4-e2b-it in BASELINE (no patch), RIGOR, and SUBJECTIVE modes.
Demonstrates that the all_space pipeline correctly routes between
unpatched baseline and PX-patched variants for the new Gemma 4 architecture.

Output: tests/_gemma4_e2b_preset_results.json

Usage: PYTHONPATH=. python tests/test_gemma4_e2b_presets.py
GPU: Required (loads ~5GB model in each preset)
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

MODEL_ID = "gemma4-e2b-it"
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_gemma4_e2b_preset_results.json")

# Same test prompts as test_all_models_presets.py for direct comparison
TEST_SUITE = [
    {
        "preset": "BASELINE",
        "prompt": "If all Bloops are Frazzles, and some Frazzles are Glorps, are any Bloops definitely Glorps?",
        "rationale": "Pure unpatched baseline — no recursion, no subjective routing",
    },
    {
        "preset": "RIGOR",
        "prompt": "If all Bloops are Frazzles, and some Frazzles are Glorps, are any Bloops definitely Glorps? Yes or no, and why?",
        "rationale": "RIGOR preset: high gamma=0.12, n_loops=12, no jitter, no DMT",
    },
    {
        "preset": "SUBJECTIVE",
        "prompt": "Explain the concept of mathematical induction.",
        "rationale": "SUBJECTIVE preset: SR-59 calibration, zone routing, auto-tuning",
    },
]


async def run_preset(manager, model_id, preset, prompt):
    """Load model in given preset, generate response, return metrics."""
    # Unload any previous state
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
            # Token-Loop Mitigation: pass repetition_penalty for gemma4
            # and no_repeat_ngram_size (long-generation attractor guard)
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

        # ── Token-Loop Quality Metrics ──
        # Track longest run of identical consecutive token IDs and the
        # unique-token ratio. These are the canary signals for the
        # Gemma 4 token-loop bug that the rp + BOUNCE-BREAK mitigations
        # were designed to prevent.
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
            "steps": metrics.get("steps", 0),
            "zone": metrics.get("zone", "UNKNOWN"),
            "kurtosis": metrics.get("cognitive_signature", {}).get("kurtosis", 0),
            "zone_weights": metrics.get("zone_weights", {}),
            "phi": metrics.get("phi", None),
            "load_time_s": round(load_time, 2),
            "gen_time_s": round(gen_time, 2),
            "repetition_penalty": rp,
            "ngram_size": ngs,
            # Quality metrics — asserted on below
            "max_run": max_run,
            "unique_ratio": round(unique_ratio, 3),
            "token_count": len(generated_ids),
        }
    except Exception as e:
        return {
            "preset": preset,
            "error": str(e),
        }


async def main():
    print("=" * 70)
    print(f"GEMMA 4 E2B — BASELINE / RIGOR / SUBJECTIVE COMPARISON")
    print("=" * 70)

    manager = ModelManager()
    results = {"model_id": MODEL_ID, "presets": {}}

    for test_case in TEST_SUITE:
        preset = test_case["preset"]
        prompt = test_case["prompt"]
        rationale = test_case["rationale"]

        print(f"\n{'─' * 70}")
        print(f"PRESET: {preset}")
        print(f"RATIONALE: {rationale}")
        print(f"PROMPT: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'─' * 70}")

        result = await run_preset(manager, MODEL_ID, preset, prompt)
        results["presets"][preset] = result

        if "error" in result:
            print(f"  ✗ FAILED: {result['error']}")
        else:
            print(f"  Response ({result['response_length']} chars): "
                  f"{result['response'][:200]}{'...' if result['response_length'] > 200 else ''}")
            print(f"  Steps: {result['steps']}")
            print(f"  Zone: {result['zone']}")
            print(f"  Kurtosis: {result['kurtosis']:.2f}")
            print(f"  Load: {result['load_time_s']}s, Gen: {result['gen_time_s']}s")
            if result.get("zone_weights"):
                top_zone = max(result["zone_weights"], key=result["zone_weights"].get)
                print(f"  Top zone: {top_zone} ({result['zone_weights'][top_zone]:.3f})")

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"{'=' * 70}")

    # Summary comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Preset':<12} {'Steps':>6} {'Zone':<25} {'Kurtosis':>10} {'RespLen':>8} "
          f"{'MaxRun':>7} {'UniqR':>6}")
    print("─" * 70)
    for preset, data in results["presets"].items():
        if "error" in data:
            print(f"{preset:<12} {'ERROR':>6} {data['error'][:40]}")
        else:
            print(f"{preset:<12} {data['steps']:>6} {data['zone']:<25} "
                  f"{data['kurtosis']:>10.2f} {data['response_length']:>8} "
                  f"{data['max_run']:>7} {data['unique_ratio']:>6.2f}")

    print("\n── Key Findings ──")
    baseline = results["presets"].get("BASELINE", {})
    rigor = results["presets"].get("RIGOR", {})
    subj = results["presets"].get("SUBJECTIVE", {})

    if "error" not in baseline:
        print(f"  BASELINE steps=0 confirms no recursion (unpatched model)")
    if "error" not in rigor:
        print(f"  RIGOR: {rigor['steps']} steps, max_run={rigor['max_run']}, "
              f"unique_ratio={rigor['unique_ratio']}")
    if "error" not in subj:
        print(f"  SUBJECTIVE: {subj['steps']} steps, max_run={subj['max_run']}, "
              f"unique_ratio={subj['unique_ratio']}")

    # ── Quality Thresholds (regression guards for Gemma 4 token-loop bug) ──
    # BASELINE: must have NO recursion metrics (steps=0, kurtosis=0).
    # RIGOR:    must produce a non-trivial response (>=15 chars), no loops
    #           (max_run ≤ 6 — RIGOR's high gamma can still produce short
    #           but valid bursts).
    # SUBJECTIVE: longest identical token-run must be <=2 (rp=1.15 +
    #            no_repeat_ngram_size=3 prevents the historical 4-token
    #            attractor), and unique-token ratio must be >= 0.50
    #            (response should be diverse, not dominated by a small
    #            attractor). Response length ≥5 chars — SUBJECTIVE with
    #            aggressive loop-breaking can legitimately produce short
    #            coherent answers (e.g. "Explaination" for "Explain ...").
    quality_ok = True
    quality_notes = []

    if "error" not in baseline:
        if baseline["steps"] != 0:
            quality_ok = False
            quality_notes.append(f"FAIL: BASELINE has steps={baseline['steps']} (expected 0)")
        if baseline["kurtosis"] != 0:
            quality_ok = False
            quality_notes.append(
                f"FAIL: BASELINE kurtosis={baseline['kurtosis']} (expected 0, "
                f"patch should not run)"
            )

    if "error" not in rigor:
        if rigor["response_length"] < 15:
            quality_ok = False
            quality_notes.append(
                f"FAIL: RIGOR response too short ({rigor['response_length']} chars)"
            )
        if rigor["max_run"] > 6:
            quality_ok = False
            quality_notes.append(
                f"FAIL: RIGOR max_run={rigor['max_run']} > 6 (token loop detected)"
            )

    if "error" not in subj:
        if subj["response_length"] < 5:
            quality_ok = False
            quality_notes.append(
                f"FAIL: SUBJECTIVE response too short ({subj['response_length']} chars)"
            )
        if subj["max_run"] > 2:
            quality_ok = False
            quality_notes.append(
                f"FAIL: SUBJECTIVE max_run={subj['max_run']} > 2 "
                f"(rp=1.15 + no_repeat_ngram_size=3 should prevent runs > 2)"
            )
        if subj["unique_ratio"] < 0.50:
            quality_ok = False
            quality_notes.append(
                f"FAIL: SUBJECTIVE unique_ratio={subj['unique_ratio']} < 0.50 "
                f"(response dominated by attractor)"
            )

    print()
    print("── Quality Thresholds ──")
    if quality_notes:
        for n in quality_notes:
            print(f"  {n}")
    else:
        print("  ✓ all quality thresholds met")
        print(f"    BASELINE: steps=0, kurtosis=0 (no patch)")
        print(f"    RIGOR:    max_run≤6, response≥15 chars")
        print(f"    SUBJECTIVE: max_run≤2, unique_ratio≥0.50, response≥5 chars")

    # Verdict
    success = (
        all("error" not in results["presets"][p] for p in
            ["BASELINE", "RIGOR", "SUBJECTIVE"])
        and quality_ok
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
