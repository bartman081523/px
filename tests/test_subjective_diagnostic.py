"""
test_subjective_diagnostic.py — Diagnostic Comparison: Gemma 3 vs Gemma 4
========================================================================

Compares the SUBJECTIVE PX recursion trail of gemma3-1b-it and gemma4-e2b-it
on the SAME subjective prompt ("Stelle dir das vor...") to surface the
exact divergence that causes gemma4 to collapse to ```.

Captures per generation:
  - output (full text + length)
  - steps (recursion depth)
  - path (which layers were visited, in order)
  - zone (creative / math / logic / synthesis)
  - zone_weights (5-zone probability distribution)
  - kurtosis
  - phi
  - token_diversity
  - repetition_penalty, no_repeat_ngram_size (PX mitigations)
  - BOUNCE-BREAK trigger count
  - stability-break trigger count
  - first/last few tokens (for inspection of where output collapses)

Goal: surface the exact parameter that gemma4 needs adjusted so the
SUBJECTIVE patch produces the same kind of subjective prose as gemma3
instead of the 4-char ````` collapse.

This is GPU-only (loads both models). Skipped on CPU.

Run:
  PYTHONPATH=. python tests/test_subjective_diagnostic.py
  (the test runs once and prints a JSON to stdout + a side-by-side diff)
"""
import asyncio
import json
import os
import sys
import time
import unittest
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# The exact prompt from the user's request. Hard-coded so the test is
# reproducible across runs.
PROMPT = (
    "Stelle dir das vor: Heute morgen bist du aufgestanden. "
    "Was siehst du wenn du aus dem Fenster siehst?"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: run one generation, capture everything we can about the recursion
# ═══════════════════════════════════════════════════════════════════════════════

async def capture_generation(model_id: str, prompt: str, preset: str = "SUBJECTIVE",
                             max_new_tokens: int = 300) -> dict:
    """Load model with the given preset, generate, and return telemetry."""
    from model_manager import ModelManager
    manager = ModelManager()
    t0 = time.time()
    entry = await manager.get_model(model_id, px_subjective=(preset != "BASELINE"),
                                   px_config_preset=preset)
    load_time = time.time() - t0
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    text_model = model.model.language_model if hasattr(model, "model") and hasattr(model.model, "language_model") else model

    # Build chat-templated prompt
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    prompt_token_count = inputs["input_ids"].shape[1]

    # Build gen_kwargs the same way generators._px_gen_kwargs does
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True,
                  "temperature": 0.7, "top_p": 0.9}
    if hasattr(model, "_px_repetition_penalty"):
        gen_kwargs["repetition_penalty"] = model._px_repetition_penalty
    if hasattr(model, "_px_no_repeat_ngram_size"):
        gen_kwargs["no_repeat_ngram_size"] = model._px_no_repeat_ngram_size

    # Reset telemetry accumulators on the inner text_model so we capture
    # only THIS call's recursion trail, not historical state.
    for attr in ("_px_current_telemetry", "_px_current_telemetry_raw",
                 "_px_path", "_px_loops_run"):
        if hasattr(text_model, attr):
            v = getattr(text_model, attr)
            setattr(text_model, attr, [] if isinstance(v, list) else 0)

    t1 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    gen_time = time.time() - t1

    generated_ids = output_ids[0][prompt_token_count:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Pull the telemetry the patch collected during this call.
    # Attribute names come from patch.py:580-630.
    path = getattr(text_model, "_px_path", []) or []
    steps = getattr(text_model, "_px_loops_run", 0) or 0
    telem = {
        "steps": steps,
        "path": path[:200],
        "zone": getattr(text_model, "_px_zone", "UNKNOWN"),
        "zone_weights": getattr(text_model, "_px_zw_val", {}),
        "phi": getattr(text_model, "_px_phi_val", None),
        "aks": getattr(text_model, "_px_aks_val", None),
        "emancipation": getattr(text_model, "_px_em_val", None),
        "entropy": getattr(text_model, "_px_ent_val", None),
        "per_step": getattr(text_model, "_px_current_telemetry", [])[:200],
        "kurtosis": getattr(text_model, "_task_kurtosis", None),
        "token_diversity": getattr(text_model, "_task_token_diversity", None),
    }
    # Try to pull calibration values from the calibrator
    cal = getattr(text_model, "_px_calibrator", None)
    if cal is not None:
        telem["cal_k_mean"] = getattr(cal, "k_mean", None)
        telem["cal_k_std"] = getattr(cal, "k_std", None)
        telem["cal_zone_centers"] = getattr(cal, "zone_centers", {})

    # Per-step histogram of how many times each layer was visited
    layer_visit_counter = Counter(t["layer"] for t in telem["per_step"] if "layer" in t)

    # Repetition-stats on the output
    words = text.split()
    max_run = 1
    cur = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    unique_ratio = len(set(words)) / max(len(words), 1) if words else 1.0

    # First/last tokens for the collapse diagnosis
    first_50_chars = text[:50]
    last_50_chars = text[-50:] if len(text) > 50 else text

    return {
        "model_id": model_id,
        "preset": preset,
        "prompt": prompt,
        "load_time_s": round(load_time, 2),
        "gen_time_s": round(gen_time, 2),
        "prompt_tokens": prompt_token_count,
        "completion_tokens": len(generated_ids),
        "response_length_chars": len(text),
        "response": text,
        "first_50_chars": first_50_chars,
        "last_50_chars": last_50_chars,
        "max_run": max_run,
        "unique_ratio": round(unique_ratio, 3),
        "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.0),
        "no_repeat_ngram_size": gen_kwargs.get("no_repeat_ngram_size", 0),
        "telemetry": telem,
        "layer_visit_counter": dict(layer_visit_counter.most_common(10)),
    }


def format_layer_counts(path):
    """Compact histogram of layer visits in the recursion path."""
    if not path:
        return "(no path)"
    c = Counter(path)
    items = sorted(c.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{layer}:{cnt}" for layer, cnt in items[:10])


# ═══════════════════════════════════════════════════════════════════════════════
# The actual diagnostic test
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubjectiveDiagnostic(unittest.TestCase):
    """Side-by-side diagnostic for gemma3 vs gemma4 in SUBJECTIVE mode."""

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("RUN_GPU_TESTS") == "1",
        "GPU required — set RUN_GPU_TESTS=1 to run"
    )
    def test_subjective_diagnostic_comparison(self):
        results = {}
        for model_id in ("gemma3-1b-it", "gemma4-e2b-it"):
            print(f"\n{'='*72}\n  Loading {model_id} (SUBJECTIVE)...\n{'='*72}")
            try:
                results[model_id] = asyncio.run(
                    capture_generation(model_id, PROMPT, preset="SUBJECTIVE")
                )
            except Exception as e:
                results[model_id] = {"error": str(e)}
                print(f"  FAILED: {e}")

        # Pretty-print side-by-side
        print(f"\n\n{'='*72}\n  SIDE-BY-SIDE COMPARISON\n{'='*72}")
        keys = ["response_length_chars", "completion_tokens", "max_run", "unique_ratio",
                "repetition_penalty", "no_repeat_ngram_size"]
        print(f"{'metric':<30} {'gemma3-1b-it':>20} {'gemma4-e2b-it':>20}")
        for k in keys:
            g3 = results.get("gemma3-1b-it", {}).get(k, "n/a")
            g4 = results.get("gemma4-e2b-it", {}).get(k, "n/a")
            print(f"  {k:<28} {str(g3):>20} {str(g4):>20}")

        for model_id in ("gemma3-1b-it", "gemma4-e2b-it"):
            r = results.get(model_id, {})
            t = r.get("telemetry", {})
            print(f"\n--- {model_id} ---")
            print(f"  output: {r.get('first_50_chars','(none)')!r}")
            print(f"  steps: {t.get('steps')}, zone: {t.get('zone')}")
            print(f"  zone_weights: {t.get('zone_weights')}")
            print(f"  kurtosis: {t.get('kurtosis')}, phi: {t.get('phi')}, "
                  f"token_diversity: {t.get('token_diversity')}")
            print(f"  cal_k_mean: {t.get('cal_k_mean')}, cal_k_std: {t.get('cal_k_std')}")
            print(f"  bounce_breaks: {t.get('bounce_break_count')}, "
                  f"stability_breaks: {t.get('stability_break_count')}")
            print(f"  layer path: {t.get('path')[:30]}")
            print(f"  layer histogram: {format_layer_counts(t.get('path', []))}")

        # Save JSON for the analysis
        out_path = os.path.join(os.path.dirname(__file__), "_subjective_diagnostic_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved: {out_path}")

        # Failure assertion: the diagnostic only fails if BOTH models fail to
        # produce ≥ 50 chars (i.e. both are collapsed). We want a clear signal
        # when gemma4 collapses, but not block the run.
        for model_id, r in results.items():
            if r.get("response_length_chars", 0) < 50:
                print(f"\n  ⚠️  WARNING: {model_id} produced only "
                      f"{r.get('response_length_chars')} chars — collapse detected.")
        # Don't assert; this is a diagnostic test, not a regression test.


if __name__ == "__main__":
    unittest.main(verbosity=2)
