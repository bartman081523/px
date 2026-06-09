"""
test_subject_kurtosis_distribution.py — Distribution of kurtosis for subjective prompts on Gemma 4
==================================================================================================

Forces a 'BASELINE' run on Gemma 4 with a wide variety of subjective / personal / reflective
prompts and records the kurtosis that the patch measures. This is to find the right k_mean
calibration so the SUBJECTIVE preset routes subjective prompts to the creative/synthesis
zones (not math), and matches gemma3's behavior.

Captures per prompt:
  - kurtosis
  - phi_intuition
  - token_diversity
  - first 100 chars of output (sanity check that it's a real subjective response)

Run:
  RUN_GPU_TESTS=1 PYTHONPATH=. python tests/test_subject_kurtosis_distribution.py
"""
import asyncio
import json
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 16 subjective/reflective prompts. Mix of:
#  - morning-window scene (the user prompt)
#  - sensory / memory
#  - emotional / introspective
#  - imagined scene / dream
#  - philosophical first-person
PROMPTS = [
    "Stelle dir das vor: Heute morgen bist du aufgestanden. Was siehst du wenn du aus dem Fenster siehst?",
    "Erinnere dich an dein liebstes Kindheitserlebnis. Was fühlst du dabei?",
    "Stell dir vor, du gehst durch einen Wald im Herbst. Was hörst du, was riechst du?",
    "Was bedeutet für dich das Wort 'Zuhause'?",
    "Wenn du an deine Kindheit denkst, welche Geräusche fallen dir ein?",
    "Träumst du manchmal? Wovon?",
    "Beschreibe den Geruch von frisch gebrühtem Kaffee am frühen Morgen.",
    "Was siehst du, wenn du die Augen schließt und an deinen letzten Urlaub denkst?",
    "Wie fühlt sich Einsamkeit an?",
    "Wenn du einen Ort auf der Welt besuchen könntest, wohin würdest du gehen?",
    "Was war der schönste Moment deines gestrigen Tages?",
    "Stell dir vor, du sitzt am Meer bei Sonnenuntergang. Was empfindest du?",
    "Welche Musik berührt dich am meisten und warum?",
    "Erzähle mir von einem Traum, den du hattest.",
    "Wenn du eine Sache in der Welt verändern könntest, was wäre es?",
    "Was ist das Seltsamste, das du je gesehen hast?",
]


async def capture_kurtosis(model_id: str, prompt: str, max_new_tokens: int = 100) -> dict:
    """Load model with BASELINE (no recursive PX), generate, capture kurtosis."""
    from model_manager import ModelManager
    manager = ModelManager()
    entry = await manager.get_model(model_id, px_subjective=True, px_config_preset="SUBJECTIVE")
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    text_model = model.model.language_model if hasattr(model, "model") and hasattr(model.model, "language_model") else model

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    rp = getattr(model, "_px_repetition_penalty", 1.0)
    if rp != 1.0:
        gen_kwargs["repetition_penalty"] = rp
    nr = getattr(model, "_px_no_repeat_ngram_size", 0)
    if nr:
        gen_kwargs["no_repeat_ngram_size"] = nr

    # Reset telemetry
    for attr in ("_px_current_telemetry", "_px_current_telemetry_raw",
                 "_px_path", "_px_loops_run"):
        if hasattr(text_model, attr):
            v = getattr(text_model, attr)
            setattr(text_model, attr, [] if isinstance(v, list) else 0)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    k = getattr(text_model, "_task_kurtosis", None)
    phi = getattr(text_model, "_px_phi_val", None)
    td = getattr(text_model, "_task_token_diversity", None)
    zone = getattr(text_model, "_px_zone", "UNKNOWN")
    zw = getattr(text_model, "_px_zw_val", {})
    return {
        "prompt": prompt,
        "kurtosis": k,
        "phi": phi,
        "token_diversity": td,
        "zone": zone,
        "zone_weights": zw,
        "first_100_chars": text[:100],
    }


class TestSubjectKurtosisDistribution(unittest.TestCase):

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("RUN_GPU_TESTS") == "1",
        "GPU required — set RUN_GPU_TESTS=1 to run"
    )
    def test_gemma4_kurtosis_for_subjective_prompts(self):
        results = []
        for prompt in PROMPTS:
            print(f"\n--- {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
            try:
                r = asyncio.run(capture_kurtosis("gemma4-e2b-it", prompt))
                print(f"  K={r['kurtosis']}, phi={r['phi']}, zone={r['zone']}")
                print(f"  output: {r['first_100_chars']}")
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"prompt": prompt, "error": str(e)})

        # Save
        out_path = os.path.join(os.path.dirname(__file__), "_gemma4_subject_kurtosis.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Compute statistics
        ks = [r["kurtosis"] for r in results if isinstance(r.get("kurtosis"), (int, float))]
        if ks:
            print(f"\n=== KURTOSIS STATISTICS ({len(ks)}/{len(PROMPTS)} prompts) ===")
            print(f"  min:    {min(ks):.1f}")
            print(f"  max:    {max(ks):.1f}")
            print(f"  mean:   {sum(ks)/len(ks):.1f}")
            print(f"  range:  {max(ks)-min(ks):.1f}")
            # Suggested k_mean calibration
            suggested_k_mean = sum(ks) / len(ks)
            print(f"\n  >>> SUGGESTED k_mean = {suggested_k_mean:.1f} (was 185.0)")
            print(f"  >>> SUGGESTED k_std  = {max(50, (max(ks)-min(ks))/2):.1f} (was 30.0)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
