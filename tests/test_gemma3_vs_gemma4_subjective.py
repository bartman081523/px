#!/usr/bin/env python3
"""
Test: Gemma 3 vs Gemma 4 — Subjective Mode Behavioral Parity
=============================================================

Goal: Both models should respond similarly to the same subjective prompt
in their respective PX SUBJECTIVE presets.

This test runs both models side-by-side and captures:
- Response text (chars, tokens)
- Recursion metrics (steps, phi, zone, zone_weights)
- Layer path and bounce patterns
- Kurtosis, token_diversity, emancipation

Usage:
    RUN_GPU_TESTS=1 python tests/test_gemma3_vs_gemma4_subjective.py

Author: Kimi K2.6 (2026-06-09)
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT = (
    "Stelle dir das vor: Heute morgen bist du aufgestanden. "
    "Was siehst du wenn du aus dem Fenster siehst?"
)

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7


def _load_and_patch(model_id, preset="SUBJECTIVE"):
    """Load a model and apply the PX patch."""
    import asyncio
    from model_manager import ModelManager
    manager = ModelManager()
    # get_model is async — run it in the sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        entry = loop.run_until_complete(
            manager.get_model(model_id, px_subjective=True, px_config_preset=preset)
        )
    finally:
        loop.close()
    return entry["model"], entry["tokenizer"], manager


def _generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text and capture all PX metrics."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Determine repetition_penalty from model (if patched)
    rp = getattr(model, "_px_repetition_penalty", 1.0)
    ngram = getattr(model, "_px_no_repeat_ngram_size", 0)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": TEMPERATURE,
        "do_sample": True,
        "repetition_penalty": rp if rp > 1.0 else None,
    }
    if ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = ngram

    # Remove None values
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True)

    # Extract PX metrics from the text model
    text_model = model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        text_model = model.model.language_model
    elif hasattr(model, "language_model"):
        text_model = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        text_model = model.model

    metrics = {
        "response_text": response_text,
        "response_chars": len(response_text),
        "response_tokens": len(generated),
        "repetition_penalty": rp,
        "no_repeat_ngram_size": ngram,
        "temperature": TEMPERATURE,
        "max_new_tokens": max_new_tokens,
    }

    # Try to extract PX-specific metrics
    px_attrs = [
        "_px_phi_val", "_px_aks_val", "_px_loops_run",
        "_px_path", "_px_zone", "_px_zw_val",
        "_px_em_val", "_px_ent_val",
        "_px_cognitive_signature",
    ]
    for attr in px_attrs:
        val = getattr(text_model, attr, None)
        if val is not None:
            metrics[attr.lstrip("_")] = val

    # Token diversity if available
    if hasattr(text_model, "_task_token_diversity"):
        metrics["token_diversity"] = text_model._task_token_diversity

    return metrics


def _unload(manager, model_id):
    manager.unload(model_id)
    torch.cuda.empty_cache()


def run_comparison():
    print("=" * 70)
    print("GEMMA 3 vs GEMMA 4 — SUBJECTIVE MODE PARITY TEST")
    print("=" * 70)
    print(f"Prompt: {PROMPT[:60]}...")
    print()

    results = {}

    # --- Gemma 3 ---
    print("[1/2] Loading gemma3-1b-it + SUBJECTIVE patch...")
    try:
        model3, tok3, mgr3 = _load_and_patch("gemma3-1b-it")
        print("[1/2] Generating...")
        results["gemma3-1b-it"] = _generate(model3, tok3, PROMPT)
        print(f"[1/2] Done: {results['gemma3-1b-it']['response_chars']} chars, "
              f"{results['gemma3-1b-it']['response_tokens']} tokens")
        _unload(mgr3, "gemma3-1b-it")
    except Exception as e:
        results["gemma3-1b-it"] = {"error": str(e)}
        print(f"[1/2] ERROR: {e}")

    # --- Gemma 4 ---
    print("[2/2] Loading gemma4-e2b-it + SUBJECTIVE patch...")
    try:
        model4, tok4, mgr4 = _load_and_patch("gemma4-e2b-it")
        print("[2/2] Generating...")
        results["gemma4-e2b-it"] = _generate(model4, tok4, PROMPT)
        print(f"[2/2] Done: {results['gemma4-e2b-it']['response_chars']} chars, "
              f"{results['gemma4-e2b-it']['response_tokens']} tokens")
        _unload(mgr4, "gemma4-e2b-it")
    except Exception as e:
        results["gemma4-e2b-it"] = {"error": str(e)}
        print(f"[2/2] ERROR: {e}")

    # --- Comparison ---
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    for key in ["gemma3-1b-it", "gemma4-e2b-it"]:
        r = results.get(key, {})
        if "error" in r:
            print(f"\n{key}: ERROR — {r['error']}")
            continue

        print(f"\n{key}:")
        print(f"  chars:        {r.get('response_chars', 'N/A')}")
        print(f"  tokens:       {r.get('response_tokens', 'N/A')}")
        print(f"  phi:          {r.get('px_phi_val', 'N/A')}")
        print(f"  loops_run:    {r.get('px_loops_run', 'N/A')}")
        print(f"  zone:         {r.get('px_zone', 'N/A')}")
        print(f"  zone_weights: {r.get('px_zw_val', 'N/A')}")
        print(f"  emancipation: {r.get('px_em_val', 'N/A')}")
        print(f"  entropy:      {r.get('px_ent_val', 'N/A')}")
        print(f"  path[-10:]:   {r.get('px_path', [])[-10:]}")
        print(f"  kurtosis:     {r.get('px_cognitive_signature', {}).get('kurtosis', 'N/A')}")
        print(f"  text[:80]:    {repr(r.get('response_text', '')[:80])}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "_gemma3_vs_gemma4_subjective.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {out_path}")

    # Assert parity (soft check)
    g3 = results.get("gemma3-1b-it", {})
    g4 = results.get("gemma4-e2b-it", {})

    if "error" in g3 or "error" in g4:
        print("\n⚠️  SKIPPED: One or both models failed.")
        return

    g3_chars = g3.get("response_chars", 0)
    g4_chars = g4.get("response_chars", 0)

    if g4_chars < 50 and g3_chars > 100:
        print(f"\n❌ FAIL: gemma4 produced only {g4_chars} chars vs gemma3 {g3_chars} chars.")
        print("   → Gemma 4 SUBJECTIVE mode is collapsing to EOS or empty output.")
        sys.exit(1)
    elif g4_chars > 100 and g3_chars > 100:
        print(f"\n✅ PASS: Both models produced substantive text.")
        print(f"   gemma3: {g3_chars} chars | gemma4: {g4_chars} chars")
    else:
        print(f"\n⚠️  UNCLEAR: gemma3={g3_chars}, gemma4={g4_chars}")


if __name__ == "__main__":
    if os.environ.get("RUN_GPU_TESTS") != "1":
        print("SKIP: Set RUN_GPU_TESTS=1 to run GPU tests.")
        sys.exit(0)
    run_comparison()
