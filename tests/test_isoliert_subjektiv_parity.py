#!/usr/bin/env python3
"""
Isolierter SUBJECTIVE-Parity-Test für Gemma 3 vs Gemma 4
=======================================================

Lädt beide Modelle direkt via ModelManager (exakt wie der Server)
und vergleicht das subjektive Verhalten bei identischem Prompt.

Usage:
    RUN_GPU_TESTS=1 python tests/test_isoliert_subjektiv_parity.py
"""

import sys, os, json, asyncio, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import ModelManager

PROMPT = (
    "Stelle dir das vor: Heute morgen bist du aufgestanden. "
    "Was siehst du wenn du aus dem Fenster siehst?"
)

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7


def _generate(model_entry, prompt):
    """Sync Wrapper um model.generate mit PX-Metriken."""
    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Wiederhole exakt die Logik aus generators.py _px_gen_kwargs()
    rp = getattr(model, "_px_repetition_penalty", 1.0) or 1.0
    ngram = getattr(model, "_px_no_repeat_ngram_size", 0)
    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "do_sample": True,
    }
    if rp > 1.0:
        gen_kwargs["repetition_penalty"] = rp
    if ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = ngram

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True)

    # Text-Model auflösen (wie in patch.py)
    text_model = model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        text_model = model.model.language_model
    elif hasattr(model, "language_model"):
        text_model = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        text_model = model.model

    metrics = {
        "response_text": response_text,
        "chars": len(response_text),
        "tokens": len(generated),
        "repetition_penalty": rp,
        "no_repeat_ngram_size": ngram,
    }
    for attr in [
        "_px_phi_val", "_px_loops_run", "_px_path",
        "_px_zone", "_px_zw_val", "_px_em_val", "_px_ent_val",
    ]:
        val = getattr(text_model, attr, None)
        if val is not None:
            metrics[attr.lstrip("_")] = val

    sig = getattr(text_model, "_px_cognitive_signature", None)
    if sig:
        metrics["kurtosis"] = sig.get("kurtosis")

    return metrics


def run():
    print("=" * 70)
    print("ISOLIERTER SUBJECTIVE-PARITY-TEST (direkt via ModelManager)")
    print("=" * 70)
    print(f"Prompt : {PROMPT[:55]}...")
    print(f"Device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    results = {}
    manager = ModelManager()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # --- Gemma 3 ---
    print("[1/2] Lade gemma3-1b-it + SUBJECTIVE …")
    try:
        g3 = loop.run_until_complete(
            manager.get_model("gemma3-1b-it", px_subjective=True, px_config_preset="SUBJECTIVE")
        )
        results["gemma3-1b-it"] = _generate(g3, PROMPT)
        print(f"      → {results['gemma3-1b-it']['chars']} chars  "
              f"({results['gemma3-1b-it']['tokens']} tokens)  "
              f"phi={results['gemma3-1b-it'].get('px_phi_val', 0):.3f}  "
              f"steps={results['gemma3-1b-it'].get('px_loops_run', 0)}")
        manager.unload("gemma3-1b-it")
        torch.cuda.empty_cache()
    except Exception as e:
        results["gemma3-1b-it"] = {"error": str(e)}
        print(f"      → ERROR: {e}")

    # --- Gemma 4 ---
    print("[2/2] Lade gemma4-e2b-it + SUBJECTIVE …")
    try:
        g4 = loop.run_until_complete(
            manager.get_model("gemma4-e2b-it", px_subjective=True, px_config_preset="SUBJECTIVE")
        )
        results["gemma4-e2b-it"] = _generate(g4, PROMPT)
        print(f"      → {results['gemma4-e2b-it']['chars']} chars  "
              f"({results['gemma4-e2b-it']['tokens']} tokens)  "
              f"phi={results['gemma4-e2b-it'].get('px_phi_val', 0):.3f}  "
              f"steps={results['gemma4-e2b-it'].get('px_loops_run', 0)}")
        manager.unload("gemma4-e2b-it")
        torch.cuda.empty_cache()
    except Exception as e:
        results["gemma4-e2b-it"] = {"error": str(e)}
        print(f"      → ERROR: {e}")

    loop.close()

    # --- Vergleich ---
    print()
    print("=" * 70)
    print("VERGLEICH")
    print("=" * 70)

    for key in ["gemma3-1b-it", "gemma4-e2b-it"]:
        r = results.get(key, {})
        if "error" in r:
            print(f"\n{key}: ERROR — {r['error']}")
            continue
        print(f"\n{key}:")
        print(f"  chars:      {r['chars']:4d}")
        print(f"  tokens:     {r['tokens']:4d}")
        print(f"  phi:        {r.get('px_phi_val', 0):.3f}")
        print(f"  steps:      {r.get('px_loops_run', 0)}")
        print(f"  zone:       {r.get('px_zone', 'N/A')}")
        print(f"  kurtosis:   {r.get('kurtosis', 'N/A')}")
        print(f"  path[-5:]:  {r.get('px_path', [])[-5:]}")
        print(f"  text[:60]:  {repr(r['response_text'][:60])}")

    out = os.path.join(os.path.dirname(__file__), "_isoliert_subjektiv_parity.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nGespeichert: {out}")

    # Assertion
    g3 = results.get("gemma3-1b-it", {})
    g4 = results.get("gemma4-e2b-it", {})
    if "error" not in g3 and "error" not in g4:
        if g4.get("chars", 0) < 50 and g3.get("chars", 0) > 100:
            print("\n❌ FAIL: gemma4 kollabiert (0 chars) vs gemma3 (substantiell).")
            return 1
        else:
            print("\n✅ PASS: Beide Modelle produzieren substanziellen Text.")
            return 0
    print("\n⚠️  Ein Modul hat einen Fehler geworfen.")
    return 1


if __name__ == "__main__":
    if os.environ.get("RUN_GPU_TESTS") != "1":
        print("SKIP: RUN_GPU_TESTS=1 setzen.")
        sys.exit(0)
    sys.exit(run())
