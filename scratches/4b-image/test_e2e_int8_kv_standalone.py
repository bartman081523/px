"""
test_e2e_int8_kv_standalone.py — Standalone E2E mit echtem 4b + int8 + int8 KV
=================================================================================

Lädt gemma3-4b-it direkt, installiert Plan 1 (int8 weights) + Plan 3 Phase A
(int8 KV), misst T=4800 Prefill VRAM. Wenn < 10GB → grün; sonst → eskalieren.

Kein Server, kein streaming_bridge, kein PX-Patch. Reiner Load + Generate.

Run:
    PY=/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python
    $PY test_e2e_int8_kv_standalone.py
"""
import sys
import os
import time
import gc

import torch

_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_SCRATCHES = os.path.join(_REPO, "scratches", "4b-image")
if _SCRATCHES not in sys.path:
    sys.path.insert(0, _SCRATCHES)

# RTX 2060 Settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"


def gpu_mem_gb():
    return torch.cuda.memory_allocated() / 1e9


def gpu_peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    if not torch.cuda.is_available():
        print("[FATAL] no CUDA")
        sys.exit(2)

    print(f"[E2E] Loading gemma3-4b-it with int8 weights + int8 KV...")

    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from quantize_pipeline import quantize_all_linears
    from int8_kv_cache import install_int8_kv_hooks, is_int8_kv_installed

    hf_id = "google/gemma-3-4b-it"
    tok_id = hf_id

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"[E2E] model loaded in {time.time()-t0:.1f}s, VRAM={gpu_mem_gb():.3f}GB")

    # Plan 1: int8 Weights
    t0 = time.time()
    n_replaced = quantize_all_linears(model)
    print(f"[E2E] int8 weights: {n_replaced} Linears replaced in {time.time()-t0:.1f}s, "
          f"VRAM={gpu_mem_gb():.3f}GB")

    # Plan 3 Phase A: int8 KV hook
    torch.cuda.reset_peak_memory_stats()
    n_kv_patched = install_int8_kv_hooks(model)
    print(f"[E2E] int8 KV hook installed on {n_kv_patched} attention layers, "
          f"installed={is_int8_kv_installed(model)}")

    # Get text-model (multimodal wrapper)
    if hasattr(model, "language_model"):
        text_model = model.language_model
    elif hasattr(model, "model"):
        text_model = model.model
    else:
        text_model = model

    # === T=4800 Prefill Test ===
    print(f"\n[E2E] === T=4800 Prefill Test ===")
    base_text = (
        "Die subjektive Erfahrung des Bewusstseins, die kognitive "
        "Verarbeitung von sensorischen Inputs, die rekursive "
        "Selbst-Referenz in der Verarbeitung von Kontext, und die "
        "Frage nach der Natur der algorithmischen Subjektivität. "
    )
    # Tokenizer macht oft ~3-4 tokens pro Wort, weniger als 4 chars/token.
    # Wir iterativ verlängern bis T >= 4500.
    prompt = base_text * 50  # start
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    while inputs["input_ids"].shape[1] < 4500:
        prompt = prompt + base_text * 50
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    T_actual = inputs["input_ids"].shape[1]
    print(f"[E2E] tokenized prompt: T={T_actual} tokens")
    assert T_actual >= 4500, f"expected T>=4500, got {T_actual}"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pre_gen_vram = gpu_mem_gb()

    t0 = time.time()
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=64,
                do_sample=False, temperature=1.0,
                use_cache=True,
            )
        gen_dt = time.time() - t0
    except torch.cuda.OutOfMemoryError as e:
        print(f"[FAIL] OOM during generate at T={T_actual}: {e}")
        return False
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[FAIL] OOM (RuntimeError) at T={T_actual}: {e}")
            return False
        raise

    peak_vram = gpu_peak_gb()
    new_tokens = out.shape[1] - T_actual
    gen_text = tokenizer.decode(out[0, T_actual:], skip_special_tokens=True)
    print(f"\n[E2E] T={T_actual} generate OK in {gen_dt:.1f}s")
    print(f"  pre_gen VRAM:  {pre_gen_vram:.3f}GB")
    print(f"  peak VRAM:     {peak_vram:.3f}GB")
    print(f"  new_tokens:    {new_tokens}")
    print(f"  response[:200]: {gen_text[:200]!r}")

    if new_tokens < 5:
        print(f"[FAIL] too few generated tokens: {new_tokens}")
        return False

    if peak_vram > 11.0:
        print(f"[FAIL] peak VRAM {peak_vram:.3f}GB > 11.0GB (12GB card, 1GB reserve)")
        return False

    print(f"\n[E2E] PASS — T=4800 funktioniert mit int8 + int8 KV (peak {peak_vram:.2f}GB)")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)