"""
test_chunked_prefill_t8000.py — Plan 3 Phase D Step 4: E2E T=8000
==================================================================

T=8000 mit 4b + int8 + PX + chunked_generate (chunk=512, full-only cache).

Akzeptanz:
  - new_tokens >= 5 (nicht leer)
  - peak < 12 GB
  - dt < 120s für 16 tokens

Run:
    python test_chunked_prefill_t8000.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:256")
import sys
import time
import torch

_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_SCRATCHES = os.path.join(_REPO, "scratches", "4b-image")
if _SCRATCHES not in sys.path:
    sys.path.insert(0, _SCRATCHES)


def main():
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from quantize_pipeline import quantize_all_linears
    from chunked_prefill import chunked_generate
    sys.path.insert(0, os.path.join(_REPO, "px_patches"))
    from gemma3_270m_px_baseline.patch import apply_px_patch

    hf_id = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    quantize_all_linears(model)
    apply_px_patch(model, config_preset="ACTIVE_MANIFOLD",
                   subjective_enabled=False, gamma=0.10)
    print("PX patch applied", flush=True)

    base = "Die subjektive Erfahrung des Bewusstseins. "
    prompt = base * 50
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    while inputs["input_ids"].shape[1] < 8000:
        prompt = prompt + base * 50
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    T = inputs["input_ids"].shape[1]
    print(f"Prompt T={T}", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pre = torch.cuda.memory_allocated() / 1e9

    t0 = time.time()
    try:
        out = chunked_generate(
            model, inputs["input_ids"], chunk_size=512,
            max_new_tokens=16, do_sample=False,
        )
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        new = out.shape[1] - T
        txt = tokenizer.decode(out[0, T:], skip_special_tokens=True)
        print(f"CHUNKED T={T}: pre={pre:.3f}GB peak={peak:.3f}GB dt={dt:.1f}s new={new}", flush=True)
        print(f"  text[:150]: {txt[:150]!r}", flush=True)
        ok = new >= 5 and peak < 12.0
        if not ok:
            print(f"[FAIL] expected new_tokens>=5, peak<12GB", flush=True)
        else:
            print(f"[OK] chunked T={T} works (peak={peak:.2f}GB dt={dt:.1f}s)", flush=True)
        return ok
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM: {str(e)[:200]}", flush=True)
        return False


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
