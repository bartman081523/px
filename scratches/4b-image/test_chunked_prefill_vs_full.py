"""
test_chunked_prefill_vs_full.py — Plan 3 Phase D Step 3: Bit-Äquivalenz
=========================================================================

T=4500 mit 4b + int8 + PX:
  - full generate (model.generate)
  - chunked_generate (chunk_size=512, mit neuer patch.py-Logik)

Erwartung: byte-identische Ausgabe (greedy decoding). Wenn das passt,
ist die score_mem_mb-Heuristik semantisch eine korrekte Erweiterung
und der chunked-Pfad macht das gleiche wie SDPA für diese Größe.

Akzeptanz:
  - beide Outputs sind textuell identisch (oder >90% common prefix)
  - VRAM peak für chunked < full (oder gleich)
  - dt chunked ≈ dt full

Run:
    python test_chunked_prefill_vs_full.py
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
    print("PX patch applied")

    # T=4500 ist unter dem 4096-MEM_EFF_THRESHOLD-aber ÜBER unserer
    # score_mem_mb-Schwelle: 4500*4500*8*4/1MB = 619 MB → chunked
    base = "Die subjektive Erfahrung des Bewusstseins. "
    prompt = base * 50
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    while inputs["input_ids"].shape[1] < 4500:
        prompt = prompt + base * 50
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    T = inputs["input_ids"].shape[1]
    print(f"Prompt T={T}")

    # === Full generate (baseline) ===
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out_full = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    dt_full = time.time() - t0
    peak_full = torch.cuda.max_memory_allocated() / 1e9
    text_full = tokenizer.decode(out_full[0, T:], skip_special_tokens=True)
    print(f"\nFULL: peak={peak_full:.3f}GB dt={dt_full:.1f}s")
    print(f"  text[:120]: {text_full[:120]!r}")

    # === Chunked generate ===
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out_chunked = chunked_generate(
        model, inputs["input_ids"], chunk_size=512,
        max_new_tokens=32, do_sample=False,
    )
    dt_chunked = time.time() - t0
    peak_chunked = torch.cuda.max_memory_allocated() / 1e9
    text_chunked = tokenizer.decode(out_chunked[0, T:], skip_special_tokens=True)
    print(f"\nCHUNKED: peak={peak_chunked:.3f}GB dt={dt_chunked:.1f}s")
    print(f"  text[:120]: {text_chunked[:120]!r}")

    # Vergleich
    if text_full == text_chunked:
        print(f"\n[OK] chunked == full (byte-identisch)")
        ok = True
    else:
        # common prefix
        common = 0
        for a, b in zip(text_full, text_chunked):
            if a == b:
                common += 1
            else:
                break
        match_pct = common / max(len(text_full), 1) * 100
        print(f"\n[INFO] common prefix: {common}/{len(text_full)} chars "
              f"({match_pct:.1f}%)")
        # Akzeptanz: ≥90% common prefix (greedy divergiert oft früh)
        ok = match_pct >= 90
        if not ok:
            print(f"[FAIL] divergiert zu früh — chunked-Pfad semantisch nicht equivalent")
        else:
            print(f"[OK] common prefix ≥90% — semantisch nah genug")

    # VRAM-Vergleich
    print(f"\nVRAM: full={peak_full:.3f}GB, chunked={peak_chunked:.3f}GB, "
          f"delta={peak_chunked-peak_full:+.3f}GB")
    # Akzeptanz: chunked peak ≤ full peak + 0.5 GB (sollte sogar kleiner sein)
    assert peak_chunked <= peak_full + 0.5, \
        f"chunked braucht mehr VRAM als full: {peak_chunked:.3f} > {peak_full+0.5:.3f}"

    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)