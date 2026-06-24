"""
profile_vision_encoding.py — Wie viel VRAM allokiert der Vision-Tower für 1/2/3 Bilder?
=========================================================================================

Frage: Ist chunked-vision-encoding (Bilder einzeln) die Lösung für Multimodal+OOM?

Hypothese:
  Aktuell: processor(text=..., images=[img1, img2, img3]) encodet alle 3 auf einmal
           → vision_tower forward mit pixel_values.shape=[3, 3, 896, 896]
           → SigLIP activations: 3 × [B, 4096, 1152] × 27 Layers × 4 bytes = ~5 GB
           → plus multi_modal_projector: weitere Allocations
           → DAS ist die 6.4GB die fehlt.

  Wenn chunked: vision_tower für 1 Bild, dann löschen + del tensors, dann nächstes
           → peak SigLIP activations: 1 × [B, 4096, 1152] × 27 × 4 = ~1.7 GB
           → DAS passt locker in headroom.

Wir messen für 1, 2, 3 Bilder:
  - VRAM-Peak bei vision_tower forward
  - VRAM-Peak bei multi_modal_projector forward
  - Output-Shape und dtype

Run:
    PYTHONPATH=. python scratches/profile_multimodal_long/profile_vision_encoding.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:256")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _vram_gb() -> float:
    import torch
    return torch.cuda.memory_allocated() / (1024 ** 3)


def _vram_peak_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _reset_peak():
    import torch
    torch.cuda.reset_peak_memory_stats()


def main():
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    from PIL import Image

    print("="*70)
    print(" PROFILE: vision_tower forward für N Bilder")
    print("="*70)

    # Load model (1:1 wie andere profiles)
    print("\n[load] Loading gemma3-4b-it (int8)...")
    t0 = time.time()
    tok_id = "google/gemma-3-4b-it"
    processor = AutoProcessor.from_pretrained(tok_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        tok_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"[load] base model loaded in {time.time()-t0:.1f}s, vram={_vram_gb():.2f}GB")

    # int8 quantize
    sys.path.insert(0, str(ROOT / "scratches" / "4b-image"))
    from quantize_pipeline import quantize_all_linears
    quantize_all_linears(model)
    print(f"[load] int8 quantized, vram={_vram_gb():.2f}GB")

    # Vision-Tower ist in model.model.vision_tower
    vision_tower = model.model.vision_tower
    multi_modal_projector = model.model.multi_modal_projector
    print(f"[load] vision_tower: {type(vision_tower).__name__}")
    print(f"[load] multi_modal_projector: {type(multi_modal_projector).__name__}")

    # Test mit 1, 2, 3 Bildern
    img = Image.new("RGB", (896, 896), color=(220, 30, 30))

    results = {}
    for n_images in [1, 2, 3]:
        print(f"\n{'='*70}")
        print(f" N = {n_images} Bilder")
        print("="*70)
        torch.cuda.empty_cache()
        gc.collect()
        _reset_peak()
        vram_before = _vram_gb()

        # PIL → pixel_values via image_processor
        images = [img] * n_images
        t0 = time.time()
        pixel_values = processor.image_processor(images, return_tensors="pt")["pixel_values"]
        # pixel_values: [N, 3, 896, 896], float32 — das ist klein (~28 MB für 3)
        pixel_values = pixel_values.to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        dt_prep = time.time() - t0
        print(f"  pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
        print(f"  after move: dt={dt_prep:.2f}s, vram={_vram_gb():.3f}GB")

        # Vision-Tower forward
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            vision_out = vision_tower(pixel_values=pixel_values, return_dict=True)
        torch.cuda.synchronize()
        dt_vision = time.time() - t0
        peak_vision = _vram_peak_gb()
        print(f"  vision_tower forward: dt={dt_vision:.2f}s, peak={peak_vision:.3f}GB")
        print(f"    output shape: {vision_out.last_hidden_state.shape}, dtype: {vision_out.last_hidden_state.dtype}")

        # Multi-Modal-Projector
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            projected = multi_modal_projector(vision_out.last_hidden_state)
        torch.cuda.synchronize()
        dt_proj = time.time() - t0
        peak_proj = _vram_peak_gb()
        print(f"  multi_modal_projector forward: dt={dt_proj:.2f}s, peak={peak_proj:.3f}GB")
        print(f"    output shape: {projected.shape}, dtype: {projected.dtype}")

        # Cleanup
        last_hidden_shape = list(vision_out.last_hidden_state.shape)
        projected_shape = list(projected.shape)
        del pixel_values, vision_out, projected
        torch.cuda.empty_cache()
        gc.collect()

        results[n_images] = {
            "dt_prep": dt_prep,
            "dt_vision": dt_vision,
            "peak_vision": peak_vision,
            "dt_proj": dt_proj,
            "peak_proj": peak_proj,
            "last_hidden_shape": last_hidden_shape,
            "projected_shape": projected_shape,
        }

    # Summary
    print("\n" + "="*70)
    print(" ZUSAMMENFASSUNG")
    print("="*70)
    print(f"  N    vision_tower    mm_projector    peak")
    print(f"       dt      peak    dt      peak")
    for n in [1, 2, 3]:
        r = results[n]
        print(f"  {n}    {r['dt_vision']:5.2f}s  {r['peak_vision']:5.2f}GB  "
              f"{r['dt_proj']:5.2f}s  {r['peak_proj']:5.2f}GB")

    print()
    if results[1]["peak_vision"] < 9.0 and results[3]["peak_vision"] > 10.0:
        print("✓ Chunked-vision-encoding LÖST das OOM-Problem!")
        print(f"  1 Bild peak: {results[1]['peak_vision']:.2f}GB → sicher in headroom")
        print(f"  3 Bilder peak: {results[3]['peak_vision']:.2f}GB → OOM (multimodal-mode)")
    else:
        print("✗ Chunked-vision-encoding bringt NICHT genug — Problem liegt woanders")

    out_path = ROOT / "scratches" / "profile_multimodal_long" / "vision_encoding_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[profile] saved → {out_path}")


if __name__ == "__main__":
    main()
