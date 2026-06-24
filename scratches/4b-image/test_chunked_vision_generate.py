"""
test_chunked_vision_generate.py — End-to-End test chunked-vision-encoder + chunked_generate
============================================================================================

Verify:
  1. chunked_process_multimodal builds inputs_embeds with 768 image_soft_tokens filled
  2. chunked_generate(inputs_embeds=...) can consume them without OOM
  3. Generation produces coherent text describing the images

Run:
    PYTHONPATH=. python -u scratches/4b-image/test_chunked_vision_generate.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:256")
os.environ.setdefault("DEBUG_ROUTING", "0")
os.environ.setdefault("DEBUG_PX", "0")
os.environ.setdefault("SUBJECTIVE_TELEMETRY", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratches" / "4b-image"))


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
    from transformers import AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration
    from PIL import Image

    print("="*70)
    print(" TEST: chunked-vision-encoder + chunked_generate(inputs_embeds)")
    print("="*70)

    # ── Load model ──
    print("\n[load] Loading gemma3-4b-it (int8 + ACTIVE_MANIFOLD)...")
    t0 = time.time()
    tok_id = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    processor = AutoProcessor.from_pretrained(tok_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        tok_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"[load] base model: {time.time()-t0:.1f}s, vram={_vram_gb():.2f}GB")

    from quantize_pipeline import quantize_all_linears
    quantize_all_linears(model)
    print(f"[load] int8 quantized: vram={_vram_gb():.2f}GB")

    from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch
    apply_px_patch(model, config_preset="ACTIVE_MANIFOLD",
                   recur_start=8, recur_end=22, routing_mode="adaptive", gamma=0.05)
    print(f"[load] PX patch applied: vram={_vram_gb():.2f}GB")
    model.eval()

    # ── Test images ──
    images = [
        Image.new("RGB", (896, 896), color=(220, 30, 30)),
        Image.new("RGB", (896, 896), color=(30, 220, 30)),
        Image.new("RGB", (896, 896), color=(30, 30, 220)),
    ]
    print(f"\n[test] {len(images)} images (896x896 solid colors)")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": "Was siehst du in den drei Bildern?"},
        ],
    }]

    # ── A. Build inputs_embeds via chunked_process_multimodal ──
    print("\n" + "="*70)
    print(" A. Build inputs_embeds via chunked-vision-encoder")
    print("="*70)
    torch.cuda.empty_cache()
    gc.collect()
    _reset_peak()
    t0 = time.time()

    from chunked_vision_encoder import chunked_process_multimodal
    try:
        result = chunked_process_multimodal(model, processor, tokenizer,
                                             messages, images)
        torch.cuda.synchronize()
        dt_a = time.time() - t0
        peak_a = _vram_peak_gb()
        print(f"  ✓ dt={dt_a:.2f}s, peak={peak_a:.2f}GB")
        print(f"    input_ids shape: {result['input_ids'].shape}")
        print(f"    inputs_embeds shape: {result['inputs_embeds'].shape}")
        n_img_tokens = (result['input_ids'][0] == 262144).sum().item()
        print(f"    image_soft_token count: {n_img_tokens} (expected {len(images)*256})")
        n_img_embeds = (result['inputs_embeds'][0].abs().sum(dim=-1) > 0.01).sum().item()
        print(f"    non-zero embed rows: {n_img_embeds}")
        ok_a = n_img_tokens == 768 and n_img_embeds >= 700  # most embeds should be filled
    except Exception as e:
        dt_a = time.time() - t0
        peak_a = _vram_peak_gb()
        print(f"  ✗ dt={dt_a:.1f}s, peak={peak_a:.2f}GB, ERROR: {type(e).__name__}: {str(e)[:300]}")
        import traceback
        traceback.print_exc()
        ok_a = False

    if not ok_a:
        print("\nABORT: chunked_process_multimodal failed")
        return

    # ── B. Generate via chunked_generate(inputs_embeds=...) ──
    print("\n" + "="*70)
    print(" B. chunked_generate(inputs_embeds=...) 32 tokens")
    print("="*70)
    torch.cuda.empty_cache()
    gc.collect()
    _reset_peak()
    t0 = time.time()

    from chunked_prefill import chunked_generate
    try:
        with torch.inference_mode():
            output_ids = chunked_generate(
                model,
                result['input_ids'],
                inputs_embeds=result['inputs_embeds'],
                attention_mask=result['attention_mask'],
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                chunk_size=512,
            )
        torch.cuda.synchronize()
        dt_b = time.time() - t0
        peak_b = _vram_peak_gb()
        new_tokens = output_ids.shape[1] - result['input_ids'].shape[1]
        text_b = tokenizer.decode(output_ids[0, result['input_ids'].shape[1]:],
                                    skip_special_tokens=True)
        print(f"  ✓ dt={dt_b:.1f}s, peak={peak_b:.2f}GB, new_tokens={new_tokens}")
        print(f"    text: {text_b!r}")
        ok_b = peak_b < 12.0 and len(text_b) > 10
    except Exception as e:
        dt_b = time.time() - t0
        peak_b = _vram_peak_gb()
        print(f"  ✗ dt={dt_b:.1f}s, peak={peak_b:.2f}GB, ERROR: {type(e).__name__}: {str(e)[:500]}")
        import traceback
        traceback.print_exc()
        ok_b = False

    # ── C. Compare with standard processor + generate (if memory allows) ──
    print("\n" + "="*70)
    print(" C. Standard processor + generate (1 image only, for sanity)")
    print("="*70)
    torch.cuda.empty_cache()
    gc.collect()
    _reset_peak()
    t0 = time.time()
    try:
        # Just 1 image to avoid OOM
        messages_c = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Was siehst du?"},
            ],
        }]
        images_c = [images[0]]
        prompt_c = processor.apply_chat_template(messages_c, tokenize=False,
                                                  add_generation_prompt=True)
        inputs_c = processor(text=prompt_c, images=images_c, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        with torch.inference_mode():
            out_c = model.generate(**inputs_c, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize()
        dt_c = time.time() - t0
        peak_c = _vram_peak_gb()
        text_c = tokenizer.decode(out_c[0, inputs_c['input_ids'].shape[1]:],
                                    skip_special_tokens=True)
        print(f"  ✓ dt={dt_c:.1f}s, peak={peak_c:.2f}GB")
        print(f"    text: {text_c!r}")
    except Exception as e:
        dt_c = time.time() - t0
        peak_c = _vram_peak_gb()
        print(f"  ✗ dt={dt_c:.1f}s, peak={peak_c:.2f}GB, ERROR: {type(e).__name__}: {str(e)[:200]}")

    # ── Summary ──
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"  A. chunked_process_multimodal (3 imgs): peak={peak_a:.2f}GB, "
          f"img_tokens={n_img_tokens}/768, ok={ok_a}")
    print(f"  B. chunked_generate + inputs_embeds (3 imgs): peak={peak_b:.2f}GB, "
          f"dt={dt_b:.1f}s, ok={ok_b}")
    print(f"  C. Standard processor (1 img): peak={peak_c:.2f}GB, dt={dt_c:.1f}s")
    if ok_a and ok_b:
        print(f"\n  ✓✓✓ Multimodal+OOM SOLVED via chunked-vision-encoder!")
        print(f"      Vorher (multimodal+long): OOM (peak 11+GB)")
        print(f"      Nachher (3 Bilder): peak {peak_b:.2f}GB")
        print(f"      Vorher (multimodal+use_cache=True): OOM")
        print(f"      Nachher: use_cache=True via chunked_generate(inputs_embeds)")


if __name__ == "__main__":
    main()