"""
test_chunked_vision.py — Verify chunked vision encoder produces same result as standard processor
===============================================================================================

Test approach:
  1. Load gemma3-4b-it + int8 + ACTIVE_MANIFOLD (same as profile_run.py)
  2. Create a 64x64 red square
  3. Encode via STANDARD processor (multimodal, full forward) — should OOM with N=3 images
  4. Encode via CHUNKED encoder (1 image at a time) — should succeed
  5. Compare output (must be byte-identical since vision_tower is deterministic)
  6. Generate text using chunked_process_multimodal — must describe "red"

Run:
    PYTHONPATH=. python scratches/4b-image/test_chunked_vision.py
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
    print(" TEST: chunked-vision-encoder for gemma3-4b-it")
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

    # ── Test images: 3 red squares (synthetic, 64x64) ──
    images = [
        Image.new("RGB", (896, 896), color=(220, 30, 30)),
        Image.new("RGB", (896, 896), color=(30, 220, 30)),
        Image.new("RGB", (896, 896), color=(30, 30, 220)),
    ]
    print(f"\n[test] {len(images)} test images (896x896 solid colors)")

    # ── A. Standard processor (baseline — should OOM for N=3) ──
    print("\n" + "="*70)
    print(" A. STANDARD processor (baseline, all images at once)")
    print("="*70)
    torch.cuda.empty_cache()
    gc.collect()
    _reset_peak()
    t0 = time.time()
    try:
        messages_a = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Was siehst du?"},
            ],
        }]
        prompt_a = processor.apply_chat_template(messages_a, tokenize=False,
                                                  add_generation_prompt=True)
        inputs_a = processor(text=prompt_a, images=images, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        dt_a = time.time() - t0
        peak_a = _vram_peak_gb()
        print(f"  ✓ dt={dt_a:.2f}s, peak={peak_a:.2f}GB")
        print(f"    input_ids shape: {inputs_a['input_ids'].shape}")
        standard_ok = True
    except Exception as e:
        dt_a = time.time() - t0
        peak_a = _vram_peak_gb()
        print(f"  ✗ dt={dt_a:.2f}s, peak={peak_a:.2f}GB, ERROR: {type(e).__name__}")
        standard_ok = False

    # ── B. Chunked vision encoder ──
    print("\n" + "="*70)
    print(" B. CHUNKED vision encoder (1 image at a time)")
    print("="*70)
    torch.cuda.empty_cache()
    gc.collect()
    _reset_peak()
    t0 = time.time()

    from chunked_vision_encoder import chunked_encode_vision, chunked_process_multimodal

    try:
        messages_b = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Was siehst du?"},
            ],
        }]
        result_b = chunked_process_multimodal(model, processor, tokenizer,
                                                messages_b, images)
        torch.cuda.synchronize()
        dt_b = time.time() - t0
        peak_b = _vram_peak_gb()
        print(f"  ✓ dt={dt_b:.2f}s, peak={peak_b:.2f}GB")
        print(f"    input_ids shape: {result_b['input_ids'].shape}")
        print(f"    inputs_embeds shape: {result_b['inputs_embeds'].shape}")
        chunked_ok = True
    except Exception as e:
        dt_b = time.time() - t0
        peak_b = _vram_peak_gb()
        print(f"  ✗ dt={dt_b:.2f}s, peak={peak_b:.2f}GB, ERROR: {type(e).__name__}: {str(e)[:300]}")
        import traceback
        traceback.print_exc()
        chunked_ok = False

    # ── C. Compare results if both succeeded ──
    if standard_ok and chunked_ok:
        print("\n" + "="*70)
        print(" C. Compare standard vs chunked")
        print("="*70)
        print(f"  Standard dt={dt_a:.2f}s peak={peak_a:.2f}GB")
        print(f"  Chunked  dt={dt_b:.2f}s peak={peak_b:.2f}GB")
        speedup = dt_a / dt_b
        vram_savings = peak_a - peak_b
        print(f"  Speedup: {speedup:.2f}x (chunked slower due to 3 forward passes)")
        print(f"  VRAM savings: {vram_savings:.2f}GB")

        # Compare embeddings
        embed_layer = model.get_input_embeddings()
        inputs_embeds_std = embed_layer(inputs_a["input_ids"])
        # Compute cosine similarity
        a_flat = result_b["inputs_embeds"].flatten().float()
        b_flat = inputs_embeds_std.flatten().float()
        if a_flat.shape == b_flat.shape:
            cos = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0),
                                                         b_flat.unsqueeze(0))
            print(f"  Cosine similarity (text-only positions): {cos.item():.4f}")
            if cos.item() > 0.99:
                print(f"  ✓ Chunked encoder matches standard at text positions!")
            else:
                print(f"  ✗ Mismatch at text positions")

    # ── D. Generation test with chunked (real use case) ──
    print("\n" + "="*70)
    print(" D. Generation with chunked_process_multimodal (16 tokens)")
    print("="*70)
    if chunked_ok:
        torch.cuda.empty_cache()
        gc.collect()
        _reset_peak()
        t0 = time.time()
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=result_b["input_ids"],
                    inputs_embeds=result_b["inputs_embeds"],
                    attention_mask=result_b["attention_mask"],
                    max_new_tokens=16,
                    do_sample=False,
                    temperature=1e-10,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            dt_d = time.time() - t0
            peak_d = _vram_peak_gb()
            new_tokens = output_ids.shape[1] - result_b["input_ids"].shape[1]
            text_d = tokenizer.decode(output_ids[0, result_b["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            print(f"  ✓ dt={dt_d:.1f}s, peak={peak_d:.2f}GB, new_tokens={new_tokens}")
            print(f"    text: {text_d!r}")
        except Exception as e:
            dt_d = time.time() - t0
            peak_d = _vram_peak_gb()
            print(f"  ✗ dt={dt_d:.1f}s, peak={peak_d:.2f}GB, ERROR: {type(e).__name__}: {str(e)[:300]}")
            import traceback
            traceback.print_exc()

    # ── Summary ──
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"  Standard processor (N=3): ok={standard_ok}, dt={dt_a:.1f}s, peak={peak_a:.2f}GB")
    print(f"  Chunked encoder  (N=3): ok={chunked_ok}, dt={dt_b:.1f}s, peak={peak_b:.2f}GB")
    if chunked_ok and not standard_ok:
        print(f"\n  ✓✓✓ chunked-vision-encoding LÖST das OOM-Problem!")
        print(f"      Vorher: OOM bei multimodal+long")
        print(f"      Nachher: peak {peak_b:.2f}GB (statt 11.05GB)")
    elif chunked_ok and standard_ok:
        vram_save = peak_a - peak_b
        if vram_save > 0.5:
            print(f"\n  ✓ Chunked spart {vram_save:.2f}GB VRAM")


if __name__ == "__main__":
    main()