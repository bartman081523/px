"""
profile_run.py — Plan 3 Phase E: VRAM + Time Profiling für multimodal + long context
=====================================================================================

Frage: Wo geht die Zeit / das VRAM hin, wenn man gemma3-4b-it + 18-entry Session
       (3 Bilder, T~6258) generieren lässt?

Setup (1:1 wie streaming_bridge → server → generators.py):
  - gemma3-4b-it, int8 quantisiert (registry default)
  - ACTIVE_MANIFOLD preset (voll: Subjective, Routing, Relay, InfLLM/ReAttention)
  - Session: profile_target_01.json (Kopie von cross_model_holographic_01)
  - Stages exakt wie generators.py:_extract_images → processor.apply_chat_template
    → processor(text=..., images=...) → model.generate / chunked_generate

Stages die wir messen:
  A. Image-Decode (PIL base64 → 3× 896x896 RGB) — CPU only
  B. apply_chat_template — CPU only
  C. processor(text=..., images=...) — Tokenize + Vision-Encoding — CPU + GPU
  D. model.generate(use_cache=True) → prefill + decode (≤16 tok) — GPU
  E. model.generate(use_cache=False) → prefill + decode (≤16 tok) — GPU (langsam, OOM möglich)
  F. chunked_generate (text-only Fallback, simuliert) → prefill + decode — GPU

Output: profile_target_01_profile.json mit allen Timings + VRAM-Peaks.

Run:
    PYTHONPATH=. python scratches/profile_multimodal_long/profile_run.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

# Setze envs VOR torch-import (PX-Code + HF-Patches brauchen das)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:256")
os.environ.setdefault("DEBUG_ROUTING", "0")
os.environ.setdefault("DEBUG_PX", "0")
os.environ.setdefault("SUBJECTIVE_TELEMETRY", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mb(b: int) -> float:
    return b / (1024 * 1024)


def _vram_gb() -> float:
    import torch
    return torch.cuda.memory_allocated() / (1024 ** 3)


def _vram_peak_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _reset_peak():
    import torch
    torch.cuda.reset_peak_memory_stats()


def _stage(label: str):
    """Decorator: misst dt + peak VRAM für eine Stage."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            import torch
            print(f"\n=== Stage {label} ===")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            _reset_peak()
            vram_before = _vram_gb()
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
                torch.cuda.synchronize()
                dt = time.time() - t0
                peak = _vram_peak_gb()
                vram_after = _vram_gb()
                print(f"  ✓ dt={dt:.2f}s | vram before={vram_before:.3f}GB "
                      f"peak={peak:.3f}GB after={vram_after:.3f}GB")
                return {"ok": True, "dt": dt, "vram_before": vram_before,
                        "vram_peak": peak, "vram_after": vram_after,
                        "result": result}
            except Exception as e:
                torch.cuda.synchronize()
                dt = time.time() - t0
                peak = _vram_peak_gb()
                err = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"  ✗ dt={dt:.2f}s | peak={peak:.3f}GB | ERROR: {err}")
                return {"ok": False, "dt": dt, "vram_peak": peak, "error": err}
        return wrapper
    return deco


def _load_session(session_id: str) -> list:
    """Lade Session (working-dir copy profile_target_01)."""
    session_path = ROOT / "sessions" / f"{session_id}.json"
    data = json.loads(session_path.read_text())
    history = data["history"]
    print(f"[profile] session={session_id} entries={len(history)}")
    n_images = sum(
        1 for h in history
        for c in (h.get("content") if isinstance(h.get("content"), list) else [])
        if isinstance(c, dict) and c.get("type") in ("image", "image_url")
    )
    print(f"[profile] images={n_images}")
    return history


def _history_to_messages(history: list, keep_images: bool = False) -> list:
    """Streaming_bridge-Mechanik: multipart-content → string-concat (siehe bridge Z.154-156).

    Mit keep_images=True: multipart-content bleibt erhalten (wie server-direkt Anfragen).
    Mit keep_images=False (default, streaming_bridge-Mode): Bilder werden zu text-konkateniert.
    """
    api_messages = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            if not keep_images:
                text = "".join([b.get("text", "") for b in content if b.get("type") == "text"])
                content = text
        api_messages.append({"role": role, "content": content})
    return api_messages


# ── Stages ──────────────────────────────────────────────────────────────────

def stage_a_decode_images(messages: list) -> list:
    """A. PIL-Image decode (das passiert in generators._extract_images).
    CPU-only. Messe nur dt."""
    from PIL import Image
    import io, base64
    images = []
    t0 = time.time()
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") in ("image", "image_url"):
                    url = (c.get("image_url") or {}).get("url", "") if c.get("type") == "image_url" else c.get("url", "")
                    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                        b64 = url.split(";base64,", 1)[1]
                        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                        images.append(img)
    dt = time.time() - t0
    print(f"  [A] decoded {len(images)} images in {dt:.2f}s")
    return images


def stage_b_apply_chat_template(processor, messages: list) -> str:
    """B. processor.apply_chat_template. CPU-only."""
    # generators._extract_images gibt cleaned messages zurück, aber für chat_template
    # brauchen wir die 'image'-Stub-Blöcke (statt 'image_url'). Wir simulieren
    # die _extract_images-Logik hier:
    cleaned = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_items = []
            text_parts = []
            for c in content:
                if isinstance(c, dict):
                    ctype = c.get("type")
                    if ctype in ("image", "image_url"):
                        new_items.append({"type": "image"})  # stub
                    elif ctype == "text":
                        text_parts.append(c.get("text", ""))
                        new_items.append(c)
                    else:
                        new_items.append(c)
            if text_parts:
                final_content = [{"type": "text", "text": "\n".join(text_parts)}]
                for it in new_items:
                    if isinstance(it, dict) and it.get("type") == "image":
                        final_content.append(it)
            else:
                final_content = new_items if new_items else ""
            cleaned.append({"role": m.get("role", "user"), "content": final_content})
        else:
            cleaned.append(m)
    t0 = time.time()
    prompt_text = processor.apply_chat_template(
        cleaned, tokenize=False, add_generation_prompt=True
    )
    dt = time.time() - t0
    print(f"  [B] chat-template rendered in {dt:.2f}s, len={len(prompt_text)} chars")
    return prompt_text


def stage_c_processor(processor, prompt_text: str, images: list, tokenizer=None) -> dict:
    """C. Tokenize + Vision-Encoding (GPU!).

    Wichtig: wenn keine Bilder vorhanden sind, geht der generators.py-Code
    über den TOKENIZER-Pfad (Zeile 445-449), nicht über processor().
    Hier folgen wir exakt dieser Mechanik.
    """
    import torch
    t0 = time.time()
    if images:
        inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    else:
        # text-only path (genau wie generators.py:445-449)
        inputs = tokenizer(prompt_text, return_tensors="pt")
    dt_before_move = time.time() - t0
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  [C] processor/tokenizer done in {dt:.2f}s (cpu={dt_before_move:.2f}s + cuda move); "
          f"input_ids shape={inputs['input_ids'].shape}, "
          f"pixel_values: {'yes' if 'pixel_values' in inputs else 'no'}")
    return inputs


def stage_d_generate_cache_true(model, inputs: dict, max_new_tokens: int = 16):
    """D. model.generate(use_cache=True) — der NORMAL-Pfad für T<4500.
    Bei T=6258 crasht das normalerweise mit OOM, weil attention-matrix [T,T]
    in 12GB nicht passt. Aber wir versuchen's trotzdem mit Memory-Tracking."""
    import torch
    gen_kwargs = dict(inputs)
    gen_kwargs.update(dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1e-10,
        use_cache=True,
    ))
    try:
        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)
        torch.cuda.synchronize()
        return {"ok": True, "output_ids": output_ids}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"ok": False, "error": f"OOM: {str(e)[:200]}"}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def stage_e_generate_cache_false(model, inputs: dict, max_new_tokens: int = 16):
    """E. model.generate(use_cache=False) — der aktuelle Fallback für multimodal + long.
    Vermeidet KV-cache aber Prefill-Attention bleibt [T,T] → SEHR langsam + OOM möglich."""
    import torch
    gen_kwargs = dict(inputs)
    gen_kwargs.update(dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1e-10,
        use_cache=False,
    ))
    try:
        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)
        torch.cuda.synchronize()
        return {"ok": True, "output_ids": output_ids}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"ok": False, "error": f"OOM: {str(e)[:200]}"}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def stage_f_chunked_generate(model, inputs: dict, max_new_tokens: int = 16):
    """F. chunked_generate (text-only path) — Plan 3 Phase D Lösung für text-only long context.
    Bei multimodal: würde crashen wegen Vision-Token-Position. Hier simulieren wir
    den text-only Pfad als Vergleich (gleicher T aber ohne Bilder)."""
    import torch
    # chunked_prefill lebt in scratches/4b-image/ — wir haben den Pfad bereits
    # oben in sys.path eingefügt (für quantize_all_linears).
    from chunked_prefill import chunked_generate
    try:
        with torch.inference_mode():
            output_ids = chunked_generate(
                model,
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=None,
                chunk_size=512,
            )
        torch.cuda.synchronize()
        return {"ok": True, "output_ids": output_ids}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"ok": False, "error": f"OOM: {str(e)[:200]}"}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


# ── Main ────────────────────────────────────────────────────────────────────

def _run_profile_mode(mode: str, model, tokenizer, processor):
    """Run profile for one mode ('text' or 'multimodal') with separate stages."""
    print("\n" + "="*70)
    print(f" PROFILE MODE: {mode}")
    print("="*70)

    history = _load_session("profile_target_01")
    if mode == "text":
        # streaming_bridge-Mode: Bilder werden zu text-only konvertiert
        messages = _history_to_messages(history, keep_images=False)
        images = []
    else:
        # multimodal-Mode: Bilder bleiben erhalten (würde von einem
        # server-direkt-Call kommen, nicht über streaming_bridge)
        messages = _history_to_messages(history, keep_images=True)
        # Bilder aus History extrahieren (für processor)
        images = stage_a_decode_images(messages)
        # Aber: messages müssen _extract_images-konformes Format haben
        # (image-stub statt image_url). Wir bauen cleaned messages:
        from generators import _extract_images
        messages_cleaned, _ = _extract_images(messages)
        messages = messages_cleaned

    # Append user-message (Session endet auf assistant → user-Frage dranhängen)
    messages.append({
        "role": "user",
        "content": "Fasse deine bisherigen Gedanken in einem Satz zusammen.",
    })

    print(f"\n[mode={mode}] messages={len(messages)}, images={len(images)}")

    # ── Stage A: image-decode ──
    print("\n" + "="*70)
    print(" STAGE A: PIL image decode (CPU)")
    print("="*70)
    t0 = time.time()
    print(f"  [A] decoded {len(images)} images in {time.time()-t0:.2f}s")
    if images:
        print(f"  → sizes: {[img.size for img in images]}")

    # ── Stage B: apply_chat_template ──
    print("\n" + "="*70)
    print(" STAGE B: processor.apply_chat_template (CPU)")
    print("="*70)
    t0 = time.time()
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"  [B] chat-template rendered in {time.time()-t0:.2f}s, len={len(prompt_text)} chars")

    # ── Stage C: processor(text=, images=) — Tokenize + Vision-Encode ──
    print("\n" + "="*70)
    print(" STAGE C: tokenizer/processor + move to CUDA")
    print("="*70)
    import torch
    torch.cuda.empty_cache()
    _reset_peak()
    vram_before_c = _vram_gb()
    t0 = time.time()
    try:
        if images:
            inputs = processor(text=prompt_text, images=images, return_tensors="pt")
        else:
            inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids_shape = inputs["input_ids"].shape
        pixel_shape = inputs.get("pixel_values").shape if "pixel_values" in inputs else None
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        torch.cuda.synchronize()
        dt_c = time.time() - t0
        peak_c = _vram_peak_gb()
        vram_after_c = _vram_gb()
        print(f"  ✓ Stage C done in {dt_c:.2f}s")
        print(f"    input_ids shape: {input_ids_shape}")
        print(f"    pixel_values shape: {pixel_shape}")
        print(f"    vram before={vram_before_c:.3f}GB peak={peak_c:.3f}GB after={vram_after_c:.3f}GB")
        stage_c_ok = True
    except Exception as e:
        dt_c = time.time() - t0
        peak_c = _vram_peak_gb()
        print(f"  ✗ Stage C failed: {type(e).__name__}: {str(e)[:300]}")
        stage_c_ok = False
        inputs = None

    if not stage_c_ok:
        return {"mode": mode, "stage_C_failed": True}

    # Decode result for each stage (max 16 tokens)
    MAX_NEW = 16
    T = inputs["input_ids"].shape[1]

    # ── Stage D: model.generate(use_cache=True) ──
    print("\n" + "="*70)
    print(" STAGE D: model.generate(use_cache=True) — NORMAL-PFAD")
    print("="*70)
    torch.cuda.empty_cache()
    _reset_peak()
    t0 = time.time()
    res_d = stage_d_generate_cache_true(model, inputs, max_new_tokens=MAX_NEW)
    dt_d = time.time() - t0
    peak_d = _vram_peak_gb()
    if res_d["ok"]:
        out_ids = res_d["output_ids"]
        new_tokens = out_ids.shape[1] - T
        text_d = tokenizer.decode(out_ids[0, T:], skip_special_tokens=True)
        print(f"  ✓ Stage D done in {dt_d:.1f}s, peak={peak_d:.3f}GB, new_tokens={new_tokens}")
        print(f"    text: {text_d[:200]!r}")
    else:
        print(f"  ✗ Stage D FAILED in {dt_d:.1f}s, peak={peak_d:.3f}GB: {res_d['error'][:200]}")

    # ── Stage E: model.generate(use_cache=False) — Fallback ──
    print("\n" + "="*70)
    print(" STAGE E: model.generate(use_cache=False) — AKTUELLER FALLBACK")
    print("="*70)
    torch.cuda.empty_cache()
    _reset_peak()
    t0 = time.time()
    res_e = stage_e_generate_cache_false(model, inputs, max_new_tokens=MAX_NEW)
    dt_e = time.time() - t0
    peak_e = _vram_peak_gb()
    if res_e["ok"]:
        out_ids = res_e["output_ids"]
        new_tokens = out_ids.shape[1] - T
        text_e = tokenizer.decode(out_ids[0, T:], skip_special_tokens=True)
        print(f"  ✓ Stage E done in {dt_e:.1f}s, peak={peak_e:.3f}GB, new_tokens={new_tokens}")
        print(f"    text: {text_e[:200]!r}")
    else:
        print(f"  ✗ Stage E FAILED in {dt_e:.1f}s, peak={peak_e:.3f}GB: {res_e['error'][:200]}")

    # ── Stage F: chunked_generate (text-only path) ──
    print("\n" + "="*70)
    print(" STAGE F: chunked_generate (text-only path) — PLAN 3 PHASE D")
    print("="*70)
    if images:
        print("  [skip] chunked_generate kann KEINE Vision-Tokens — bei multimodal Mode übersprungen")
        dt_f, peak_f, ok_f, err_f = None, None, None, "skipped (multimodal)"
    else:
        torch.cuda.empty_cache()
        _reset_peak()
        t0 = time.time()
        res_f = stage_f_chunked_generate(model, inputs, max_new_tokens=MAX_NEW)
        dt_f = time.time() - t0
        peak_f = _vram_peak_gb()
        if res_f["ok"]:
            out_ids = res_f["output_ids"]
            new_tokens = out_ids.shape[1] - T
            text_f = tokenizer.decode(out_ids[0, T:], skip_special_tokens=True)
            print(f"  ✓ Stage F done in {dt_f:.1f}s, peak={peak_f:.3f}GB, new_tokens={new_tokens}")
            print(f"    text: {text_f[:200]!r}")
            ok_f = True
            err_f = ""
        else:
            print(f"  ✗ Stage F FAILED in {dt_f:.1f}s, peak={peak_f:.3f}GB: {res_f['error'][:200]}")
            ok_f = False
            err_f = res_f["error"]

    # ── Summary ──
    print("\n" + "="*70)
    print(f" ZUSAMMENFASSUNG MODE={mode}")
    print("="*70)
    print(f"Session: profile_target_01 (18 entries, {len(images)} images)")
    print(f"input_ids T = {T}")
    print(f"Max-new-tokens = {MAX_NEW}")
    print()
    print(f"Stage D (use_cache=True)        : dt={dt_d:7.1f}s  peak={peak_d:5.2f}GB  ok={res_d['ok']}")
    print(f"Stage E (use_cache=False)       : dt={dt_e:7.1f}s  peak={peak_e:5.2f}GB  ok={res_e['ok']}")
    if dt_f is not None:
        print(f"Stage F (chunked, text-only)    : dt={dt_f:7.1f}s  peak={peak_f:5.2f}GB  ok={ok_f}")

    return {
        "mode": mode,
        "T": T,
        "n_images": len(images),
        "max_new_tokens": MAX_NEW,
        "stage_C": {"dt": dt_c, "vram_peak_gb": peak_c, "vram_before_gb": vram_before_c,
                    "vram_after_gb": vram_after_c if 'vram_after_c' in dir() else None,
                    "input_ids_shape": list(input_ids_shape),
                    "pixel_values_shape": list(pixel_shape) if pixel_shape else None},
        "stage_D_use_cache_true": {"dt": dt_d, "vram_peak_gb": peak_d, "ok": res_d["ok"],
                                    "error": res_d.get("error", "")},
        "stage_E_use_cache_false": {"dt": dt_e, "vram_peak_gb": peak_e, "ok": res_e["ok"],
                                     "error": res_e.get("error", "")},
        "stage_F_chunked_text_only": ({"dt": dt_f, "vram_peak_gb": peak_f, "ok": ok_f,
                                       "error": err_f} if dt_f is not None else {"skipped": "multimodal"}),
    }


def main():
    print("="*70)
    print(" PROFILE: gemma3-4b-it + ACTIVE_MANIFOLD, profile_target_01 session")
    print("="*70)

    # ── Load model (1:1 wie model_manager._load_model mit ACTIVE_MANIFOLD) ──
    import torch
    from transformers import AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration

    print("\n[load] Loading gemma3-4b-it (int8 + ACTIVE_MANIFOLD patch)...")
    t0 = time.time()
    tok_id = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    processor = AutoProcessor.from_pretrained(tok_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        tok_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"[load] base model loaded in {time.time()-t0:.1f}s, vram={_vram_gb():.2f}GB")

    # Apply int8 quantization
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "scratches" / "4b-image"))
    from quantize_pipeline import quantize_all_linears
    n_replaced = quantize_all_linears(model)
    print(f"[load] int8 quantization: {n_replaced} Linears replaced in {time.time()-t0:.1f}s, "
          f"vram={_vram_gb():.2f}GB")

    # Apply PX patch (ACTIVE_MANIFOLD)
    patch_kwargs = dict(
        config_preset="ACTIVE_MANIFOLD",
        recur_start=8, recur_end=22, routing_mode="adaptive", gamma=0.05,
    )
    print(f"[load] Applying PX patch (kwargs={patch_kwargs})...")
    t0 = time.time()
    from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch
    apply_px_patch(model, **patch_kwargs)
    print(f"[load] PX patch applied in {time.time()-t0:.1f}s, vram={_vram_gb():.2f}GB")
    model.eval()

    # ── Profile Mode 1: text-only (wie streaming_bridge es heute macht) ──
    res_text = _run_profile_mode("text", model, tokenizer, processor)

    # ── Profile Mode 2: multimodal (volle Session mit Bildern) ──
    res_mm = _run_profile_mode("multimodal", model, tokenizer, processor)

    # ── Save both profiles ──
    out = {
        "session": "profile_target_01",
        "mode_text": res_text,
        "mode_multimodal": res_mm,
    }
    out_path = ROOT / "scratches" / "profile_multimodal_long" / "profile_target_01_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[profile] saved → {out_path}")


if __name__ == "__main__":
    main()
