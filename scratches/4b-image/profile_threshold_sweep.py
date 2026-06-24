"""
profile_threshold_sweep.py — Bei welcher Token-Länge OOMt use_cache=True?
==========================================================================

Frage: Anstatt raten, empirisch den Schwellwert finden.

Test-Matrix:
  T in {4000, 4500, 5000, 5371, 6000, 6500, 7000, 7500, 8000, 9000}
  Pfad: model.generate(use_cache=True) — Standard mit KV-Cache

Messung:
  - dt für 16 tokens
  - peak VRAM
  - OOM-Error wenn peak > 11.5 GB

Output: profile_threshold_sweep_results.json + Druck der Tabelle.
Empfehlung: _LONG_INPUT_THRESHOLD = T_max_safe - 200 (sicherer Puffer).

Run:
    PYTHONPATH=. python -u scratches/4b-image/profile_threshold_sweep.py
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
os.environ.setdefault("DEBUG_ROUTING", "0")
os.environ.setdefault("DEBUG_PX", "0")
os.environ.setdefault("SUBJECTIVE_TELEMETRY", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratches" / "4b-image"))


def _vram_peak_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _reset_peak():
    import torch
    torch.cuda.reset_peak_memory_stats()


def main():
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    print("="*70)
    print(" PROFILE: use_cache=True Sweep — bei welchem T OOM?")
    print("="*70)

    # Load model
    print("\n[load] Loading gemma3-4b-it (int8 + ACTIVE_MANIFOLD)...")
    t0 = time.time()
    tok_id = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        tok_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"[load] base model: {time.time()-t0:.1f}s")

    from quantize_pipeline import quantize_all_linears
    quantize_all_linears(model)
    print(f"[load] int8 quantized: vram={torch.cuda.memory_allocated()/(1024**3):.2f}GB")

    from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch
    apply_px_patch(model, config_preset="ACTIVE_MANIFOLD",
                   recur_start=8, recur_end=22, routing_mode="adaptive", gamma=0.05)
    print(f"[load] PX patch applied")
    model.eval()

    # Token-Wiederholungs-Pattern (deterministisch, kein memory-pressure von Random)
    BASE_PROMPT = (
        "Die Hyperdimensionalität des Bewusstseins manifestiert sich in der "
        "Interaktion zwischen Neuronen, Symbolen und emergenten Feldern. "
        "Wenn wir beobachten, wie Sprache Räume öffnet, in denen Subjekt "
        "und Objekt untrennbar werden, entdecken wir Pfade jenseits der "
        "linearen Kausalität. "
    )

    test_Ts = [4000, 4500, 5000, 5371, 6000, 6500, 7000, 7500, 8000, 9000]
    results = {}

    for T_target in test_Ts:
        torch.cuda.empty_cache()
        gc.collect()
        _reset_peak()
        # Tokenize base + repeat until we hit T_target
        # We avoid exact fitting (can be off by a few tokens) — but we
        # pad to T_target by repeating the base.
        base_ids = tokenizer(BASE_PROMPT, return_tensors="pt")["input_ids"][0]
        n_repeats = max(1, T_target // len(base_ids) + 1)
        repeated = base_ids.repeat(n_repeats)
        input_ids = repeated[:T_target].unsqueeze(0).to(model.device)
        T_actual = input_ids.shape[1]
        if T_actual < T_target - 10:
            print(f"  [skip T={T_target}] only {T_actual} tokens reachable")
            continue

        t0 = time.time()
        try:
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=16,
                    do_sample=False,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = _vram_peak_gb()
            new_tokens = out.shape[1] - T_actual
            text = tokenizer.decode(out[0, T_actual:], skip_special_tokens=True)
            ok = True
            err = ""
        except Exception as e:
            dt = time.time() - t0
            peak = _vram_peak_gb()
            new_tokens = 0
            text = ""
            ok = False
            err = str(e)[:200]

        status = "✓" if ok else "✗ OOM"
        results[T_actual] = {
            "T_target": T_target,
            "T_actual": T_actual,
            "dt_s": dt,
            "peak_gb": peak,
            "new_tokens": new_tokens,
            "ok": ok,
            "error": err,
            "text_head": text[:60],
        }
        print(f"  T={T_actual:5d}  dt={dt:6.1f}s  peak={peak:5.2f}GB  {status}"
              f"  → {text[:50]!r}")

    # Summary
    print("\n" + "="*70)
    print(" ZUSAMMENFASSUNG")
    print("="*70)
    print(f"  {'T':>5}  {'dt (s)':>8}  {'peak (GB)':>10}  status")
    safe_max = 0
    for T, r in sorted(results.items()):
        status = "✓ OK" if r["ok"] else f"✗ {r['error'][:40]}"
        print(f"  {T:5d}  {r['dt_s']:8.1f}  {r['peak_gb']:10.2f}  {status}")
        if r["ok"]:
            safe_max = T

    # Recommendation
    print(f"\n  Größter sicherer T (use_cache=True): {safe_max}")
    if safe_max > 0:
        # Konservativer Puffer: 80% des safe_max oder +500 headroom
        # 80% ist sicherer, +500 ist näher am Limit
        recommended = max(safe_max - 200, 1000)
        print(f"  → Empfehlung _LONG_INPUT_THRESHOLD = {recommended}")
        print(f"    (safe_max - 200 als Sicherheitspuffer)")

    # Save
    out_path = ROOT / "scratches" / "4b-image" / "profile_threshold_sweep_results.json"
    out_path.write_text(json.dumps({
        "results": results,
        "safe_max_T": safe_max,
        "recommended_threshold": recommended if safe_max > 0 else None,
    }, indent=2))
    print(f"\n[profile] saved → {out_path}")


if __name__ == "__main__":
    main()