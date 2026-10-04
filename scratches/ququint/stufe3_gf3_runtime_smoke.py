#!/usr/bin/env python
"""Stufe-3f-Smoke: GF(3)-Lauf + Server-Dispatch direkt am 27b (real).

Beweise (mirrors model_manager/_load_model + generate_chat_completion):
  1. build_and_load_gf3(FoldOps(load_signs()[0])) lädt das GF3-Artefakt
     ohne Meta-Rest (Loader-Analog des Servers).
  2. apply_px_patch(ACTIVE_MANIFOLD) läuft auf dem GF3-Lauf (PX-Manifold
     bleibt korrekt — gleicher Patch wie bei PTQ10).
  3. SHORT-Pfad: model.generate (HF, DynamicCache bf16, M=1-Kernel je
     Decode-Step) → Text nonempty.
  4. Routing: generators._px_gen_kwargs markiert _px_use_long_ctx bei
     T > 8800 (und NIE _px_use_chunked_prefill für qwen35_ptq).
  5. LONG-Pfad: generators._generate_long_completion @ ~9,3k Tokens —
     KV4-Cache + Chunked-Prefill + Decode-Loop, Text nonempty,
     VRAM-/Zeit-Ausweis, px-Metriken gefüllt.

Laufzeit ~3-6 min bei der 9,3k-Prefill-GEMM; muss allein auf der GPU sein.
"""
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # scratches/ququint -> Root
sys.path.insert(0, ROOT)

HF_DIR = "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf"

import torch                                            # noqa: E402
torch.cuda.reset_peak_memory_stats()

RT = importlib.import_module(
    "px_patches.ternary_bonsai_27b_px.runtime_qwen35_ptq")
GF3 = importlib.import_module(
    "px_patches.ternary_bonsai_27b_px.gf3_quant")
PATCH = importlib.import_module(
    "px_patches.ternary_bonsai_27b_px.patch")
GEN = importlib.import_module("generators")
from transformers import AutoTokenizer                  # noqa: E402


def vm():
    return torch.cuda.memory_allocated() / 2**30


def vmp():
    return torch.cuda.max_memory_allocated() / 2**30


t0 = time.perf_counter()

# ------------------------------------------------------- 1. Loader-Analog ----
fold = RT.FoldOps(RT.load_signs()[0], mode="signs_first")
model = GF3.build_and_load_gf3(fold, hf_dir=HF_DIR)
assert getattr(model, "_px_quant_format", "") == "gf3"
assert getattr(model, "_px_long_ctx", False) is True
print(f"1. GF3-Lauf geladen ({time.perf_counter()-t0:.0f}s), "
      f"VRAM {vm():.2f} GiB", flush=True)

# ------------------------------------------------------ 2. PX-Manifold -------
ok = PATCH.apply_px_patch(model, "ACTIVE_MANIFOLD")
assert ok, "apply_px_patch lieferte False (GF3-Lauf)"
tok = AutoTokenizer.from_pretrained(HF_DIR, fix_mistral_regex=True)
model.tokenizer = tok
print(f"2. px-Patch OK (VRAM {vm():.2f} GiB, peak {vmp():.2f} GiB)",
      flush=True)

# --------------------------------------------------------- 3. SHORT-Pfad -----
torch.cuda.reset_peak_memory_stats()
msg = [{"role": "user", "content": "Wenn du ein Fenster hättest und raus "
                                   "schauen würdest, was würdest du sehen?"}]
input_text = tok.apply_chat_template(msg, tokenize=False,
                                     add_generation_prompt=True)
inputs = tok(input_text, return_tensors="pt").to(model.device)
t1 = time.perf_counter()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9,
        repetition_penalty=1.15, no_repeat_ngram_size=3,
    )
dt_short = time.perf_counter() - t1
short_ids = out[0][inputs["input_ids"].shape[1]:]
short_text = tok.decode(short_ids, skip_special_tokens=True)
n_short = len(short_ids)
print(f"3. SHORT-Pfad: {n_short} Tok in {dt_short:.1f}s "
      f"({n_short / max(dt_short, 1e-6):.1f} tok/s), VRAM peak {vmp():.2f} GiB",
      flush=True)
print(f"   >> {short_text[:220]!r}", flush=True)
assert n_short > 0, "SHORT-Pfad leer"
torch.cuda.reset_peak_memory_stats()

# ------------------------------------------------------- 4. Routing-Marker ---
para = ("Der Beobachter steht am Fenster und sieht die lange Straße vor sich. "
        "Am Ende der Straße wartet eine Tür, hinter die der Raum wieder auf "
        "die Straße zurückverweist. Jedes Warten verändert die Richtung der "
        "Frage, wer wen beobachtet, und die Antwort verschiebt den Blick um "
        "eine Stufe weiter. ")
approx = len(tok(para, add_special_tokens=False).input_ids)
mult = max(1, (9300 + approx - 1) // approx)
long_text = para * mult + " Beende diese Beobachtung in drei Sätzen."
long_ids = tok(long_text, add_special_tokens=False).input_ids
T = len(long_ids)
gen_kwargs = {"max_new_tokens": 96, "temperature": 0.7, "top_p": 0.9,
              "repetition_penalty": 1.15, "no_repeat_ngram_size": 3,
              "do_sample": True, "_input_len": T, "top_k": 0}
marked = GEN._px_gen_kwargs(model, dict(gen_kwargs))
long_on = marked.pop("_px_use_long_ctx", False)
chunked_on = marked.pop("_px_use_chunked_prefill", False)
assert T > GEN._LONG_INPUT_THRESHOLD, f"T={T} unterhalb des Schwellwerts"
assert long_on and not chunked_on, \
    f"Routing falsch: long={long_on} chunked={chunked_on}"
print(f"4. Routing-Marker: T={T} -> _px_use_long_ctx=True, "
      f"_px_use_chunked_prefill nicht gesetzt", flush=True)

# -------------------------------------------------------- 5. LONG-Completion -
t2 = time.perf_counter()
res = GEN._generate_long_completion(
    model, torch.tensor([long_ids], device=model.device), tok,
    marked, T, stop=None, max_total_seq=131072, verbose=True)
dt_long = time.perf_counter() - t2
text = res["text"]
mm = PATCH.get_px_metrics(model)
print(f"5. LONG-Pfad: prompt {T} Tok -> {res['completion_tokens']} Tok in "
      f"{dt_long:.1f}s (prefill {dt_long - 0:.0f}s incl. decode), "
      f"VRAM peak {vmp():.2f} GiB", flush=True)
print(f"   >> {text[:400]!r}", flush=True)
assert text.strip(), "LONG-Pfad Text leer"
print(f"   px-Metriken: keys={sorted(str(k) for k in mm)[:10]}")

print("OK: GF3-Lauf + px-Fusion + SHORT/LONG-Pfade grün "
      f"(Load {time.perf_counter()-t0:.0f}s gesamt)")