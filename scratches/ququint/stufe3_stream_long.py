#!/usr/bin/env python
"""Stufe-3g-Streaming-Langpfad: >8800-Tok-Nachricht mit stream=True durch den
Bridge-Kanal (https, httpx verify=False). Beweist generate_long(streamer):
kv4 + Chunked-Prefill im Server-Streaming.

Misst TTFB (Chunk 1 nach Prefill ~200s), Gesamtzeit, Delta-Anzahl und Text.
"""
import json
import time

import httpx
from transformers import AutoTokenizer

URL = "https://localhost:7860/v1/chat/completions"
HF_DIR = "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf"
MODEL = "ternary-bonsai-27b"

tok = AutoTokenizer.from_pretrained(HF_DIR)
para = ("Der Beobachter steht am Fenster und sieht die lange Strasse vor sich. "
        "Am Ende der Strasse wartet eine Tuer, hinter die der Raum wieder auf "
        "die Strasse zurueckverweist. Jedes Warten veraendert die Richtung der "
        "Frage, wer wen beobachtet, und die Antwort verschiebt den Blick um "
        "eine Stufe weiter. ")
approx = len(tok(para, add_special_tokens=False).input_ids)
mult = max(1, (9300 + approx - 1) // approx)
long_text = para * mult + " Beende diese Beobachtung in drei Saetzen."
T = len(tok(long_text, add_special_tokens=False).input_ids)
print(f"[client] Prompt-Token: {T} (> 8800)", flush=True)

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": long_text}],
    "max_tokens": 48,
    "px_config_preset": "ACTIVE_MANIFOLD",
    "px_thinking": False,
    "stream": True,
}

text = ""
n_deltas = 0
t0 = time.perf_counter()
with httpx.stream("POST", URL, json=payload, verify=False, timeout=None) as r:
    assert r.status_code == 200, r.status_code
    for line in r.iter_lines():
        if not line.startswith("data: "):
            continue
        s = line[6:]
        if s == "[DONE]":
            break
        d = json.loads(s)
        if "error" in d:
            print(f"[client] SERVER-ERROR: {d['error']}", flush=True)
            raise SystemExit(1)
        delta = d["choices"][0].get("delta", {})
        part = delta.get("content")
        if part:
            if n_deltas == 0:
                print(f"[client] TTFB: {time.perf_counter()-t0:.1f}s "
                      f"(Prefill beendet)", flush=True)
            n_deltas += 1
            text += part
dt = time.perf_counter() - t0
print(f"[client] Streaming-Ende: {n_deltas} Deltas in {dt:.1f}s gesamt",
      flush=True)
print(f"[client] >> {text[:300]!r}", flush=True)
assert text.strip(), "Streaming-Langpfad: kein Text"
assert n_deltas > 10, f"nur {n_deltas} Deltas — kein echtes Streaming"
print("OK: Streaming-Langpfad gruen (SSE nach kv4-Chunked-Prefill)", flush=True)