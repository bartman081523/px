#!/usr/bin/env python
"""Stufe-3f-Langpfad am Server: >8800-Token-Prompt -> _generate_long_completion
(kv4-Cache + Chunked-Prefill 2048 + px-Write-Guard). Vor dem Guard crashed
dieser Pfad im 2. Chunk (sdpa 6144-vs-4096); der Standalone-Smoke war gruen —
hier der Beweis durch den echten Server-Request.

Prompt = Beobachter-Paragraph x mult (identisch zum Standalone-Runtime-Smoke),
Antwort: 64 Tok bei ACTIVE_MANIFOLD, px_thinking=False.
"""
import json
import subprocess
import time
import urllib.request

from transformers import AutoTokenizer

URL = "http://localhost:7860"
MODEL = "ternary-bonsai-27b"
HF_DIR = "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf"

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
print(f"[client] Prompt-Token (chattemplate-frei gemessen): {T}", flush=True)
assert T > 8800, f"T={T} unterhalb Long-Threshold"

body = {
    "model": MODEL,
    "messages": [{"role": "user", "content": long_text}],
    "max_tokens": 64,
    "px_config_preset": "ACTIVE_MANIFOLD",
    "px_thinking": False,
}


def _vram():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20)
    return int(out.stdout.strip().splitlines()[0])


v0 = _vram()
t0 = time.perf_counter()
req = urllib.request.Request(
    URL + "/v1/chat/completions", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=900) as r:
    res = json.loads(r.read())
dt = time.perf_counter() - t0
peak = _vram()
txt = res["choices"][0]["message"]["content"]
usage = res.get("usage", {})
print(f"[client] Server-Antwort nach {dt:.1f}s, VRAM {v0}->{peak} MiB, "
      f"usage={usage}", flush=True)
print(f"[client] >> {txt[:400]!r}", flush=True)
metrics = json.loads(urllib.request.urlopen(
    f"{URL}/v1/px/metrics/{MODEL}", timeout=30).read())
print(f"[client] px-Metriken keys={sorted(metrics.keys())} "
      f"phi={metrics.get('phi')} steps={metrics.get('steps')}", flush=True)
print("OK: Server-Langpfad gruen (kv4 + chunked prefill durch /v1-API)",
      flush=True)