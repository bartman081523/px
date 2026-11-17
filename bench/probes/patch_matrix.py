#!/usr/bin/env python
"""Modell × Patch-Matrix gegen den Harness-Prompt.

Ziel: für jedes Modell und jeden PX-Preset prüfen, ob der Harness-Prompt
(~1 400 Tokens, Tool-Schema) ein sauberes Ergebnis liefert oder in einen
Degenerations-Loop läuft — plus Latenz und Tool-Call-Erkennung.

Modelle: gemma3-270m-it, gemma3-1b-it, gemma3-4b-it, gemma4-e2b-it
         (bf16 und int8), minicpm5-1b, ternary-bonsai-27b.
Presets: BASELINE, ACTIVE_MANIFOLD, ACTIVE_MANIFOLD_LEAN,
         ACTIVE_MANIFOLD_RELAY(+1), ACTIVE_MANIFOLD_RELAY(-1).

GF(5)-Artefakte sind bewusst NICHT dabei (verworfen, siehe Roadmap §B3).

Aufruf:
    python bench/probes/patch_matrix.py --models ... --presets ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_prompt_compare import SYSTEM, TASK, TOOLS, repeat_ratio  # noqa: E402

API = os.environ.get("PX_API", "http://127.0.0.1:7860/v1")

DEFAULT_MODELS = [
    "gemma3-270m-it", "gemma3-1b-it", "gemma3-4b-it",
    "gemma4-e2b-it", "minicpm5-1b", "ternary-bonsai-27b",
]
DEFAULT_PRESETS = [
    "BASELINE", "ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN",
    "ACTIVE_MANIFOLD_RELAY:+1", "ACTIVE_MANIFOLD_RELAY:-1",
]


def ask(model, preset, quantization=None, max_tokens=256, timeout=1800):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": TASK}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "tools": TOOLS,
    }
    if preset and preset != "BASELINE":
        name, _, sign = preset.partition(":")
        body["px_config_preset"] = name
        if sign:
            body["px_relay_sign"] = int(sign)
            body["px_relay_alpha"] = 0.3
    else:
        body["px_config_preset"] = "BASELINE"
    if quantization:
        body["quantization"] = quantization
    req = urllib.request.Request(f"{API}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    c = d["choices"][0]
    text = c["message"].get("content") or ""
    return {
        "wall_s": round(dt, 1),
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "completion_tokens": d["usage"]["completion_tokens"],
        "finish": c["finish_reason"],
        "tool_calls": bool(c["message"].get("tool_calls")),
        "repeat_ratio": round(repeat_ratio(text), 3),
        "preview": text[:160],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--presets", default=",".join(DEFAULT_PRESETS))
    ap.add_argument("--out", default="bench/results/patch_matrix.json")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    presets = [p.strip() for p in a.presets.split(",") if p.strip()]

    rows = []
    for model in models:
        quants = [None, "int8"] if model == "gemma4-e2b-it" else [None]
        for quant in quants:
            for preset in presets:
                label = f"{model}{'/' + quant if quant else ''}/{preset}"
                print(f"\n=== {label} ===", flush=True)
                try:
                    r = ask(model, preset, quant)
                except Exception as e:  # noqa: BLE001
                    r = {"error": f"{type(e).__name__}: {str(e)[:140]}"}
                r.update({"model": model, "quantization": quant, "preset": preset})
                rows.append(r)
                print(json.dumps(r, ensure_ascii=False)[:260], flush=True)
                with open(a.out, "w") as f:
                    json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
