#!/usr/bin/env python
"""Stufe-3f-Preset-Matrix am Server (Stufe-3e-Wiring, Registry-Default gf3).

Fuer jedes der 4 validen Presets (BASELINE, ACTIVE_MANIFOLD,
ACTIVE_MANIFOLD_LEAN, ACTIVE_MANIFOLD_RELAY) ein Kurz-Completion-Request
(px_thinking=False, 48 Tok) + px-Metriken + GPU-VRAM. Erster Request triggert
den Lazy-Load (GF3-Runtime ~5 s + px-Patch), Preset-Wechsel laufen ueber
_reapply_patch (kein Weight-Reload).

Erwartung aus Repack: max|Delta(lm_head-GF3 vs PTQ1_0)| = 0.0000 -> Ausgaben
sollten zwischen den GF3-Presets qualitativ stabil sein.
"""
import json
import subprocess
import sys
import time
import urllib.request

URL = "http://localhost:7860"
MODEL = "ternary-bonsai-27b"

PROMPT = ("Du stehst am Fenster und siehst auf die lange Strasse. "
          "In drei Saetzen: Was siehst du?")
PRESETS = ["BASELINE", "ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN",
           "ACTIVE_MANIFOLD_RELAY"]


def _post(path, body):
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def _get(path):
    with urllib.request.urlopen(URL + path, timeout=30) as r:
        return json.loads(r.read())


def _vram():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20)
    return int(out.stdout.strip().splitlines()[0])


def short(preset):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 48,
        "temperature": 0.7,
        "px_config_preset": preset,
        "px_thinking": False,
    }
    t0 = time.perf_counter()
    res = _post("/v1/chat/completions", body)
    dt = time.perf_counter() - t0
    txt = res["choices"][0]["message"]["content"]
    usage = res.get("usage", {})
    return dt, txt, usage


def main():
    print(f"== Preset-Matrix @ {URL} ({MODEL}) ==", flush=True)
    info = _get(f"/v1/models/{MODEL}")
    print(f"   Registry: px_loaded={info['px_loaded']} "
          f"model_type={info['model_type']}", flush=True)
    for preset in PRESETS:
        before = _vram()
        try:
            dt, txt, usage = short(preset)
        except Exception as exc:
            body = str(exc)
            if hasattr(exc, "read"):          # HTTPError-Body mitnehmen
                body = exc.read().decode(errors="replace")[:400]
            print(f"   {preset}: FEHLER {body}", flush=True)
            print("FAIL: Preset-Matrix rot", flush=True)
            sys.exit(1)
        peak = _vram()
        metrics = _get(f"/v1/px/metrics/{MODEL}")
        keys = sorted(metrics.keys())
        phi = metrics.get("phi", "?")
        ent = metrics.get("entropy", "?")
        path = metrics.get("path", "?")
        steps = metrics.get("steps", "?")
        ct = usage.get("completion_tokens", "?")
        print(f"   {preset}: {dt:.1f}s, {ct} Tok, VRAM {before}->{peak} MiB, "
              f"phi={phi} ent={ent} path={path} steps={steps}", flush=True)
        print(f"      keys={keys}", flush=True)
        print(f"      >> {txt[:160]!r}", flush=True)
    print("OK: Preset-Matrix gruen (4/4 Presets, Kurzpfad)", flush=True)


if __name__ == "__main__":
    main()