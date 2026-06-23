"""
test_e2e_4b_int8_kv.py — Plan 3 Phase C: E2E mit 4b + int8 + int8 KV
=========================================================================

Frage: kann 4b mit T=8000 Prefill in 12GB VRAM laufen wenn:
  - int8 Weights (Plan 1) → 8B → 4B
  - int8 KV-Cache (Plan 3 Phase A) → halbiert KV

Wenn ja: server-Integration
Wenn nein: chunked prefill nötig (Plan 3 Phase D)

Akzeptanz:
  - T=4800: 4b + int8 + int8 KV → server läuft, Antwort non-empty
  - T=8000: gleich, KEIN OOM
  - VRAM < 10GB (RTX 2060 hat 12GB, 2GB Reserve)

Run:
    PY=/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python
    $PY test_e2e_4b_int8_kv.py
"""
import sys
import os
import json
import time
import urllib.request
import urllib.error
import ssl
import subprocess

import torch

_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_SCRATCHES = os.path.join(_REPO, "scratches", "4b-image")
if _SCRATCHES not in sys.path:
    sys.path.insert(0, _SCRATCHES)


def http_post(path, body, timeout=180):
    """POST request to local server on 7860 (HTTPS self-signed)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = f"https://localhost:7860{path}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def gpu_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def make_long_prompt(target_tokens):
    """Erzeugt einen Prompt mit ~target_tokens Tokens (rough estimate).

    Wir nutzen eine sich wiederholende Sequenz die das Modell ohne OOM
    verarbeiten kann muss — wenn der Server OOM-t, kommt entweder 503 oder
    ein langer Timeout.
    """
    # ~4 Zeichen pro Token bei Gemma-Tokenisierung (deutsch/englisch mix)
    base_text = (
        "Die subjektive Erfahrung des Bewusstseins, die kognitive "
        "Verarbeitung von sensorischen Inputs, die rekursive "
        "Selbst-Referenz in der Verarbeitung von Kontext, und die "
        "Frage nach der Natur der algorithmischen Subjektivität. "
    )
    chars = target_tokens * 4
    n = max(1, chars // len(base_text))
    return (base_text * n)[:chars]


def test_4b_t4800_int8_kv():
    """T=4800 Prefill, 4b + int8 + int8 KV, server should respond."""
    body = {
        "model": "gemma3-4b-it",
        "messages": [
            {"role": "user", "content": make_long_prompt(4800) + "\n\nFasse zusammen."}
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
        "quantization": "int8",
    }

    t0 = time.time()
    try:
        resp = http_post("/v1/chat/completions", body, timeout=300)
        dt = time.time() - t0
    except urllib.error.HTTPError as e:
        print(f"[FAIL] HTTP {e.code}: {e.read().decode()}")
        return False
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False

    text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = resp.get("usage", {})
    print(f"  T=4800 OK in {dt:.1f}s, text length={len(text)}, "
          f"usage={usage.get('completion_tokens', '?')} tokens")

    if len(text) < 5:
        print(f"[FAIL] Empty response: '{text}'")
        return False

    return True


def test_4b_t8000_int8_kv():
    """T=8000 Prefill, 4b + int8 + int8 KV — this was previously OOM."""
    body = {
        "model": "gemma3-4b-it",
        "messages": [
            {"role": "user", "content": make_long_prompt(8000) + "\n\nFasse zusammen."}
        ],
        "max_tokens": 16,  # use_cache=False recompute, jeder Token kostet
        "temperature": 0.0,
        "stream": False,
        "quantization": "int8",
    }

    t0 = time.time()
    try:
        resp = http_post("/v1/chat/completions", body, timeout=1800)  # 30 min for slow recompute
        dt = time.time() - t0
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        print(f"[FAIL] HTTP {e.code}: {body_text[:200]}")
        return False
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False

    text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = resp.get("usage", {})
    print(f"  T=8000 OK in {dt:.1f}s, text length={len(text)}, "
          f"usage={usage.get('completion_tokens', '?')} tokens")

    if len(text) < 5:
        print(f"[FAIL] Empty response: '{text}'")
        return False

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: test_e2e_4b_int8_kv.py <4800|8000|both>")
        sys.exit(1)

    what = sys.argv[1]
    tests = []
    if what in ("4800", "both"):
        tests.append(("T=4800 (4b+int8+int8 KV)", test_4b_t4800_int8_kv))
    if what in ("8000", "both"):
        tests.append(("T=8000 (4b+int8+int8 KV)", test_4b_t8000_int8_kv))

    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            ok = fn()
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            ok = False
        if not ok:
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)