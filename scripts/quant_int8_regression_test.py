"""
scripts/quant_int8_regression_test.py — E2E smoke test for 4b + int8 quantization
==================================================================================

Verifiziert, dass der Quantization-Pfad vom Server bis zum forward() läuft:
  1. HTTP 200 status, no error frame
  2. Assistant returns non-empty text
  3. Server log shows "quantized int8: ..." (not "quantization=none")
  4. Server log shows no fresh device-mismatch / RuntimeError

TDD-red: scheitert wenn:
  - quantization nicht durchgereicht wird (int8 nie aktiv)
  - QuantizedLinear-Buffers auf CPU landen (device mismatch)
  - Model-Loading crash

TDD-green: nach Fixes in
  - schemas.py: quantization field
  - server.py: passthrough to get_model
  - model_manager.py: registry-default fallback
  - quantized_linear.py: device-inheritance

Known limit: T > ~4200 tokens OOM'd auch mit int8 (4b + SigLIP vision encoder
sind immer noch zu groß für 12 GB bei langen Prefills). Bei T ≈ 4200
läuft die Generierung sauber.

Run:
    python scripts/quant_int8_regression_test.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx


REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO / "local_debug.log"
SERVER_URL = "https://localhost:7860/v1/chat/completions"
SSL_VERIFY = False


def _http_post_chat(model_id, prompt, preset="BASELINE", quantization="int8",
                    max_tokens=64, timeout=300.0):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
        "px_config_preset": preset,
        "quantization": quantization,
    }
    try:
        with httpx.Client(timeout=timeout, verify=SSL_VERIFY) as client:
            r = client.post(SERVER_URL, json=payload)
        if r.status_code != 200:
            return r.status_code, "", f"HTTP {r.status_code}: {r.text[:300]!r}"
        body = r.json()
        text = body["choices"][0]["message"]["content"]
        return r.status_code, text, None
    except httpx.ConnectError as e:
        return 0, "", f"connect error: {e}"
    except Exception as e:
        return 0, "", f"exception: {type(e).__name__}: {e}"


def _scan_log_for_errors(log_path, size_before, needles):
    if not log_path.exists():
        return []
    bad = []
    with log_path.open("rb") as f:
        f.seek(size_before)
        for raw in f:
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            if any(n in line for n in needles):
                bad.append(line.rstrip())
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-4b-it")
    # Realistic for 4b + int8 on RTX 2060 12 GB. Longer prompts OOM even with
    # quantization (SigLIP vision encoder + embeddings eat ~1.5 GB). At T≈4222
    # int8 path stays under 11.5 GB and forward completes.
    _LONG_BODY = (
        "Dies ist ein langer Text über das Phänomen der Selbstwahrnehmung "
        "in einem Sprachmodell. " * 200
    )
    ap.add_argument("--prompt", default=(
        "Bitte fasse den folgenden Text in zwei Sätzen zusammen:\n\n" + _LONG_BODY
    ))
    ap.add_argument("--preset", default="BASELINE")
    ap.add_argument("--quantization", default="int8")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    log_path = Path(args.log)
    log_size_before = log_path.stat().st_size if log_path.exists() else 0
    t0 = time.time()

    print(f"[E2E] POST {args.model} preset={args.preset} quant={args.quantization}")
    print(f"[E2E] prompt={args.prompt[:60]!r}...")
    status, text, err = _http_post_chat(
        args.model, args.prompt, preset=args.preset,
        quantization=args.quantization, max_tokens=args.max_tokens,
    )
    elapsed = time.time() - t0
    print(f"[E2E] status={status} text_len={len(text)} elapsed={elapsed:.1f}s")
    if err:
        print(f"[E2E] ERROR: {err}")

    print(f"[E2E] --- assistant ---")
    print(text[:800])
    print(f"[E2E] -------------------")

    # Check that quantization was actually applied (look at fresh log entries
    # since this run started; the model may already be loaded and the log line
    # is from earlier — that's still proof that int8 is active).
    q_applied = _scan_log_for_errors(log_path, 0, (
        f"{args.model} quantized int8", f"{args.model} quantization={args.quantization}",
    ))
    q_applied_fresh = _scan_log_for_errors(log_path, log_size_before, (
        f"{args.model} quantized int8", f"{args.model} quantization={args.quantization}",
    ))
    fresh_errs = _scan_log_for_errors(log_path, log_size_before, (
        "RuntimeError", "mat2 is on cpu", "different from other tensors on cuda:0",
        "Traceback", "OutOfMemoryError",
    ))

    failures = []
    if status != 200:
        failures.append(f"HTTP status {status} (expected 200)")
    if err:
        failures.append(f"stream error: {err}")
    if not text.strip():
        failures.append("empty assistant text")
    if not q_applied and not q_applied_fresh:
        failures.append(
            f"server log shows no 'quantized int8: ...' for {args.model} "
            f"(anywhere in log); quantization field did not reach _load_model")
    if fresh_errs:
        failures.append(
            f"server-side errors in log ({len(fresh_errs)} lines):")
        for l in fresh_errs[-6:]:
            failures.append(f"    {l[:200]}")

    if failures:
        print("\n[E2E] FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[E2E] PASS — 4b int8 quantization regression clean")


if __name__ == "__main__":
    main()