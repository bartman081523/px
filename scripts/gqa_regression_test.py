"""
scripts/gqa_regression_test.py — E2E smoke test for GQA on the 4b model
========================================================================

The bug: _chunked_attention crashes for Gemma3 4b (Hq=8, Hkv=4 — 2:1 GQA)
because torch.matmul(qc, k.transpose) treats Hq/Hkv as batch dims and they
must match (or one be 1). Hq=8 vs Hkv=4 fails.

This test fires a real chat completion against the running server with
gemma3-4b-it, parses the streaming response, and verifies:
  1. HTTP 200 status, no error frame in SSE stream
  2. Assistant returns non-empty content
  3. Server log does NOT contain a fresh GQA/RuntimeError traceback

TDD-red: fails on the buggy _chunked_attention.
TDD-green: passes after the GQA fix in patch.py.

Run:
    python scripts/gqa_regression_test.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx


REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO / "local_debug.log"
SERVER_URL = "https://localhost:7860/v1/chat/completions"
SSL_VERIFY = False  # self-signed


def _http_post_chat(model_id, prompt, preset="BASELINE", max_tokens=128, timeout=300.0):
    """Send one streaming chat completion; return (status_code, full_text, error)."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "preset": preset,
    }
    try:
        with httpx.stream(
            "POST", SERVER_URL, json=payload, timeout=timeout, verify=SSL_VERIFY,
        ) as r:
            if r.status_code != 200:
                body = r.read()
                return r.status_code, "", f"HTTP {r.status_code}: {body[:300]!r}"

            chunks = []
            err = None
            for raw in r.iter_lines():
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception as e:
                    return 200, "", f"JSON parse error on chunk: {data!r} ({e})"
                # error frame?
                if "error" in obj:
                    err = obj["error"]
                    continue
                # text chunk?
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        chunks.append(content)
            text = "".join(chunks)
            if err:
                return 200, "", f"SSE error frame: {err}"
            return r.status_code, text, None
    except httpx.ConnectError as e:
        return 0, "", f"connect error: {e}"
    except Exception as e:
        return 0, "", f"exception: {type(e).__name__}: {e}"


def _scan_log_for_gqa_errors(log_path, size_before):
    """Return RuntimeError/GQA/OOM lines APPENDED after the given byte offset."""
    if not log_path.exists():
        return []
    bad = []
    needles = ("RuntimeError", "GQA", "OutOfMemoryError",
               "tensor a (8) must match tensor b (4)")
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
    # Long prompt (>4096 tokens) forces prefill T > MEM_EFF_THRESHOLD in
    # _mem_eff_attention_forward, which is what triggers the chunked path
    # where the GQA bug lives. Short prompts take the SDPA fast path that
    # handles GQA natively and never reaches _chunked_attention.
    # 4b + RTX 2060 12GB: at T=4100 with chunked path (chunk=256) per-layer
    # score matrix is 8*256*4100*4B = 33 MB. Total chunked memory fits in
    # the ~1 GB headroom after 4b model + KV cache (~10.5 GB). Larger T
    # or larger chunk pushes OOM past GPU free memory.
    _LONG_BODY = (
        "Dies ist ein langer Text über das Phänomen der Selbstwahrnehmung "
        "in einem Sprachmodell. " * 490
    )
    ap.add_argument("--prompt", default=(
        "Bitte fasse den folgenden Text in zwei Sätzen zusammen:\n\n" + _LONG_BODY
    ))
    ap.add_argument("--preset", default="BASELINE")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    log_path = Path(args.log)
    log_size_before = log_path.stat().st_size if log_path.exists() else 0
    t0 = time.time()

    print(f"[E2E] POST {args.model} (preset={args.preset}) prompt={args.prompt[:60]!r}...")
    status, text, err = _http_post_chat(
        args.model, args.prompt, preset=args.preset, max_tokens=args.max_tokens,
    )

    elapsed = time.time() - t0
    print(f"[E2E] status={status} text_len={len(text)} elapsed={elapsed:.1f}s")
    if err:
        print(f"[E2E] ERROR: {err}")

    print(f"[E2E] --- assistant ---")
    print(text[:800])
    print(f"[E2E] -------------------")

    gqa_errors = _scan_log_for_gqa_errors(log_path, log_size_before)
    # All post-call errors are real failures: GQA mismatch, CUDA OOM, etc.
    fresh_errs = [l for l in gqa_errors if (
        "Traceback" in l or "RuntimeError" in l or "OutOfMemoryError" in l
    )]

    failures = []
    if status != 200:
        failures.append(f"HTTP status {status} (expected 200)")
    if err:
        failures.append(f"stream error: {err}")
    if not text.strip():
        failures.append("empty assistant text")
    if fresh_errs:
        failures.append(f"server-side errors in log ({len(fresh_errs)} lines):")
        for l in fresh_errs[-6:]:
            failures.append(f"    {l[:200]}")

    if failures:
        print("\n[E2E] FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[E2E] PASS — 4b GQA regression clean")


if __name__ == "__main__":
    main()