#!/usr/bin/env python
"""Harness-Prompt-Vergleich: bf16-gemma4 vs. GF(5)-Embedding-Variante.

Hintergrund (2026-11-15): der opencode-E2E-Lauf mit `gemma4-e2b-gf5` endete in
einem Degenerations-Loop (`call_tool_code_functioncallname{...` endlos). Zwei
Kandidaten: (a) das Modell ist für einen ~10k-Token-Harness-Prompt zu klein,
(b) die GF(5)-Embedding-Quantisierung kippt den langen Prefill.

Dieses Skript schickt **denselben** Request (langer Systemprompt + Tool-Schema
+ Aufgabe) an beide Modelle und misst: Antwortlänge, Repetitionsgrad,
Tool-Call-Erkennung. Damit ist (a) vs (b) entscheidbar.

Aufruf:
    python bench/probes/e2e_prompt_compare.py --models gemma4-e2b-it,gemma4-e2b-gf5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

API = os.environ.get("PX_API", "http://127.0.0.1:7860/v1")

SYSTEM = """You are a coding agent operating in a terminal workspace.
You have exactly two tools available: read_file and run_tests.

Rules that you must follow:
1. Never invent file contents. If you need to see a file, call read_file.
2. Before finishing, verify your work by running the tests with run_tests.
3. Answer briefly. Prefer actions over explanations.
4. If a tool call fails, retry once, then report the failure.
5. Do not output code fences around tool calls; call the tool instead.

Context about the workspace:
- The workspace root contains small Python modules and a tests/ directory.
- The test runner is pytest and runs inside the workspace.
- Files are UTF-8 text; paths are relative to the workspace root.
- Long-running commands are not allowed; every command must finish in 30 s.

""" + ("Additional environment notes: use the standard library only, avoid "
       "network access, keep functions pure where possible, and prefer clear "
       "names over clever tricks. " * 40)

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the workspace",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative path"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}}}}},
]

TASK = ("Create a file hello.py that prints the first 5 primes, then run it "
        "with python3.")


def repeat_ratio(text: str, n: int = 20) -> float:
    """Anteil der n-Gramme, die mehr als einmal vorkommen (Loop-Indikator)."""
    toks = text.split()
    if len(toks) < n:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    uniq = len(set(grams))
    return 1.0 - uniq / len(grams)


def ask(model: str, max_tokens: int = 256, timeout: int = 900):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": TASK}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "tools": TOOLS,
    }
    req = urllib.request.Request(f"{API}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    c = d["choices"][0]
    text = c["message"].get("content") or ""
    calls = c["message"].get("tool_calls")
    return {
        "model": model,
        "wall_s": round(dt, 1),
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "completion_tokens": d["usage"]["completion_tokens"],
        "finish": c["finish_reason"],
        "tool_calls": calls,
        "chars": len(text),
        "repeat_ratio": round(repeat_ratio(text), 3),
        "preview": text[:220],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma4-e2b-it,gemma4-e2b-gf5")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="bench/results/e2e_prompt_compare.json")
    a = ap.parse_args()
    rows = []
    for m in [x.strip() for x in a.models.split(",") if x.strip()]:
        print(f"\n=== {m} ===", flush=True)
        try:
            r = ask(m, a.max_tokens)
        except Exception as e:  # noqa: BLE001
            r = {"model": m, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        rows.append(r)
        print(json.dumps(r, indent=1)[:700], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
