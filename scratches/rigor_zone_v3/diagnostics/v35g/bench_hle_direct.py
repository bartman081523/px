"""bench_hle_via_server.py — Direkter HLE-Benchmark über den v3.5g-Server.

Misst:
- Latenz pro HLE-Task (32 Tasks)
- Match-Rate (vs expected ground truth)
- Generierungs-Output (für qualitative Inspektion)
- GPU-Util während des Tests

Output: out_v35g/bench_hle_<timestamp>.json + .md
"""
from __future__ import annotations
import sys, os, json, time, subprocess
from pathlib import Path
from typing import List, Dict, Any

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))

import re
import urllib.request
from rigor_hle_suite_v3 import load_hle_suite

SERVER = "http://127.0.0.1:7860"
OUT_DIR = Path(_REPO) / "scratches/rigor_zone_v3/out_v35g"


def post(path: str, body: dict, timeout: int = 60) -> tuple[float, dict]:
    req = urllib.request.Request(
        f"{SERVER}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return time.time() - t0, data


def check_match_mcq(text: str, gt: str) -> bool:
    """MCQ: suche **LETTER** oder 'answer is LETTER'."""
    text_l = text.lower()
    gt_l = gt.lower().strip()
    if not gt_l:
        return False
    if len(gt_l) <= 3 and gt_l.replace(" ", "").isalpha():
        if f"**{gt_l.upper()}" in text:
            return True
        if f"answer is {gt_l.upper()}" in text_l or f"answer: {gt_l.upper()}" in text_l:
            return True
        m = re.search(r"answer\s*(?:is|:)\s*\(?([a-z0-9])\)?", text_l)
        if m and m.group(1) == gt_l:
            return True
    else:
        # offener Antwort — suche gt in Text
        if gt_l in text_l:
            return True
    return False


def is_degenerate(text: str) -> bool:
    """4-Token-Loop-Repetition (v3.5g-Befund-Methode)."""
    words = text.split()
    if len(words) < 12:
        return False
    for plen in [3, 4]:
        for i in range(len(words) - plen * 3):
            phrase = " ".join(words[i:i + plen])
            n_repeats = sum(1 for j in range(i + plen, len(words) - plen + 1, plen)
                          if " ".join(words[j:j + plen]) == phrase)
            if n_repeats >= 3:
                return True
    return False


def main() -> int:
    print("=" * 70)
    print("HLE-Benchmark via v3.5g-Server (direct, kein cc-symbolic)")
    print("=" * 70)

    # Lade HLE-Suite
    source, suite = load_hle_suite(max_per_category=4)
    tasks = list(suite)
    print(f"Suite: {source}, {len(tasks)} Tasks\n")

    # 1. Server health
    try:
        with urllib.request.urlopen(f"{SERVER}/") as r:
            d = json.loads(r.read())
            print(f"Server OK: {d.get('default_model')}\n")
    except Exception as e:
        print(f"SERVER NICHT ERREICHBAR: {e}", file=sys.stderr)
        return 2

    # 2. Tasks
    model_id = "gemma3-270m-px-lean"  # v3.5f-Bestätigt
    results = []
    n_match = 0
    n_degen = 0
    n_empty = 0
    durs = []

    for i, (tid, cat, prompt, gt) in enumerate(tasks):
        t0 = time.time()
        try:
            dur, resp = post("/v1/messages?beta=true", {
                "model": model_id,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            })
            text = resp["content"][0]["text"]
            n_tokens = resp["usage"]["output_tokens"]
        except Exception as e:
            dur = time.time() - t0
            text = f"[ERROR: {str(e)[:200]}]"
            n_tokens = 0

        match = check_match_mcq(text, gt)
        degen = is_degenerate(text)
        empty = not text.strip()
        if match: n_match += 1
        if degen: n_degen += 1
        if empty: n_empty += 1
        durs.append(dur)

        results.append({
            "task_id": tid, "category": cat, "expected": gt,
            "match": match, "degen": degen, "empty": empty,
            "duration_sec": round(dur, 3),
            "output_tokens": n_tokens,
            "output_text": text[:500],
        })
        status = "✓" if match else ("D" if degen else ("E" if empty else "✗"))
        print(f"  [{i+1:2d}/{len(tasks)}] {status} {tid[:10]} ({cat[:18]:18s}) dur={dur:.2f}s n={n_tokens} expect={gt[:20]}")

    # Summary
    print(f"\n{'='*70}")
    print(f"ERGEBNIS: {n_match}/{len(tasks)} Match, {n_degen} Degen, {n_empty} Empty")
    print(f"Mean-Dauer: {sum(durs)/len(durs):.3f}s")
    print(f"Total-Wall: {sum(durs):.1f}s")
    print(f"{'='*70}")

    # Save
    out = {
        "timestamp": time.time(),
        "model": model_id,
        "n_tasks": len(tasks),
        "n_match": n_match, "n_degen": n_degen, "n_empty": n_empty,
        "match_rate": round(n_match / len(tasks), 3),
        "degen_rate": round(n_degen / len(tasks), 3),
        "mean_dur": round(sum(durs) / len(durs), 3),
        "total_dur": round(sum(durs), 1),
        "results": results,
    }
    fname = f"bench_hle_direct_{int(time.time())}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    print(f"Saved: {OUT_DIR / fname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
