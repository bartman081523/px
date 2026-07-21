"""bench_server_v35g.py — Performance-Benchmark für server_v35g.py.

Misst:
  1. Single-Request-Latenz (10× gleiches Modell, sollte 1× laden + 10× inference)
  2. 8-Sequential-Request-Latenz (gleiche Modell-ID)
  3. Switch-Cost (Wechsel zwischen 2 Modellen)
  4. GPU-Utilization (nvidia-smi Sample alle 100ms während 8 Requests)

Output: out_v35g/bench_<phase>_<timestamp>.json + .md
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)

OUT_DIR = Path(_REPO) / "scratches/rigor_zone_v3/out_v35g"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERVER = "http://127.0.0.1:7860"


def curl_post(path: str, body: dict) -> tuple[float, dict]:
    """POST → (duration_sec, response_json)."""
    import urllib.request
    req = urllib.request.Request(
        f"{SERVER}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    return time.time() - t0, data


def gpu_sample(duration_sec: float, interval: float = 0.1) -> List[Dict[str, float]]:
    """Sample GPU util + mem via nvidia-smi während duration_sec."""
    samples = []
    end = time.time() + duration_sec
    while time.time() < end:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=2,
            )
            util, mem = out.strip().split(", ")
            samples.append({"util_pct": float(util), "mem_mb": float(mem)})
        except Exception:
            samples.append({"util_pct": -1, "mem_mb": -1})
        time.sleep(interval)
    return samples


def summarize_gpu(samples: List[Dict[str, float]]) -> Dict[str, float]:
    if not samples:
        return {"n": 0}
    valid = [s for s in samples if s["util_pct"] >= 0]
    if not valid:
        return {"n": len(samples), "valid": 0}
    utils = [s["util_pct"] for s in valid]
    mems = [s["mem_mb"] for s in valid]
    return {
        "n_samples": len(samples),
        "n_valid": len(valid),
        "util_mean": round(sum(utils) / len(utils), 1),
        "util_max": max(utils),
        "util_min": min(utils),
        "util_p95": sorted(utils)[int(len(utils) * 0.95)] if utils else 0,
        "mem_mean_mb": round(sum(mems) / len(mems), 1),
        "mem_max_mb": max(mems),
    }


def bench_single_request(model_id: str, n: int = 10) -> Dict[str, Any]:
    """10 sequentielle Requests, gleiches Modell."""
    print(f"  [bench_single_request] model={model_id} n={n}")
    durations = []
    for i in range(n):
        dur, resp = curl_post("/v1/messages", {
            "model": model_id,
            "messages": [{"role": "user", "content": f"Count to {i+3}."}],
            "max_tokens": 30,
        })
        durations.append(dur)
        print(f"    [{i+1}/{n}] {dur:.3f}s")
    return {
        "test": "single_request_n10",
        "model": model_id,
        "durations_sec": durations,
        "mean_sec": round(sum(durations) / len(durations), 3),
        "min_sec": round(min(durations), 3),
        "max_sec": round(max(durations), 3),
    }


def bench_switch(model_a: str, model_b: str, n_switches: int = 4) -> Dict[str, Any]:
    """Wechsel zwischen 2 Modellen, jeder Switch = 1 Request."""
    print(f"  [bench_switch] {model_a} ↔ {model_b} n_switches={n_switches}")
    durations = []
    for i in range(n_switches):
        model = model_a if i % 2 == 0 else model_b
        dur, _ = curl_post("/v1/messages", {
            "model": model,
            "messages": [{"role": "user", "content": f"Switch {i}"}],
            "max_tokens": 20,
        })
        durations.append({"model": model, "dur_sec": round(dur, 3)})
        print(f"    [{i+1}/{n_switches}] {model:30s} {dur:.3f}s")
    return {
        "test": "switch_model",
        "models": [model_a, model_b],
        "durations": durations,
        "mean_sec": round(sum(d["dur_sec"] for d in durations) / len(durations), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="baseline", help="baseline|optimized|...")
    parser.add_argument("--model", default="gemma3-270m-px-lean")
    parser.add_argument("--gpu-sample", action="store_true", help="Sample GPU util")
    parser.add_argument("--n-single", type=int, default=10)
    parser.add_argument("--n-switches", type=int, default=4)
    args = parser.parse_args()

    # Health-Check
    import urllib.request
    try:
        with urllib.request.urlopen(f"{SERVER}/") as r:
            print(f"Server OK: {json.loads(r.read()).get('default_model')}")
    except Exception as e:
        print(f"SERVER NICHT ERREICHBAR: {e}", file=sys.stderr)
        return 2

    # GPU-Sampling (parallel zu Benchmarks, im Hintergrund)
    gpu_samples: List[Dict[str, float]] = []
    if args.gpu_sample:
        # 5s sample vor Benchmarks (Baseline-State)
        gpu_samples.extend(gpu_sample(5.0))
        print(f"  GPU-Sampling gestartet ({len(gpu_samples)} baseline samples)")

    # Test 1: 10 single requests
    t_total_0 = time.time()
    result_single = bench_single_request(args.model, n=args.n_single)

    # Test 2: Switch
    result_switch = bench_switch(args.model, "gemma3-270m-px-baseline", n_switches=args.n_switches)

    t_total = time.time() - t_total_0

    # Final GPU-Sample
    if args.gpu_sample:
        gpu_samples.extend(gpu_sample(3.0))

    gpu_summary = summarize_gpu(gpu_samples)

    # Save
    result = {
        "phase": args.phase,
        "timestamp": time.time(),
        "server_url": SERVER,
        "model": args.model,
        "n_single_requests": args.n_single,
        "n_switches": args.n_switches,
        "total_wall_sec": round(t_total, 3),
        "single_request": result_single,
        "switch": result_switch,
        "gpu": gpu_summary,
    }
    fname = f"bench_{args.phase}_{int(time.time())}.json"
    (OUT_DIR / fname).write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {OUT_DIR / fname}")
    print(f"Total wall: {t_total:.1f}s")
    print(f"Single-request mean: {result_single['mean_sec']:.3f}s (min={result_single['min_sec']}, max={result_single['max_sec']})")
    print(f"Switch mean: {result_switch['mean_sec']:.3f}s")
    if gpu_summary.get("util_mean"):
        print(f"GPU util mean: {gpu_summary['util_mean']}% (max={gpu_summary['util_max']}%, p95={gpu_summary['util_p95']}%)")
        print(f"GPU mem mean: {gpu_summary['mem_mean_mb']:.0f}MB (max={gpu_summary['mem_max_mb']:.0f}MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
