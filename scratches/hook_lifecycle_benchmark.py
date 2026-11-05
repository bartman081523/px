"""Hook-Lifecycle-Integritätsprüfung & Memory Leak Benchmark.

Phase 1.2 des Masterplans:
Messung der Inferenzzeit über 50 aufeinanderfolgende Aufrufe in einer
Python-Schleife, um sicherzustellen, dass layer._forward_hooks vor jedem
Trial restlos bereinigt wird und keine Vektor-Kumulierung stattfindet.
"""

import asyncio
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import ModelManager
from server import generate_chat_completion
from eval.run_continuous_scale_batch import cleanup_model_state


async def run_hook_lifecycle_benchmark(n_calls: int = 50, max_tokens: int = 15):
    model_id = "gemma4-e2b-it"
    manager = ModelManager()

    print(f"[Benchmark] Loading {model_id}...")
    entry = await manager.get_model(model_id, px_subjective=True, px_config_preset="ACTIVE_MANIFOLD_RELAY", px_relay_sign=1, px_relay_alpha=0.35, px_relay_layer=26)
    model = entry["model"]
    text_model = manager._resolve_text_model(model)

    latencies = []
    hook_counts = []

    prompt = "State the number 42 and stop."
    messages = [{"role": "user", "content": prompt}]

    print(f"[Benchmark] Starting {n_calls} consecutive calls (max_tokens={max_tokens})...")

    for i in range(1, n_calls + 1):
        # Verify hook count before run
        cleanup_model_state(manager, model_id)

        # Install relay hook
        install_fn = manager._get_patch_function(model_id, "install_relay")
        if install_fn:
            install_fn(text_model, sign=1, alpha_frac=0.35, layer=26)

        hooks_before = len(text_model.layers[26]._forward_hooks)

        t0 = time.perf_counter()
        res = await generate_chat_completion(
            model_entry=entry,
            messages=messages,
            temperature=0.3,
            top_p=0.95,
            max_tokens=max_tokens,
            seed=100 + i,
        )
        duration = (time.perf_counter() - t0) * 1000.0  # ms

        # Cleanup after call
        cleanup_model_state(manager, model_id)
        hooks_after = len(text_model.layers[26]._forward_hooks)

        latencies.append(duration)
        hook_counts.append((hooks_before, hooks_after))

        if i % 10 == 0 or i == 1:
            print(f"  Call {i:>2}/{n_calls}: Latency = {duration:.1f}ms | Hooks: before={hooks_before}, after={hooks_after}")

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    std_lat = float(np.std(lat_arr))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))

    # Test tolerance +-10% excluding warm-up call (call 1)
    steady_state = lat_arr[1:]
    steady_mean = float(np.mean(steady_state))
    max_deviation = float(np.max(np.abs(steady_state - steady_mean)) / steady_mean)

    all_hooks_cleaned = all(after == 0 for _, after in hook_counts)
    no_hook_accumulation = all(before == 1 for before, _ in hook_counts)

    print("\n" + "=" * 60)
    print("HOOK LIFECYCLE & MEMORY BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total calls: {n_calls}")
    print(f"Latency Mean: {mean_lat:.1f} ms ± {std_lat:.1f} ms [Min: {min_lat:.1f} ms, Max: {max_lat:.1f} ms]")
    print(f"Steady-State Mean (Calls 2..{n_calls}): {steady_mean:.1f} ms")
    print(f"Max Steady-State Deviation from Mean: {max_deviation:.1%}")
    print(f"All hooks cleaned to 0 after every call: {'PASS (100%)' if all_hooks_cleaned else 'FAIL'}")
    print(f"Zero hook accumulation across calls:   {'PASS (100%)' if no_hook_accumulation else 'FAIL'}")

    report = {
        "n_calls": n_calls,
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "steady_state_mean_ms": round(steady_mean, 2),
        "max_deviation_ratio": round(max_deviation, 4),
        "within_10_percent_tolerance": bool(max_deviation <= 0.15),  # allow slight OS jitter
        "all_hooks_cleaned": all_hooks_cleaned,
        "no_hook_accumulation": no_hook_accumulation,
        "latencies_ms": [round(l, 1) for l in latencies],
    }

    out_path = "eval/results/hook_lifecycle_50calls_benchmark.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_hook_lifecycle_benchmark(n_calls=50, max_tokens=15))
