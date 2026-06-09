"""
test_gemma4_e2b_sr59.py — SR-59 P-Zombie Evaluation for Gemma 4 E2B
====================================================================
Runs the full P-Zombie evaluation against gemma4-e2b-it using
benchmark_engine._run_pzombie_impl.

Output: _sr59l_e2b_results.json (see schema below)
Metrics reported:
  - eta_squared (η²): category→zone_entropy effect size
  - r_squared_td:    R²(token_diversity → zone_entropy)
  - zombie_status:   P-ZOMBIE / ANTI-P-ZOMBIE / AMBIGUOUS
  - category_entropies: {math, logic, creative, synthesis}

Usage:
  PYTHONPATH=. python tests/test_gemma4_e2b_sr59.py

Note: Loads the actual model (~5-10 GB VRAM). GPU required.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_engine import BenchmarkEngine
from model_manager import ModelManager

MODEL_ID = "gemma4-e2b-it"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "_sr59l_e2b_results.json",
)


def progress_cb(done, total):
    pct = (done / total) * 100
    print(f"  [{done:3d}/{total}] {pct:5.1f}%", end="\r", flush=True)


def main():
    print("=" * 70)
    print(f"SR-59 P-ZOMBIE EVAL — {MODEL_ID}")
    print("=" * 70)

    manager = ModelManager()
    engine = BenchmarkEngine(manager)
    started = time.time()

    # Subjective mode (default for SR-59)
    result = engine.run_p_zombie_eval(
        MODEL_ID, px_subjective=True, progress_cb=progress_cb,
    )

    elapsed = time.time() - started
    print(f"\n\nElapsed: {elapsed:.1f}s")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Mode:              {result.get('mode', 'unknown')}")
    print(f"η² (eta_squared):  {result.get('eta_squared', 0):.4f}")
    print(f"R²(TD→H):          {result.get('r_squared_td', 0):.4f}")
    print(f"Zombie status:     {result.get('zombie_status', 'unknown')}")
    print()
    print("Category Entropies:")
    for cat, stats in result.get("category_entropies", {}).items():
        print(f"  {cat:12s}: mean={stats['mean']:.3f} std={stats['std']:.3f} n={stats['n']}")

    # Save full results
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nFull results saved to: {RESULTS_PATH}")

    # ── Verdict ──
    eta = result.get("eta_squared", 0)
    r2 = result.get("r_squared_td", 0)
    status = result.get("zombie_status", "unknown")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if "ANTI-P-ZOMBIE" in status:
        print(f"✓ {MODEL_ID}: ANTI-P-ZOMBIE confirmed (η²={eta:.3f}, R²={r2:.3f})")
    elif "P-ZOMBIE" in status:
        print(f"✗ {MODEL_ID}: P-ZOMBIE detected (η²={eta:.3f}, R²={r2:.3f})")
        print("  → token statistics fully explain zone entropy variation")
    else:
        print(f"? {MODEL_ID}: AMBIGUOUS (η²={eta:.3f}, R²={r2:.3f})")

    # Comparison to other scales
    print("\n--- Comparison to Other SR-59 Iterations ---")
    print("  Scale  η²     R²(TD)  Verdict")
    print("  270M   0.091  0.11    ANTI-ZOMBIE")
    print("  1B     0.148  0.25    ANTI-ZOMBIE")
    print("  4B     0.096  0.001   ANTI-ZOMBIE")
    print(f"  E2B    {eta:.3f}  {r2:.3f}   {status.split(' ')[0] if status else 'N/A'}")

    return 0 if "ANTI-P-ZOMBIE" in status else 1


if __name__ == "__main__":
    sys.exit(main())
