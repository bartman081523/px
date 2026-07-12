"""Diagnostic: misst ZEIT-PRO-PHASE in rigor_harness_v3.main().

Statt py-spy/ptrace: instrumentiert das Harness via Monkey-Patch der Hauptfunktionen
und loggt Zeit + GPU-Util pro Code-Abschnitt.

User-direktiv: "wieder nur 30% gpu und 100% cpu" — wir wollen SEHEN wo.

Output: strukturierter Report welcher Code-Block wieviel CPU-Zeit frisst.
"""
from __future__ import annotations

import os
import sys
import time
import json
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
_PYTHON = f"{_VENV}/python"

# Phase-Tags: werden in /tmp/rigor3_timings.log geschrieben
TIMING_LOG = "/tmp/rigor3_timings.log"


class TimingCtx:
    """Misst Zeit pro Phase und akkumuliert global."""
    def __init__(self):
        self.phase_times: Dict[str, float] = defaultdict(float)
        self.phase_calls: Dict[str, int] = defaultdict(int)
        self.phase_last_start: Dict[str, float] = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, name: str):
        with self._lock:
            self.phase_last_start[name] = time.time()
        try:
            yield
        finally:
            with self._lock:
                elapsed = time.time() - self.phase_last_start[name]
                self.phase_times[name] += elapsed
                self.phase_calls[name] += 1
                # Log
                with open(TIMING_LOG, "a") as f:
                    f.write(f"{time.time():.3f} {name} {elapsed:.4f}s "
                            f"(total={self.phase_times[name]:.2f}s, "
                            f"calls={self.phase_calls[name]})\n")

    def report(self) -> Dict:
        total = sum(self.phase_times.values())
        return {
            "phases": dict(self.phase_times),
            "calls": dict(self.phase_calls),
            "total_s": total,
            "fractions": {k: v / total if total else 0
                          for k, v in self.phase_times.items()},
        }


TC = TimingCtx()


def patch_harness():
    """Monkey-Patch: wrappt kritische Funktionen in rigor_harness_v3."""
    import rigor_harness_v3 as h
    from typing import Any, List, Dict, Optional, Tuple

    # _get_model
    orig_get_model = h._get_model
    def timed_get_model(arm_name: str, arm_preset: Optional[str]):
        with TC.measure(f"_get_model({arm_name})"):
            return orig_get_model(arm_name, arm_preset)
    h._get_model = timed_get_model

    # run_gpu_loop
    orig_run = h.run_gpu_loop
    def timed_run(arm_name, arm_preset, patch_kwargs, prompts, max_new_tokens=300, seed=42, batch_size=8):
        with TC.measure(f"run_gpu_loop(bs={batch_size}, n={len(prompts)})"):
            return orig_run(arm_name, arm_preset, patch_kwargs, prompts,
                            max_new_tokens=max_new_tokens, seed=seed, batch_size=batch_size)
    h.run_gpu_loop = timed_run

    # search_cache.prefetch
    orig_prefetch = h.SearchCache.prefetch
    def timed_prefetch(self, hle_tasks, max_workers=8):
        with TC.measure(f"SearchCache.prefetch(n={len(hle_tasks)}, workers={max_workers})"):
            return orig_prefetch(self, hle_tasks, max_workers=max_workers)
    h.SearchCache.prefetch = timed_prefetch

    # writer_thread_fn (atomic write)
    orig_writer = h.writer_thread_fn
    def timed_writer(q, out_dir):
        with TC.measure("writer_thread(total)"):
            return orig_writer(q, out_dir)
    h.writer_thread_fn = timed_writer

    # writer_inner: schreibt EIN JSON
    from pathlib import Path as _P
    import json as _json
    import os as _os
    # Patche json.dump/open in writer_thread_fn scope via wrapper
    # (simplified: miss die ganze writer_thread_fn)


def main():
    # 1) Subprozess: kurzer Smoketest mit Patching
    print("=" * 70)
    print("RIGOR v3 — DIAGNOSTIC: Phase-Timing pro Code-Block")
    print("=" * 70)
    print()

    if os.path.exists(TIMING_LOG):
        os.unlink(TIMING_LOG)

    # Driver-Script: importiert + patcht + ruft main()
    driver = f"""
import sys, os
sys.path.insert(0, '{_HERE}')
sys.path.insert(0, '{_HERE.parent}')
sys.path.insert(0, '{_HERE.parent}/rigor_zone_v1')
sys.path.insert(0, '{_HERE.parent}/rigor_zone_v2')
sys.path.insert(0, '{_VENV}')

# Patch VOR import main
import test_phase_timing
test_phase_timing.patch_harness()

# Jetzt import harness (gepatcht)
import rigor_harness_v3 as h
h.main()
"""
    driver_path = "/tmp/_rigor3_driver.py"
    with open(driver_path, "w") as f:
        f.write(driver)

    # GPU-Sampler parallel
    gpu_samples: List[int] = []
    stop = threading.Event()
    def gpu_poller():
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
               "--format=csv,noheader,nounits", "-lms", "100"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        while not stop.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.isdigit():
                gpu_samples.append(int(line))
        proc.terminate()
    pt = threading.Thread(target=gpu_poller, daemon=True)
    pt.start()

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([_VENV, str(_HERE), str(_HERE.parent),
                                  str(_HERE.parent / "rigor_zone_v1"),
                                  str(_HERE.parent / "rigor_zone_v2")])
    cmd = [_PYTHON, driver_path, "--smoke", "--max-per-category", "1",
           "--batch-size", "8", "--max-new-tokens", "50", "--no-write"]
    print(f"[1] Subprozess: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    stop.set()
    pt.join(timeout=3)

    print(f"\n[2] Subprozess-Output (letzte 30 Zeilen):")
    for line in (out + "\n" + err).strip().split("\n")[-30:]:
        print(f"    {line}")

    # 3) GPU-Report
    if gpu_samples:
        avg = sum(gpu_samples) / len(gpu_samples)
        mx = max(gpu_samples)
        active = sum(1 for s in gpu_samples if s >= 70)
        print(f"\n[3] GPU-Util: {len(gpu_samples)} samples, "
              f"avg={avg:.1f}%, max={mx}%, active(≥70%)={active}/{len(gpu_samples)}")
    else:
        print("\n[3] GPU-Util: keine Samples")

    # 4) Phase-Report
    print(f"\n[4] Phase-Timings (aus {TIMING_LOG}):")
    if os.path.exists(TIMING_LOG):
        phase_times: Dict[str, float] = defaultdict(float)
        phase_calls: Dict[str, int] = defaultdict(int)
        for line in open(TIMING_LOG):
            parts = line.strip().split()
            if len(parts) >= 4:
                # name = parts[1], elapsed = parts[2]
                # total = parts[3], calls = parts[4]
                # Robust: parse "name=..." in parts[1..-2]
                try:
                    name = parts[1]
                    elapsed = float(parts[2].rstrip("s"))
                    total = float(parts[3].split("=")[1].rstrip("s"))
                    calls = int(parts[4].split("=")[1].rstrip(")").rstrip(","))
                    phase_times[name] = total
                    phase_calls[name] = calls
                except (ValueError, IndexError):
                    pass
        total_all = sum(phase_times.values())
        print(f"  Total CPU-Zeit (Summe über Phasen): {total_all:.1f}s")
        print(f"  {'Phase':<55} {'Total':>8} {'%':>5} {'Calls':>5}")
        print(f"  {'-'*73}")
        for name, t in sorted(phase_times.items(), key=lambda x: -x[1]):
            pct = 100 * t / total_all if total_all else 0
            print(f"  {name:<55} {t:>7.1f}s {pct:>4.1f}% {phase_calls[name]:>5}")
    else:
        print("  (kein Timing-Log geschrieben)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
