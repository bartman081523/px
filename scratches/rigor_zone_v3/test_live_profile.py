"""TDD-Profiling: Lokalisiert den ECHTEN CPU-Bottleneck im rigor_harness_v3.py.

Methodik (TDD, User-direktiv: "TDD bitte und best practices"):
  1. Starte rigor_harness_v3.main() als Subprozess (klein: --smoke, 1 Task, 2 Arms)
  2. Polle ALLE 50ms: nvidia-smi GPU-Util, ps CPU-%, py-spy dump (welche Funktion läuft?)
  3. Sammle alle Hotspots (welche Python-Funktionen am meisten Zeit fressen)
  4. FAIL wenn: GPU-Util < 70% avg ODER Hotspot ist eine Python-Funktion
     die nicht zur Generate-Phase gehört (ddgs/HTTP/JSON-write/regex)

RED-Phase: Profilen, Bottleneck identifizieren.
GREEN-Phase: Bottleneck gefixt, GPU-Util ≥ 70% avg.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import threading
import json
import signal
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

_HERE = Path(__file__).resolve().parent
_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
sys.path.insert(0, str(_HERE))


# ═══════════════════════════════════════════════════════════════════════════════
# Konfiguration
# ═══════════════════════════════════════════════════════════════════════════════
PYTHON = f"{_VENV}/python"
HARNESS = str(_HERE / "rigor_harness_v3.py")
GPU_UTIL_AVG_MIN = 70.0  # User's strikte Forderung
SAMPLE_INTERVAL_MS = 100
PROFILE_DUMP_INTERVAL = 1.0  # py-spy dump alle 1s
RUN_DURATION_S = 20  # Smoketest-Run für 20s


def _start_subprocess() -> subprocess.Popen:
    """Startet rigor_harness_v3 als Subprozess mit kleinem Skopus."""
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        _VENV,
        str(_HERE),
        str(_HERE.parent.parent),
        str(_HERE.parent / "rigor_zone_v1"),
        str(_HERE.parent / "rigor_zone_v2"),
    ])
    # KLEINER Skopus: 1 Task pro Kategorie, nur baseline + rigor_low
    cmd = [
        PYTHON, HARNESS,
        "--smoke", "--max-per-category", "1",
        "--batch-size", "8",
        "--max-new-tokens", "50",  # kurz für schnelles Sampling
    ]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )


def _poll_gpu(stop_event: threading.Event) -> List[int]:
    """Pollt nvidia-smi alle SAMPLE_INTERVAL_MS."""
    samples = []
    cmd = [
        "nvidia-smi", "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits", "-lms", str(SAMPLE_INTERVAL_MS),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.isdigit():
                samples.append(int(line))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    return samples


def _profile_dump(pid: int) -> str:
    """py-spy dump für laufenden Prozess."""
    try:
        out = subprocess.run(
            ["py-spy", "dump", "--pid", str(pid)],
            capture_output=True, text=True, timeout=3,
        )
        return out.stdout
    except Exception:
        return ""


def _aggregate_hotspots(dumps: List[str]) -> Dict[str, int]:
    """Aggregiert py-spy dumps: zählt wie oft jede Python-Funktion als Top-Frame auftaucht."""
    counter = Counter()
    for dump in dumps:
        lines = dump.split("\n")
        # Top-Frame ist die erste nicht-leere Zeile (höchster %-Anteil)
        for line in lines:
            line = line.strip()
            if not line or line.startswith("Process") or line.startswith("Threads"):
                continue
            # Format: "FuncName (file.py:123)"
            m = re.match(r"^([\w\.<>\[\]'\"]+)", line)
            if m:
                counter[m.group(1)] += 1
                break  # nur Top-Frame pro dump
    return dict(counter)


def main() -> int:
    print("=" * 70)
    print("RIGOR v3 — LIVE-PROFILING (echter Subprozess + py-spy + nvidia-smi)")
    print("=" * 70)
    print()

    # 1) Subprozess starten
    print(f"[1] Starte rigor_harness_v3 (smoke, 20s window)...")
    proc = _start_subprocess()
    print(f"    PID={proc.pid}")
    time.sleep(2)  # Modell-Loading abwarten

    # 2) GPU + py-spy parallel samplen
    print(f"[2] Sampling: nvidia-smi ({SAMPLE_INTERVAL_MS}ms) + py-spy (1s)...")
    stop = threading.Event()
    gpu_samples_box: List[List[int]] = [[]]

    def gpu_poller():
        gpu_samples_box[0] = _poll_gpu(stop)

    poller = threading.Thread(target=gpu_poller, daemon=True)
    poller.start()

    dumps = []
    t_start = time.time()
    while time.time() - t_start < RUN_DURATION_S:
        dumps.append(_profile_dump(proc.pid))
        time.sleep(PROFILE_DUMP_INTERVAL)
    stop.set()
    poller.join(timeout=3)

    # 3) Subprozess beenden (oder weiterlaufen lassen für vollständigen Run)
    try:
        proc.terminate()
        out, err = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()

    # 4) Auswertung
    gpu_samples = gpu_samples_box[0]
    if not gpu_samples:
        print("  [WARN] keine GPU-Samples — GPU idle den ganzen Test?")
    else:
        avg = sum(gpu_samples) / len(gpu_samples)
        mx = max(gpu_samples)
        n_active = sum(1 for s in gpu_samples if s >= 70)
        print(f"  GPU-Util: {len(gpu_samples)} samples, avg={avg:.1f}%, max={mx}%, "
              f"active(≥70%)={n_active}/{len(gpu_samples)}={100*n_active/len(gpu_samples):.0f}%")

    hotspots = _aggregate_hotspots(dumps)
    print(f"\n  Top-10 Hotspots ({len(dumps)} dumps):")
    for name, count in sorted(hotspots.items(), key=lambda x: -x[1])[:10]:
        print(f"    {count:3d}× {name}")

    # 5) Verdict
    print()
    print("=" * 70)
    print("Verdict:")
    print("=" * 70)
    errors = []
    if gpu_samples:
        avg = sum(gpu_samples) / len(gpu_samples)
        if avg < GPU_UTIL_AVG_MIN:
            errors.append(
                f"GPU-Util avg {avg:.1f}% < {GPU_UTIL_AVG_MIN}% → Bottleneck NICHT behoben!"
            )
    # Bekannte CPU-Bottleneck-Signaturen
    bad_signatures = [
        "ddgs", "DDGS", "web_search", "urllib", "requests",
        "json.dumps", "json.load", "Path.write_text",
        "re.search", "re.match", "Counter", "compute_self_report",
    ]
    bad_hits = {sig: count for sig, count in hotspots.items() if any(b in sig for b in bad_signatures)}
    if bad_hits:
        errors.append(
            f"CPU-Bottleneck-Hotspots gefunden: {bad_hits}. "
            "Diese Funktionen dürfen NICHT im GPU-Loop laufen!"
        )

    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    print(f"  ✓ GPU-Util OK, keine CPU-Bottleneck-Hotspots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
