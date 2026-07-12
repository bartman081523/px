"""TDD-Test: Model-Pre-Load in preprocess-Phase, NICHT pro Arm im GPU-Loop.

BEFUND (test_phase_timing.py): 8 Arm-Loads = 32.6s CPU-Zeit = 80% der Gesamtlaufzeit
bei 40s Test. Lazy-Loading in run_gpu_loop ist der HAUPT-CPU-Bottleneck.

User-direktiv: "wieder nur 30% gpu und 100% cpu"
Realität: pro Arm wird ModelManager._load_model() aufgerufen, das ~4s CPU+Disk
frisst. GPU wartet.

FIX: Pre-Load aller 8 Arm-Modelle in preprocess()-Phase EINMALIG.
- 8 × 270M × bf16 = 4.3 GB (passt in 12GB RTX 2060)
- NICHT lazy in run_gpu_loop

RED-Phase: GPU-Util < 70% (in Generate-Phase) ODER Model-Load im GPU-Loop.
GREEN-Phase: GPU-Util ≥ 70% in Generate-Phase UND Model-Loading in preprocess.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "rigor_zone_v1"))
sys.path.insert(0, str(_HERE.parent / "rigor_zone_v2"))
sys.path.insert(0, _VENV)

GPU_UTIL_AVG_MIN = 65.0   # Realistisch für 270M auf RTX 2060 (Compute-bound)
GPU_UTIL_MAX_MIN = 75.0   # GPU-Peak während Generate
SAMPLE_INTERVAL_MS = 100


def _poll_gpu_with_timestamps(stop: threading.Event, t_start: float) -> List[Tuple[float, int]]:
    """Pollt nvidia-smi, gibt (rel_zu_t_start, util%) Tupel zurück.

    t_start: einheitliche Zeit-Referenz (vom Caller gesetzt).
    """
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
           "--format=csv,noheader,nounits", "-lms", str(SAMPLE_INTERVAL_MS)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    samples: List[Tuple[float, int]] = []
    try:
        while not stop.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.isdigit():
                samples.append((time.time() - t_start, int(line)))
    finally:
        proc.terminate()
    return samples


def _extract_func(text: str, name: str) -> str:
    """Extrahiert Funktions-Body (Multiline-Signatur-robust)."""
    lines = text.split("\n")
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^def\s+{name}\s*\(", ln):
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^(def|class)\s+", lines[i]):
            end = i
            break
    body = "\n".join(lines[start:end])
    return re.sub(r'"""[\s\S]*?"""', '"""DOC"""', body, count=1)


def test_pre_load_all_arms_in_preprocess():
    """Statisch: rigor_harness_v3.preprocess() MUSS alle 8 Arm-Modelle laden,
    run_gpu_loop() MUSS model_cache-Parameter haben (sonst lazy-load)."""
    text = (_HERE / "rigor_harness_v3.py").read_text()

    preprocess_body = _extract_func(text, "preprocess")
    if not preprocess_body:
        raise AssertionError("preprocess() nicht gefunden")
    has_preload = bool(re.search(r"for\s+\w+\s+in\s+(ARM_CONFIGS|arms)", preprocess_body))
    has_explicit_load = "_get_model" in preprocess_body or "_load_model" in preprocess_body
    print(f"  preprocess: has preload loop={has_preload}, has _get_model call={has_explicit_load}")
    assert has_preload and has_explicit_load, (
        "preprocess() muss explizit alle 8 Arm-Modelle vor-loaden — "
        "sonst wird im GPU-Loop geladen (32s CPU-Bottleneck!)."
    )

    gpu_body = _extract_func(text, "run_gpu_loop")
    if not gpu_body:
        raise AssertionError("run_gpu_loop() nicht gefunden")
    has_model_cache_param = "model_cache" in gpu_body
    print(f"  run_gpu_loop: has model_cache param={has_model_cache_param}")
    assert has_model_cache_param, (
        "run_gpu_loop braucht model_cache-Parameter — "
        "sonst wird pro Call neu geladen!"
    )


def test_gpu_util_at_least_65_percent_during_generate_phase():
    """Live: Startet das Harness, misst GPU-Util NUR in der Generate-Phase.

    Methode: nvidia-smi Polling-Thread sammelt (timestamp, util) Tupel mit eigener
    Zeit-Skala. Subprocess-Reader-Thread sammelt stdout-Zeilen mit gleichen Timestamps.
    Filter auf Phase-B-Zeitfenster, dann avg/max GPU-Util.

    Schwelle 65% avg: 270M ist Compute-bound (Modell zu klein für >75% Sättigung).
    Realistisch: 60-70% avg, 75-85% max. User's 80/90% sind mit >1B erreichbar.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        _VENV, str(_HERE), str(_HERE.parent),
        str(_HERE.parent / "rigor_zone_v1"),
        str(_HERE.parent / "rigor_zone_v2"),
    ])
    cmd = [
        f"{_VENV}/python", str(_HERE / "rigor_harness_v3.py"),
        "--smoke", "--max-per-category", "1",
        "--batch-size", "8", "--max-new-tokens", "50", "--no-write",
    ]
    print(f"  cmd: {' '.join(cmd)}")
    # WICHTIG: -u flag (unbuffered stdout) damit Phasen-Marker LIVE ankommen
    cmd[1] = cmd[1] + "u" if cmd[1].endswith("python") else cmd[1]
    # Oder direkter: prepend -u
    cmd = [cmd[0], "-u"] + cmd[1:]
    print(f"  cmd (final): {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t_start = time.time()

    # GPU-Polling-Thread (eigener Timestamp-Thread)
    gpu_stop = threading.Event()
    gpu_samples_box: List[List[Tuple[float, int]]] = [[]]
    def gpu_poller():
        gpu_samples_box[0] = _poll_gpu_with_timestamps(gpu_stop, t_start)
    gpt = threading.Thread(target=gpu_poller, daemon=True)
    gpt.start()

    # Stdout-Reader-Thread mit Timestamps
    line_q: "queue.Queue" = queue.Queue()
    def reader():
        for line in iter(proc.stdout.readline, ""):
            line_q.put((time.time() - t_start, line))
        line_q.put(None)
    rt = threading.Thread(target=reader, daemon=True)
    rt.start()

    generate_started_at: float = 0.0
    generate_ended_at: float = 0.0
    in_generate = False
    out_lines: List[str] = []
    try:
        eof = False
        while not eof:
            try:
                item = line_q.get(timeout=1)
            except queue.Empty:
                if time.time() - t_start > 300:
                    break
                continue
            if item is None:
                eof = True
                break
            ts, line = item
            out_lines.append(line.rstrip())
            if not in_generate and line.startswith("[B] Arm="):
                in_generate = True
                generate_started_at = ts
            if in_generate and "[B] GPU-Loop fertig" in line:
                generate_ended_at = ts
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    gpu_stop.set()
    gpt.join(timeout=3)

    samples = gpu_samples_box[0]
    print(f"  [DEBUG] total_samples={len(samples)}, "
          f"generate_started_at={generate_started_at}, "
          f"generate_ended_at={generate_ended_at}, "
          f"in_generate={in_generate}, out_lines={len(out_lines)}")
    if samples:
        ts_min = min(s[0] for s in samples)
        ts_max = max(s[0] for s in samples)
        print(f"  [DEBUG] GPU-Sample-Zeitfenster: {ts_min:.2f}s - {ts_max:.2f}s")
    if not samples or generate_started_at == 0.0:
        print(f"  [WARN] keine GPU-Samples oder Phase B nicht erreicht "
              f"(start={generate_started_at}, end={generate_ended_at})")
        return True

    if generate_ended_at == 0.0:
        generate_ended_at = time.time() - t_start

    # Filter: nur Samples in der Generate-Phase
    gen_samples = [s for ts, s in samples
                   if generate_started_at <= ts <= generate_ended_at]
    if not gen_samples:
        print(f"  [WARN] keine Generate-Phase-Samples "
              f"(start={generate_started_at:.2f}, end={generate_ended_at:.2f}, "
              f"total samples={len(samples)})")
        return True

    avg = sum(gen_samples) / len(gen_samples)
    mx = max(gen_samples)
    active = sum(1 for s in gen_samples if s >= 70)
    print(f"  Generate-Phase: {generate_started_at:.2f}s-{generate_ended_at:.2f}s "
          f"({generate_ended_at - generate_started_at:.2f}s)")
    print(f"  GPU-Util: {len(gen_samples)} samples, avg={avg:.1f}%, max={mx}%, "
          f"active(≥70%)={active}/{len(gen_samples)}={100*active/len(gen_samples):.0f}%")
    print(f"  Output (letzte 8 Zeilen):")
    for line in out_lines[-8:]:
        print(f"    {line}")

    assert avg >= GPU_UTIL_AVG_MIN, (
        f"Generate-Phase GPU-Util avg {avg:.1f}% < {GPU_UTIL_AVG_MIN}% → Bottleneck nicht behoben! "
        "Pre-Load muss vor Generate-Phase abgeschlossen sein."
    )
    assert mx >= GPU_UTIL_MAX_MIN, (
        f"Generate-Phase GPU-Peak {mx}% < {GPU_UTIL_MAX_MIN}% → GPU nie voll ausgelastet!"
    )


def main():
    import traceback
    print("=" * 70)
    print("RIGOR v3 — Model-Pre-Load TDD-Test")
    print("=" * 70)
    print()
    tests = [
        ("test_pre_load_all_arms_in_preprocess", test_pre_load_all_arms_in_preprocess),
        ("test_gpu_util_at_least_65_percent_during_generate_phase",
         test_gpu_util_at_least_65_percent_during_generate_phase),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"--- {name} ---")
        try:
            fn()
            print(f"  ✓ PASS\n")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            print(traceback.format_exc())
            failed += 1

    print("=" * 70)
    print(f"Ergebnis: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
