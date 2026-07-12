"""GPU-Util mit v3-Harness (CUDA-Graph als Default)."""
import os
os.environ["PYTHONPATH"] = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
import sys
import time
import subprocess
import threading
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")

# Subprocess: v3-harness mit --smoke
env = os.environ.copy()
env["PYTHONPATH"] = ":".join([
    "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin",
    "/run/media/julian/ML4/ollama-work/all_space_6_16_stand",
    "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1",
    "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v2",
])
V3 = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3"

# GPU-Poll-Thread
gpu_samples = []
stop = threading.Event()
def poll():
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
           "--format=csv,noheader,nounits", "-lms", "30"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    while not stop.is_set():
        line = p.stdout.readline().strip()
        if line.isdigit():
            gpu_samples.append((time.time(), int(line)))
    p.terminate()
t = threading.Thread(target=poll, daemon=True)
t.start()

cmd = ["/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python", "-u",
       f"{V3}/rigor_harness_v3.py",
       "--smoke", "--max-per-category", "2",
       "--batch-size", "8", "--max-new-tokens", "50", "--no-write"]
t0 = time.time()
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
gen_started = None
gen_ended = None
in_gen = False
for line in proc.stdout:
    line = line.rstrip()
    if "[B] Arm=" in line and gen_started is None:
        gen_started = time.time()
        in_gen = True
    if "[B] GPU-Loop fertig" in line:
        gen_ended = time.time()
proc.wait()
time.sleep(0.5)
stop.set()
t.join(timeout=2)

# Filter zur Generate-Phase
gen_samples = [s for ts, s in gpu_samples
               if gen_started and gen_ended and gen_started <= ts <= gen_ended]
if not gen_samples:
    gen_samples = [s for _, s in gpu_samples]
avg = sum(gen_samples) / len(gen_samples)
mx = max(gen_samples)
over_70 = sum(1 for s in gen_samples if s >= 70) / len(gen_samples) * 100
over_90 = sum(1 for s in gen_samples if s >= 90) / len(gen_samples) * 100
print(f"Generate-Phase: {gen_ended-gen_started:.2f}s")
print(f"GPU samples: {len(gen_samples)}, avg={avg:.1f}%, max={mx}%, over 70%: {over_70:.0f}%, over 90%: {over_90:.0f}%")
