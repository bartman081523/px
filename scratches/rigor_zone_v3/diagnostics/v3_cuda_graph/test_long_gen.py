"""Live-Test: GPU-Util mit 200 tokens, bs=8 → längere Generate-Phase."""
import os, sys, queue, subprocess, threading, time
from pathlib import Path

_HERE = Path("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
env = os.environ.copy()
env["PYTHONPATH"] = ":".join([_VENV, str(_HERE), str(_HERE.parent)])
cmd = [f"{_VENV}/python", "-u", str(_HERE / "rigor_harness_v3.py"),
       "--smoke", "--max-per-category", "4", "--batch-size", "8",
       "--max-new-tokens", "200", "--no-write"]
print(f"cmd: {' '.join(cmd)}")
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
t_start = time.time()

# GPU polling
gpu_stop = threading.Event()
gpu_samples = []
def gpu_poller():
    p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu",
                          "--format=csv,noheader,nounits", "-lms", "50"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    while not gpu_stop.is_set():
        l = p.stdout.readline()
        if not l: break
        l = l.strip()
        if l.isdigit():
            gpu_samples.append((time.time() - t_start, int(l)))
    p.terminate()
threading.Thread(target=gpu_poller, daemon=True).start()

# Reader
line_q = queue.Queue()
def reader():
    for l in iter(proc.stdout.readline, ""):
        line_q.put((time.time() - t_start, l))
    line_q.put(None)
threading.Thread(target=reader, daemon=True).start()

gen_start = 0.0
gen_end = 0.0
in_gen = False
out = []
while True:
    try:
        item = line_q.get(timeout=1)
    except queue.Empty:
        if time.time() - t_start > 300: break
        continue
    if item is None: break
    ts, line = item
    out.append(line.rstrip())
    if not in_gen and line.startswith("[B] Arm="):
        in_gen = True; gen_start = ts
    if in_gen and "[B] GPU-Loop fertig" in line:
        gen_end = ts
gpu_stop.set()
proc.terminate()
try: proc.wait(timeout=5)
except: proc.kill()

if gen_start == 0: gen_start = 0.01
if gen_end == 0: gen_end = time.time() - t_start
gen_s = [s for ts, s in gpu_samples if gen_start <= ts <= gen_end]
if gen_s:
    avg = sum(gen_s)/len(gen_s); mx = max(gen_s)
    print(f"\nGenerate-Phase: {gen_start:.2f}s-{gen_end:.2f}s ({gen_end-gen_start:.2f}s)")
    print(f"GPU-Util: {len(gen_s)} samples, avg={avg:.1f}%, max={mx}%, "
          f"active(>=70%)={sum(1 for s in gen_s if s>=70)}/{len(gen_s)}")
    print(f"Output (last 5 lines):")
    for l in out[-5:]: print(f"  {l}")
