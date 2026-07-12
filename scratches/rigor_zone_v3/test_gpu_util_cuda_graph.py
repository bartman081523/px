"""Live-Test: CUDA-Graph Runner auf 270m → 100% GPU-Util?
Misst GPU-Util bei 50 Decode-Steps mit bs=8.
"""
import os
os.environ["PYTHONPATH"] = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
import sys
import time
import subprocess
import threading
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from px_patches_v3.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig

MODEL = "google/gemma-3-270m-it"
device = "cuda"
print(f"Lade {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

# GPU-Poll
gpu_samples = []
stop = threading.Event()
def poll():
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
           "--format=csv,noheader,nounits", "-lms", "50"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    while not stop.is_set():
        line = p.stdout.readline().strip()
        if line.isdigit():
            gpu_samples.append((time.time(), int(line)))
    p.terminate()
t = threading.Thread(target=poll, daemon=True)
t.start()

# Setup
prompts = [f"What is 2+{i}?" for i in range(8)]
tpl = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
enc = tok(tpl, padding="longest", return_tensors="pt", add_special_tokens=False)
ids = enc["input_ids"].to("cuda")
am = enc["attention_mask"].to("cuda")

cfg = CUDAGraphRunnerConfig(batch_size=8, max_seq_len=64)
runner = CUDAGraphRunner(m, cfg)
first = runner.setup(ids, am)
print(f"Setup done, max_seq={cfg.max_seq_len}")

# Warmup
for _ in range(10):
    nxt = runner.step()
    runner.append(nxt)
time.sleep(0.5)

# Measure: 100 Replays
t0 = time.time()
n_steps = 100
for _ in range(n_steps):
    nxt = runner.step()
    runner.append(nxt)
torch.cuda.synchronize()
dt = time.time() - t0
time.sleep(0.5)
stop.set()
t.join(timeout=2)

# Filter GPU samples to decode phase
gen_samples = [s for ts, s in gpu_samples if t0 <= ts <= t0 + dt]
if not gen_samples:
    # Use the whole range
    gen_samples = [s for _, s in gpu_samples]

avg = sum(gen_samples) / len(gen_samples)
mx = max(gen_samples)
over_70 = sum(1 for s in gen_samples if s >= 70)
over_90 = sum(1 for s in gen_samples if s >= 90)

print(f"\n=== RESULTS ===")
print(f"  Steps: {n_steps}, Time: {dt:.2f}s, {n_steps/dt:.1f} steps/s, {n_steps*8/dt:.0f} tok/s")
print(f"  GPU samples: {len(gen_samples)}")
print(f"  avg={avg:.1f}%, max={mx}%, over 70%={over_70}/{len(gen_samples)} ({100*over_70/max(len(gen_samples),1):.0f}%)")
print(f"  over 90%={over_90}/{len(gen_samples)} ({100*over_90/max(len(gen_samples),1):.0f}%)")
