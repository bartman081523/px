"""GPU-Util bei 270M-Generate (baseline) messen.
Mit model.generate() — keine PX-Patches.
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

MODEL = "google/gemma-3-270m-it"
device = "cuda"
print(f"Lade {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device).eval()

prompts = ["What is 2+" + str(i) + "?" for i in range(8)]
tokenizer = tok
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tpl = [tokenizer.apply_chat_template(
    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
) for p in prompts]
enc = tokenizer(tpl, padding="longest", return_tensors="pt", add_special_tokens=False)
ids = enc["input_ids"].to(device)
am = enc["attention_mask"].to(device)
print(f"  bs={ids.shape[0]}, prompt_len={ids.shape[1]}")

# GPU-Poll thread
gpu_samples = []
stop = threading.Event()
def poll():
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
           "--format=csv,noheader,nounits", "-lms", "50"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    while not stop.is_set():
        line = p.stdout.readline().strip()
        if line.isdigit():
            gpu_samples.append(int(line))
    p.terminate()
t = threading.Thread(target=poll, daemon=True)
t.start()

# Warmup
with torch.inference_mode():
    _ = m.generate(input_ids=ids, attention_mask=am, max_new_tokens=10, do_sample=False, use_cache=True)
time.sleep(0.5)

# Measure
t0 = time.perf_counter()
with torch.inference_mode():
    out = m.generate(input_ids=ids, attention_mask=am, max_new_tokens=50, do_sample=False, use_cache=True)
dt = time.perf_counter() - t0
time.sleep(0.5)
stop.set()
t.join(timeout=2)

print(f"\nbs={ids.shape[0]}, 50 tokens, {dt:.2f}s = {50/dt:.1f} tok/s, {50/(dt*ids.shape[0]):.1f} tok/s/seq")
print(f"GPU samples: {len(gpu_samples)}")
if gpu_samples:
    print(f"  avg: {sum(gpu_samples)/len(gpu_samples):.1f}%")
    print(f"  max: {max(gpu_samples)}%")
    over_70 = sum(1 for s in gpu_samples if s >= 70)
    print(f"  over 70%: {over_70}/{len(gpu_samples)} ({100*over_70/len(gpu_samples):.0f}%)")
