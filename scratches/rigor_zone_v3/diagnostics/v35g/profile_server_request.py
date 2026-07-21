"""profile_server_request.py — Profiliere was im Server langsam ist.

Misst: GPU-Util + Wall-Time während eines einzelnen Generate-Calls.
"""
import sys, os, time, subprocess, json
sys.path.insert(0, "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch

def gpu_sample(duration, interval=0.05):
    samples = []
    end = time.time() + duration
    while time.time() < end:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=2,
            )
            parts = out.strip().split(", ")
            samples.append({"util": float(parts[0]), "mem": float(parts[1]), "pwr": float(parts[2])})
        except Exception:
            pass
        time.sleep(interval)
    return samples

def summarize(samples, label):
    if not samples:
        return
    utils = [s["util"] for s in samples]
    mems = [s["mem"] for s in samples]
    pwrs = [s["pwr"] for s in samples]
    print(f"  [{label}] n={len(samples)} util: mean={sum(utils)/len(utils):.1f}% max={max(utils):.1f}% p95={sorted(utils)[int(len(utils)*0.95)]:.1f}%")
    print(f"  [{label}] mem: mean={sum(mems)/len(mems):.0f}MB max={max(mems):.0f}MB")
    print(f"  [{label}] pwr: mean={sum(pwrs)/len(pwrs):.1f}W max={max(pwrs):.1f}W")

print("=" * 70)
print("Profile: einzelner Generate-Call mit LEAN-Patch")
print("=" * 70)
print("Loading model...")
tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m-it", dtype=torch.bfloat16
).to("cuda").eval()
apply_px_patch(model.model, config_preset="ACTIVE_MANIFOLD_LEAN")
print("Model loaded.")

ids = tok("What is 2+3? Answer with just the number.", return_tensors="pt").to("cuda")
n = 30

# Idle GPU sample
print("\n[IDLE]")
samples_idle = gpu_sample(2.0, interval=0.1)
summarize(samples_idle, "IDLE")

# Generate with GPU sampling
print("\n[GENERATE n=30, do_sample=False]")
samples_gen = []
end = time.time() + 5.0
import threading
def sample_thread():
    while time.time() < end:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True, timeout=2,
            )
            samples_gen.append(float(out.strip()))
        except Exception:
            pass
        time.sleep(0.02)
t = threading.Thread(target=sample_thread, daemon=True)
t.start()
torch.cuda.synchronize()
t0 = time.time()
with torch.inference_mode():
    out = model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
gen_dur = time.time() - t0
time.sleep(0.5)  # let sampler finish
summarize([{"util": u, "mem": 0, "pwr": 0} for u in samples_gen], "GENERATE")
print(f"\n  Wall-time: {gen_dur:.3f}s for {n} tokens ({gen_dur/n*1000:.0f}ms/token)")

# Same without PX-patch (baseline)
print("\n[NO PX-PATCH, just generate n=30]")
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m-it", dtype=torch.bfloat16
).to("cuda").eval()
print("Fresh model loaded (no patch).")
samples_gen2 = []
end = time.time() + 5.0
t = threading.Thread(target=sample_thread, daemon=True)
t.start()
torch.cuda.synchronize()
t0 = time.time()
with torch.inference_mode():
    out2 = model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
gen2_dur = time.time() - t0
time.sleep(0.5)
summarize([{"util": u, "mem": 0, "pwr": 0} for u in samples_gen2], "NO-PATCH")
print(f"\n  Wall-time: {gen2_dur:.3f}s for {n} tokens ({gen2_dur/n*1000:.0f}ms/token)")
