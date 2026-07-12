"""Quick-Test: Funktioniert torch.compile() auf gemma3-270m?
- mode="reduce-overhead" aktiviert CUDA-Graph
- mode="default" ist nur Inductor-Compile
"""
import os
os.environ["PYTHONPATH"] = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
import sys
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-3-270m-it"
device = "cuda"
print(f"Lade {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device).eval()

prompt = "What is 2+2?"
ids = tok(prompt, return_tensors="pt").input_ids.to(device)
print(f"  prompt len: {ids.shape[1]}")

# Warmup
with torch.inference_mode():
    out = m.generate(input_ids=ids, max_new_tokens=20, do_sample=False, use_cache=True)
print(f"  warmup output len: {out.shape[1]}")

# 1) Baseline (no compile)
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(3):
        out = m.generate(input_ids=ids, max_new_tokens=50, do_sample=False, use_cache=True)
torch.cuda.synchronize()
dt_base = (time.perf_counter() - t0) / 3
print(f"\nBaseline (no compile): {dt_base*1000:.0f}ms / 50 tokens = {50/dt_base:.1f} tok/s")

# 2) torch.compile - default
print("\nCompile mode=default...")
m_c = torch.compile(m, mode="default", fullgraph=False)
torch.cuda.synchronize()
t0 = time.perf_counter()
try:
    with torch.inference_mode():
        for _ in range(3):
            out = m_c.generate(input_ids=ids, max_new_tokens=50, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    dt_def = (time.perf_counter() - t0) / 3
    print(f"  mode=default: {dt_def*1000:.0f}ms / 50 tokens = {50/dt_def:.1f} tok/s")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
    dt_def = None

# 3) torch.compile - reduce-overhead (CUDA-Graph)
print("\nCompile mode=reduce-overhead (CUDA-Graph)...")
m_c2 = torch.compile(m, mode="reduce-overhead", fullgraph=False)
torch.cuda.synchronize()
t0 = time.perf_counter()
try:
    with torch.inference_mode():
        for _ in range(3):
            out = m_c2.generate(input_ids=ids, max_new_tokens=50, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    dt_ro = (time.perf_counter() - t0) / 3
    print(f"  mode=reduce-overhead: {dt_ro*1000:.0f}ms / 50 tokens = {50/dt_ro:.1f} tok/s")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
    dt_ro = None
