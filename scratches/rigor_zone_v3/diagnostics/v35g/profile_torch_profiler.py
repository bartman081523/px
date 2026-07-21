"""profile_torch_profiler.py — Detailliertes Profiling mit torch.profiler."""
import sys, os
sys.path.insert(0, "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch

print("Loading model with LEAN-Patch...")
tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m-it", dtype=torch.bfloat16
).to("cuda").eval()
apply_px_patch(model.model, config_preset="ACTIVE_MANIFOLD_LEAN")

ids = tok("What is 2+3? Answer with just the number.", return_tensors="pt").to("cuda")

# Warmup
with torch.inference_mode():
    _ = model.generate(**ids, max_new_tokens=2, do_sample=False, pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()

# Profiling
print("Profiling 30 token generate...")
from torch.profiler import profile, ProfilerActivity, record_function
with torch.inference_mode():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        with record_function("model_generate"):
            out = model.generate(**ids, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()

# Print top time consumers
print("\n=== TOP 20 CPU-OPS (by self_cuda_time_total) ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

# CPU time (Python overhead)
print("\n=== TOP 20 CPU-OPS (by self_cpu_time_total) ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

# Save detailed
prof.export_chrome_trace("/tmp/v3_trace.json")
print("\nTrace saved: /tmp/v3_trace.json")
