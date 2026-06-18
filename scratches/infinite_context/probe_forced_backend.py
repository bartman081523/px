"""Probe: force each SDPA backend at 16k on base gemma3-1b-it. Find a lossless, fast, no-OOM backend.
Runs in foreground (no timeout). Tests FLASH, MEM_EFF, EFFICIENT_ATTENTION forced.
"""
import torch, time, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

BACKENDS = {
    "FLASH": [SDPBackend.FLASH_ATTENTION],
    "EFFICIENT": [SDPBackend.EFFICIENT_ATTENTION],
    "CUDNN": [SDPBackend.CUDNN_ATTENTION],
}
# EFFICIENT_ATTENTION == mem_efficient in modern torch

def run(T, backend):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    ids = torch.randint(10, tok.vocab_size, (1, T), device="cuda")
    try:
        t0 = time.time()
        with torch.no_grad(), sdpa_kernel(backend):
            out = model.generate(ids, max_new_tokens=4, do_sample=False)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        return f"OK peak={peak:.2f}GB t={dt:.1f}s"
    except Exception as e:
        torch.cuda.empty_cache()
        return f"ERR {type(e).__name__}: {str(e)[:90]}"

for name, bk in BACKENDS.items():
    for T in (2000, 16000):
        print(f"[{name}] T={T}: {run(T, bk)}")