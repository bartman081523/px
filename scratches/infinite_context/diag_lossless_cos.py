"""Losslessness check for the REAL px_lossless mem-eff patch:
 compare logits[0,-1] of STOCK forward vs PATCHED (mem-eff) forward at a T where
 the chunked path activates (T>4096) AND stock still fits. cos must be ~0.999.
"""
import sys, os, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transformers import AutoModelForCausalLM, AutoTokenizer
from px_lossless import patch as pxpatch

MODEL = "google/gemma-3-1b-it"; T = 5000
tok = AutoTokenizer.from_pretrained(MODEL)
ids = torch.randint(10, tok.vocab_size, (1, T), device="cuda")

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
with torch.no_grad(): stock = model(ids).logits[0, -1].float()
del model; torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
pxpatch.apply_mem_eff_attention_patch(model.model)
# confirm chunked path will be taken
print("threshold:", pxpatch.MEM_EFF_THRESHOLD, "T:", T, "-> chunked:", T > pxpatch.MEM_EFF_THRESHOLD)
with torch.no_grad(): chunk = model(ids).logits[0, -1].float()

cos = (stock @ chunk / (stock.norm()*chunk.norm())).item()
print("cos(stock, mem_eff_chunked):", round(cos, 6))
print("argmax stock:", stock.argmax().item(), "argmax chunk:", chunk.argmax().item())
print("max_abs_diff:", round((stock-chunk).abs().max().item(), 4))