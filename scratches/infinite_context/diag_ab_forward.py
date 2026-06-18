"""A/B isolate the 14k degradation source:
 (A) PX-patched model, STOCK forward restored (only mem-eff attention patch active)
     -> if GOOD: degradation is in the PX custom forward (layer segmentation/positions).
 (B) PX-patched model, PX forward active (full ACTIVE_MANIFOLD) -> the degenerate case.
Both use the SAME mem-eff chunked attention, so attention is held constant.
"""
import sys, os, json, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transformers import AutoModelForCausalLM, AutoTokenizer
from px_lossless import patch as pxpatch

MODEL = "google/gemma-3-1b-it"
SESSION = os.path.join(HERE, "..", "..", "sessions", "3479b4f9.json")

def gen(model, tok, ids, tag):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache(); t0 = __import__("time").time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=48, do_sample=False,
                             repetition_penalty=1.15, no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id)
    dt = __import__("time").time() - t0
    print(f"[{tag}] peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB t={dt:.1f}s")
    print(f"[{tag}] OUT:", repr(tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)[:300]))

tok = AutoTokenizer.from_pretrained(MODEL)
hist = json.load(open(SESSION))["history"]
ids = tok(tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
print("tokens:", ids["input_ids"].shape[1])

# (A) mem-eff attention patch ONLY, stock forward
print("="*60); print("(A) mem-eff attention only, STOCK forward")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
pxpatch.apply_mem_eff_attention_patch(model.model)
gen(model, tok, ids, "A-stockfwd")
del model; torch.cuda.empty_cache()

# (B) full PX ACTIVE_MANIFOLD (mem-eff attention + PX custom forward)
print("="*60); print("(B) full PX ACTIVE_MANIFOLD")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
pxpatch.apply_px_patch(model, config_preset="ACTIVE_MANIFOLD")
gen(model, tok, ids, "B-pxfwd")