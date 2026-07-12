"""Vergleiche Baseline vs PX-Generate Speed (ms/Token, tok/s)."""
import os, sys, time, importlib
from pathlib import Path
sys.path.insert(0, "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
os.chdir("/run/media/julian/ML4/ollama-work/all_space_6_16_stand")

# Setup v3 patch module alias FIRST
_v3_dir = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/px_patches_v3"
if _v3_dir not in sys.path: sys.path.insert(0, _v3_dir)
v3 = importlib.import_module("px_patches_v3.patch")
fake_pkg = type(sys)("gemma3_270m_px_baseline")
fake_pkg.__path__ = []  # make it a package
sys.modules["gemma3_270m_px_baseline"] = fake_pkg
sys.modules["gemma3_270m_px_baseline.patch"] = v3

from model_manager import ModelManager, _migrate_preset
import torch
from transformers import AutoTokenizer

mm = ModelManager()
print("Loading baseline...")
t0 = time.time()
entry = mm._load_model("gemma3-270m-it", px_subjective=False, px_config_preset="BASELINE")
print(f"  baseline loaded in {time.time()-t0:.1f}s")
model_b = entry["model"]; tok_b = entry["tokenizer"]

# IMPORTANT: re-apply sys.modules patch after second load
sys.modules["gemma3_270m_px_baseline.patch"] = v3

print("Loading PX...")
t0 = time.time()
entry2 = mm._load_model("gemma3-270m-it", px_subjective=True, px_config_preset="ACTIVE_MANIFOLD")
print(f"  PX loaded in {time.time()-t0:.1f}s")
model_p = entry2["model"]; tok_p = entry2["tokenizer"]

device = next(model_b.parameters()).device
torch.cuda.synchronize()

def gen(model, tok, prompt, max_new=50):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(device)
    t0 = time.time()
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, 
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    torch.cuda.synchronize()
    return time.time() - t0, out.shape[1] - inp["input_ids"].shape[1]

# Warmup
gen(model_b, tok_b, "Hello", 5)
gen(model_p, tok_p, "Hello", 5)

prompts = ["What is 2+2?", "What is the capital of France?", "Tell me a story about a cat."]
print("\n--- BASELINE ---")
total_t_b, total_n = 0, 0
for p in prompts:
    t, n = gen(model_b, tok_b, p, 50)
    total_t_b += t; total_n += n
    print(f"  '{p[:30]}' → {n} tok in {t:.2f}s ({n/t:.1f} tok/s)")
print(f"  TOTAL: {total_n} tok in {total_t_b:.2f}s = {total_n/total_t_b:.1f} tok/s")

print("\n--- PX (active_manifold) ---")
total_t_p, total_n = 0, 0
for p in prompts:
    t, n = gen(model_p, tok_p, p, 50)
    total_t_p += t; total_n += n
    print(f"  '{p[:30]}' → {n} tok in {t:.2f}s ({n/t:.1f} tok/s)")
print(f"  TOTAL: {total_n} tok in {total_t_p:.2f}s = {total_n/total_t_p:.1f} tok/s")

print(f"\nPX-Speedup: baseline/PX = {total_t_p/total_t_b:.1f}x (höher = PX langsamer)")
