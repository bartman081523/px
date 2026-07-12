"""Batched generate speed: bs=8, verschiedene PX-Modi."""
import os, sys, time, importlib
sys.path.insert(0, "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
os.chdir("/run/media/julian/ML4/ollama-work/all_space_6_16_stand")

_v3_dir = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/px_patches_v3"
if _v3_dir not in sys.path: sys.path.insert(0, _v3_dir)
v3 = importlib.import_module("px_patches_v3.patch")
fake_pkg = type(sys)("gemma3_270m_px_baseline")
fake_pkg.__path__ = []
sys.modules["gemma3_270m_px_baseline"] = fake_pkg
sys.modules["gemma3_270m_px_baseline.patch"] = v3

from model_manager import ModelManager
import torch

mm = ModelManager()
print("Loading baseline...")
t0 = time.time()
entry = mm._load_model("gemma3-270m-it", px_subjective=False, px_config_preset="BASELINE")
print(f"  baseline loaded in {time.time()-t0:.1f}s")
model_b = entry["model"]; tok = entry["tokenizer"]

sys.modules["gemma3_270m_px_baseline.patch"] = v3
print("Loading PX...")
t0 = time.time()
entry2 = mm._load_model("gemma3-270m-it", px_subjective=True, px_config_preset="ACTIVE_MANIFOLD")
print(f"  PX loaded in {time.time()-t0:.1f}s")
model_p = entry2["model"]

device = next(model_b.parameters()).device

# Prepare 8 prompts
prompts = [f"What is {i}+{i+1}?" for i in range(8)]
def batched_gen(model, max_new=50):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    inp = tok(texts, return_tensors="pt", padding="longest").to(device)
    t0 = time.time()
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, 
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    n_gen = (out.shape[1] - inp["input_ids"].shape[1]) * out.shape[0]
    return elapsed, n_gen, out.shape

# Warmup
batched_gen(model_b, 5)
batched_gen(model_p, 5)

# Time BASELINE
print("\n--- BASELINE bs=8 ---")
t, n, shape = batched_gen(model_b, 50)
print(f"  8 prompts, 50 max_new → {n} tok in {t:.2f}s = {n/t:.1f} tok/s")
print(f"  shape={shape}")

# Time PX
print("\n--- PX bs=8 ---")
t, n, shape = batched_gen(model_p, 50)
print(f"  8 prompts, 50 max_new → {n} tok in {t:.2f}s = {n/t:.1f} tok/s")
print(f"  shape={shape}")
