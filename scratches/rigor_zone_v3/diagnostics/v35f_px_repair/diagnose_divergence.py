"""Diagnose: Wo genau divergiert ACTIVE_MANIFOLD von BASELINE?"""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch
from rigor_hle_suite_v3 import load_hle_suite

source, suite = load_hle_suite(max_per_category=4)
TASKS = [(tid, cat, prompt, gt) for tid, cat, prompt, gt in suite]
target_tid = "66b2c7c979"
prompt = next(p for tid, c, p, g in TASKS if tid.startswith(target_tid))
print(f"Prompt: {prompt[:100]}")

MAX_NEW = 50

def load_model(preset, **kwargs):
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", dtype=torch.bfloat16).to("cuda").eval()
    if preset != "BASELINE":
        apply_px_patch(model.model, config_preset=preset, **kwargs)
    return tok, model

def gen_tokens(tok, model, prompt, max_new=MAX_NEW):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    new_ids = out[0, ids["input_ids"].shape[1]:].tolist()
    return new_ids

# Run both
print("\n=== BASELINE ===")
tok, model = load_model("BASELINE")
b_ids = gen_tokens(tok, model, prompt)
print(f"Tokens: {b_ids[:30]}")
b_text = tok.decode(b_ids, skip_special_tokens=True)
print(f"Text: {b_text[:200]}")
del model; torch.cuda.empty_cache()

print("\n=== ACTIVE_MANIFOLD ===")
tok, model = load_model("ACTIVE_MANIFOLD")
am_ids = gen_tokens(tok, model, prompt)
print(f"Tokens: {am_ids[:30]}")
am_text = tok.decode(am_ids, skip_special_tokens=True)
print(f"Text: {am_text[:200]}")
del model; torch.cuda.empty_cache()

# Find divergence
print("\n=== DIVERGENZ-ANALYSE ===")
i = 0
while i < min(len(b_ids), len(am_ids)) and b_ids[i] == am_ids[i]:
    i += 1
print(f"Identische Tokens: {i}")
print(f"BASELINE Token {i}: {b_ids[i]} = {tok.decode([b_ids[i]])!r}")
print(f"ACTIVE  Token {i}: {am_ids[i]} = {tok.decode([am_ids[i]])!r}")
print(f"BASELINE  Folge: {tok.decode(b_ids[i:i+10])!r}")
print(f"ACTIVE    Folge: {tok.decode(am_ids[i:i+10])!r}")
