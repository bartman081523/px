"""Diagnose v11: Alle 8 HLE-Kategorien (1 pro Kat)."""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch
from rigor_hle_suite_v3 import load_hle_suite

# Lade alle 8 HLE Tasks
source, suite = load_hle_suite(max_per_category=1)
print(f"Loaded {len(suite)} HLE tasks")
TASKS = [(cat, prompt, gt) for tid, cat, prompt, gt in suite]
for cat, prompt, gt in TASKS:
    print(f"  - {cat}: {prompt[:50]!r}...")

MAX_NEW = 60

def load_model(preset, **kwargs):
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", dtype=torch.bfloat16).to("cuda").eval()
    if preset != "BASELINE":
        apply_px_patch(model.model, config_preset=preset, **kwargs)
    return tok, model

def generate(tok, model, prompt, max_new=MAX_NEW):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)

def analyze(text):
    if not text.strip(): return "EMPTY"
    words = text.split()
    for plen in [3, 4]:
        for i in range(len(words) - plen*3):
            phrase = " ".join(words[i:i+plen])
            n_repeats = sum(1 for j in range(i+plen, len(words)-plen+1, plen) if " ".join(words[j:j+plen]) == phrase)
            if n_repeats >= 3: return f"LOOPY_p{plen}"
    if "0, 0, 0" in text or "\\_\\" in text: return "NUMERIC"
    return "OK"

results = {}
for preset, label in [("BASELINE", "BASELINE"), ("ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD"), ("ACTIVE_MANIFOLD_LEAN", "LEAN")]:
    print(f"\n=== {label} ===")
    results[label] = {"degen": 0, "ok": 0}
    tok, model = load_model(preset)
    for cat, prompt, gt in TASKS:
        text = generate(tok, model, prompt)
        status = analyze(text)
        match = "✓" if gt.lower() in text.lower() else "✗"
        if status == "OK": results[label]["ok"] += 1
        else: results[label]["degen"] += 1
        print(f"  [{cat:30s}] [{status:6s}] {match} | {text[:60]!r}")
    del model; torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("RESULTATE")
print("=" * 70)
for label, r in results.items():
    print(f"  {label:30s}: {r['ok']:2d}/{len(TASKS)} ok, {r['degen']:2d} degen")
