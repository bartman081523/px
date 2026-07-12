"""Diagnose v13: 32 HLE tasks, alle 3 presets."""
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
print(f"Loaded {len(TASKS)} HLE tasks")

MAX_NEW = 200

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

def check_match(text, gt):
    text_l = text.lower()
    gt_l = gt.lower().strip()
    if not gt_l: return False
    if len(gt_l) <= 3 and gt_l.replace(" ", "").isalpha():
        if f"**{gt_l.upper()}" in text or f"answer is {gt_l.upper()}" in text_l or f"answer: {gt_l.upper()}" in text_l:
            return True
        if f"\n{gt_l.upper()}. " in text or f"{gt_l.upper()}. " in text[:50]:
            return True
    else:
        if gt_l in text_l:
            return True
    return False

results = {}
for preset, label in [("BASELINE", "BASELINE"), ("ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD"), ("ACTIVE_MANIFOLD_LEAN", "LEAN")]:
    print(f"\n=== {label} ===", flush=True)
    results[label] = {"ok": 0, "degen": 0, "matches": 0, "tasks": []}
    tok, model = load_model(preset)
    for tid, cat, prompt, gt in TASKS:
        text = generate(tok, model, prompt)
        status = analyze(text)
        match = check_match(text, gt)
        results[label]["tasks"].append((tid, cat, match, status))
        if status == "OK": results[label]["ok"] += 1
        else: results[label]["degen"] += 1
        if match: results[label]["matches"] += 1
        print(f"  [{cat[:20]:20s}] {tid[:10]:10s} match={int(match)} degen={status}", flush=True)
    del model; torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("VERGLEICH (32 HLE tasks)")
print("=" * 70)
for label, r in results.items():
    pct_m = r['matches']*100/len(TASKS)
    pct_d = r['degen']*100/len(TASKS)
    print(f"  {label:30s}: {r['ok']:2d}/{len(TASKS)} ok, {r['matches']:2d} matches ({pct_m:5.1f}%), {r['degen']:2d} degen ({pct_d:5.1f}%)")
