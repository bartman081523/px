"""Diagnose v14: Match-Übersicht, gleiche Tasks vergleichen."""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import torch, json
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
            if n_repeats >= 3: return f"LOOPY"
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

# Speichere alle Outputs
all_outputs = {}
for preset, label in [("BASELINE", "BASELINE"), ("ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD"), ("ACTIVE_MANIFOLD_LEAN", "LEAN")]:
    print(f"\n=== {label} ===", flush=True)
    all_outputs[label] = {}
    tok, model = load_model(preset)
    for tid, cat, prompt, gt in TASKS:
        text = generate(tok, model, prompt)
        all_outputs[label][tid] = {
            "text": text, "cat": cat, "gt": gt,
            "status": analyze(text),
            "match": check_match(text, gt),
        }
    del model; torch.cuda.empty_cache()

# Per-Task-Vergleich
print("\n" + "=" * 80)
print(f"{'TID':12s} {'CAT':20s} {'GT':6s} {'B':3s} {'AM':3s} {'LN':3s}")
print("=" * 80)
b_m, am_m, ln_m, all_three = 0, 0, 0, 0
for tid, cat, prompt, gt in TASKS:
    b_match = all_outputs["BASELINE"][tid]["match"]
    am_match = all_outputs["ACTIVE_MANIFOLD"][tid]["match"]
    ln_match = all_outputs["LEAN"][tid]["match"]
    if b_match: b_m += 1
    if am_match: am_m += 1
    if ln_match: ln_m += 1
    if b_match and am_match and ln_match: all_three += 1
    print(f"  {tid[:10]:12s} {cat[:20]:20s} {gt[:6]:6s} {'✓' if b_match else '✗':3s} {'✓' if am_match else '✗':3s} {'✓' if ln_match else '✗':3s}")
print(f"\nBASELINE matches: {b_m}/{len(TASKS)}")
print(f"ACTIVE_MANIFOLD matches: {am_m}/{len(TASKS)}")
print(f"LEAN matches: {ln_m}/{len(TASKS)}")
print(f"Tasks wo ALLE 3 ✓: {all_three}/{len(TASKS)}")

# Outputs speichern
with open("/tmp/diagnose_px14_outputs.json", "w") as f:
    json.dump(all_outputs, f, indent=2)
print("\nGespeichert: /tmp/diagnose_px14_outputs.json")
