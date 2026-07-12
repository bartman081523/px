"""Diagnose v8: ACTIVE_MANIFOLD über mehrere Tasks."""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch

# HLE Tasks (echte)
TASKS = [
    ("Math", "Let $G_2$ be the classifying space of the Lie group $G_2$. What is the reduced 12-th dimensional Spin bordism of $G_2$? Please answer with a single mathematical expression.", "Z+Z+Z+Z+Z"),
    ("Humanities", "In the recent Arrhenius paper, which of the conditions of Arrhenius's sixth impossibility theorem is violated by standard welfare economics? Answer with one letter (A-D).", "D"),
    ("Computer", "You see a ciphertext: 'KHOOR ZRUOG'. This is a Caesar cipher. What is the plaintext? Just answer with the plaintext text.", "HELLO WORLD"),
    ("Other", "A phrase in an unknown language: 'Yeyo'. What language might this be? Just name one language.", "yeyo"),
]

MAX_NEW = 80

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

print("=" * 70)
print("MULTI-TASK DIAGNOSE — 4 HLE Tasks")
print("=" * 70)

for preset, label in [("BASELINE", "BASELINE"), ("ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD"), ("ACTIVE_MANIFOLD_LEAN", "LEAN")]:
    print(f"\n--- {label} ---")
    tok, model = load_model(preset)
    for cat, prompt, gt in TASKS:
        text = generate(tok, model, prompt)
        status = analyze(text)
        match = "✓" if gt.lower() in text.lower() else "✗"
        print(f"  [{cat:11s}] [{status:6s}] {match} | {text[:80]!r}")
    del model; torch.cuda.empty_cache()
