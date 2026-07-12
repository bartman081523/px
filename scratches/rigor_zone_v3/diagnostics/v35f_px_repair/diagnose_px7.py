"""Diagnose v7: Verifizieren dass baseline+PX(no recursion) == baseline."""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch

TASK_PROMPT = """Let $G_2$ be the classifying space of the Lie group $G_2$. What is the reduced 12-th dimensional Spin bordism of $G_2$? Please answer with a single mathematical expression."""
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

# A: BASELINE
print("--- A_BASELINE ---")
tok, model = load_model("BASELINE")
text_a = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text_a):6s}] {text_a[:120]!r}")
del model; torch.cuda.empty_cache()

# B: ACTIVE_MANIFOLD mit recur_start=18, recur_end=18 (n_loops via Calibrator)
print("--- B_ACTIVE_MANIFOLD recur_start=18, recur_end=18 ---")
tok, model = load_model("ACTIVE_MANIFOLD", recur_start=18, recur_end=18)
text_b = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text_b):6s}] {text_b[:120]!r}")
print(f"  EQUAL TO BASELINE: {text_a == text_b}")
del model; torch.cuda.empty_cache()

# C: ACTIVE_MANIFOLD mit recur_start=0, recur_end=18
print("--- C_ACTIVE_MANIFOLD recur_start=0, recur_end=18 ---")
tok, model = load_model("ACTIVE_MANIFOLD", recur_start=0, recur_end=18)
text_c = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text_c):6s}] {text_c[:120]!r}")
del model; torch.cuda.empty_cache()

# D: ACTIVE_MANIFOLD mit recur_start=0, recur_end=0 (REASONING leer, PRELUDE 0, CODA alle)
print("--- D_ACTIVE_MANIFOLD recur_start=0, recur_end=0 (CODA alle) ---")
tok, model = load_model("ACTIVE_MANIFOLD", recur_start=0, recur_end=0)
text_d = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text_d):6s}] {text_d[:120]!r}")
print(f"  EQUAL TO BASELINE: {text_a == text_d}")
del model; torch.cuda.empty_cache()
