"""Diagnose v5: Test dynamic_start/end full-range."""
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

print("=" * 70)
print("DIAGNOSE v5 — Was wenn dynamic_start=0, dynamic_end=18?")
print("=" * 70)

# Variante 1: BASELINE
print("\n--- A_BASELINE ---")
tok, model = load_model("BASELINE")
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Variante 2: full range = PRELUDE=0, REASONING=0-17
print("\n--- B_ACTIVE_MANIFOLD full-range (recur_start=0, recur_end=18) ---")
tok, model = load_model("ACTIVE_MANIFOLD", recur_start=0, recur_end=18, n_loops=0)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Variante 3: PRELUDE nur, REASONING leer (recur_start=18, recur_end=18)
print("\n--- C_ACTIVE_MANIFOLD PRELUDE only (recur_start=18, recur_end=18) ---")
tok, model = load_model("ACTIVE_MANIFOLD", recur_start=18, recur_end=18, n_loops=0)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Variante 4: PRELUDE none, REASONING full (recur_start=0, recur_end=0, default end via cfg)
print("\n--- D_ACTIVE_MANIFOLD REASONING only (recur_start=0, recur_end=0 + last) ---")
# Hmm das geht nicht direkt. Statt dessen: PRELUDE = Layer 0-17, REASONING = Layer 0-17 (= same as 18)
