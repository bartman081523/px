"""Diagnose v6: PRELUDE vs REASONING ZONE — was verursacht Degeneration?"""
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

# ACHTUNG: cfg["prelude_end"] = defaults["recur_start"]
# Wenn ich recur_start=0 setze → prelude_end=0 → PRELUDE loop 0 mal
# Wenn ich recur_start=18 setze → prelude_end=18 → PRELUDE läuft alle Layer
# REASONING ZONE läuft [dynamic_start, dynamic_end) = [recur_start, recur_end)

# Test: PRELUDE=18 Layer (alle), REASONING=leer
print("--- recur_start=18, recur_end=18 (PRELUDE alle, REASONING leer) ---")
tok, model = load_model("ACTIVE_MANIFOLD", recur_start=18, recur_end=18, n_loops=0)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Test: PRELUDE=0 Layer, REASONING=alle 18 Layer (mit _layer_step)
print("--- recur_start=0, recur_end=18 (PRELUDE 0, REASONING alle) ---")
tok, model = load_model("ACTIVE_MANIFOLD", recur_start=0, recur_end=18, n_loops=0)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Test: PRELUDE=0-5, REASONING=6-18 (PRELUDE Standard, REASONING _layer_step)
print("--- recur_start=5, recur_end=18 (PRELUDE 0-5, REASONING 5-18) ---")
tok, model = load_model("ACTIVE_MANIFOLD", recur_start=5, recur_end=18, n_loops=0)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Test: PRELUDE 0-17, REASONING 17-18 (nur Layer 17 in REASONING)
print("--- recur_start=17, recur_end=18 (PRELUDE 0-17, REASONING 17) ---")
tok, model = load_model("ACTIVE_MANIFOLD", recur_start=17, recur_end=18, n_loops=0)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()
