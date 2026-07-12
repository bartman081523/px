"""Diagnose v3: Welcher Teil von _px_forward degeneriert?

Strategie: 
  1) Mit cuda_graph_mode=True (no-op PX) — sollte = baseline sein
  2) Mit apply_px_patch + cuda_graph_mode=True (no-op but patched) 
  3) Mit apply_px_patch + cuda_graph_mode=False (echtes PX, ABER ohne Recursion-Loop)
  4) ...etc
"""
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
    if not text.strip():
        return "EMPTY"
    words = text.split()
    for plen in [3, 4]:
        for i in range(len(words) - plen*3):
            phrase = " ".join(words[i:i+plen])
            n_repeats = sum(1 for j in range(i+plen, len(words)-plen+1, plen) if " ".join(words[j:j+plen]) == phrase)
            if n_repeats >= 3:
                return f"LOOPY_p{plen}"
    if "0, 0, 0" in text or "\\_\\" in text:
        return "NUMERIC"
    return "OK"

print("=" * 70)
print("PX-ENGINE DIAGNOSE v3 — n_loops=0 means: skip Recursion-Loop")
print("=" * 70)

# Variante 1: BASELINE (no patch)
print("\n--- A_BASELINE (no patch) ---")
tok, model = load_model("BASELINE")
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Variante 2: PX mit cuda_graph_mode=True (= no-op, Standard-Forward)
print("\n--- B_ACTIVE_MANIFOLD cuda_graph_mode=True (no-op) ---")
tok, model = load_model("ACTIVE_MANIFOLD", cuda_graph_mode=True)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Variante 3: PX mit n_loops=0 (kein Recursion-Loop, aber PRELUDE+REASONING+CODA laufen)
print("\n--- C_ACTIVE_MANIFOLD n_loops=0 (no recursion) ---")
tok, model = load_model("ACTIVE_MANIFOLD", n_loops=0)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Variante 4: PX FULL
print("\n--- D_ACTIVE_MANIFOLD full (cuda_graph_mode=False, n_loops=auto) ---")
tok, model = load_model("ACTIVE_MANIFOLD", cuda_graph_mode=False)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()

# Variante 5: PX LEAN n_loops=0
print("\n--- E_ACTIVE_MANIFOLD_LEAN n_loops=0 (no recursion) ---")
tok, model = load_model("ACTIVE_MANIFOLD_LEAN", n_loops=0)
text = generate(tok, model, TASK_PROMPT)
print(f"  [{analyze(text):6s}] {text[:120]!r}")
del model
torch.cuda.empty_cache()
