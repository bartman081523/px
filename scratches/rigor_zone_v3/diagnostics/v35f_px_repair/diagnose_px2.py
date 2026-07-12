"""Diagnose v2: Teste LEAN vs FULL auf 270m Math-Task.

Hypothese: SubjectiveSensor (Introspection-Loop) oder AksSensor 
verursachen Degeneration. LEAN = ohne diese.
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
        return {"degenerated": True, "reason": "EMPTY", "len": 0}
    words = text.split()
    for plen in [3, 4, 5]:
        for i in range(len(words) - plen*3):
            phrase = " ".join(words[i:i+plen])
            n_repeats = sum(1 for j in range(i+plen, len(words)-plen+1, plen) if " ".join(words[j:j+plen]) == phrase)
            if n_repeats >= 3:
                return {"degenerated": True, "reason": f"LOOPY_p{plen}", "len": len(text)}
    if "0, 0, 0" in text or "1, 1, 1" in text or "\\_\\" in text or "  \\" in text:
        return {"degenerated": True, "reason": "NUMERIC", "len": len(text)}
    return {"degenerated": False, "reason": "OK", "len": len(text)}

print("=" * 70)
print("PX-ENGINE DIAGNOSE v2 — 1 Math-Task")
print("=" * 70)
print(f"GT: Z+Z+Z+Z+Z")
print()

for label, preset, kwargs in [
    ("A_baseline", "BASELINE", {}),
    ("B_ACTIVE_MANIFOLD (full)", "ACTIVE_MANIFOLD", {}),
    ("C_ACTIVE_MANIFOLD_LEAN", "ACTIVE_MANIFOLD_LEAN", {}),
    ("D_ACTIVE_MANIFOLD_RELAY", "ACTIVE_MANIFOLD_RELAY", {}),
]:
    print(f"--- {label} ---")
    try:
        tok, model = load_model(preset, **kwargs)
        text = generate(tok, model, TASK_PROMPT)
        r = analyze(text)
        status = "DEGEN" if r["degenerated"] else "OK"
        print(f"  [{status:5s}] reason={r['reason']:10s} len={r['len']:3d}")
        print(f"  output: {text[:160]!r}")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
    print()
