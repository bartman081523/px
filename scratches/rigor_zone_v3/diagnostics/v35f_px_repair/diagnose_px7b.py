"""Diagnose v7b: Genauere Inspections."""
import sys, os, traceback
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
    try:
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        new_tokens = out[0, ids["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)
    except Exception as e:
        return f"ERROR: {e}"

# BASELINE first
print("--- A_BASELINE ---")
tok, model = load_model("BASELINE")
text_a = generate(tok, model, TASK_PROMPT)
print(f"  {text_a[:120]!r}")
del model; torch.cuda.empty_cache()

# Now baseline for testing — use NO PX
# Same as A but with BASELINE check
print("--- B_ACTIVE_MANIFOLD (default, ohne override) ---")
tok, model = load_model("ACTIVE_MANIFOLD")
text_b = generate(tok, model, TASK_PROMPT)
print(f"  {text_b[:120]!r}")
del model; torch.cuda.empty_cache()
