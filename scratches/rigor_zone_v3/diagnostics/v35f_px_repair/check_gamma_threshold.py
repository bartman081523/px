"""Check: was ist die echte entropy für die Math-Task?"""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['DEBUG_PX'] = '1'  # Aktiviere Debug-Prints

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch
from rigor_hle_suite_v3 import load_hle_suite

source, suite = load_hle_suite(max_per_category=4)
TASKS = [(tid, cat, prompt, gt) for tid, cat, prompt, gt in suite]
target_tid = "66b2c7c979"
prompt = next(p for tid, c, p, g in TASKS if tid.startswith(target_tid))

MAX_NEW = 30

tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", dtype=torch.bfloat16).to("cuda").eval()
apply_px_patch(model.model, config_preset="ACTIVE_MANIFOLD")

msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
