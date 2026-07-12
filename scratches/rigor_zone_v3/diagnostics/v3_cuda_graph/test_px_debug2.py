import os
os.environ["PYTHONPATH"] = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
import sys
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from px_patches_v3 import patch as v3_patch

MODEL = "google/gemma-3-270m-it"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"
v3_patch.apply_px_patch(m, "ACTIVE_MANIFOLD")

prompts = [f"What is 2+{i}?" for i in range(8)]
tpl = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
enc = tok(tpl, padding="longest", return_tensors="pt", add_special_tokens=False)
ids = enc["input_ids"].to("cuda")
am = enc["attention_mask"].to("cuda")

from transformers.cache_utils import DynamicCache
past = DynamicCache(config=m.config)
# Use PX-forward with m.model
with torch.inference_mode():
    o = m.model(input_ids=ids, attention_mask=am, past_key_values=past, use_cache=True)
# Check all layers
for i in range(18):
    print(f"  L{i:2d}: keys={past.layers[i].keys.shape if past.layers[i].keys is not None else 'None'}, seq_len={past.get_seq_length(i)}")
