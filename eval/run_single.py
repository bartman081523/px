import sys, os, json
sys.path.insert(0, ".")
from model_manager import ModelManager
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch
from generators import _px_gen_kwargs
import torch

model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
apply_px_patch(model, config_preset="ACTIVE_MANIFOLD")

prompt = "Create a metaphor for consciousness."
messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

gen_kwargs = {"max_new_tokens": 60, "do_sample": False, "use_cache": False}
gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
out = model.generate(**inputs, **gen_kwargs)
print(tokenizer.decode(out[0], skip_special_tokens=True))
