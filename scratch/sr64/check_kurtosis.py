import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir = "/run/media/julian/ML4/ollama-work/all_space"
sys.path.insert(0, base_dir)

from px_patches.gemma3_270m_px_baseline.patch import get_px_metrics, apply_px_patch

model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

apply_px_patch(model, config_preset="ACTIVE_MANIFOLD")

prompts = [
    ("Ultra-Short", "Hi"),
    ("Short", "What is 2+2?"),
    ("Long", "Please provide a detailed, step-by-step explanation of the history of the Roman Empire, focusing specifically on the transition from the Republic to the Empire under Augustus. Include multiple paragraphs and cite specific historical events and political changes that led to this monumental shift in governance.")
]

for name, text in prompts:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    token_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2)
    
    kurtosis = getattr(model.model, "_task_kurtosis", 0.0)
    print(f"[{name} Prompt] Token Len: {token_len} | Native Kurtosis: {kurtosis:.2f}")

