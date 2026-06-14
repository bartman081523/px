import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir = "/run/media/julian/ML4/ollama-work/all_space"
sys.path.insert(0, base_dir)

from config import MODEL_REGISTRY
from eval.master_psychology_prompts import get_master_prompt_collection
from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, get_px_metrics

model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

apply_px_patch(model, config_preset="ACTIVE_MANIFOLD")

prompts = get_master_prompt_collection()[:5]

for i, (prompt_text, cat) in enumerate(prompts):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    
    metrics = get_px_metrics(model)
    c_idx = getattr(model, "_px_focus_index", metrics.get("cognitive_signature", {}).get("focus_index", -1.0))
    print(f"[{i+1}/5] C={c_idx:.3f} | {cat} | Text: {prompt_text[:30]}")
