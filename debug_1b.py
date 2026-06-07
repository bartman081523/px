
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Ensure we can import from all_space
sys.path.insert(0, os.getcwd())

from all_space.px_patches.gemma3_270m_px.patch import apply_px_patch, _resolve_text_model

def debug_1b_structure(model_id="google/gemma-3-1b-it"):
    print(f"--- Debugging Model Structure: {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cpu")
    
    print(f"Model class: {type(model)}")
    if hasattr(model, "model"):
        print(f"model.model class: {type(model.model)}")
    
    tm = _resolve_text_model(model)
    print(f"Resolved text model class: {type(tm)}")
    
    print(f"Applying patch...")
    apply_px_patch(model, config_preset="SUBJECTIVE")
    
    print(f"Attribute check on resolved text model:")
    print(f"  Has _px_injection: {hasattr(tm, '_px_injection')}")
    print(f"  Forward is patched: {tm.forward.__name__ == '_px_forward' if hasattr(tm.forward, '__name__') else False}")
    
    # Check if there's another hidden text model
    for name, module in model.named_modules():
        if "Gemma3TextModel" in type(module).__name__:
            print(f"Found {name} ({type(module)}): _px_injection={hasattr(module, '_px_injection')}")

if __name__ == "__main__":
    debug_1b_structure()
