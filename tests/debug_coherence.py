
import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Ensure we can import from all_space
sys.path.insert(0, os.getcwd())

def run_debug_test(model_id="google/gemma-3-270m-it", config_preset="SUBJECTIVE", jitter=0.0):
    print(f"--- Debug Coherence Test: {model_id} (Preset={config_preset}, Jitter={jitter}) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    
    if config_preset != "BASELINE":
        from all_space.px_patches.gemma3_270m_px.patch import apply_px_patch
        apply_px_patch(model, config_preset=config_preset, jitter_mag=jitter)
    else:
        print("[Debug] BASELINE: Skipping PX patch.")
    
    test_prompts = [
        "What is the capital of France?",
        "Solve: 15 + 27 * 2"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=32, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

if __name__ == "__main__":
    # Test 0: BASELINE (unpatched)
    run_debug_test(config_preset="BASELINE", jitter=0.0)
    
    # Test 1: Subjective with NO Jitter
    run_debug_test(config_preset="SUBJECTIVE", jitter=0.0)
