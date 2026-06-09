
import asyncio
import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure we can import from all_space
sys.path.insert(0, os.getcwd())

from all_space.generators import generate_chat_completion
from all_space.px_patches.gemma3_270m_px_baseline.patch import apply_px_patch

async def run_official_style_test(model_id="google/gemma-3-270m-it", config_preset="SUBJECTIVE"):
    print(f"--- Official Style Coherence Test: {model_id} (Preset={config_preset}) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    
    if config_preset != "BASELINE":
        apply_px_patch(model, config_preset=config_preset)
    
    model_entry = {
        "model": model,
        "tokenizer": tokenizer
    }
    
    test_prompts = [
        "What is the capital of France?",
        "Solve: 15 + 27 * 2"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        
        result = await generate_chat_completion(
            model_entry=model_entry,
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=128
        )
        
        print(f"Response: {result['text']}")

if __name__ == "__main__":
    asyncio.run(run_official_style_test(config_preset="BASELINE"))
    asyncio.run(run_official_style_test(config_preset="SUBJECTIVE"))
    asyncio.run(run_official_style_test(config_preset="RIGOR"))
