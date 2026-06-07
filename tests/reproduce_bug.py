
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Ensure we can import from all_space
sys.path.insert(0, os.getcwd())

from all_space.px_patches.gemma3_270m_px.patch import apply_px_patch

def reproduce_error(model_id="google/gemma-3-270m-it"):
    print(f"--- Attempting to reproduce RuntimeError for {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    
    apply_px_patch(model, config_preset="SUBJECTIVE")
    
    # Simulate a long chat history or a specific prompt length
    # The numbers 2036 and 1525 might come from a large input
    prompt = "Explain the history of the world in great detail. " * 50
    messages = [{"role": "user", "content": prompt}]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    print(f"Input length: {inputs['input_ids'].shape[1]}")
    print(f"Model config: {model.config}")
    
    # Try multiple lengths
    for length in [1024, 1525, 2036]:
        print(f"\n--- Testing length {length} ---")
        prompt = "test " * length
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=True,
                    temperature=0.7
                )
            print(f"Length {length} success.")
        except Exception as e:
            print(f"Length {length} FAILED: {e}")
            # traceback.print_exc()

if __name__ == "__main__":
    reproduce_error()
