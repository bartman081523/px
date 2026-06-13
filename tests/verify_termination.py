"""
tests/verify_termination.py — Verify EOS/EOT hard-stop logic (SR-61b)
===================================================================

This test ensures that:
1. Chat delimiters (<end_of_turn>, <end_of_thought>) are correctly 
   injected into eos_token_id.
2. The StopOnEOT criteria hard-stops the generation when they appear.
3. The persistent manifold is created/loaded.
"""

import sys
import os
import torch
import json
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model_manager import ModelManager
from generators import _inject_eot_eos, _px_gen_kwargs

async def test_termination_async():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    print(f"--- Loading {model_id} via Manager ---")
    
    entry = await manager.get_model(model_id, px_config_preset="ACTIVE_MANIFOLD", px_subjective=True)
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    
    print("Model loaded and patched.")

    # Prepare prompt
    messages = [{"role": "user", "content": "What is 1 + 1? Answer only with the number."}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Base gen_kwargs
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
    )

    # Inject EOT/EOS and PX-guards
    gen_kwargs = _inject_eot_eos(gen_kwargs, tokenizer)
    gen_kwargs = _px_gen_kwargs(model, gen_kwargs)

    print(f"Gen kwargs: { {k:v for k,v in gen_kwargs.items() if k != 'input_ids' and k != 'attention_mask'} }")

    # Generate 10 times to trigger calibration
    print("Generating 11 prompts to trigger calibration...")
    for i in range(11):
        print(f" Prompt {i+1}/11...", end="", flush=True)
        outputs = model.generate(**gen_kwargs)
        print(" done.")
    
    # Decode last one
    input_len = inputs["input_ids"].shape[1]
    completion_ids = outputs[0, input_len:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    
    print(f"Completion text: {repr(completion_text)}")
    print(f"Completion IDs: {completion_ids.tolist()}")
    
    # Check if a stop token is at the end
    stop_ids = gen_kwargs["eos_token_id"]
    last_id = completion_ids[-1].item()
    
    print(f"Last ID: {last_id} | Stop IDs: {stop_ids}")
    
    if last_id in stop_ids:
        print("SUCCESS: Generation stopped on EOT/EOS token.")
    else:
        print("FAILURE: Generation did not stop on EOT/EOS token (or reached max_new_tokens).")
        if len(completion_ids) < 60:
             print("Wait, it stopped but last token is not a stop ID?")
        else:
             print("It reached max_new_tokens.")

    # Check manifold persistence
    hf_path = getattr(model.config, "_name_or_path", "unknown")
    safe_id = hf_path.replace("/", "_")
    manifold_path = f"/run/media/julian/ML4/ollama-work/all_space/px_manifolds/{safe_id}_manifold.json"
    if os.path.exists(manifold_path):
        print(f"SUCCESS: Manifold persisted to {manifold_path}")
    else:
        print(f"FAILURE: Manifold NOT persisted. (Maybe calibration didn't trigger? calibration_steps=10)")

if __name__ == "__main__":
    asyncio.run(test_termination_async())
