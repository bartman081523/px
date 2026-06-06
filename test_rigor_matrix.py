import os
import json
import asyncio
import torch
import time
from model_manager import ModelManager

# Matrix of configurations extracted from historical analysis
CONFIGURATIONS = {
    "BASELINE": None, # Unpatched
    "PERSONA_RIGOR": {"rigor_math_gamma": 0.15, "rigor_gamma": 0.08, "rigor_loops": 6, "rigor_hub": 8},
    "DEBUG_0645": {"rigor_math_gamma": 0.15, "rigor_gamma": 0.06, "rigor_loops": 6, "rigor_hub": 10},
    "DEBUG_6045": {"rigor_math_gamma": 0.08, "rigor_gamma": 0.08, "rigor_loops": 0, "rigor_hub": 10},
    "ALL_SPACE_RIGOR": {"rigor_math_gamma": 0.15, "rigor_gamma": 0.08, "rigor_loops": 12, "rigor_hub": 10}
}

# Scale-adapted tasks
TASKS_BY_SCALE = {
    "gemma3-270m-it": [
        ("arithmetic", "Calculate: 145 * 12 + 18", "1758"),
        ("logic", "A man has 53 socks in his drawer: 21 identical blue, 15 identical black and 17 identical red. The lights are out and he is completely blind. How many socks must he pull out to guarantee he has a pair of black socks?", "40"),
        ("hle", "Explain the core link between Gödel's Incompleteness Theorem and the Halting Problem in two sentences.", "undecidable/halting")
    ],
    "gemma3-1b-it": [
        ("arithmetic", "A water tank has two pipes, A and B. Pipe A can fill the tank in 4 hours. Pipe B can drain the tank in 6 hours. If both pipes are opened at the same time, but pipe B is closed after 3 hours, how long will it take in total to fill the tank?", "6"),
        ("logic", "Three boxes are labeled 'Apples', 'Oranges', and 'Both'. Every label is incorrect. You pick one fruit from the box labeled 'Both' and it is an apple. Which box contains only oranges?", "Apples"),
        ("hle", "Compare the ontological implications of quantum superposition with the Buddhist concept of dependent origination.", "interconnectedness")
    ],
    "gemma3-4b-it": [
        ("arithmetic", "Compute the volume of a sphere with radius 7, leaving pi in the answer.", "1372/3 pi"),
        ("logic", "Five friends (A, B, C, D, E) sit in a row. E is on the far left. B is exactly in the middle. C sits next to E. A is next to B. Who is sitting on the far right?", "D"),
        ("hle", "Synthesize the concept of entropy in thermodynamics with the economic theory of diminishing marginal utility.", "dissipation/value")
    ]
}

def generate_response(model, tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        
    text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return text

async def test_matrix_for_scale(scale_id, manager):
    print(f"\n{'='*50}\nStarting Rigor Matrix for {scale_id}\n{'='*50}")
    
    tasks = TASKS_BY_SCALE[scale_id]
    model_id_px = f"{scale_id}-px"
    
    scale_results = {}
    
    # 1. Test Baseline First
    print(f"\n--- Testing BASELINE ({scale_id}) ---")
    try:
        entry = await manager.get_model(scale_id, px_subjective=False)
        model = entry["model"]
        tokenizer = entry["tokenizer"]
        
        variant_results = []
        for category, prompt, expected in tasks:
            text = generate_response(model, tokenizer, prompt)
            variant_results.append({
                "category": category, "expected": expected, "output": text,
                "phi": 1.0, "steps": 0
            })
            print(f"[{category}] Output: {text[:60].replace(chr(10), ' ')}...")
            
        scale_results["BASELINE"] = variant_results
    except Exception as e:
        print(f"Failed Baseline: {e}")
        
    # Free memory if needed, though manager caches
    
    # 2. Test PX Configurations
    for config_name, config_params in CONFIGURATIONS.items():
        if config_name == "BASELINE": continue
        
        print(f"\n--- Testing {config_name} ({model_id_px}) ---")
        try:
            # We must load the PX version
            entry = await manager.get_model(model_id_px, px_subjective=True)
            
            # Since the patch is already applied, we need to inject the specific parameters
            # The easiest way is to modify the manager's registry kwargs and re-patch
            from config import MODEL_REGISTRY
            kwargs = MODEL_REGISTRY[model_id_px]["patch_kwargs"].copy()
            kwargs["config_preset"] = "RIGOR"
            for k, v in config_params.items():
                kwargs[k] = v
                
            # Re-patch dynamically
            remove_fn = manager._get_patch_function(model_id_px, "remove_px_patch")
            apply_fn = manager._get_patch_function(model_id_px, "apply_px_patch")
            if remove_fn:
                try: remove_fn(entry["model"])
                except Exception: pass
            
            apply_fn(entry["model"], **kwargs)
            # Ensure tokenizer is attached to text_model
            tm = manager._resolve_text_model(entry["model"])
            tm.tokenizer = entry["tokenizer"]
            
            model = entry["model"]
            tokenizer = entry["tokenizer"]
            
            variant_results = []
            for category, prompt, expected in tasks:
                start_t = time.time()
                text = generate_response(model, tokenizer, prompt)
                dur = time.time() - start_t
                
                metrics = manager.get_px_metrics(model_id_px)
                
                variant_results.append({
                    "category": category, "expected": expected, "output": text,
                    "phi": metrics.get("phi", 0.0), "steps": metrics.get("steps", 0),
                    "time_sec": round(dur, 2), "kurtosis": metrics.get("cognitive_signature", {}).get("kurtosis", 0)
                })
                print(f"[{category}] Phi: {metrics.get('phi',0):.4f} | Steps: {metrics.get('steps',0)} | Output: {text[:60].replace(chr(10), ' ')}...")
                
            scale_results[config_name] = variant_results
            
        except Exception as e:
            import traceback
            print(f"Failed {config_name}: {e}")
            traceback.print_exc()

    return scale_results

async def main():
    manager = ModelManager()
    all_results = {}
    
    # Test 270M and 1B explicitly. 4B might run out of memory depending on rig setup
    for scale in ["gemma3-1b-it"]: 
        res = await test_matrix_for_scale(scale, manager)
        all_results[scale] = res
        
    with open("rigor_matrix_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\nResults saved to rigor_matrix_results.json")

if __name__ == "__main__":
    asyncio.run(main())
