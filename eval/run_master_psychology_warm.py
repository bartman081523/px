"""
eval/run_master_psychology_warm.py — Fast SR-64 Multi-Scale Evaluation
======================================================================
Loads the model ONCE per scale and iterates through the Master Prompt Collection.
Updates and saves the persistent manifolds with the new SR-64 methodology.
"""

import argparse
import json
import os
import sys
import time
import torch
import gc

# Project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model_manager import ModelManager
from config import MODEL_REGISTRY
from eval.master_psychology_prompts import get_master_prompt_collection
from generators import _px_gen_kwargs

def shannon_entropy(weights_dict):
    vals = list(weights_dict.values()) if weights_dict else []
    total = sum(vals)
    if total < 1e-10: return 0.0
    probs = [v / total for v in vals]
    return -sum(p * torch.tensor(p).log2().item() for p in probs if p > 0)

def run_warm_eval(scale, preset, output_dir, limit=None):
    if scale not in ["270M", "1B", "4B", "E2B"]:
        print(f"Error: Unknown scale {scale}")
        return

    model_key = {
        "270M": "gemma3-270m-it",
        "1B":   "gemma3-1b-it",
        "4B":   "gemma3-4b-it",
        "E2B":  "gemma4-e2b-it",
    }[scale]

    os.makedirs(output_dir, exist_ok=True)
    
    from config import MODEL_REGISTRY
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    spec = MODEL_REGISTRY[model_key]
    model_id = spec["hf_id"]
    patch_dir = spec.get("patch_dir")

    print(f"=== Loading Model: {model_id} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    # Identify patch for metrics
    if patch_dir == "gemma4_2b_px":
        from px_patches.gemma4_2b_px.patch import apply_px_patch, get_px_metrics
    elif patch_dir == "gemma3_270m_px_baseline":
        from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, get_px_metrics
    else:
        from px_patches.minicpm5_1b_px.patch import apply_px_patch, get_px_metrics

    print(f"=== Patching Model with {preset} ===")
    apply_px_patch(model, config_preset=preset)

    master_prompts = get_master_prompt_collection()
    if limit:
        master_prompts = master_prompts[:limit]
        
    print(f"=== Starting Evaluation for {scale} ({len(master_prompts)} prompts) ===")
    
    all_results = []
    
    for i, (prompt_text, cat) in enumerate(master_prompts):
        print(f"[{i+1}/{len(master_prompts)}] {scale} | {cat} | {prompt_text[:50]}...")
        
        # Tokenize
        messages = [{"role": "user", "content": prompt_text}]
        try:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            input_text = prompt_text
            
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        
        # Generate
        t0 = time.time()
        gen_kwargs = {
            "max_new_tokens": 60,
            "do_sample": False,
            "temperature": 1.0,
            "use_cache": False,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        gen_time = time.time() - t0
        
        new_tokens = outputs[0][input_len:]
        completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Collect Metrics
        metrics = get_px_metrics(model)
        zw = metrics.get("zone_weights", {})
        phi = metrics.get("phi", 1.0)
        zone = metrics.get("zone", "UNKNOWN")
        sig = metrics.get("cognitive_signature", {})
        
        # Prevent VRAM leak from accumulating telemetry across prompts
        if hasattr(model, "_px_current_telemetry_raw"):
            model._px_current_telemetry_raw = []
        
        res = {
            "prompt": prompt_text,
            "category": cat,
            "completion": completion_text,
            "phi": phi,
            "zone": zone,
            "focus_index": sig.get("focus_index", 0.5),
            "kurtosis": sig.get("kurtosis", 0.0),
            "gamma": sig.get("gamma", 0.08),
            "loops": metrics.get("steps", 0),
            "gen_time": gen_time,
            "tokens": len(new_tokens)
        }
        
        print(f"  ✓ phi={phi:.3f} | C={res['focus_index']:.3f} | zone={zone} | {len(new_tokens)}tok")
        all_results.append(res)
        
        # Save individual result
        safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt_text[:20]])
        res_file = os.path.join(output_dir, f"{i:03d}_{cat}_{safe_prompt}.json")
        with open(res_file, "w") as f:
            json.dump(res, f, indent=2)
            
        # Memory maintenance
        del outputs, inputs
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Final Manifold Save happens automatically via AutoCalibrator, 
    # but we can trigger it or verify it here.
    
    # Aggregate save
    agg_path = os.path.join(output_dir, f"{scale}_master_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"=== Finished {scale}. Aggregate results in {agg_path} ===")
    
    # Cleanup model for next scale
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="eval/results/SR64_MASTER")
    args = parser.parse_args()
    
    run_warm_eval(args.scale, "ACTIVE_MANIFOLD", args.output_dir, limit=args.limit)
