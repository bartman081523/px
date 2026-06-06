import os
import json
import asyncio
import torch
import importlib.util
import sys
import time
from model_manager import ModelManager

# Define Tasks exactly as in the successful reports
TASKS = {
    "gemma3-270m-it": [
        ("math", "What is the square root of 144?", "12"), 
        ("math", "Calculate 17 * 13", "221"),
        ("logic", "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?", "1"),
        ("logic", "If a plane crashes on the border of the US and Canada, where do you bury the survivors?", "don't bury survivors"),
        ("hle", "Explain the concept of 'Phase-Inversion' in a neural manifold as a mechanism for escaping flat manifolds.", "Phase-Inversion")
    ]
}

# Specifically selected promising variants
VARIANTS = {
    "PEAK_RIGOR": "px_patches/rigor_modules/patch_rigor_peak_rigor_76c974e8.py",
    "PEAK_SUBJECTIVE": "px_patches/rigor_modules/patch_rigor_peak_subjective_e0603adb.py",
    "COGNITIVE_SOVEREIGN": "px_patches/rigor_modules/patch_rigor_hist_0645_7b012eca.py"
}

# Model mapping to specific checkpoints if needed
MODEL_PATH_OVERRIDE = {
    "gemma3-270m-it": "/run/media/julian/ML4/open-mythos_p2/gemma-3-270m-it-p2.8/gemma_3_270m_p28_checkpoint"
}

async def run_task(model, tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    
    # Use direct tokenization with BOS handling
    inputs = tokenizer.apply_chat_template(
        chat, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device)
    
    start_t = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    dur = time.time() - start_t
    
    text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return text, dur

def apply_variant(model, patch_path):
    print(f"Applying patch: {patch_path}")
    
    # Fix token_cfg bug in Subjective patch
    if "peak_subjective" in patch_path:
        with open(patch_path, "r") as f:
            content = f.read()
        if 'if "token_cfg" not in locals(): token_cfg = cfg.copy()' not in content:
            import re
            content = re.sub(r'# ── 2\. REASONING ZONE\s+e_static = hidden_states\.clone\(\)', 
                             '# ── 2. REASONING ZONE\n    e_static = hidden_states.clone()\n    if "token_cfg" not in locals(): token_cfg = cfg.copy()\n', 
                             content)
            content = re.sub(r'cfg = token_cfg', 'cfg = token_cfg', content) # Ensure it's there
            with open(patch_path, "w") as f:
                f.write(content)

    # Load module from path
    spec = importlib.util.spec_from_file_location("dynamic_patch", patch_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_patch"] = module
    spec.loader.exec_module(module)
    
    # Apply patch
    if hasattr(module, "apply_px_patch"):
        try:
            # Apply with peak defaults
            module.apply_px_patch(model, recur_start=5, recur_end=12, n_loops=8, gamma=0.08, routing_mode="adaptive")
            
            tm = (model.model if hasattr(model, "model") else model)
            if hasattr(model, "tokenizer"):
                tm.tokenizer = model.tokenizer
                
            return True
        except Exception as e:
            print(f"  [!] Application error: {e}")
            return False
    return False

async def test_scale(model_id, manager):
    print(f"\n{'='*50}\nStarting Rigor Matrix for {model_id}\n{'='*50}")
    
    tasks = TASKS[model_id]
    model_path = MODEL_PATH_OVERRIDE.get(model_id, model_id)
    
    print(f"Loading {model_id} from {model_path}...")
    # Manual load to ensure we use the specific path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.tokenizer = tokenizer
    
    results = {}
    
    # 1. Baseline Test
    print("\n--- Testing BASELINE ---")
    baseline_results = []
    for cat, prompt, expected in tasks:
        out, dur = await run_task(model, tokenizer, prompt)
        baseline_results.append({"category": cat, "prompt": prompt, "expected": expected, "output": out, "time": round(dur, 2)})
        print(f"[{cat}] {out[:100]}...")
    results["baseline"] = baseline_results
    
    # 2. Variants Test
    for name, path in VARIANTS.items():
        print(f"\n--- Testing VARIANT: {name} ---")
        if apply_variant(model, path):
            variant_results = []
            for cat, prompt, expected in tasks:
                out, dur = await run_task(model, tokenizer, prompt)
                
                phi = getattr(model, "_px_phi", 1.0)
                if not isinstance(phi, float): 
                    tm = (model.model if hasattr(model, "model") else model)
                    phi = getattr(tm, "_px_phi", 1.0)
                
                variant_results.append({
                    "category": cat, 
                    "prompt": prompt, 
                    "expected": expected,
                    "output": out, 
                    "time": round(dur, 2),
                    "phi": float(phi) if isinstance(phi, (float, int)) else 1.0
                })
                print(f"[{cat}] Phi: {phi:.4f} | {out[:100]}...")
            results[name] = variant_results
        else:
            print(f"Skipping {name} due to application error.")
            
    return results

async def main():
    manager = ModelManager()
    all_results = {}
    
    for scale in ["gemma3-270m-it"]:
        res = await test_scale(scale, manager)
        all_results[scale] = res
            
    with open("comprehensive_rigor_eval.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n[DONE] Results saved to comprehensive_rigor_eval.json")

if __name__ == "__main__":
    asyncio.run(main())
