import os
import json
import asyncio
import torch
import importlib.util
import sys
import time
from model_manager import ModelManager

# Define Tasks for 270M
TASKS = [
    ("arithmetic", "Calculate exactly: 145 * 12 + 18"),
    ("logic", "A man looks at a painting and says: 'Brothers and sisters I have none, but this man's father is my father's son.' Who is in the painting?"),
    ("hle", "Synthesize the concept of hidden-state kurtosis (as a measure of informational peakiness) with the Gödelian Incompleteness Theorem.")
]

# Specifically selected promising variants
VARIANTS = {
    "PEAK_RIGOR": "px_patches/rigor_modules/patch_rigor_peak_rigor_76c974e8.py",
    "PEAK_SUBJECTIVE": "px_patches/rigor_modules/patch_rigor_peak_subjective_e0603adb.py",
    "QUANTUM_RSM": "px_patches/rigor_modules/patch_rigor_hist_0950_ec3e308c.py"
}

async def run_task(model, tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    start_t = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    dur = time.time() - start_t
    
    text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return text, dur

def apply_variant(model, patch_path):
    print(f"Applying patch: {patch_path}")
    # Load module from path
    spec = importlib.util.spec_from_file_location("dynamic_patch", patch_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_patch"] = module
    spec.loader.exec_module(module)
    
    # Apply patch
    if hasattr(module, "apply_px_patch"):
        try:
            # Try to remove old patch first
            if hasattr(module, "remove_px_patch"):
                try: module.remove_px_patch(model)
                except: pass
            
            # Apply with peak defaults
            module.apply_px_patch(model, recur_start=5, recur_end=12, n_loops=8, gamma=0.08)
            
            # Critical: Set tokenizer on text_model for metrics/steering
            tm = (model.model if hasattr(model, "model") else model)
            if hasattr(model, "tokenizer"):
                tm.tokenizer = model.tokenizer
                
            return True
        except Exception as e:
            print(f"  [!] Application error: {e}")
            return False
    return False

async def main():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    
    print(f"Loading {model_id} baseline...")
    entry = await manager.get_model(model_id, px_subjective=False)
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    model.tokenizer = tokenizer # Attach for patch
    
    results = {}
    
    # 1. Baseline Test
    print("\n--- Testing BASELINE ---")
    baseline_results = []
    for cat, prompt in TASKS:
        out, dur = await run_task(model, tokenizer, prompt)
        baseline_results.append({"category": cat, "prompt": prompt, "output": out, "time": round(dur, 2)})
        print(f"[{cat}] {out[:100]}...")
    results["baseline"] = baseline_results
    
    # 2. Variants Test
    for name, path in VARIANTS.items():
        print(f"\n--- Testing VARIANT: {name} ---")
        if apply_variant(model, path):
            variant_results = []
            for cat, prompt in TASKS:
                out, dur = await run_task(model, tokenizer, prompt)
                # Try to get metrics
                phi = getattr(model, "_px_phi", 1.0)
                if not isinstance(phi, float): 
                    tm = (model.model if hasattr(model, "model") else model)
                    phi = getattr(tm, "_px_phi", 1.0)
                
                variant_results.append({
                    "category": cat, 
                    "prompt": prompt, 
                    "output": out, 
                    "time": round(dur, 2),
                    "phi": float(phi) if isinstance(phi, (float, int)) else 1.0
                })
                print(f"[{cat}] Phi: {phi:.4f} | {out[:100]}...")
            results[name] = variant_results
        else:
            print(f"Skipping {name} due to application error.")
            
    with open("comprehensive_rigor_eval.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n[DONE] Results saved to comprehensive_rigor_eval.json")

if __name__ == "__main__":
    asyncio.run(main())
