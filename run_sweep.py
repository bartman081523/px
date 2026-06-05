import json
import asyncio
from model_manager import ModelManager
from benchmark_engine import BenchmarkEngine

def run_sweep():
    manager = ModelManager()
    engine = BenchmarkEngine(manager)
    
    gammas = [0.04, 0.08, 0.12]
    loops = [4, 8, 10]
    
    results = {}
    
    print("Starting PX Parameter Sweep on MiniCPM5-1B-PX (Peak Mode)...")
    
    for g in gammas:
        for l in loops:
            label = f"gamma_{g}_loops_{l}"
            print(f"\n--- Testing {label} ---")
            
            from config import MODEL_REGISTRY
            registry = MODEL_REGISTRY["minicpm5-1b-px"]
            registry["patch_kwargs"]["gamma"] = g
            registry["patch_kwargs"]["n_loops"] = l
            
            # Using _run_capability_impl which handles loading
            res = engine._run_capability_impl("minicpm5-1b-px", False, None)
            uh_res = engine._run_ultra_hard_impl("minicpm5-1b-px", False, None)
            
            print(f"Capability Acc: {res['overall_accuracy']}")
            print(f"Ultra Hard Acc: {uh_res['overall_accuracy']}")
            
            results[label] = {
                "capability": res['overall_accuracy'],
                "ultra_hard": uh_res['overall_accuracy']
            }
            
            manager.unload("minicpm5-1b-px")
            
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nSweep completed. Results saved to sweep_results.json")

if __name__ == "__main__":
    run_sweep()
