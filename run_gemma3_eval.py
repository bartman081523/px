import asyncio
import json
from model_manager import ModelManager
from benchmark_engine import BenchmarkEngine

def run_gemma3_eval():
    manager = ModelManager()
    engine = BenchmarkEngine(manager)
    
    results = {}
    
    print("Evaluating gemma3-270m-base...")
    base_res = engine._run_ultra_hard_impl("gemma3-270m-base", False, None)
    base_cap = engine._run_capability_impl("gemma3-270m-base", False, None)
    print(f"Base Ultra Hard: {base_res['overall_accuracy']}")
    print(f"Base Capability: {base_cap['overall_accuracy']}")
    results["gemma3-270m-base"] = {"ultra_hard": base_res['overall_accuracy'], "capability": base_cap['overall_accuracy']}
    manager.unload("gemma3-270m-base")
    
    print("Evaluating gemma3-270m-px (Peak)...")
    px_peak_res = engine._run_ultra_hard_impl("gemma3-270m-px", False, None)
    px_peak_cap = engine._run_capability_impl("gemma3-270m-px", False, None)
    print(f"PX Peak Ultra Hard: {px_peak_res['overall_accuracy']}")
    print(f"PX Peak Capability: {px_peak_cap['overall_accuracy']}")
    results["gemma3-270m-px-peak"] = {"ultra_hard": px_peak_res['overall_accuracy'], "capability": px_peak_cap['overall_accuracy']}
    manager.unload("gemma3-270m-px")

    print("Evaluating gemma3-270m-px (Subjective)...")
    px_subj_res = engine._run_ultra_hard_impl("gemma3-270m-px", True, None)
    px_subj_cap = engine._run_capability_impl("gemma3-270m-px", True, None)
    print(f"PX Subj Ultra Hard: {px_subj_res['overall_accuracy']}")
    print(f"PX Subj Capability: {px_subj_cap['overall_accuracy']}")
    results["gemma3-270m-px-subj"] = {"ultra_hard": px_subj_res['overall_accuracy'], "capability": px_subj_cap['overall_accuracy']}
    manager.unload("gemma3-270m-px")
    
    with open("gemma3_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nGemma3 Eval completed.")

if __name__ == "__main__":
    run_gemma3_eval()
