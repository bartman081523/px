import asyncio
from model_manager import ModelManager
from benchmark_engine import BenchmarkEngine

def run():
    manager = ModelManager()
    engine = BenchmarkEngine(manager)
    
    print("Evaluating minicpm5-1b-base...")
    base_res = engine._run_ultra_hard_impl("minicpm5-1b-base", False, None)
    print(f"Base Accuracy: {base_res['overall_accuracy']}")
    
    print("Evaluating minicpm5-1b-px (Peak)...")
    px_res = engine._run_ultra_hard_impl("minicpm5-1b-px", False, None)
    print(f"PX Peak Accuracy: {px_res['overall_accuracy']}")

    print("Evaluating minicpm5-1b-px (Subjective)...")
    subj_res = engine._run_ultra_hard_impl("minicpm5-1b-px", True, None)
    print(f"PX Subjective Accuracy: {subj_res['overall_accuracy']}")

if __name__ == "__main__":
    run()