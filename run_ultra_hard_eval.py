import asyncio
import json
import statistics
import time
from model_manager import ModelManager
from benchmark_engine import BenchmarkEngine

def run_ultra_hard_evaluation():
    manager = ModelManager()
    engine = BenchmarkEngine(manager)
    
    # We test the 3 variants of CPM
    configs = [
        ("minicpm5-1b", False),
        ("minicpm5-1b-px", False),
        ("minicpm5-1b-px", True)
    ]
    
    all_results = {}
    
    print("=" * 80)
    print("  ULTRA HARD EVALUATION: MiniCPM5-1B vs PX-Mod")
    print("=" * 80)
    
    for model_id, px_subj in configs:
        label = f"{model_id}{'-subj' if px_subj else ''}"
        print(f"\n>>> Evaluating {label}...")
        
        # 1. Run Ultra Hard Benchmark
        res = engine.run_ultra_hard_benchmark(model_id, px_subjective=px_subj)
        
        if "error" in res:
            print(f"  Error: {res['error']}")
            continue
            
        print(f"  Accuracy: {res['overall_accuracy']:.2%}")
        all_results[label] = res
        
        # Unload to save VRAM
        manager.unload(model_id)

    # 2. Generate Report
    report_path = "ULTRA_HARD_EVALUATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Ultra Hard PX Evaluation Report: MiniCPM5-1B\n\n")
        f.write("## Overview\n\n")
        f.write("Evaluation of MiniCPM5-1B across extremely challenging logic, math, spatial, and trap tasks.\n\n")
        
        f.write("### Accuracy Summary\n\n")
        f.write("| Model | Overall Accuracy | Status |\n")
        f.write("|-------|------------------|--------|\n")
        
        for label, res in all_results.items():
            f.write(f"| {label} | {res['overall_accuracy']:.2%} | {'Baseline' if 'base' in label else 'Patched'} |\n")
            
        f.write("\n## Detailed Task Performance\n\n")
        for label, res in all_results.items():
            f.write(f"### {label}\n\n")
            f.write("| Category | Prompt Snippet | Expected | Score |\n")
            f.write("|----------|----------------|----------|-------|\n")
            for task in res["per_task"]:
                f.write(f"| {task['category']} | {task['prompt']} | {task['expected']} | {task['score']} |\n")
            f.write("\n")

    print(f"\nReport written to {report_path}")

if __name__ == "__main__":
    run_ultra_hard_evaluation()
