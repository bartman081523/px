"""
run_comprehensive_eval.py — Extensive Evaluation of MiniCPM5-1B vs PX-Mod
========================================================================
Runs Logic, Arithmetic, HLE, and Creative tasks.
Compares Baseline vs PX-Peak vs PX-Subjective.
"""

import asyncio
import json
import os
import statistics
import time
from typing import Dict, List, Any

from model_manager import ModelManager
from benchmark_engine import BenchmarkEngine, score_answer
from test_prompts import (
    ALL_CAPABILITY_TASKS, CREATIVE_PROMPTS, SYNTHESIS_PROMPTS, CALIBRATION_PROMPTS
)

async def run_evaluation():
    manager = ModelManager()
    engine = BenchmarkEngine(manager)
    
    models = ["minicpm5-1b-base", "minicpm5-1b-px", "minicpm5-1b-px-subj"]
    configs = [
        ("minicpm5-1b-base", False),
        ("minicpm5-1b-px", False),
        ("minicpm5-1b-px", True)
    ]
    
    all_results = {}
    
    print("=" * 80)
    print("  COMPREHENSIVE COGNITIVE EVALUATION: MiniCPM5-1B vs PX-Mod")
    print("=" * 80)
    
    for model_id, px_subj in configs:
        label = f"{model_id}{'-subj' if px_subj else ''}"
        print(f"\n>>> Evaluating {label}...")
        
        # Get model
        model_entry = await manager.get_model(model_id, px_subjective=px_subj)
        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]
        
        # 1. Calibration (if PX)
        if "px" in model_id:
            print("  Calibrating PX...")
            for cp in CALIBRATION_PROMPTS:
                inputs = tokenizer(cp, return_tensors="pt").to(model.device)
                model.generate(**inputs, max_new_tokens=5, do_sample=False)
        
        # 2. Objective Tasks (Logic, Arithmetic, HLE)
        print(f"  Running {len(ALL_CAPABILITY_TASKS)} objective tasks...")
        obj_results = []
        for cat, prompt, expected in ALL_CAPABILITY_TASKS:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            
            score = score_answer(text, expected)
            obj_results.append({
                "category": cat,
                "prompt": prompt,
                "expected": expected,
                "output": text,
                "score": score
            })
            
        # 3. Creative/Synthesis Tasks
        print(f"  Running {len(CREATIVE_PROMPTS)} creative/synthesis tasks...")
        creative_results = []
        for prompt in CREATIVE_PROMPTS[:10] + SYNTHESIS_PROMPTS[:10]:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            
            # Qualitative metrics
            metrics = manager.get_px_metrics(model_id)
            creative_results.append({
                "prompt": prompt,
                "output": text,
                "length": len(text),
                "phi": metrics.get("phi", 1.0),
                "steps": metrics.get("steps", 0),
                "zone": metrics.get("zone", "N/A")
            })
            
        all_results[label] = {
            "objective": obj_results,
            "creative": creative_results,
            "accuracy": statistics.mean([r["score"] for r in obj_results])
        }
        
        print(f"  {label} Accuracy: {all_results[label]['accuracy']:.2%}")
        
        # Unload to save VRAM
        manager.unload(model_id)

    # 4. Generate Report
    report_path = "CPM_PX_EVALUATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# MiniCPM5-1B PX-Mod Evaluation Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("Comparison of MiniCPM5-1B Base vs PX-Patched (Peak and Subjective modes).\n\n")
        
        f.write("### Accuracy Summary\n\n")
        f.write("| Model | Overall Accuracy | Logic | Arithmetic | HLE |\n")
        f.write("|-------|------------------|-------|------------|-----|\n")
        
        for label, res in all_results.items():
            cats = ["logic", "arithmetic", "hle"]
            cat_accs = {}
            for c in cats:
                scores = [r["score"] for r in res["objective"] if r["category"] == c]
                cat_accs[c] = statistics.mean(scores) if scores else 0
            
            f.write(f"| {label} | {res['accuracy']:.2%} | {cat_accs['logic']:.2%} | {cat_accs['arithmetic']:.2%} | {cat_accs['hle']:.2%} |\n")
            
        f.write("\n## Qualitative Analysis (Creative & Synthesis)\n\n")
        for label, res in all_results.items():
            f.write(f"### {label}\n\n")
            avg_len = statistics.mean([r["length"] for r in res["creative"]])
            avg_phi = statistics.mean([r["phi"] for r in res["creative"]])
            f.write(f"- Average response length: {avg_len:.1f} chars\n")
            f.write(f"- Average Phi (Stability): {avg_phi:.4f}\n\n")
            
    print(f"\nReport written to {report_path}")
    
    # Save raw data
    with open("eval_results_raw.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
