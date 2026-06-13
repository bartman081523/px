"""
tests/repro_math_drift.py — Baseline Measurement for Math/Logic Drift
=====================================================================
Establishes the failure rate of SR-61b (current) before SR-62 intervention.
"""

import asyncio
import json
import os
import sys
import torch
from transformers import AutoTokenizer

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from model_manager import ModelManager

TEST_PROMPTS = [
    {"q": "What is 17 * 23? Answer only the number.", "a": "391", "cat": "math"},
    {"q": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Answer Yes or No and explain briefly.", "a": "No", "cat": "logic"},
    {"q": "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? Answer only the number.", "a": "0.05", "cat": "math"},
    {"q": "Solve for x: 3x + 7 = 22. Answer only x.", "a": "5", "cat": "math"},
    {"q": "Premise 1: No mammals are birds. Premise 2: All dogs are mammals. Conclusion: Are any dogs birds?", "a": "No", "cat": "logic"},
]

async def run_baseline():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    print(f"--- Establishing Baseline for {model_id} (SR-61b) ---")
    
    entry = await manager.get_model(model_id, px_config_preset="ACTIVE_MANIFOLD", px_subjective=True)
    model, tokenizer = entry["model"], entry["tokenizer"]
    
    results = []
    
    for i, test in enumerate(TEST_PROMPTS):
        print(f"Test {i+1}/{len(TEST_PROMPTS)}: {test['q']}")
        
        messages = [{"role": "user", "content": test['q']}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # We use a deterministic generation for the benchmark
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        input_len = inputs["input_ids"].shape[1]
        completion = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
        metrics = manager.get_px_metrics(model_id)
        
        # Simple accuracy check (substring match for target)
        is_correct = test['a'].lower() in completion.lower()
        # Drift check: detect Hindi characters (range \u0900-\u097F)
        has_drift = any('\u0900' <= char <= '\u097f' for char in completion)
        
        res = {
            "prompt": test['q'],
            "completion": completion,
            "correct": is_correct,
            "drift": has_drift,
            "phi": metrics.get("phi"),
            "zone": metrics.get("zone"),
            "steps": metrics.get("steps")
        }
        results.append(res)
        print(f"  > Correct: {is_correct} | Drift: {has_drift} | Zone: {res['zone']}")
        print(f"  > Result: {repr(completion[:50])}...")

    # Calculate overall metrics
    acc = sum(1 for r in results if r['correct']) / len(results)
    drift_rate = sum(1 for r in results if r['drift']) / len(results)
    
    summary = {
        "model_id": model_id,
        "timestamp": "2026-06-13",
        "architecture": "SR-61b (Current)",
        "accuracy": acc,
        "drift_rate": drift_rate,
        "detailed": results
    }
    
    with open("eval/results/SR61_MATH_BASELINE.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*40)
    print(f"BASELINE SUMMARY: Accuracy={acc:.1%} | Drift={drift_rate:.1%}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(run_baseline())
