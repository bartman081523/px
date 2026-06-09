import torch
import json
import os
import sys
import asyncio
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_manager import ModelManager
from benchmark_engine import score_answer

BENCHMARK_FILE = "/run/media/julian/ML4/open-mythos_p2/gemma-3-270m-it-p2.8/data/ultra_diverse_bench.json"

CONFIGS = [
    {"name": "Baseline", "model_id": "gemma3-270m-it", "px_kwargs": {}},
    {"name": "Peak", "model_id": "gemma3-270m-it-px", "px_kwargs": {"subjective_enabled": False, "persona_enabled": False, "dmt_protocol_enabled": False}},
    {"name": "Subjective", "model_id": "gemma3-270m-it-px", "px_kwargs": {"subjective_enabled": True, "persona_enabled": False, "dmt_protocol_enabled": False}},
    {"name": "Persona", "model_id": "gemma3-270m-it-px", "px_kwargs": {"subjective_enabled": False, "persona_enabled": True, "dmt_protocol_enabled": False}},
    {"name": "DMT-Full", "model_id": "gemma3-270m-it-px", "px_kwargs": {"subjective_enabled": True, "persona_enabled": True, "dmt_protocol_enabled": True}},
]

PERSONA = "You are a highly logical and creative reasoning engine. Analyze the following request deeply and provide a precise, grounded answer."

async def run_benchmark():
    manager = ModelManager()
    
    with open(BENCHMARK_FILE, "r") as f:
        bench_data = json.load(f)
    
    all_results = {}
    
    for config in CONFIGS:
        print(f"\n[Bench] Running {config['name']}...")
        
        # Load/Patch model
        model_id = config["model_id"]
        px_kwargs = config["px_kwargs"]
        
        # We need to manually apply patch for custom kwargs since ModelManager is limited
        model_entry = await manager.get_model(model_id)
        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]
        
        if "px" in model_id:
            # 2026-06-09: routed to isolated baseline (gemma3)
            from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch
            apply_px_patch(model, **px_kwargs)
            if px_kwargs.get("persona_enabled"):
                model.persona = PERSONA
        
        config_results = []
        domain_scores = {}
        
        for domain_data in bench_data:
            domain = domain_data["domain"]
            tasks = domain_data["tasks"]
            print(f"  Domain: {domain}")
            
            domain_results = []
            for task in tasks:
                q, expected = task["q"], task["expected"]
                
                inputs = tokenizer(q, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                
                input_len = inputs["input_ids"].shape[1]
                output_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                
                # Check for multiple expected variants separated by |
                if "|" in expected:
                    score = 0.0
                    for variant in expected.split("|"):
                        if score_answer(output_text, variant) > 0:
                            score = 1.0; break
                else:
                    score = score_answer(output_text, expected)
                
                domain_results.append({
                    "q": q,
                    "expected": expected,
                    "output": output_text,
                    "score": score,
                    "tag": task.get("tag", "generic")
                })
            
            avg_domain_score = sum(r["score"] for r in domain_results) / len(domain_results)
            domain_scores[domain] = avg_domain_score
            config_results.append({
                "domain": domain,
                "score": avg_domain_score,
                "tasks": domain_results
            })
            
        overall_score = sum(domain_scores.values()) / len(domain_scores)
        all_results[config["name"]] = {
            "overall_score": overall_score,
            "domain_scores": domain_scores,
            "detailed_results": config_results
        }
        print(f"  [Bench] {config['name']} Overall: {overall_score:.4f}")
        
        # Unload to save memory
        manager.unload(model_id)
        torch.cuda.empty_cache()

    with open("ultra_diverse_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[Bench] Completed. Results saved to ultra_diverse_results.json")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
