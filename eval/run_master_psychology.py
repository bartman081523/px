"""
eval/run_master_psychology.py — Comprehensive SR-64 Cross-Scale Evaluation
==========================================================================
Runs the diverse Master Prompt Collection across all model scales.
Measures: η², Accuracy, Focus Index (C), and Drift.
Saves updated manifolds for each model.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import random

# Project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.master_psychology_prompts import get_master_prompt_collection

SCALE_TO_MODEL = {
    "270M": "gemma3-270m-it",
    "1B":   "gemma3-1b-it",
    "4B":   "gemma3-4b-it",
    "E2B":  "gemma4-e2b-it",
}

def _spawn_runner(prompt_text, model_id, preset, max_new_tokens, result_path, timeout=600):
    cfg = {
        "prompt": prompt_text,
        "model_id": model_id,
        "preset": preset,
        "max_new_tokens": max_new_tokens,
        "result_path": result_path,
    }
    cfg_path = result_path + ".cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    cmd = [
        sys.executable,
        os.path.join(_ROOT, "eval", "runner.py"),
        cfg_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "prompt": prompt_text}

    if proc.returncode != 0:
        return {"error": f"exit_code_{proc.returncode}", "prompt": prompt_text, "stderr": proc.stderr}
    
    try:
        with open(result_path) as f:
            return json.load(f)
    except:
        return {"error": "read_failed", "prompt": prompt_text}

def run_master_eval(scale, preset, output_dir):
    print(f"DEBUG: Starting run_master_eval for {scale}")
    if scale not in SCALE_TO_MODEL:
        print(f"DEBUG: Scale {scale} not found")
        return
    model_id = SCALE_TO_MODEL[scale]
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"DEBUG: Getting master prompts")
    master_prompts = get_master_prompt_collection()
    print(f"=== Starting Master Evaluation for {scale} ({len(master_prompts)} prompts) ===")
    
    results = []
    for i, (prompt, cat) in enumerate(master_prompts[:5]):
        safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt[:30]])
        res_path = os.path.join(output_dir, f"{i:03d}_{cat}_{safe_prompt}.json")
        
        print(f"[{i+1}/{len(master_prompts)}] {scale} | {cat} | {prompt[:50]}...")
        res = _spawn_runner(prompt, model_id, preset, 60, res_path)
        
        if "error" in res:
            print(f"  FAILED: {res.get('error')}")
        else:
            phi = res.get("phi", 1.0)
            focus = res.get("cognitive_signature", {}).get("focus_index", 0.5)
            zone = res.get("zone", "UNKNOWN")
            print(f"  ✓ phi={phi:.3f} | C={focus:.3f} | zone={zone}")
            results.append(res)
            
    # Save aggregate
    agg_path = os.path.join(output_dir, f"{scale}_master_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"=== Finished {scale}. Results in {agg_path} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="270M")
    parser.add_argument("--preset", type=str, default="ACTIVE_MANIFOLD")
    parser.add_argument("--output-dir", type=str, default="eval/results/SR64_MASTER")
    args = parser.parse_args()
    
    run_master_eval(args.scale, args.preset, args.output_dir)
