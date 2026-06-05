"""
test_telemetry_quality.py — Qualitative Analysis of PX Thinking Steps
======================================================================
Compares telemetry JSONs from dmt_space_50 and all_space.
Metrics:
  - Phi Variance: Higher is better (indicates active reasoning).
  - Path Diversity: Number of unique layers visited.
  - AKS Activity: Frequency of sensory refreshes.
  - Emancipation: Final distance from anchor.
"""

import json
import os
import glob
import statistics
from typing import List, Dict, Any

def analyze_telemetry_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        return {}
    
    phis = [step.get("phi", 0.0) for step in data if "phi" in step]
    layers = [step.get("layer", 0) for step in data if "layer" in step]
    
    if not phis:
        return {}
        
    return {
        "steps": len(data),
        "phi_avg": statistics.mean(phis),
        "phi_std": statistics.stdev(phis) if len(phis) > 1 else 0.0,
        "phi_min": min(phis),
        "unique_layers": len(set(layers)),
        "path": [step.get("routing", "NEXT") for step in data]
    }

def run_quality_comparison():
    dmt_files = glob.glob("/run/media/julian/ML4/ollama-work/dmt_space_50/px_telemetry_*.json")
    all_files = glob.glob("/run/media/julian/ML4/ollama-work/all_space/px_telemetry_*.json")
    
    print("=== PX Telemetry Qualitative Comparison ===")
    
    for label, files in [("dmt_space_50 (Phase 58)", dmt_files), ("all_space (Current)", all_files)]:
        if not files:
            print(f"\n{label}: No telemetry files found.")
            continue
            
        stats_list = []
        for f in files[:20]: # Sample last 20
            s = analyze_telemetry_file(f)
            if s: stats_list.append(s)
            
        if not stats_list:
            continue
            
        avg_steps = statistics.mean([s["steps"] for s in stats_list])
        avg_phi_std = statistics.mean([s["phi_std"] for s in stats_list])
        avg_unique_layers = statistics.mean([s["unique_layers"] for s in stats_list])
        
        print(f"\n{label}:")
        print(f"  - Avg Steps per Token: {avg_steps:.1f}")
        print(f"  - Avg Phi Std Dev:    {avg_phi_std:.4f} (Higher = More active)")
        print(f"  - Avg Unique Layers:  {avg_unique_layers:.1f}")
        
        # Check for 'BACK' routing
        back_count = sum(1 for s in stats_list for r in s["path"] if "BACK" in r)
        print(f"  - Total Backtracks:   {back_count} (Indicates active self-correction)")

if __name__ == "__main__":
    run_quality_comparison()
