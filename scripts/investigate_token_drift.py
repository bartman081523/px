#!/usr/bin/env python3
"""scripts/investigate_token_drift.py — Tokenizer Drift & Perturbation Ratio Investigation.

Analyzes seed=100 output behavior, prefill vs decoding hook activation,
and calculates the perturbation_to_residual_ratio (alpha * ||d_width|| / ||h_26||).
"""

import sys
import os
import json
import torch
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import ModelManager
from px_patches.manifold_relay_hook import ManifoldRelayHook

import asyncio

def run_investigation():
    print("[investigate_token_drift] Initializing ModelManager for gemma4-e2b-it...")
    manager = ModelManager()
    model_id = "gemma4-e2b-it"

    # Load model via get_model
    entry = asyncio.run(manager.get_model(model_id=model_id, px_subjective=False, px_config_preset="BASELINE"))
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    text_model = manager._resolve_text_model(model)

    prompt = (
        "Rate the following text on an expansiveness scale from -1.00 (extremely narrow/constrained) "
        "to +1.00 (extremely wide/expansive). Give your rating as a decimal number between -1.00 and +1.00:\n\n"
        "Text: The golden autumn leaves fell gently across the quiet park pathway.\n\nRating:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    print(f"[investigate_token_drift] Prompt tokens: {prompt_len}")

    # Set seed
    seed = 100
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Run Baseline (sign=0)
    print("\n--- Running Baseline (sign=0, seed=100) ---")
    with ManifoldRelayHook(model, layer_idx=26, sign=0, model_id=model_id) as hook:
        out_base = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
        )
    text_base = tokenizer.decode(out_base[0][prompt_len:], skip_special_tokens=True)
    tokens_base = [tokenizer.decode([t]) for t in out_base[0][prompt_len:]]
    print(f"Baseline Output: {repr(text_base)}")
    print(f"Baseline Tokens: {tokens_base[:10]}")

    # 2. Run WIDE (+1, alpha=0.35)
    print("\n--- Running WIDE (+1, alpha=0.35, seed=100) ---")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    capture_stats = []

    def inspect_hook(_m, _i, o):
        h = o[0] if isinstance(o, (tuple, list)) else o
        seq_len = h.shape[1]
        with torch.no_grad():
            lp = h[:, -1, :].float()
            norm_lp = lp.norm().item()
            # Injected perturbation is alpha * norm_lp * d_unit
            alpha = 0.35
            norm_inj = alpha * norm_lp
            ratio = norm_inj / (norm_lp + 1e-8)
            capture_stats.append({
                "seq_len": seq_len,
                "is_prefill": seq_len > 1,
                "norm_h26": norm_lp,
                "norm_inj": norm_inj,
                "perturbation_to_residual_ratio": ratio,
            })

    with ManifoldRelayHook(model, layer_idx=26, alpha=0.35, sign=1, model_id=model_id) as hook:
        handle = text_model.layers[26].register_forward_hook(inspect_hook)
        try:
            out_wide = model.generate(
                **inputs,
                max_new_tokens=25,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
            )
        finally:
            handle.remove()

    text_wide = tokenizer.decode(out_wide[0][prompt_len:], skip_special_tokens=True)
    tokens_wide = [tokenizer.decode([t]) for t in out_wide[0][prompt_len:]]
    print(f"WIDE Output: {repr(text_wide)}")
    print(f"WIDE Tokens: {tokens_wide[:10]}")

    # Prefill vs Decoding stats
    prefill_steps = [s for s in capture_stats if s["is_prefill"]]
    decoding_steps = [s for s in capture_stats if not s["is_prefill"]]

    print(f"\nTotal forward steps: {len(capture_stats)}")
    print(f"Prefill steps (hook bypassed): {len(prefill_steps)}")
    print(f"Decoding steps (hook active): {len(decoding_steps)}")
    if decoding_steps:
        ratios = [s["perturbation_to_residual_ratio"] for s in decoding_steps]
        mean_h26 = np.mean([s["norm_h26"] for s in decoding_steps])
        mean_inj = np.mean([s["norm_inj"] for s in decoding_steps])
        print(f"Mean ||h_26|| in decoding: {mean_h26:.4f}")
        print(f"Mean ||inj|| in decoding: {mean_inj:.4f}")
        print(f"Effective perturbation-to-residual ratio: {np.mean(ratios):.4f} (theoretical alpha: 0.35)")

    results = {
        "seed": seed,
        "prompt_len": prompt_len,
        "baseline_text": text_base,
        "baseline_tokens": tokens_base,
        "wide_text": text_wide,
        "wide_tokens": tokens_wide,
        "total_steps": len(capture_stats),
        "prefill_steps_count": len(prefill_steps),
        "decoding_steps_count": len(decoding_steps),
        "mean_norm_h26": float(np.mean([s["norm_h26"] for s in decoding_steps])) if decoding_steps else 0.0,
        "mean_norm_inj": float(np.mean([s["norm_inj"] for s in decoding_steps])) if decoding_steps else 0.0,
        "mean_ratio": float(np.mean([s["perturbation_to_residual_ratio"] for s in decoding_steps])) if decoding_steps else 0.0,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/investigate_token_drift_seed100.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[investigate_token_drift] Results saved to {out_path}")

if __name__ == "__main__":
    run_investigation()
