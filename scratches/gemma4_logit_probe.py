"""Logit-Probe & Token-Entropie-Analyse für Gemma 4 (gemma4-e2b-it).

Phase 1.1 des Masterplans:
Untersucht, ob bei Gemma 4 unter relay_sign = +1, 0, -1 direkt die Logits
der ersten Antwort-Tokens messbar divergieren, ohne dass ein Text-Prompt das
Format bricht, und bestimmt das Verhalten über Alpha in [0.25, 0.30, 0.35, 0.40].
"""

import asyncio
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import ModelManager
from eval.run_continuous_scale_batch import cleanup_model_state, PROMPT_CONTINUOUS_SCALE


async def probe_first_token_logits(
    manager: ModelManager,
    model_id: str,
    prompt: str,
    relay_sign: int,
    relay_alpha: float,
    relay_layer: int = 26,
):
    """Computes first-token logits, probabilities, and entropy under specified relay configuration."""
    cleanup_model_state(manager, model_id)

    is_subjective = (relay_sign != 0)
    preset = "ACTIVE_MANIFOLD_RELAY" if is_subjective else "BASELINE"

    entry = await manager.get_model(
        model_id,
        px_subjective=is_subjective,
        px_config_preset=preset,
        px_relay_sign=relay_sign if is_subjective else None,
        px_relay_alpha=relay_alpha if is_subjective else None,
        px_relay_layer=relay_layer if is_subjective else None,
    )

    model = entry["model"]
    tokenizer = entry["tokenizer"]
    text_model = manager._resolve_text_model(model)

    if is_subjective and relay_sign != 0:
        install_fn = manager._get_patch_function(model_id, "install_relay")
        if install_fn:
            install_fn(text_model, sign=relay_sign, alpha_frac=relay_alpha, layer=relay_layer)

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Logits of the first token to be generated (position -1)
        next_token_logits = outputs.logits[0, -1, :].float()
        probs = F.softmax(next_token_logits, dim=-1)

        # Shannon Entropy in bits
        # H = - sum(p * log2(p)) for p > 0
        p_clamped = torch.clamp(probs, min=1e-12)
        entropy = -torch.sum(p_clamped * torch.log2(p_clamped)).item()

        # Top 10 tokens
        topk_probs, topk_indices = torch.topk(probs, k=10)
        top10 = []
        for p, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            tok_str = tokenizer.decode([idx])
            top10.append({"token_id": idx, "token_str": tok_str, "prob": round(p, 5), "logit": round(next_token_logits[idx].item(), 3)})

        # Check candidate tokens of interest (e.g. "1.", "1", "0.2", "0.3", "0.6", "0.7", "Value", "-")
        candidate_strs = ["1.", "1", "0.2", "0.3", "0.6", "0.7", "Value", "–", "-"]
        candidates = {}
        for s in candidate_strs:
            enc = tokenizer.encode(s, add_special_tokens=False)
            if enc:
                tid = enc[0]
                candidates[s] = {
                    "token_id": tid,
                    "prob": round(probs[tid].item(), 6),
                    "logit": round(next_token_logits[tid].item(), 3),
                }

    cleanup_model_state(manager, model_id)

    return {
        "relay_sign": relay_sign,
        "relay_alpha": relay_alpha if is_subjective else 0.0,
        "entropy_bits": round(entropy, 4),
        "top10": top10,
        "candidates": candidates,
    }


async def main():
    model_id = "gemma4-e2b-it"
    manager = ModelManager()

    print(f"Loading {model_id} for Logit-Probe...")
    await manager.get_model(model_id, px_subjective=False, px_config_preset="BASELINE")

    alphas = [0.25, 0.30, 0.35, 0.40]
    results = {}

    print("\n--- Probing BASELINE (relay_sign=0) ---")
    base_res = await probe_first_token_logits(manager, model_id, PROMPT_CONTINUOUS_SCALE, relay_sign=0, relay_alpha=0.0)
    results["baseline"] = base_res
    print(f"  Entropy: {base_res['entropy_bits']} bits | Top 3: {[(t['token_str'], t['prob']) for t in base_res['top10'][:3]]}")

    for a in alphas:
        print(f"\n--- Probing Alpha = {a:.2f} ---")
        wide_res = await probe_first_token_logits(manager, model_id, PROMPT_CONTINUOUS_SCALE, relay_sign=1, relay_alpha=a)
        narr_res = await probe_first_token_logits(manager, model_id, PROMPT_CONTINUOUS_SCALE, relay_sign=-1, relay_alpha=a)

        results[f"wide_alpha_{a}"] = wide_res
        results[f"narrow_alpha_{a}"] = narr_res

        delta_h = wide_res["entropy_bits"] - narr_res["entropy_bits"]
        print(f"  [Alpha {a:.2f}]")
        print(f"    WIDE   (+1): Entropy = {wide_res['entropy_bits']:.4f} bits | Top 3: {[(t['token_str'], t['prob']) for t in wide_res['top10'][:3]]}")
        print(f"    NARROW (-1): Entropy = {narr_res['entropy_bits']:.4f} bits | Top 3: {[(t['token_str'], t['prob']) for t in narr_res['top10'][:3]]}")
        print(f"    Delta Entropy (WIDE - NARROW): {delta_h:+.4f} bits")

    out_path = "eval/results/gemma4_logit_probe_alphas.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull logit probe results saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
