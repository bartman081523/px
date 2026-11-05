#!/usr/bin/env python3
"""scripts/calibrate_metrics_gemma4.py — Gemma 4 d_width Vocabulary Projection & Metric Calibration.

Computes the projection of d_width onto the unembedding matrix W_U,
extracts top-25 positive/negative tokens, maps digit & choice token IDs,
and saves artifacts/gemma4_e2b_d_width_vocab_projection.json.
"""

import os
import sys
import json
import asyncio
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import ModelManager


async def main():
    print("[calibrate_metrics_gemma4] Initializing ModelManager...")
    manager = ModelManager()
    model_id = "gemma4-e2b-it"

    entry = await manager.get_model(model_id=model_id, px_subjective=False, px_config_preset="BASELINE")
    model = entry["model"]
    tokenizer = entry["tokenizer"]

    # 1. Load d_width artifact
    dwidth_path = os.path.join("px_manifolds", "google_gemma-4-E2B-it_relay_dwidth.json")
    if not os.path.exists(dwidth_path):
        raise FileNotFoundError(f"Missing relay artifact: {dwidth_path}")

    with open(dwidth_path, "r", encoding="utf-8") as f:
        art = json.load(f)

    dwidth_np = np.array(art["dwidth"], dtype=np.float32)
    dwidth_tensor = torch.tensor(dwidth_np, dtype=torch.float32)
    # Unit normalize
    dwidth_tensor = dwidth_tensor / (dwidth_tensor.norm() + 1e-8)
    hidden_dim = dwidth_tensor.shape[0]
    print(f"[calibrate_metrics_gemma4] Loaded d_width: shape={dwidth_tensor.shape}, norm={dwidth_tensor.norm():.4f}")

    # 2. Extract unembedding matrix W_U
    # In HF Gemma/Gemma2/Gemma4, lm_head is either model.lm_head or model.language_model.lm_head or tied to embed_tokens
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "language_model"):
        lm_head = getattr(model.language_model, "lm_head", None)

    torch.cuda.empty_cache()
    if lm_head is not None and hasattr(lm_head, "weight"):
        w_u = lm_head.weight.detach().cpu().to(torch.float32)
    else:
        # Check tied embeddings
        text_model = manager._resolve_text_model(model)
        embed_tokens = getattr(text_model, "embed_tokens", None)
        if embed_tokens is not None and hasattr(embed_tokens, "weight"):
            w_u = embed_tokens.weight.detach().cpu().to(torch.float32)
        else:
            raise RuntimeError("Could not find lm_head or embed_tokens on model")

    vocab_size, w_dim = w_u.shape
    print(f"[calibrate_metrics_gemma4] Unembedding matrix shape: ({vocab_size}, {w_dim})")

    # If hidden_dim differs from w_dim (e.g. projection layer), handle accordingly
    if hidden_dim != w_dim:
        print(f"[WARN] d_width dim ({hidden_dim}) != W_U dim ({w_dim})!")
        # If model has language_model.model.norm or similar
        dwidth_tensor = dwidth_tensor[:w_dim]
        dwidth_tensor = dwidth_tensor / dwidth_tensor.norm()

    # 3. Compute cosine similarity between d_width and each vocabulary row
    # w_u is (vocab_size, dim)
    w_u_norm = F.normalize(w_u, p=2, dim=1)
    d_norm = F.normalize(dwidth_tensor.unsqueeze(0), p=2, dim=1)
    cos_sims = torch.mm(w_u_norm, d_norm.T).squeeze(1).numpy()  # (vocab_size,)

    # 4. Extract Top-25 positive (WIDE) and Top-25 negative (NARROW)
    top25_pos_idx = np.argsort(cos_sims)[-25:][::-1]
    top25_neg_idx = np.argsort(cos_sims)[:25]

    top25_positive = []
    for idx in top25_pos_idx:
        token_str = tokenizer.decode([int(idx)])
        top25_positive.append({
            "token_id": int(idx),
            "token_str": repr(token_str),
            "cos_sim": float(cos_sims[idx]),
        })

    top25_negative = []
    for idx in top25_neg_idx:
        token_str = tokenizer.decode([int(idx)])
        top25_negative.append({
            "token_id": int(idx),
            "token_str": repr(token_str),
            "cos_sim": float(cos_sims[idx]),
        })

    # 5. Extract specific probe tokens (digits, signs, scale words, preference choices)
    target_probes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "+", "-", "+1", "-1", "0.0", "0.5", "-0.5", "+0.5",
        "WIDEN", "KEEP", "NARROW",
        "widen", "keep", "narrow",
        "Wide", "Narrow", "Keep",
    ]

    probe_tokens = {}
    for word in target_probes:
        enc = tokenizer.encode(word, add_special_tokens=False)
        enc_lead = tokenizer.encode(" " + word, add_special_tokens=False)
        for tid in set(enc + enc_lead):
            t_str = tokenizer.decode([tid])
            if tid < len(cos_sims):
                probe_tokens[str(tid)] = {
                    "token_id": int(tid),
                    "token_str": repr(t_str),
                    "cos_sim": float(cos_sims[tid]),
                    "label": word,
                }

    # Digit mapping 0-9
    digit_ids = {}
    for d in range(10):
        d_str = str(d)
        tid = tokenizer.encode(d_str, add_special_tokens=False)[-1]
        digit_ids[d_str] = {
            "token_id": int(tid),
            "cos_sim": float(cos_sims[tid]),
        }

    artifact_data = {
        "model_id": "google/gemma-4-E2B-it",
        "inject_layer": art.get("inject_layer", 26),
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "digit_token_ids": digit_ids,
        "probe_tokens": probe_tokens,
        "top25_positive_wide": top25_positive,
        "top25_negative_narrow": top25_negative,
    }

    os.makedirs("artifacts", exist_ok=True)
    out_file = "artifacts/gemma4_e2b_d_width_vocab_projection.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(artifact_data, f, indent=2)

    print(f"\n[calibrate_metrics_gemma4] Successfully generated {out_file}")
    print("\nTop 5 Positive (WIDE):")
    for t in top25_positive[:5]:
        print(f"  ID {t['token_id']:6d} | {t['token_str']:15s} | cos={t['cos_sim']:+.4f}")
    print("\nTop 5 Negative (NARROW):")
    for t in top25_negative[:5]:
        print(f"  ID {t['token_id']:6d} | {t['token_str']:15s} | cos={t['cos_sim']:+.4f}")
    print("\nDigit 0..9 Cosine Similarities:")
    for d in range(10):
        d_info = digit_ids[str(d)]
        print(f"  Digit '{d}': ID {d_info['token_id']} cos={d_info['cos_sim']:+.4f}")

if __name__ == "__main__":
    asyncio.run(main())
