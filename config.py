"""
config.py — Model Registry and Server Configuration
=====================================================
Defines available models, their HuggingFace IDs, and default patch info.
Redundant patched variants removed; PX mode is now a dynamic parameter.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    # ── Gemma3 270M ──
    # NOTE: As of 2026-06-09 isolation, gemma3 models route to the
    # BYTE-IDENTICAL pre-gemma4 baseline patch directory. Do NOT point
    # these to gemma3_270m_px or gemma4_2b_px — that would re-introduce
    # the cross-contamination this isolation was meant to fix.
    "gemma3-270m": {
        "hf_id": "google/gemma-3-270m",
        "tokenizer_id": "google/gemma-3-270m-it",
        "patch_dir": "gemma3_270m_px_baseline",
        "patch_kwargs": {"recur_start": 5, "recur_end": 12, "routing_mode": "adaptive", "gamma": 0.08},
        "chat_template_manual": "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n' }}{% else %}{{ 'Assistant: ' + message['content'] + '\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}",
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 2048,
    },
    "gemma3-270m-it": {
        "hf_id": "google/gemma-3-270m-it",
        "tokenizer_id": "google/gemma-3-270m-it",
        "patch_dir": "gemma3_270m_px_baseline",
        "patch_kwargs": {"recur_start": 5, "recur_end": 12, "routing_mode": "adaptive", "gamma": 0.08},
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 2048,
    },

    # ── Gemma3 1B ──
    "gemma3-1b": {
        "hf_id": "google/gemma-3-1b-pt",
        "tokenizer_id": "google/gemma-3-1b-it",
        "patch_dir": "gemma3_270m_px_baseline",
        "patch_kwargs": {"recur_start": 10, "recur_end": 20, "routing_mode": "adaptive", "gamma": 0.12},
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 4096,
    },
    "gemma3-1b-it": {
        "hf_id": "google/gemma-3-1b-it",
        "tokenizer_id": "google/gemma-3-1b-it",
        "patch_dir": "gemma3_270m_px_baseline",
        "patch_kwargs": {"recur_start": 10, "recur_end": 20, "routing_mode": "adaptive", "gamma": 0.12},
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 4096,
    },

    # ── Gemma3 4B ──
    "gemma3-4b": {
        "hf_id": "google/gemma-3-4b-pt",
        "tokenizer_id": "google/gemma-3-4b-it",
        "patch_dir": "gemma3_270m_px_baseline",
        "patch_kwargs": {"recur_start": 8, "recur_end": 22, "routing_mode": "adaptive", "gamma": 0.05},
        "model_type": "gemma3_conditional",
        "dtype": "bfloat16",
        "max_length": 4096,
    },
    "gemma3-4b-it": {
        "hf_id": "google/gemma-3-4b-it",
        "tokenizer_id": "google/gemma-3-4b-it",
        "patch_dir": "gemma3_270m_px_baseline",
        "patch_kwargs": {"recur_start": 8, "recur_end": 22, "routing_mode": "adaptive", "gamma": 0.05},
        "model_type": "gemma3_conditional",
        "dtype": "bfloat16",
        "max_length": 4096,
        # Plan 1 Phase D: int8 by default to fit 4b on 12 GB GPUs for long
        # prefills. bf16 path OOM'd at the MLP layer (T=4800 needed >200 MB
        # headroom that bf16 weight memory didn't leave). int8 monkey-patches
        # every nn.Linear with QuantizedLinear (~50% weight-VRAM reduction).
        # Override per-request with `quantization="none"` if you need bf16.
        "quantization": "int8",
    },

    # ── Gemma4 E2B ──
    # Own isolated patch directory (gemma4_2b_px) with model_type-specific
    # behavior. Does NOT share patch code with gemma3 — the previous setup
    # had gemma4 reusing gemma3_270m_px with model_type conditionals, but
    # several of the token-loop mitigations and SCALE_DEFAULTS[1536] entries
    # leaked into gemma3 behavior. Isolating here keeps gemma3 bit-identical
    # to the pre-gemma4 baseline (5e46ed2).
    "gemma4-e2b-it": {
        "hf_id": "google/gemma-4-E2B-it",
        "tokenizer_id": "google/gemma-4-E2B-it",
        "patch_dir": "gemma4_2b_px",
        "patch_kwargs": {"routing_mode": "adaptive"},
        "model_type": "gemma4_conditional",
        "dtype": "bfloat16",
        "max_length": 4096,
    },

    # ── MiniCPM5 1B ──
    "minicpm5-1b": {
        "hf_id": "openbmb/MiniCPM5-1B",
        "tokenizer_id": "openbmb/MiniCPM5-1B",
        "patch_dir": "minicpm5_1b_px",
        "patch_kwargs": {"routing_mode": "adaptive"},
        "model_type": "llama",
        "dtype": "bfloat16",
        "max_length": 4096,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Server Configuration
# ═══════════════════════════════════════════════════════════════════════════════

import os

SERVER_CONFIG = {
    "host": os.environ.get("PX_HOST", "0.0.0.0"),
    "port": int(os.environ.get("PX_PORT", 7860)),
    "default_model": "gemma3-270m-it",
    "default_max_tokens": 512,
    "default_temperature": 0.7,
    "default_top_p": 0.9,
    "ssl_cert": os.environ.get("SSL_CERTFILE"),
    "ssl_key": os.environ.get("SSL_KEYFILE"),
}
