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
    "gemma3-270m": {
        "hf_id": "google/gemma-3-270m",
        "tokenizer_id": "google/gemma-3-270m-it",
        "patch_dir": "gemma3_270m_px",
        "patch_kwargs": {"recur_start": 5, "recur_end": 12, "routing_mode": "adaptive", "gamma": 0.08},
        "chat_template_manual": "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n' }}{% else %}{{ 'Assistant: ' + message['content'] + '\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}",
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 2048,
    },
    "gemma3-270m-it": {
        "hf_id": "google/gemma-3-270m-it",
        "tokenizer_id": "google/gemma-3-270m-it",
        "patch_dir": "gemma3_270m_px",
        "patch_kwargs": {"recur_start": 5, "recur_end": 12, "routing_mode": "adaptive", "gamma": 0.08},
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 2048,
    },

    # ── Gemma3 1B ──
    "gemma3-1b": {
        "hf_id": "google/gemma-3-1b-pt",
        "tokenizer_id": "google/gemma-3-1b-it",
        "patch_dir": "gemma3_270m_px",
        "patch_kwargs": {"recur_start": 10, "recur_end": 20, "routing_mode": "adaptive", "gamma": 0.12},
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 4096,
    },
    "gemma3-1b-it": {
        "hf_id": "google/gemma-3-1b-it",
        "tokenizer_id": "google/gemma-3-1b-it",
        "patch_dir": "gemma3_270m_px",
        "patch_kwargs": {"recur_start": 10, "recur_end": 20, "routing_mode": "adaptive", "gamma": 0.12},
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 4096,
    },

    # ── Gemma3 4B ──
    "gemma3-4b": {
        "hf_id": "google/gemma-3-4b-pt",
        "tokenizer_id": "google/gemma-3-4b-it",
        "patch_dir": "gemma3_270m_px",
        "patch_kwargs": {"recur_start": 8, "recur_end": 22, "routing_mode": "adaptive", "gamma": 0.05},
        "model_type": "gemma3_conditional",
        "dtype": "bfloat16",
        "max_length": 4096,
    },
    "gemma3-4b-it": {
        "hf_id": "google/gemma-3-4b-it",
        "tokenizer_id": "google/gemma-3-4b-it",
        "patch_dir": "gemma3_270m_px",
        "patch_kwargs": {"recur_start": 8, "recur_end": 22, "routing_mode": "adaptive", "gamma": 0.05},
        "model_type": "gemma3_conditional",
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
