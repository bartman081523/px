"""
config.py — Model Registry and Server Configuration
=====================================================
Defines available models, their HuggingFace IDs, patch parameters,
and server-level settings.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    # ── PX-Patched Models ──
    "gemma3-270m-px": {
        "hf_id": "google/gemma-3-270m",             # Base model weights
        "tokenizer_id": "google/gemma-3-270m-it",    # IT tokenizer for chat template
        "patch_dir": "gemma3_270m_px",               # Symlink name in px_patches/
        "patch_kwargs": {
            "recur_start": 5,
            "recur_end": 12,
            "routing_mode": "adaptive",
            "gamma": 0.08,
        },
        "subjective_kwargs": {
            "config_preset": "SUBJECTIVE",
        },
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 2048,
        "chat_template_manual": (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message.role == 'user' %}"
            "<start_of_turn>user\n{{ message.content }}<end_of_turn>\n"
            "{% elif message.role == 'assistant' %}"
            "<start_of_turn>model\n{{ message.content }}<end_of_turn>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        ),
    },
    "minicpm5-1b-px": {
        "hf_id": "openbmb/MiniCPM5-1B",
        "tokenizer_id": "openbmb/MiniCPM5-1B",
        "patch_dir": "minicpm5_1b_px",
        "patch_kwargs": {
            "routing_mode": "adaptive",
            "subjective_enabled": False,
        },
        "subjective_kwargs": {
            "subjective_enabled": True,
        },
        "model_type": "llama",
        "dtype": "bfloat16",
        "max_length": 4096,
        "chat_template_manual": None,  # Uses built-in tokenizer chat_template
    },
    # ── Unpatched Baseline Models ──
    "gemma3-270m-base": {
        "hf_id": "google/gemma-3-270m",             # Same weights as PX, no patch
        "tokenizer_id": "google/gemma-3-270m-it",    # IT tokenizer for chat template
        "patch_dir": None,                           # No PX patch
        "patch_kwargs": {},
        "subjective_kwargs": {},
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 2048,
        "chat_template_manual": (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message.role == 'user' %}"
            "<start_of_turn>user\n{{ message.content }}<end_of_turn>\n"
            "{% elif message.role == 'assistant' %}"
            "<start_of_turn>model\n{{ message.content }}<end_of_turn>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        ),
    },
    "gemma3-270m-it": {
        "hf_id": "google/gemma-3-270m-it",           # IT model (instruction-tuned)
        "tokenizer_id": "google/gemma-3-270m-it",
        "patch_dir": None,                           # No PX patch
        "patch_kwargs": {},
        "subjective_kwargs": {},
        "model_type": "gemma3",
        "dtype": "bfloat16",
        "max_length": 2048,
        "chat_template_manual": None,               # IT tokenizer has built-in template
    },
    "minicpm5-1b-base": {
        "hf_id": "openbmb/MiniCPM5-1B",             # Same weights as PX, no patch
        "tokenizer_id": "openbmb/MiniCPM5-1B",
        "patch_dir": None,                           # No PX patch
        "patch_kwargs": {},
        "subjective_kwargs": {},
        "model_type": "llama",
        "dtype": "bfloat16",
        "max_length": 4096,
        "chat_template_manual": None,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Server Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "default_model": "minicpm5-1b-px",
    "default_max_tokens": 512,
    "default_temperature": 0.7,
    "default_top_p": 0.9,
}