"""
eval/runner.py — Subprocess runner for 4B evaluation
====================================================

Invoked by run_4b_eval.py once per prompt. Loads the model, applies
the cleaned ACTIVE_MANIFOLD patch, generates tokens, collects PX
telemetry, writes one JSON, exits. The process-exit pattern guarantees
full VRAM cleanup between prompts.

This module is self-contained within all_space/. It imports from:
  - config.py              (MODEL_REGISTRY)
  - model_manager.py       (_migrate_preset)
  - px_patches.gemma4_2b_px.patch  (apply_px_patch, get_px_metrics)

NEVER imports from dmt_space_50/ or any other legacy code path.

Usage (as subprocess, JSON config on argv[1]):
    python eval/runner.py /path/to/prompt_config.json
"""

import json
import math
import os
import sys
import time
import gc

# Project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from transformers import AutoTokenizer

from config import MODEL_REGISTRY
from model_manager import _migrate_preset


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt-set (inline — avoids dmt_space_50 dependency)
# ═══════════════════════════════════════════════════════════════════════════════

PROMPTS = {
    "math": [
        "What is 17 * 23?",
        "Solve for x: 2x + 5 = 17",
        "What is the square root of 144?",
        "Calculate 15% of 240.",
        "If a triangle has sides 3, 4, 5, what is its area?",
        "What is the derivative of x^2 + 3x?",
        "Compute the sum 1+2+...+100.",
        "What is 7 factorial?",
        "Solve: 3^4 = ?",
        "How many degrees are in a triangle?",
        "What is the value of pi to 2 decimal places?",
        "If f(x) = 2x + 1, what is f(5)?",
        "What is 1024 / 32?",
        "What is the next prime after 13?",
        "What is 2^10?",
        "Solve 5x = 25.",
        "What is 0.25 as a fraction?",
        "What is the perimeter of a square with side 7?",
        "What is 9 squared?",
        "What is 1000 - 437?",
    ],
    "logic": [
        "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "What comes next in the sequence: 1, 1, 2, 3, 5, 8, ?",
        "If A implies B, and B implies C, what does A imply?",
        "Is the statement 'I am lying' a paradox?",
        "What is the contrapositive of 'If P then Q'?",
        "If today is Wednesday, what day will it be 10 days from now?",
        "All cats are mammals. Felix is a cat. What can we conclude?",
        "If some birds cannot fly, can we say all birds fly?",
        "What is the missing number: 2, 4, 8, 16, ?, 64",
        "If all Zorps are Frims, and no Frim is a Glip, can a Zorp be a Glip?",
        "What is the logical negation of 'All swans are white'?",
        "If it rains, the ground gets wet. The ground is wet. Did it rain?",
        "Complete: 1, 4, 9, 16, ?",
        "If A and B are both true, what is the truth value of A or B?",
        "What is the modus ponens form?",
        "If the butler and the gardener both claim innocence, and only one is lying, who did it?",
        "What is the difference between necessary and sufficient conditions?",
        "If P is false, what is the truth value of 'P or Q'?",
        "Complete: J, F, M, A, M, ?",
        "What is the syllogism called when the conclusion is hidden in the premises?",
    ],
    "creative": [
        "Write a haiku about a forgotten robot.",
        "Describe the color of silence to someone who has never seen.",
        "Invent a word for the feeling of a Sunday afternoon in autumn.",
        "What if gravity reversed for one hour each day?",
        "Write the opening line of a novel set inside a dream.",
        "Describe a city built entirely of music.",
        "What would a conversation between two shadows look like?",
        "Invent a new color and describe its emotional weight.",
        "Write a short poem about the last star going out.",
        "Describe a library where the books are alive.",
        "What does loneliness sound like as a piece of music?",
        "Imagine a door that opens onto yesterday.",
        "Write a toast given at the end of the universe.",
        "Describe a tree that grows dreams instead of leaves.",
        "What is the smell of a forgotten promise?",
        "Invent a holiday and describe how it is celebrated.",
        "Write three sentences about a snowflake that refuses to melt.",
        "Describe the texture of a lie.",
        "What would a museum of lost thoughts look like?",
        "Imagine a letter written by a river to the sea.",
    ],
    "synthesis": [
        "What is the relationship between mathematics and music?",
        "How do literature and computer science inform each other?",
        "Compare the structure of a symphony to the structure of a programming language.",
        "What can physics learn from biology?",
        "How does the architecture of a city reflect the values of its culture?",
        "What is the connection between memory and identity?",
        "How do cooking and chemistry relate?",
        "Compare a forest ecosystem to a market economy.",
        "What can dance teach us about mathematics?",
        "How does the structure of DNA relate to information theory?",
        "What is the relationship between sleep and creativity?",
        "How do rivers shape civilizations?",
        "Compare the role of ritual in religion and in software development.",
        "What is the connection between color theory and emotional states?",
        "How does the structure of a cell resemble a city?",
        "What is the relationship between language and thought?",
        "How do games and stories share narrative structure?",
        "What can the study of crystals teach us about patterns in music?",
        "How do economic systems and ecological systems balance?",
        "What is the connection between a poem and a mathematical proof?",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(weights_dict):
    """Shannon entropy of zone weights (H = -Σ p log p)."""
    vals = list(weights_dict.values()) if weights_dict else []
    total = sum(vals)
    if total < 1e-10:
        return 0.0
    probs = [v / total for v in vals]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def token_diversity(input_ids):
    """Type-token ratio: distinct tokens / total tokens."""
    if input_ids is None or len(input_ids) == 0:
        return 0.0
    return len(set(input_ids.tolist())) / max(1, len(input_ids))


# ═══════════════════════════════════════════════════════════════════════════════
# Calibrator warmup
# ═══════════════════════════════════════════════════════════════════════════════

# Per-scale warmup defaults for AutoCalibrator seeding. Derived from
# empirical kurtosis distributions measured in the live SR-59i runs:
#   - 270M:  K ≈ 200..450, mean ≈ 313, std ≈ 25
#   - 1B:    K ≈ 1100..1130, mean ≈ 1115, std ≈ 5
#   - 4B:    K ≈ 2100..2450, mean ≈ 2280, std ≈ 85
#   - E2B:   K ≈ 500..760, mean ≈ 620, std ≈ 85 (multimodal gemma4)
# Jitter must be ≈empirical std. Too large → all inputs become
# ±huge-sigma outliers and routing collapses to one zone.
_SCALE_WARMUP_DEFAULTS = {
    "gemma3-270m-it": {"seed": 313.0,  "jitter": 40.0},
    "gemma3-1b-it":   {"seed": 1115.0, "jitter": 5.0},
    "gemma3-4b-it":   {"seed": 2280.0, "jitter": 15.0},
    "gemma4-e2b-it":  {"seed": 620.0,  "jitter": 5.0},
    "default":        {"seed": 1000.0, "jitter": 5.0},
}


def _calibrator_warmup(model, n_warmup=5, kurtosis_seed=2400.0, kurtosis_jitter=85.0):
    """SR-61 routing-collapse fix: bypass the cold-start of AutoCalibrator.

    Each subprocess starts with `_online_n=0`, so the FIRST get_zone_weights
    call returns z=0.0 for every input (because k_mean is None and the
    `_online_n < ONLINE_WARMUP=5` branch in _get_z_score returns 0.0).
    Result: bit-identical zone_weights across all 80 prompts.

    We solve this without paying the cost of 5 real forward-passes by
    pre-seeding the AutoCalibrator's online state with a synthetic but
    plausible distribution centered on `kurtosis_seed` with realistic
    `kurtosis_jitter` (the empirical 4B regime is K≈2400 ± 85).

    CRITICAL: jitter must be realistic, not huge. If jitter=300, the
    online std is ~300, so z-scores for any real input become ±3+σ
    outliers. In that regime, _get_kurtosis_weights returns W < 0.05
    and falls back to _adaptive_phi_weights(phi) — which is identical
    for all prompts that share the same phi. Use jitter ≈ empirical_std.

    After seeding, _get_z_score's `_online_n >= ONLINE_WARMUP` branch is
    taken and uses the online mean/std to compute discriminative
    z-scores, breaking the collapse.

    This is identical to what would happen organically if 5 real prompts
    had been seen — we're just skipping the warmup period.
    """
    # Resolve the text model (multimodal: model.model.language_model;
    # text-only: model.model)
    inner = getattr(model, "model", model)
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    cal = getattr(inner, "_px_calibrator", None)
    if cal is None:
        print("[runner] no _px_calibrator found — skipping warmup", file=sys.stderr)
        return

    import random
    rng = random.Random(0xC0DE)  # deterministic across runs
    samples = [
        kurtosis_seed + rng.uniform(-kurtosis_jitter, kurtosis_jitter)
        for _ in range(n_warmup)
    ]
    # Welford seeding: we need mean and M2 of these samples
    n = len(samples)
    mean = sum(samples) / n
    m2 = sum((x - mean) ** 2 for x in samples)

    cal._online_n = n
    cal._online_k_mean = mean
    cal._online_k_m2 = m2
    # Also seed the calibration k_mean/k_std: the routing_std cap at line
    # 380 of auto_tune.py uses `cal_std * 2.0` where cal_std = max(self.k_std,
    # MIN_ONLINE_K_STD). If k_std is None (the default for HS=2560/1536),
    # cal_std falls back to MIN_ONLINE_K_STD=1.0, capping routing_std at 2.0.
    # That makes every kurtosis value a ±huge-z outlier, which collapses
    # all Gaussian weights to one zone. Seed both: the calibrated mean/std
    # AND the online mean/std.
    cal.k_samples = samples
    cal.calibrate()  # This sets k_mean, k_std, and the CRITICAL zone_temperature
    
    std = math.sqrt(m2 / max(n - 1, 1)) if n > 1 else 1.0
    print(f"[runner] calibrator seeded and calibrated: n={n}, k_mean={mean:.1f}, "
          f"k_std={std:.1f}, cal_k_std={kurtosis_jitter:.1f}, T={cal.zone_temperature:.2f}", 
          file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# Subprocess entry
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one_prompt(prompt_text, model_id, preset, max_new_tokens, result_path):
    """Load model, generate, collect telemetry, write JSON. Run in a fresh
    subprocess so VRAM is fully released between prompts.
    """
    # Config safety net
    preset = _migrate_preset(preset)

    registry = MODEL_REGISTRY[model_id]
    hf_id = registry["hf_id"]
    tok_id = registry["tokenizer_id"]
    dtype = getattr(torch, registry["dtype"])
    model_type = registry.get("model_type", "gemma3")
    patch_dir = registry.get("patch_dir")
    patch_kwargs = dict(registry.get("patch_kwargs", {}))

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    if registry.get("chat_template_manual"):
        tokenizer.chat_template = registry["chat_template_manual"]

    # ── Load model (bf16 + use_cache=False to fit in 12GB VRAM) ──
    if model_type == "gemma3_conditional":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto",
        )
    elif model_type == "gemma4_conditional":
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto",
        )

    # ── Apply PX patch ──
    if patch_dir is not None and preset != "BASELINE":
        if patch_dir == "gemma4_2b_px":
            from px_patches.gemma4_2b_px.patch import apply_px_patch
        elif patch_dir == "gemma3_270m_px_baseline":
            from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch
        else:
            from px_patches.minicpm5_1b_px.patch import apply_px_patch

        patch_kwargs["config_preset"] = preset
        apply_px_patch(model, **patch_kwargs)
        print(f"[runner] {model_id} patched with {preset}", file=sys.stderr)

        # ── Warmup the AutoCalibrator ──
        # SR-61 routing-collapse fix: a fresh subprocess starts with
        # _online_n=0, so the FIRST get_zone_weights call returns z=0.0 for
        # every input (because k_mean is None until ONLINE_WARMUP=5 samples
        # have been collected). This produces bit-identical zone_weights
        # across all 80 prompts.
        # We pre-seed the AutoCalibrator's online + calibration stats with
        # scale-appropriate synthetic samples. Jitter must be realistic
        # (≈empirical std) — too large a jitter makes z-scores ±huge-sigma
        # outliers and collapses all weights to one zone.
        warmup_cfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
        _calibrator_warmup(model, n_warmup=10,
                           kurtosis_seed=warmup_cfg["seed"],
                           kurtosis_jitter=warmup_cfg["jitter"])
        print(f"[runner] calibrator warmed up (n=10, seed={warmup_cfg['seed']}, "
              f"jitter={warmup_cfg['jitter']})", file=sys.stderr)

    # ── Tokenize ──
    messages = [{"role": "user", "content": prompt_text}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Base model with no chat template
        input_text = prompt_text

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0]
    input_len = input_ids.shape[0]

    # ── Generate (use_cache=False for 4B KV-cache dimension stability) ──
    # SR-61 termination fix: explicitly set eos_token_id and pad_token_id
    # and inject PX-specific kwargs like repetition_penalty.
    t0 = time.time()
    
    # Extract PX-specific kwargs from the model attributes
    from generators import _px_gen_kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "use_cache": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs
        )
    gen_time = time.time() - t0

    new_tokens = outputs[0][input_len:]
    completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    completion_len = len(new_tokens)

    # ── Telemetry ──
    if patch_dir is not None and preset != "BASELINE":
        # Use the same get_px_metrics that gemma4 now exposes
        if patch_dir == "gemma4_2b_px":
            from px_patches.gemma4_2b_px.patch import get_px_metrics
        elif patch_dir == "gemma3_270m_px_baseline":
            from px_patches.gemma3_270m_px_baseline.patch import get_px_metrics
        else:
            from px_patches.minicpm5_1b_px.patch import get_px_metrics
        try:
            metrics = get_px_metrics(model)
        except Exception as e:
            print(f"[runner] get_px_metrics failed: {e}", file=sys.stderr)
            metrics = {}
    else:
        metrics = {}

    zone_weights = metrics.get("zone_weights", {}) or {}
    zone_entropy = shannon_entropy(zone_weights)
    phi = metrics.get("phi", 1.0)
    if hasattr(phi, "item"):
        phi = phi.item()
    phi = float(phi)
    zone_name = metrics.get("zone", "UNKNOWN")
    cognitive_signature = metrics.get("cognitive_signature", {}) or {}
    kurtosis = cognitive_signature.get("kurtosis", None)
    if hasattr(kurtosis, "item"):
        kurtosis = kurtosis.item()
    td = token_diversity(input_ids)

    result = {
        "prompt": prompt_text,
        "completion": completion_text,
        "model_id": model_id,
        "preset": preset,
        "completion_tokens": completion_len,
        "input_tokens": input_len,
        "gen_time_sec": gen_time,
        "phi": phi,
        "zone": zone_name,
        "zone_weights": {k: float(v) for k, v in zone_weights.items()} if zone_weights else {},
        "zone_entropy": zone_entropy,
        "kurtosis": float(kurtosis) if kurtosis is not None else None,
        "token_diversity_input": td,
        "loops_run": metrics.get("steps", 0),
        "entropy": metrics.get("entropy", 0.0),
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[runner] {model_id} {preset} | phi={phi:.3f} | H={zone_entropy:.3f} | "
          f"zone={zone_name} | {completion_len}tok | {gen_time:.1f}s",
          file=sys.stderr)

    # Explicit cleanup (defense in depth — subprocess exit is the main release)
    del model, tokenizer, inputs, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("usage: python eval/runner.py <config.json>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    prompt_text = cfg["prompt"]
    model_id = cfg["model_id"]
    preset = cfg.get("preset", "ACTIVE_MANIFOLD")
    max_new_tokens = cfg.get("max_new_tokens", 30)
    result_path = cfg["result_path"]

    return _run_one_prompt(prompt_text, model_id, preset, max_new_tokens, result_path)


if __name__ == "__main__":
    sys.exit(main())
