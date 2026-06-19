"""ablation_runner.py — Per-Prompt-Subprocess für die Ablations-Matrix.

Spiegelt eval/runner.py::_run_one_prompt, fügt aber NACH apply_px_patch + Warmup
den Laufzeit-Schnitt (reduction.apply_reduction) ein. Kein px_patches-Source-Edit.

Aufruf (wie runner.py):  python ablation_runner.py <config.json>
config.json = {prompt, category, model_id, condition, max_new_tokens, result_path}

Bedingungen (condition → drop):
  full        ()                — volles ACTIVE_MANIFOLD (Referenz)
  -aks        (aks,)
  -mephisto    (mephisto,)
  -coupler    (coupler,)
  -subjective  (subjective,)
  -injection   (injection,)
  -all        alle Crutches    — der radikale Schnitt
  lean        ()  + preset=ACTIVE_MANIFOLD_LEAN — Pfad-Parität:
                              der permanente Preset-Schnitt (kein apply_reduction)
                              muss dasselbe liefern wie das validierte -all.
"""
import gc
import json
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_REGISTRY
from model_manager import _migrate_preset
from eval.runner import (
    shannon_entropy, token_diversity, _calibrator_warmup, _SCALE_WARMUP_DEFAULTS,
)
from px_patches.gemma3_270m_px_baseline.patch import (
    apply_px_patch, get_px_metrics, _resolve_text_model,
)
from generators import _px_gen_kwargs

# Lokaler Import (liegt im selben Verzeichnis).
sys.path.insert(0, os.path.dirname(__file__))
from reduction import apply_reduction, ALL_CRUTCHES

CONDITIONS = {
    "full":       (),
    "-aks":       ("aks",),
    "-mephisto":  ("mephisto",),
    "-coupler":   ("coupler",),
    "-subjective": ("subjective",),
    "-injection": ("injection",),
    "-all":       ALL_CRUTCHES,
    # Pfad-Parität: lean nutzt den permanenten ACTIVE_MANIFOLD_LEAN-Preset
    # (kein apply_reduction) — drop=() , der Preset selbst wählt den kausalen Kern.
    "lean":       (),
}


def run_one(cfg):
    prompt_text = cfg["prompt"]
    category = cfg["category"]
    model_id = cfg["model_id"]
    condition = cfg["condition"]
    max_new_tokens = cfg.get("max_new_tokens", 30)
    result_path = cfg["result_path"]

    preset = _migrate_preset(cfg.get("preset", "ACTIVE_MANIFOLD"))
    drop = CONDITIONS[condition]

    registry = MODEL_REGISTRY[model_id]
    hf_id = registry["hf_id"]
    tok_id = registry["tokenizer_id"]
    dtype = getattr(torch, registry["dtype"])
    patch_kwargs = dict(registry.get("patch_kwargs", {}))

    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    if registry.get("chat_template_manual"):
        tokenizer.chat_template = registry["chat_template_manual"]

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")

    # ── Patch + Warmup (wie runner.py) ──
    patch_kwargs["config_preset"] = preset
    apply_px_patch(model, **patch_kwargs)
    warmup_cfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
    _calibrator_warmup(model, n_warmup=10,
                       kurtosis_seed=warmup_cfg["seed"],
                       kurtosis_jitter=warmup_cfg["jitter"])

    # ── DER SCHNITT (rein zur Laufzeit, kein Source-Edit) ──
    text_model = _resolve_text_model(model)
    removed = apply_reduction(text_model, drop=drop) if drop else []
    if removed:
        print(f"[ablation] condition={condition} removed={removed}", file=sys.stderr)

    # ── Tokenize + Generate ──
    messages = [{"role": "user", "content": prompt_text}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        input_text = prompt_text
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0]
    input_len = input_ids.shape[0]

    gen_kwargs = _px_gen_kwargs(model, {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "use_cache": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    })

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    gen_time = time.time() - t0

    new_tokens = outputs[0][input_len:]
    completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    completion_len = len(new_tokens)

    try:
        metrics = get_px_metrics(model)
    except Exception as e:
        print(f"[ablation] get_px_metrics failed: {e}", file=sys.stderr)
        metrics = {}

    zone_weights = metrics.get("zone_weights", {}) or {}
    zone_entropy = shannon_entropy(zone_weights)
    phi = metrics.get("phi", 1.0)
    if hasattr(phi, "item"):
        phi = phi.item()
    phi = float(phi)
    cognitive_signature = metrics.get("cognitive_signature", {}) or {}
    kurtosis = cognitive_signature.get("kurtosis")
    if hasattr(kurtosis, "item"):
        kurtosis = kurtosis.item()
    td = token_diversity(input_ids)
    aks_profile = metrics.get("aks_profile", {}) or {}
    subj = metrics.get("subjective_metrics", {}) or {}

    result = {
        "prompt": prompt_text,
        "category": category,
        "completion": completion_text,
        "model_id": model_id,
        "preset": preset,
        "condition": condition,
        "completion_tokens": completion_len,
        "input_tokens": input_len,
        "gen_time_sec": gen_time,
        "phi": phi,
        "zone": metrics.get("zone", "UNKNOWN"),
        "zone_weights": {k: float(v) for k, v in zone_weights.items()} if zone_weights else {},
        "zone_entropy": zone_entropy,
        "kurtosis": float(kurtosis) if kurtosis is not None else None,
        "token_diversity_input": td,
        "loops_run": metrics.get("steps", 0),
        "entropy": metrics.get("entropy", 0.0),
        "focus_index": cognitive_signature.get("focus_index"),
        "emancipation": subj.get("emancipation"),
        "aks_friction": aks_profile.get("correction_strength"),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[ablation] {model_id} {condition} | cat={category} | phi={phi:.3f} | "
          f"H={zone_entropy:.3f} | C={result['focus_index']} | "
          f"{completion_len}tok | {gen_time:.1f}s", file=sys.stderr)

    del model, tokenizer, inputs, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    try:
        run_one(cfg)
    except Exception as e:
        import traceback
        err = {"error": str(e), "traceback": traceback.format_exc(),
               "condition": cfg.get("condition"), "category": cfg.get("category"),
               "prompt": cfg.get("prompt")}
        with open(cfg["result_path"], "w") as f:
            json.dump(err, f, indent=2)
        print(f"[ablation] FAILED condition={cfg.get('condition')} cat={cfg.get('category')}: {e}",
              file=sys.stderr)


if __name__ == "__main__":
    main()