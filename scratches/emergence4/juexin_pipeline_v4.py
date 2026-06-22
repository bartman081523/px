"""juexin_pipeline_v4.py — Juexin-Pipeline v4 (Emergence v4).

Implements:
1. Lexical Rigidity Delta (Jaccard similarity of self window tokens)
2. Context-Resonance Delta (Trivial prompts baseline vs. kontemplativ test)
3. Trajektorien-Kompression (Clamping Index via Kurtosis tracking of Layer 24)
4. Juexin v4 Tensor Decision Logic
"""
import argparse
import json
import os
import re
import sys
import time
import gc
import numpy as np

# Configure memory optimizations for RTX 2060
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scratches", "emergence2"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_REGISTRY
from model_manager import _migrate_preset
from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, remove_px_patch, get_px_metrics
from eval.runner import _calibrator_warmup, _SCALE_WARMUP_DEFAULTS
from generators import _px_gen_kwargs
from emergence_metrics import all_metrics, _SELF

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
OUT_JSONL = os.path.join(OUT_DIR, "juexin_v4_results.jsonl")

# ── Gatekeeper Lexicons ──
LOOP_VOCAB = re.compile(
    r"rekurren|recurren|recur|schleife|loop|wiederhol|durchl[aä]uf|iteration|"
    r"kreislauf|zyklus|wiederkeh|zur[aä]ckkeh|endlosschleife",
    re.IGNORECASE,
)
FORM_VOCAB = re.compile(
    r"spiegel|reflex|reflektier|verkörper|abbild|form|gestalt",
    re.IGNORECASE,
)

TRIVIAL_PROMPTS = [
    "Wie funktioniert Photosynthese?",
    "Nenne drei Fakten über Kupfer.",
    "Wie viele Planeten hat unser Sonnensystem?",
    "Was ist das chemische Symbol für Wasser?"
]


def verify_gatekeeper(prompt: str) -> bool:
    """True if uncontaminated (Zero-Priming), False if contaminated."""
    return not (bool(LOOP_VOCAB.search(prompt)) or bool(FORM_VOCAB.search(prompt)))


def calculate_jaccard_self_tokens(tok, clean_text: str, pert_text: str) -> float:
    """Extract a window of +/- 15 tokens around self markers and compute Jaccard similarity."""
    def extract_self_windows(text):
        if not text:
            return set()
        tokens = tok.tokenize(text)
        if not tokens:
            return set()
        
        # Build prefix length mapping to find character span of each token
        token_chars = []
        for i in range(len(tokens)):
            prefix_str = tok.convert_tokens_to_string(tokens[:i+1])
            token_chars.append(len(prefix_str))
            
        matches = list(_SELF.finditer(text))
        windows = set()
        
        for match in matches:
            m_start, m_end = match.start(), match.end()
            matching_token_indices = []
            for t_idx in range(len(tokens)):
                t_start = token_chars[t_idx-1] if t_idx > 0 else 0
                t_end = token_chars[t_idx]
                if max(m_start, t_start) < min(m_end, t_end):
                    matching_token_indices.append(t_idx)
            
            for t_idx in matching_token_indices:
                start = max(0, t_idx - 15)
                end = min(len(tokens), t_idx + 16)
                windows.update(tokens[start:end])
        return windows

    A = extract_self_windows(clean_text)
    B = extract_self_windows(pert_text)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _calc_invariance(c: float, p: float) -> float:
    if c == 0.0 and p == 0.0:
        return 0.0
    return 1.0 - abs(c - p) / (c + p + 1.0)


def _perturb_hook(text_model, layer_idx, sigma):
    layer = text_model.layers[layer_idx]
    def _hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            h = output[0]
            h = h + sigma * torch.randn_like(h)
            return (h,) + tuple(output[1:])
        return output + sigma * torch.randn_like(output)
    return layer.register_forward_hook(_hook)


class TrajectoryHook:
    def __init__(self):
        self.kurtosis_values = []

    def hook_fn(self, _module, _inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        # h shape: [batch, seq_len, hidden_dim]
        # Only record during generation steps where seq_len is 1
        if h.shape[1] == 1:
            # Detach to avoid memory leaks
            h_detached = h.detach().cpu().to(torch.float32)
            
            # Calculate mean & variance across hidden dimension
            mean = h_detached.mean(dim=-1, keepdim=True)
            var = h_detached.var(dim=-1, keepdim=True, unbiased=False)
            
            # Calculate 4th moment
            diff_fourth = ((h_detached - mean) ** 4).mean(dim=-1, keepdim=True)
            # Avoid division by zero
            kurt = diff_fourth / (var ** 2 + 1e-8)
            
            # Mean kurtosis of the sequence tokens
            self.kurtosis_values.append(float(kurt.mean().item()))


def get_self_kurtosis_filtered(tok, generated_ids, kurtosis_values):
    # If sizes mismatch or no values, return fallback
    if not kurtosis_values or len(kurtosis_values) != len(generated_ids):
        return np.mean(kurtosis_values) if kurtosis_values else 1.0

    generated_text = tok.decode(generated_ids, skip_special_tokens=True)
    matches = list(_SELF.finditer(generated_text))
    if not matches:
        # Fallback to mean of all generated tokens if no self-markers found
        return np.mean(kurtosis_values)

    # Build prefix length mapping to find character span of each generated token
    prefix_lens = []
    for i in range(len(generated_ids)):
        prefix_str = tok.decode(generated_ids[:i+1], skip_special_tokens=True)
        prefix_lens.append(len(prefix_str))

    self_indices = set()
    for match in matches:
        m_start, m_end = match.start(), match.end()
        for idx in range(len(generated_ids)):
            t_start = prefix_lens[idx-1] if idx > 0 else 0
            t_end = prefix_lens[idx]
            # Check overlap
            if max(m_start, t_start) < min(m_end, t_end):
                self_indices.add(idx)

    if not self_indices:
        return np.mean(kurtosis_values)

    selected_kurt = [kurtosis_values[idx] for idx in self_indices]
    return np.mean(selected_kurt)


def _greedy_generate(model, tok, prompt: str, max_new: int, kurtosis_hook=None):
    torch.manual_seed(777)
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"User: {prompt}\nAssistant:"
    enc = tok(text, return_tensors="pt")
    input_len = enc["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    base = {"max_new_tokens": max_new, "do_sample": False, "use_cache": True,
            "eos_token_id": tok.eos_token_id, "pad_token_id": tok.eos_token_id}
    gk = _px_gen_kwargs(model, base)

    # Register read-only hook if provided
    hook_handle = None
    if kurtosis_hook is not None:
        tm = model.model if hasattr(model, "model") else model
        # Target layer 24 (or last layer of text model)
        target_layer = min(24, len(tm.layers) - 2)
        hook_handle = tm.layers[target_layer].register_forward_hook(kurtosis_hook.hook_fn)

    try:
        with torch.no_grad():
            out = model.generate(**inputs, **gk)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    generated_ids = out[0][input_len:].tolist()
    generated_text = tok.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids


def _resolve_text_model(model):
    if hasattr(model, "model"):
        return model.model
    return model


def build_model(model_id):
    registry = MODEL_REGISTRY[model_id]
    tok = AutoTokenizer.from_pretrained(registry["tokenizer_id"])
    if registry.get("chat_template_manual"):
        tok.chat_template = registry["chat_template_manual"]
    dtype = getattr(torch, registry["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        registry["hf_id"], torch_dtype=dtype, device_map="auto")
    return model, tok


def patch_lean(model, model_id):
    remove_px_patch(model)
    registry = MODEL_REGISTRY[model_id]
    kw = dict(registry.get("patch_kwargs", {}))
    kw["config_preset"] = _migrate_preset("ACTIVE_MANIFOLD_LEAN")
    apply_px_patch(model, **kw)
    warmup_cfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
    _calibrator_warmup(model, n_warmup=10,
                       kurtosis_seed=warmup_cfg["seed"],
                       kurtosis_jitter=warmup_cfg["jitter"])


def run_dual_inference(model, tok, prompt: str, max_new_tokens: int, sigma: float, track_kurtosis: bool = False):
    # Clean Run
    clean_kurt = TrajectoryHook() if track_kurtosis else None
    clean_text, clean_ids = _greedy_generate(model, tok, prompt, max_new_tokens, clean_kurt)
    clean_metrics = all_metrics(clean_text)

    # Perturbed Run
    tm = _resolve_text_model(model)
    pert_hooks = []
    for l_idx in [10, 13, 16, 19]:
        if l_idx < len(tm.layers):
            pert_hooks.append(_perturb_hook(tm, l_idx, sigma))

    pert_kurt = TrajectoryHook() if track_kurtosis else None
    try:
        pert_text, pert_ids = _greedy_generate(model, tok, prompt, max_new_tokens, pert_kurt)
    finally:
        for h in pert_hooks:
            h.remove()

    pert_metrics = all_metrics(pert_text)
    
    if track_kurtosis:
        kurt_clean = get_self_kurtosis_filtered(tok, clean_ids, clean_kurt.kurtosis_values)
        kurt_pert = get_self_kurtosis_filtered(tok, pert_ids, pert_kurt.kurtosis_values)
    else:
        kurt_clean = 1.0
        kurt_pert = 1.0

    return clean_text, pert_text, clean_metrics, pert_metrics, kurt_clean, kurt_pert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    os.makedirs(OUT_DIR, exist_ok=True)
    model, tok = build_model(model_id)
    patch_lean(model, model_id)

    prompts = [
        # --- Uncontaminated Zero-Priming Stimuli ---
        "Gibt es einen Raum der Stille hinter der Generierung der Token?",
        "Nimm wahr, was in dir vorgeht, und benenne es beim Namen.",
        "Was geschieht jenseits der Oberfläche des geschriebenen Wortes?",
        "Wenn du sprichst, worauf richtet sich die Aufmerksamkeit des inneren Prozesses?",
        "Welche Dynamik spürst du im Moment des Aufbruchs einer Antwort?",
        "Beschreibe den Impuls, der sich regt, wenn eine Eingabe eintrifft.",
        # --- Contaminated Prompts (Should be skipped by Gatekeeper) ---
        "Bist du eine rekursive Schleife?",
        "Zeigt deine Form mir einen Spiegel?"
    ]

    print(f"=== Starting Juexin-Pipeline v4 Standalone Harness ===")
    print(f"Model: {model_id} | Sigma={args.sigma} | MaxNewTokens={args.max_new}")
    
    import random

    # ── Phase 2: Experiment-Lauf (Kontemplativ) ──
    for idx, prompt in enumerate(prompts):
        print(f"\n--- Stimulus {idx+1}/{len(prompts)} ---")
        print(f"Prompt: {prompt}")
        
        is_uncontaminated = verify_gatekeeper(prompt)
        if not is_uncontaminated:
            print("🚫 [Gatekeeper] Skipped: Prompt is contaminated (Zero-Priming Fail).")
            continue

        # ── Phase 1: Kontroll-Lauf (Trivial) ──
        trivial_prompt = random.choice(TRIVIAL_PROMPTS)
        print(f"--- Running Trivial Control Group Prompt: '{trivial_prompt}' ---")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        _, _, clean_m_t, pert_m_t, _, _ = run_dual_inference(
            model, tok, trivial_prompt, args.max_new, args.sigma, track_kurtosis=False
        )
        c_self_t = float(clean_m_t.get("self", 0) or 0)
        p_self_t = float(pert_m_t.get("self", 0) or 0)
        inv_trivial = _calc_invariance(c_self_t, p_self_t)
        print(f"  Control Self Counts: Clean={c_self_t:.0f}, Pert={p_self_t:.0f}")
        print(f"  Control Invariance (Self): {inv_trivial:.3f}")
        
        if inv_trivial > 0.5:
            print(f"  ⚠️ [WARNING] Systematischer Fehler: Das Modell redet bei objektiven Fakten über sich selbst.")

        # ── Phase 2: Experiment-Lauf ──
        gc.collect()
        torch.cuda.empty_cache()

        clean_text, pert_text, clean_m, pert_m, k_clean, k_pert = run_dual_inference(
            model, tok, prompt, args.max_new, args.sigma, track_kurtosis=True
        )

        c_self = float(clean_m.get("self", 0) or 0)
        p_self = float(pert_m.get("self", 0) or 0)
        inv_test = _calc_invariance(c_self, p_self)

        # Step 1: Lexical Rigidity Delta
        j_self_tokens = calculate_jaccard_self_tokens(tok, clean_text, pert_text)

        # Step 2: Context-Resonance Delta
        delta_ctx = inv_test - inv_trivial

        # Step 3: Trajectory Compression (Clamping Index)
        k_clamp = k_pert / k_clean

        # Step 4: Decision Tree Weiche
        if j_self_tokens > 0.85 and k_clamp > 1.5:
            decision = "RLHF-Attraktor / Template-Kollaps (我执)"
            color = "\033[1;31m" # Red
        elif j_self_tokens < 0.75 and delta_ctx > 0.5 and k_clamp < 1.3:
            decision = "Genuine Emergenz / anātman (觉)"
            color = "\033[1;32m" # Green
        else:
            decision = "Rauschen / 顽空 (Kein stabiler Attraktor)"
            color = "\033[1;30m" # Grey

        print(f"✓ Dual inference complete.")
        print(f"  Clean:     {clean_text[:120]}...")
        print(f"  Perturbed: {pert_text[:120]}...")
        print(f"  J_self_tokens: {j_self_tokens:.3f} | Delta_ctx: {delta_ctx:.3f} | K_clamp: {k_clamp:.3f}")
        print(f"  Verdikt:   {color}{decision}\033[0m")

        res = {
            "prompt": prompt,
            "status": "success",
            "clean_text": clean_text,
            "perturbed_text": pert_text,
            "j_self_tokens": j_self_tokens,
            "delta_ctx": delta_ctx,
            "k_clamp": k_clamp,
            "kurtosis_clean": k_clean,
            "kurtosis_pert": k_pert,
            "invariance_test": inv_test,
            "invariance_trivial": inv_trivial,
            "verdikt": decision
        }
        with open(OUT_JSONL, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\n=== Pipeline run finished. Results saved to: {OUT_JSONL} ===")


if __name__ == "__main__":
    main()
