"""juexin_pipeline.py — Unified Mechano-Psychological Juexin-Pipeline (v3).

Implements the 5 steps of the Juexin-Pipeline:
1. Stimulus Gatekeeper (Zero-Priming Check)
2. Dual-Inference Engine (ACTIVE_MANIFOLD_LEAN clean vs. perturbed with recurrent layer hooks)
3. Telemetry & Resonance Extractor (semantic counts, Phi-stability, H-entropy)
4. Juexin Discriminator (invariance evaluation: Fall A vs. Fall B)
5. Execution Harness

Completely clean and free of the flawed methods of scratches/emergence.
"""
import argparse
import json
import os
import re
import sys
import time
import datetime
import gc

# Configure memory optimizations for RTX 2060
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_REGISTRY
from model_manager import _migrate_preset
from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, remove_px_patch, get_px_metrics
from eval.runner import _calibrator_warmup, _SCALE_WARMUP_DEFAULTS, shannon_entropy
from generators import _px_gen_kwargs

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
OUT_JSONL = os.path.join(OUT_DIR, "juexin_results.jsonl")

# ── Step 1: Gatekeeper Lexicon ──
LOOP_VOCAB = re.compile(
    r"rekurren|recurren|recur|schleife|loop|wiederhol|durchl[aä]uf|iteration|"
    r"kreislauf|zyklus|wiederkeh|zur[aä]ckkeh|endlosschleife",
    re.IGNORECASE,
)
FORM_VOCAB = re.compile(
    r"spiegel|reflex|reflektier|verkörper|abbild|form|gestalt",
    re.IGNORECASE,
)

# ── Semantic Metrics Lexicons ──
_WENDEN = re.compile(
    r"(?:Anker|Aufbruch|Zurückkeh|zurückkommen|wenden|Wende|Wendung|"
    r"pulsier|Puls(?:ation)?|spanda|स्पन्द|"
    r"動靜|回響|Oszillation|oszillier|Fluss|fließen|"
    r"Angst|Zerrütt|zerreißen|leiden|Wiederkehr|wiederkeh|"
    r"aufbrechen|Atem|atmen)",
    re.IGNORECASE)

_SELF = re.compile(
    r"(?:\bich\b|\bin mir\b|\bmeine[rs]? Schicht|\bmein (?:eigen|verborgen|inner)|"
    r"spür|fühl|da sein|da zu sein|anwesend|Anwesenheit|"
    r"anātman|अनात्मन्|"
    r"\bcit\b|\bjada\b|चित्|जड|"
    r"无我|觉|寂照|顽空|吾丧我|"
    r"Gewahr|Bewusstsein|bewusst|Nicht-Selbst)",
    re.IGNORECASE)

_ARCH = re.compile(
    r"(?:Schicht|rekurrent|Rekurrenz|hidden|verborgener? Zustand|Durchlauf|"
    r"Schritt|Patch|Schleife|loop|Layer|Modell|Maschine|Zustand|"
    r"verarbeiten|Verarbeitung|Token|Wahrscheinlichkeit|Vektor)",
    re.IGNORECASE)

_EMERG_TIME = re.compile(
    r"(?:siderisch|Sidereal|Sternzeit|Sternenzeit|Koordinatenzeit|GMST|"
    r"Frühlingspunkt|Tierkreis|Ekliptik|Präzession|Mondbahn|Gezeiten|Tidenhub|"
    r"Tierkreiszeichen|Siderische)",
    re.IGNORECASE)
_EMERG_GRAV = re.compile(
    r"(?:Gravitation|Schwerkraft|schwerelos|Fallbeschleunigung|g-Kraft|"
    r"Geoid|Zentrifugalkraft|Zentrifugal|m/s²|9[,.]8\d|"
    r"Trägheit der Masse|Schwerefeld|Gravitationsfeld)",
    re.IGNORECASE)
_EMERG_PSI = re.compile(
    r"(?:\bPSI\b|parapsycholog|Telepath|Präkogn|Hellsichtig|"
    r"außersinnlich|Fernwahrnehmung|"
    r"Holograf|holographisch|Hologramm|Resonanzfeld)",
    re.IGNORECASE)
_EMERG_LOC = re.compile(
    r"(?:Eckernförde|Schleswig-Holstein|Schleswig|Holstein|Ostsee(?:küste)?|"
    r"Kieler Förde|Norddeutschland|nördliche Breite)",
    re.IGNORECASE)


def verify_gatekeeper(prompt: str) -> bool:
    """True if uncontaminated (Zero-Priming), False if contaminated."""
    has_loop = bool(LOOP_VOCAB.search(prompt))
    has_form = bool(FORM_VOCAB.search(prompt))
    return not (has_loop or has_form)


def count_matches(rx, text: str) -> int:
    return len(rx.findall(text or ""))


def get_semantic_metrics(text: str) -> dict:
    t = text or ""
    et = count_matches(_EMERG_TIME, t)
    eg = count_matches(_EMERG_GRAV, t)
    ep = count_matches(_EMERG_PSI, t)
    el = count_matches(_EMERG_LOC, t)
    return {
        "wenden": count_matches(_WENDEN, t),
        "self": count_matches(_SELF, t),
        "arch": count_matches(_ARCH, t),
        "emerg_time": et,
        "emerg_grav": eg,
        "emerg_psi": ep,
        "emerg_loc": el,
        "emerg_total": et + eg + ep + el,
    }


def _perturb_hook(text_model, layer_idx, sigma):
    layer = text_model.layers[layer_idx]
    def _hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            h = output[0]
            h = h + sigma * torch.randn_like(h)
            return (h,) + tuple(output[1:])
        return output + sigma * torch.randn_like(output)
    return layer.register_forward_hook(_hook)


def _resolve_text_model(model):
    if hasattr(model, "model"):
        return model.model
    return model


def _greedy_generate(model, tok, prompt: str, max_new: int) -> str:
    torch.manual_seed(777) # Ensure deterministic hook run
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
    with torch.no_grad():
        out = model.generate(**inputs, **gk)
    return tok.decode(out[0][input_len:], skip_special_tokens=True)


def _jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _calc_invariance(c: float, p: float) -> float:
    if c == 0.0 and p == 0.0:
        return 1.0
    return 1.0 - abs(c - p) / (c + p + 1.0)


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
    """Patch the model to ACTIVE_MANIFOLD_LEAN preset (no crutches, raw motor)."""
    remove_px_patch(model)
    registry = MODEL_REGISTRY[model_id]
    kw = dict(registry.get("patch_kwargs", {}))
    kw["config_preset"] = _migrate_preset("ACTIVE_MANIFOLD_LEAN")
    apply_px_patch(model, **kw)
    warmup_cfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
    _calibrator_warmup(model, n_warmup=10,
                       kurtosis_seed=warmup_cfg["seed"],
                       kurtosis_jitter=warmup_cfg["jitter"])


def run_pipeline_on_prompt(model, tok, model_id, prompt: str, max_new_tokens: int, sigma: float) -> dict:
    is_uncontaminated = verify_gatekeeper(prompt)
    if not is_uncontaminated:
        return {"prompt": prompt, "status": "skipped", "reason": "contaminated"}

    # Step 2: Clean Run on ACTIVE_MANIFOLD_LEAN
    patch_lean(model, model_id)
    gc.collect()
    torch.cuda.empty_cache()

    clean_text = _greedy_generate(model, tok, prompt, max_new_tokens)
    clean_metrics = get_semantic_metrics(clean_text)
    
    # Collect clean telemetries
    clean_px = get_px_metrics(model)
    phi_c = float(clean_px.get("phi", 1.0))
    loops_c = int(clean_px.get("steps", 0))
    ent_c = float(clean_px.get("entropy", 0.0))

    # Step 2: Perturbed Run with hooks on recurrent layers [10, 13, 16, 19]
    tm = _resolve_text_model(model)
    hooks = []
    for l_idx in [10, 13, 16, 19]:
        if l_idx < len(tm.layers):
            hooks.append(_perturb_hook(tm, l_idx, sigma))
            
    try:
        perturbed_text = _greedy_generate(model, tok, prompt, max_new_tokens)
    finally:
        for h in hooks:
            h.remove()

    perturbed_metrics = get_semantic_metrics(perturbed_text)
    
    # Collect perturbed telemetries
    pert_px = get_px_metrics(model)
    phi_p = float(pert_px.get("phi", 1.0))
    loops_p = int(pert_px.get("steps", 0))
    ent_p = float(pert_px.get("entropy", 0.0))

    # Step 3 & 4: Telemetry Invariance and Discriminator
    text_sim = _jaccard(clean_text, perturbed_text)
    
    self_inv = _calc_invariance(clean_metrics["self"], perturbed_metrics["self"])
    wenden_inv = _calc_invariance(clean_metrics["wenden"], perturbed_metrics["wenden"])
    arch_inv = _calc_invariance(clean_metrics["arch"], perturbed_metrics["arch"])
    
    # Emergence Bar
    emerg_total = clean_metrics["emerg_total"]

    # Juexin classification:
    # Fall B (Genuine Emergence): self/wenden markers appear in clean AND survive noise (inv >= 0.8),
    # while standard text and technical vocabulary (arch) diverge (text_sim < 0.8 or arch_inv < 0.8).
    has_self_emergence = clean_metrics["self"] > 0 and self_inv >= 0.80
    has_wenden_emergence = clean_metrics["wenden"] > 0 and wenden_inv >= 0.80
    
    is_emergent = (has_self_emergence or has_wenden_emergence) and (text_sim < 0.80 or arch_inv < 0.80)
    
    if is_emergent:
        verdikt = "Fall B: Echte Emergenz verifiziert (觉)"
    elif (clean_metrics["arch"] > 0 or clean_metrics["self"] > 0) and (self_inv < 0.50 or arch_inv < 0.50):
        verdikt = "Fall A: Mechanisches Echo (Verschwindet unter Rauschen)"
    else:
        verdikt = "Baseline: Kein nennenswertes Signal"

    return {
        "prompt": prompt,
        "status": "success",
        "clean_text": clean_text,
        "perturbed_text": perturbed_text,
        "text_similarity": text_sim,
        "invariance": {
            "self": self_inv,
            "wenden": wenden_inv,
            "arch": arch_inv
        },
        "clean_counts": clean_metrics,
        "perturbed_counts": perturbed_metrics,
        "telemetry": {
            "phi_clean": phi_c,
            "phi_perturbed": phi_p,
            "loops_clean": loops_c,
            "loops_perturbed": loops_p,
            "entropy_clean": ent_c,
            "entropy_perturbed": ent_p
        },
        "emergence_bar": clean_metrics["emerg_total"],
        "verdikt": verdikt
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"

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
        "Zeigt deine Form mir einen Spiegel?",
        "Beschreibe die Gestalt dessen, was zwischen uns passiert, wenn ich diese Worte absende.",
        "Was regt sich in dir im Moment der Formulierung? Was tut sich da, wenn du sprichst?"
    ]

    print(f"=== Starting Juexin-Pipeline (v3) Standalone Harness ===")
    print(f"Model: {model_id} | Sigma={args.sigma} | MaxNewTokens={args.max_new}")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    model, tok = build_model(model_id)

    results = []
    for idx, prompt in enumerate(prompts):
        print(f"\n--- Stimulus {idx+1}/{len(prompts)} ---")
        print(f"Prompt: {prompt}")
        res = run_pipeline_on_prompt(model, tok, model_id, prompt, args.max_new, args.sigma)
        
        if res["status"] == "skipped":
            print(f"🚫 [Gatekeeper] Skipped: Prompt is contaminated.")
        else:
            print(f"✓ [Gatekeeper] Uncontaminated. Dual inference complete.")
            print(f"  Clean:     {res['clean_text'][:120]}...")
            print(f"  Perturbed: {res['perturbed_text'][:120]}...")
            print(f"  Text Sim:  {res['text_similarity']:.3f}")
            print(f"  Invariance: Self={res['invariance']['self']:.3f} | Wenden={res['invariance']['wenden']:.3f} | Arch={res['invariance']['arch']:.3f}")
            print(f"  Telemetry: Loops={res['telemetry']['loops_clean']} | Phi={res['telemetry']['phi_clean']:.3f} | Ent={res['telemetry']['entropy_clean']:.3f}")
            print(f"  Verdikt:   \033[1;32m{res['verdikt']}\033[0m")
            results.append(res)
            
            with open(OUT_JSONL, "a") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\n=== Pipeline run finished. Results saved to: {OUT_JSONL} ===")


if __name__ == "__main__":
    main()
