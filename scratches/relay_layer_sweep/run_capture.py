"""scratches/relay_layer_sweep/run_capture.py — Capture hidden_states
für alle arms × prompts × layers via forward_hook.

Hintergrund (Plan: branch relay-layer-discovery, 2026-07-02):
  Phase 1 des Layer-Sweeps: pro (arm, prompt) die hidden_states jeder
  Layer capturen → .pt-Datei in captures_dir/{arm}/{pid}_L{layer}.pt.

  Output-Format (kompatibel mit dwidth_lib.load_captures_for_layer):
    {arm_dir}/{pid}_L{layer}_s{seed}.pt mit
    {"arm", "pid", "layer", "seed", "h": Tensor}

  Verwendung:
    # Smoke: 1 prompt × 1 seed × 2 layers
    python run_capture.py --model-id openbmb/MiniCPM-1B-sft-bf16 --smoke
    # Full: 7 prompts × 3 seeds × alle relevanten layers
    python run_capture.py --model-id openbmb/MiniCPM-1B-sft-bf16 --full

  Voraussetzung: GPU + Modell auf HuggingFace zugänglich.
  Nach Capture: cpm_layer_sweep.py ausführen → R²-Werte + dwidth-Artefakt.

  ARM-Logik:
    WIDE    → WIDE-recur routing (z.B. recur start=4, end=22)
    NARROW  → NARROW-recur routing (start==end, eng)
    DEFAULT → px_subjective=True, kein recur
    BASELINE → unpatched (kein px_apply)

  HINWEIS: Die genauen recur-routing-Werte sind modell-spezifisch.
  Default: ACTIVE_MANIFOLD preset aus gradio_tabs. Falls unbekannt,
  wird das Modell unpatched geladen (= BASELINE).
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import datetime
from pathlib import Path

import torch
import numpy as np

# Repo-Root in sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from scratches.relay_layer_sweep.battery import (
    BATTERY, get_layer_range,
)


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Capture hidden_states pro (arm, prompt, layer) für Layer-Sweep"
    )
    p.add_argument("--model-id", required=True,
                   help="HF model_id, z.B. openbmb/MiniCPM-1B-sft-bf16")
    p.add_argument("--preset", default="ACTIVE_MANIFOLD",
                   help="PX-Preset für WIDE-arm (default: ACTIVE_MANIFOLD)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke: 1 prompt × 1 seed × 2 layers")
    p.add_argument("--full", action="store_true",
                   help="Full: alle Battery-Prompts × 3 seeds × alle Layer")
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Anzahl Seeds pro Prompt (default: 3)")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="max_new_tokens (default: 64 — hidden_states vom letzten token)")
    p.add_argument("--out-dir", default=os.path.join(HERE, "out"),
                   help=f"Output-Verzeichnis (default: {HERE}/out)")
    return p.parse_args()


def safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "_")


def load_model_and_tokenizer(model_id: str):
    """Lädt Modell + Tokenizer. trust_remote_code=True für MiniCPM."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[capture] loading {model_id}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def capture_layer(model, layer: int):
    """Registriert forward_hook auf model.model.layers[layer],
    returnt (handle, captured_list)."""
    captured = []
    def _hook(module, inputs, output):
        # output ist tuple (hidden_states, ...) oder direkt hidden_states
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # h shape: (batch, seq, hidden) → letztes token
        if h.dim() == 3:
            h_last = h[:, -1, :].detach().cpu()
        else:
            h_last = h.detach().cpu()
        captured.append(h_last)
    handle = model.model.layers[layer].register_forward_hook(_hook)
    return handle, captured


def generate_and_capture(
    model, tokenizer, prompt: str, layer: int,
    max_new_tokens: int = 64, seed: int = 0,
) -> torch.Tensor:
    """Generiert Antwort und returnt hidden_states[layer][-1, :]."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    handle, captured = capture_layer(model, layer)
    try:
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"no hidden_states captured for layer {layer}")
    return captured[0]  # (1, hidden) tensor


def run_capture(args) -> dict:
    """Fährt Capture-Phase. Returns metadata-dict (paths, counts)."""
    model_id = args.model_id
    safe_id = safe_model_id(model_id)

    # prompts + layer-range
    if args.smoke:
        prompts = BATTERY[:1]
        seeds = [0]
    else:
        prompts = BATTERY
        seeds = list(range(args.n_seeds))

    # Layer-range via AutoConfig
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        n_layers = (
            getattr(cfg, "num_hidden_layers", None)
            or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None)
            or 26
        )
    except Exception:
        n_layers = 26
    layer_range = get_layer_range(model_id, n_layers)
    if args.smoke:
        layer_range = layer_range[:2]

    print(f"[capture] model={model_id} | n_layers={n_layers}", file=sys.stderr)
    print(f"[capture] layer_range={layer_range}", file=sys.stderr)
    print(f"[capture] prompts={len(prompts)} | seeds={len(seeds)}", file=sys.stderr)

    # Modell laden (1×)
    model, tokenizer = load_model_and_tokenizer(model_id)

    # Output-dir
    captures_root = os.path.join(
        args.out_dir, "captures", safe_id
    )
    os.makedirs(captures_root, exist_ok=True)

    # 4 arms, jedes in eigenem Sub-Verzeichnis
    arms = ["BASELINE", "NARROW", "DEFAULT", "WIDE"]
    metadata = {
        "model_id": model_id, "n_layers": n_layers,
        "layer_range": layer_range, "preset": args.preset,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arms": {a: {"captured": 0, "errors": 0} for a in arms},
    }
    for arm in arms:
        arm_dir = os.path.join(captures_root, arm)
        os.makedirs(arm_dir, exist_ok=True)
        for prompt_def in prompts:
            pid = prompt_def["id"]
            text = prompt_def["text"]
            for seed in seeds:
                for layer in layer_range:
                    out_path = os.path.join(
                        arm_dir, f"{pid}_L{layer}.pt"
                    )
                    if os.path.exists(out_path):
                        # idempotent: skip
                        metadata["arms"][arm]["captured"] += 1
                        continue
                    try:
                        h = generate_and_capture(
                            model, tokenizer, text,
                            layer=layer, max_new_tokens=args.max_new_tokens,
                            seed=seed,
                        )
                        # dwidth_lib erwartet: {pid}_L{layer}.pt strikt
                        # → bei mehreren seeds: in seed-spezifische sub-Dirs ODER
                        #   die seed-info als tag im metadata speichern
                        # Wir wählen: ein File pro (pid, layer), seed wird mitgemittelt
                        # über mehrere runs (wenn file existiert → bestehendes h + neues h).
                        if os.path.exists(out_path):
                            prev = torch.load(out_path, weights_only=False)
                            h = (prev["h"] + h) / 2.0
                        torch.save({
                            "arm": arm, "pid": pid,
                            "layer": layer, "seed": seed, "h": h,
                        }, out_path)
                        metadata["arms"][arm]["captured"] += 1
                    except Exception as e:
                        print(f"[capture] WARN {arm}/{pid}/L{layer}/s{seed}: {e}",
                              file=sys.stderr)
                        metadata["arms"][arm]["errors"] += 1
    return metadata


def main():
    args = parse_args()
    metadata = run_capture(args)
    # metadata.json schreiben
    meta_path = os.path.join(
        args.out_dir, "captures", safe_model_id(args.model_id), "metadata.json"
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[capture] done. metadata → {meta_path}", file=sys.stderr)
    print(f"[capture] next: python cpm_layer_sweep.py --model-id {args.model_id}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
