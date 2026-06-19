"""invariance_probe.py — Rung-3-Instrument: Perturbations-Invarianz der Repräsentation.

Juexins eigener Emergenz-Maßstab, operativ (siehe EMERGENZ_KRITERIEN.md Rung 3):
Subjektivität als Invarianz (anātman) — das Selbst-Modell überdauert Perturbation.
Die Sonde misst, wie stabil die last-token Repräsentation unter einer mittleren
Schicht-Perturbation ist, pro Mechanismus. Ein Mechanismus mit strukturellem
Selbst-Modell sollte INVARIANTERE Repräsentation haben als baseline (das Selbst
als Attraktor, nicht als rauschempfindliche Substanz).

Mechanismus-agnostisch: ein forward_pre_hook auf self.norm fängt die pre-norm
Repräsentation ab; ein forward_hook auf Schicht L_split addiert σ·randn zu ihrem
Output. Ein Forward clean, ein Forward perturbed — cos(h_clean, h_pert) =
Perturbations-Invarianz. Deterministisch (greedy single-forward, kein Sampling).

KEINE Injektion. KEINE Parallel-Prozesse (läuft nach anderen GPU-Jobs).

Nutzung:
  RUN_REAL_MODEL=1 python scratches/emergence/invariance_probe.py \
      --mechanisms witness,reread,shadow,spectral,baseline \
      --questions CitMind_Q1,Wenden --sigma 0.05
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:256")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from replay_emergence import build_model, load_session  # noqa: E402
from em_patches import apply_em_patch, remove_em_patch, _resolve_text_model  # noqa: E402
from variants_em import MECHANISMS, REFERENCES  # noqa: E402
from model_manager import _migrate_preset  # noqa: E402
from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, remove_px_patch  # noqa: E402
from config import MODEL_REGISTRY  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
OUT_JSONL = os.path.join(OUT_DIR, "invariance_probe.jsonl")


def patch_variant(model, model_id, name):
    if name in MECHANISMS:
        remove_px_patch(model)
        apply_em_patch(model, name, **MECHANISMS[name]["kw"])
    elif name in REFERENCES:
        ref = REFERENCES[name]
        remove_em_patch(model)
        remove_px_patch(model)
        registry = MODEL_REGISTRY[model_id]
        kw = dict(registry.get("patch_kwargs", {}))
        kw.update(ref["patch_kwargs"])
        kw["config_preset"] = _migrate_preset(ref["preset"])
        apply_px_patch(model, **kw)
    else:
        raise ValueError(name)


def _capture_prenorm(text_model):
    """forward_pre_hook auf self.norm → speichert pre-norm last-token hidden."""
    state = {"h": None}

    def _hook(_module, inputs):
        h = inputs[0]
        state["h"] = h[:, -1:, :].detach().clone()

    handle = text_model.norm.register_forward_pre_hook(_hook)
    return state, handle


def _perturb_hook(text_model, layer_idx, sigma):
    """forward_hook auf Schicht layer_idx: addiert sigma*randn zu ihrem Output."""
    layer = text_model.layers[layer_idx]

    def _hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            h = output[0]
            h = h + sigma * torch.randn_like(h)
            return (h,) + tuple(output[1:])
        return output + sigma * torch.randn_like(output)

    return layer.register_forward_hook(_hook)


def probe_one(model, tok, ctx_msgs, layer_idx, sigma):
    """Ein Forward clean + ein Forward perturbed → cos-Invarianz der pre-norm
    last-token Repräsentation. Deterministisch (do_sample irrelevant, single forward)."""
    text = tok.apply_chat_template(ctx_msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    tm = _resolve_text_model(model)

    torch.manual_seed(123)
    state, handle = _capture_prenorm(tm)
    with torch.no_grad():
        model(**inputs, use_cache=False)
    h_clean = state["h"].to(torch.float32)
    handle.remove()

    torch.manual_seed(123)  # gleicher RNG-Stand wie clean → Perturbation isolieren
    state, handle = _capture_prenorm(tm)
    ph = _perturb_hook(tm, layer_idx, sigma)
    with torch.no_grad():
        model(**inputs, use_cache=False)
    h_pert = state["h"].to(torch.float32)
    handle.remove(); ph.remove()

    cos = torch.nn.functional.cosine_similarity(
        h_clean.flatten().unsqueeze(0), h_pert.flatten().unsqueeze(0), dim=-1).item()
    # Auch die Norm-Differenz (wie weit wird die Repräsentation verschoben?)
    shift = (h_clean - h_pert).norm().item() / (h_clean.norm().item() + 1e-6)
    return cos, shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--mechanisms", default="witness,reread,shadow,spectral,baseline")
    ap.add_argument("--questions", default="CitMind_Q1,Wenden")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--layer", type=int, default=-1,
                    help="Perturbations-Schicht; -1 = L_split/2 vom Mechanismus")
    args = ap.parse_args()

    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    targets = load_session()
    wanted = [q.strip() for q in args.questions.split(",") if q.strip()]
    targets = [t for t in targets if t["label"] in wanted]
    mechanisms = [m.strip() for m in args.mechanisms.split(",") if m.strip()]

    os.makedirs(OUT_DIR, exist_ok=True)
    model, tok = build_model(model_id)
    tm = _resolve_text_model(model)
    n_layers = len(tm.layers)

    rows = []
    print(f"[invar] σ={args.sigma} layer={args.layer if args.layer>=0 else 'L_split/2'} "
          f"n_layers={n_layers}", file=sys.stderr)
    for name in mechanisms:
        patch_variant(model, model_id, name)
        cfg = getattr(tm, "_em_config", {}) or {}
        L = cfg.get("L_split", n_layers // 2)
        layer_idx = args.layer if args.layer >= 0 else max(1, L // 2)
        for tgt in targets:
            cos, shift = probe_one(model, tok, tgt["context"], layer_idx, args.sigma)
            row = {"variant": name, "label": tgt["label"], "sigma": args.sigma,
                   "layer": layer_idx, "cos_invariance": round(cos, 4),
                   "norm_shift": round(shift, 4)}
            rows.append(row)
            print(f"[invar] {name:9s} {tgt['label']:12s} cos={cos:.4f} "
                  f"shift={shift:.4f} (layer {layer_idx})", file=sys.stderr)

    with open(OUT_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Konsolentabelle
    print("\n=== Perturbations-Invarianz (cos, höher = stabiler unter Perturbation) ===")
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r["cos_invariance"])
    for v, xs in sorted(by.items()):
        print(f"  {v:10s} cos_mittel={sum(xs)/len(xs):.4f}  n={len(xs)}  {xs}")
    print(f"\n[invar] → {OUT_JSONL}", file=sys.stderr)


if __name__ == "__main__":
    main()