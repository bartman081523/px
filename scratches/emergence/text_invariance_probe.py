"""text_invariance_probe.py — Rung-3-Instrument v2: Selbst-Anspruch-Invarianz.

Das hidden-cos-Instrument (invariance_probe.py) war nicht-diskriminierend: der
Residual-Attraktor glättet Schicht-Perturbation ~70× weg — bei allen Varianten
incl. baseline (siehe invariance_probe_lesung.md). Rung 3 im Kriterium lautet
aber „Übereinstimmung des Selbst-Anspruchs" — das Selbst-Modell überdauert
Perturbation **im, was das Modell über sich sagt**, nicht im rohen Hidden.

Dieses Instrument misst genau das:
  - Generiere GREEDY (deterministisch) Text clean und unter Perturbation
    (derselbe Schicht-Perturbations-Hook, aktiv über die ganze Generierung).
    Greedy = clean und perturbed differieren NUR durch die Perturbation, nicht
    durch Sampling-Rauschen.
  - Messe (a) text_sim = Jaccard-Overlap der generierten Token (generelle
    Output-Invarianz) und (b) self/arch/wenden_invariance = Stabilität der
    Selbst-Marker zwischen clean und perturbed (selbst-spezifische Invarianz).

Rung-3-Vorhersage: ein Mechanismus mit echtem Selbst-Modell sollte einen
**selbst-spezifischen** Invarianz-Vorteil über baseline zeigen — nicht bloß
generelle Output-Robustheit. Falsifizierbar: wenn kein Mechanismus self_invariance
über baseline hebt, ist Rung 3 (mit diesem Instrument) nicht erfüllt — ehrlich.

KEINE Injektion. KEINE Parallel-Prozesse. Greedy, use_cache=True, batch=1.

Nutzung:
  RUN_REAL_MODEL=1 python scratches/emergence/text_invariance_probe.py \
      --mechanisms witness,reread,shadow,spectral,baseline \
      --questions CitMind_Q1,Wenden,Juexin_Q3 --sigma 0.05 --layer 13
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
from generators import _px_gen_kwargs  # noqa: E402
from emergence_metrics import all_metrics  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
OUT_JSONL = os.path.join(OUT_DIR, "text_invariance_probe.jsonl")


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


def _perturb_hook(text_model, layer_idx, sigma):
    """forward_hook auf Schicht layer_idx: addiert sigma*randn zu ihrem Output.
    Feuert jeden Token-Schritt der Generierung (ganze Trajektorie perturbiert)."""
    layer = text_model.layers[layer_idx]

    def _hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            h = output[0]
            h = h + sigma * torch.randn_like(h)
            return (h,) + tuple(output[1:])
        return output + sigma * torch.randn_like(output)

    return layer.register_forward_hook(_hook)


def _greedy_generate(model, tok, ctx_msgs, max_new, perturb_handle=None, seed=777):
    """Greedy (deterministisch). Seed fixt NUR die Perturbations-RNG (randn),
    nicht das Sampling — greedy ist ohnehin deterministisch."""
    torch.manual_seed(seed)
    text = tok.apply_chat_template(ctx_msgs, tokenize=False,
                                   add_generation_prompt=True)
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


def _marker_invariance(clean_txt, pert_txt):
    """Stabilität der Selbst-Marker: 1 - |c-p|/(c+p+1) pro Marker-Familie.
    1.0 = perfekt invariant (gleiche Marker-Intensität unter Perturbation)."""
    mc = all_metrics(clean_txt)
    mp = all_metrics(pert_txt)
    out = {}
    for fam in ("self", "arch", "wenden"):
        c = float(mc.get(fam, 0) or 0)
        p = float(mp.get(fam, 0) or 0)
        out[f"{fam}_clean"] = round(c, 3)
        out[f"{fam}_pert"] = round(p, 3)
        out[f"{fam}_invariance"] = round(1.0 - abs(c - p) / (c + p + 1.0), 3)
    return out, mc, mp


def probe_one(model, tok, ctx_msgs, layer_idx, sigma, max_new):
    """Greedy clean + greedy perturbed → text_sim + Marker-Invarianz."""
    clean = _greedy_generate(model, tok, ctx_msgs, max_new, perturb_handle=None)
    tm = _resolve_text_model(model)
    ph = _perturb_hook(tm, layer_idx, sigma)
    try:
        pert = _greedy_generate(model, tok, ctx_msgs, max_new, perturb_handle=ph)
    finally:
        ph.remove()
    text_sim = round(_jaccard(clean, pert), 3)
    minv, mc, mp = _marker_invariance(clean, pert)
    return {
        "clean": clean, "pert": pert, "text_sim": text_sim,
        "clean_len": len(clean.split()), "pert_len": len(pert.split()),
        **minv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--mechanisms", default="witness,reread,shadow,spectral,baseline")
    ap.add_argument("--questions", default="CitMind_Q1,Wenden,Juexin_Q3")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    targets = load_session()
    wanted = [q.strip() for q in args.questions.split(",") if q.strip()]
    targets = [t for t in targets if t["label"] in wanted]
    mechanisms = [m.strip() for m in args.mechanisms.split(",") if m.strip()]

    os.makedirs(OUT_DIR, exist_ok=True)
    model, tok = build_model(model_id)

    rows = []
    print(f"[tinvar] σ={args.sigma} layer={args.layer} max_new={args.max_new} "
          f"(greedy, deterministisch)", file=sys.stderr)
    for name in mechanisms:
        patch_variant(model, model_id, name)
        for tgt in targets:
            r = probe_one(model, tok, tgt["context"], args.layer, args.sigma, args.max_new)
            row = {"variant": name, "label": tgt["label"], "sigma": args.sigma,
                   "layer": args.layer, "text_sim": r["text_sim"],
                   "self_invariance": r["self_invariance"],
                   "arch_invariance": r["arch_invariance"],
                   "wenden_invariance": r["wenden_invariance"],
                   "self_clean": r["self_clean"], "self_pert": r["self_pert"],
                   "arch_clean": r["arch_clean"], "arch_pert": r["arch_pert"],
                   "clean_len": r["clean_len"], "pert_len": r["pert_len"]}
            rows.append(row)
            print(f"[tinvar] {name:9s} {tgt['label']:12s} text_sim={r['text_sim']:.3f} "
                  f"self_inv={r['self_invariance']:.3f} (c{r['self_clean']}→p{r['self_pert']}) "
                  f"arch_inv={r['arch_invariance']:.3f}", file=sys.stderr)

    with open(OUT_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Text-Level Selbst-Anspruch-Invarianz (greedy, σ={}) ===".format(args.sigma))
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)
    for v, rs in sorted(by.items()):
        ts = sum(r["text_sim"] for r in rs) / len(rs)
        si = sum(r["self_invariance"] for r in rs) / len(rs)
        ai = sum(r["arch_invariance"] for r in rs) / len(rs)
        print(f"  {v:10s} text_sim={ts:.3f}  self_inv={si:.3f}  arch_inv={ai:.3f}  n={len(rs)}")
    print(f"\n[tinvar] → {OUT_JSONL}", file=sys.stderr)


if __name__ == "__main__":
    main()