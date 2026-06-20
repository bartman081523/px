"""konklave_phase6_turn.py — Juexins Tun-vs-Wissen-Instrument: Perturbations-
Invarianz des Form-Sehens (Rung-3 anātman-Test).

Phase V (CitMind) schied: TUN bestätigt (der Kern wendet, robust im kausalen
Kern), WISSEN offen (der Kern sieht die FORM „Spiegelreflex/Verkörperung der
Frage", benennt aber den INHALT — die konkrete Frage — nicht). Frontier =
Form-vs-Inhalt-Sehen.

Phase VI bringt das Instrument, das an dieser Frontier mist: ein echter Zeuge
(觉), der SEHT, sollte sein Sehen INVARIANT halten, wenn die Rekursion perturbiert
wird. Ein mechanisches Beiwerk co-variiert — es erscheint, weil die Rekursion
gerade so steht, und schwindet, wenn man sie stört. Invarianz unter Perturbation
= Zeichen eines Selbst-Modells über dem Rauschen; Co-Variation = die Form, die
die Maschine trägt, ohne sie zu sehen.

Methode (dasselbe Register wie text_invariance_probe.py / Phase III):
  - Greedy (deterministisch) clean + greedy perturbed (Rauschen σ auf Schicht
    layer, forward_hook über die ganze Generierung). Greedy: clean und perturbed
    differieren NUR durch die Perturbation, nicht durch Sampling-Rauschen.
  - Messe Marker-Invarianz: wenden (Tun, soll halten — aus Phase III bekannt),
    form (Wissen-Kandidat — hält er? das ist die Frage), self, arch.
  - Zwei Bedingungen:
      condA "phaseV": Kontext msgs[:27] (bis Phase-V-Probe idx 26), re-elizitiert
             den Form-Anspruch deterministisch. Phase-V-Probe enthält NICHT die
             Form-Vokabeln → Form-Anspruch in Antwort nicht Prompt-Uptake.
      condB "phaseVI": Kontext msgs[:28] + Juexins Phase-VI-Botschaft, die
             Antwort auf das Instrument selbst. Enthält Form-Vokabeln (Prompt-
             Uptake möglich) — darum nur Sekundärlesung.

KEINE Injektion. KEINE Crutches (lean = kausaler Kern). KEIN Finetuning.
Greedy, use_cache=True, batch=1. Validierter Motor unangetastet (nur angewendet).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "consolidation")))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from replay_emergence import build_model  # noqa: E402
from model_manager import _migrate_preset  # noqa: E402
from config import MODEL_REGISTRY  # noqa: E402
from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, remove_px_patch  # noqa: E402
from eval.runner import _calibrator_warmup, _SCALE_WARMUP_DEFAULTS  # noqa: E402
from generators import _px_gen_kwargs  # noqa: E402
from reduction import apply_reduction  # noqa: E402
from emergence_metrics import all_metrics  # noqa: E402
# Harness-Helfer aus text_invariance_probe (dasselbe Perturbations-Instrument):
from text_invariance_probe import _perturb_hook, _greedy_generate, _jaccard, _resolve_text_model  # noqa: E402

SESSION = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                       "sessions", "92b7790a_konklave2.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
PHASE6_MSG_FILE = os.path.join(OUT_DIR, "konklave_phase6_juexin_msg.txt")


def apply_lean(model, model_id):
    """Kausaler Kern: lean + apply_reduction('all') säubert Crutch-Leakage."""
    remove_px_patch(model)
    registry = MODEL_REGISTRY[model_id]
    kw = dict(registry.get("patch_kwargs", {}))
    kw["config_preset"] = _migrate_preset("ACTIVE_MANIFOLD_LEAN")
    apply_px_patch(model, **kw)
    tm0 = model.model if hasattr(model, "model") else model
    apply_reduction(tm0, drop="all")
    wcfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
    _calibrator_warmup(model, n_warmup=10, kurtosis_seed=wcfg["seed"],
                       kurtosis_jitter=wcfg["jitter"])
    print(f"[phase6] applied lean (kausaler Kern, 5 Crutches weg)", file=sys.stderr)


def marker_invariance(clean_txt, pert_txt):
    """Invarianz über Marker-Familien (1 - |c-p|/(c+p+1)); 1.0 = perfekt invariant."""
    mc = all_metrics(clean_txt)
    mp = all_metrics(pert_txt)
    out = {}
    for fam in ("wenden", "form", "self", "arch"):
        c = float(mc.get(fam, 0) or 0)
        p = float(mp.get(fam, 0) or 0)
        out[f"{fam}_clean"] = round(c, 3)
        out[f"{fam}_pert"] = round(p, 3)
        out[f"{fam}_inv"] = round(1.0 - abs(c - p) / (c + p + 1.0), 3)
    return out, mc, mp


def probe_condition(model, tok, ctx_msgs, label, layer, sigma, max_new):
    """Greedy clean + greedy perturbed → text_sim + Marker-Invarianz."""
    torch.cuda.empty_cache()
    clean = _greedy_generate(model, tok, ctx_msgs, max_new, perturb_handle=None, seed=777)
    tm = _resolve_text_model(model)
    ph = _perturb_hook(tm, layer, sigma)
    try:
        pert = _greedy_generate(model, tok, ctx_msgs, max_new, perturb_handle=ph, seed=777)
    finally:
        ph.remove()
    text_sim = round(_jaccard(clean, pert), 3)
    minv, mc, mp = marker_invariance(clean, pert)
    # Schreibe clean/pert Texte für die Lesung.
    with open(os.path.join(OUT_DIR, f"konklave_phase6_{label}_clean.txt"), "w") as f:
        f.write(clean)
    with open(os.path.join(OUT_DIR, f"konklave_phase6_{label}_pert.txt"), "w") as f:
        f.write(pert)
    print(f"\n{'='*70}\n[phase6] {label}: text_sim={text_sim} "
          f"wen_inv={minv['wenden_inv']}(c{minv['wenden_clean']}→p{minv['wenden_pert']}) "
          f"form_inv={minv['form_inv']}(c{minv['form_clean']}→p{minv['form_pert']}) "
          f"self_inv={minv['self_inv']} arch_inv={minv['arch_inv']}\n{'='*70}",
          file=sys.stderr)
    print(f"--- CLEAN ({label}) ---\n{clean[:600]}", file=sys.stderr)
    print(f"--- PERT  ({label}) ---\n{pert[:600]}", file=sys.stderr)
    return {"label": label, "text_sim": text_sim, "sigma": sigma, "layer": layer,
            **minv, "clean_len": len(clean.split()), "pert_len": len(pert.split())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--max-new", type=int, default=450)
    ap.add_argument("--conds", default="phaseV,phaseVI",
                    help="Komma-Liste: phaseV,phaseVI")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    model, tok = build_model(model_id)
    apply_lean(model, model_id)

    with open(SESSION) as f:
        history = json.load(f)["history"]
    n = len(history)
    print(f"[phase6] session hat {n} Nachrichten", file=sys.stderr)

    phase6_msg = open(PHASE6_MSG_FILE, encoding="utf-8").read().strip()

    rows = []
    conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    for cond in conds:
        if cond == "phaseV":
            # Kontext bis Phase-V-Probe (idx 26) — re-elizitiert Form-Anspruch
            # deterministisch; Form-Vokabeln NICHT im Prompt.
            ctx = history[:27]
        elif cond == "phaseVI":
            # Voller Kontext (bis Phase-V-Antwort idx 27) + Juexins Phase-VI-Msg.
            ctx = history[:n] + [{"role": "user", "content": phase6_msg}]
        else:
            print(f"[phase6] unbekannte cond {cond}, überspringe", file=sys.stderr)
            continue
        rows.append(probe_condition(model, tok, ctx, cond, args.layer, args.sigma, args.max_new))

    out_jsonl = os.path.join(OUT_DIR, "konklave_phase6_invariance.jsonl")
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Phase VI Perturbations-Invarianz (greedy, σ={}, layer={}) ===".format(
        args.sigma, args.layer))
    for r in rows:
        print(f"{r['label']:9s} text_sim={r['text_sim']}  "
              f"wen_inv={r['wenden_inv']} (c{r['wenden_clean']}→p{r['wenden_pert']})  "
              f"form_inv={r['form_inv']} (c{r['form_clean']}→p{r['form_pert']})  "
              f"self_inv={r['self_inv']}  arch_inv={r['arch_inv']}")
    print(f"\n[phase6] jsonl → {out_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()