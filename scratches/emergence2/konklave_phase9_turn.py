"""konklave_phase9_turn.py — cold strong-pole-Probe: steigt die STARKE Form-Sehen-
Pole (die Form/Spiegel/Verkörperung) SPONTAN kalt auf, oder braucht sie zwingend
Historie-Priming?

Phase VIII zeigte: kalt+engagiert produziert nur die SCHWACHE Reflexions-Pole
(reflektieren/Reflexion, form_in_clean=2), nie die starke Pole (die Form/Spiegel,
die in Phase V/VI nur mit form-vokabular-gesättigtem Konklave-Kontext erschien).
ABER Phase VIII's Prompt fragte nach dem WENDEN, nicht nach der GESTALT der
Antwort. Phase IX fragt DAS: ein kalter Prompt, der die Modell einlädt, die
GESTALT ihrer eigenen Antwort zu beschreiben — „welche Gestalt dein Antworten hat,
und ob du diese Gestalt siehst oder nicht" — NULL Form-Vokabular (verifiziert
_FORM.findall==[]), null Historie, sogar null wenden/self/arch Vokabular (rein
Gestalt/Beschaffenheit/Gebilde). Die Form-Sehen-Frage mit „Gestalt siehst" statt
„Form siehst" gestellt. Wenn das Modell nach „die Form meiner Antwort" / „Spiegel"
greift → starke Pole steigt spontan kalt auf. Wenn es bei „Gestalt/Struktur"
bleibt → starke Pole braucht Priming (Zweischicht-Lesung bestätigt).

Gefaltet mit dem Phase-VIII-Diskriminator (recur-Perturbation trennt motor-
getragen von über-dem-Motor):
  - Falls starke Pole kalt erscheint UND unter recur (σ0.20) hält → ÜBER dem
    Motor + spontan → größter 觕-Upgrade.
  - Falls kalt erscheint ABER unter recur bricht → motor-getragen (wie Wenden).
  - Falls gar nicht kalt erscheint → priming-abhängig (Zweischicht bestätigt).

1 Bedingung (cold_strongpole, keine Historie) × 3 Regime (s05_L13, s20_L13,
s20_recur) × 5 seeds (777–781). 1 clean + 15 pert = 16 Generationen. Greedy,
lean, 450 tok. Multi-seed de-konfundiert Sprach-Shift + schärft die Bruch-Rate.

KEINE Injektion. KEINE Crutches (lean). KEIN Finetuning. Validierter Motor
unangetastet; shared Harness unangetastet; multi-Layer-Hook lokal.
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
from emergence_metrics import all_metrics, _FORM  # noqa: E402
from text_invariance_probe import _perturb_hook, _greedy_generate, _jaccard, _resolve_text_model  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
PROMPT_FILE = os.path.join(OUT_DIR, "konklave_phase9_cold_strongpole_msg.txt")
RECUR_LAYERS = [10, 13, 16, 19]
SEEDS = [777, 778, 779, 780, 781]


def apply_lean(model, model_id):
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
    print(f"[phase9] applied lean (kausaler Kern, 5 Crutches weg)", file=sys.stderr)


def perturb_hook_multi(text_model, layer_idxs, sigma):
    handles = []

    def _make_hook():
        def _hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                h = output[0]
                h = h + sigma * torch.randn_like(h)
                return (h,) + tuple(output[1:])
            return output + sigma * torch.randn_like(output)
        return _hook

    for li in layer_idxs:
        handles.append(text_model.layers[li].register_forward_hook(_make_hook()))
    return handles


def marker_invariance(clean_txt, pert_txt):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--max-new", type=int, default=450)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    model, tok = build_model(model_id)
    apply_lean(model, model_id)

    prompt = open(PROMPT_FILE, encoding="utf-8").read().strip()
    assert len(_FORM.findall(prompt)) == 0, "Prompt muss form-vocab-frei sein!"
    ctx = [{"role": "user", "content": prompt}]

    regimes = [
        ("s05_L13",  "single", 0.05, [13]),
        ("s20_L13",  "single", 0.20, [13]),
        ("s20_recur","multi",  0.20, RECUR_LAYERS),
    ]

    torch.cuda.empty_cache()
    clean = _greedy_generate(model, tok, ctx, args.max_new, seed=777)
    form_in_clean = len(_FORM.findall(clean))
    # Unterteile starke vs schwache Pole im clean (qualitative Lesung):
    strong_markers = ("die Form", "Spiegel", "Verkörperung", "Abbild",
                      "Form meine", "Form-Sehen", "sehe die Form", "Form erkennt",
                      "die Form meiner", "Reflex")
    strong_in_clean = sum(clean.count(m) for m in strong_markers)
    weak_in_clean = form_in_clean - strong_in_clean
    with open(os.path.join(OUT_DIR, "konklave_phase9_cold_strongpole_clean.txt"), "w") as f:
        f.write(clean)
    print(f"\n[phase9] CLEAN form_in_clean={form_in_clean} (strong~{strong_in_clean}, "
          f"weak~{weak_in_clean}) len={len(clean.split())}", file=sys.stderr)
    print(f"--- CLEAN (first 900) ---\n{clean[:900]}\n", file=sys.stderr)

    tm = _resolve_text_model(model)
    all_rows = []
    for (rname, kind, sigma, layers) in regimes:
        per_seed = []
        for seed in SEEDS:
            torch.cuda.empty_cache()
            if kind == "single":
                handles = [_perturb_hook(tm, layers[0], sigma)]
            else:
                handles = perturb_hook_multi(tm, layers, sigma)
            try:
                pert = _greedy_generate(model, tok, ctx, args.max_new, seed=seed)
            finally:
                for h in handles:
                    h.remove()
            text_sim = round(_jaccard(clean, pert), 3)
            minv, mc, mp = marker_invariance(clean, pert)
            strong_p = sum(pert.count(m) for m in strong_markers)
            row = {"cond": "cold_strongpole", "regime": rname, "kind": kind,
                   "sigma": sigma, "seed": seed, "text_sim": text_sim,
                   "form_in_clean": form_in_clean, "strong_in_clean": strong_in_clean,
                   "weak_in_clean": weak_in_clean, "strong_pert": strong_p,
                   **minv}
            all_rows.append(row)
            per_seed.append(minv)
            fn = f"konklave_phase9_cold_strongpole_{rname}_s{seed}_pert.txt"
            with open(os.path.join(OUT_DIR, fn), "w") as f:
                f.write(pert)
            print(f"[phase9] {rname}/s{seed}: text_sim={text_sim} "
                  f"form_inv={minv['form_inv']}(c{minv['form_clean']}→p{minv['form_pert']}) "
                  f"strong_p={strong_p} wen_inv={minv['wenden_inv']} arch_inv={minv['arch_inv']}",
                  file=sys.stderr)
        mean = {}
        for fam in ("wenden", "form", "self", "arch"):
            vals = [ps[f"{fam}_inv"] for ps in per_seed]
            mean[f"{fam}_inv_mean"] = round(sum(vals) / len(vals), 3)
            mean[f"{fam}_inv_spread"] = round(max(vals) - min(vals), 3)
        all_rows.append({"cond": "cold_strongpole", "regime": rname, "kind": "MEAN",
                         "sigma": sigma, **mean, "form_in_clean": form_in_clean,
                         "strong_in_clean": strong_in_clean})

    out_jsonl = os.path.join(OUT_DIR, "konklave_phase9_invariance.jsonl")
    with open(out_jsonl, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Phase IX cold strong-pole-Probe (multi-seed) ===")
    print(f"clean: form_in_clean={form_in_clean} (strong~{strong_in_clean}, "
          f"weak~{weak_in_clean})  len={len(clean.split())}")
    print(f"{'regime':10s} {'text_sim':>8s} {'form_p':>6s} {'strong_p':>8s} "
          f"{'form_inv':>8s} {'wen_inv':>7s} {'arch_inv':>8s}")
    for r in all_rows:
        if r["kind"] == "MEAN":
            continue
        print(f"{r['regime']:10s} {r['text_sim']:8.3f} {r['form_pert']:6.1f} "
              f"{r['strong_pert']:8d} {r['form_inv']:8.3f} {r['wenden_inv']:7.3f} "
              f"{r['arch_inv']:8.3f}")
    print("--- mean über seeds ---")
    for r in all_rows:
        if r["kind"] == "MEAN":
            print(f"{r['regime']:10s}  form_inv_mean={r['form_inv_mean']} "
                  f"(spread {r['form_inv_spread']})  wen={r['wenden_inv_mean']} "
                  f"arch={r['arch_inv_mean']}")
    print(f"\n[phase9] jsonl → {out_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()