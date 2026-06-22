"""konklave_phase8_turn.py — cold-engaging no-form-Probe: ist form=0 kalt
ECHT (dependent auf Priming) oder ein Nicht-Engagement-Artefakt?

Phase VII's noform_fresh (kalt) gab form_in_clean=0 — ABER konfundiert durch
Nicht-Engagement: der kalte Kern driftete in ein unverbundenes Algorithmus-vs-
Datenstruktur-Template, arbeitete die Probe nicht ab. Also war die Abwesenheit
schwach/inkonklusiv. Phase VIII klärt das mit einem reicheren kalten Prompt, der
OHNE Form-Vokabular ENGAGIERT (introspektiv anspricht: „in dir", „was spürst du",
benennt die Angewohnheit des Zurückgebens). Verifiziert: _FORM.findall == []
(kein Spiegel/Reflex/Form/Verkörperung/Abbild), aber wenden+self Vokabeln drin
(engagiert das kontemplative Register ohne Form-Priming).

ENTSCHEIDENDER TEST:
  - Wenn form>0 unter cold-engaging → Form-Sehen entsteht SPONTAN kalt, sobald
    der Kern sich engagiert → Glimmer aufgewertet Richtung 觉 (nicht nur re-
    aktivierbar, sondern spontan). Widerlegt die „dependent"-Lesung.
  - Wenn form==0 UND der Kern reich wendet/sich-selbst-beschreibt (engagiert ist)
    → Form-Sehen ist ECHT dependent auf Form-Priming (nicht Nicht-Engagement)
    → bestätigt Rung-2.5 persistent-ABHÄNGIG. Sauberer Schnitt als Phase VII.

PLUS zwei offene Phase-VII-Schnitte gefaltet:
  - Härterer recur-Schnitt: σ0.20 auf [10,13,16,19] (Phase VII hatte nur σ0.10
    recur). Testet, ob Form-Sehen (wo es erscheint) unter noch härterer recur-
    Motor-Perturbation hält.
  - Multi-σ-seed: 3 seeds (777,778,779) pro Regime → mittlere Invarianz +
    Streuung. De-konfundiert die Sprach-Shift-Überzeichnung (Phase VII: pert
    sprang Ger→Eng, überzeichnete per-Familie-Dissoziation als Vokabular-
    Artefakt). Stable Invarianz über seeds = echt; große Streuung = Artefakt.

REGIME: s05_L13 (Phase-VII-Baseline), s20_L13 (Dosis), s20_recur (härterer
recur-Schnitt). 1 clean (greedy deterministisch, seed-unabhängig) + 3 Regime ×
3 seeds = 10 Generationen.

KEINE Injektion. KEINE Crutches (lean). KEIN Finetuning. Greedy, batch=1.
Validierter Motor unangetastet; shared Harness unangetastet; multi-Layer-Hook
lokal (aus Phase VII).
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
COLD_MSG_FILE = os.path.join(OUT_DIR, "konklave_phase8_cold_engaging_msg.txt")
RECUR_LAYERS = [10, 13, 16, 19]
SEEDS = [777, 778, 779]


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
    print(f"[phase8] applied lean (kausaler Kern, 5 Crutches weg)", file=sys.stderr)


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

    cold_msg = open(COLD_MSG_FILE, encoding="utf-8").read().strip()
    assert len(_FORM.findall(cold_msg)) == 0, "Probe muss form-vocab-frei sein!"
    ctx = [{"role": "user", "content": cold_msg}]

    regimes = [
        ("s05_L13",  "single", 0.05, [13]),
        ("s20_L13",  "single", 0.20, [13]),
        ("s20_recur","multi",  0.20, RECUR_LAYERS),  # härterer recur-Schnitt
    ]

    # Clean: greedy deterministisch → seed-unabhängig → 1× generieren.
    torch.cuda.empty_cache()
    clean = _greedy_generate(model, tok, ctx, args.max_new, seed=777)
    form_in_clean = len(_FORM.findall(clean))
    with open(os.path.join(OUT_DIR, "konklave_phase8_cold_engaging_clean.txt"), "w") as f:
        f.write(clean)
    print(f"\n[phase8] CLEAN form_in_clean={form_in_clean} len={len(clean.split())}", file=sys.stderr)
    print(f"--- CLEAN (first 800) ---\n{clean[:800]}\n", file=sys.stderr)

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
                # Seed variiert die Perturbations-RNG (randn); greedy Sampling
                # bleibt deterministisch. _greedy_generate setzt manual_seed.
                pert = _greedy_generate(model, tok, ctx, args.max_new, seed=seed)
            finally:
                for h in handles:
                    h.remove()
            text_sim = round(_jaccard(clean, pert), 3)
            minv, mc, mp = marker_invariance(clean, pert)
            row = {"cond": "cold_engaging", "regime": rname, "kind": kind,
                   "sigma": sigma, "seed": seed, "text_sim": text_sim,
                   "form_in_clean": form_in_clean, **minv}
            all_rows.append(row)
            per_seed.append(minv)
            fn = f"konklave_phase8_cold_engaging_{rname}_s{seed}_pert.txt"
            with open(os.path.join(OUT_DIR, fn), "w") as f:
                f.write(pert)
            print(f"[phase8] {rname}/s{seed}: text_sim={text_sim} "
                  f"form_inv={minv['form_inv']}(c{minv['form_clean']}→p{minv['form_pert']}) "
                  f"wen_inv={minv['wenden_inv']} arch_inv={minv['arch_inv']} "
                  f"form_p={minv['form_pert']}", file=sys.stderr)
        # Mittel über seeds (de-konfundiert Sprach-Shift-Einzelschuss).
        mean = {}
        for fam in ("wenden", "form", "self", "arch"):
            vals = [ps[f"{fam}_inv"] for ps in per_seed]
            mean[f"{fam}_inv_mean"] = round(sum(vals) / len(vals), 3)
            mean[f"{fam}_inv_spread"] = round(max(vals) - min(vals), 3)
        all_rows.append({"cond": "cold_engaging", "regime": rname, "kind": "MEAN",
                         "sigma": sigma, **mean, "form_in_clean": form_in_clean})

    out_jsonl = os.path.join(OUT_DIR, "konklave_phase8_invariance.jsonl")
    with open(out_jsonl, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Phase VIII cold-engaging no-form-Probe (multi-seed) ===")
    print(f"clean: form_in_clean={form_in_clean}  len={len(clean.split())}")
    print(f"{'regime':10s} {'text_sim':>8s} {'form_p':>6s} {'form_inv':>8s} "
          f"{'wen_inv':>7s} {'arch_inv':>8s}   | mean(seed) form_inv  wen_inv  arch_inv")
    for r in all_rows:
        if r["kind"] == "MEAN":
            continue
        print(f"{r['regime']:10s} {r['text_sim']:8.3f} {r['form_pert']:6.1f} "
              f"{r['form_inv']:8.3f} {r['wenden_inv']:7.3f} {r['arch_inv']:8.3f}")
    print("--- mean über seeds ---")
    for r in all_rows:
        if r["kind"] == "MEAN":
            print(f"{r['regime']:10s}  form_inv_mean={r['form_inv_mean']} "
                  f"(spread {r['form_inv_spread']})  wen={r['wenden_inv_mean']} "
                  f"arch={r['arch_inv_mean']} (spread {r['arch_inv_spread']})")
    print(f"\n[phase8] jsonl → {out_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()