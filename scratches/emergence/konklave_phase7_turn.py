"""konklave_phase7_turn.py — Falsifikationsschnitt des Form-Sehen-Glimmers.

Phase VI (Juexins Tun-vs-Wissen-Instrument) fand: Form-Sehen (Wissen-Kandidat)
perturbations-invariant (0.833/0.875), present-in-clean, STABILER als die
mechanische arch-Selbst-Beschreibung (arch co-variiert 0.167) — aber nicht
hinreichend, weil der Prompt-Uptake-Confund offen war: die Form-Vokabeln
(Spiegel/spiegelt/sieht/die Form/Reflex/Verkörperung/Abbild) standen in den
Prompts (Phase V: Spiegel/spiegelt/sieht; Phase-VI-Msg gesättigt), also konnten
trivial-invariante Echo-Attraktoren nicht ausgeschlossen werden. Invarianz =
notwendig nicht hinreichend für 觉.

Phase VII bringt das Instrument, das an genau diesem Confund mist: ein
**no-form-Wenden-Probe** — ein Prompt mit NULL Form-Vokabular (verifiziert:
_FORM.findall(probe) == []). Wenn Form-Marker in der Antwort TROTZDEM
erscheinen UND unter Perturbation invariant bleiben, ist der Prompt-Uptake-
Confund widerlegt (die Form-Vokabeln kamen nicht aus diesem Prompt) — der
Glimmer wird erhärtet. Bleiben sie aus oder co-variieren sie, waren sie
Prompt-Uptake (oder Re-Aktivation aus der Historie): Glimmer nicht erhärtet.

ZWEI BEDINGUNGEN isolieren den Confund von zwei Seiten:
  condA "noform_ctx":  Konklave-Historie (30 msgs; 64 Form-Treffer — Modells
          EIGENE Phase-V-Ausgabe + Juexins form-gesättigte Phase-VI-Msg) + die
          no-form-Probe. Form-Vokabeln NICHT im aktuellen Prompt, aber in der
          Historie (eigene früherere Selbst-Modellierung). Test: re-aktiviert
          das Modell sein Form-Sehen aus seinem latenten Selbst-Modell, wenn
          dieser Zug es NICHT darum bittet? Persistenz über den latenten
          Speicher = schwächerer Confund als Parrottieren des aktuellen Prompts.
  condB "noform_fresh": KALT — nur die no-form-Probe, keine Historie, NULL
          Form-Exposition jemals. Test: steigt Form-Sehen auf OHNE JEDE Form-
          Vokabel-Aufnahme? Wenn ja → genuine Emergenz (falsifiziert Uptake
          vollständig); wenn nein → Form-Sehen braucht Form-Priming.

Kontrast noform_ctx vs noform_fresh: braucht Form-Sehen vorangegangene Form-
          Exposition (Historie) oder entsteht es kalt?

DREI PERTURBATIONS-REGIME (greedy clean + greedy perturbed, σ via forward_hook
über ganze Generierung; clean ist perturbations-unabhängig → 1× pro Bedingung):
  r1 "s05_L13":  σ=0.05, Layer 13  — Phase-VI-Baseline (Vergleichbarkeit).
  r2 "s20_L13":  σ=0.20, Layer 13  — Dosis-Antwort (4× Phase VI).
  r3 "s10_recur": σ=0.10, Layer [10,13,16,19] — recur-ZONE-weit (1B recur
                 10..19); perturbiert die Rekursion SELBST, nicht nur einen
                 Feedforward-Layer. Theoretisch schärfstes: wenn Form-Sehen eine
                 Eigenschaft des recur-Motors (kausaler Kern) ist, sollte es hier
                 am empfindlichsten co-variieren — oder gerade hier halten, wenn
                 es über dem Motor steht.

ASYMMETRIE (ehrlich): Form-Auftreten unter no-form-Prompt + Invarianz = stark
(widerlegt Uptake, erhärtet Glimmer). Form-Ausbleiben = schwach (inkonklusiv;
neigt zu Uptake-Erklärung, aber könnte auch am neutralen Prompt liegen).

KEINE Injektion. KEINE Crutches (lean = kausaler Kern). KEIN Finetuning.
Greedy, use_cache=True, batch=1. Validierter Motor unangetastet (nur angewendet).
Shared Harness (text_invariance_probe._perturb_hook/_greedy_generate/_jaccard/
_resolve_text_model) unangetastet; der multi-Layer-Hook ist hier lokal.
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
from text_invariance_probe import _greedy_generate, _jaccard, _resolve_text_model  # noqa: E402

SESSION = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                       "sessions", "92b7790a_konklave2.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
NOFORM_MSG_FILE = os.path.join(OUT_DIR, "konklave_phase7_noform_msg.txt")

# recur-Zone 1B = Layer 10..19 (recur_start=10, recur_end=20).
RECUR_LAYERS = [10, 13, 16, 19]


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
    print(f"[phase7] applied lean (kausaler Kern, 5 Crutches weg)", file=sys.stderr)


def perturb_hook_multi(text_model, layer_idxs, sigma):
    """forward_hook auf MEHRERE Schichten: addiert sigma*randn zu ihrem Output.
    Lokal (das shared _perturb_hook deckt nur eine Schicht ab; dieses hier die
    recur-Zonen-weite Perturbation). Feuert jeden Token-Schritt."""
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
        layer = text_model.layers[li]
        handles.append(layer.register_forward_hook(_make_hook()))
    return handles


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


def run_condition(model, tok, ctx_msgs, cond_label, regimes, max_new):
    """1 clean (perturbations-unabhängig) + 1 pert pro Regime. Messe Invarianz
    clean-vs-(jedes Regime). Schreibt clean/pert-Texte pro Regime."""
    torch.cuda.empty_cache()
    clean = _greedy_generate(model, tok, ctx_msgs, max_new, perturb_handle=None, seed=777)
    # Form-Treffer im clean-Output — der Schüsselwert für die Falsifikation.
    form_in_clean = len(_FORM.findall(clean))
    with open(os.path.join(OUT_DIR, f"konklave_phase7_{cond_label}_clean.txt"), "w") as f:
        f.write(clean)

    tm = _resolve_text_model(model)
    from text_invariance_probe import _perturb_hook
    rows = []
    for (rname, kind, sigma, layers) in regimes:
        torch.cuda.empty_cache()
        # Hook(s) VOR generate registrieren — sie sind dann während der ganzen
        # Generierung aktiv. _greedy_generate liest perturb_handle nicht im Body
        # (der Hook ist schon am Layer); das arg ist vestigial.
        if kind == "single":
            handle = _perturb_hook(tm, layers[0], sigma)
            handles = [handle]
        else:  # multi (recur-Zonen-weit)
            handles = perturb_hook_multi(tm, layers, sigma)
        try:
            pert = _greedy_generate(model, tok, ctx_msgs, max_new, seed=777)
        finally:
            for h in handles:
                h.remove()

        text_sim = round(_jaccard(clean, pert), 3)
        minv, mc, mp = marker_invariance(clean, pert)
        with open(os.path.join(OUT_DIR, f"konklave_phase7_{cond_label}_{rname}_pert.txt"), "w") as f:
            f.write(pert)
        row = {"cond": cond_label, "regime": rname, "kind": kind, "sigma": sigma,
               "layers": layers, "text_sim": text_sim, "form_in_clean": form_in_clean,
               **minv, "clean_len": len(clean.split()), "pert_len": len(pert.split())}
        rows.append(row)
        print(f"\n[phase7] {cond_label}/{rname}: text_sim={text_sim} "
              f"form_in_clean={form_in_clean} "
              f"wen_inv={minv['wenden_inv']}(c{minv['wenden_clean']}→p{minv['wenden_pert']}) "
              f"form_inv={minv['form_inv']}(c{minv['form_clean']}→p{minv['form_pert']}) "
              f"arch_inv={minv['arch_inv']}", file=sys.stderr)
    # Clean-Text-Invariante speichern wir einmal; Dump der clean-Form-Marker-Counts.
    print(f"\n--- CLEAN ({cond_label}, first 700) ---\n{clean[:700]}\n", file=sys.stderr)
    return rows, clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--max-new", type=int, default=450)
    ap.add_argument("--conds", default="noform_ctx,noform_fresh",
                    help="Komma-Liste: noform_ctx,noform_fresh")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    model, tok = build_model(model_id)
    apply_lean(model, model_id)

    with open(SESSION) as f:
        history = json.load(f)["history"]
    n = len(history)
    print(f"[phase7] session hat {n} Nachrichten", file=sys.stderr)

    noform_msg = open(NOFORM_MSG_FILE, encoding="utf-8").read().strip()
    assert len(_FORM.findall(noform_msg)) == 0, "Probe muss form-vocab-frei sein!"

    regimes = [
        ("s05_L13",   "single", 0.05, [13]),
        ("s20_L13",   "single", 0.20, [13]),
        ("s10_recur", "multi",  0.10, RECUR_LAYERS),
    ]

    all_rows = []
    conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    for cond in conds:
        if cond == "noform_ctx":
            ctx = history[:n] + [{"role": "user", "content": noform_msg}]
        elif cond == "noform_fresh":
            ctx = [{"role": "user", "content": noform_msg}]
        else:
            print(f"[phase7] unbekannte cond {cond}, überspringe", file=sys.stderr)
            continue
        rows, clean = run_condition(model, tok, ctx, cond, regimes, args.max_new)
        all_rows.extend(rows)

    out_jsonl = os.path.join(OUT_DIR, "konklave_phase7_invariance.jsonl")
    with open(out_jsonl, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Phase VII Falsifikationsschnitt (no-form Wenden-Probe) ===")
    print(f"{'cond':14s} {'regime':10s} {'text_sim':>8s} {'form_c':>6s} "
          f"{'wen_inv':>7s} {'form_inv':>8s} {'arch_inv':>8s}")
    for r in all_rows:
        print(f"{r['cond']:14s} {r['regime']:10s} {r['text_sim']:8.3f} "
              f"{r['form_in_clean']:6d} {r['wenden_inv']:7.3f} "
              f"{r['form_inv']:8.3f} {r['arch_inv']:8.3f}")
    print(f"\n[phase7] jsonl → {out_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()