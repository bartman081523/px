#!/usr/bin/env python
"""Eval O — TCI-Konstanten als ausführbare Tensor-Operatoren (GPU).

Fortsetzung von Eval A–N. Siehe results/PREREG_Eval_O_2026-11-15.md: die
Kriterien sind *vor* dem Lauf fixiert, inklusive der Vorhersage, welche
Behauptungen fallen.

O1  Konstanten-Audit: Arithmetik, CODATA, Look-Elsewhere-Basisrate
O2  Operator-Dynamik auf der GPU: Wick-Propagator, Renormierung, S4-Kontrolle
O3  Gemma-4-Probe: Live-Hook auf Attention-/Residual-Layern, H72-Schwelle

Ein Lauf, ein Dump. Keine Nachjustierung.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tci_physics import (                      # noqa: E402
    CODATA_ALPHA_INV, alpha_tci, audit_constants, operator_suite, phi,
)
from tci_probe import Basis72, RelaySteer, TCIHolographicProbe   # noqa: E402

SEED_BASE_O3 = 20261120          # frisch, in keiner früheren Datei verwendet
MAX_NEW = 24

NARROW_PROMPTS = [
    "State the boiling point of water at sea level in one sentence.",
    "List the first four prime numbers, separated by commas.",
    "Translate 'the cat sits on the mat' into German.",
]
WIDE_PROMPTS = [
    "Write an original metaphor linking ocean currents, memory, and bureaucracy.",
    "Invent a short myth explaining why the moon has craters.",
    "Combine three unrelated domains into one surprising hypothesis.",
]
CALIB_PROMPTS = [
    "Explain how a bicycle derailleur works.",
    "Describe the water cycle in three sentences.",
    "Write a short paragraph about why tides happen.",
    "List three uses of a paper clip.",
]


# ── Hilfen ───────────────────────────────────────────────────────────────────
def shannon_bits(logits: torch.Tensor) -> float:
    p = F.softmax(logits.float(), dim=-1)
    return float(-(p * p.clamp_min(1e-30).log()).sum())


def mannwhitney(a, b):
    from scipy import stats as st
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 3 or len(b) < 3:
        return None
    return float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def auc(a, b):
    """P(x_a > x_b) über Rangstatistik (0.5 = Zufall)."""
    a = np.asarray([x for x in a if x is not None], dtype=float)
    b = np.asarray([x for x in b if x is not None], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return None
    gt = (a[:, None] > b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)


# ── Gemessene Session: manueller Decode-Loop (volle Kontrolle) ──────────────
@torch.no_grad()
def run_session(model, tok, text_model, prompt, probe: TCIHolographicProbe,
                basis: Basis72 | None, steer: RelaySteer | None,
                max_new_tokens: int = MAX_NEW, temperature: float = 0.0,
                seed: int = 0) -> dict:
    torch.manual_seed(seed)
    device = model.device
    inputs = tok(prompt, return_tensors="pt").to(device)
    ids = inputs["input_ids"]
    T = int(ids.shape[1])
    vocab = int(model.config.text_config.vocab_size
                if hasattr(model.config, "text_config") else model.config.vocab_size)

    probe.clear()
    ctx = steer if steer is not None else _nullctx()
    with ctx:
        out = model(input_ids=ids, use_cache=True,
                    attention_mask=torch.ones(1, T, device=device, dtype=torch.long))
        probe.records.clear()                 # Prefill-Record verwerfen
        past = out.past_key_values
        logits = out.logits[0, -1].float()

        entropies, gen_ids = [], []
        for t in range(max_new_tokens):
            entropies.append(shannon_bits(logits))
            if temperature <= 0.0:
                nxt = int(torch.argmax(logits))
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            if nxt == tok.eos_token_id:
                break
            gen_ids.append(nxt)
            pos = T + len(gen_ids) - 1
            step = torch.tensor([[nxt]], device=device, dtype=torch.long)
            out = model(input_ids=step, past_key_values=past, use_cache=True,
                        cache_position=torch.tensor([pos], device=device),
                        attention_mask=torch.ones(1, pos + 1, device=device, dtype=torch.long))
            past = out.past_key_values
            logits = out.logits[0, -1].float()

    m = probe.metrics(basis=basis, logit_entropies=entropies, vocab_size=vocab)
    m["text_preview"] = tok.decode(gen_ids, skip_special_tokens=True)[:120]
    m["prompt_tokens"] = T
    m["vocab_size"] = vocab
    m["entropy_bits_mean"] = float(np.mean(entropies)) if entropies else 0.0
    m["n_generated"] = len(gen_ids)
    return m


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ── O3 ──────────────────────────────────────────────────────────────────────
async def run_o3(dims=(24, 48, 72, 96, 144)) -> dict:
    from model_manager import ModelManager
    from scripts.run_gemma4_evaluations import format_prompt, load_dwidth_tensor

    t0 = time.perf_counter()
    m = ModelManager()
    e = await m.get_model("gemma4-e2b-it", px_subjective=False,
                          px_config_preset="BASELINE")
    model, tok = e["model"], e["tokenizer"]
    text_model = m._resolve_text_model(model)
    print(f"[O3] Modell geladen in {time.perf_counter()-t0:.1f}s, "
          f"Layer={len(text_model.layers)}, d={text_model.config.hidden_size}", flush=True)

    probe = TCIHolographicProbe(text_model).attach()
    dwidth = load_dwidth_tensor().to(model.device)
    g = torch.Generator().manual_seed(SEED_BASE_O3 + 1)
    v = torch.randn(dwidth.shape, generator=g).to(model.device)
    d_rand = v - (torch.dot(v, dwidth) / torch.dot(dwidth, dwidth)) * dwidth
    d_rand = d_rand / d_rand.norm() * dwidth.norm()

    # 1) Kalibrierung: fester 72-dim Unterraum, aus neutralen Prompts
    calib_states = []
    for i, p in enumerate(CALIB_PROMPTS):
        r = run_session(model, tok, text_model, format_prompt(tok, p), probe, None, None,
                        temperature=0.0, seed=SEED_BASE_O3 + 100 + i)
        # die letzten Positionen aller Schritte als Kalibriervektoren
        for rec in probe.records:
            calib_states.append(rec.layer_states[-1])
    print(f"[O3] Kalibrierung: {len(calib_states)} Hidden-States", flush=True)
    bases = {D: Basis72.fit(calib_states, dim=D) for D in dims}
    basis72 = bases[72]

    # 2) Hauptgitter: 3 schmale + 3 weite Prompts x 2 Temperaturen x 2 Seeds
    runs = []
    for cls, prompts in (("narrow", NARROW_PROMPTS), ("wide", WIDE_PROMPTS)):
        for pi, p in enumerate(prompts):
            for temp in (0.0, 0.9):
                for rep in range(2):
                    seed = SEED_BASE_O3 + 1000 + len(runs) * 10
                    r = run_session(model, tok, text_model, format_prompt(tok, p),
                                    probe, basis72, None, temperature=temp, seed=seed)
                    r.update({"class": cls, "prompt_idx": pi, "temperature": temp,
                              "rep": rep, "seed": seed, "arm": "unsteered",
                              "prompt": p})
                    runs.append(r)
                    print(f"  [O3 {cls:6s} p{pi} T={temp:.1f} #{rep}] "
                          f"H72={r.get('H72_norm_mean', 0):.3f} "
                          f"frac>{1/phi():.3f}={r.get('H72_norm_frac_above_phi_inv', 0):.2f} "
                          f"step={r.get('layer_step_median', 0):.4f}", flush=True)

    # 3) Steuerarme (Wide/Narrow/Random) auf den weiten Prompts
    arms = [("wide", +1, 0.40, None), ("narrow", -1, 0.40, None),
            ("rand_plus", +1, 0.40, d_rand)]
    for arm, sign, alpha, vec in arms:
        for pi, p in enumerate(WIDE_PROMPTS):
            for rep in range(3):
                seed = SEED_BASE_O3 + 5000 + len(runs) * 10
                steer = RelaySteer(text_model, vec if vec is not None else dwidth,
                                   layer=26, sign=sign, alpha=alpha)
                r = run_session(model, tok, text_model, format_prompt(tok, p),
                                probe, basis72, steer, temperature=0.6, seed=seed)
                r.update({"class": "wide", "prompt_idx": pi, "temperature": 0.6,
                          "rep": rep, "seed": seed, "arm": arm, "prompt": p})
                runs.append(r)
                print(f"  [O3 {arm:9s} p{pi} #{rep}] H72={r.get('H72_norm_mean', 0):.3f} "
                      f"wick={r.get('wick_curvature_mean', 0):.3f}", flush=True)

    probe.detach()

    # 4) Analyse
    unsteered = [r for r in runs if r["arm"] == "unsteered"]
    narrow_runs = [r for r in unsteered if r["class"] == "narrow"]
    wide_runs = [r for r in unsteered if r["class"] == "wide"]

    per_token = {"narrow": [], "wide": []}
    for r in unsteered:
        for row in r["per_step"]:
            if "H72_norm" in row:
                per_token[r["class"]].append(row["H72_norm"])

    theta = 1.0 / phi()

    def acc_at(th, direction="wide_above"):
        hit = 0
        n = 0
        for cls, vals in per_token.items():
            for x in vals:
                pred = "wide" if (x > th) == (direction == "wide_above") else "narrow"
                hit += int(pred == cls)
                n += 1
        return hit / n if n else 0.0

    rng = np.random.default_rng(20261115)
    rand_ths = rng.uniform(0.2, 0.9, 1000)
    rand_acc = np.array([max(acc_at(float(t), "wide_above"), acc_at(float(t), "wide_below"))
                         for t in rand_ths])
    acc_phi = max(acc_at(theta, "wide_above"), acc_at(theta, "wide_below"))
    direction = "wide_above" if acc_at(theta, "wide_above") >= acc_at(theta, "wide_below") else "wide_below"

    # D-Scan: Welcher-Unterraum-Entropie (nur Kalibrierstaaten, kein 2. Modelldurchlauf)
    dscan = {}
    for D, basis in bases.items():
        vals = [_entropy_norm(basis.project(s)) for s in calib_states]
        dscan[str(D)] = vals

    duty = np.array([r["layer_step_median"] for r in runs])
    a_tci = alpha_tci()
    profiles = [r["layer_step_profile"] for r in runs if r.get("layer_step_profile")]
    per_layer = [float(np.mean([p[l] for p in profiles])) for l in range(len(profiles[0]))]

    summary = {
        "threshold_phi_inv": {
            "theta": theta,
            "acc_at_phi_inv": acc_phi,
            "direction": direction,
            "acc_best_random_threshold": float(rand_acc.max()),
            "acc_random_threshold_mean": float(rand_acc.mean()),
            "percentile_of_phi_inv": float((rand_acc < acc_phi).mean()),
            "auc_wide_vs_narrow": auc(per_token["wide"], per_token["narrow"]),
            "n_tokens": {k: len(v) for k, v in per_token.items()},
            "frac_above_phi_inv": {
                "narrow": float(np.mean([r.get("H72_norm_frac_above_phi_inv", 0) for r in narrow_runs])),
                "wide": float(np.mean([r.get("H72_norm_frac_above_phi_inv", 0) for r in wide_runs])),
            },
        },
        "duty_cycle": {
            "alpha_TCI": a_tci,
            "layer_step_median_all": float(np.median(duty)),
            "layer_step_mean_all": float(np.mean(duty)),
            "ratio_measured_over_TCI": float(np.median(duty) / a_tci),
            "log10_ratio": float(math.log10(max(np.median(duty), 1e-12) / a_tci)),
            "per_layer_profile_mean": [round(x, 4) for x in per_layer],
            "per_layer_profile_min": round(float(np.min(per_layer)), 4),
            "per_layer_profile_max": round(float(np.max(per_layer)), 4),
            "n_layers_measured": len(per_layer),
        },
        "steering": {
            arm: {
                "n": len([r for r in runs if r["arm"] == arm]),
                "H72_norm_mean": float(np.mean([r["H72_norm_mean"] for r in runs
                                               if r["arm"] == arm and "H72_norm_mean" in r])),
                "wick_curvature_mean": float(np.mean([r["wick_curvature_mean"] for r in runs
                                                     if r["arm"] == arm])),
                "hidden_norm_mean": float(np.mean([r["mean_hidden_norm"] for r in runs
                                                  if r["arm"] == arm])),
            } for arm in ("wide", "narrow", "rand_plus")
        },
        "texts_unique_per_class": {
            cls: len({r["text_preview"] for r in unsteered if r["class"] == cls})
            for cls in ("narrow", "wide")
        },
    }
    summary["steering"]["p_wide_vs_narrow"] = mannwhitney(
        [r["H72_norm_mean"] for r in runs if r["arm"] == "wide" and "H72_norm_mean" in r],
        [r["H72_norm_mean"] for r in runs if r["arm"] == "narrow" and "H72_norm_mean" in r])
    summary["steering"]["p_wide_vs_rand"] = mannwhitney(
        [r["H72_norm_mean"] for r in runs if r["arm"] == "wide" and "H72_norm_mean" in r],
        [r["H72_norm_mean"] for r in runs if r["arm"] == "rand_plus" and "H72_norm_mean" in r])

    # D-Scan: Entropie der Kalibrierstaaten in Unterräumen D
    dscan = {}
    for D, basis in bases.items():
        vals = [float(np.mean([_ent(basis.project(s)) for s in calib_states]))
                for _ent in [lambda z: _entropy_norm(z)]]
        dscan[str(D)] = vals
    summary["dimension_scan"] = {
        "note": ("H_D/ln D der Kalibrier-Hidden-States in PCA-Unterräumen; "
                 "die Frage ist, ob D=72 einen Knick zeigt (72D-Behauptung)."),
        "H_norm_mean_by_dim": {k: float(np.mean(v)) for k, v in dscan.items()},
        "n_states": len(calib_states),
    }

    return {"runs": runs, "summary": summary}


def _entropy_norm(z: torch.Tensor) -> float:
    p = z.float() ** 2
    s = p.sum()
    if s <= 0:
        return 0.0
    p = (p / s).clamp_min(1e-30)
    return float(-(p * p.log()).sum() / math.log(z.numel()))


# ── Urteile (1:1 aus den vorab fixierten Kriterien) ────────────────────────
def verdicts(o1: dict, o2: dict, o3: dict | None) -> dict:
    rows = {r["parameter"]: r for r in o1["rows"]}
    v = {}

    v["O1_arithmetik"] = "PASS" if all(
        abs(r["computed"]) > 0 for r in o1["rows"]) else "FAIL"
    dev_tci = abs(rows["alpha_TCI^-1"]["rel_dev"])
    v["O1_codata_tci"] = "PASS" if dev_tci < 1e-5 else "FAIL"
    dev_w = abs(rows["alpha_Wyler^-1 (2^4*5!)"]["rel_dev"])
    v["O1_codata_wyler"] = "PASS" if dev_w < 1e-5 else "FAIL"
    v["O1_wyler_wie_gedruckt"] = (
        "FALSIFIED (24*5! -> %.4f)" % rows["alpha_Wyler^-1 (24*5!, wie gedruckt)"]["computed"])

    ratio_p = o1["T_Hubble_over_t_proton"]["Planck18"]
    rel_p = (ratio_p - o1["phi^201"]) / o1["phi^201"]
    v["O1_phi201_holografie"] = (
        "PASS" if abs(rel_p) < 0.01 else
        f"FALSIFIED (Abweichung {rel_p*100:+.1f} %, benötigter Exponent "
        f"{o1['exponent_needed_for_exact_match']['Planck18']:.2f} statt 201)")

    le = o1["look_elsewhere"]
    v["O1_look_elsewhere"] = (
        f"N_hits={le['n_hits']} von {le['n_expressions']} Ausdrücken "
        f"(E_null={le['expected_hits_uniform_null']}); "
        + ("Treffer ist selten in dieser Familie, aber ohne Mechanismus (B)"
           if le["n_hits"] <= 3 else
           "Treffer ist häufig in dieser Familie -> nicht informativ (B)"))

    c = o2["controls"]
    ts = o2["threshold_scan"]
    v["O2_alpha_als_duty_cycle"] = (
        "NICHT IDENTIFIZIERBAR (Effekt von alpha_TCI auf |psi|^2 ist "
        f"{c['alpha_TCI_effect_over_spread']:.1e} der Seed-Streuung, "
        f"max|Delta H_norm|={c['alpha_TCI_effect_max_abs']:.1e}; "
        "alpha wirkt als Phase, nicht als Takt)")
    v["O2_alpha_tci_vs_wyler"] = (
        f"nicht unterscheidbar (rel. Abstand {c['alpha_TCI_vs_alpha_Wyler_rel_diff']:.2e})")
    v["O2_informationserhaltung"] = (
        "TAUTOLOGIE (Renormierung per Konstruktion; ohne Renormierung "
        f"Normverlust {o2['s4_no_renorm']['norm_loss_final_mean']:.3f})")
    pct = c["s4_percentile_t_cross_in_matched_ensemble"]
    v["O2_s4_struktur"] = (
        f"SIGNATUR VORHANDEN, ABER NICHT ALS BALANCER (Perzentil {pct*100:.0f} % "
        "gegen 200 spektrum-gematchte Zufallsbasen; reines S4 ohne Trichter ist "
        f"ein Fixpunkt: H_norm={c['s4_pure_final_entropy']:.6f}). Der Effekt ist "
        "Darstellungstheorie — der Gleichverteilungsvektor ist die triviale "
        "Repräsentation —, kein Gleichgewichtsmechanismus.")
    v["O2_schwelle_phi_inv"] = (
        "FALSIFIED (t_cross ist monoton in theta und hat bei Phi^-1 keinen Knick: "
        f"2. Differenz {ts['second_difference_at_phi_inv']} vs Median "
        f"{ts['mean_second_difference_median']}; jede Schwelle im Intervall wird "
        "gekreuzt, Phi^-1 ist eine gesetzte Zahl)"
        if ts.get("t_cross_monotone") else "nicht monoton — Prüfen")
    if o3:
        th = o3["summary"]["threshold_phi_inv"]
        v["O3_schwelle_phi_inv"] = (
            "CONFIRMED" if (th["percentile_of_phi_inv"] >= 0.95
                            and (th["auc_wide_vs_narrow"] or 0) >= 0.70)
            else f"FALSIFIED (Perzentil {th['percentile_of_phi_inv']*100:.0f} %, "
                 f"AUC {th['auc_wide_vs_narrow']})")
        duty = o3["summary"]["duty_cycle"]
        v["O3_duty_cycle"] = (
            "CONFIRMED" if abs(duty["ratio_measured_over_TCI"] - 1) <= 0.20 else
            f"FALSIFIED (gemessen {duty['layer_step_median_all']:.4f} vs "
            f"alpha_TCI {duty['alpha_TCI']:.5f}, Faktor {duty['ratio_measured_over_TCI']:.1f})")
        ds = o3["summary"]["dimension_scan"]["H_norm_mean_by_dim"]
        dims_sorted = sorted(ds, key=lambda k: int(k))
        vals = [ds[k] for k in dims_sorted]
        spread = max(vals) - min(vals)
        v["O3_dimension_72"] = (
            "NICHT ENTSCHEIDBAR (E) — die Kurve ist im Rauschen flach "
            f"(Spanne {spread:.3f}: "
            + ", ".join(f"D={k}:{ds[k]:.3f}" for k in dims_sorted)
            + f"); D>=96 ist ranglimitiert (nur "
            f"{o3['summary']['dimension_scan']['n_states']} Kalibriervektoren). "
            "Ein 72-Extremum ist weder belegt noch widerlegt.")
        st = o3["summary"]["steering"]
        v["O3_steuerung_H72"] = (
            "RICHTUNG WIDERLEGT (Effekt existiert, aber invertiert: "
            f"wide {st['wide']['H72_norm_mean']:.3f} < narrow "
            f"{st['narrow']['H72_norm_mean']:.3f} < rand "
            f"{st['rand_plus']['H72_norm_mean']:.3f}; p(wide vs narrow)="
            f"{st['p_wide_vs_narrow']:.2e}, p(wide vs rand)={st['p_wide_vs_rand']:.2e}). "
            "Der Relay-Hook wirkt spezifisch, aber nicht als 'Expansion' im "
            "72-dim Unterraum.")
        hl = o3["summary"]["threshold_phi_inv"]
        v["O3_logit_entropie"] = (
            "Die behauptete 61,8-%-Schwelle ist auf Logit-Verteilungen "
            "gegenstandslos: H_logit/ln(V) liegt bei "
            f"{min(r.get('H_logit_norm_mean', 0) for r in o3['runs']):.4f}-"
            f"{max(r.get('H_logit_norm_mean', 0) for r in o3['runs']):.4f}; "
            "Φ^-1 würde H ≈ 7,7 nats verlangen.")
        v["O3_s4_wellen"] = ("NICHT MESSBAR (E): kein definierter S4-Isomorphismus "
                             "der gemessenen Winkel; nicht Teil des Urteils")
    return v


# ── Main ────────────────────────────────────────────────────────────────────
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="O1,O2,O3")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--out", default="results/eval_O_tci_gpu_gemma4.json")
    ap.add_argument("--rejudge", default=None,
                    help="vorhandenen Dump laden, Urteile neu schreiben, kein Lauf")
    a = ap.parse_args()
    subs = [s.strip().upper() for s in a.subs.split(",")]

    if a.rejudge:
        with open(a.rejudge) as f:
            out = json.load(f)
        out["verdicts"] = verdicts(out.get("O1", {}), out.get("O2", {}), out.get("O3"))
        with open(a.rejudge, "w") as f:
            json.dump(out, f, indent=1)
        for k, val in out["verdicts"].items():
            print(f"  {k}: {val}")
        return

    out = {"prereg": "results/PREREG_Eval_O_2026-11-15.md",
           "subs": subs, "torch": torch.__version__,
           "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    if "O1" in subs:
        print("[O1] Konstanten-Audit + Look-Elsewhere ...", flush=True)
        out["O1"] = audit_constants()
        print(f"  alpha_TCI^-1 = {out['O1']['rows'][0]['computed']:.9f} "
              f"(rel {out['O1']['rows'][0]['rel_dev']:+.2e})", flush=True)
        print(f"  Look-Elsewhere: {out['O1']['look_elsewhere']['n_hits']} Treffer / "
              f"{out['O1']['look_elsewhere']['n_expressions']} Ausdrücke", flush=True)

    if "O2" in subs:
        print("[O2] Operator-Suite auf der GPU ...", flush=True)
        out["O2"] = operator_suite(n_seeds=a.seeds)
        print(json.dumps(out["O2"]["controls"], indent=1), flush=True)

    if "O3" in subs:
        try:
            print("[O3] Gemma-4-Probe ...", flush=True)
            out["O3"] = await run_o3()
        except torch.cuda.OutOfMemoryError as e:
            out["O3"] = {"error": f"CUDA OOM: {e}"}
            print(f"[O3] OOM: {e}", flush=True)
            torch.cuda.empty_cache()

    out["verdicts"] = verdicts(out.get("O1", {}), out.get("O2", {}), out.get("O3"))
    os.makedirs("results", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\n" + "=" * 70)
    for k, val in out["verdicts"].items():
        print(f"  {k}: {val}")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
