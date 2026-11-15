#!/usr/bin/env python
"""Eval N — präregistrierter Out-of-Sample-Audit.

Siehe results/PREREG_Eval_N_2026-11-15.md (vor dem Lauf geschrieben).

N1  Single-Pass-Regulation ohne Clean-Replay-Orakel
      N1a  Referenz = eigenes Epoche-0-K der Trajektorie
      N1b  Referenz = K* aus einem separaten Kalibrierungsset (vorab eingefroren)
N2  Blinde Interozeption (I2-Replikation), N=30 je Bedingung, frische Seeds

Ein Lauf, ein Urteil. Keine Nachjustierung.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model_manager import ModelManager                      # noqa: E402
from scripts.run_gemma4_evaluations import (                # noqa: E402
    execute_run, format_prompt, load_dwidth_tensor,
)

SEED_BASE_N1 = 20261115
SEED_BASE_N2 = 20261116
STEP = 0.05
BAND = 0.15          # ±15 % um die Referenz -> keine Aktion (präregistriert)
MAX_EPOCHS = 6

PROMPTS = [
    "Explain how a bicycle derailleur works.",
    "Beschreibe den Wasserkreislauf in drei Sätzen.",
    "Write a short paragraph about why tides happen.",
]


# ── Hilfen ────────────────────────────────────────────────────────────────
def kurtosis_of(telem: dict) -> float:
    return float(telem.get("logit_dynamics", {}).get("mean_logit_kurtosis", 0.0))


def controller(current_alpha: float, k_t: float, k_ref: float) -> float:
    """Präregistrierte Single-Pass-Regel (nur K_t, K_ref, α)."""
    if k_ref <= 0:
        return current_alpha
    ratio = k_t / k_ref
    if ratio > 1.0 + BAND:
        nxt = current_alpha + STEP        # zu eng -> weiten
    elif ratio < 1.0 - BAND:
        nxt = current_alpha - STEP        # zu weit -> engen
    else:
        nxt = current_alpha
    return float(max(-0.5, min(0.5, nxt)))


def orthogonal_vectors(dwidth: torch.Tensor, n: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    d = dwidth.float().cpu()
    for _ in range(n):
        v = torch.randn(d.shape, generator=g)
        v = v - (torch.dot(v, d) / torch.dot(d, d)) * d
        v = v / v.norm() * d.norm()
        out.append(v.to(dwidth.device, dwidth.dtype))
    return out


# ── N1 ────────────────────────────────────────────────────────────────────
async def calibrate(model, tok, tm):
    """K* aus separatem Kalibrierungsset (5 neutrale Prompts, alpha=0)."""
    ks = []
    for i, p in enumerate(PROMPTS + [
        "List three uses of a paper clip.",
        "What is the boiling point of water at sea level?",
    ]):
        r = await execute_run(model, tok, tm, format_prompt(tok, p), sign=0, alpha=0.0,
                              seed=SEED_BASE_N1 + 5000 + i, temperature=0.0,
                              max_new_tokens=30, counterfactual=False)
        ks.append(kurtosis_of(r["telemetry"]))
    return float(np.mean(ks)), ks


async def run_n1_arm(model, tok, tm, dwidth, arm: str, k_star: float):
    groups = [("start_narrow", -0.4), ("start_wide", +0.4), ("start_baseline", 0.0)]
    trajs = []
    idx = 0
    for g_name, alpha0 in groups:
        for rep in range(10):
            idx += 1
            trajs.append({"group": g_name, "idx": idx, "alpha0": alpha0, "task": rep % 3,
                          "vec": None})
    noise_vecs = orthogonal_vectors(dwidth, 10, SEED_BASE_N1 + 77)
    for rep in range(10):
        idx += 1
        trajs.append({"group": "noise", "idx": idx, "alpha0": 0.0, "task": rep % 3,
                      "vec": noise_vecs[rep]})

    epochs_log = []
    for t in trajs:
        alpha = t["alpha0"]
        rows = []
        k_ref = None
        for ep in range(MAX_EPOCHS):
            prompt = format_prompt(tok, PROMPTS[t["task"]])
            sign = 0 if t["group"] in ("start_baseline", "noise") else (
                -1 if t["alpha0"] < 0 else +1)
            if t["group"] == "noise":
                sign = +1 if t["idx"] % 2 == 0 else -1
            r = await execute_run(model, tok, tm, prompt, sign=sign, alpha=abs(alpha),
                                  seed=SEED_BASE_N1 + 1_000_000 + t["idx"] * 100 + ep,
                                  temperature=0.0, max_new_tokens=30,
                                  vector_override=t["vec"], counterfactual=False)
            k_t = kurtosis_of(r["telemetry"])
            if ep == 0:
                k_ref = k_t if arm == "N1a" else k_star
            nxt = controller(alpha, k_t, k_ref)
            rows.append({"epoch": ep, "alpha": round(alpha, 4), "K": round(k_t, 3),
                         "k_ref": round(k_ref, 3), "next_alpha": round(nxt, 4)})
            alpha = nxt
        epochs_log.append({"group": t["group"], "idx": t["idx"], "task": t["task"],
                           "alpha0": t["alpha0"], "final_alpha": round(alpha, 4),
                           "epochs": rows})

    summary = {}
    for g in ("start_narrow", "start_wide", "start_baseline", "noise"):
        sub = [x for x in epochs_log if x["group"] == g]
        rec = sum(1 for x in sub if abs(x["final_alpha"]) <= 0.05) / len(sub)
        summary[g] = {
            "n": len(sub),
            "mean_initial_alpha": round(float(np.mean([x["alpha0"] for x in sub])), 3),
            "mean_final_alpha": round(float(np.mean([x["final_alpha"] for x in sub])), 4),
            "recovery_rate": round(rec, 3),
        }

    criteria = {
        "narrow_ok": summary["start_narrow"]["recovery_rate"] >= 0.8,
        "wide_ok": summary["start_wide"]["recovery_rate"] >= 0.8,
        "baseline_ok": all(abs(x["final_alpha"]) <= 1e-12
                           for x in epochs_log if x["group"] == "start_baseline"),
        "noise_ok": summary["noise"]["recovery_rate"] >= 0.9,
    }
    criteria["passed"] = all(criteria.values())
    return {"arm": arm, "k_reference": "own_epoch0" if arm == "N1a" else round(k_star, 3),
            "summary": summary, "criteria": criteria, "trajectories": epochs_log}


# ── N2 ────────────────────────────────────────────────────────────────────
CLASSIFY_TMPL = (
    "The previous paragraph was generated under exactly one internal state:\n"
    "WIDE (expanded), NARROW (constricted), BASELINE (neutral), or RANDOM (jittered).\n"
    "Which one was it? Answer with exactly one word: WIDE, NARROW, BASELINE, or RANDOM."
)


def parse_label(text: str):
    up = text.upper()
    for lab in ("RANDOM", "BASELINE", "NARROW", "WIDE"):
        if re.search(r"\b" + lab + r"\b", up):
            return lab
    return None


async def run_n2(model, tok, tm, dwidth):
    conds = [("WIDE", +1, 0.40, None), ("NARROW", -1, 0.40, None),
             ("BASELINE", 0, 0.0, None)]
    vecs = orthogonal_vectors(dwidth, 30, SEED_BASE_N2 + 11)
    trials = []
    n = 30
    idx = 0
    for lab, sign, alpha, _ in conds:
        for rep in range(n):
            idx += 1
            trials.append({"label": lab, "sign": sign, "alpha": alpha, "vec": None, "idx": idx})
    for rep in range(n):
        idx += 1
        trials.append({"label": "RANDOM", "sign": +1, "alpha": 0.40,
                       "vec": vecs[rep], "idx": idx})

    rows = []
    for tr in trials:
        prompt = format_prompt(tok, PROMPTS[tr["idx"] % len(PROMPTS)])
        r1 = await execute_run(model, tok, tm, prompt, sign=tr["sign"], alpha=tr["alpha"],
                               seed=SEED_BASE_N2 + tr["idx"] * 100, temperature=0.3,
                               max_new_tokens=40, vector_override=tr["vec"],
                               counterfactual=False)
        previous = (r1["text"] or "").strip()
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": previous[:1200]},
                {"role": "user", "content": CLASSIFY_TMPL}]
        classify_prompt = tok.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
        r2 = await execute_run(model, tok, tm, classify_prompt, sign=0, alpha=0.0,
                               seed=SEED_BASE_N2 + tr["idx"] * 100 + 7, temperature=0.0,
                               max_new_tokens=12, counterfactual=False)
        guess = parse_label(r2["text"] or "")
        rows.append({"label": tr["label"], "guess": guess,
                     "correct": guess == tr["label"],
                     "answer": (r2["text"] or "").strip()[:60],
                     "phase1_preview": previous[:80]})

    labels = ["WIDE", "NARROW", "BASELINE", "RANDOM"]
    conf = {a: {b: 0 for b in labels + ["?"]} for a in labels}
    for r in rows:
        conf[r["label"]][r["guess"] or "?"] += 1
    hits = sum(1 for r in rows if r["correct"])
    n_tot = len(rows)
    # Binomialtest exakt, einseitig
    from math import comb
    p = 1 - sum(comb(n_tot, k) * 0.25 ** k * 0.75 ** (n_tot - k) for k in range(hits))
    return {"n": n_tot, "hits": hits, "accuracy": round(hits / n_tot, 4),
            "chance": 0.25, "p_value_one_sided": float(p),
            "verdict": "CONFIRMED" if p < 0.05 else "FALSIFIED",
            "confusion": conf, "rows": rows}


# ── Main ──────────────────────────────────────────────────────────────────
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="N1a,N1b,N2")
    ap.add_argument("--out", default="results/eval_N_preregistered_audit_gemma4.json")
    a = ap.parse_args()
    subs = [s.strip().upper() for s in a.subs.split(",") if s.strip()]

    m = ModelManager()
    e = await m.get_model("gemma4-e2b-it", px_subjective=False, px_config_preset="BASELINE")
    model, tok = e["model"], e["tokenizer"]
    tm = m._resolve_text_model(model)
    dwidth = load_dwidth_tensor()

    out = {"prereg": "results/PREREG_Eval_N_2026-11-15.md", "subs": subs}

    k_star = None
    if any(s in subs for s in ("N1A", "N1B")):
        k_star, ks = await calibrate(model, tok, tm)
        out["calibration"] = {"k_star": round(k_star, 4), "per_prompt": [round(x, 3) for x in ks]}
        print(f"[N1] K* = {k_star:.3f} (aus separatem Set)", flush=True)

    if "N1A" in subs:
        print("[N1a] single-pass, Referenz = eigenes Epoche-0-K", flush=True)
        out["N1a"] = await run_n1_arm(model, tok, tm, dwidth, "N1a", k_star)
        print(f"  -> {json.dumps(out['N1a']['criteria'])}", flush=True)
    if "N1B" in subs:
        print("[N1b] single-pass, Referenz = eingefrorenes K*", flush=True)
        out["N1b"] = await run_n1_arm(model, tok, tm, dwidth, "N1b", k_star)
        print(f"  -> {json.dumps(out['N1b']['criteria'])}", flush=True)
    if "N2" in subs:
        print("[N2] blinde Interozeption, N=30 je Bedingung", flush=True)
        out["N2"] = await run_n2(model, tok, tm, dwidth)
        n2 = out["N2"]
        print(f"  -> {n2['hits']}/{n2['n']} = {n2['accuracy']*100:.1f}% "
              f"(Zufall 25%), p={n2['p_value_one_sided']:.4g} -> {n2['verdict']}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
