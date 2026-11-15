#!/usr/bin/env python
"""Eval R — semantischer Attraktor statt Skalar.

Siehe results/PREREG_Eval_R_2026-11-15.md. 45 frische Prompts x 4 Arme,
gleicher Seed, 24 Tokens, Texte gespeichert.

R1  Template-Attraktor (binär): Listen-/Angebots-Schema am Textanfang
R2  Semantische Divergenz δ = 1 - cos(e_arm, e_baseline), e = mean-pooled
    finale Hidden-Repräsentation der 24 Tokens (basisinvariant)
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

from scripts.run_eval_q_relay_predictor import (      # noqa: E402
    ARMS, N_NEW, PROMPTS, SEED_BASE, TEMP, session,
)
from tci_probe import RelaySteer, TCIHolographicProbe   # noqa: E402
from scripts.analyze_steering_texts import feats       # noqa: E402

EXCLUDE = {24, 25, 26}          # die drei O3/P2-Prompts der Hypothesenbildung
TEMPLATE_KEYS = ("listing_prefix", "meta_offer", "numbered_list")


def is_template(text: str) -> bool:
    f = feats(text)
    return any(f[k] for k in TEMPLATE_KEYS)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: binäre Paare. Gibt b, c und den exakten zweiseitigen p-Wert."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return {"n01": n01, "n10": n10, "p": min(1.0, p)}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, prev = {}, True
    for i, (name, p) in enumerate(items):
        thr = alpha / (len(items) - i)
        out[name] = {"p": float(p), "threshold": thr,
                     "reject": bool(prev and p <= thr)}
        prev = out[name]["reject"]
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/eval_R_semantic_attractor_gemma4.json")
    ap.add_argument("--rejudge", default=None)
    a = ap.parse_args()

    if a.rejudge:
        with open(a.rejudge) as f:
            out = json.load(f)
        out["verdicts"] = verdicts(out)
        with open(a.rejudge, "w") as f:
            json.dump(out, f, indent=1)
        for k, v in out["verdicts"].items():
            print(f"  {k}: {v}")
        return

    from model_manager import ModelManager
    from scripts.run_gemma4_evaluations import format_prompt, load_dwidth_tensor

    t0 = time.perf_counter()
    m = ModelManager()
    e = await m.get_model("gemma4-e2b-it", px_subjective=False, px_config_preset="BASELINE")
    model, tok = e["model"], e["tokenizer"]
    tm = m._resolve_text_model(model)
    print(f"[R] Modell geladen in {time.perf_counter()-t0:.1f}s", flush=True)

    dwidth = load_dwidth_tensor().to(model.device)
    g = torch.Generator().manual_seed(SEED_BASE + 2)
    v = torch.randn(dwidth.shape, generator=g).to(model.device)
    d_rand = v - (torch.dot(v, dwidth) / torch.dot(dwidth, dwidth)) * dwidth
    d_rand = d_rand / d_rand.norm() * dwidth.norm()

    hprobe = TCIHolographicProbe(tm).attach()
    prompts = [(i, p) for i, p in enumerate(PROMPTS) if i not in EXCLUDE]
    rows = []
    for n, (i, p) in enumerate(prompts):
        seed = SEED_BASE + 7000 + i * 100
        row = {"idx": i, "prompt": p, "seed": seed,
               "class": "factual" if i < 24 else "creative"}
        emb = {}
        for arm, sign, alpha, vec in ARMS:
            steer = None if sign == 0 else RelaySteer(
                tm, d_rand if vec == "rand" else dwidth, layer=26,
                sign=sign, alpha=alpha)
            r = session(model, tok, format_prompt(tok, p), hprobe, steer, seed)
            emb[arm] = r.pop("emb")
            r["template"] = is_template(r["text"])
            row[arm] = r
        for arm, *_ in ARMS:
            d = 1.0 - float(F.cosine_similarity(emb[arm], emb["baseline"], dim=0))
            row[arm]["delta_sem"] = d
        rows.append(row)
        print(f"  [R {n+1:2d}/{len(prompts)}] tmpl base={int(row['baseline']['template'])} "
              f"wide={int(row['wide']['template'])} narrow={int(row['narrow']['template'])} "
              f"rand={int(row['rand_plus']['template'])} | d_sem wide={row['wide']['delta_sem']:.4f} "
              f"rand={row['rand_plus']['delta_sem']:.4f}", flush=True)

    hprobe.detach()
    out = {"prereg": "results/PREREG_Eval_R_2026-11-15.md",
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "seeds_base": SEED_BASE, "n_prompts": len(prompts),
           "n_new_tokens": N_NEW, "temperature": TEMP,
           "arms": [x[0] for x in ARMS], "rows": rows}
    out["contrasts"] = {
        "wide_vs_rand": contrast(rows, "wide", "rand_plus"),
        "wide_vs_narrow": contrast(rows, "wide", "narrow"),
    }
    out["descriptives"] = {
        arm: {"template_rate": float(np.mean([r[arm]["template"] for r in rows])),
              "delta_sem_mean": float(np.mean([r[arm]["delta_sem"] for r in rows])),
              "delta_sem_sd": float(np.std([r[arm]["delta_sem"] for r in rows], ddof=1))}
        for arm, *_ in ARMS}
    for cls in ("factual", "creative"):
        sub = [r for r in rows if r["class"] == cls]
        out["descriptives"][f"{cls}_wide_minus_rand"] = {
            "template_rate_diff": float(np.mean([r["wide"]["template"] for r in sub])
                                        - np.mean([r["rand_plus"]["template"] for r in sub])),
            "delta_sem_diff": float(np.mean([r["wide"]["delta_sem"] - r["rand_plus"]["delta_sem"]
                                             for r in sub])),
        }
    out["verdicts"] = verdicts(out)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\n" + "=" * 70)
    for k, v in out["verdicts"].items():
        print(f"  {k}: {v}")
    print(f"\nSaved {a.out}")


def contrast(rows, a, b):
    from scipy import stats as st
    ta = np.array([1 if r[a]["template"] else 0 for r in rows])
    tb = np.array([1 if r[b]["template"] else 0 for r in rows])
    mc = mcnemar_exact(ta, tb)
    d = np.array([r[a]["delta_sem"] - r[b]["delta_sem"] for r in rows])
    p_w = float(st.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    h = holm({"template": mc["p"], "delta_sem": p_w})
    return {
        "a": a, "b": b, "n": len(rows),
        "template": {"rate_a": float(ta.mean()), "rate_b": float(tb.mean()),
                     "n01": mc["n01"], "n10": mc["n10"], "p": mc["p"],
                     "holm": h["template"]},
        "delta_sem": {"mean_diff": float(d.mean()),
                      "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
                      "p": p_w, "holm": h["delta_sem"],
                      "direction_consistency": float(np.mean(np.sign(d) == np.sign(d.mean())))},
    }


def verdicts(out):
    c = out.get("contrasts", {}).get("wide_vs_rand")
    if not c:
        return {}
    r1 = c["template"]["holm"]["reject"] and c["template"]["rate_a"] < c["template"]["rate_b"]
    r2 = c["delta_sem"]["holm"]["reject"] and c["delta_sem"]["mean_diff"] > 0
    v = {"R1_template": f"template wide {c['template']['rate_a']:.2f} vs rand "
                        f"{c['template']['rate_b']:.2f} (McNemar p={c['template']['p']:.3g}, "
                        f"Holm={'ja' if c['template']['holm']['reject'] else 'nein'})",
         "R2_delta_sem": f"δ wide−rand {c['delta_sem']['mean_diff']:+.4f} "
                         f"(dz={c['delta_sem']['dz']:+.2f}, p={c['delta_sem']['p']:.3g}, "
                         f"Richtung {c['delta_sem']['direction_consistency']*100:.0f} %)"}
    v["urteil"] = ("CONFIRMED — R1 und R2 im primären Kontrast signifikant und "
                   "in der vorhergesagten Richtung" if r1 and r2 else
                   ("PARTIAL — nur " + ("R1" if r1 else "R2") if (r1 or r2) else
                    "FALSIFIED — kein Endpunkt im primären Kontrast"))
    return v


if __name__ == "__main__":
    asyncio.run(main())
