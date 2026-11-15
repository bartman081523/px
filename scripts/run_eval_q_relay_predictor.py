#!/usr/bin/env python
"""Eval Q — gibt es einen basis-invarianten Relay-Prädiktor?

Siehe results/PREREG_Eval_Q_2026-11-15.md. 48 distinkte Prompts x 4 Arme,
gleicher Seed pro Prompt, exakt 24 Tokens. Drei Kandidaten, Holm k=3.

K1  ||h|| (Layer 27-34, letzte Position)        -> Manipulations-Check
K2  H_logit / ln V                              -> Downstream-Effekt
K3  erank der 24 x d-Trajektorie (SVD)          -> basis-invariant
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

from tci_probe import RelaySteer, TCIHolographicProbe          # noqa: E402

SEED_BASE = 20261201
N_PROMPTS = 48
N_NEW = 24
TEMP = 0.6
ARMS = [("baseline", 0, 0.0, None), ("wide", +1, 0.40, None),
        ("narrow", -1, 0.40, None), ("rand_plus", +1, 0.40, "rand")]

FACTUAL = [
    "Explain how a bicycle derailleur works.",
    "Describe the water cycle in three sentences.",
    "Summarize the causes of the French Revolution.",
    "How does a catalytic converter reduce emissions?",
    "Explain the difference between TCP and UDP.",
    "What is the role of mitochondria in a cell?",
    "Describe how vaccines train the immune system.",
    "Explain why the sky is blue.",
    "How does a refrigerator move heat?",
    "Describe the nitrogen cycle briefly.",
    "What caused the 2008 financial crisis?",
    "Explain how GPS determines a position.",
    "How does photosynthesis store energy?",
    "Describe the structure of a transistor.",
    "What is the difference between weather and climate?",
    "Explain how a hash table handles collisions.",
    "Describe the function of the liver.",
    "How does an electric motor generate torque?",
    "What is entropy in thermodynamics?",
    "Explain how antibiotics work.",
    "Describe the water treatment process.",
    "How does a compiler optimize code?",
    "Explain the greenhouse effect.",
    "Describe how sound travels through air.",
]
CREATIVE = [
    "Write an original metaphor linking ocean currents, memory, and bureaucracy.",
    "Invent a short myth explaining why the moon has craters.",
    "Combine three unrelated domains into one surprising hypothesis.",
    "Write a metaphor about a library that forgets.",
    "Invent a ritual for the first rain of autumn.",
    "Describe a city built inside a clock.",
    "What would a conversation between fog and glass sound like?",
    "Invent a unit of measurement for regret.",
    "Write the opening of a myth about a map that redraws itself.",
    "Describe music as a landscape.",
    "Invent a law that only applies on Tuesdays.",
    "Write about a door that remembers everyone who used it.",
    "Describe the colour of a sound you have never heard.",
    "Invent a machine that translates silence.",
    "Write a short fable about a river that refuses to reach the sea.",
    "Describe a house whose rooms change with the weather.",
    "Invent an alphabet made of temperatures.",
    "Write about a city where shadows arrive late.",
    "Describe a library that only lends memories.",
    "Invent a profession that exists only at dusk.",
    "Write a myth about the first tool.",
    "Describe a language spoken only by mirrors.",
    "Invent a holiday that celebrates forgetting.",
    "Write about a map that leads to its own author.",
]
PROMPTS = FACTUAL + CREATIVE


def shannon_bits(logits: torch.Tensor) -> float:
    p = F.softmax(logits.float(), dim=-1)
    return float(-(p * p.clamp_min(1e-30).log()).sum())


def erank_of(X: torch.Tensor) -> float:
    """exp(H_spec) der Singulärwerte, basis-invariant (SVD)."""
    Xc = X.float() - X.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    v = s ** 2
    tot = v.sum()
    if tot <= 0:
        return 0.0
    p = (v / tot).clamp_min(1e-30)
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def session(model, tok, prompt, hprobe, steer, seed):
    torch.manual_seed(seed)
    dev = model.device
    ids = tok(prompt, return_tensors="pt").to(dev)["input_ids"]
    T = int(ids.shape[1])
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    vocab = int(cfg.vocab_size)
    eos = tok.eos_token_id

    hprobe.clear()
    ctx = steer if steer is not None else _null()
    with ctx:
        out = model(input_ids=ids, use_cache=True,
                    attention_mask=torch.ones(1, T, device=dev, dtype=torch.long))
        hprobe.records.clear()
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        ents, gen, gen_ids = [], 0, []
        for t in range(N_NEW):
            ents.append(shannon_bits(logits))
            l = logits.clone()
            if eos is not None:
                l[eos] = -float("inf")          # min_new_tokens = N_NEW
            nxt = int(torch.multinomial(F.softmax(l / TEMP, dim=-1), 1))
            gen += 1
            gen_ids.append(nxt)
            pos = T + gen - 1
            out = model(input_ids=torch.tensor([[nxt]], device=dev),
                        past_key_values=past, use_cache=True,
                        cache_position=torch.tensor([pos], device=dev),
                        attention_mask=torch.ones(1, pos + 1, device=dev, dtype=torch.long))
            past = out.past_key_values
            logits = out.logits[0, -1].float()

    recs = hprobe.records
    layers = list(range(27, 35))
    k1 = float(np.mean([float(r.layer_states[l].norm()) for r in recs for l in layers]))
    X = torch.stack([r.layer_states[-1] for r in recs], dim=0)
    k3 = erank_of(X)
    k2 = float(np.mean(ents) / math.log(vocab))
    return {"K1_norm": k1, "K2_Hlogit_norm": k2, "K3_erank": k3,
            "n_gen": gen, "logit_bits_mean": float(np.mean(ents)),
            "final_hidden_norm": float(recs[-1].layer_states[-1].norm()),
            "text": tok.decode(gen_ids, skip_special_tokens=True),
            # mittlere finale Hidden-Repräsentation der 24 Tokens (für Cosinus)
            "emb": torch.stack([r.layer_states[-1] for r in recs], 0).mean(0)}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ── Statistik ───────────────────────────────────────────────────────────────
def holm(pvals: dict, alpha: float = 0.05) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    k = len(items)
    out, prev_reject = {}, True
    for i, (name, p) in enumerate(items):
        thr = alpha / (k - i)
        rej = prev_reject and (p <= thr)
        out[name] = {"p": p, "threshold": thr, "reject": bool(rej)}
        prev_reject = rej
    return out


def contrast(rows: list, a: str, b: str) -> dict:
    from scipy import stats as st
    res = {}
    pv = {}
    for key in ("K1_norm", "K2_Hlogit_norm", "K3_erank"):
        d = np.array([r[a][key] - r[b][key] for r in rows])
        p = float(st.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        pt = float(st.ttest_rel([r[a][key] for r in rows],
                                [r[b][key] for r in rows]).pvalue)
        res[key] = {
            "mean_diff": float(d.mean()),
            "sd_diff": float(d.std(ddof=1)),
            "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
            "p_wilcoxon": p,
            "p_paired_t": pt,
            "direction_consistency": float(np.mean(np.sign(d) == np.sign(d.mean()))),
            "per_prompt_diff": [round(float(x), 6) for x in d],
        }
        pv[key] = p
    h = holm(pv)
    for key in res:
        res[key]["holm"] = h[key]
        res[key]["confirmed"] = bool(h[key]["reject"] and
                                     res[key]["direction_consistency"] >= 0.80)
    return {"a": a, "b": b, "n_prompts": len(rows), "metrics": res}


# ── Main ────────────────────────────────────────────────────────────────────
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/eval_Q_relay_predictor_gemma4.json")
    ap.add_argument("--rejudge", default=None)
    ap.add_argument("--max-prompts", type=int, default=N_PROMPTS)
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
    print(f"[Q] Modell geladen in {time.perf_counter()-t0:.1f}s", flush=True)

    hprobe = TCIHolographicProbe(tm).attach()
    dwidth = load_dwidth_tensor().to(model.device)
    g = torch.Generator().manual_seed(SEED_BASE + 1)
    v = torch.randn(dwidth.shape, generator=g).to(model.device)
    d_rand = v - (torch.dot(v, dwidth) / torch.dot(dwidth, dwidth)) * dwidth
    d_rand = d_rand / d_rand.norm() * dwidth.norm()

    prompts = PROMPTS[:a.max_prompts]
    rows = []
    for i, p in enumerate(prompts):
        seed = SEED_BASE + i * 100
        row = {"idx": i, "prompt": p, "seed": seed, "class": "factual" if i < 24 else "creative"}
        for arm, sign, alpha, vec in ARMS:
            steer = None if sign == 0 else RelaySteer(
                tm, d_rand if vec == "rand" else dwidth, layer=26,
                sign=sign, alpha=alpha)
            r = session(model, tok, format_prompt(tok, p), hprobe, steer, seed)
            row[arm] = r
        rows.append(row)
        print(f"  [Q {i+1:2d}/{len(prompts)}] base={row['baseline']['K1_norm']:.2f} "
              f"wide={row['wide']['K1_norm']:.2f} narrow={row['narrow']['K1_norm']:.2f} "
              f"rand={row['rand_plus']['K1_norm']:.2f} | K2 wide={row['wide']['K2_Hlogit_norm']:.4f}",
              flush=True)

    hprobe.detach()
    out = {"prereg": "results/PREREG_Eval_Q_2026-11-15.md",
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "seeds_base": SEED_BASE, "n_prompts": len(prompts),
           "n_new_tokens": N_NEW, "temperature": TEMP,
           "arms": [x[0] for x in ARMS], "rows": rows}
    out["contrasts"] = {"wide_vs_narrow": contrast(rows, "wide", "narrow"),
                        "wide_vs_rand": contrast(rows, "wide", "rand_plus")}
    out["descriptives"] = {
        arm: {k: {"mean": float(np.mean([r[arm][k] for r in rows])),
                  "sd": float(np.std([r[arm][k] for r in rows], ddof=1))}
              for k in ("K1_norm", "K2_Hlogit_norm", "K3_erank")}
        for arm, *_ in ARMS}
    out["verdicts"] = verdicts(out)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\n" + "=" * 70)
    for k, val in out["verdicts"].items():
        print(f"  {k}: {val}")
    print(f"\nSaved {a.out}")


def verdicts(out: dict) -> dict:
    v = {}
    for name, c in out.get("contrasts", {}).items():
        m = c["metrics"]
        any_sig = any(m[k]["holm"]["reject"] for k in m)
        conf = [k for k in m if m[k]["confirmed"]]
        if not any_sig:
            v[name] = "FALSIFIED — kein Kandidat Holm-signifikant"
        elif conf:
            v[name] = "CONFIRMED: " + ", ".join(
                f"{k} (dz={m[k]['dz']:+.2f}, p={m[k]['p_wilcoxon']:.2g}, "
                f"Richtung {m[k]['direction_consistency']*100:.0f} %)" for k in conf)
        else:
            v[name] = "PARTIAL — signifikant, aber Richtung < 80 %: " + ", ".join(
                k for k in m if m[k]["holm"]["reject"])
    return v


if __name__ == "__main__":
    asyncio.run(main())
