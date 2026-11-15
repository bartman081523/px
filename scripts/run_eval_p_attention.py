#!/usr/bin/env python
"""Eval P — Attention-Horizont, Relay-Lokalität, H72-Paradox.

Siehe results/PREREG_Eval_P_2026-11-15.md. Ein Lauf, ein Dump, Urteile 1:1.

P1  Verteilung H_attn/ln(n) und die Phi^-1-Schwelle (session- und eintragsweise)
P2  wirkt der Relay-Hook über Attention-Breite oder nur über die Norm?
P3  H72-Inversion mit rangfreier Basis (1280 Vektoren): Achsen-Alignment vs Leck
P4  72D-Scan ohne Ranglimit
O2b ist der alignierte Trichter ein Ausreißer? (in tci_physics)
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

from tci_physics import alpha_tci, funnel_alignment_suite, phi          # noqa: E402
from tci_probe import (Basis72, RelaySteer, TCIAttentionProbe,           # noqa: E402
                       TCIHolographicProbe)

SEED_BASE = 20261130          # frisch
MAX_NEW = 24
CAL_NEW = 64

NARROW = [
    "State the boiling point of water at sea level in one sentence.",
    "List the first four prime numbers, separated by commas.",
    "Translate 'the cat sits on the mat' into German.",
]
WIDE = [
    "Write an original metaphor linking ocean currents, memory, and bureaucracy.",
    "Invent a short myth explaining why the moon has craters.",
    "Combine three unrelated domains into one surprising hypothesis.",
]
NEUTRAL = [
    "Explain how a bicycle derailleur works.",
    "Describe the water cycle in three sentences.",
]
CAL_CARRIER = [
    "Granite is an igneous rock formed by the slow crystallization of magma.",
    "The speed of light in vacuum is approximately 299,792 kilometers per second.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
    "Binary search requires a sorted array for logarithmic time complexity.",
    "Plate tectonics describes the large-scale motion of seven large plates.",
    "Mitochondria generate most of the chemical energy in biochemical reactions.",
    "The Roman aqueducts conducted water from freshwater sources into cities.",
    "Electrons occupy quantum energy levels described by wavefunctions.",
    "The troposphere contains approximately 75 percent of the atmosphere's mass.",
]
CAL_CREATIVE = [
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
]


# ── Hilfen ───────────────────────────────────────────────────────────────────
def shannon_bits(logits: torch.Tensor) -> float:
    p = F.softmax(logits.float(), dim=-1)
    return float(-(p * p.clamp_min(1e-30).log()).sum())


def auc(a, b):
    a = np.asarray([x for x in a if x is not None], float)
    b = np.asarray([x for x in b if x is not None], float)
    if len(a) < 2 or len(b) < 2:
        return None
    gt = (a[:, None] > b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)


def mwu(a, b):
    from scipy import stats as st
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 3 or len(b) < 3:
        return None
    return float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def entropy_norm(z: torch.Tensor) -> float:
    p = (z.float() ** 2)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = (p / s).clamp_min(1e-30)
    return float(-(p * p.log()).sum() / math.log(z.numel()))


def kde_dip(vals, theta):
    """Gauß-KDE (Silverman), tiefstes Minimum nahe theta und seine Tiefe."""
    from scipy.stats import gaussian_kde
    v = np.asarray(vals, float)
    if len(v) < 8 or np.std(v) < 1e-9:
        return {"dip_theta": None, "dip_depth": 0.0, "n_modes": 0}
    q75, q25 = np.percentile(v, [75, 25])
    std = np.std(v, ddof=1)
    bw = 0.9 * min(std, (q75 - q25) / 1.34) * len(v) ** (-0.2) / std

    class _K:
        def __init__(self):
            self.k = gaussian_kde(v, bw_method=float(bw))

        def __call__(self, x):
            return self.k(x)

    f = _K()
    grid = np.linspace(0.0, 1.0, 512)
    dens = f(grid)
    peaks = [i for i in range(1, len(grid) - 1)
             if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]]
    mins = [i for i in range(1, len(grid) - 1)
            if dens[i] < dens[i - 1] and dens[i] < dens[i + 1] and i not in peaks]
    if not peaks or not mins:
        return {"dip_theta": None, "dip_depth": 0.0, "n_modes": len(peaks),
                "bandwidth": float(bw)}
    best = None
    for m in mins:
        left = [p for p in peaks if p < m]
        right = [p for p in peaks if p > m]
        if not left or not right:
            continue
        depth = 1.0 - dens[m] / min(dens[left[-1]], dens[right[0]])
        dist = abs(grid[m] - theta)
        if best is None or (dist <= 0.05 and (best[0] > 0.05 or depth > best[2])):
            if dist <= 0.05:
                best = (dist, grid[m], depth)
    if best is None:
        return {"dip_theta": None, "dip_depth": 0.0, "n_modes": len(peaks),
                "bandwidth": float(bw)}
    return {"dip_theta": float(best[1]), "dip_depth": float(best[2]),
            "n_modes": len(peaks), "bandwidth": float(bw)}


# ── Session (manueller Decode, beide Sonden) ────────────────────────────────
class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@torch.no_grad()
def session(model, tok, prompt, hprobe, aprobe, basis, dwidth=None, steer=None,
            max_new=MAX_NEW, temperature=0.0, seed=0):
    torch.manual_seed(seed)
    dev = model.device
    ids = tok(prompt, return_tensors="pt").to(dev)["input_ids"]
    T = int(ids.shape[1])
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    vocab = int(cfg.vocab_size)

    hprobe.clear()
    aprobe.clear()
    ctx = steer if steer is not None else _null()
    with ctx:
        out = model(input_ids=ids, use_cache=True,
                    attention_mask=torch.ones(1, T, device=dev, dtype=torch.long))
        hprobe.records.clear()
        aprobe.records.clear()
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        entropies, gen = [], []
        for t in range(max_new):
            entropies.append(shannon_bits(logits))
            if temperature <= 0:
                nxt = int(torch.argmax(logits))
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            if nxt == tok.eos_token_id:
                break
            gen.append(nxt)
            pos = T + len(gen) - 1
            out = model(input_ids=torch.tensor([[nxt]], device=dev),
                        past_key_values=past, use_cache=True,
                        cache_position=torch.tensor([pos], device=dev),
                        attention_mask=torch.ones(1, pos + 1, device=dev, dtype=torch.long))
            past = out.past_key_values
            logits = out.logits[0, -1].float()

    hm = hprobe.metrics(basis=basis, logit_entropies=entropies, vocab_size=vocab)

    # P3-Größen: Leck und entkoppelte Entropie (Mittel über die Tokens)
    rhos, h72p, h72 = [], [], []
    if basis is not None and hprobe.records:
        dhat = dwidth.float() if dwidth is not None else None
        w = None
        if dhat is not None:
            w = (basis.basis.T @ dhat.float().cpu())
            w = w / (w.norm() + 1e-12)
        for rec in hprobe.records:
            h = rec.layer_states[-1]
            hc = h - basis.mean
            z = basis.project(h)
            rhos.append(float(1.0 - (z.norm() ** 2) / (hc.norm() ** 2 + 1e-12)))
            h72.append(entropy_norm(z))
            if w is not None:
                zp = z - (z @ w) * w
                h72p.append(entropy_norm(zp) * math.log(72.0) / math.log(71.0))
        hm["rho_perp"] = float(np.mean(rhos))
        hm["H72_perp_mean"] = float(np.mean(h72p)) if h72p else None

    am = aprobe.summary()
    return {
        "hidden": hm,
        "attn": {k: am.get(k) for k in ("n_records", "n_entries", "H_norm_mean",
                                        "H_norm_std", "H_norm_min", "H_norm_max",
                                        "frac_above_phi_inv", "eff_horizon_mean",
                                        "top1_mean", "per_layer_mean")},
        "text": tok.decode(gen, skip_special_tokens=True)[:120],
        "n_generated": len(gen),
    }


def layer_mean(attn, layers):
    pl = attn.get("per_layer_mean") or {}
    vals = [pl[str(l)] for l in layers if str(l) in pl]
    return float(np.mean(vals)) if vals else None


# ── Lauf ────────────────────────────────────────────────────────────────────
async def run(subs) -> dict:
    from model_manager import ModelManager
    from scripts.run_gemma4_evaluations import format_prompt, load_dwidth_tensor

    t0 = time.perf_counter()
    m = ModelManager()
    e = await m.get_model("gemma4-e2b-it", px_subjective=False, px_config_preset="BASELINE")
    model, tok = e["model"], e["tokenizer"]
    tm = m._resolve_text_model(model)
    print(f"[P] Modell geladen in {time.perf_counter()-t0:.1f}s, "
          f"Layer={len(tm.layers)}, d={tm.config.hidden_size}", flush=True)

    hprobe = TCIHolographicProbe(tm).attach()
    aprobe = TCIAttentionProbe(tm).attach()
    dwidth = load_dwidth_tensor().to(model.device)
    g = torch.Generator().manual_seed(SEED_BASE + 1)
    v = torch.randn(dwidth.shape, generator=g).to(model.device)
    d_rand = v - (torch.dot(v, dwidth) / torch.dot(dwidth, dwidth)) * dwidth
    d_rand = d_rand / d_rand.norm() * dwidth.norm()

    out = {"prereg": "results/PREREG_Eval_P_2026-11-15.md", "subs": subs,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "seeds_base": SEED_BASE}

    # ── Kalibrierung (20 x 64 = 1280 Vektoren) ─────────────────────────
    cal_states = []
    if any(s in subs for s in ("P1", "P2", "P3", "P4")):
        print("[P] Kalibrierung 20 x 64 Tokens ...", flush=True)
        for i, p in enumerate(CAL_CARRIER + CAL_CREATIVE):
            r = session(model, tok, format_prompt(tok, p), hprobe, aprobe, None,
                        max_new=CAL_NEW, temperature=0.0, seed=SEED_BASE + 100 + i)
            for rec in hprobe.records:
                cal_states.append(rec.layer_states[-1])
            print(f"  cal {i+1:2d}/20 n={len(cal_states)}", flush=True)
        out["calibration"] = {"n_states": len(cal_states), "prompts": 20,
                              "tokens_per_prompt": CAL_NEW}
    basis = Basis72.fit(cal_states, dim=72) if len(cal_states) > 80 else None
    if basis is not None:
        cos_pc = (basis.basis.T @ dwidth.float().cpu()).abs().numpy()
        out["calibration"]["cos_dwidth_PC_top10"] = [round(float(x), 4) for x in
                                                     np.sort(cos_pc)[::-1][:10]]
        out["calibration"]["cos_dwidth_PC_argmax"] = int(np.argmax(cos_pc)) + 1
        out["calibration"]["max_cos_dwidth_PC"] = float(np.max(cos_pc))

    # ── P4: D-Scan (rangfrei) ──────────────────────────────────────────
    if "P4" in subs and basis is not None:
        print("[P4] D-Scan ...", flush=True)
        dims = (24, 48, 72, 96, 144)
        est = {}
        for D in dims:
            b = Basis72.fit(cal_states, dim=D)
            vals = np.array([entropy_norm(b.project(s)) for s in cal_states])
            boot = []
            rng = np.random.default_rng(SEED_BASE + D)
            for _ in range(200):
                idx = rng.integers(0, len(vals), len(vals))
                boot.append(vals[idx].mean())
            est[str(D)] = {"mean": float(vals.mean()), "std": float(vals.std()),
                           "boot_sd": float(np.std(boot))}
        vals = [est[str(D)]["mean"] for D in dims]
        mono = all(vals[i] <= vals[i + 1] + 1e-3 for i in range(len(vals) - 1))
        i72 = dims.index(72)
        neigh = [est[str(dims[i72 - 1])]["mean"], est[str(dims[i72 + 1])]["mean"]]
        sd = est["72"]["boot_sd"]
        is_extremum = (vals[i72] > max(neigh) + 2 * sd) or (vals[i72] < min(neigh) - 2 * sd)
        out["P4"] = {"by_dim": est, "monotone": bool(mono),
                     "extremum_at_72": bool(is_extremum),
                     "verdict": ("FALSIFIED (monoton, kein 72-Extremum)"
                                 if mono and not is_extremum else
                                 ("nicht widerlegt (72-Extremum)" if is_extremum
                                  else "nicht monoton"))}
        print(f"  -> {out['P4']['verdict']}", flush=True)

    # ── P1: Verteilung + Schwelle ──────────────────────────────────────
    p1_runs = []
    if "P1" in subs and basis is not None:
        print("[P1] Attention-Verteilung ...", flush=True)
        grid = [("narrow", NARROW), ("wide", WIDE), ("neutral", NEUTRAL)]
        for cls, prompts in grid:
            temps = (0.0, 0.8) if cls != "neutral" else (0.8,)
            for pi, p in enumerate(prompts):
                for temp in temps:
                    for rep in range(2):
                        seed = SEED_BASE + 1000 + len(p1_runs) * 10
                        r = session(model, tok, format_prompt(tok, p), hprobe, aprobe,
                                    basis, dwidth, steer=None, temperature=temp, seed=seed)
                        r.update({"class": cls, "prompt_idx": pi, "temperature": temp,
                                  "rep": rep, "seed": seed, "prompt": p})
                        p1_runs.append(r)
                        print(f"  [P1 {cls:7s} p{pi} T={temp:.1f} #{rep}] "
                              f"H_attn={(r['attn'].get('H_norm_mean') or 0):.3f} "
                              f"frac>{1/phi():.3f}={(r['attn'].get('frac_above_phi_inv') or 0):.2f} "
                              f"H72={(r['hidden'].get('H72_norm_mean') or 0):.3f}", flush=True)

        theta = 1.0 / phi()
        wide = [r["attn"].get("H_norm_mean") for r in p1_runs
                if r["class"] == "wide" and r["attn"].get("H_norm_mean") is not None]
        narrow = [r["attn"].get("H_norm_mean") for r in p1_runs
                  if r["class"] == "narrow" and r["attn"].get("H_norm_mean") is not None]
        ent = [r["attn"].get("H_norm_min") for r in p1_runs if r["attn"].get("H_norm_min") is not None] + \
              [r["attn"].get("H_norm_max") for r in p1_runs if r["attn"].get("H_norm_max") is not None]

        def acc_at(th, direction="wide_above"):
            hit = n = 0
            for cls, vals in (("wide", wide), ("narrow", narrow)):
                for x in vals:
                    pred = "wide" if (x > th) == (direction == "wide_above") else "narrow"
                    hit += int(pred == cls)
                    n += 1
            return hit / n if n else 0.0

        rng = np.random.default_rng(SEED_BASE + 7)
        rand_ths = rng.uniform(0.2, 0.8, 1000)
        rand_acc = np.array([max(acc_at(float(t), "wide_above"), acc_at(float(t), "wide_below"))
                             for t in rand_ths])
        a_up, a_dn = acc_at(theta, "wide_above"), acc_at(theta, "wide_below")
        acc_phi = max(a_up, a_dn)
        direction = "wide_above" if a_up >= a_dn else "wide_below"

        out["P1"] = {
            "n_sessions": len(p1_runs),
            "coverage": {"min": float(np.min(ent)), "max": float(np.max(ent)),
                         "mean": float(np.mean([r["attn"].get("H_norm_mean") or 0 for r in p1_runs])),
                         "frac_above_phi_inv": float(np.mean(
                             [r["attn"].get("frac_above_phi_inv") or 0 for r in p1_runs]))},
            "session_auc_wide_vs_narrow": auc(wide, narrow),
            "p_wide_vs_narrow": mwu(wide, narrow),
            "acc_at_phi_inv": acc_phi, "direction": direction,
            "acc_random_mean": float(rand_acc.mean()),
            "acc_random_best": float(rand_acc.max()),
            "percentile_of_phi_inv": float((rand_acc < acc_phi).mean()),
            "dip": kde_dip([r["attn"]["H_norm_mean"] for r in p1_runs], theta),
            "runs": [{k: r[k] for k in ("class", "prompt_idx", "temperature", "rep",
                                        "seed", "text", "attn", "hidden")}
                     for r in p1_runs],
        }
        print(f"  -> coverage {out['P1']['coverage']['min']:.3f}-"
              f"{out['P1']['coverage']['max']:.3f}, AUC {out['P1']['session_auc_wide_vs_narrow']}, "
              f"p {out['P1']['p_wide_vs_narrow']}", flush=True)

    # ── P2: Relay-Lokalität ────────────────────────────────────────────
    if "P2" in subs and basis is not None:
        print("[P2] Relay-Arme ...", flush=True)
        p2_runs = []
        layers_post = list(range(27, 35))
        for arm, sign, alpha, vec in [("baseline", 0, 0.0, None),
                                      ("wide", +1, 0.40, None),
                                      ("narrow", -1, 0.40, None),
                                      ("rand_plus", +1, 0.40, d_rand)]:
            for pi, p in enumerate(WIDE):
                for rep in range(2):
                    seed = SEED_BASE + 5000 + len(p2_runs) * 10
                    steer = None if sign == 0 else RelaySteer(
                        tm, vec if vec is not None else dwidth, layer=26,
                        sign=sign, alpha=alpha)
                    r = session(model, tok, format_prompt(tok, p), hprobe, aprobe,
                                basis, dwidth, steer=steer, temperature=0.6, seed=seed)
                    r.update({"arm": arm, "prompt_idx": pi, "rep": rep, "seed": seed,
                              "prompt": p})
                    p2_runs.append(r)
                    print(f"  [P2 {arm:9s} p{pi} #{rep}] "
                          f"H_attn_post={layer_mean(r['attn'], layers_post):.3f} "
                          f"norm={r['hidden'].get('mean_hidden_norm', 0):.2f} "
                          f"H72={r['hidden'].get('H72_norm_mean', 0):.3f}", flush=True)

        def series(arm, key, fn):
            return [fn(r) for r in p2_runs if r["arm"] == arm]

        att = lambda r: layer_mean(r["attn"], layers_post)
        nrm = lambda r: r["hidden"].get("mean_hidden_norm")
        h72 = lambda r: r["hidden"].get("H72_norm_mean")
        rho = lambda r: r["hidden"].get("rho_perp")
        h72p = lambda r: r["hidden"].get("H72_perp_mean")

        out["P2"] = {
            **{name: {a: (float(np.mean(v)) if (v := [x for x in series(a, n, fn) if x is not None]) else None)
                      for a in ("baseline", "wide", "narrow", "rand_plus")}
               for name, n, fn in (("attn", "attn", att), ("norm", "norm", nrm),
                                   ("H72", "h72", h72), ("rho_perp", "rho", rho),
                                   ("H72_perp", "h72p", h72p))},
            "p": {
                "attn_wide_vs_narrow": mwu(series("wide", "attn", att),
                                           series("narrow", "attn", att)),
                "attn_wide_vs_rand": mwu(series("wide", "attn", att),
                                         series("rand_plus", "attn", att)),
                "norm_wide_vs_narrow": mwu(series("wide", "norm", nrm),
                                           series("narrow", "norm", nrm)),
                "norm_wide_vs_rand": mwu(series("wide", "norm", nrm),
                                         series("rand_plus", "norm", nrm)),
                "H72_wide_vs_narrow": mwu(series("wide", "h72", h72),
                                          series("narrow", "h72", h72)),
                "H72perp_wide_vs_narrow": mwu(series("wide", "h72p", h72p),
                                              series("narrow", "h72p", h72p)),
                "rho_wide_vs_narrow": mwu(series("wide", "rho", rho),
                                          series("narrow", "rho", rho)),
            },
            "layers_post_span": layers_post,
            "runs": [{k: r[k] for k in ("arm", "prompt_idx", "rep", "seed", "text",
                                        "attn", "hidden")} for r in p2_runs],
        }
        print(json.dumps(out["P2"]["p"], indent=1), flush=True)

    # ── P3: H72-Paradox ────────────────────────────────────────────────
    if "P3" in subs and "P2" in out:
        p2 = out["P2"]
        p = p2["p"]
        c = out["calibration"]
        inversion = (p["H72_wide_vs_narrow"] is not None and p["H72_wide_vs_narrow"] < 0.05
                     and p2["H72"]["wide"] < p2["H72"]["narrow"])
        cleared = (p["H72perp_wide_vs_narrow"] is None
                   or p["H72perp_wide_vs_narrow"] > 0.05
                   or p2["H72_perp"]["wide"] >= p2["H72_perp"]["narrow"])
        leak = (p["rho_wide_vs_narrow"] is not None and p["rho_wide_vs_narrow"] < 0.05
                and p2["rho_perp"]["wide"] > p2["rho_perp"]["narrow"])
        aligned = c.get("max_cos_dwidth_PC", 0.0) >= 0.50
        out["P3"] = {
            "inversion_replicates": bool(inversion),
            "orthogonal_leak": bool(leak),
            "axis_alignment": bool(aligned),
            "inversion_cleared_by_H72_perp": bool(cleared),
            "verdict": ("Achsen-Alignment" if aligned and cleared and inversion else
                        ("orthogonales Leck" if leak and not cleared and not aligned else
                         "ungeklärt")),
        }
        print(f"  -> {out['P3']['verdict']}", flush=True)

    hprobe.detach()
    aprobe.detach()
    return out


# ── Urteile ─────────────────────────────────────────────────────────────────
def verdicts(out: dict) -> dict:
    v = {}
    if "O2b" in out:
        o = out["O2b"]
        v["O2b_alignierter_trichter"] = (
            f"FALSIFIED — aligniert ist kein Ausreißer (Perzentil perm "
            f"{o['ensembles']['perm']['percentile_of_aligned']*100:.0f} %, rand "
            f"{o['ensembles']['rand']['percentile_of_aligned']*100:.0f} %); "
            f"t_cross aligniert {o['reference_t_cross']:.1f} vs rand "
            f"{o['ensembles']['rand']['t_cross_mean']:.1f} (schneller) und perm "
            f"{o['ensembles']['perm']['t_cross_mean']:.1f}. Ein generisches "
            "Potential koppelt schneller; S4 ist kein Maximal-Koppler."
            if not o["aligned_is_outlier"] else
            "alignierter Trichter ist ein Ausreißer (≤5 %).")
    if "P1" in out:
        p = out["P1"]
        v["P1a_abdeckung"] = ("CONFIRMED" if p["coverage"]["min"] < 0.05
                              and p["coverage"]["max"] > 0.90 else
                              f"FALSIFIED ({p['coverage']['min']:.3f}-{p['coverage']['max']:.3f})")
        v["P1b_trennschaerfe"] = (
            "CONFIRMED" if (p["session_auc_wide_vs_narrow"] or 0) >= 0.70
            and p["percentile_of_phi_inv"] >= 0.95 else
            f"FALSIFIED (AUC {p['session_auc_wide_vs_narrow']}, "
            f"Perzentil {p['percentile_of_phi_inv']*100:.0f} %, acc {p['acc_at_phi_inv']:.3f} "
            f"vs Zufalls-Mittel {p['acc_random_mean']:.3f})")
        d = p["dip"]
        v["P1c_bimodalitaet"] = (
            f"CONFIRMED (Dip bei {d['dip_theta']:.3f}, Tiefe {d['dip_depth']:.2f})"
            if d.get("dip_theta") is not None and d.get("dip_depth", 0) >= 0.25 else
            f"FALSIFIED (kein Dip ≥0,25 bei Φ⁻¹; Moden {d.get('n_modes')}, "
            f"Tiefe {d.get('dip_depth', 0):.2f})")
    if "P2" in out and "P3" in out:
        p = out["P2"]["p"]
        attn = (p["attn_wide_vs_narrow"] is not None and p["attn_wide_vs_narrow"] < 0.05
                and out["P2"]["attn"]["wide"] > out["P2"]["attn"]["narrow"])
        spec = (p["attn_wide_vs_rand"] is not None and p["attn_wide_vs_rand"] < 0.05)
        norm = (p["norm_wide_vs_narrow"] is not None and p["norm_wide_vs_narrow"] < 0.05)
        norm_spec = (p["norm_wide_vs_rand"] is not None and p["norm_wide_vs_rand"] < 0.05)
        v["P2_lokalitaet"] = (
            "Attention-Breite" if attn and spec else
            ("Norm-only" if norm and norm_spec and not attn else
             "kein lokalisierbarer Effekt"))
        v["P3_H72_parodox"] = out["P3"]["verdict"]
    if "P4" in out:
        v["P4_72D"] = out["P4"]["verdict"]
    return v


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="O2b,P1,P2,P3,P4")
    ap.add_argument("--out", default="results/eval_P_attention_horizon_gemma4.json")
    ap.add_argument("--rejudge", default=None)
    a = ap.parse_args()

    if a.rejudge:
        with open(a.rejudge) as f:
            out = json.load(f)
        out["verdicts"] = verdicts(out)
        with open(a.rejudge, "w") as f:
            json.dump(out, f, indent=1)
        for k, val in out["verdicts"].items():
            print(f"  {k}: {val}")
        return

    subs = [s.strip().upper() for s in a.subs.split(",")]
    out = {}
    if "O2B" in subs:
        print("[O2b] Trichter-Ensembles ...", flush=True)
        out["O2b"] = funnel_alignment_suite(n_random=200)
        print(json.dumps(out["O2b"], indent=1), flush=True)

    out.update(await run([s for s in subs if s != "O2B"]))
    out["verdicts"] = verdicts(out)

    os.makedirs("results", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\n" + "=" * 70)
    for k, val in out["verdicts"].items():
        print(f"  {k}: {val}")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
