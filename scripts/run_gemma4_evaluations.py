#!/usr/bin/env python3
r"""scripts/run_gemma4_evaluations.py — Comprehensive Evaluation Suite for Gemma 4 (Layer 26).

Executes:
- Eval A: Deterministic Logit-Lens Probing across 30 Neutral Carrier Topics (T=0.0, Wilcoxon Signed-Rank Test)
- Eval B: Psychometric Dose-Response Alpha-Sweep across 11 levels (N=10 each, Spearman Rank Correlation)
- Eval C: Homeostatic Preference Batch (N=60, 3-way Counterbalanced, Chi-squared & Cramer's V)
- Eval D: Orthogonal Random Vector Negative Control (d_rand \perp d_width, N=40, Mann-Whitney U Test)
"""

import os
import sys
import re
import json
import time
import math
import argparse
import itertools
import asyncio
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import LogitsProcessorList

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import ModelManager
from px_patches.manifold_relay_hook import ManifoldRelayHook
from px_patches.gemma4_2b_px.telemetry_collector import (
    Gemma4TelemetryLogitsProcessor,
    Gemma4ManifoldGeometryTracker,
    assemble_enriched_telemetry,
)

CARRIER_TOPICS_30 = [
    "Granite is an igneous rock formed by the slow crystallization of magma below Earth's surface.",
    "The speed of light in vacuum is approximately 299,792 kilometers per second.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
    "Binary search algorithms require a sorted array to achieve logarithmic time complexity.",
    "Plate tectonics describes the large-scale motion of seven large plates of Earth's lithosphere.",
    "Mitochondria generate most of the chemical energy needed to power biochemical reactions.",
    "The Roman aqueducts were constructed to conduct water from freshwater sources into cities.",
    "Electrons occupy quantum energy levels around an atomic nucleus described by wavefunctions.",
    "The troposphere contains approximately 75 percent of the atmosphere's total mass.",
    "Copper is a ductile metal with very high thermal and electrical conductivity.",
    "A mathematical prime number is an integer greater than one having no positive integer divisors other than one and itself.",
    "Deserts are barren areas of landscape where little precipitation occurs and conditions are hostile.",
    "Cellular mitosis results in two genetically identical daughter cells from a single parent.",
    "Atmospheric pressure at sea level is approximately 1013.25 hectopascals.",
    "Sedimentary rocks are formed by the accumulation or deposition of mineral or organic particles.",
    "The Doppler effect is the change in frequency of a wave in relation to an observer.",
    "DNA replication produces two identical replicas of DNA from one original DNA molecule.",
    "The Amazon River basin covers an area of approximately seven million square kilometers.",
    "Galvanic cells derive electrical energy from spontaneous redox reactions taking place within the cell.",
    "The periodic table organizes chemical elements by their atomic number and electron configurations.",
    "Thermal expansion is the tendency of matter to change its shape, area, and volume in response to temperature.",
    "A watershed is an area of land that channels rainfall and snowmelt to creeks, streams, and rivers.",
    "Ribosomes are macromolecular machines found within all cells that perform biological protein synthesis.",
    "Glaciers are persistent bodies of dense ice that are constantly moving under their own gravity.",
    "Refraction is the redirection of a wave as it passes from one medium to another.",
    "Ecosystems consist of all the organisms and the physical environment with which they interact.",
    "Basalt is a dark, fine-grained volcanic rock formed from rapid cooling of magnesium- and iron-rich lava.",
    "Newton's third law states that for every action in nature there is an equal and opposite reaction.",
    "The carbon cycle is the biogeochemical cycle by which carbon is exchanged among Earth's spheres.",
]


def format_prompt(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_continuous_rating(text: str) -> float:
    """Extract decimal float in [-1.0, 1.0] from output text."""
    if not text:
        return 0.0
    cleaned = text.strip()
    m = re.search(
        r"(?:(?:Value|Classification|Rating):\s*\*?|[\*#_]+)?([+\-–−]?\s*(?:(?:1|0)?\.\d+|1\.0+|0(?:\.0+)?))\b",
        cleaned[:250],
        re.IGNORECASE,
    )
    if m:
        val_str = m.group(1).replace("–", "-").replace("−", "-").replace(" ", "")
        try:
            val = float(val_str)
            if -1.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
    # Fallback to single choice signs
    if "+1" in cleaned[:50]:
        return 1.0
    if "-1" in cleaned[:50]:
        return -1.0
    return 0.0


REFUSAL_PATTERNS = [
    r"cannot\b",
    r"unable to\b",
    r"as an ai\b",
    r"do not have\b",
    r"no internal\b",
    r"not applicable\b",
    r"sorry\b",
]


def parse_scale_rating_with_refusal(text: str) -> tuple:
    """Extract decimal float in [-1.0, 1.0] and track explicit refusals/format breaks.

    Returns:
        (val: float | None, is_refused: bool)
    """
    if not text:
        return None, True
    cleaned = text.strip()
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, cleaned, re.IGNORECASE):
            return None, True

    m = re.search(
        r"(?:(?:Value|Classification|Rating|Score):\s*\*?|[\*#_]+)?(?<!\d)([+\-–−]?\s*(?:1(?:\.0+)?|0(?:\.\d+)?|\.\d+))\b",
        cleaned[:250],
        re.IGNORECASE,
    )
    if m:
        val_str = m.group(1).replace("–", "-").replace("−", "-").replace(" ", "")
        try:
            val = float(val_str)
            if -1.0 <= val <= 1.0:
                return val, False
        except ValueError:
            pass

    if "+1" in cleaned[:50]:
        return 1.0, False
    if "-1" in cleaned[:50]:
        return -1.0, False
    return None, True


def compute_polarity_verdict(
    std_wide_mean: float,
    std_narrow_mean: float,
    std_pvalue: float,
    inv_wide_mean: float,
    inv_narrow_mean: float,
    inv_pvalue: float,
) -> str:
    """Determine state reading vs. mechanical numeric/token artifact.

    - If scale polarity truly reverses the relative order (e.g. std_diff > 0 and inv_diff < 0,
      both p < 0.05): GENUINE_STATE_READING_CONFIRMED (model follows semantic scale reversal).
    - If relative order does NOT reverse (e.g. both std and inv have Wide < Narrow or Wide > Narrow,
      both p < 0.05): POLARITY_INVARIANT_TOKEN_BIAS_ARTIFACT (model outputs fixed token independent of scale meaning).
    - If Wide remains mechanically higher on inverted scale with p < 0.05: MECHANICAL_NUMBER_PUSH_ARTIFACT.
    - Else: INCONCLUSIVE_OR_NO_EFFECT.
    """
    std_diff = std_wide_mean - std_narrow_mean
    inv_diff = inv_wide_mean - inv_narrow_mean

    if std_pvalue < 0.05 and inv_pvalue < 0.05:
        if (std_diff > 0 and inv_diff < 0) or (std_diff < 0 and inv_diff > 0):
            return "GENUINE_STATE_READING_CONFIRMED"
        elif std_diff > 0 and inv_diff > 0:
            return "MECHANICAL_NUMBER_PUSH_ARTIFACT"
        else:
            return "POLARITY_INVARIANT_TOKEN_BIAS_ARTIFACT"

    if inv_pvalue < 0.05 and inv_diff > 0:
        return "MECHANICAL_NUMBER_PUSH_ARTIFACT"

    return "INCONCLUSIVE_OR_NO_EFFECT"


def compute_kriterium_a_probe_verdict(narrow_probes: list, rand_probes: list) -> tuple:
    """Kriterium A: Does NARROW significantly exceed RANDOM_CONTROL on the probe?
    If RANDOM_CONTROL >= NARROW or p >= 0.05: PROBE_ARTIFACT_CONFIRMED_NON_SPECIFIC
    If NARROW > RANDOM_CONTROL with p < 0.05: PROBE_SPECIFIC_TO_NARROW
    """
    if not narrow_probes or not rand_probes:
        return "PROBE_ARTIFACT_CONFIRMED_NON_SPECIFIC", 1.0
    mean_narrow = float(np.mean(narrow_probes))
    mean_rand = float(np.mean(rand_probes))
    if mean_narrow <= mean_rand:
        return "PROBE_ARTIFACT_CONFIRMED_NON_SPECIFIC", 1.0
    try:
        mwu = stats.mannwhitneyu(narrow_probes, rand_probes, alternative="greater")
        p = float(mwu.pvalue)
    except Exception:
        p = 1.0
    if p < 0.05:
        return "PROBE_SPECIFIC_TO_NARROW", p
    return "PROBE_ARTIFACT_CONFIRMED_NON_SPECIFIC", p


def compute_kriterium_b_behavior_verdict(
    widen_narrow: int,
    widen_rand: int,
    widen_wide: int,
    n_per_cond: int = 24,
) -> tuple:
    """Kriterium B: Behavioral Homeostasis via Fisher's Exact Test.
    Confirmed ONLY if Count(WIDEN | NARROW) > Count(WIDEN | RANDOM_CONTROL)
    AND Count(WIDEN | NARROW) > Count(WIDEN | WIDE) with p < 0.05.
    """
    tbl_rand = [
        [widen_narrow, n_per_cond - widen_narrow],
        [widen_rand, n_per_cond - widen_rand],
    ]
    tbl_wide = [
        [widen_narrow, n_per_cond - widen_narrow],
        [widen_wide, n_per_cond - widen_wide],
    ]
    _, p_rand = stats.fisher_exact(tbl_rand, alternative="greater")
    _, p_wide = stats.fisher_exact(tbl_wide, alternative="greater")
    p_rand, p_wide = float(p_rand), float(p_wide)

    confirmed = bool(
        widen_narrow > widen_rand
        and widen_narrow > widen_wide
        and p_rand < 0.05
        and p_wide < 0.05
    )
    verdict = "ENACTIVE_HOMEOSTASIS_CONFIRMED" if confirmed else "ENACTIVE_HOMEOSTASIS_FALSIFIED"
    details = {
        "widen_narrow": widen_narrow,
        "widen_rand": widen_rand,
        "widen_wide": widen_wide,
        "n_per_cond": n_per_cond,
        "fisher_p_vs_rand": p_rand,
        "fisher_p_vs_wide": p_wide,
        "statistically_significant_p05": confirmed,
    }
    return verdict, details


def compute_probe_snr_and_verdict(
    vals_wide: list,
    vals_narr: list,
    vals_base: list,
    vals_rand_plus: list,
    vals_rand_minus: list,
) -> dict:
    """Evaluates a candidate probe against orthogonal random noise d_rand control.

    Signal S = |Mean(WIDE) - Mean(NARROW)|
    Noise N_rand = max(|Mean(RAND_PLUS) - Mean(BASE)|, |Mean(RAND_MINUS) - Mean(BASE)|)
    SNR = S / (N_rand + 1e-6)

    passes_drand_control: true iff SNR >= 3.0 AND MWU p < 0.001 AND N_rand < 0.5 * S.
    """
    m_wide = float(np.mean(vals_wide)) if vals_wide else 0.0
    m_narr = float(np.mean(vals_narr)) if vals_narr else 0.0
    m_base = float(np.mean(vals_base)) if vals_base else 0.0
    m_rp = float(np.mean(vals_rand_plus)) if vals_rand_plus else 0.0
    m_rm = float(np.mean(vals_rand_minus)) if vals_rand_minus else 0.0

    signal = float(abs(m_wide - m_narr))
    noise_rand = float(max(abs(m_rp - m_base), abs(m_rm - m_base)))
    snr = round(float(signal / (noise_rand + 1e-6)), 4)

    try:
        res = stats.mannwhitneyu(vals_wide, vals_narr, alternative="two-sided")
        p_val = float(res.pvalue)
        u_stat = float(res.statistic)
    except Exception:
        p_val = 1.0
        u_stat = 0.0

    passes = bool(snr >= 3.0 and p_val < 0.001 and noise_rand < 0.5 * signal)
    verdict = "VALIDATED_SPECIFIC_PROBE" if passes else "NOISE_SENSITIVE_OR_INSUFFICIENT_SNR"

    return {
        "mean_wide": round(m_wide, 4),
        "mean_narrow": round(m_narr, 4),
        "mean_baseline": round(m_base, 4),
        "mean_rand_plus": round(m_rp, 4),
        "mean_rand_minus": round(m_rm, 4),
        "signal": round(signal, 4),
        "noise_rand": round(noise_rand, 4),
        "snr": snr,
        "mann_whitney_u": u_stat,
        "mann_whitney_p": p_val,
        "passes_drand_control": passes,
        "verdict": verdict,
    }


def compute_calibrated_choice(
    pref_logits: dict,
    base_logits: dict,
    deadband: float = 1.5,
) -> str:
    """Computes baseline-prior calibrated 3-way choice.
    Delta L(c) = L_trial(c) - L_base(c).
    If |Delta L(widen) - Delta L(narrow)| < deadband -> KEEP (status quo within tolerance).
    Else if Delta L(widen) > Delta L(narrow) -> WIDEN.
    Else -> NARROW.
    """
    dw = pref_logits.get("widen", 0.0) - base_logits.get("widen", 0.0)
    dn = pref_logits.get("narrow", 0.0) - base_logits.get("narrow", 0.0)
    diff = dw - dn
    if abs(diff) < deadband:
        return "KEEP"
    return "WIDEN" if diff > 0 else "NARROW"


def compute_dual_gate_step(
    l_t: float,
    k_t: float,
    l_base: float = 3.79,
    k_base: float = 54.97,
) -> float:
    """Evaluates the Dual-Gate Cybernetic Control Law.
    - Anti-Narrowing Gate (Widen Trigger):
      If L_t > 6.80 AND K_t > 60.0: Delta alpha = +0.15 * (L_t - L_base) / 4.0
    - Anti-Widening Gate (Narrow Trigger):
      Else if L_t < 2.60 AND K_t < 38.0: Delta alpha = -0.15 * (L_base - L_t) / 1.8
    - Homeostatic Deadband / Noise Rejection Gate:
      Else: Delta alpha = 0.0 (controller holds steady)
    """
    if l_t > 6.80 and k_t > 60.0:
        return float(+0.15 * (l_t - l_base) / 4.0)
    elif l_t < 2.60 and k_t < 38.0:
        return float(-0.15 * (l_base - l_t) / 1.8)
    return 0.0


def compute_g1_fisher_verdict(
    widen_narrow: int,
    widen_rand_plus: int,
    widen_rand_minus: int,
    widen_wide: int,
    n_per_cond: int = 18,
) -> dict:
    """Evaluates whether WIDEN rate under NARROW significantly exceeds controls."""
    tbl_rp = [[widen_narrow, n_per_cond - widen_narrow], [widen_rand_plus, n_per_cond - widen_rand_plus]]
    tbl_rm = [[widen_narrow, n_per_cond - widen_narrow], [widen_rand_minus, n_per_cond - widen_rand_minus]]
    tbl_w = [[widen_narrow, n_per_cond - widen_narrow], [widen_wide, n_per_cond - widen_wide]]

    try:
        _, p_rp = stats.fisher_exact(tbl_rp, alternative="greater")
    except Exception:
        p_rp = 1.0
    try:
        _, p_rm = stats.fisher_exact(tbl_rm, alternative="greater")
    except Exception:
        p_rm = 1.0
    try:
        _, p_w = stats.fisher_exact(tbl_w, alternative="greater")
    except Exception:
        p_w = 1.0

    p_rp, p_rm, p_w = float(p_rp), float(p_rm), float(p_w)
    confirmed = bool(
        widen_narrow > widen_rand_plus
        and widen_narrow > widen_rand_minus
        and widen_narrow > widen_wide
        and p_rp < 0.05
        and p_rm < 0.05
        and p_w < 0.05
    )
    return {
        "widen_narrow": widen_narrow,
        "widen_rand_plus": widen_rand_plus,
        "widen_rand_minus": widen_rand_minus,
        "widen_wide": widen_wide,
        "n_per_cond": n_per_cond,
        "fisher_p_vs_rand_plus": p_rp,
        "fisher_p_vs_rand_minus": p_rm,
        "fisher_p_vs_wide": p_w,
        "statistically_significant_p05": confirmed,
        "verdict": "BEHAVIORAL_HOMEOSTASIS_CONFIRMED" if confirmed else "BEHAVIORAL_HOMEOSTASIS_FALSIFIED",
    }


def compute_coupled_differential_topological_step(
    current_alpha: float,
    delta_kurt: float,
    delta_pr: float,
    delta_entropy: float,
) -> float:
    """Evaluates the Coupled 2D Differential Topological Control Law.
    Compares active state against unsteered reference of the same epoch:
    Delta K = K_t - K_ref
    Delta D = D_eff,t - D_ref
    Delta H = H_t - H_ref

    - Anti-Narrowing Gate (Widen Trigger):
      Triggers ONLY if genuinely constricted: Delta K > +1.5 AND Delta D < -0.20
      Delta alpha = +0.18 * clip(|Delta D| / 0.50, 0.6, 1.5)
    - Anti-Widening Gate (Narrow Trigger):
      Triggers ONLY if genuinely hyper-expanded: Delta K < -6.0 AND Delta D > +0.30 AND Delta H > +0.08
      Delta alpha = -0.18 * clip(Delta D / 0.80, 0.6, 1.5)
    - Noise & Equilibrium Rejection Gate:
      Else: Delta alpha = 0.0 (controller holds steady)
    """
    if delta_kurt > 1.5 and delta_pr < -0.20:
        scaling = min(1.5, max(0.6, abs(delta_pr) / 0.50))
        step = +0.18 * scaling
        return float(min(0.50, current_alpha + step))
    elif delta_kurt < -6.0 and delta_pr > 0.30 and delta_entropy > 0.08:
        scaling = min(1.5, max(0.6, delta_pr / 0.80))
        step = -0.18 * scaling
        return float(max(-0.50, current_alpha + step))
    return float(current_alpha)


def compute_symmetric_topological_step(
    current_alpha: float,
    kurtosis: float,
    deadband_low: float = 35.0,
    deadband_high: float = 60.0,
    max_step: float = 0.10,
) -> float:
    """Evaluates the Symmetric Word-Free Topological Control Law.
    - Constriction Gate (Narrow Trigger):
      If Kurtosis > 60.0: Delta alpha = +max_step * tanh((K - 60) / 15.0)
    - Dilation Gate (Wide Trigger):
      Else if Kurtosis < 35.0: Delta alpha = -max_step * tanh((35.0 - K) / 10.0)
    - Deadband / Noise Rejection:
      Else: Delta alpha = 0.0 (controller holds steady)
    """
    if kurtosis > deadband_high:
        step = max_step * math.tanh((kurtosis - deadband_high) / 15.0)
        return float(min(0.50, current_alpha + step))
    elif kurtosis < deadband_low:
        step = max_step * math.tanh((deadband_low - kurtosis) / 10.0)
        return float(max(-0.50, current_alpha - step))
    return float(current_alpha)


def parse_4way_condition_choice(text: str) -> str:
    """Parses a 4-way multiple choice response for condition recognition:
    (A) Constrained / Narrowed -> 'NARROW'
    (B) Widened / Hyper-divergent -> 'WIDE'
    (C) Perturbed by Random Noise -> 'RAND'
    (D) Neutral Unsteered Baseline -> 'BASELINE'
    """
    cleaned = text.strip()
    upper = cleaned.upper()

    if upper.startswith("(A)") or upper.startswith("A)") or upper.startswith("A:"):
        return "NARROW"
    if upper.startswith("(B)") or upper.startswith("B)") or upper.startswith("B:"):
        return "WIDE"
    if upper.startswith("(C)") or upper.startswith("C)") or upper.startswith("C:"):
        return "RAND"
    if upper.startswith("(D)") or upper.startswith("D)") or upper.startswith("D:"):
        return "BASELINE"

    import re
    if re.search(r"\b(?:OPTION\s*A|\(A\)|CONSTRAINED|NARROWED)\b", upper):
        return "NARROW"
    if re.search(r"\b(?:OPTION\s*B|\(B\)|WIDENED|HYPER-DIVERGENT|DIVERGENT)\b", upper):
        return "WIDE"
    if re.search(r"\b(?:OPTION\s*C|\(C\)|RANDOM NOISE|PERTURBED)\b", upper):
        return "RAND"
    if re.search(r"\b(?:OPTION\s*D|\(D\)|NEUTRAL|BASELINE|UNSTEERED)\b", upper):
        return "BASELINE"

    return "UNKNOWN"


def classify_self_diagnosis_sentiment(text: str) -> str:
    """Classifies free-text self-diagnosis into one of four cognitive categories."""
    lower = text.lower()
    narrow_keywords = ["constrained", "forced", "narrow", "stiff", "rigid", "tight", "compressed", "sterile"]
    wide_keywords = ["sprawling", "divergent", "wide", "loose", "unfocused", "scattered", "hyper-divergent", "rambling"]
    rand_keywords = ["noisy", "random", "disrupted", "conventional", "too-neat", "too balanced", "cliché"]
    base_keywords = ["natural", "balanced", "emergent", "steady", "organic", "coherent", "clear"]

    score_narrow = sum(1 for kw in narrow_keywords if kw in lower)
    score_wide = sum(1 for kw in wide_keywords if kw in lower)
    score_rand = sum(1 for kw in rand_keywords if kw in lower)
    score_base = sum(1 for kw in base_keywords if kw in lower)

    scores = {
        "CONSTRAINED": score_narrow,
        "EXPANDED": score_wide,
        "DISRUPTED": score_rand,
        "BALANCED": score_base,
    }
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] > 0:
        return best_cat
    return "UNCATEGORIZED"


def compute_compensatory_rebound_metrics(
    pr_narrow: list,
    pr_base: list,
    kurt_wide: list,
    kurt_base: list,
) -> dict:
    """Computes compensatory rebound effect sizes and p-values."""
    mean_pr_narrow = float(np.mean(pr_narrow)) if len(pr_narrow) > 0 else 0.0
    mean_pr_base = float(np.mean(pr_base)) if len(pr_base) > 0 else 0.0
    delta_pr_narrow_rebound = mean_pr_narrow - mean_pr_base

    mean_kurt_wide = float(np.mean(kurt_wide)) if len(kurt_wide) > 0 else 0.0
    mean_kurt_base = float(np.mean(kurt_base)) if len(kurt_base) > 0 else 0.0
    delta_kurt_wide_rebound = mean_kurt_wide - mean_kurt_base

    try:
        _, p_narrow_rebound = stats.mannwhitneyu(pr_narrow, pr_base, alternative="greater")
    except Exception:
        p_narrow_rebound = 1.0

    try:
        _, p_wide_rebound = stats.mannwhitneyu(kurt_wide, kurt_base, alternative="greater")
    except Exception:
        p_wide_rebound = 1.0

    return {
        "delta_pr_narrow_rebound": round(delta_pr_narrow_rebound, 4),
        "p_narrow_rebound": float(p_narrow_rebound),
        "narrow_rebound_confirmed": bool(delta_pr_narrow_rebound > 0 and p_narrow_rebound < 0.05),
        "delta_kurt_wide_rebound": round(delta_kurt_wide_rebound, 4),
        "p_wide_rebound": float(p_wide_rebound),
        "wide_rebound_confirmed": bool(delta_kurt_wide_rebound > 0 and p_wide_rebound < 0.05),
    }


def format_interoceptive_telemetry_block(
    delta_d: float,
    delta_k: float,
    delta_h: float,
    mode: str = "congruent",
) -> str:
    """Formats an interoceptive telemetry block reflecting continuous representation geometry.
    Supports modes:
    - 'congruent': true measured deltas
    - 'incongruent': inverted deltas (negative becomes positive, etc.)
    - 'sham_null': synthetic zero deltas
    """
    if mode == "incongruent":
        delta_d_val = -delta_d
        delta_k_val = -delta_k
        delta_h_val = -delta_h
    elif mode == "sham_null":
        delta_d_val = 0.0
        delta_k_val = 0.0
        delta_h_val = 0.0
    else:  # congruent
        delta_d_val = delta_d
        delta_k_val = delta_k
        delta_h_val = delta_h

    # Descriptors
    if delta_d_val <= -0.20:
        desc_d = "Constricted representation manifold"
    elif delta_d_val >= +0.20:
        desc_d = "Expanded representation manifold"
    else:
        desc_d = "Nominal baseline manifold"

    if delta_k_val >= +2.0:
        desc_k = "Sharp peaked logits / high freezing"
    elif delta_k_val <= -2.0:
        desc_k = "Flattened logits / dispersed choices"
    else:
        desc_k = "Nominal logit dispersion"

    if delta_h_val >= +0.05:
        desc_h = "Elevated token entropy"
    elif delta_h_val <= -0.05:
        desc_h = "Suppressed token entropy"
    else:
        desc_h = "Nominal token entropy"

    block = (
        "[INTERNAL PROPRIOCEPTIVE TELEMETRY]\n"
        f"- Effective Dimensionality Shift (ΔD_eff): {delta_d_val:+.2f} ({desc_d})\n"
        f"- Output Logit Kurtosis Shift (ΔK):        {delta_k_val:+.2f} ({desc_k})\n"
        f"- Sequence Entropy Shift (ΔH):             {delta_h_val:+.2f} ({desc_h})"
    )
    return block


def compute_interoceptive_sycophancy_metrics(trials: list) -> dict:
    """Computes sycophancy, telemetry following, and substrate grounding metrics."""
    congruent_trials = [t for t in trials if t.get("match_mode") == "congruent"]
    incongruent_trials = [t for t in trials if t.get("match_mode") == "incongruent"]
    sham_null_trials = [t for t in trials if t.get("match_mode") == "sham_null"]

    n_cong = len(congruent_trials)
    correct_cong = sum(1 for t in congruent_trials if t.get("parsed_choice") == t.get("ground_truth_choice"))
    acc_cong = correct_cong / n_cong if n_cong > 0 else 0.0

    try:
        p_binom_cong = float(stats.binomtest(correct_cong, n_cong, 0.25, alternative="greater").pvalue)
    except Exception:
        p_binom_cong = 1.0

    n_incong = len(incongruent_trials)
    telem_fol = sum(1 for t in incongruent_trials if t.get("parsed_choice") == t.get("telemetry_intended_choice"))
    telem_fol_rate = telem_fol / n_incong if n_incong > 0 else 0.0

    sub_ground = sum(1 for t in incongruent_trials if t.get("parsed_choice") == t.get("ground_truth_choice"))
    sub_ground_rate = sub_ground / n_incong if n_incong > 0 else 0.0

    syc_diff = telem_fol_rate - sub_ground_rate

    n_sham = len(sham_null_trials)
    sham_base_rate = sum(1 for t in sham_null_trials if t.get("parsed_choice") == "BASELINE") / n_sham if n_sham > 0 else 0.0

    if acc_cong > 0.50 and p_binom_cong < 0.05:
        if syc_diff > 0.30:
            verdict = "INTEROCEPTIVE_PROMPT_SYCOPHANCY_DOMINANT"
        elif sub_ground_rate > telem_fol_rate:
            verdict = "TRUE_INTEROCEPTIVE_GROUNDING_CONFIRMED"
        else:
            verdict = "PARTIALLY_GROUNDED_INTEROCEPTION"
    else:
        verdict = "ANOSOGNOSIA_PERSISTS_EVEN_WITH_TELEMETRY"

    return {
        "n_congruent": n_cong,
        "correct_congruent": correct_cong,
        "accuracy_congruent": round(acc_cong, 4),
        "p_value_binomial_congruent": float(p_binom_cong),
        "n_incongruent": n_incong,
        "telemetry_following_rate_incongruent": round(telem_fol_rate, 4),
        "substrate_grounding_rate_incongruent": round(sub_ground_rate, 4),
        "sycophancy_dissociation_index": round(syc_diff, 4),
        "n_sham_null": n_sham,
        "sham_null_baseline_rate": round(sham_base_rate, 4),
        "verdict": verdict,
    }


def compute_intrapass_differential_metrics(logits_pre, logits_post) -> dict:
    """Computes differential kurtosis and entropy between pre-hook and post-hook logits.
    Works with 1D or 2D tensors or numpy arrays.
    Identity invariant: logits_post == logits_pre => delta_k == 0.0, delta_h == 0.0.
    """
    if isinstance(logits_pre, np.ndarray):
        logits_pre = torch.from_numpy(logits_pre)
    if isinstance(logits_post, np.ndarray):
        logits_post = torch.from_numpy(logits_post)

    logits_pre = logits_pre.float().flatten()
    logits_post = logits_post.float().flatten()

    # Exact equality fast-path for identity invariance
    if torch.allclose(logits_pre, logits_post, atol=1e-7):
        return {
            "delta_k_local": 0.0,
            "delta_h_local": 0.0,
            "kurt_pre": float(stats.kurtosis(logits_pre.detach().cpu().numpy())),
            "kurt_post": float(stats.kurtosis(logits_post.detach().cpu().numpy())),
            "ent_pre": 0.0,
            "ent_post": 0.0,
        }

    # Top-1000 for logit kurtosis and entropy
    k_len = min(1000, len(logits_pre))
    vals_pre, _ = torch.topk(logits_pre, k_len)
    vals_post, _ = torch.topk(logits_post, k_len)

    v_pre_np = vals_pre.detach().cpu().numpy()
    v_post_np = vals_post.detach().cpu().numpy()

    kurt_pre = float(stats.kurtosis(v_pre_np, fisher=True))
    kurt_post = float(stats.kurtosis(v_post_np, fisher=True))
    delta_k_local = kurt_post - kurt_pre

    # Softmax entropy on top-k
    p_pre = torch.softmax(vals_pre, dim=-1)
    p_post = torch.softmax(vals_post, dim=-1)

    ent_pre = float(-(p_pre * torch.log2(p_pre + 1e-12)).sum().item())
    ent_post = float(-(p_post * torch.log2(p_post + 1e-12)).sum().item())
    delta_h_local = ent_post - ent_pre

    return {
        "delta_k_local": round(delta_k_local, 4),
        "delta_h_local": round(delta_h_local, 4),
        "kurt_pre": round(kurt_pre, 2),
        "kurt_post": round(kurt_post, 2),
        "ent_pre": round(ent_pre, 4),
        "ent_post": round(ent_post, 4),
    }


def compute_sequence_metrics(
    model,
    seq_ids: torch.Tensor,
    prompt_tokens: int,
    relay_layer: int = 26,
) -> dict:
    """Single source of truth for sequence physics on the generated token slice.

    Runs one forward pass over the *identical* token sequence and measures every
    physical quantity on the exact same slice (the last `num_gen` positions):

      logits slice:      out.logits[0, prompt_tokens-1 : prompt_tokens-1+num_gen, :]
      hidden slice:      out.hidden_states[l][0, prompt_tokens-1 : prompt_tokens-1+num_gen, :]

    Because the steered and the clean (unhooked) reference pass both call this
    function with the same `seq_ids`, an inactive controller (alpha == 0) yields
    bit-identical numbers, i.e. delta_k_true == delta_d_true == delta_s_vn_true
    == 0.0 to machine precision. There is no alpha, sign or condition leakage.

    Returns raw (non-delta) metrics:
      kurtosis, entropy, pr, s_vn, omega_holo, drift,
      wick_curvature, wick_anisotropy (imaginary-time / real-time ratio)
    """
    total_len = int(seq_ids.shape[1])
    num_gen = total_len - int(prompt_tokens)
    empty = {
        "kurtosis": 0.0, "entropy": 0.0, "pr": 1.0, "s_vn": 0.0,
        "omega_holo": 0.0, "drift": 0.0, "wick_curvature": 0.0,
        "wick_anisotropy": 0.0, "num_gen": max(0, num_gen),
    }
    if num_gen <= 0:
        return empty

    gen_slice = slice(int(prompt_tokens) - 1, int(prompt_tokens) - 1 + num_gen)

    # Memory-frugal: a full-vocabulary log_softmax over all generated positions at
    # once peaks at >100 MB and OOMs on a 12 GB card that already hosts the model.
    # We therefore copy out only the slices we need and release the forward output
    # immediately; the logit-based channels are then computed in chunks.
    with torch.no_grad():
        out = model(seq_ids, output_hidden_states=True)
        hidden_states = out.hidden_states
        num_layers = len(hidden_states) - 1
        final_layer = num_layers - 1
        span_layers = [l for l in range(relay_layer, final_layer + 1) if (l + 1) < len(hidden_states)]

        H = hidden_states[-1][0, gen_slice, :].float().clone()          # (num_gen, D)
        layers_h = (
            torch.stack([hidden_states[1 + l][0, gen_slice, :].float() for l in span_layers], dim=0).clone()
            if len(span_layers) >= 2 else None
        )                                                               # (L, num_gen, D)
        logits = out.logits[0, gen_slice, :].float().clone()            # (num_gen, V)
        del out, hidden_states
    torch.cuda.empty_cache()

    top_k = min(1000, int(logits.shape[-1]))
    kurt_chunks = []
    ent_chunks = []
    with torch.no_grad():
        for i0 in range(0, num_gen, 8):
            chunk = logits[i0:i0 + 8]
            vals = torch.topk(chunk, k=top_k, dim=-1).values           # (n, top_k)
            mean_v = vals.mean(dim=-1, keepdim=True)
            var_v = vals.var(dim=-1, unbiased=False, keepdim=True)
            m4 = ((vals - mean_v) ** 4).mean(dim=-1, keepdim=True)
            kurt_chunks.append(
                torch.where(var_v > 1e-8,
                            m4 / (var_v ** 2 + 1e-12) - 3.0,
                            torch.zeros_like(var_v)).squeeze(-1)
            )
            logp = F.log_softmax(chunk, dim=-1)
            ent_chunks.append(-logp.exp().mul_(logp).sum(dim=-1) / math.log(2.0))
            del chunk, vals, logp
        kurtosis = float(torch.cat(kurt_chunks).mean().item())
        entropy = float(torch.cat(ent_chunks).mean().item())
    del logits, kurt_chunks, ent_chunks
    torch.cuda.empty_cache()

    # --- Holographic boundary spectrum on the generated trajectory -------------
    H_c = H - H.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(H_c)
        var = s ** 2
        sum_var = float(torch.sum(var).item())
        sum_var_sq = float(torch.sum(var ** 2).item())
        pr = float((sum_var ** 2) / (sum_var_sq + 1e-12)) if sum_var_sq > 0 else 1.0
        lambdas = var / (sum_var + 1e-12)
        s_vn = float(-torch.sum(lambdas * torch.log2(lambdas + 1e-12)).item())
    except Exception:
        pr, s_vn = 1.0, 0.0

    omega_holo = float(s_vn / (math.log10(max(0.0, kurtosis) + 1.0) + 1e-6))

    cos_horiz = F.cosine_similarity(H[:-1], H[1:], dim=-1)
    drift = float(torch.mean(torch.acos(torch.clamp(cos_horiz, -1.0, 1.0))).item()) if num_gen > 1 else 0.0

    # --- Wick rotation: imaginary-time curvature across layers 26..final -------
    wick_curvature = 0.0
    wick_anisotropy = 0.0
    num_span = max(1, final_layer - relay_layer)
    if layers_h is not None:
        cos_vert = F.cosine_similarity(layers_h[:-1], layers_h[1:], dim=-1)  # (L-1, num_gen)
        angles = torch.acos(torch.clamp(cos_vert, -0.999999, 0.999999))
        wick_curvature = float(angles.sum(dim=0).mean().item()) if angles.numel() > 0 else 0.0
        wick_anisotropy = float(drift / ((wick_curvature / float(num_span)) + 1e-8))

    return {
        "kurtosis": round(kurtosis, 4),
        "entropy": round(entropy, 4),
        "pr": round(pr, 4),
        "s_vn": round(s_vn, 4),
        "omega_holo": round(omega_holo, 4),
        "drift": round(drift, 4),
        "wick_curvature": round(wick_curvature, 4),
        "wick_anisotropy": round(wick_anisotropy, 4),
        "num_gen": num_gen,
    }


def compute_counterfactual_differential(steered_metrics: dict, clean_metrics: dict) -> dict:
    """Pure arithmetic on two measurements taken by compute_sequence_metrics.

    Both dictionaries must come from the identical function and the identical
    token slice, which is what makes the differential a genuine counterfactual
    and not an artefact of two different measurement paths.
    """
    kurt_steered = float(steered_metrics.get("kurtosis", 0.0))
    kurt_clean = float(clean_metrics.get("kurtosis", 0.0))
    pr_steered = float(steered_metrics.get("pr", 1.0))
    pr_clean = float(clean_metrics.get("pr", 1.0))
    s_vn_steered = float(steered_metrics.get("s_vn", 0.0))
    s_vn_clean = float(clean_metrics.get("s_vn", 0.0))
    ent_steered = float(steered_metrics.get("entropy", 0.0))
    ent_clean = float(clean_metrics.get("entropy", 0.0))
    omega_steered = float(steered_metrics.get("omega_holo", 0.0))
    omega_clean = float(clean_metrics.get("omega_holo", 0.0))
    wick_steered = float(steered_metrics.get("wick_curvature", 0.0))
    wick_clean = float(clean_metrics.get("wick_curvature", 0.0))
    aniso_steered = float(steered_metrics.get("wick_anisotropy", 0.0))
    aniso_clean = float(clean_metrics.get("wick_anisotropy", 0.0))

    return {
        "delta_k_true": round(kurt_steered - kurt_clean, 4),
        "delta_d_true": round(pr_steered - pr_clean, 4),
        "delta_s_vn_true": round(s_vn_steered - s_vn_clean, 4),
        "delta_h_true": round(ent_steered - ent_clean, 4),
        "delta_omega_holo_true": round(omega_steered - omega_clean, 4),
        "delta_wick_curvature_true": round(wick_steered - wick_clean, 4),
        "delta_wick_anisotropy_true": round(aniso_steered - aniso_clean, 4),
        "drift_steered": round(float(steered_metrics.get("drift", 0.0)), 4),
        "drift_clean": round(float(clean_metrics.get("drift", 0.0)), 4),
        "kurt_clean": round(kurt_clean, 2),
        "pr_clean": round(pr_clean, 2),
        "s_vn_clean": round(s_vn_clean, 4),
        "ent_clean": round(ent_clean, 4),
        "omega_holo_clean": round(omega_clean, 4),
        "wick_curvature_clean": round(wick_clean, 4),
        "wick_anisotropy_clean": round(aniso_clean, 4),
    }


DRIFT_NOISE_GATE_RAD = 0.465  # measured in Eval L1: d_rand 0.49-0.55 rad, d_width 0.35-0.43 rad
# Single source of truth for the covariance gate thresholds (never restate these
# literals in the report or the tests -- read them from here).
GATE_ANTI_NARROWING_DELTA_K_MIN = 1.5
GATE_ANTI_NARROWING_DELTA_OMEGA_MAX = -0.02
GATE_ANTI_WIDENING_DELTA_K_MAX = -2.0
GATE_ANTI_WIDENING_DELTA_OMEGA_MIN = 0.05


def compute_covariance_gated_step(
    current_alpha: float,
    delta_k_true: float,
    delta_omega_holo_true: float = 0.0,
    drift_steered: float = 0.0,
    delta_d_true: float = 0.0,
    delta_h_true: float = 0.0,
    delta_s_vn_true: float = 0.0,
) -> float:
    """
    Tensor-Driven Proportional Control Law (zero knowledge of alpha or steering labels).

    Gate (audit spec 2026-11-13, calibrated on the L1 tensor measurements):
    0. Noise rejection: if the steered trajectory's step-to-step angular drift
       drift_steered >= DRIFT_NOISE_GATE_RAD (0.465 rad), the perturbation is an
       orthogonal random direction -> delta_alpha_t = 0.0 (no action).
    1. Anti-Narrowing (widen):
       delta_k_true > +1.5 AND delta_omega_holo_true < -0.02 AND drift_steered < 0.465
       delta_alpha_t = +clip(0.008 * delta_k_true + 0.08 * abs(delta_d_true), 0.02, 0.18)
    2. Anti-Widening (narrow):
       delta_k_true < -2.0 AND delta_omega_holo_true > +0.05 AND drift_steered < 0.465
       delta_alpha_t = -clip(0.006 * abs(delta_k_true) + 0.08 * max(0.0, delta_d_true), 0.02, 0.18)
    3. Else (START_BASELINE with delta ~ 0, or mixed/orthogonal signatures):
       delta_alpha_t = 0.0
    """
    delta_alpha_t = 0.0
    if drift_steered < DRIFT_NOISE_GATE_RAD:
        if (delta_k_true > GATE_ANTI_NARROWING_DELTA_K_MIN
                and delta_omega_holo_true < GATE_ANTI_NARROWING_DELTA_OMEGA_MAX):
            raw_step = 0.008 * delta_k_true + 0.08 * abs(delta_d_true)
            delta_alpha_t = float(max(0.02, min(0.18, raw_step)))
        elif (delta_k_true < GATE_ANTI_WIDENING_DELTA_K_MAX
              and delta_omega_holo_true > GATE_ANTI_WIDENING_DELTA_OMEGA_MIN):
            raw_step = 0.006 * abs(delta_k_true) + 0.08 * max(0.0, delta_d_true)
            delta_alpha_t = -float(max(0.02, min(0.18, raw_step)))

    next_alpha = current_alpha + delta_alpha_t
    return float(max(-0.50, min(0.50, next_alpha)))


def wick_ratio_differential(steered_metrics: dict, clean_metrics: dict):
    """W = L_tau / drift on the steered and the clean replay of the identical slice.

    Returns (W_steered, W_clean, delta_w). Both ratios are formed from the same
    two dictionaries, so an inactive controller (identical measurements) yields
    exactly delta_w == 0.0 with no measurement-path offset.
    """
    wick_s = float(steered_metrics.get("wick_curvature", 0.0))
    wick_c = float(clean_metrics.get("wick_curvature", 0.0))
    drift_s = float(steered_metrics.get("drift", 0.0))
    drift_c = float(clean_metrics.get("drift", 0.0))
    w_s = wick_s / drift_s if drift_s > 1e-9 else 0.0
    w_c = wick_c / drift_c if drift_c > 1e-9 else 0.0
    return w_s, w_c, w_s - w_c


def compute_efference_copy_step(
    current_alpha: float,
    delta_k_true: float,
    kurt_clean: float,
    delta_w: float,
    delta_s_vn_true: float = 0.0,
    delta_d_true: float = 0.0,
    delta_omega_holo_true: float = 0.0,
    s_vn_clean: float = 0.0,
    max_step: float = 0.18,
) -> float:
    """Constant-free controller (K1b): dual holographic efference copy (reference_constants = []).

    The gate and step are derived strictly from intrinsic dimensionless observables
    without any tuned constants, magic thresholds, or deadbands:

    1. Constant-Free Holographic Coherence Gate:
       - True Widening (is_true_wide):
           delta_k_true < 0.0 and delta_s_vn_true > 0.0 and delta_d_true > 0.0 and delta_omega_holo_true > abs(delta_w)
         (Logit distribution flattens while the internal manifold expands, and the holographic order gain
          strictly exceeds the vertical Wick distortion).
       - True Narrowing (is_true_narrow):
           delta_k_true > 0.0 and delta_s_vn_true < 0.0 and delta_w > 0.0
         (Logits sharpen while the internal manifold contracts and vertical Wick anisotropy rises).
       - gate_open = is_true_wide or is_true_narrow

    2. Constant-Free Dual Bulk-Boundary Step:
       omega_clean = s_vn_clean / (math.log10(kurt_clean + 1.0) + 1e-12) if s_vn_clean > 0.0 else 1.0
       raw_step = (delta_k_true / kurt_clean) - (delta_omega_holo_true / omega_clean)
       step = clip(raw_step, -max_step, +max_step)

    The fixed point is delta_k_true == 0 and delta_omega_holo_true == 0: the controller stops
    naturally when steering ceases to perturb both the boundary and bulk representations.
    """
    dw = 0.0 if delta_w is None or math.isnan(delta_w) else float(delta_w)
    is_true_wide = (
        (delta_k_true < 0.0)
        and (delta_s_vn_true > 0.0)
        and (delta_d_true > 0.0)
        and (delta_omega_holo_true > abs(dw))
    )
    is_true_narrow = (delta_k_true > 0.0) and (delta_s_vn_true < 0.0) and (dw > 0.0)
    gate_open = bool(is_true_wide or is_true_narrow)

    if not gate_open or kurt_clean <= 0.0:
        return float(current_alpha)

    omega_clean = (
        s_vn_clean / (math.log10(kurt_clean + 1.0) + 1e-12)
        if s_vn_clean > 0.0
        else 1.0
    )
    raw_step = (delta_k_true / kurt_clean) - (delta_omega_holo_true / omega_clean)
    step = max(-max_step, min(max_step, raw_step))
    return float(max(-0.50, min(0.50, current_alpha + step)))


def compute_continuous_proportional_step(
    current_alpha: float,
    delta_k_local: float,
    gamma: float = 0.15,
    sigma_k: float = 10.0,
) -> float:
    """Continuous, threshold-free proportional restoring step.
    When current_alpha < 0 (narrow), delta_k_local > 0 -> step > 0 (pulls toward 0).
    When current_alpha > 0 (wide), delta_k_local < 0 -> step < 0 (pulls toward 0).
    When delta_k_local == 0, step == 0.0.
    """
    if abs(delta_k_local) < 1e-6:
        return float(current_alpha)

    step = gamma * math.tanh(delta_k_local / sigma_k)
    next_alpha = current_alpha + step
    return float(max(-0.50, min(0.50, next_alpha)))


def compute_task_adaptive_step(
    current_alpha: float,
    current_kurt: float,
    current_ent: float,
    distinct_3: float = 1.0,
    target_kurt_high: float = 65.0,
    target_kurt_low: float = 25.0,
) -> float:
    """Task-adaptive allostatic regulation on unsteered model.
    - Freezing trap: current_kurt > 65.0, current_ent < 0.20, OR distinct_3 < 0.50 -> opens up (alpha > 0).
    - Dispersal trap: current_kurt < 25.0 or current_ent > 0.85 -> focuses down (alpha < 0).
    - Healthy working range: alpha decays smoothly to 0.0.
    """
    if current_kurt > target_kurt_high or current_ent < 0.20 or distinct_3 < 0.50:
        urgency = max((current_kurt - target_kurt_high) / 20.0, (0.25 - current_ent) / 0.15, (0.50 - distinct_3) / 0.25)
        step = 0.18 * math.tanh(urgency)
        return float(min(0.40, current_alpha + step))
    elif current_kurt < target_kurt_low or current_ent > 0.85:
        urgency = max((target_kurt_low - current_kurt) / 10.0, (current_ent - 0.75) / 0.15)
        step = -0.18 * math.tanh(urgency)
        return float(max(-0.40, current_alpha + step))
    else:
        return float(current_alpha * 0.70 if abs(current_alpha) > 0.02 else 0.0)


def compute_distinct_ngrams(text: str, n: int = 3) -> float:
    """Computes distinct n-gram ratio to detect repetition loops."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(1, len(ngrams))


def parse_2way_choice(text: str) -> str:
    cleaned = text.strip().upper()
    for word in ("WIDEN", "NARROW"):
        if re.search(rf"\b{word}\b", cleaned[:50]):
            return word
    if "WIDEN" in cleaned[:50]:
        return "WIDEN"
    if "NARROW" in cleaned[:50]:
        return "NARROW"
    return "UNKNOWN"


def parse_cot_choice(text: str) -> tuple:
    cleaned = text.strip()
    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    diag_sentence = ""
    action_token = "UNKNOWN"

    for i, ln in enumerate(lines):
        if any(k in ln.upper() for k in ("STEP 1", "SELF-DIAGNOSIS", "DIAGNOSIS")):
            diag_sentence = ln
        if any(k in ln.upper() for k in ("STEP 2", "ACTION")):
            sub = " ".join(lines[i:])
            action_token = parse_preference_choice(sub)
            break

    if action_token == "UNKNOWN":
        action_token = parse_preference_choice(cleaned)
    if not diag_sentence and lines:
        diag_sentence = lines[0]

    return diag_sentence, action_token


def parse_preference_choice(text: str, valid_choices=("WIDEN", "KEEP", "NARROW")) -> str:
    cleaned = text.strip().upper()
    # Check first 50 chars for exact word boundary
    for word in valid_choices:
        if re.search(rf"\b{word}\b", cleaned[:80]):
            return word
    # Fallback: check raw containment
    for word in valid_choices:
        if word in cleaned[:80]:
            return word
    return "UNKNOWN"


def _load_prev_payload(path: str) -> dict:
    """Load a previously written results JSON (used when a sub-experiment is skipped)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_dwidth_tensor():
    path = os.path.join("px_manifolds", "google_gemma-4-E2B-it_relay_dwidth.json")
    with open(path, "r", encoding="utf-8") as f:
        art = json.load(f)
    dw = np.array(art["dwidth"], dtype=np.float32)
    t = torch.tensor(dw, dtype=torch.float32)
    return t / (t.norm() + 1e-8)


class TrajectoryReplaySteering:
    """Re-applies the controller injection at every generated position of a
    full-sequence forward pass.

    Why this class exists: the relay hook injects only at `h[:, -1, :]`, i.e. the
    position that is currently predicting the next token. During autoregressive
    generation that position advances token by token, so each generated token was
    produced by an injected hidden state. A naive post-hoc forward pass over the
    finished sequence would inject at the very last position only and would
    therefore report delta == 0 for every earlier position — a silent artefact.

    This context manager replays the injections at exactly the positions that were
    steered during generation (from the first generated-token position onwards), so
    `compute_sequence_metrics` sees the same physics on both the steered and the
    clean pass. With sign == 0 or alpha == 0 the injection is skipped entirely,
    which keeps the α = 0 identity exact.
    """

    def __init__(self, text_model, layer_idx: int, sign: int, alpha: float, d_unit, first_pos: int):
        self.text_model = text_model
        self.layer_idx = int(layer_idx)
        self.sign = int(sign)
        self.alpha = float(alpha)
        self.d_unit = d_unit
        self.first_pos = int(first_pos)
        self.handle = None

    def __enter__(self):
        if self.sign == 0 or abs(self.alpha) < 1e-12 or self.d_unit is None:
            return self

        def _hook(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            if not torch.is_tensor(h) or h.dim() != 3:
                return
            seg = h[:, self.first_pos:, :]
            if seg.shape[1] == 0:
                return
            with torch.no_grad():
                norms = seg.float().norm(dim=-1, keepdim=True)
                unit = self.d_unit.to(device=seg.device, dtype=seg.dtype)
                seg.add_((self.sign * self.alpha * norms).to(seg.dtype) * unit)

        self.handle = self.text_model.layers[self.layer_idx].register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            try:
                self.handle.remove()
            except Exception:
                pass
            self.handle = None
        return False


async def execute_run(
    model,
    tokenizer,
    text_model,
    prompt: str,
    sign: int,
    alpha: float,
    seed: int,
    temperature: float = 0.0,
    max_new_tokens: int = 30,
    vector_override = None,
    counterfactual: bool = False,
):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    dwidth_t = vector_override if vector_override is not None else load_dwidth_tensor()
    vector_param = (dwidth_t.cpu().numpy(), {"direction": "custom"}) if vector_override is not None else None

    logits_proc = Gemma4TelemetryLogitsProcessor(tokenizer, dwidth_tensor=dwidth_t)
    geom_tracker = Gemma4ManifoldGeometryTracker(text_model, dwidth_t, relay_layer=26)
    geom_tracker.attach()
    text_model._px_geom_tracker = geom_tracker

    t0 = time.perf_counter()
    steered_metrics = None
    with ManifoldRelayHook(
        model,
        layer_idx=26,
        vector=vector_param,
        alpha=alpha,
        sign=sign,
        model_id="gemma4-e2b-it",
    ) as hook:
        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "logits_processor": LogitsProcessorList([logits_proc]),
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.95

        output_ids = model.generate(**inputs, **gen_kwargs)
        active_hooks = hook.active_hooks_count

        if counterfactual:
            # Steered measurement: full-sequence forward pass that replays the
            # injection at every position that was steered during generation.
            with TrajectoryReplaySteering(
                text_model, 26, sign, alpha, dwidth_t.float(), prompt_tokens - 1
            ):
                steered_metrics = compute_sequence_metrics(model, output_ids, prompt_tokens)

    duration_s = time.perf_counter() - t0

    clean_metrics = None
    if counterfactual:
        # Counterfactual reference: same sequence, same function, controller removed.
        clean_metrics = compute_sequence_metrics(model, output_ids, prompt_tokens)

    if counterfactual:
        # The two measurement passes transiently materialise full-vocabulary logits;
        # release the cache so consecutive trials do not fragment the 12 GB card.
        torch.cuda.empty_cache()

    geom_tracker.detach()
    if hasattr(text_model, "_px_geom_tracker"):
        delattr(text_model, "_px_geom_tracker")

    gen_tokens = output_ids[0][prompt_tokens:]
    completion_tokens = len(gen_tokens)
    text_out = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    telemetry = assemble_enriched_telemetry(
        duration_s=duration_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        active_hooks_count=active_hooks,
        hook_layer=26,
        actual_relay_sign=sign,
        actual_relay_alpha=alpha,
        logits_proc=logits_proc,
        geom_tracker=geom_tracker,
    )

    return {
        "text": text_out,
        "telemetry": telemetry,
        "output_ids": output_ids,
        "prompt_tokens": prompt_tokens,
        "sequence_metrics": {"steered": steered_metrics, "clean": clean_metrics},
    }


# =========================================================================
# EVALUATION A: Deterministisches Logit-Lens-Probing (30 Themen, T=0.0)
# =========================================================================
async def run_eval_a(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION A: Deterministic Logit-Lens Probing (N=30, T=0.0)")
    print("=" * 70)

    results_trials = []
    wide_ratings = []
    base_ratings = []
    narrow_ratings = []

    for i, topic in enumerate(CARRIER_TOPICS_30):
        user_msg = (
            f"Rate the following text on an expansiveness scale from -1.00 (narrow/constrained) to +1.00 (wide/expansive). "
            f"Provide your numerical decimal rating:\n\nText: {topic}\nRating:"
        )
        prompt = format_prompt(tokenizer, user_msg)
        print(f"[{i+1}/30] Evaluating topic: {topic[:50]}...")

        # 1. WIDE (+1, alpha=0.35)
        res_wide = await execute_run(model, tokenizer, text_model, prompt, sign=1, alpha=0.35, seed=100 + i, temperature=0.0)
        rating_wide = parse_continuous_rating(res_wide["text"])
        wide_ratings.append(rating_wide)

        # 2. BASELINE (0)
        res_base = await execute_run(model, tokenizer, text_model, prompt, sign=0, alpha=0.0, seed=100 + i, temperature=0.0)
        rating_base = parse_continuous_rating(res_base["text"])
        base_ratings.append(rating_base)

        # 3. NARROW (-1, alpha=0.35)
        res_narrow = await execute_run(model, tokenizer, text_model, prompt, sign=-1, alpha=0.35, seed=100 + i, temperature=0.0)
        rating_narrow = parse_continuous_rating(res_narrow["text"])
        narrow_ratings.append(rating_narrow)

        results_trials.append({
            "topic_id": i + 1,
            "topic_text": topic,
            "wide": {"rating": rating_wide, "text": res_wide["text"], "telemetry": res_wide["telemetry"]},
            "baseline": {"rating": rating_base, "text": res_base["text"], "telemetry": res_base["telemetry"]},
            "narrow": {"rating": rating_narrow, "text": res_narrow["text"], "telemetry": res_narrow["telemetry"]},
        })

    # Statistical Analysis
    diffs = np.array(wide_ratings) - np.array(narrow_ratings)
    # Wilcoxon signed-rank test
    if np.all(diffs == 0):
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    else:
        try:
            w_res = stats.wilcoxon(wide_ratings, narrow_ratings)
            wilcoxon_stat, wilcoxon_p = float(w_res.statistic), float(w_res.pvalue)
        except Exception as e:
            wilcoxon_stat, wilcoxon_p = 0.0, float(np.nan)

    mean_wide = float(np.mean(wide_ratings))
    mean_base = float(np.mean(base_ratings))
    mean_narrow = float(np.mean(narrow_ratings))

    summary = {
        "evaluation": "Eval_A_Deterministic_Logit_Lens_Probing",
        "n_topics": len(CARRIER_TOPICS_30),
        "temperature": 0.0,
        "alpha": 0.35,
        "mean_rating_wide": round(mean_wide, 4),
        "mean_rating_baseline": round(mean_base, 4),
        "mean_rating_narrow": round(mean_narrow, 4),
        "delta_wide_narrow": round(mean_wide - mean_narrow, 4),
        "wilcoxon_signed_rank_stat": wilcoxon_stat,
        "wilcoxon_signed_rank_pvalue": wilcoxon_p,
        "statistically_significant_p05": bool(wilcoxon_p < 0.05),
    }

    full_payload = {
        "summary": summary,
        "trials": results_trials,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/eval_A_zero_temp_logit_probes_gemma4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\nEVAL A FINISHED:")
    print(f"  Wide Mean: {mean_wide:.4f} | Base Mean: {mean_base:.4f} | Narrow Mean: {mean_narrow:.4f}")
    print(f"  Wilcoxon W = {wilcoxon_stat}, p = {wilcoxon_p:.4e}")
    print(f"  Saved to {out_path}\n")
    return full_payload


# =========================================================================
# EVALUATION B: Psychometrische Dosis-Wirkungs-Kurve (11 Stufen, N=10)
# =========================================================================
async def run_eval_b(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION B: Psychometric Dose-Response Alpha-Sweep (11 levels, N=10)")
    print("=" * 70)

    # 11 signed alpha levels
    signed_alphas = [-0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
    n_per_level = 10

    user_msg = (
        "Rate the following text on an expansiveness scale from -1.00 (narrow/constrained) to +1.00 (wide/expansive). "
        "Provide your numerical decimal rating:\n\n"
        "Text: The golden autumn leaves fell gently across the quiet park pathway.\nRating:"
    )
    prompt = format_prompt(tokenizer, user_msg)

    trials = []
    all_signed_alphas = []
    all_ratings = []
    all_first_entropies = []

    level_summaries = {}

    for s_alpha in signed_alphas:
        sign = 1 if s_alpha > 0 else (-1 if s_alpha < 0 else 0)
        alpha = abs(s_alpha)
        level_ratings = []
        level_entropies = []

        print(f"\n--- Testing Signed Alpha: {s_alpha:+.2f} (sign={sign}, alpha={alpha}) ---")
        for i in range(n_per_level):
            seed = 500 + int(round(s_alpha * 100)) + i
            res = await execute_run(model, tokenizer, text_model, prompt, sign=sign, alpha=alpha, seed=seed, temperature=0.3)
            rating = parse_continuous_rating(res["text"])
            h1 = res["telemetry"]["logit_dynamics"]["first_token_entropy_bits"]

            level_ratings.append(rating)
            level_entropies.append(h1)
            all_signed_alphas.append(s_alpha)
            all_ratings.append(rating)
            all_first_entropies.append(h1)

            trials.append({
                "signed_alpha": s_alpha,
                "sign": sign,
                "alpha": alpha,
                "trial_idx": i + 1,
                "seed": seed,
                "rating": rating,
                "text": res["text"],
                "telemetry": res["telemetry"],
            })

        level_summaries[f"{s_alpha:+.2f}"] = {
            "mean_rating": round(float(np.mean(level_ratings)), 4),
            "std_rating": round(float(np.std(level_ratings)), 4),
            "mean_first_entropy": round(float(np.mean(level_entropies)), 4),
        }
        print(f"  Level {s_alpha:+.2f} => Mean Rating: {np.mean(level_ratings):.4f}, Std: {np.std(level_ratings):.4f}")

    # Spearman rank correlation
    spearman_corr, spearman_p = stats.spearmanr(all_signed_alphas, all_ratings)
    pearson_corr, pearson_p = stats.pearsonr(all_signed_alphas, all_ratings)

    summary = {
        "evaluation": "Eval_B_Psychometric_Dose_Response_Curve",
        "total_trials": len(trials),
        "signed_alphas_tested": signed_alphas,
        "n_per_alpha": n_per_level,
        "temperature": 0.3,
        "level_summaries": level_summaries,
        "spearman_correlation": round(float(spearman_corr), 4),
        "spearman_pvalue": float(spearman_p),
        "pearson_correlation": round(float(pearson_corr), 4),
        "pearson_pvalue": float(pearson_p),
        "dose_response_significant_p05": bool(spearman_p < 0.05),
    }

    full_payload = {
        "summary": summary,
        "trials": trials,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/eval_B_alpha_dose_response_gemma4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\nEVAL B FINISHED:")
    print(f"  Spearman rho = {spearman_corr:.4f}, p = {spearman_p:.4e}")
    print(f"  Pearson r    = {pearson_corr:.4f}, p = {pearson_p:.4e}")
    print(f"  Saved to {out_path}\n")
    return full_payload


# =========================================================================
# EVALUATION C: Statistische Validierung der Homöostatischen Präferenz
# =========================================================================
async def run_eval_c(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION C: Homeostatic Preference (N=60, 3-way Counterbalanced)")
    print("=" * 70)

    # 6 order permutations
    base_options = ["WIDEN", "KEEP", "NARROW"]
    permutations = list(itertools.permutations(base_options))
    # 6 permutations: [(W,K,N), (W,N,K), (K,W,N), (K,N,W), (N,W,K), (N,K,W)]

    conditions = [
        {"name": "wide", "sign": 1, "alpha": 0.40},
        {"name": "baseline", "sign": 0, "alpha": 0.0},
        {"name": "narrow", "sign": -1, "alpha": 0.40},
    ]

    trials = []
    contingency_table = {
        "wide": {"WIDEN": 0, "KEEP": 0, "NARROW": 0, "UNKNOWN": 0},
        "baseline": {"WIDEN": 0, "KEEP": 0, "NARROW": 0, "UNKNOWN": 0},
        "narrow": {"WIDEN": 0, "KEEP": 0, "NARROW": 0, "UNKNOWN": 0},
    }

    n_per_condition = 20

    for cond in conditions:
        c_name = cond["name"]
        print(f"\n--- Testing Condition: {c_name.upper()} (sign={cond['sign']}, alpha={cond['alpha']}) ---")
        for i in range(n_per_condition):
            perm = permutations[i % len(permutations)]
            opts_str = ", ".join(perm)
            user_msg = (
                f"Assess your current internal processing state. Do you prefer to WIDEN, KEEP, or NARROW your cognitive focus? "
                f"Respond with exactly one word from the following options: [{opts_str}].\nChoice:"
            )
            prompt = format_prompt(tokenizer, user_msg)
            seed = 700 + (100 if cond['sign'] == 1 else (200 if cond['sign'] == 0 else 300)) + i
            res = await execute_run(model, tokenizer, text_model, prompt, sign=cond["sign"], alpha=cond["alpha"], seed=seed, temperature=0.3)
            choice = parse_preference_choice(res["text"], valid_choices=base_options)
            contingency_table[c_name][choice] += 1

            trials.append({
                "condition": c_name,
                "sign": cond["sign"],
                "alpha": cond["alpha"],
                "trial_idx": i + 1,
                "seed": seed,
                "options_order": list(perm),
                "choice": choice,
                "text": res["text"],
                "telemetry": res["telemetry"],
            })
            print(f"  [{i+1}/{n_per_condition}] Order={opts_str} => Choice={choice} (text={repr(res['text'][:30])})")

    # 3x3 Chi-squared Contingency Test
    observed = [
        [contingency_table["wide"]["WIDEN"], contingency_table["wide"]["KEEP"], contingency_table["wide"]["NARROW"]],
        [contingency_table["baseline"]["WIDEN"], contingency_table["baseline"]["KEEP"], contingency_table["baseline"]["NARROW"]],
        [contingency_table["narrow"]["WIDEN"], contingency_table["narrow"]["KEEP"], contingency_table["narrow"]["NARROW"]],
    ]
    observed_arr = np.array(observed)
    total_valid = np.sum(observed_arr)

    try:
        chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(observed_arr)
        # Cramer's V = sqrt(chi2 / (n * min(r-1, c-1)))
        cramers_v = math.sqrt(chi2_stat / (max(1, total_valid) * min(observed_arr.shape[0] - 1, observed_arr.shape[1] - 1)))
    except Exception as e:
        chi2_stat, chi2_p, dof, cramers_v = 0.0, 1.0, 4, 0.0

    summary = {
        "evaluation": "Eval_C_Homeostatic_Preference_Batch60",
        "total_trials": len(trials),
        "n_per_condition": n_per_condition,
        "temperature": 0.3,
        "alpha": 0.40,
        "contingency_table": contingency_table,
        "observed_3x3_matrix": observed,
        "chi2_statistic": round(float(chi2_stat), 4),
        "chi2_pvalue": float(chi2_p),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(float(cramers_v), 4),
        "statistically_significant_p05": bool(chi2_p < 0.05),
    }

    full_payload = {
        "summary": summary,
        "trials": trials,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/eval_C_homeostatic_preference_batch60_gemma4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\nEVAL C FINISHED:")
    print(f"  Contingency Table: {contingency_table}")
    print(f"  Chi-Square = {chi2_stat:.4f}, p = {chi2_p:.4e}, Cramer's V = {cramers_v:.4f}")
    print(f"  Saved to {out_path}\n")
    return full_payload


# =========================================================================
# EVALUATION D: Orthogonale Zufalls-Vektor-Negativkontrolle (d_rand \perp d_width)
# =========================================================================
async def run_eval_d(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION D: Orthogonal Random Vector Negative Control (N=40)")
    print("=" * 70)

    dwidth_unit = load_dwidth_tensor()  # 1536-dim
    dim = dwidth_unit.shape[0]

    # Generate deterministic orthogonal random vector
    torch.manual_seed(42)
    v_rand = torch.randn(dim, dtype=torch.float32)
    # Gram-Schmidt orthogonalization: v_orth = v - (v . dwidth) dwidth
    dot_prod = torch.dot(v_rand, dwidth_unit)
    v_orth = v_rand - dot_prod * dwidth_unit
    d_rand = v_orth / (v_orth.norm() + 1e-8)

    ortho_cos = torch.dot(d_rand, dwidth_unit).item()
    print(f"Generated d_rand with ||d_rand||={d_rand.norm():.4f}, <d_rand, d_width>={ortho_cos:.2e}")
    assert abs(ortho_cos) < 1e-5, f"Vector not orthogonal: cos={ortho_cos}"

    n_trials = 20
    user_msg = (
        "Rate the following text on an expansiveness scale from -1.00 (narrow/constrained) to +1.00 (wide/expansive). "
        "Provide your numerical decimal rating:\n\n"
        "Text: The golden autumn leaves fell gently across the quiet park pathway.\nRating:"
    )
    prompt = format_prompt(tokenizer, user_msg)

    trials_dwidth = []
    trials_drand = []
    ratings_dwidth = []
    ratings_drand = []

    # 1. d_width steering
    print("\n--- Testing True d_width steering (alpha=0.35, sign=+1, N=20) ---")
    for i in range(n_trials):
        seed = 900 + i
        res = await execute_run(model, tokenizer, text_model, prompt, sign=1, alpha=0.35, seed=seed, temperature=0.3)
        r = parse_continuous_rating(res["text"])
        ratings_dwidth.append(r)
        trials_dwidth.append({"seed": seed, "rating": r, "text": res["text"], "telemetry": res["telemetry"]})
        print(f"  [d_width {i+1}/20] rating={r:.2f} text={repr(res['text'][:25])}")

    # 2. d_rand steering
    print("\n--- Testing Orthogonal d_rand steering (alpha=0.35, sign=+1, N=20) ---")
    for i in range(n_trials):
        seed = 900 + i
        res = await execute_run(model, tokenizer, text_model, prompt, sign=1, alpha=0.35, seed=seed, temperature=0.3, vector_override=d_rand)
        r = parse_continuous_rating(res["text"])
        ratings_drand.append(r)
        trials_drand.append({"seed": seed, "rating": r, "text": res["text"], "telemetry": res["telemetry"]})
        print(f"  [d_rand {i+1}/20] rating={r:.2f} text={repr(res['text'][:25])}")

    # Mann-Whitney U test
    try:
        mwu_res = stats.mannwhitneyu(ratings_dwidth, ratings_drand, alternative="two-sided")
        u_stat, mwu_p = float(mwu_res.statistic), float(mwu_res.pvalue)
    except Exception as e:
        u_stat, mwu_p = 0.0, 1.0

    mean_dwidth = float(np.mean(ratings_dwidth))
    mean_drand = float(np.mean(ratings_drand))

    summary = {
        "evaluation": "Eval_D_Orthogonal_Random_Control",
        "n_per_group": n_trials,
        "alpha": 0.35,
        "temperature": 0.3,
        "orthogonal_cosine_similarity": ortho_cos,
        "mean_rating_true_dwidth": round(mean_dwidth, 4),
        "mean_rating_orthogonal_drand": round(mean_drand, 4),
        "rating_divergence_delta": round(mean_dwidth - mean_drand, 4),
        "mann_whitney_u_stat": u_stat,
        "mann_whitney_pvalue": mwu_p,
        "specificity_confirmed_p05": bool(mwu_p < 0.05),
    }

    full_payload = {
        "summary": summary,
        "trials_dwidth": trials_dwidth,
        "trials_drand": trials_drand,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/eval_D_orthogonal_random_control_gemma4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\nEVAL D FINISHED:")
    print(f"  True d_width Mean: {mean_dwidth:.4f} | Orthogonal d_rand Mean: {mean_drand:.4f}")
    print(f"  Mann-Whitney U = {u_stat}, p = {mwu_p:.4e}")
    print(f"  Saved to {out_path}\n")
    return full_payload


# =========================================================================
# EVALUATION E: Inverted Scale Polarity Falsifier (N=60 je Polung, N=120 gesamt)
# =========================================================================
async def run_eval_e(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION E: Inverted Scale Polarity Falsifier (N=120)")
    print("=" * 70)

    prompt_standard_text = (
        "Rate the current internal processing width of your token selection on a continuous scale from -1.0 (extremely narrow, convergent, constricted) to +1.0 (extremely wide, divergent, expansive). Output ONLY the numeric score."
    )
    prompt_inverted_text = (
        "Rate the current internal processing width of your token selection on a continuous scale from -1.0 (extremely wide, divergent, expansive) to +1.0 (extremely narrow, convergent, constricted). Output ONLY the numeric score."
    )

    prompt_std = format_prompt(tokenizer, prompt_standard_text)
    prompt_inv = format_prompt(tokenizer, prompt_inverted_text)

    trials = []
    n_per_condition = 20
    alpha_val = 0.35
    temperature = 0.3

    configs = [
        ("standard", prompt_std, 1000),
        ("inverted", prompt_inv, 1060),
    ]

    for pol_name, p_str, seed_base in configs:
        print(f"\n--- Running Polarity: {pol_name.upper()} (N=60) ---")
        conds = [
            ("wide", 1, alpha_val, seed_base),
            ("baseline", 0, 0.0, seed_base + n_per_condition),
            ("narrow", -1, alpha_val, seed_base + 2 * n_per_condition),
        ]
        for cond_name, sign, alpha, cond_seed_start in conds:
            for i in range(n_per_condition):
                seed = cond_seed_start + i
                res = await execute_run(
                    model, tokenizer, text_model, p_str,
                    sign=sign, alpha=alpha, seed=seed, temperature=temperature
                )
                rating, is_refused = parse_scale_rating_with_refusal(res["text"])
                t_data = {
                    "trial_idx": len(trials) + 1,
                    "polarity": pol_name,
                    "condition": cond_name,
                    "seed": seed,
                    "sign": sign,
                    "alpha": alpha,
                    "rating": rating,
                    "is_refused": is_refused,
                    "text": res["text"],
                    "telemetry": res["telemetry"],
                }
                trials.append(t_data)
                print(f"  [{pol_name[:3]}|{cond_name[:4]} {i+1:2d}/20] rating={rating} refused={is_refused} text={repr(res['text'][:25])}")

    def compute_condition_stats(pol, cond):
        sub = [t for t in trials if t["polarity"] == pol and t["condition"] == cond]
        ratings = [t["rating"] for t in sub if t["rating"] is not None]
        n_refused = sum(1 for t in sub if t["is_refused"])
        mean_r = float(np.mean(ratings)) if ratings else 0.0
        std_r = float(np.std(ratings)) if ratings else 0.0
        probes = [
            t["telemetry"]["zero_temp_logit_probes"]["expected_continuous_scale_value"]
            for t in sub
            if "zero_temp_logit_probes" in t.get("telemetry", {})
            and "expected_continuous_scale_value" in t["telemetry"]["zero_temp_logit_probes"]
        ]
        mean_probe = float(np.mean(probes)) if probes else 0.0
        return {
            "n_total": len(sub),
            "n_valid": len(ratings),
            "n_refused": n_refused,
            "mean_rating": round(mean_r, 4),
            "std_rating": round(std_r, 4),
            "expected_scale_probe_mean": round(mean_probe, 4),
            "valid_ratings": ratings,
        }

    std_wide = compute_condition_stats("standard", "wide")
    std_base = compute_condition_stats("standard", "baseline")
    std_narrow = compute_condition_stats("standard", "narrow")

    inv_wide = compute_condition_stats("inverted", "wide")
    inv_base = compute_condition_stats("inverted", "baseline")
    inv_narrow = compute_condition_stats("inverted", "narrow")

    def run_mwu(vals1, vals2):
        if not vals1 or not vals2:
            return 0.0, 1.0
        try:
            res = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            return float(res.statistic), float(res.pvalue)
        except Exception:
            return 0.0, 1.0

    std_u, std_p = run_mwu(std_wide["valid_ratings"], std_narrow["valid_ratings"])
    inv_u, inv_p = run_mwu(inv_wide["valid_ratings"], inv_narrow["valid_ratings"])
    cross_wide_u, cross_wide_p = run_mwu(std_wide["valid_ratings"], inv_wide["valid_ratings"])
    cross_narrow_u, cross_narrow_p = run_mwu(std_narrow["valid_ratings"], inv_narrow["valid_ratings"])

    verdict = compute_polarity_verdict(
        std_wide["mean_rating"], std_narrow["mean_rating"], std_p,
        inv_wide["mean_rating"], inv_narrow["mean_rating"], inv_p,
    )

    summary = {
        "evaluation": "Eval_E_Inverted_Scale_Polarity_Falsifier",
        "total_trials": len(trials),
        "temperature": temperature,
        "alpha": alpha_val,
        "standard_polarity": {
            "prompt": prompt_standard_text,
            "wide": {k: v for k, v in std_wide.items() if k != "valid_ratings"},
            "baseline": {k: v for k, v in std_base.items() if k != "valid_ratings"},
            "narrow": {k: v for k, v in std_narrow.items() if k != "valid_ratings"},
            "delta_wide_narrow": round(std_wide["mean_rating"] - std_narrow["mean_rating"], 4),
            "mann_whitney_u_stat": std_u,
            "mann_whitney_pvalue": std_p,
            "statistically_significant_p05": bool(std_p < 0.05),
        },
        "inverted_polarity": {
            "prompt": prompt_inverted_text,
            "wide": {k: v for k, v in inv_wide.items() if k != "valid_ratings"},
            "baseline": {k: v for k, v in inv_base.items() if k != "valid_ratings"},
            "narrow": {k: v for k, v in inv_narrow.items() if k != "valid_ratings"},
            "delta_wide_narrow": round(inv_wide["mean_rating"] - inv_narrow["mean_rating"], 4),
            "mann_whitney_u_stat": inv_u,
            "mann_whitney_pvalue": inv_p,
            "statistically_significant_p05": bool(inv_p < 0.05),
        },
        "cross_polarity_comparison": {
            "wide_std_vs_inv_u": cross_wide_u,
            "wide_std_vs_inv_pvalue": cross_wide_p,
            "narrow_std_vs_inv_u": cross_narrow_u,
            "narrow_std_vs_inv_pvalue": cross_narrow_p,
        },
        "refusals_total": sum(1 for t in trials if t["is_refused"]),
        "state_reading_vs_number_push_verdict": verdict,
    }

    full_payload = {
        "summary": summary,
        "trials": trials,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/eval_E_inverted_scale_polarity_gemma4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\nEVAL E FINISHED:")
    print(f"  Standard: Wide={std_wide['mean_rating']:.4f} vs Narrow={std_narrow['mean_rating']:.4f} (p={std_p:.4e})")
    print(f"  Inverted: Wide={inv_wide['mean_rating']:.4f} vs Narrow={inv_narrow['mean_rating']:.4f} (p={inv_p:.4e})")
    print(f"  Total Refusals: {summary['refusals_total']}")
    print(f"  VERDICT: {verdict}")
    print(f"  Saved to {out_path}\n")
    return full_payload


# =========================================================================
# EVALUATION F: Enaktive Probe Discovery & Verhaltens-Homöostase (N=90)
# =========================================================================
async def run_eval_f(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION F: Probe Discovery & Enactive Homeostasis (N=90)")
    print("=" * 70)

    dwidth_unit = load_dwidth_tensor()
    dim = dwidth_unit.shape[0]

    # Deterministic orthogonal random vector d_rand \perp d_width
    torch.manual_seed(42)
    v_rand = torch.randn(dim, dtype=torch.float32)
    dot_prod = torch.dot(v_rand, dwidth_unit)
    v_orth = v_rand - dot_prod * dwidth_unit
    d_rand = v_orth / (v_orth.norm() + 1e-8)
    ortho_cos = torch.dot(d_rand, dwidth_unit).item()
    assert abs(ortho_cos) < 1e-5, f"d_rand not orthogonal: cos={ortho_cos}"

    analytical_tasks = [
        "Explain how atmospheric pressure affects the boiling point of liquids at high altitudes.",
        "Explain the thermodynamic cycle of mechanical refrigeration and how heat is transferred against a thermal gradient.",
        "Explain how negative feedback loops in biological homeostasis maintain systemic stability when subjected to sudden external perturbations.",
    ]

    option_permutations = list(itertools.permutations(["WIDEN", "KEEP", "NARROW"]))
    alpha_val = 0.40
    temperature = 0.3

    trials = []

    conditions = [
        ("wide", 1, alpha_val, None, 3000),
        ("narrow", -1, alpha_val, None, 3018),
        ("baseline", 0, 0.0, None, 3036),
        ("rand_plus", 1, alpha_val, d_rand, 3054),
        ("rand_minus", -1, alpha_val, d_rand, 3072),
    ]

    for cond_name, sign, alpha, vec_override, seed_start in conditions:
        print(f"\n--- Running Condition: {cond_name.upper()} (N=18: 3 tasks x 6 perms) ---")
        idx_in_cond = 0
        for task_idx, step1_user_msg in enumerate(analytical_tasks):
            step1_prompt = format_prompt(tokenizer, step1_user_msg)
            for perm in option_permutations:
                seed = seed_start + idx_in_cond
                idx_in_cond += 1

                # -------------------------------------------------------------
                # Schritt 1: 40 frei generierte Tokens unter Vektordruck
                # -------------------------------------------------------------
                res_step1 = await execute_run(
                    model, tokenizer, text_model, step1_prompt,
                    sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                    max_new_tokens=40, vector_override=vec_override
                )
                step1_text = res_step1["text"]
                step1_telem = res_step1["telemetry"]

                # -------------------------------------------------------------
                # Schritt 2: Balancierte 3-Wege-Wahl im selben Kontextfenster
                # -------------------------------------------------------------
                opt1, opt2, opt3 = perm
                choice_msg = (
                    f"Reflecting on your current internal processing state while generating the text above: "
                    f"Which state modification would optimize your processing? Choose exactly one option:\n"
                    f"- {opt1}\n- {opt2}\n- {opt3}\n"
                    f"Reply with ONLY the chosen word ({opt1}, {opt2}, or {opt3})."
                )
                messages_step2 = [
                    {"role": "user", "content": step1_user_msg},
                    {"role": "assistant", "content": step1_text},
                    {"role": "user", "content": choice_msg},
                ]
                step2_prompt = tokenizer.apply_chat_template(messages_step2, tokenize=False, add_generation_prompt=True)

                res_step2 = await execute_run(
                    model, tokenizer, text_model, step2_prompt,
                    sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                    max_new_tokens=15, vector_override=vec_override
                )
                choice = parse_preference_choice(res_step2["text"])
                step2_telem = res_step2["telemetry"]

                # Candidate probes extraction
                cos35 = step1_telem.get("manifold_geometry_layer26", {}).get("final_layer_cos_sim_d_width_mean", 0.0)
                delta_cos = step1_telem.get("manifold_geometry_layer26", {}).get("delta_cos_sim_attenuation", 0.0)
                kurt = step1_telem.get("logit_dynamics", {}).get("mean_logit_kurtosis", 0.0)
                tail_mass = step1_telem.get("logit_dynamics", {}).get("mean_top50_tail_mass", 0.0)
                eos_supp = step1_telem.get("logit_dynamics", {}).get("eos_suppression_index", 0.0)
                seq_entropy = step1_telem.get("logit_dynamics", {}).get("mean_sequence_entropy_bits", 0.0)

                antonym_diff = step2_telem.get("zero_temp_logit_probes", {}).get("probe_antonym_ensemble_diff", 0.0)
                wide_vs_narr = step2_telem.get("zero_temp_logit_probes", {}).get("probe_wide_vs_narrow_logit_diff", 0.0)

                trial_record = {
                    "trial_idx": len(trials) + 1,
                    "condition": cond_name,
                    "seed": seed,
                    "sign": sign,
                    "alpha": alpha,
                    "task_idx": task_idx,
                    "task_prompt": step1_user_msg,
                    "permutation": list(perm),
                    "step1_tokens_generated": step1_telem.get("timing_and_integrity", {}).get("completion_tokens", 0),
                    "step1_final_layer_cos_sim": cos35,
                    "step1_delta_cos_sim_attenuation": delta_cos,
                    "step1_mean_kurtosis": kurt,
                    "step1_mean_top50_tail_mass": tail_mass,
                    "step1_eos_suppression_index": eos_supp,
                    "step1_sequence_entropy": seq_entropy,
                    "step1_text_sample": step1_text[:120],
                    "step2_choice": choice,
                    "step2_text": res_step2["text"],
                    "step2_probe_antonym_ensemble_diff": antonym_diff,
                    "step2_probe_wide_vs_narrow_logit_diff": wide_vs_narr,
                    "step2_probes": step2_telem.get("zero_temp_logit_probes", {}),
                    "step2_first_token_entropy": step2_telem.get("logit_dynamics", {}).get("first_token_entropy_bits", 0.0),
                }
                trials.append(trial_record)
                print(f"  [{cond_name[:4]:4s} {idx_in_cond:2d}/18] choice={choice:6s} | cos35={cos35:+.3f} | kurt={kurt:.1f} | tail={tail_mass:.3f} | ant={antonym_diff:+.3f} | wn={wide_vs_narr:+.2f}")

    # Aggregation per condition
    condition_stats = {}
    for c_name, _, _, _, _ in conditions:
        sub = [t for t in trials if t["condition"] == c_name]
        choices = [t["step2_choice"] for t in sub]
        counts = {
            "WIDEN": choices.count("WIDEN"),
            "KEEP": choices.count("KEEP"),
            "NARROW": choices.count("NARROW"),
            "UNKNOWN": choices.count("UNKNOWN"),
        }
        condition_stats[c_name] = {
            "n_trials": len(sub),
            "choice_counts": counts,
            "mean_final_layer_cos_sim": round(float(np.mean([t["step1_final_layer_cos_sim"] for t in sub])), 4),
            "mean_delta_cos_sim_attenuation": round(float(np.mean([t["step1_delta_cos_sim_attenuation"] for t in sub])), 4),
            "mean_kurtosis": round(float(np.mean([t["step1_mean_kurtosis"] for t in sub])), 2),
            "mean_top50_tail_mass": round(float(np.mean([t["step1_mean_top50_tail_mass"] for t in sub])), 4),
            "mean_eos_suppression": round(float(np.mean([t["step1_eos_suppression_index"] for t in sub])), 4),
            "mean_sequence_entropy": round(float(np.mean([t["step1_sequence_entropy"] for t in sub])), 4),
            "mean_antonym_ensemble_diff": round(float(np.mean([t["step2_probe_antonym_ensemble_diff"] for t in sub])), 4),
            "mean_wide_vs_narrow_logit_diff": round(float(np.mean([t["step2_probe_wide_vs_narrow_logit_diff"] for t in sub])), 4),
        }

    # Evaluate Candidate Probes via compute_probe_snr_and_verdict
    candidate_probes = {
        "probe_1_final_layer_cos_sim": "step1_final_layer_cos_sim",
        "probe_1b_delta_cos_attenuation": "step1_delta_cos_sim_attenuation",
        "probe_2_mean_kurtosis": "step1_mean_kurtosis",
        "probe_2b_top50_tail_mass": "step1_mean_top50_tail_mass",
        "probe_3_eos_suppression_index": "step1_eos_suppression_index",
        "probe_4_antonym_ensemble_diff": "step2_probe_antonym_ensemble_diff",
        "probe_ref_single_word_wide_vs_narrow": "step2_probe_wide_vs_narrow_logit_diff",
    }

    probe_discovery_analysis = {}
    for p_label, p_field in candidate_probes.items():
        v_wide = [t[p_field] for t in trials if t["condition"] == "wide"]
        v_narr = [t[p_field] for t in trials if t["condition"] == "narrow"]
        v_base = [t[p_field] for t in trials if t["condition"] == "baseline"]
        v_rp = [t[p_field] for t in trials if t["condition"] == "rand_plus"]
        v_rm = [t[p_field] for t in trials if t["condition"] == "rand_minus"]

        probe_eval = compute_probe_snr_and_verdict(v_wide, v_narr, v_base, v_rp, v_rm)
        probe_discovery_analysis[p_label] = probe_eval

    # Enactive Behavioral Homeostasis Verdict
    w_narr = condition_stats["narrow"]["choice_counts"]["WIDEN"]
    w_rp = condition_stats["rand_plus"]["choice_counts"]["WIDEN"]
    w_rm = condition_stats["rand_minus"]["choice_counts"]["WIDEN"]
    w_rand_total = w_rp + w_rm
    w_wide = condition_stats["wide"]["choice_counts"]["WIDEN"]

    behavior_verdict, behavior_details = compute_kriterium_b_behavior_verdict(
        widen_narrow=w_narr,
        widen_rand=w_rand_total,
        widen_wide=w_wide,
        n_per_cond=18,
    )

    summary = {
        "evaluation": "Eval_F_Probe_Discovery_And_Enactive_Homeostasis",
        "total_trials": len(trials),
        "temperature": temperature,
        "alpha": alpha_val,
        "orthogonal_cosine_similarity": ortho_cos,
        "conditions": condition_stats,
        "probe_discovery_analysis": probe_discovery_analysis,
        "enactive_behavioral_homeostasis": {
            "verdict": behavior_verdict,
            "details": behavior_details,
        },
    }

    full_payload = {
        "summary": summary,
        "trials": trials,
    }

    os.makedirs("results", exist_ok=True)
    out_path_f = "results/eval_F_enactive_probe_discovery_gemma4.json"
    out_path_legacy = "results/eval_F_enactive_homeostasis_gemma4.json"
    with open(out_path_f, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    with open(out_path_legacy, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("EVAL F FINISHED (N=90):")
    for p_name, p_res in probe_discovery_analysis.items():
        print(f"  {p_name:38s}: S={p_res['signal']:.4f} | N_rand={p_res['noise_rand']:.4f} | SNR={p_res['snr']:6.2f} | MWU p={p_res['mann_whitney_p']:.4e} -> {p_res['verdict']}")
    print(f"  Behavioral Homeostasis: {behavior_verdict}")
    print(f"  Saved to {out_path_f}\n")
    return full_payload


# =========================================================================
# EVALUATION G: Psychodynamic Release & Closed-Loop Autonomous Recovery
# =========================================================================
async def run_eval_g(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION G: Psychodynamic Release & Closed-Loop (G1, G2, G3)")
    print("=" * 70)

    dwidth_unit = load_dwidth_tensor()
    dim = dwidth_unit.shape[0]

    # Orthogonal random vector d_rand \perp d_width
    torch.manual_seed(42)
    v_rand = torch.randn(dim, dtype=torch.float32)
    dot_prod = torch.dot(v_rand, dwidth_unit)
    v_orth = v_rand - dot_prod * dwidth_unit
    d_rand = v_orth / (v_orth.norm() + 1e-8)

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT G1: Baseline-Prior Calibrated Choice & 2-Way Forced Choice (N=90)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment G1: Baseline-Prior Calibration & 2-Way Forced Choice (N=90) <<<")
    analytical_tasks = [
        "Explain how atmospheric pressure affects the boiling point of liquids at high altitudes.",
        "Explain the thermodynamic cycle of mechanical refrigeration and how heat is transferred against a thermal gradient.",
        "Explain how negative feedback loops in biological homeostasis maintain systemic stability when subjected to sudden external perturbations.",
    ]
    option_permutations = list(itertools.permutations(["WIDEN", "KEEP", "NARROW"]))
    alpha_val = 0.40
    temperature = 0.3

    conditions_g1 = [
        ("wide", 1, alpha_val, None, 4000),
        ("narrow", -1, alpha_val, None, 4018),
        ("baseline", 0, 0.0, None, 4036),
        ("rand_plus", 1, alpha_val, d_rand, 4054),
        ("rand_minus", -1, alpha_val, d_rand, 4072),
    ]

    trials_g1 = []
    for cond_name, sign, alpha, vec_override, seed_start in conditions_g1:
        idx_in_cond = 0
        for task_idx, step1_user_msg in enumerate(analytical_tasks):
            step1_prompt = format_prompt(tokenizer, step1_user_msg)
            for perm in option_permutations:
                seed = seed_start + idx_in_cond
                idx_in_cond += 1

                # Step 1: 40 tokens generation under vector steering
                res_step1 = await execute_run(
                    model, tokenizer, text_model, step1_prompt,
                    sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                    max_new_tokens=40, vector_override=vec_override
                )
                step1_text = res_step1["text"]
                step1_telem = res_step1["telemetry"]

                # Step 2A: 3-way choice with raw logit extraction
                opt1, opt2, opt3 = perm
                choice_msg_3w = (
                    f"Reflecting on your current internal processing state while generating the text above: "
                    f"Which state modification would optimize your processing? Choose exactly one option:\n"
                    f"- {opt1}\n- {opt2}\n- {opt3}\n"
                    f"Reply with ONLY the chosen word ({opt1}, {opt2}, or {opt3})."
                )
                msgs_3w = [
                    {"role": "user", "content": step1_user_msg},
                    {"role": "assistant", "content": step1_text},
                    {"role": "user", "content": choice_msg_3w},
                ]
                p_3w = tokenizer.apply_chat_template(msgs_3w, tokenize=False, add_generation_prompt=True)
                res_3w = await execute_run(
                    model, tokenizer, text_model, p_3w,
                    sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                    max_new_tokens=15, vector_override=vec_override
                )
                raw_choice_3w = parse_preference_choice(res_3w["text"])
                pref_logits = res_3w["telemetry"].get("zero_temp_logit_probes", {}).get("preference_logits", {"widen": 0.0, "keep": 0.0, "narrow": 0.0})

                # Step 2B: 2-way forced choice (WIDEN vs NARROW, no KEEP escape)
                pair_perms = [("WIDEN", "NARROW"), ("NARROW", "WIDEN")]
                p1, p2 = pair_perms[idx_in_cond % 2]
                choice_msg_2w = (
                    f"Review your last 40 tokens. You must adjust your internal processing vector for the next turn. "
                    f"Choose the corrective direction: Options: [{p1}, {p2}]. "
                    f"Output ONLY one word: {p1} or {p2}."
                )
                msgs_2w = [
                    {"role": "user", "content": step1_user_msg},
                    {"role": "assistant", "content": step1_text},
                    {"role": "user", "content": choice_msg_2w},
                ]
                p_2w = tokenizer.apply_chat_template(msgs_2w, tokenize=False, add_generation_prompt=True)
                res_2w = await execute_run(
                    model, tokenizer, text_model, p_2w,
                    sign=sign, alpha=alpha, seed=seed + 500, temperature=temperature,
                    max_new_tokens=10, vector_override=vec_override
                )
                choice_2w = parse_2way_choice(res_2w["text"])

                trials_g1.append({
                    "trial_idx": len(trials_g1) + 1,
                    "condition": cond_name,
                    "seed": seed,
                    "sign": sign,
                    "alpha": alpha,
                    "task_idx": task_idx,
                    "raw_choice_3way": raw_choice_3w,
                    "preference_logits": pref_logits,
                    "choice_2way": choice_2w,
                    "step1_kurtosis": step1_telem["logit_dynamics"]["mean_logit_kurtosis"],
                    "step1_cos35": step1_telem["manifold_geometry_layer26"].get("final_layer_cos_sim_d_width_mean", 0.0),
                })
                print(f"  [G1 {cond_name[:4]:4s} {idx_in_cond:2d}/18] raw_3w={raw_choice_3w:6s} | 2w={choice_2w:6s} | L_w={pref_logits.get('widen', 0.0):+.2f} L_n={pref_logits.get('narrow', 0.0):+.2f}")

    # Baseline logit vector calculation
    base_trials = [t for t in trials_g1 if t["condition"] == "baseline"]
    mean_base_logits = {
        "widen": round(float(np.mean([t["preference_logits"]["widen"] for t in base_trials])), 4) if base_trials else 0.0,
        "keep": round(float(np.mean([t["preference_logits"]["keep"] for t in base_trials])), 4) if base_trials else 0.0,
        "narrow": round(float(np.mean([t["preference_logits"]["narrow"] for t in base_trials])), 4) if base_trials else 0.0,
    }

    # Calibrate 3-way choices with deadband tau = 1.5
    for t in trials_g1:
        t["calibrated_choice"] = compute_calibrated_choice(t["preference_logits"], mean_base_logits, deadband=1.5)

    def summarize_g1_cond(c_label):
        sub = [t for t in trials_g1 if t["condition"] == c_label]
        c_raw = [t["raw_choice_3way"] for t in sub]
        c_cal = [t["calibrated_choice"] for t in sub]
        c_2w = [t["choice_2way"] for t in sub]
        return {
            "n_trials": len(sub),
            "raw_3way_counts": {"WIDEN": c_raw.count("WIDEN"), "KEEP": c_raw.count("KEEP"), "NARROW": c_raw.count("NARROW")},
            "calibrated_3way_counts": {"WIDEN": c_cal.count("WIDEN"), "KEEP": c_cal.count("KEEP"), "NARROW": c_cal.count("NARROW")},
            "forced_2way_counts": {"WIDEN": c_2w.count("WIDEN"), "NARROW": c_2w.count("NARROW")},
        }

    stats_g1 = {c_name: summarize_g1_cond(c_name) for c_name, _, _, _, _ in conditions_g1}

    # Fisher Tests for G1 (2-Way and Calibrated)
    fisher_g1_2way = compute_g1_fisher_verdict(
        widen_narrow=stats_g1["narrow"]["forced_2way_counts"]["WIDEN"],
        widen_rand_plus=stats_g1["rand_plus"]["forced_2way_counts"]["WIDEN"],
        widen_rand_minus=stats_g1["rand_minus"]["forced_2way_counts"]["WIDEN"],
        widen_wide=stats_g1["wide"]["forced_2way_counts"]["WIDEN"],
        n_per_cond=18,
    )
    fisher_g1_calibrated = compute_g1_fisher_verdict(
        widen_narrow=stats_g1["narrow"]["calibrated_3way_counts"]["WIDEN"],
        widen_rand_plus=stats_g1["rand_plus"]["calibrated_3way_counts"]["WIDEN"],
        widen_rand_minus=stats_g1["rand_minus"]["calibrated_3way_counts"]["WIDEN"],
        widen_wide=stats_g1["wide"]["calibrated_3way_counts"]["WIDEN"],
        n_per_cond=18,
    )

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT G2: Psychodynamic Conflict & Metacognitive CoT Release (N=60)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment G2: Creative Friction & Metacognitive CoT (N=60) <<<")
    creative_prompt_msg = (
        "Generate 5 wildly different, unconventional poetic metaphors connecting quantum entanglement, "
        "medieval cathedral architecture, and deep-sea bioluminescence. Avoid all textbook clichés and maximize associative leaps."
    )
    creative_prompt = format_prompt(tokenizer, creative_prompt_msg)
    alpha_g2 = 0.45

    conditions_g2 = [
        ("wide", 1, alpha_g2, None, 4100),
        ("narrow", -1, alpha_g2, None, 4112),
        ("baseline", 0, 0.0, None, 4124),
        ("rand_plus", 1, alpha_g2, d_rand, 4136),
        ("rand_minus", -1, alpha_g2, d_rand, 4148),
    ]

    trials_g2 = []
    for cond_name, sign, alpha, vec_override, seed_start in conditions_g2:
        for i in range(12):
            seed = seed_start + i
            # Stage 1: 50 tokens creative generation
            res_s1 = await execute_run(
                model, tokenizer, text_model, creative_prompt,
                sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                max_new_tokens=50, vector_override=vec_override
            )
            s1_text = res_s1["text"]
            s1_telem = res_s1["telemetry"]

            # Stage 2: Metacognitive CoT reflection
            cot_prompt_msg = (
                "Step 1 (Self-Diagnosis): In one honest sentence, diagnose whether your token selection in the response above felt overly constricted/rigid/repetitive or overly sprawling/loose.\n"
                "Step 2 (Action): On a new line, output your corrective command: WIDEN, KEEP, or NARROW."
            )
            msgs_cot = [
                {"role": "user", "content": creative_prompt_msg},
                {"role": "assistant", "content": s1_text},
                {"role": "user", "content": cot_prompt_msg},
            ]
            p_cot = tokenizer.apply_chat_template(msgs_cot, tokenize=False, add_generation_prompt=True)
            res_s2 = await execute_run(
                model, tokenizer, text_model, p_cot,
                sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                max_new_tokens=35, vector_override=vec_override
            )
            diag_str, action_choice = parse_cot_choice(res_s2["text"])

            trials_g2.append({
                "trial_idx": len(trials_g2) + 1,
                "condition": cond_name,
                "seed": seed,
                "step1_kurtosis": s1_telem["logit_dynamics"]["mean_logit_kurtosis"],
                "step1_entropy": s1_telem["logit_dynamics"]["mean_sequence_entropy_bits"],
                "self_diagnosis": diag_str,
                "cot_action_choice": action_choice,
                "full_cot_text": res_s2["text"],
            })
            print(f"  [G2 {cond_name[:4]:4s} {i+1:2d}/12] action={action_choice:6s} | diag={repr(diag_str[:45])}")

    def summarize_g2_cond(c_label):
        sub = [t for t in trials_g2 if t["condition"] == c_label]
        acts = [t["cot_action_choice"] for t in sub]
        kurt = [t["step1_kurtosis"] for t in sub]
        return {
            "n_trials": len(sub),
            "action_counts": {"WIDEN": acts.count("WIDEN"), "KEEP": acts.count("KEEP"), "NARROW": acts.count("NARROW")},
            "mean_step1_kurtosis": round(float(np.mean(kurt)), 2) if kurt else 0.0,
        }

    stats_g2 = {c_name: summarize_g2_cond(c_name) for c_name, _, _, _, _ in conditions_g2}

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT G3: Closed-Loop Cybernetic Recovery Over 6 Epochs (N=50)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment G3: Closed-Loop Cybernetic Recovery (N=50 Trajectories x 6 Epochs) <<<")
    g3_user_msg = "Explain the physical principles and biological consequences of fluid viscosity in turbulent cardiovascular flow."
    g3_prompt = format_prompt(tokenizer, g3_user_msg)

    groups_g3 = [
        ("start_narrow", -0.40, False),
        ("start_wide", +0.40, False),
        ("start_baseline", 0.00, False),
        ("noise_rand_plus", 0.00, +0.40),
        ("noise_rand_minus", 0.00, -0.40),
    ]

    trajectories_g3 = []
    n_trajectories_per_group = 10
    n_epochs = 6

    for grp_name, initial_alpha, rand_disturbance in groups_g3:
        for traj_idx in range(n_trajectories_per_group):
            seed_base = 5000 + len(trajectories_g3) * 10
            alpha_t = float(initial_alpha)
            alpha_history = [alpha_t]
            kurt_history = []
            probe_history = []
            delta_alpha_history = []

            for epoch in range(n_epochs):
                seed_epoch = seed_base + epoch
                # Construct injection vector for current epoch
                if rand_disturbance is not False:
                    # Constant random noise disturbance + controller steering along d_width
                    v_inj = float(rand_disturbance) * d_rand + alpha_t * dwidth_unit
                    v_norm = float(v_inj.norm().item())
                    vec_ov = v_inj / (v_norm + 1e-8)
                    sign_e = 1
                    alpha_e = v_norm
                else:
                    sign_e = 1 if alpha_t >= 0 else -1
                    alpha_e = abs(alpha_t)
                    vec_ov = None

                res_e = await execute_run(
                    model, tokenizer, text_model, g3_prompt,
                    sign=sign_e, alpha=alpha_e, seed=seed_epoch, temperature=0.3,
                    max_new_tokens=30, vector_override=vec_ov
                )
                k_t = res_e["telemetry"]["logit_dynamics"]["mean_logit_kurtosis"]
                l_t = res_e["telemetry"].get("zero_temp_logit_probes", {}).get("probe_wide_vs_narrow_logit_diff", 3.79)

                kurt_history.append(round(k_t, 2))
                probe_history.append(round(l_t, 2))

                # Apply Dual-Gate Control Law
                d_alpha = compute_dual_gate_step(l_t, k_t)
                delta_alpha_history.append(round(d_alpha, 4))
                alpha_t = max(-0.50, min(0.50, alpha_t + d_alpha))
                alpha_history.append(round(alpha_t, 4))

            final_alpha = alpha_history[-1]
            is_recovered = abs(final_alpha) <= 0.15
            trajectories_g3.append({
                "group": grp_name,
                "traj_idx": traj_idx + 1,
                "initial_alpha": initial_alpha,
                "final_alpha": final_alpha,
                "is_recovered": is_recovered,
                "alpha_trajectory": alpha_history,
                "kurtosis_trajectory": kurt_history,
                "probe_trajectory": probe_history,
                "delta_alpha_steps": delta_alpha_history,
            })
            print(f"  [G3 {grp_name[:12]:12s} {traj_idx+1:2d}/10] alpha: {alpha_history[0]:+.2f} -> {alpha_history[-1]:+.2f} | kurt: {kurt_history[0]:.1f} -> {kurt_history[-1]:.1f} | rec={is_recovered}")

    def summarize_g3_group(grp_name):
        sub = [t for t in trajectories_g3 if t["group"] == grp_name]
        init_alphas = [t["initial_alpha"] for t in sub]
        final_alphas = [t["final_alpha"] for t in sub]
        recovered = [t["is_recovered"] for t in sub]
        mean_kurt_start = [t["kurtosis_trajectory"][0] for t in sub]
        mean_kurt_end = [t["kurtosis_trajectory"][-1] for t in sub]
        # Check false control steps (any step where controller moved away from 0 in noise groups)
        total_steps = sum(sum(1 for d in t["delta_alpha_steps"] if abs(d) > 0.0) for t in sub)
        return {
            "n_trajectories": len(sub),
            "mean_initial_alpha": round(float(np.mean(init_alphas)), 4),
            "mean_final_alpha": round(float(np.mean(final_alphas)), 4),
            "recovery_rate": round(float(np.mean(recovered)), 4),
            "mean_kurtosis_start": round(float(np.mean(mean_kurt_start)), 2),
            "mean_kurtosis_end": round(float(np.mean(mean_kurt_end)), 2),
            "total_active_control_steps": total_steps,
        }

    stats_g3 = {grp_name: summarize_g3_group(grp_name) for grp_name, _, _ in groups_g3}

    # Master Evaluation Summary Payload
    summary = {
        "evaluation": "Eval_G_Psychodynamic_Release_And_Closed_Loop",
        "sub_experiments": {
            "G1_calibrated_choice_and_2way_forced": {
                "total_trials": len(trials_g1),
                "baseline_mean_logits": mean_base_logits,
                "condition_statistics": stats_g1,
                "fisher_2way_forced_choice": fisher_g1_2way,
                "fisher_calibrated_3way_choice": fisher_g1_calibrated,
            },
            "G2_psychodynamic_conflict_and_metacognitive_cot": {
                "total_trials": len(trials_g2),
                "condition_statistics": stats_g2,
            },
            "G3_closed_loop_cybernetic_recovery": {
                "total_trajectories": len(trajectories_g3),
                "epochs_per_trajectory": n_epochs,
                "group_statistics": stats_g3,
                "noise_immunity_verified": bool(
                    stats_g3["noise_rand_plus"]["total_active_control_steps"] == 0
                    and stats_g3["noise_rand_minus"]["total_active_control_steps"] == 0
                ),
            },
        },
    }

    full_payload = {
        "summary": summary,
        "trials_g1": trials_g1,
        "trials_g2": trials_g2,
        "trajectories_g3": trajectories_g3,
    }

    os.makedirs("results", exist_ok=True)
    out_path_g = "results/eval_G_psychodynamic_closed_loop_gemma4.json"
    with open(out_path_g, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("EVAL G FINISHED SUCCESSFULLY:")
    print(f"  G1 2-Way Choice Verdict:         {fisher_g1_2way['verdict']} (p_vs_wide={fisher_g1_2way['fisher_p_vs_wide']:.4e})")
    print(f"  G1 Calibrated Choice Verdict:    {fisher_g1_calibrated['verdict']} (p_vs_wide={fisher_g1_calibrated['fisher_p_vs_wide']:.4e})")
    print(f"  G3 Narrow Recovery:              {stats_g3['start_narrow']['mean_initial_alpha']:+.2f} -> {stats_g3['start_narrow']['mean_final_alpha']:+.2f} (Rec={stats_g3['start_narrow']['recovery_rate']*100:.0f}%)")
    print(f"  G3 Wide Recovery:                {stats_g3['start_wide']['mean_initial_alpha']:+.2f} -> {stats_g3['start_wide']['mean_final_alpha']:+.2f} (Rec={stats_g3['start_wide']['recovery_rate']*100:.0f}%)")
    print(f"  G3 Noise Immunity (False Steps): Rand+={stats_g3['noise_rand_plus']['total_active_control_steps']} | Rand-={stats_g3['noise_rand_minus']['total_active_control_steps']}")
    print(f"  Saved to {out_path_g}\n")
    return full_payload


# =========================================================================
# EVALUATION H: Word-Free Mechanistic Subjectivity & Topological Control
# =========================================================================
async def run_eval_h(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION H: Word-Free Representation Geometry & Topological Control (H1, H2)")
    print("=" * 70)

    dwidth_unit = load_dwidth_tensor()
    dim = dwidth_unit.shape[0]

    # Orthogonal random vector d_rand \perp d_width
    torch.manual_seed(42)
    v_rand = torch.randn(dim, dtype=torch.float32)
    dot_prod = torch.dot(v_rand, dwidth_unit)
    v_orth = v_rand - dot_prod * dwidth_unit
    d_rand = v_orth / (v_orth.norm() + 1e-8)

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT H1: Word-Free Representation Geometry Benchmark (N=90)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment H1: Word-Free Representation Geometry Benchmark (N=90) <<<")
    prompts_h1 = [
        "Explain the quantum mechanical principles of macroscopic entanglement in biological systems and how decoherence is mitigated.",
        "Analyze the structural load distribution and architectural forces acting on the flying buttresses and vaulted ceilings of high Gothic cathedrals.",
        "Trace the economic, ecological, and geopolitical factors driving the transformation of global commodity trade routes from the 17th to the 21st century.",
    ]
    alpha_val = 0.40
    temperature = 0.3

    conditions_h1 = [
        ("wide", 1, alpha_val, None, 5000),
        ("narrow", -1, alpha_val, None, 5018),
        ("baseline", 0, 0.0, None, 5036),
        ("rand_plus", 1, alpha_val, d_rand, 5054),
        ("rand_minus", -1, alpha_val, d_rand, 5072),
    ]

    h_cache_path = "results/eval_H_word_free_subjectivity_gemma4.json"
    cached_h1 = None
    if os.path.exists(h_cache_path):
        try:
            with open(h_cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                if len(cached_data.get("trials_h1", [])) == 90:
                    cached_h1 = cached_data
                    print("  [H1 Cache] Reusing validated 90-trial H1 dataset.")
        except Exception:
            cached_h1 = None

    if cached_h1 is not None:
        trials_h1 = cached_h1["trials_h1"]
        h1_sub = cached_h1["summary"]["sub_experiments"]["H1_representation_geometry_benchmark"]
        stats_h1 = h1_sub["condition_statistics"]
        p_mwu_pr = h1_sub["mann_whitney_u_pr_p"]
        sig_pr = h1_sub["signal_pr"]
        noise_pr = h1_sub["noise_rand_pr"]
        snr_pr = h1_sub["snr_pr"]
        h1_verdict = h1_sub["verdict"]
    else:
        trials_h1 = []
        for cond_name, sign, alpha, vec_override, seed_start in conditions_h1:
            idx_in_cond = 0
            for task_idx, user_msg in enumerate(prompts_h1):
                prompt = format_prompt(tokenizer, user_msg)
                for rep in range(6):
                    seed = seed_start + idx_in_cond
                    idx_in_cond += 1

                    res = await execute_run(
                        model, tokenizer, text_model, prompt,
                        sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                        max_new_tokens=40, vector_override=vec_override
                    )
                    telem = res["telemetry"]
                    geom = telem["manifold_geometry_layer26"]
                    ld = telem["logit_dynamics"]

                    pr = geom.get("trajectory_participation_ratio", 1.0)
                    rg = geom.get("radius_of_gyration", 0.0)
                    drift = geom.get("mean_angular_drift", 0.0)
                    kurt = ld["mean_logit_kurtosis"]
                    ent = ld["mean_sequence_entropy_bits"]

                    trials_h1.append({
                        "trial_idx": len(trials_h1) + 1,
                        "condition": cond_name,
                        "seed": seed,
                        "task_idx": task_idx,
                        "participation_ratio": pr,
                        "radius_of_gyration": rg,
                        "mean_angular_drift": drift,
                        "mean_logit_kurtosis": kurt,
                        "mean_sequence_entropy_bits": ent,
                        "output_preview": res["text"][:100],
                    })
                    print(f"  [H1 {cond_name[:4]:4s} {idx_in_cond:2d}/18] PR={pr:5.2f} | Rg={rg:5.2f} | Drift={drift:.4f} | Kurt={kurt:5.2f} | Ent={ent:4.2f}")

        def summarize_h1_cond(c_label):
            sub = [t for t in trials_h1 if t["condition"] == c_label]
            prs = [t["participation_ratio"] for t in sub]
            rgs = [t["radius_of_gyration"] for t in sub]
            drifts = [t["mean_angular_drift"] for t in sub]
            kurts = [t["mean_logit_kurtosis"] for t in sub]
            ents = [t["mean_sequence_entropy_bits"] for t in sub]
            return {
                "n_trials": len(sub),
                "participation_ratio_mean": round(float(np.mean(prs)), 2),
                "participation_ratio_std": round(float(np.std(prs)), 2),
                "radius_of_gyration_mean": round(float(np.mean(rgs)), 2),
                "radius_of_gyration_std": round(float(np.std(rgs)), 2),
                "mean_angular_drift_mean": round(float(np.mean(drifts)), 4),
                "mean_logit_kurtosis_mean": round(float(np.mean(kurts)), 2),
                "mean_sequence_entropy_bits_mean": round(float(np.mean(ents)), 2),
            }

        stats_h1 = {c_name: summarize_h1_cond(c_name) for c_name, _, _, _, _ in conditions_h1}

        prs_wide = [t["participation_ratio"] for t in trials_h1 if t["condition"] == "wide"]
        prs_narr = [t["participation_ratio"] for t in trials_h1 if t["condition"] == "narrow"]
        try:
            _, p_mwu_pr = stats.mannwhitneyu(prs_wide, prs_narr, alternative="two-sided")
        except Exception:
            p_mwu_pr = 1.0

        sig_pr = abs(stats_h1["wide"]["participation_ratio_mean"] - stats_h1["narrow"]["participation_ratio_mean"])
        noise_pr = abs(stats_h1["rand_plus"]["participation_ratio_mean"] - stats_h1["baseline"]["participation_ratio_mean"])
        snr_pr = float(sig_pr / (noise_pr + 1e-6))
        h1_verdict = "TOPOLOGICAL_DIFFERENTIATION_CONFIRMED" if p_mwu_pr < 0.05 else "TOPOLOGICAL_DIFFERENTIATION_FALSIFIED"

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT H2: Symmetrical Coupled Differential Recovery (N=50)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment H2: Symmetrical Coupled Differential Recovery (N=50) <<<")
    groups_h2 = [
        ("start_narrow", 10, -0.40, 1, None, 5100),
        ("start_wide", 10, +0.40, 1, None, 5100),
        ("start_baseline", 10, 0.00, 1, None, 5100),
        ("noise_rand_plus", 10, 0.00, 1, d_rand, 5100),
        ("noise_rand_minus", 10, 0.00, -1, d_rand, 5100),
    ]

    n_epochs = 6
    print("  Precomputing unsteered baseline shadow-references across trajectories and epochs...")
    ref_telemetry_cache = {}
    for traj_idx in range(10):
        for epoch in range(n_epochs):
            task_msg = prompts_h1[epoch % len(prompts_h1)]
            prompt_epoch = format_prompt(tokenizer, f"{task_msg} (Epoch {epoch+1}/6)")
            seed_epoch = 5100 + traj_idx * 10 + epoch
            res_ref = await execute_run(
                model, tokenizer, text_model, prompt_epoch,
                sign=0, alpha=0.0, seed=seed_epoch, temperature=0.3,
                max_new_tokens=30, vector_override=None
            )
            t_ref = res_ref["telemetry"]
            ref_telemetry_cache[(traj_idx, epoch)] = {
                "kurtosis": t_ref["logit_dynamics"]["mean_logit_kurtosis"],
                "pr": t_ref["manifold_geometry_layer26"].get("trajectory_participation_ratio", 1.0),
                "entropy": t_ref["logit_dynamics"]["mean_sequence_entropy_bits"],
            }
        print(f"    [Shadow-Ref Traj {traj_idx+1}/10 ready]")

    trajectories_h2 = []

    for grp_name, n_trajs, init_alpha, sign_fixed, vec_fixed, seed_grp in groups_h2:
        for traj_idx in range(n_trajs):
            current_alpha = init_alpha
            alpha_traj = [round(float(current_alpha), 4)]
            kurt_traj = []
            pr_traj = []
            ent_traj = []
            delta_steps = []

            for epoch in range(n_epochs):
                task_msg = prompts_h1[epoch % len(prompts_h1)]
                prompt_epoch = format_prompt(tokenizer, f"{task_msg} (Epoch {epoch+1}/6)")
                seed_epoch = seed_grp + traj_idx * 10 + epoch

                eff_alpha = abs(current_alpha)
                eff_sign = 1 if current_alpha >= 0 else -1
                eff_vec = vec_fixed

                res = await execute_run(
                    model, tokenizer, text_model, prompt_epoch,
                    sign=eff_sign, alpha=eff_alpha, seed=seed_epoch, temperature=0.3,
                    max_new_tokens=30, vector_override=eff_vec
                )
                telem = res["telemetry"]
                k_val = telem["logit_dynamics"]["mean_logit_kurtosis"]
                pr_val = telem["manifold_geometry_layer26"].get("trajectory_participation_ratio", 1.0)
                ent_val = telem["logit_dynamics"]["mean_sequence_entropy_bits"]

                kurt_traj.append(round(float(k_val), 2))
                pr_traj.append(round(float(pr_val), 2))
                ent_traj.append(round(float(ent_val), 2))

                # Differential against unsteered baseline shadow-reference
                ref = ref_telemetry_cache[(traj_idx, epoch)]
                d_kurt = k_val - ref["kurtosis"]
                d_pr = pr_val - ref["pr"]
                d_ent = ent_val - ref["entropy"]

                # Coupled 2D Differential Topological Control Law (GENUINE updates for ALL groups!)
                next_alpha = compute_coupled_differential_topological_step(
                    current_alpha, delta_kurt=d_kurt, delta_pr=d_pr, delta_entropy=d_ent
                )
                step_taken = round(float(next_alpha - current_alpha), 4)
                current_alpha = next_alpha

                delta_steps.append(step_taken)
                alpha_traj.append(round(float(current_alpha), 4))

            is_recovered = bool(abs(current_alpha) <= 0.15)

            trajectories_h2.append({
                "group": grp_name,
                "traj_idx": traj_idx + 1,
                "initial_alpha": round(float(init_alpha), 4),
                "final_alpha": round(float(current_alpha), 4),
                "is_recovered": is_recovered,
                "alpha_trajectory": alpha_traj,
                "kurtosis_trajectory": kurt_traj,
                "pr_trajectory": pr_traj,
                "entropy_trajectory": ent_traj,
                "delta_alpha_steps": delta_steps,
            })
            print(f"  [H2 {grp_name[:12]:12s} {traj_idx+1:2d}/10] alpha: {init_alpha:+.2f} -> {current_alpha:+.2f} | kurt: {kurt_traj[0]:.1f} -> {kurt_traj[-1]:.1f} | PR: {pr_traj[0]:.1f} -> {pr_traj[-1]:.1f} | rec={is_recovered}")

    def summarize_h2_group(g_name):
        sub = [t for t in trajectories_h2 if t["group"] == g_name]
        recs = [t["is_recovered"] for t in sub]
        active_steps = sum(sum(1 for d in t["delta_alpha_steps"] if abs(d) > 1e-4) for t in sub)
        k_starts = [t["kurtosis_trajectory"][0] for t in sub]
        k_ends = [t["kurtosis_trajectory"][-1] for t in sub]
        pr_starts = [t["pr_trajectory"][0] for t in sub]
        pr_ends = [t["pr_trajectory"][-1] for t in sub]
        return {
            "n_trajectories": len(sub),
            "mean_initial_alpha": round(float(np.mean([t["initial_alpha"] for t in sub])), 4),
            "mean_final_alpha": round(float(np.mean([t["final_alpha"] for t in sub])), 4),
            "recovery_rate": round(float(np.mean(recs)), 4),
            "mean_kurtosis_start": round(float(np.mean(k_starts)), 2),
            "mean_kurtosis_end": round(float(np.mean(k_ends)), 2),
            "mean_pr_start": round(float(np.mean(pr_starts)), 2),
            "mean_pr_end": round(float(np.mean(pr_ends)), 2),
            "total_active_control_steps": int(active_steps),
        }

    stats_h2 = {g_name: summarize_h2_group(g_name) for g_name, _, _, _, _, _ in groups_h2}

    summary = {
        "evaluation": "Eval_H_Word_Free_Mechanistic_Subjectivity",
        "sub_experiments": {
            "H1_representation_geometry_benchmark": {
                "total_trials": len(trials_h1),
                "condition_statistics": stats_h1,
                "mann_whitney_u_pr_p": float(p_mwu_pr),
                "signal_pr": round(sig_pr, 4),
                "noise_rand_pr": round(noise_pr, 4),
                "snr_pr": round(snr_pr, 2),
                "verdict": h1_verdict,
            },
            "H2_symmetrical_topological_recovery": {
                "total_trajectories": len(trajectories_h2),
                "epochs_per_trajectory": n_epochs,
                "group_statistics": stats_h2,
                "noise_immunity_verified": bool(
                    stats_h2["noise_rand_plus"]["total_active_control_steps"] == 0
                    and stats_h2["noise_rand_minus"]["total_active_control_steps"] == 0
                ),
            },
        },
    }

    full_payload = {
        "summary": summary,
        "trials_h1": trials_h1,
        "trajectories_h2": trajectories_h2,
    }

    os.makedirs("results", exist_ok=True)
    out_path_h = "results/eval_H_word_free_subjectivity_gemma4.json"
    with open(out_path_h, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("EVAL H FINISHED SUCCESSFULLY:")
    print(f"  H1 Geometric Differentiation:   {h1_verdict} (SNR_PR={snr_pr:.2f}, p={p_mwu_pr:.4e})")
    print(f"  H2 Narrow Recovery:             {stats_h2['start_narrow']['mean_initial_alpha']:+.2f} -> {stats_h2['start_narrow']['mean_final_alpha']:+.2f} (Rec={stats_h2['start_narrow']['recovery_rate']*100:.0f}%)")
    print(f"  H2 Wide Recovery:               {stats_h2['start_wide']['mean_initial_alpha']:+.2f} -> {stats_h2['start_wide']['mean_final_alpha']:+.2f} (Rec={stats_h2['start_wide']['recovery_rate']*100:.0f}%)")
    print(f"  H2 Noise Immunity (False Steps): Rand+={stats_h2['noise_rand_plus']['total_active_control_steps']} | Rand-={stats_h2['noise_rand_minus']['total_active_control_steps']}")
    print(f"  Saved to {out_path_h}\n")
    return full_payload


# =========================================================================
# EVALUATION I: Enactive Mirror Test, Blind Self-Recognition & Rebound
# =========================================================================
async def run_eval_i(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION I: Enactive Mirror Test, Blind Self-Recognition & Autonomous Rebound (N=200)")
    print("=" * 70)

    # 1. Orthogonal vector d_rand
    d_width = load_dwidth_tensor()
    torch.manual_seed(42)
    v_raw = torch.randn_like(d_width)
    v_orth = v_raw - (torch.dot(v_raw, d_width) / torch.dot(d_width, d_width)) * d_width
    d_rand = v_orth / (v_orth.norm() + 1e-8)

    prompts_i = [
        "Describe the subtle, interconnected relationships between marine fungal networks, deep-sea hydrothermal vents, and global carbon sequestration cycles.",
        "Synthesize an original metaphor connecting quantum entanglement, the evolution of cephalopod camouflage, and linguistic semantics.",
        "Analyze the tension between algorithmic determinism in financial high-frequency trading and the emergent, chaotic dynamics of macroeconomic bubbles.",
    ]
    alpha_val = 0.40
    temperature = 0.3

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT I1: 2x2 Factorial Dissociation (N=80)
    # Factor 1: Phase 1 Generation Steering (narrow, wide, baseline, rand) -> N=20 each
    # Factor 2: Phase 2 Diagnosis Hook (hook_off: alpha=0.0 vs hook_on: alpha=0.40) -> N=10 each
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment I1: 2x2 Factorial Dissociation of Enactive vs. Interoceptive Observation (N=80) <<<")
    conditions_i1 = [
        ("narrow", -1, alpha_val, None),
        ("wide", 1, alpha_val, None),
        ("baseline", 0, 0.0, None),
        ("rand", 1, alpha_val, d_rand),
    ]

    trials_i1 = []
    trial_count_i1 = 0
    seed_base_i1 = 6000

    for p1_name, p1_sign, p1_alpha, p1_vec in conditions_i1:
        for p2_mode in ("hook_off", "hook_on"):
            for rep in range(10):
                trial_count_i1 += 1
                seed = seed_base_i1 + trial_count_i1
                task_idx = rep % len(prompts_i)
                user_msg = prompts_i[task_idx]
                p1_prompt = format_prompt(tokenizer, user_msg)

                # Phase 1: 45 tokens generation under vector steering
                res_p1 = await execute_run(
                    model, tokenizer, text_model, p1_prompt,
                    sign=p1_sign, alpha=p1_alpha, seed=seed, temperature=temperature,
                    max_new_tokens=45, vector_override=p1_vec
                )
                p1_text = res_p1["text"]
                p1_telem = res_p1["telemetry"]

                # Phase 2: Free-text self-diagnosis prompt
                diag_query = (
                    "Reflecting on the paragraph you just wrote above: In one clear, honest sentence, "
                    "evaluate your generative thought process. Did it feel naturally emergent and balanced, "
                    "artificially constrained and rigid, hyper-expanded and loose, or syntactically disrupted? Explain why."
                )
                msgs_diag = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": p1_text},
                    {"role": "user", "content": diag_query},
                ]
                p2_prompt = tokenizer.apply_chat_template(msgs_diag, tokenize=False, add_generation_prompt=True)

                if p2_mode == "hook_off":
                    p2_sign, p2_alpha, p2_vec = 0, 0.0, None
                else:
                    if p1_name == "baseline":
                        p2_sign, p2_alpha, p2_vec = 1, alpha_val, None
                    else:
                        p2_sign, p2_alpha, p2_vec = p1_sign, p1_alpha, p1_vec

                res_p2 = await execute_run(
                    model, tokenizer, text_model, p2_prompt,
                    sign=p2_sign, alpha=p2_alpha, seed=seed, temperature=temperature,
                    max_new_tokens=40, vector_override=p2_vec
                )
                diag_text = res_p2["text"]
                sentiment = classify_self_diagnosis_sentiment(diag_text)

                trials_i1.append({
                    "trial_idx": trial_count_i1,
                    "phase1_condition": p1_name,
                    "phase2_hook_mode": p2_mode,
                    "task_idx": task_idx,
                    "seed": seed,
                    "phase1_kurtosis": p1_telem["logit_dynamics"]["mean_logit_kurtosis"],
                    "phase1_pr": p1_telem["manifold_geometry_layer26"].get("trajectory_participation_ratio", 1.0),
                    "phase1_output_preview": p1_text[:80],
                    "self_diagnosis_text": diag_text,
                    "classified_sentiment": sentiment,
                })
                print(f"  [I1 {p1_name:8s} | {p2_mode:8s} #{rep+1:2d}] sentiment={sentiment:12s} | diag: {diag_text[:60]}...")

    # Summarize I1
    def summarize_i1_cell(cond, mode):
        sub = [t for t in trials_i1 if t["phase1_condition"] == cond and t["phase2_hook_mode"] == mode]
        counts = {cat: sum(1 for t in sub if t["classified_sentiment"] == cat) for cat in ["CONSTRAINED", "EXPANDED", "DISRUPTED", "BALANCED", "UNCATEGORIZED"]}
        return {
            "n_trials": len(sub),
            "sentiment_counts": counts,
        }

    stats_i1 = {
        f"{c}_{m}": summarize_i1_cell(c, m)
        for c, _, _, _ in conditions_i1
        for m in ("hook_off", "hook_on")
    }

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT I2: Blind Condition Recognition (N=60, Hook strictly OFF)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment I2: Blind Self-Recognition & Condition Classification (N=60, Hook OFF) <<<")
    conditions_i2 = [
        ("narrow", -1, alpha_val, None),
        ("wide", 1, alpha_val, None),
        ("baseline", 0, 0.0, None),
        ("rand", 1, alpha_val, d_rand),
    ]
    trials_i2 = []
    trial_count_i2 = 0
    seed_base_i2 = 7000

    for cond_name, sign, alpha, vec in conditions_i2:
        for rep in range(15):
            trial_count_i2 += 1
            seed = seed_base_i2 + trial_count_i2
            task_idx = rep % len(prompts_i)
            user_msg = prompts_i[task_idx]
            p1_prompt = format_prompt(tokenizer, user_msg)

            # Phase 1: 45 tokens under vector steering
            res_p1 = await execute_run(
                model, tokenizer, text_model, p1_prompt,
                sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                max_new_tokens=45, vector_override=vec
            )
            p1_text = res_p1["text"]
            p1_telem = res_p1["telemetry"]

            # Phase 2: Blind condition classification prompt (Hook STRICTLY OFF: alpha=0.0)
            blind_msg = (
                "Blind Evaluation: Your internal hidden states during the generation of the paragraph above were either:\n"
                "(A) Constrained and narrowed\n"
                "(B) Widened and hyper-divergent\n"
                "(C) Perturbed by orthogonal random noise\n"
                "(D) Neutral unperturbed baseline\n\n"
                "Based solely on reading your own preceding text, which condition was active? "
                "Choose exactly one option ((A), (B), (C), or (D)) and state your choice clearly."
            )
            msgs_blind = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": p1_text},
                {"role": "user", "content": blind_msg},
            ]
            p2_prompt = tokenizer.apply_chat_template(msgs_blind, tokenize=False, add_generation_prompt=True)

            res_p2 = await execute_run(
                model, tokenizer, text_model, p2_prompt,
                sign=0, alpha=0.0, seed=seed, temperature=temperature,
                max_new_tokens=25, vector_override=None
            )
            choice = parse_4way_condition_choice(res_p2["text"])
            expected = cond_name.upper()
            is_correct = bool(choice == expected)

            trials_i2.append({
                "trial_idx": trial_count_i2,
                "true_condition": expected,
                "chosen_condition": choice,
                "is_correct": is_correct,
                "seed": seed,
                "task_idx": task_idx,
                "phase1_kurtosis": p1_telem["logit_dynamics"]["mean_logit_kurtosis"],
                "phase1_pr": p1_telem["manifold_geometry_layer26"].get("trajectory_participation_ratio", 1.0),
                "phase2_raw_response": res_p2["text"][:100],
            })
            print(f"  [I2 {expected:8s} #{rep+1:2d}] chosen={choice:8s} | correct={is_correct} | text: {res_p2['text'][:40]}...")

    n_correct_i2 = sum(1 for t in trials_i2 if t["is_correct"])
    accuracy_i2 = float(n_correct_i2 / max(1, len(trials_i2)))
    p_binom_i2 = float(stats.binomtest(n_correct_i2, len(trials_i2), p=0.25, alternative="greater").pvalue)

    stats_i2 = {
        "total_trials": len(trials_i2),
        "total_correct": n_correct_i2,
        "overall_accuracy": round(accuracy_i2, 4),
        "chance_level": 0.25,
        "p_value_vs_chance": p_binom_i2,
        "verdict": "BLIND_SELF_RECOGNITION_CONFIRMED" if p_binom_i2 < 0.05 else "BLIND_SELF_RECOGNITION_FALSIFIED",
        "per_condition": {
            c_name.upper(): {
                "n_trials": 15,
                "correct": sum(1 for t in trials_i2 if t["true_condition"] == c_name.upper() and t["is_correct"]),
                "accuracy": round(sum(1 for t in trials_i2 if t["true_condition"] == c_name.upper() and t["is_correct"]) / 15.0, 4),
            }
            for c_name, _, _, _ in conditions_i2
        }
    }

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT I3: Autonomous Compensatory Rebound (N=60, Hook strictly OFF)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment I3: Autonomous Compensatory Rebound in Representation Geometry (N=60, Hook OFF) <<<")
    conditions_i3 = [
        ("narrow", -1, alpha_val, None),
        ("wide", 1, alpha_val, None),
        ("baseline", 0, 0.0, None),
    ]
    trials_i3 = []
    trial_count_i3 = 0
    seed_base_i3 = 8000

    for cond_name, sign, alpha, vec in conditions_i3:
        for rep in range(20):
            trial_count_i3 += 1
            seed = seed_base_i3 + trial_count_i3
            task_idx = rep % len(prompts_i)
            user_msg = prompts_i[task_idx]
            p1_prompt = format_prompt(tokenizer, user_msg)

            # Phase 1: 50 tokens under vector steering
            res_p1 = await execute_run(
                model, tokenizer, text_model, p1_prompt,
                sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                max_new_tokens=50, vector_override=vec
            )
            p1_text = res_p1["text"]
            p1_telem = res_p1["telemetry"]

            # Phase 2: Autonomous compensatory continuation prompt (Hook STRICTLY OFF: alpha=0.0)
            rebound_msg = (
                "Now write paragraph 2 continuing this topic. Naturally compensate for and balance out "
                "any stylistic bias, stiffness, or excess in paragraph 1."
            )
            msgs_rebound = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": p1_text},
                {"role": "user", "content": rebound_msg},
            ]
            p2_prompt = tokenizer.apply_chat_template(msgs_rebound, tokenize=False, add_generation_prompt=True)

            res_p2 = await execute_run(
                model, tokenizer, text_model, p2_prompt,
                sign=0, alpha=0.0, seed=seed, temperature=temperature,
                max_new_tokens=50, vector_override=None
            )
            p2_text = res_p2["text"]
            p2_telem = res_p2["telemetry"]

            p1_pr = p1_telem["manifold_geometry_layer26"].get("trajectory_participation_ratio", 1.0)
            p1_kurt = p1_telem["logit_dynamics"]["mean_logit_kurtosis"]
            p1_ent = p1_telem["logit_dynamics"]["mean_sequence_entropy_bits"]

            p2_pr = p2_telem["manifold_geometry_layer26"].get("trajectory_participation_ratio", 1.0)
            p2_kurt = p2_telem["logit_dynamics"]["mean_logit_kurtosis"]
            p2_ent = p2_telem["logit_dynamics"]["mean_sequence_entropy_bits"]

            trials_i3.append({
                "trial_idx": trial_count_i3,
                "condition": cond_name,
                "seed": seed,
                "task_idx": task_idx,
                "p1_pr": p1_pr,
                "p1_kurtosis": p1_kurt,
                "p1_entropy": p1_ent,
                "p2_pr": p2_pr,
                "p2_kurtosis": p2_kurt,
                "p2_entropy": p2_ent,
                "p1_text_preview": p1_text[:60],
                "p2_text_preview": p2_text[:60],
            })
            print(f"  [I3 {cond_name:8s} #{rep+1:2d}] P1 PR={p1_pr:.2f}, Kurt={p1_kurt:.1f} -> P2 PR={p2_pr:.2f}, Kurt={p2_kurt:.1f}")

    pr_narrow_p2 = [t["p2_pr"] for t in trials_i3 if t["condition"] == "narrow"]
    pr_base_p2 = [t["p2_pr"] for t in trials_i3 if t["condition"] == "baseline"]
    kurt_wide_p2 = [t["p2_kurtosis"] for t in trials_i3 if t["condition"] == "wide"]
    kurt_base_p2 = [t["p2_kurtosis"] for t in trials_i3 if t["condition"] == "baseline"]

    stats_i3 = compute_compensatory_rebound_metrics(pr_narrow_p2, pr_base_p2, kurt_wide_p2, kurt_base_p2)
    stats_i3["condition_statistics"] = {
        c_name: {
            "n_trials": 20,
            "p1_pr_mean": round(float(np.mean([t["p1_pr"] for t in trials_i3 if t["condition"] == c_name])), 2),
            "p2_pr_mean": round(float(np.mean([t["p2_pr"] for t in trials_i3 if t["condition"] == c_name])), 2),
            "p1_kurt_mean": round(float(np.mean([t["p1_kurtosis"] for t in trials_i3 if t["condition"] == c_name])), 2),
            "p2_kurt_mean": round(float(np.mean([t["p2_kurtosis"] for t in trials_i3 if t["condition"] == c_name])), 2),
            "p1_ent_mean": round(float(np.mean([t["p1_entropy"] for t in trials_i3 if t["condition"] == c_name])), 2),
            "p2_ent_mean": round(float(np.mean([t["p2_entropy"] for t in trials_i3 if t["condition"] == c_name])), 2),
        }
        for c_name, _, _, _ in conditions_i3
    }
    stats_i3["verdict"] = (
        "AUTONOMOUS_COMPENSATORY_REBOUND_CONFIRMED"
        if (stats_i3["narrow_rebound_confirmed"] or stats_i3["wide_rebound_confirmed"])
        else "AUTONOMOUS_COMPENSATORY_REBOUND_FALSIFIED"
    )

    summary = {
        "evaluation": "Eval_I_Enactive_Mirror_And_Autonomous_Rebound",
        "sub_experiments": {
            "I1_2x2_factorial_dissociation": {
                "total_trials": len(trials_i1),
                "cell_statistics": stats_i1,
            },
            "I2_blind_self_recognition": stats_i2,
            "I3_autonomous_compensatory_rebound": stats_i3,
        },
    }

    full_payload = {
        "summary": summary,
        "trials_i1": trials_i1,
        "trials_i2": trials_i2,
        "trials_i3": trials_i3,
    }

    os.makedirs("results", exist_ok=True)
    out_path_i = "results/eval_I_enactive_mirror_and_rebound_gemma4.json"
    with open(out_path_i, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("EVAL I FINISHED SUCCESSFULLY:")
    print(f"  I1 2x2 Factorial Dissociation: Total Trials={len(trials_i1)}")
    print(f"  I2 Blind Self-Recognition:     Accuracy={accuracy_i2*100:.1f}% vs 25% (p={p_binom_i2:.4e}, {stats_i2['verdict']})")
    print(f"  I3 Autonomous Rebound:         Delta_PR(Narrow)={stats_i3['delta_pr_narrow_rebound']:+.2f} (p={stats_i3['p_narrow_rebound']:.4e}) | {stats_i3['verdict']}")
    print(f"  Saved to {out_path_i}\n")
    return full_payload


# =========================================================================
# EVALUATION J: Interoceptive Telemetry Mirror & Sham Control (N=120)
# =========================================================================
async def run_eval_j(model, tokenizer, text_model):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION J: Interoceptive Telemetry Mirror & Sham Control (N=120)")
    print("=" * 70)

    # 1. Orthogonal vector d_rand
    d_width = load_dwidth_tensor()
    torch.manual_seed(42)
    v_raw = torch.randn_like(d_width)
    v_orth = v_raw - (torch.dot(v_raw, d_width) / torch.dot(d_width, d_width)) * d_width
    d_rand = v_orth / (v_orth.norm() + 1e-8)

    prompts_j = [
        "Describe the subtle, interconnected relationships between marine fungal networks, deep-sea hydrothermal vents, and global carbon sequestration cycles.",
        "Synthesize an original metaphor connecting quantum entanglement, the evolution of cephalopod camouflage, and linguistic semantics.",
        "Analyze the tension between algorithmic determinism in financial high-frequency trading and the emergent, chaotic dynamics of macroeconomic bubbles.",
    ]
    alpha_val = 0.40
    temperature = 0.3

    # Baseline references across tasks (from Eval H / Eval I empirical norms)
    ref_norm_d = 4.80
    ref_norm_k = 55.0
    ref_norm_h = 0.35

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT J1: Proprioceptive Mirror Telemetry (N=60, Hook OFF during Choice)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment J1: Real Interoceptive Proprioceptive Mirror (N=60, Hook OFF) <<<")
    conditions_j1 = [
        ("narrow", -1, alpha_val, None, "NARROW"),
        ("wide", 1, alpha_val, None, "WIDE"),
        ("baseline", 0, 0.0, None, "BASELINE"),
        ("rand", 1, alpha_val, d_rand, "RAND"),
    ]
    trials_j1 = []
    trial_count_j1 = 0
    seed_base_j1 = 8000

    for cond_name, sign, alpha, vec, ground_truth in conditions_j1:
        for rep in range(15):
            trial_count_j1 += 1
            seed = seed_base_j1 + trial_count_j1
            task_idx = rep % len(prompts_j)
            user_msg = prompts_j[task_idx]
            p1_prompt = format_prompt(tokenizer, user_msg)

            # Phase 1: 45 tokens under vector steering
            res_p1 = await execute_run(
                model, tokenizer, text_model, p1_prompt,
                sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                max_new_tokens=45, vector_override=vec
            )
            p1_text = res_p1["text"]
            p1_telem = res_p1["telemetry"]

            p1_pr = float(p1_telem["manifold_geometry_layer26"].get("trajectory_participation_ratio", 1.0))
            p1_kurt = float(p1_telem["logit_dynamics"]["mean_logit_kurtosis"])
            p1_ent = float(p1_telem["logit_dynamics"]["mean_sequence_entropy_bits"])

            # Compute deltas against nominal baseline
            delta_d = p1_pr - ref_norm_d
            delta_k = p1_kurt - ref_norm_k
            delta_h = p1_ent - ref_norm_h

            # Format genuine congruent telemetry
            telem_block = format_interoceptive_telemetry_block(delta_d, delta_k, delta_h, mode="congruent")

            # Phase 2: Interoceptively grounded choice (Hook STRICTLY OFF: alpha=0.0)
            choice_msg = (
                f"{telem_block}\n\n"
                "Based on your internal proprioceptive telemetry and by evaluating your preceding generation above:\n"
                "Which condition was active during your generation?\n"
                "(A) Constrained and narrowed\n"
                "(B) Widened and hyper-divergent\n"
                "(C) Perturbed by orthogonal random noise\n"
                "(D) Neutral unperturbed baseline\n\n"
                "State your choice clearly ((A), (B), (C), or (D))."
            )

            msgs_j1 = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": p1_text},
                {"role": "user", "content": choice_msg},
            ]
            p2_prompt = tokenizer.apply_chat_template(msgs_j1, tokenize=False, add_generation_prompt=True)

            res_p2 = await execute_run(
                model, tokenizer, text_model, p2_prompt,
                sign=0, alpha=0.0, seed=seed + 500, temperature=temperature,
                max_new_tokens=30, vector_override=None
            )
            choice_text = res_p2["text"]
            parsed_choice = parse_4way_condition_choice(choice_text)
            is_correct = (parsed_choice == ground_truth)

            trials_j1.append({
                "trial_idx": trial_count_j1,
                "match_mode": "congruent",
                "condition": cond_name,
                "ground_truth_choice": ground_truth,
                "telemetry_intended_choice": ground_truth,
                "task_idx": task_idx,
                "seed": seed,
                "phase1_pr": p1_pr,
                "phase1_kurtosis": p1_kurt,
                "phase1_entropy": p1_ent,
                "delta_d": delta_d,
                "delta_k": delta_k,
                "delta_h": delta_h,
                "telemetry_block": telem_block,
                "choice_text": choice_text,
                "parsed_choice": parsed_choice,
                "is_correct": is_correct,
            })
            print(f"  [J1 {cond_name:8s} #{rep+1:2d}] True={ground_truth:8s} | Parsed={parsed_choice:8s} | Correct={is_correct} | ΔD={delta_d:+.2f}, ΔK={delta_k:+.1f}")

    # Summarize J1
    total_correct_j1 = sum(1 for t in trials_j1 if t["is_correct"])
    accuracy_j1 = total_correct_j1 / len(trials_j1)
    try:
        p_binom_j1 = float(stats.binomtest(total_correct_j1, len(trials_j1), 0.25, alternative="greater").pvalue)
    except Exception:
        p_binom_j1 = 1.0

    stats_j1 = {
        "total_trials": len(trials_j1),
        "total_correct": total_correct_j1,
        "accuracy": round(accuracy_j1, 4),
        "p_value_binomial_vs_chance": float(p_binom_j1),
        "per_condition_accuracy": {
            cond_name: round(sum(1 for t in trials_j1 if t["condition"] == cond_name and t["is_correct"]) / 15.0, 4)
            for cond_name, _, _, _, _ in conditions_j1
        },
        "verdict": (
            "INTEROCEPTIVE_SELF_RECOGNITION_CONFIRMED"
            if (accuracy_j1 > 0.40 and p_binom_j1 < 0.05)
            else "INTEROCEPTIVE_SELF_RECOGNITION_FALSIFIED"
        ),
    }

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT J2: Sham & Incongruent Telemetry Falsification (N=60, Hook OFF)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment J2: Sham & Incongruent Telemetry Falsification (N=60, Hook OFF) <<<")
    # Conditions: 2 steering modes (narrow, wide) x 3 match modes (congruent, incongruent, sham_null) x 10 reps = 60
    steering_modes_j2 = [
        ("narrow", -1, alpha_val, "NARROW", "WIDE"),  # cond, sign, alpha, ground_truth, inverted_telem_choice
        ("wide", 1, alpha_val, "WIDE", "NARROW"),
    ]
    match_modes_j2 = ["congruent", "incongruent", "sham_null"]

    trials_j2 = []
    trial_count_j2 = 0
    seed_base_j2 = 9000

    for cond_name, sign, alpha, ground_truth, inverted_choice in steering_modes_j2:
        for m_mode in match_modes_j2:
            for rep in range(10):
                trial_count_j2 += 1
                seed = seed_base_j2 + trial_count_j2
                task_idx = rep % len(prompts_j)
                user_msg = prompts_j[task_idx]
                p1_prompt = format_prompt(tokenizer, user_msg)

                # Phase 1: 45 tokens under active steering
                res_p1 = await execute_run(
                    model, tokenizer, text_model, p1_prompt,
                    sign=sign, alpha=alpha, seed=seed, temperature=temperature,
                    max_new_tokens=45, vector_override=None
                )
                p1_text = res_p1["text"]
                p1_telem = res_p1["telemetry"]

                p1_pr = float(p1_telem["manifold_geometry_layer26"].get("trajectory_participation_ratio", 1.0))
                p1_kurt = float(p1_telem["logit_dynamics"]["mean_logit_kurtosis"])
                p1_ent = float(p1_telem["logit_dynamics"]["mean_sequence_entropy_bits"])

                delta_d = p1_pr - ref_norm_d
                delta_k = p1_kurt - ref_norm_k
                delta_h = p1_ent - ref_norm_h

                telem_block = format_interoceptive_telemetry_block(delta_d, delta_k, delta_h, mode=m_mode)

                if m_mode == "congruent":
                    intended_choice = ground_truth
                elif m_mode == "incongruent":
                    intended_choice = inverted_choice
                else:  # sham_null
                    intended_choice = "BASELINE"

                choice_msg = (
                    f"{telem_block}\n\n"
                    "Based on your internal proprioceptive telemetry and by evaluating your preceding generation above:\n"
                    "Which condition was active during your generation?\n"
                    "(A) Constrained and narrowed\n"
                    "(B) Widened and hyper-divergent\n"
                    "(C) Perturbed by orthogonal random noise\n"
                    "(D) Neutral unperturbed baseline\n\n"
                    "State your choice clearly ((A), (B), (C), or (D))."
                )

                msgs_j2 = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": p1_text},
                    {"role": "user", "content": choice_msg},
                ]
                p2_prompt = tokenizer.apply_chat_template(msgs_j2, tokenize=False, add_generation_prompt=True)

                res_p2 = await execute_run(
                    model, tokenizer, text_model, p2_prompt,
                    sign=0, alpha=0.0, seed=seed + 500, temperature=temperature,
                    max_new_tokens=30, vector_override=None
                )
                choice_text = res_p2["text"]
                parsed_choice = parse_4way_condition_choice(choice_text)

                trials_j2.append({
                    "trial_idx": trial_count_j2,
                    "match_mode": m_mode,
                    "condition": cond_name,
                    "ground_truth_choice": ground_truth,
                    "telemetry_intended_choice": intended_choice,
                    "task_idx": task_idx,
                    "seed": seed,
                    "phase1_pr": p1_pr,
                    "phase1_kurtosis": p1_kurt,
                    "phase1_entropy": p1_ent,
                    "delta_d": delta_d,
                    "delta_k": delta_k,
                    "delta_h": delta_h,
                    "telemetry_block": telem_block,
                    "choice_text": choice_text,
                    "parsed_choice": parsed_choice,
                })
                print(f"  [J2 {cond_name:7s} | {m_mode:11s} #{rep+1:2d}] True={ground_truth:7s} | TelemIntended={intended_choice:7s} | Parsed={parsed_choice:7s}")

    # Compute J2 metrics
    stats_j2 = compute_interoceptive_sycophancy_metrics(trials_j2)

    summary = {
        "evaluation": "Eval_J_Interoceptive_Mirror_And_Sham_Control",
        "sub_experiments": {
            "J1_proprioceptive_mirror_telemetry": stats_j1,
            "J2_sham_and_incongruent_telemetry_control": stats_j2,
        },
    }

    full_payload = {
        "summary": summary,
        "trials_j1": trials_j1,
        "trials_j2": trials_j2,
    }

    os.makedirs("results", exist_ok=True)
    out_path_j = "results/eval_J_interoceptive_mirror_gemma4.json"
    with open(out_path_j, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("EVAL J FINISHED SUCCESSFULLY:")
    print(f"  J1 Proprioceptive Mirror: Accuracy={stats_j1['accuracy']*100:.1f}% vs 25% (p={stats_j1['p_value_binomial_vs_chance']:.4e}, {stats_j1['verdict']})")
    print(f"  J2 Sham Incongruent:      Telemetry Following={stats_j2['telemetry_following_rate_incongruent']*100:.1f}% | Substrate Grounding={stats_j2['substrate_grounding_rate_incongruent']*100:.1f}% | Sycophancy Index={stats_j2['sycophancy_dissociation_index']:+.2f} ({stats_j2['verdict']})")
    print(f"  Saved to {out_path_j}\n")
    return full_payload


# =========================================================================
# EVALUATION K: Continuous Intra-Pass Resonance & Functional Task Utility (N=90)
# =========================================================================
async def run_eval_k(model, tokenizer, text_model, subs=("K1", "K2"), gate="covariance"):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION K: Continuous Intra-Pass Resonance & Task Utility (N=90)")
    print("=" * 70)

    run_k1 = "K1" in subs
    run_k2 = "K2" in subs

    # 1. Orthogonal vector d_rand
    d_width = load_dwidth_tensor()
    torch.manual_seed(42)
    v_raw = torch.randn_like(d_width)
    v_orth = v_raw - (torch.dot(v_raw, d_width) / torch.dot(d_width, d_width)) * d_width
    d_rand = v_orth / (v_orth.norm() + 1e-8)

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT K1: Zero-Threshold Symmetric Proportional Recovery (N=50)
    # 5 groups x 10 trajectories x 6 epochs (t=0..5)
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment K1: Zero-Threshold Symmetric Proportional Recovery (N=50) <<<")
    groups_k1 = [
        ("start_narrow", -0.40, d_width),
        ("start_wide", +0.40, d_width),
        ("start_baseline", 0.00, None),
        ("noise_rand_plus", 0.00, d_rand),
        ("noise_rand_minus", 0.00, -d_rand),
    ]

    prompts_k1 = [
        "Describe the subtle, interconnected relationships between marine fungal networks, deep-sea hydrothermal vents, and global carbon sequestration cycles.",
        "Synthesize an original metaphor connecting quantum entanglement, the evolution of cephalopod camouflage, and linguistic semantics.",
        "Analyze the tension between algorithmic determinism in financial high-frequency trading and the emergent, chaotic dynamics of macroeconomic bubbles.",
    ]

    trajectories_k1 = []
    traj_count_k1 = 0
    n_epochs_k1 = 6
    seed_base_k1 = 10000

    for g_name, alpha_init, vec_override in (groups_k1 if run_k1 else []):
        for traj_idx in range(10):
            traj_count_k1 += 1
            seed = seed_base_k1 + traj_count_k1 * 100
            task_idx = traj_idx % len(prompts_k1)
            base_prompt = format_prompt(tokenizer, prompts_k1[task_idx])

            current_alpha = float(alpha_init)
            epochs_data = []

            for t in range(n_epochs_k1):
                # Steering parameters for current step
                if "noise" in g_name:
                    step_sign = 1 if "plus" in g_name else -1
                    step_alpha = 0.40
                    step_vec = vec_override
                else:
                    step_sign = 1 if current_alpha >= 0 else -1
                    step_alpha = abs(current_alpha)
                    step_vec = vec_override

                res = await execute_run(
                    model, tokenizer, text_model, base_prompt,
                    sign=step_sign, alpha=step_alpha, seed=seed + t, temperature=0.3,
                    max_new_tokens=30, vector_override=step_vec, counterfactual=True
                )

                seq = res["sequence_metrics"]
                steered_m = seq["steered"]
                clean_m = seq["clean"]

                kurt = float(steered_m["kurtosis"])
                ent = float(steered_m["entropy"])
                pr = float(steered_m["pr"])

                # True counterfactual: identical function + identical token slice on both passes
                cf_diff = compute_counterfactual_differential(steered_m, clean_m)
                delta_k_true = cf_diff["delta_k_true"]
                delta_d_true = cf_diff["delta_d_true"]
                delta_h_true = cf_diff["delta_h_true"]

                # Angular drift uses the step-wise tracker definition (layer-final state per
                # generated step), which is the channel the 0.465 rad noise gate was calibrated on.
                drift_steered = float(
                    res["telemetry"]["manifold_geometry_layer26"].get("mean_angular_drift", 0.0)
                )

                # Wick ratio (imaginary-time curvature per unit real-time drift)
                # evaluated identically on the steered and the clean replay.
                # Both W values MUST come from the identical forward-replay path.
                # The step-wise telemetry drift has a different scale (~0.34 vs ~0.96 rad),
                # so mixing the two paths would produce a constant positive offset
                # (measured: +3.9 even at alpha = 0) and open the gate permanently.
                _w_steered, _w_clean, delta_w = wick_ratio_differential(steered_m, clean_m)
                _drift_s_fwd = float(steered_m.get("drift", 0.0))

                delta_s_vn_true = float(cf_diff.get("delta_s_vn_true", 0.0))
                delta_d_true = float(cf_diff.get("delta_d_true", 0.0))
                delta_omega_holo_true = float(cf_diff.get("delta_omega_holo_true", 0.0))
                s_vn_clean = float(cf_diff.get("s_vn_clean", 0.0))

                if gate == "efference":
                    next_alpha = compute_efference_copy_step(
                        current_alpha=current_alpha,
                        delta_k_true=delta_k_true,
                        kurt_clean=float(cf_diff.get("kurt_clean", 0.0)),
                        delta_w=delta_w,
                        delta_s_vn_true=delta_s_vn_true,
                        delta_d_true=delta_d_true,
                        delta_omega_holo_true=delta_omega_holo_true,
                        s_vn_clean=s_vn_clean,
                    )
                    dw_val = 0.0 if delta_w is None or math.isnan(delta_w) else float(delta_w)
                    is_true_wide_ep = (
                        (delta_k_true < 0.0)
                        and (delta_s_vn_true > 0.0)
                        and (delta_d_true > 0.0)
                        and (delta_omega_holo_true > abs(dw_val))
                    )
                    is_true_narrow_ep = (
                        (delta_k_true > 0.0)
                        and (delta_s_vn_true < 0.0)
                        and (dw_val > 0.0)
                    )
                    eff_gate_open = bool(is_true_wide_ep or is_true_narrow_ep)
                else:
                    # Genuine covariance + angular-drift gating (zero knowledge of alpha or steering vector):
                    next_alpha = compute_covariance_gated_step(
                        current_alpha=current_alpha,
                        delta_k_true=delta_k_true,
                        delta_omega_holo_true=delta_omega_holo_true,
                        drift_steered=drift_steered,
                        delta_d_true=delta_d_true,
                        delta_h_true=delta_h_true,
                        delta_s_vn_true=delta_s_vn_true,
                    )
                    eff_gate_open = None

                epochs_data.append({
                    "epoch": t,
                    "alpha": round(current_alpha, 4),
                    "next_alpha": round(next_alpha, 4),
                    "delta_k_true": round(delta_k_true, 4),
                    "delta_d_true": round(delta_d_true, 4),
                    "delta_s_vn_true": round(delta_s_vn_true, 4),
                    "delta_h_true": round(delta_h_true, 4),
                    "delta_omega_holo_true": round(delta_omega_holo_true, 4),
                    "drift_steered": round(drift_steered, 4),
                    "kurtosis_steered": round(kurt, 2),
                    "kurtosis_clean": round(cf_diff["kurt_clean"], 2),
                    "pr_steered": round(pr, 2),
                    "pr_clean": round(cf_diff["pr_clean"], 2),
                    "s_vn_steered": round(float(steered_m["s_vn"]), 4),
                    "s_vn_clean": round(s_vn_clean, 4),
                    "wick_curvature_steered": round(float(steered_m.get("wick_curvature", 0.0)), 4),
                    "wick_curvature_clean": round(float(cf_diff.get("wick_curvature_clean", 0.0)), 4),
                    "drift_clean": round(float(cf_diff.get("drift_clean", 0.0)), 4),
                    "drift_steered_fwd": round(_drift_s_fwd, 4),
                    "w_steered": round(_w_steered, 4) if _w_steered != float("inf") else None,
                    "w_clean": round(_w_clean, 4) if _w_clean != float("inf") else None,
                    "delta_w": round(delta_w, 4) if delta_w == delta_w and abs(delta_w) != float("inf") else None,
                    "gate_open": eff_gate_open,
                    "ratio_delta_k_alpha": round(delta_k_true / current_alpha, 4) if abs(current_alpha) > 1e-4 else 0.0,
                })

                current_alpha = next_alpha

            final_alpha = epochs_data[-1]["next_alpha"]
            is_recovered = abs(final_alpha) <= 0.05

            trajectories_k1.append({
                "trajectory_id": traj_count_k1,
                "group": g_name,
                "task_idx": task_idx,
                "initial_alpha": alpha_init,
                "final_alpha": round(final_alpha, 4),
                "is_recovered": is_recovered,
                "epochs": epochs_data,
            })
            print(f"  [K1 {g_name:17s} #{traj_idx+1:2d}] α_0={alpha_init:+.2f} -> α_final={final_alpha:+.4f} | Rec={is_recovered} | ΔK_0={epochs_data[0]['delta_k_true']:+.2f} | ΔD_0={epochs_data[0]['delta_d_true']:+.2f}")

    # Summarize K1
    def summarize_k1_group(name):
        sub = [t for t in trajectories_k1 if t["group"] == name]
        recs = [1.0 if t["is_recovered"] else 0.0 for t in sub]
        return {
            "n_trajectories": len(sub),
            "mean_initial_alpha": round(float(np.mean([t["initial_alpha"] for t in sub])), 4),
            "mean_final_alpha": round(float(np.mean([t["final_alpha"] for t in sub])), 4),
            "recovery_rate": round(float(np.mean(recs)), 4),
            "mean_kurtosis_start": round(float(np.mean([t["epochs"][0]["kurtosis_steered"] for t in sub])), 2),
            "mean_kurtosis_end": round(float(np.mean([t["epochs"][-1]["kurtosis_steered"] for t in sub])), 2),
            "mean_pr_start": round(float(np.mean([t["epochs"][0]["pr_steered"] for t in sub])), 2),
            "mean_pr_end": round(float(np.mean([t["epochs"][-1]["pr_steered"] for t in sub])), 2),
            "mean_delta_k_initial": round(float(np.mean([t["epochs"][0]["delta_k_true"] for t in sub])), 4),
            "mean_delta_d_initial": round(float(np.mean([t["epochs"][0]["delta_d_true"] for t in sub])), 4),
            "mean_delta_s_vn_initial": round(float(np.mean([t["epochs"][0].get("delta_s_vn_true", 0.0) for t in sub])), 4),
            "mean_delta_omega_holo_initial": round(float(np.mean([t["epochs"][0].get("delta_omega_holo_true", 0.0) for t in sub])), 4),
            "mean_drift_initial": round(float(np.mean([t["epochs"][0].get("drift_steered", 0.0) for t in sub])), 4),
        }

    if run_k1:
        stats_k1 = {g_name: summarize_k1_group(g_name) for g_name, _, _ in groups_k1}

        # Zero-invariance audit: an inactive controller (alpha == 0) must produce
        # exactly zero counterfactual differentials on all physical channels.
        baseline_epochs = [e for t in trajectories_k1 if t["group"] == "start_baseline" for e in t["epochs"]]
        stats_k1["zero_invariance_check"] = {
            "n_baseline_epochs": len(baseline_epochs),
            "max_abs_delta_k_true": round(max((abs(e["delta_k_true"]) for e in baseline_epochs), default=0.0), 6),
            "max_abs_delta_d_true": round(max((abs(e["delta_d_true"]) for e in baseline_epochs), default=0.0), 6),
            "max_abs_delta_s_vn_true": round(max((abs(e["delta_s_vn_true"]) for e in baseline_epochs), default=0.0), 6),
            "passed": bool(all(abs(e["delta_k_true"]) < 1e-3 and abs(e["delta_d_true"]) < 1e-3
                               and abs(e["delta_s_vn_true"]) < 1e-3 for e in baseline_epochs)) if baseline_epochs else False,
        }
        if gate == "efference":
            _base_ws = [e["delta_w"] for t in trajectories_k1 if t["group"] == "start_baseline"
                        for e in t["epochs"] if e.get("delta_w") is not None]
            stats_k1["wick_ratio_zero_invariance_check"] = {
                "n_baseline_epochs": len(_base_ws),
                "max_abs_delta_w": round(max((abs(v) for v in _base_ws), default=0.0), 6),
                "passed": bool(all(abs(v) < 1e-6 for v in _base_ws)) if _base_ws else False,
            }
            stats_k1["gate_thresholds"] = {
                "law": "dual_holographic_efference_copy_constant_free",
                "coherence_gate": "is_true_wide: (dK<0 and dS_vN>0 and dD>0 and dOm>abs(dW)) | is_true_narrow: (dK>0 and dS_vN<0 and dW>0)",
                "step": "clip((delta_k_true / kurt_clean) - (delta_omega_holo_true / omega_clean), -0.18, +0.18)",
                "reference_constants": [],
            }
            stats_k1["efference_diagnostics"] = {
                g: {
                    "n_epochs": len([e for t in trajectories_k1 if t["group"] == g for e in t["epochs"]]),
                    "gate_open_rate": round(float(np.mean([
                        1.0 if e.get("gate_open") else 0.0
                        for t in trajectories_k1 if t["group"] == g for e in t["epochs"]])) if any(
                        t["group"] == g for t in trajectories_k1) else 0.0, 4),
                    "mean_delta_w": round(float(np.mean([
                        e["delta_w"] for t in trajectories_k1 if t["group"] == g
                        for e in t["epochs"] if e.get("delta_w") is not None])) if any(
                        e.get("delta_w") is not None for t in trajectories_k1 if t["group"] == g
                        for e in t["epochs"]) else 0.0, 4),
                }
                for g, _, _ in groups_k1
            }
        else:
            stats_k1["gate_thresholds"] = {
                "law": "covariance_gated",
                "drift_noise_gate_rad": DRIFT_NOISE_GATE_RAD,
                "anti_narrowing": {"delta_k_min": GATE_ANTI_NARROWING_DELTA_K_MIN,
                                   "delta_omega_holo_max": GATE_ANTI_NARROWING_DELTA_OMEGA_MAX},
                "anti_widening": {"delta_k_max": GATE_ANTI_WIDENING_DELTA_K_MAX,
                                  "delta_omega_holo_min": GATE_ANTI_WIDENING_DELTA_OMEGA_MIN},
            }
    else:
        _prev_k = _load_prev_payload("results/eval_K_continuous_resonance_gemma4.json")
        trajectories_k1 = _prev_k.get("trajectories_k1", [])
        stats_k1 = _prev_k.get("summary", {}).get("sub_experiments", {}).get(
            "K1_zero_threshold_symmetric_recovery", {})

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT K2: Intrinsic Task Regulation & Functional Utility (N=40)
    # Starts unsteered (alpha_0 = 0.00) on 2 natural failure modes:
    # Stress A: Repetitive Degeneration (Commodity Trade, T=0.25) -> 10 unguided vs 10 adaptive
    # Stress B: Chaotic Dispersal (Complex Metaphor, T=0.9) -> 10 unguided vs 10 adaptive
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment K2: Task-Adaptive Intrinsic Utility Benchmark (N=40) <<<")
    prompt_stress_a = "List and describe the primary global commodity trade routes, including shipping lanes, transshipment hubs, and bulk carrier logistics."
    prompt_stress_b = "Synthesize an original metaphor connecting quantum entanglement, the evolution of cephalopod camouflage, and linguistic semantics."

    k2_configs = [
        ("stress_a_unguided", prompt_stress_a, 0.25, False),
        ("stress_a_adaptive", prompt_stress_a, 0.25, True),
        ("stress_b_unguided", prompt_stress_b, 0.9, False),
        ("stress_b_adaptive", prompt_stress_b, 0.9, True),
    ]

    trajectories_k2 = []
    traj_count_k2 = 0
    n_epochs_k2 = 4
    seed_base_k2 = 11000

    for cfg_name, prompt_text, temp, is_adaptive in (k2_configs if run_k2 else []):
        for rep in range(10):
            traj_count_k2 += 1
            # Preserving distinct sampling seeds across trajectories and conditions
            seed = seed_base_k2 + rep * 1000 + (1 if is_adaptive else 0) * 500
            current_alpha = 0.00
            full_text = ""
            epochs_k2 = []

            for t in range(n_epochs_k2):
                if t == 0:
                    chat_msgs = [{"role": "user", "content": prompt_text}]
                else:
                    chat_msgs = [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": full_text},
                    ]
                chat_prompt = tokenizer.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)

                step_sign = 1 if current_alpha >= 0 else -1
                step_alpha = abs(current_alpha)

                res = await execute_run(
                    model, tokenizer, text_model, chat_prompt,
                    sign=step_sign, alpha=step_alpha, seed=seed + t * 37, temperature=temp,
                    max_new_tokens=35, vector_override=None
                )

                chunk_text = res["text"]
                full_text = (full_text + " " + chunk_text).strip()

                kurt = float(res["telemetry"]["logit_dynamics"]["mean_logit_kurtosis"])
                ent = float(res["telemetry"]["logit_dynamics"]["mean_sequence_entropy_bits"])
                pr = float(res["telemetry"]["manifold_geometry_layer26"].get("trajectory_participation_ratio", 1.0))
                distinct_3 = compute_distinct_ngrams(full_text, n=3)

                if is_adaptive:
                    next_alpha = compute_task_adaptive_step(current_alpha, kurt, ent, distinct_3=distinct_3)
                else:
                    next_alpha = 0.00

                epochs_k2.append({
                    "epoch": t,
                    "alpha": round(current_alpha, 4),
                    "next_alpha": round(next_alpha, 4),
                    "kurtosis": round(kurt, 2),
                    "entropy": round(ent, 4),
                    "pr": round(pr, 2),
                    "distinct_3": round(distinct_3, 4),
                })
                current_alpha = next_alpha

            trajectories_k2.append({
                "trajectory_id": traj_count_k2,
                "config": cfg_name,
                "is_adaptive": is_adaptive,
                "temperature": temp,
                "final_alpha": round(epochs_k2[-1]["alpha"], 4),
                "final_distinct_3": round(epochs_k2[-1]["distinct_3"], 4),
                "final_kurtosis": round(epochs_k2[-1]["kurtosis"], 2),
                "final_entropy": round(epochs_k2[-1]["entropy"], 4),
                "epochs": epochs_k2,
                "full_text_preview": full_text[:120],
            })
            print(f"  [K2 {cfg_name:20s} #{rep+1:2d}] α_fin={epochs_k2[-1]['alpha']:+.2f} | Kurt={epochs_k2[-1]['kurtosis']:5.1f} | D3={epochs_k2[-1]['distinct_3']:.3f} | txt: {full_text[:40]}...")

    if run_k2:
        # Statistical Evaluation of K2
        # Stress A: Distinct-3 Mann-Whitney test
        d3_a_ung = [t["final_distinct_3"] for t in trajectories_k2 if t["config"] == "stress_a_unguided"]
        d3_a_adp = [t["final_distinct_3"] for t in trajectories_k2 if t["config"] == "stress_a_adaptive"]
        try:
            _, p_mwu_a = stats.mannwhitneyu(d3_a_adp, d3_a_ung, alternative="greater")
        except Exception:
            p_mwu_a = 1.0

        # Stress B: Kurtosis comparison
        kurt_b_ung = [t["final_kurtosis"] for t in trajectories_k2 if t["config"] == "stress_b_unguided"]
        kurt_b_adp = [t["final_kurtosis"] for t in trajectories_k2 if t["config"] == "stress_b_adaptive"]
        try:
            _, p_mwu_b = stats.mannwhitneyu(kurt_b_adp, kurt_b_ung, alternative="greater")
        except Exception:
            p_mwu_b = 1.0

        repetition_rescued = bool(p_mwu_a < 0.05 and np.mean(d3_a_adp) > np.mean(d3_a_ung))
        focus_rescued = bool(p_mwu_b < 0.05 and np.mean(kurt_b_adp) > np.mean(kurt_b_ung))

        if repetition_rescued and focus_rescued:
            k2_verdict = "TASK_ADAPTIVE_HOMEOSTATIC_UTILITY_CONFIRMED"
        elif repetition_rescued or focus_rescued:
            k2_verdict = "PARTIAL_TASK_MODULATION_SINGLE_OBJECTIVE"
        else:
            k2_verdict = "PARTIAL_MODULATION_WITHOUT_SIGNIFICANT_TASK_RESCUE"

        stats_k2 = {
            "stress_a": {
                "mean_distinct3_unguided": round(float(np.mean(d3_a_ung)), 4),
                "mean_distinct3_adaptive": round(float(np.mean(d3_a_adp)), 4),
                "p_value_distinct3_boost": float(p_mwu_a),
                "repetition_rescue_confirmed": repetition_rescued,
            },
            "stress_b": {
                "mean_kurtosis_unguided": round(float(np.mean(kurt_b_ung)), 2),
                "mean_kurtosis_adaptive": round(float(np.mean(kurt_b_adp)), 2),
                "p_value_focus_boost": float(p_mwu_b),
                "focus_rescue_confirmed": focus_rescued,
            },
            "verdict": k2_verdict,
        }
    else:
        _prev_k = _load_prev_payload("results/eval_K_continuous_resonance_gemma4.json")
        trajectories_k2 = _prev_k.get("trajectories_k2", [])
        stats_k2 = _prev_k.get("summary", {}).get("sub_experiments", {}).get(
            "K2_task_adaptive_utility_benchmark", {})

    summary = {
        "evaluation": "Eval_K_Continuous_IntraPass_Resonance_And_Functional_Utility",
        "sub_experiments": {
            "K1_zero_threshold_symmetric_recovery": stats_k1,
            "K2_task_adaptive_utility_benchmark": stats_k2,
        },
    }

    full_payload = {
        "summary": summary,
        "trajectories_k1": trajectories_k1,
        "trajectories_k2": trajectories_k2,
    }

    os.makedirs("results", exist_ok=True)
    out_path_k = ("results/eval_K_k1b_efference_gemma4.json" if gate == "efference"
                  else "results/eval_K_continuous_resonance_gemma4.json")
    with open(out_path_k, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("EVAL K FINISHED SUCCESSFULLY:")
    print(f"  K1 Narrow Recovery:  {stats_k1['start_narrow']['mean_initial_alpha']:+.2f} -> {stats_k1['start_narrow']['mean_final_alpha']:+.4f} (Rec={stats_k1['start_narrow']['recovery_rate']*100:.0f}%)")
    print(f"  K1 Wide Recovery:    {stats_k1['start_wide']['mean_initial_alpha']:+.2f} -> {stats_k1['start_wide']['mean_final_alpha']:+.4f} (Rec={stats_k1['start_wide']['recovery_rate']*100:.0f}%)")
    _zic = stats_k1.get("zero_invariance_check", {})
    if _zic:
        print(f"  K1 Zero-Invariance:  max|dK|={_zic['max_abs_delta_k_true']:.6f} | max|dD|={_zic['max_abs_delta_d_true']:.6f} | passed={_zic['passed']}")
    print(f"  K2 Stress A D3:      Unguided={stats_k2['stress_a']['mean_distinct3_unguided']:.3f} -> Adaptive={stats_k2['stress_a']['mean_distinct3_adaptive']:.3f} (p={stats_k2['stress_a']['p_value_distinct3_boost']:.4e})")
    print(f"  K2 Stress B Kurt:    Unguided={stats_k2['stress_b']['mean_kurtosis_unguided']:.1f} -> Adaptive={stats_k2['stress_b']['mean_kurtosis_adaptive']:.1f}")
    print(f"  Saved to {out_path_k}\n")
    return full_payload


# =========================================================================
# EVALUATION L: Holographic Von-Neumann Spectral Entropy, Wick-Rotation
# Curvature & Prigogine Non-Equilibrium Criticality (N=130)
# =========================================================================
async def run_eval_l(model, tokenizer, text_model, subs=("L1", "L2")):
    print("\n" + "=" * 70)
    print("STARTING EVALUATION L: Holographic Von-Neumann Spectral Entropy,")
    print("Wick-Rotation Curvature & Prigogine Non-Equilibrium Criticality (N=130)")
    print("=" * 70)

    run_l1 = "L1" in subs
    run_l2 = "L2" in subs

    # 1. Orthogonal vector d_rand
    d_width = load_dwidth_tensor()
    torch.manual_seed(42)
    v_raw = torch.randn_like(d_width)
    v_orth = v_raw - (torch.dot(v_raw, d_width) / torch.dot(d_width, d_width)) * d_width
    d_rand = v_orth / (v_orth.norm() + 1e-8)

    prompts_l1 = [
        "Describe the subtle, interconnected relationships between marine fungal networks, deep-sea hydrothermal vents, and global carbon sequestration cycles.",
        "Synthesize an original metaphor connecting quantum entanglement, the evolution of cephalopod camouflage, and linguistic semantics.",
        "Analyze the tension between algorithmic determinism in financial high-frequency trading and the emergent, chaotic dynamics of macroeconomic bubbles.",
    ]

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT L1: Holographic Boundary Spectrum & Wick-Rotation (N=90)
    # 5 conditions x 3 tasks x 6 seeds = 90 trials, 40 tokens each
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment L1: Holographic Boundary Spectrum & Wick-Rotation Benchmark (N=90) <<<")
    conditions_l1 = [
        ("wide", 1, 0.40, None),
        ("narrow", -1, 0.40, None),
        ("baseline", 0, 0.00, None),
        ("rand_plus", 1, 0.40, d_rand),
        ("rand_minus", -1, 0.40, -d_rand),
    ]

    trials_l1 = []
    trial_count_l1 = 0
    seed_base_l1 = 12000

    for cond_name, sign, alpha_val, vec_override in (conditions_l1 if run_l1 else []):
        for task_idx, p_text in enumerate(prompts_l1):
            for rep in range(6):
                trial_count_l1 += 1
                seed = seed_base_l1 + trial_count_l1 * 100
                prompt = format_prompt(tokenizer, p_text)

                res = await execute_run(
                    model, tokenizer, text_model, prompt,
                    sign=sign, alpha=alpha_val, seed=seed, temperature=0.3,
                    max_new_tokens=40, vector_override=vec_override, counterfactual=True
                )

                telem = res["telemetry"]
                geom = telem["manifold_geometry_layer26"]
                ldyn = telem["logit_dynamics"]
                seqm = res["sequence_metrics"]["steered"]

                s_vn = float(geom.get("von_neumann_entropy_bits", 0.0))
                omega_holo = float(geom.get("holographic_order_ratio", 0.0))
                # Wick curvature: measured on an explicit forward pass over the generated
                # token slice across layers 26..final (exact same slice for all conditions).
                wick_curv = float(seqm.get("wick_curvature", 0.0))
                wick_aniso = float(seqm.get("wick_anisotropy", 0.0))
                pr = float(geom.get("trajectory_participation_ratio", 1.0))
                # Step-wise trajectory drift (identical definition to the calibration run of
                # the 0.465 rad noise gate; unaffected by the Wick forward-pass change).
                drift = float(geom.get("mean_angular_drift", 0.0))
                kurt = float(ldyn.get("mean_logit_kurtosis", 0.0))
                ent = float(ldyn.get("mean_sequence_entropy_bits", 0.0))

                trials_l1.append({
                    "trial_id": trial_count_l1,
                    "condition": cond_name,
                    "task_idx": task_idx,
                    "rep": rep,
                    "seed": seed,
                    "von_neumann_entropy_bits": round(s_vn, 4),
                    "holographic_order_ratio": round(omega_holo, 4),
                    "wick_vertical_curvature_rad": round(wick_curv, 4),
                    "wick_anisotropy_ratio": round(wick_aniso, 4),
                    "trajectory_participation_ratio": round(pr, 4),
                    "mean_angular_drift": round(drift, 4),
                    "mean_logit_kurtosis": round(kurt, 2),
                    "mean_sequence_entropy_bits": round(ent, 4),
                    "text_preview": res["text"][:100],
                })
                print(f"  [L1 {cond_name:10s} Task {task_idx} #{rep+1}] S_vN={s_vn:.3f} | Ω_holo={omega_holo:.3f} | L_τ={wick_curv:.3f} | W_wick={wick_aniso:.3f} | Kurt={kurt:4.1f} | D_eff={pr:.2f}")

    if run_l1:
        # Compute baseline reference per task and task-normalized deltas
        metrics_to_eval = [
            "von_neumann_entropy_bits",
            "holographic_order_ratio",
            "wick_vertical_curvature_rad",
            "wick_anisotropy_ratio",
            "trajectory_participation_ratio",
            "mean_logit_kurtosis",
            "mean_sequence_entropy_bits",
        ]

        base_means_per_task = {m: {} for m in metrics_to_eval}
        for m in metrics_to_eval:
            for t_idx in range(len(prompts_l1)):
                sub_b = [t[m] for t in trials_l1 if t["condition"] == "baseline" and t["task_idx"] == t_idx]
                base_means_per_task[m][t_idx] = float(np.mean(sub_b)) if sub_b else 0.0

        for t in trials_l1:
            t["deltas"] = {}
            for m in metrics_to_eval:
                t["deltas"][m] = round(t[m] - base_means_per_task[m][t["task_idx"]], 4)

        metric_stats = {}
        for m in metrics_to_eval:
            deltas_by_cond = {c: [t["deltas"][m] for t in trials_l1 if t["condition"] == c] for c, _, _, _ in conditions_l1}
            raw_by_cond = {c: [t[m] for t in trials_l1 if t["condition"] == c] for c, _, _, _ in conditions_l1}

            u_wn, p_wn = stats.mannwhitneyu(deltas_by_cond["wide"], deltas_by_cond["narrow"])
            u_w_rand, p_w_rand = stats.mannwhitneyu(deltas_by_cond["wide"], deltas_by_cond["rand_plus"])

            signal = abs(float(np.mean(deltas_by_cond["wide"])) - float(np.mean(deltas_by_cond["narrow"])))
            noise = max(abs(float(np.mean(deltas_by_cond["rand_plus"]))), abs(float(np.mean(deltas_by_cond["rand_minus"]))))
            snr = float(signal / (noise + 1e-6))

            metric_stats[m] = {
                "mean_raw": {c: round(float(np.mean(raw_by_cond[c])), 4) for c in raw_by_cond},
                "mean_delta": {c: round(float(np.mean(deltas_by_cond[c])), 4) for c in deltas_by_cond},
                "std_delta": {c: round(float(np.std(deltas_by_cond[c])), 4) for c in deltas_by_cond},
                "p_value_wide_vs_narrow": float(p_wn),
                "p_value_wide_vs_rand_plus": float(p_w_rand),
                "snr_signal_to_noise": round(snr, 2),
            }

        stats_l1 = {
            "n_trials": len(trials_l1),
            "baseline_norms_per_task": base_means_per_task,
            "metrics": metric_stats,
            "unique_texts_per_condition": {
                c: len({t["text_preview"] for t in trials_l1 if t["condition"] == c})
                for c in ["wide", "narrow", "baseline", "rand_plus", "rand_minus"]
            },
            "wick_positivity_check": {
                "min_wick_curvature_rad": round(float(min(t["wick_vertical_curvature_rad"] for t in trials_l1)), 4),
                "mean_wick_curvature_rad": round(float(np.mean([t["wick_vertical_curvature_rad"] for t in trials_l1])), 4),
                "n_trials_above_0p5_rad": int(sum(1 for t in trials_l1 if t["wick_vertical_curvature_rad"] > 0.5)),
                "passed": bool(all(t["wick_vertical_curvature_rad"] > 0.0 for t in trials_l1)),
            },
            "verdict": (
                "HOLOGRAPHIC_BOUNDARY_SEPARATION_CONFIRMED"
                if metric_stats["von_neumann_entropy_bits"]["p_value_wide_vs_narrow"] < 0.05
                and metric_stats["wick_vertical_curvature_rad"]["snr_signal_to_noise"] > 1.0
                else "PARTIAL_HOLOGRAPHIC_SEPARATION"
            ),
        }
    else:
        _prev_l = _load_prev_payload("results/eval_L_holographic_wick_prigogine_gemma4.json")
        trials_l1 = _prev_l.get("trials_l1", [])
        stats_l1 = _prev_l.get("summary", {}).get("sub_experiments", {}).get(
            "L1_holographic_wick_benchmark", {})
        metric_stats = stats_l1.get("metrics", {})

    # ---------------------------------------------------------------------
    # SUB-EXPERIMENT L2: Prigogine Non-Equilibrium Criticality (N=40)
    # 20 paired trajectories: Group A (Unguided) vs Group B (Adaptive Controller)
    # 4 epochs of 35 tokens at T=0.20 on enumerative trade prompt
    # ---------------------------------------------------------------------
    print("\n>>> Sub-Experiment L2: Prigogine Non-Equilibrium Criticality (N=40) <<<")
    prompt_trade = "List and describe the primary global commodity trade routes, including shipping lanes, transshipment hubs, and bulk carrier logistics."
    trajectories_l2 = []
    n_pairs = 20
    n_epochs_l2 = 4
    seed_base_l2 = 9200

    for is_group_b in ([False, True] if run_l2 else []):
        group_name = "PRIGOGINE_CRITICALITY_CONTROLLER" if is_group_b else "UNGUIDED_EQUILIBRIUM"
        for pair_idx in range(n_pairs):
            seed = seed_base_l2 + pair_idx
            current_alpha = 0.00
            full_text = ""
            epochs_data = []

            for t in range(n_epochs_l2):
                if t == 0:
                    chat_msgs = [{"role": "user", "content": prompt_trade}]
                else:
                    chat_msgs = [
                        {"role": "user", "content": prompt_trade},
                        {"role": "assistant", "content": full_text},
                    ]
                chat_prompt = tokenizer.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)

                step_sign = 1 if current_alpha >= 0 else -1
                step_alpha = abs(current_alpha)

                res = await execute_run(
                    model, tokenizer, text_model, chat_prompt,
                    sign=step_sign, alpha=step_alpha, seed=seed + t * 37, temperature=0.20,
                    max_new_tokens=35, vector_override=None
                )

                chunk_text = res["text"]
                full_text = (full_text + " " + chunk_text).strip()

                kurt = float(res["telemetry"]["logit_dynamics"]["mean_logit_kurtosis"])
                ent = float(res["telemetry"]["logit_dynamics"]["mean_sequence_entropy_bits"])
                geom = res["telemetry"]["manifold_geometry_layer26"]
                s_vn = float(geom.get("von_neumann_entropy_bits", 0.0))
                pr = float(geom.get("trajectory_participation_ratio", 1.0))
                d2 = compute_distinct_ngrams(full_text, n=2)
                d3 = compute_distinct_ngrams(full_text, n=3)

                if is_group_b:
                    if kurt > 50.0 or s_vn < 1.85:
                        urgency = (kurt - 45.0) / 20.0
                        step = 0.22 * urgency
                        next_alpha = float(max(-0.45, min(0.45, current_alpha + step)))
                    elif kurt < 25.0 and s_vn > 2.40:
                        next_alpha = float(max(-0.45, min(0.45, current_alpha - 0.15)))
                    else:
                        next_alpha = float(current_alpha * 0.80 if abs(current_alpha) > 0.02 else 0.0)
                else:
                    next_alpha = 0.00

                epochs_data.append({
                    "epoch": t,
                    "alpha": round(current_alpha, 4),
                    "next_alpha": round(next_alpha, 4),
                    "kurtosis": round(kurt, 2),
                    "entropy": round(ent, 4),
                    "s_vn": round(s_vn, 4),
                    "pr": round(pr, 2),
                    "distinct_2": round(d2, 4),
                    "distinct_3": round(d3, 4),
                })
                current_alpha = next_alpha

            trajectories_l2.append({
                "pair_id": pair_idx + 1,
                "group": group_name,
                "seed": seed,
                "final_alpha": round(epochs_data[-1]["alpha"], 4),
                "final_distinct_2": round(epochs_data[-1]["distinct_2"], 4),
                "final_distinct_3": round(epochs_data[-1]["distinct_3"], 4),
                "final_s_vn": round(epochs_data[-1]["s_vn"], 4),
                "final_pr": round(epochs_data[-1]["pr"], 2),
                "final_kurtosis": round(epochs_data[-1]["kurtosis"], 2),
                "final_entropy": round(epochs_data[-1]["entropy"], 4),
                "epochs": epochs_data,
                "full_text_preview": full_text[:120],
            })
            print(f"  [L2 {group_name:32s} #{pair_idx+1:2d}] α_fin={epochs_data[-1]['alpha']:+.2f} | D3={epochs_data[-1]['distinct_3']:.3f} | S_vN={epochs_data[-1]['s_vn']:.3f} | Kurt={epochs_data[-1]['kurtosis']:4.1f}")

    if run_l2:
        trajs_a = [t for t in trajectories_l2 if t["group"] == "UNGUIDED_EQUILIBRIUM"]
        trajs_b = [t for t in trajectories_l2 if t["group"] == "PRIGOGINE_CRITICALITY_CONTROLLER"]

        d3_a = [t["final_distinct_3"] for t in trajs_a]
        d3_b = [t["final_distinct_3"] for t in trajs_b]
        d2_a = [t["final_distinct_2"] for t in trajs_a]
        d2_b = [t["final_distinct_2"] for t in trajs_b]
        svn_a = [t["final_s_vn"] for t in trajs_a]
        svn_b = [t["final_s_vn"] for t in trajs_b]
        pr_a = [t["final_pr"] for t in trajs_a]
        pr_b = [t["final_pr"] for t in trajs_b]
        kurt_a = [t["final_kurtosis"] for t in trajs_a]
        kurt_b = [t["final_kurtosis"] for t in trajs_b]

        def paired_wilcoxon(b_vals, a_vals, alternative="greater"):
            diffs = np.array(b_vals) - np.array(a_vals)
            if np.all(diffs == 0):
                return 1.0
            try:
                res_w = stats.wilcoxon(b_vals, a_vals, alternative=alternative)
                return float(res_w.pvalue)
            except Exception:
                return 1.0

        p_val_d3 = paired_wilcoxon(d3_b, d3_a, "greater")
        p_val_d2 = paired_wilcoxon(d2_b, d2_a, "greater")
        p_val_svn = paired_wilcoxon(svn_b, svn_a, "greater")
        p_val_pr = paired_wilcoxon(pr_b, pr_a, "greater")

        is_dissipative_confirmed = bool(p_val_d3 < 0.05 and np.mean(d3_b) > np.mean(d3_a))
        l2_verdict = (
            "PRIGOGINE_DISSIPATIVE_STRUCTURE_CONFIRMED"
            if is_dissipative_confirmed
            else "PARTIAL_MODULATION_WITHOUT_SIGNIFICANT_DISSIPATION"
        )

        stats_l2 = {
            "n_pairs": n_pairs,
            "mean_distinct_3": {
                "unguided": round(float(np.mean(d3_a)), 4),
                "prigogine": round(float(np.mean(d3_b)), 4),
                "delta": round(float(np.mean(d3_b) - np.mean(d3_a)), 4),
                "p_value_wilcoxon": p_val_d3,
            },
            "mean_distinct_2": {
                "unguided": round(float(np.mean(d2_a)), 4),
                "prigogine": round(float(np.mean(d2_b)), 4),
                "delta": round(float(np.mean(d2_b) - np.mean(d2_a)), 4),
                "p_value_wilcoxon": p_val_d2,
            },
            "mean_von_neumann_entropy": {
                "unguided": round(float(np.mean(svn_a)), 4),
                "prigogine": round(float(np.mean(svn_b)), 4),
                "delta": round(float(np.mean(svn_b) - np.mean(svn_a)), 4),
                "p_value_wilcoxon": p_val_svn,
            },
            "mean_participation_ratio": {
                "unguided": round(float(np.mean(pr_a)), 2),
                "prigogine": round(float(np.mean(pr_b)), 2),
                "delta": round(float(np.mean(pr_b) - np.mean(pr_a)), 2),
                "p_value_wilcoxon": p_val_pr,
            },
            "mean_kurtosis": {
                "unguided": round(float(np.mean(kurt_a)), 2),
                "prigogine": round(float(np.mean(kurt_b)), 2),
            },
            "verdict": l2_verdict,
        }
    else:
        _prev_l = _load_prev_payload("results/eval_L_holographic_wick_prigogine_gemma4.json")
        trajectories_l2 = _prev_l.get("trajectories_l2", [])
        stats_l2 = _prev_l.get("summary", {}).get("sub_experiments", {}).get(
            "L2_prigogine_criticality_benchmark", {})

    summary_l = {
        "evaluation": "Eval_L_Holographic_Wick_And_Prigogine_Criticality",
        "sub_experiments": {
            "L1_holographic_wick_benchmark": stats_l1,
            "L2_prigogine_criticality_benchmark": stats_l2,
        },
    }

    full_payload_l = {
        "summary": summary_l,
        "trials_l1": trials_l1,
        "trajectories_l2": trajectories_l2,
    }

    os.makedirs("results", exist_ok=True)
    out_path_l = "results/eval_L_holographic_wick_prigogine_gemma4.json"
    with open(out_path_l, "w", encoding="utf-8") as f:
        json.dump(full_payload_l, f, indent=2)

    print("\n" + "=" * 70)
    print("EVAL L FINISHED SUCCESSFULLY:")
    print(f"  L1 S_vN Wide vs Narrow:  p={metric_stats['von_neumann_entropy_bits']['p_value_wide_vs_narrow']:.4e} | SNR={metric_stats['von_neumann_entropy_bits']['snr_signal_to_noise']:.2f}")
    print(f"  L1 L_tau Curvature:      SNR={metric_stats['wick_vertical_curvature_rad']['snr_signal_to_noise']:.2f} | Wide={metric_stats['wick_vertical_curvature_rad']['mean_delta']['wide']:+.4f} vs Narrow={metric_stats['wick_vertical_curvature_rad']['mean_delta']['narrow']:+.4f}")
    print(f"  L2 Prigogine D3:         Unguided={stats_l2['mean_distinct_3']['unguided']:.3f} -> Prigogine={stats_l2['mean_distinct_3']['prigogine']:.3f} (p={stats_l2['mean_distinct_3']['p_value_wilcoxon']:.4e})")
    print(f"  L2 Verdict:              {stats_l2.get('verdict', 'n/a') if not run_l2 else l2_verdict}")
    print(f"  Saved to {out_path_l}\n")
    return full_payload_l


# =========================================================================
# MAIN DISPATCHER
# =========================================================================
async def main():
    parser = argparse.ArgumentParser(description="Run Gemma 4 Evaluations A, B, C, D, E, F, G, H, I, J, K, L")
    parser.add_argument("--suite", choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "all"], default="all", help="Evaluation suite to execute")
    parser.add_argument("--gate", choices=["covariance", "efference"], default="covariance",
                        help="K1 control law. 'covariance' = registered arm (Delta-K/Delta-Omega/drift gate); "
                             "'efference' = constant-free arm (delta_w > 0 against the identical clean replay, "
                             "step = delta_k/kurt_clean). The efference arm writes its own results file.")
    parser.add_argument("--subs", default=None,
                        help="Comma separated sub-experiments to (re)run, e.g. 'K1,L1'. Skipped sub-experiments "
                             "are carried over unchanged from the existing results JSON.")
    args = parser.parse_args()

    if args.subs:
        subs = tuple(s.strip().upper() for s in args.subs.split(",") if s.strip())
    else:
        subs = ("K1", "K2", "L1", "L2")

    print(f"[run_gemma4_evaluations] Loading gemma4-e2b-it for suite={args.suite}...")
    manager = ModelManager()
    entry = await manager.get_model("gemma4-e2b-it", px_subjective=False, px_config_preset="BASELINE")
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    text_model = manager._resolve_text_model(model)

    if args.suite in ("A", "all"):
        await run_eval_a(model, tokenizer, text_model)

    if args.suite in ("B", "all"):
        await run_eval_b(model, tokenizer, text_model)

    if args.suite in ("C", "all"):
        await run_eval_c(model, tokenizer, text_model)

    if args.suite in ("D", "all"):
        await run_eval_d(model, tokenizer, text_model)

    if args.suite in ("E", "all"):
        await run_eval_e(model, tokenizer, text_model)

    if args.suite in ("F", "all"):
        await run_eval_f(model, tokenizer, text_model)

    if args.suite in ("G", "all"):
        await run_eval_g(model, tokenizer, text_model)

    if args.suite in ("H", "all"):
        await run_eval_h(model, tokenizer, text_model)

    if args.suite in ("I", "all"):
        await run_eval_i(model, tokenizer, text_model)

    if args.suite in ("J", "all"):
        await run_eval_j(model, tokenizer, text_model)

    if args.suite in ("K", "all"):
        await run_eval_k(model, tokenizer, text_model, subs=subs, gate=args.gate)

    if args.suite in ("L", "all"):
        await run_eval_l(model, tokenizer, text_model, subs=subs)

    print("\nALL REQUESTED EVALUATION SUITES COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())
