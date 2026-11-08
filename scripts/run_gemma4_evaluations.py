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


def load_dwidth_tensor():
    path = os.path.join("px_manifolds", "google_gemma-4-E2B-it_relay_dwidth.json")
    with open(path, "r", encoding="utf-8") as f:
        art = json.load(f)
    dw = np.array(art["dwidth"], dtype=np.float32)
    t = torch.tensor(dw, dtype=torch.float32)
    return t / (t.norm() + 1e-8)


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

    duration_s = time.perf_counter() - t0
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
# MAIN DISPATCHER
# =========================================================================
async def main():
    parser = argparse.ArgumentParser(description="Run Gemma 4 Evaluations A, B, C, D, E, F")
    parser.add_argument("--suite", choices=["A", "B", "C", "D", "E", "F", "all"], default="all", help="Evaluation suite to execute")
    args = parser.parse_args()

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

    print("\nALL REQUESTED EVALUATION SUITES COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())
