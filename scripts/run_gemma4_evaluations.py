#!/usr/bin/env python3
"""scripts/run_gemma4_evaluations.py — Comprehensive Evaluation Suite for Gemma 4 (Layer 26).

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
# MAIN DISPATCHER
# =========================================================================
async def main():
    parser = argparse.ArgumentParser(description="Run Gemma 4 Evaluations A, B, C, D")
    parser.add_argument("--suite", choices=["A", "B", "C", "D", "all"], default="all", help="Evaluation suite to execute")
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

    print("\nALL REQUESTED EVALUATION SUITES COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())
