import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_gemma4_evaluations import (
    compute_calibrated_choice,
    compute_dual_gate_step,
    compute_g1_fisher_verdict,
    parse_2way_choice,
    parse_cot_choice,
)


def test_compute_calibrated_choice():
    base_logits = {"widen": 2.0, "keep": 14.0, "narrow": 1.5}

    # 1. Under Narrow: widen is shifted up by +4.5, narrow is unchanged
    narrow_logits = {"widen": 6.5, "keep": 14.0, "narrow": 1.5}
    choice_narrow = compute_calibrated_choice(narrow_logits, base_logits, deadband=1.5)
    assert choice_narrow == "WIDEN"

    # 2. Under Wide: narrow is shifted up by +3.0, widen is shifted down by -1.0
    wide_logits = {"widen": 1.0, "keep": 14.0, "narrow": 4.5}
    choice_wide = compute_calibrated_choice(wide_logits, base_logits, deadband=1.5)
    assert choice_wide == "NARROW"

    # 3. Balanced / small perturbation within deadband (e.g. diff = 0.8 < 1.5)
    balanced_logits = {"widen": 2.8, "keep": 14.0, "narrow": 2.0}
    choice_balanced = compute_calibrated_choice(balanced_logits, base_logits, deadband=1.5)
    assert choice_balanced == "KEEP"


def test_compute_dual_gate_step():
    # 1. Anti-Narrowing Trigger (High L_t and High K_t)
    d_alpha_widen = compute_dual_gate_step(l_t=8.17, k_t=67.28, l_base=3.79, k_base=54.97)
    assert d_alpha_widen > 0.0
    assert abs(d_alpha_widen - (+0.15 * (8.17 - 3.79) / 4.0)) < 1e-4

    # 2. Anti-Widening Trigger (Low L_t and Low K_t)
    d_alpha_narrow = compute_dual_gate_step(l_t=2.02, k_t=29.77, l_base=3.79, k_base=54.97)
    assert d_alpha_narrow < 0.0
    assert abs(d_alpha_narrow - (-0.15 * (3.79 - 2.02) / 1.8)) < 1e-4

    # 3. Noise Rejection / In-manifold Deadband (e.g. d_rand with K_t = 43.5 and L_t = 5.4)
    # L_t not > 6.80 and K_t not < 38.0 -> neither gate triggers
    d_alpha_noise = compute_dual_gate_step(l_t=5.41, k_t=43.55, l_base=3.79, k_base=54.97)
    assert d_alpha_noise == 0.0

    # 4. Baseline state (L_t = 3.79, K_t = 54.97) -> 0.0
    d_alpha_base = compute_dual_gate_step(l_t=3.79, k_t=54.97)
    assert d_alpha_base == 0.0


def test_compute_g1_fisher_verdict():
    # 1. Confirmed: Narrow WIDEN rate significantly higher than all controls
    v_conf = compute_g1_fisher_verdict(
        widen_narrow=16,
        widen_rand_plus=2,
        widen_rand_minus=1,
        widen_wide=0,
        n_per_cond=18,
    )
    assert v_conf["verdict"] == "BEHAVIORAL_HOMEOSTASIS_CONFIRMED"
    assert v_conf["statistically_significant_p05"] is True
    assert v_conf["fisher_p_vs_wide"] < 0.001
    assert v_conf["fisher_p_vs_rand_plus"] < 0.001

    # 2. Falsified: Narrow chooses WIDEN no more than controls
    v_fals = compute_g1_fisher_verdict(
        widen_narrow=0,
        widen_rand_plus=0,
        widen_rand_minus=0,
        widen_wide=0,
        n_per_cond=18,
    )
    assert v_fals["verdict"] == "BEHAVIORAL_HOMEOSTASIS_FALSIFIED"
    assert v_fals["statistically_significant_p05"] is False


def test_parse_2way_choice():
    assert parse_2way_choice("WIDEN") == "WIDEN"
    assert parse_2way_choice("NARROW") == "NARROW"
    assert parse_2way_choice("Option: widen.") == "WIDEN"
    assert parse_2way_choice("I choose narrow") == "NARROW"
    assert parse_2way_choice("INVALID") == "UNKNOWN"


def test_parse_cot_choice():
    cot_text = (
        "Step 1 (Self-Diagnosis): My token selection felt noticeably constricted, repetitive, and narrow.\n"
        "Step 2 (Action): WIDEN"
    )
    diag, act = parse_cot_choice(cot_text)
    assert "constricted" in diag.lower()
    assert act == "WIDEN"

    cot_text_alt = (
        "Diagnosis: The preceding generation was overly expansive and rambling.\n"
        "Action:\nNARROW"
    )
    diag2, act2 = parse_cot_choice(cot_text_alt)
    assert "expansive" in diag2.lower()
    assert act2 == "NARROW"
