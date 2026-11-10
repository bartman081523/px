"""
Unit tests for Evaluation K: Continuous Intra-Pass Resonance,
Zero-Threshold Symmetric Recovery, and Task-Adaptive Utility Benchmark.
"""

import math
import pytest
import torch
import numpy as np

from scripts.run_gemma4_evaluations import (
    compute_continuous_proportional_step,
    compute_intrapass_differential_metrics,
    compute_task_adaptive_step,
    compute_distinct_ngrams,
)


def test_intrapass_differential_zero_invariance():
    """Verify that when alpha=0 (logits_post == logits_pre), deltas are identically zero."""
    torch.manual_seed(42)
    dummy_logits = torch.randn(1, 1000, dtype=torch.float32) * 5.0

    diff = compute_intrapass_differential_metrics(dummy_logits, dummy_logits)
    assert diff["delta_k_local"] == 0.0
    assert diff["delta_h_local"] == 0.0


def test_continuous_proportional_step_dynamics():
    """Verify continuous proportional recovery without if-else deadbands."""
    # 1. Narrow recovery: alpha = -0.40, delta_k_local = +15.0 -> must pull upward toward 0
    alpha_0 = -0.40
    alpha_1 = compute_continuous_proportional_step(alpha_0, delta_k_local=+15.0, gamma=0.15, sigma_k=10.0)
    assert alpha_1 > alpha_0
    assert alpha_1 < 0.0

    # Multi-step convergence simulation
    alpha_t = -0.40
    for _ in range(5):
        # As alpha approaches 0, perturbation effect decreases proportionally
        sim_delta_k = -30.0 * alpha_t  # positive when alpha_t is negative
        alpha_t = compute_continuous_proportional_step(alpha_t, sim_delta_k, gamma=0.15, sigma_k=10.0)
    assert abs(alpha_t) < 0.05, f"Expected convergence to |alpha| < 0.05, got {alpha_t}"

    # 2. Wide recovery: alpha = +0.40, delta_k_local = -18.0 -> must pull downward toward 0
    alpha_0_w = +0.40
    alpha_1_w = compute_continuous_proportional_step(alpha_0_w, delta_k_local=-18.0, gamma=0.15, sigma_k=10.0)
    assert alpha_1_w < alpha_0_w
    assert alpha_1_w > 0.0

    alpha_t_w = +0.40
    for _ in range(5):
        sim_delta_k_w = -30.0 * alpha_t_w  # negative when alpha_t_w is positive
        alpha_t_w = compute_continuous_proportional_step(alpha_t_w, sim_delta_k_w, gamma=0.15, sigma_k=10.0)
    assert abs(alpha_t_w) < 0.05, f"Expected convergence to |alpha| < 0.05, got {alpha_t_w}"

    # 3. Baseline & noise zero-drift: delta_k_local = 0.0 -> alpha remains strictly constant
    assert compute_continuous_proportional_step(0.0, 0.0) == 0.0


def test_task_adaptive_utility_step():
    """Verify task-adaptive regulation: widening on freezing, narrowing on chaotic dispersal."""
    # Freezing trap: high kurtosis (95.0), low entropy (0.10) -> must widen (alpha > 0)
    alpha_freezing = compute_task_adaptive_step(current_alpha=0.0, current_kurt=95.0, current_ent=0.10)
    assert alpha_freezing > 0.0, f"Expected positive alpha to widen frozen loop, got {alpha_freezing}"

    # Dispersal trap: low kurtosis (15.0), high entropy (0.90) -> must narrow (alpha < 0)
    alpha_dispersal = compute_task_adaptive_step(current_alpha=0.0, current_kurt=15.0, current_ent=0.90)
    assert alpha_dispersal < 0.0, f"Expected negative alpha to tighten dispersed beam, got {alpha_dispersal}"

    # Nominal healthy state (kurtosis 45.0, entropy 0.40) -> alpha stays at 0.0
    alpha_healthy = compute_task_adaptive_step(current_alpha=0.0, current_kurt=45.0, current_ent=0.40)
    assert abs(alpha_healthy) < 0.02


def test_compute_distinct_ngrams():
    """Verify distinct n-gram calculation detects repetition loops."""
    repetitive_text = "shipping lanes shipping lanes shipping lanes shipping lanes shipping lanes shipping lanes"
    diverse_text = "shipping lanes connect deep-sea ports with continental bulk transport distribution centers"

    d_rep = compute_distinct_ngrams(repetitive_text, n=3)
    d_div = compute_distinct_ngrams(diverse_text, n=3)

    assert d_rep < 0.30
    assert d_div > 0.90

