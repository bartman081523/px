"""
Unit tests for Evaluation K: Continuous Intra-Pass Resonance,
Zero-Threshold Symmetric Recovery, and Task-Adaptive Utility Benchmark.
"""

import math
import pytest
import torch
import numpy as np

from scripts.run_gemma4_evaluations import (
    DRIFT_NOISE_GATE_RAD,
    compute_covariance_gated_step,
    compute_counterfactual_differential,
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


def test_covariance_gated_step_narrow_recovery():
    """Verify Covariance Gating on Narrow perturbation (ΔK > 0, ΔD < 0)."""
    alpha_0 = -0.40
    # Narrow signature: kurtosis increased (+20.0), Omega_holo decreased (-0.05), low drift
    alpha_1 = compute_covariance_gated_step(
        alpha_0, delta_k_true=+20.0, delta_omega_holo_true=-0.05,
        drift_steered=0.40, delta_d_true=-0.7,
    )
    assert alpha_1 > alpha_0, "Narrow recovery must pull alpha upward towards 0"
    assert alpha_1 < 0.0

    # Multi-step convergence test with empirical decay
    alpha_t = -0.40
    for _ in range(5):
        # Physical model: deltas scale with remaining perturbation
        delta_k = -50.0 * alpha_t  # positive when alpha_t < 0
        delta_omega = 0.55 * alpha_t  # negative when alpha_t < 0 (Omega_holo follows S_vN/K)
        delta_d = 1.5 * alpha_t    # negative when alpha_t < 0
        delta_h = 0.5 * alpha_t    # negative when alpha_t < 0
        alpha_t = compute_covariance_gated_step(
            alpha_t, delta_k, delta_omega_holo_true=delta_omega,
            drift_steered=0.40, delta_d_true=delta_d, delta_h_true=delta_h,
        )
    # The Omega_holo gate (delta_omega < -0.02) is a deliberate dead zone: once the
    # remaining perturbation no longer produces a measurable holographic signature the
    # controller freezes, so convergence is monotone but gate-limited (< 0.10).
    assert abs(alpha_t) < 0.10, f"Expected gate-limited convergence, got {alpha_t}"


def test_covariance_gated_step_wide_recovery():
    """Verify Covariance Gating on Wide perturbation (ΔK < 0, ΔD > 0)."""
    alpha_0 = +0.40
    # Wide signature: kurtosis decreased (-22.0), Omega_holo increased (+0.08), low drift
    alpha_1 = compute_covariance_gated_step(
        alpha_0, delta_k_true=-22.0, delta_omega_holo_true=+0.08,
        drift_steered=0.40, delta_d_true=+0.8, delta_h_true=+0.25,
    )
    assert alpha_1 < alpha_0, "Wide recovery must pull alpha downward towards 0"
    assert alpha_1 > 0.0

    # Multi-step convergence test with empirical decay
    alpha_t = +0.40
    for _ in range(5):
        delta_k = -50.0 * alpha_t  # negative when alpha_t > 0
        delta_omega = 1.00 * alpha_t  # positive when alpha_t > 0 (Omega_holo follows S_vN/K)
        delta_d = 1.5 * alpha_t    # positive when alpha_t > 0
        delta_h = 0.5 * alpha_t    # positive when alpha_t > 0
        alpha_t = compute_covariance_gated_step(
            alpha_t, delta_k, delta_omega_holo_true=delta_omega,
            drift_steered=0.40, delta_d_true=delta_d, delta_h_true=delta_h,
        )
    # The Omega_holo gate (delta_omega < -0.02) is a deliberate dead zone: once the
    # remaining perturbation no longer produces a measurable holographic signature the
    # controller freezes, so convergence is monotone but gate-limited (< 0.10).
    assert abs(alpha_t) < 0.10, f"Expected gate-limited convergence, got {alpha_t}"


def test_covariance_gated_step_noise_and_baseline_immunity():
    """Verify that orthogonal noise (same signs) or zero drift produces strict zero step."""
    # Orthogonal noise case: both positive or both negative
    alpha_rand1 = compute_covariance_gated_step(0.0, delta_k_true=+8.0, delta_d_true=+0.4)
    assert alpha_rand1 == 0.0, "Same-sign noise must be rejected by covariance gate"

    alpha_rand2 = compute_covariance_gated_step(0.0, delta_k_true=-5.0, delta_d_true=-0.3)
    assert alpha_rand2 == 0.0, "Same-sign noise must be rejected by covariance gate"

    # Baseline case: ~0 deltas
    alpha_base = compute_covariance_gated_step(0.0, delta_k_true=0.0, delta_d_true=0.0)
    assert alpha_base == 0.0, "Zero deltas must produce exactly zero alpha shift"


def test_task_adaptive_utility_step():
    """Verify task-adaptive regulation: widening on repetition/freezing, narrowing on chaotic dispersal."""
    # Freezing trap by kurtosis: high kurtosis (95.0), low entropy (0.10) -> must widen (alpha > 0)
    alpha_freezing = compute_task_adaptive_step(current_alpha=0.0, current_kurt=95.0, current_ent=0.10)
    assert alpha_freezing > 0.0, f"Expected positive alpha to widen frozen loop, got {alpha_freezing}"

    # Repetition trap by distinct n-grams: distinct_3 = 0.35 -> must widen (alpha > 0)
    alpha_rep = compute_task_adaptive_step(current_alpha=0.0, current_kurt=45.0, current_ent=0.35, distinct_3=0.35)
    assert alpha_rep > 0.0, f"Expected positive alpha to break repetitive n-grams, got {alpha_rep}"

    # Dispersal trap: low kurtosis (15.0), high entropy (0.90) -> must narrow (alpha < 0)
    alpha_dispersal = compute_task_adaptive_step(current_alpha=0.0, current_kurt=15.0, current_ent=0.90)
    assert alpha_dispersal < 0.0, f"Expected negative alpha to tighten dispersed beam, got {alpha_dispersal}"

    # Nominal healthy state (kurtosis 45.0, entropy 0.40, distinct_3 0.85) -> alpha stays at 0.0
    alpha_healthy = compute_task_adaptive_step(current_alpha=0.0, current_kurt=45.0, current_ent=0.40, distinct_3=0.85)
    assert abs(alpha_healthy) < 0.02


def test_compute_distinct_ngrams():
    """Verify distinct n-gram calculation detects repetition loops."""
    repetitive_text = "shipping lanes shipping lanes shipping lanes shipping lanes shipping lanes shipping lanes"
    diverse_text = "shipping lanes connect deep-sea ports with continental bulk transport distribution centers"

    d_rep = compute_distinct_ngrams(repetitive_text, n=3)
    d_div = compute_distinct_ngrams(diverse_text, n=3)

    assert d_rep < 0.30
    assert d_div > 0.90


def test_angular_drift_noise_gate():
    """Orthogonal d_rand produces high angular jitter -> the controller must not act."""
    alpha_next = compute_covariance_gated_step(
        0.0, delta_k_true=+15.0, delta_omega_holo_true=-0.20,
        drift_steered=DRIFT_NOISE_GATE_RAD + 0.01, delta_d_true=-0.4,
    )
    assert alpha_next == 0.0, "Drift above the noise gate must freeze the controller"


def test_counterfactual_differential_zero_invariance():
    """Identical steered/clean measurements (inactive controller) -> exactly 0.0 deltas."""
    metrics = {
        "kurtosis": 41.25, "entropy": 3.10, "pr": 5.42, "s_vn": 3.37,
        "omega_holo": 0.90, "drift": 0.40, "wick_curvature": 3.12,
        "wick_anisotropy": 1.28, "num_gen": 30,
    }
    diff = compute_counterfactual_differential(metrics, dict(metrics))
    assert diff["delta_k_true"] == 0.0
    assert diff["delta_d_true"] == 0.0
    assert diff["delta_s_vn_true"] == 0.0
    assert diff["delta_h_true"] == 0.0
    assert diff["delta_omega_holo_true"] == 0.0
    assert diff["delta_wick_curvature_true"] == 0.0
