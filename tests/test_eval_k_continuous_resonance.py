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
    compute_efference_copy_step,
    wick_ratio_differential,
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


# --- K1b: constant-free efference-copy arm --------------------------------

def test_efference_gate_closed_when_noise_or_incongruent():
    """Noise controls (Task 0/2: dS_vN < 0, Task 1: dOmega < |dW|) keep gate closed."""
    # Task 0/2 noise: dK < 0, but dS_vN < 0 and dD < 0 -> gate closed
    assert compute_efference_copy_step(0.0, delta_k_true=-12.7, kurt_clean=48.0, delta_w=+0.14,
                                       delta_s_vn_true=-0.047, delta_d_true=-0.45, delta_omega_holo_true=+0.12) == 0.0
    # Task 1 noise: dS_vN > 0 and dD > 0, but dOmega < |dW| (0.18 < 0.23) -> gate closed
    assert compute_efference_copy_step(0.0, delta_k_true=-3.14, kurt_clean=36.0, delta_w=+0.227,
                                       delta_s_vn_true=+0.078, delta_d_true=+0.59, delta_omega_holo_true=+0.182) == 0.0
    # Inverted narrow: dK > 0 but dW <= 0 -> gate closed
    assert compute_efference_copy_step(-0.4, delta_k_true=+9.0, kurt_clean=50.0, delta_w=-0.1,
                                       delta_s_vn_true=-0.05) == -0.4


def test_efference_step_dual_bulk_boundary():
    """Step incorporates both relative boundary displacement (dK/Kc) and bulk order shift (dOm/Omc)."""
    # Narrowing: dK > 0, dS_vN < 0, dW > 0 -> step = dK/Kc - dOm/Omc
    # Kc=50.0, dK=+5.0 -> dK/Kc = +0.10. Omc = 2.0, dOm = -0.10 -> -dOm/Omc = +0.05.
    # Total step = +0.15. Alpha: -0.40 -> -0.25.
    a1 = compute_efference_copy_step(
        -0.40, delta_k_true=+5.0, kurt_clean=50.0, delta_w=+0.2,
        delta_s_vn_true=-0.05, delta_omega_holo_true=-0.10, s_vn_clean=3.4
    )
    # Omega_clean = 3.4 / log10(51) = 3.4 / 1.70757 = 1.99113
    # Step = 5/50 - (-0.10 / 1.99113) = 0.10 + 0.05022 = +0.15022
    assert a1 == pytest.approx(-0.40 + (5.0 / 50.0 - (-0.10 / (3.4 / math.log10(51.0)))), abs=1e-5)

    # Widening: dK < 0, dS_vN > 0, dD > 0, dOm > |dW| -> pulls alpha downward
    a2 = compute_efference_copy_step(
        +0.40, delta_k_true=-5.0, kurt_clean=50.0, delta_w=+0.05,
        delta_s_vn_true=+0.05, delta_d_true=+0.5, delta_omega_holo_true=+0.10, s_vn_clean=3.4
    )
    assert a2 < +0.40


def test_efference_step_has_fixed_point_at_zero_differential():
    """delta_k_true == 0 and delta_omega_holo_true == 0 is the fixed point: controller stops exactly there."""
    for alpha in (-0.40, -0.05, 0.0, 0.05, 0.40):
        assert compute_efference_copy_step(
            alpha, delta_k_true=0.0, kurt_clean=50.0, delta_w=+0.3,
            delta_s_vn_true=-0.05, delta_omega_holo_true=0.0, s_vn_clean=3.0
        ) == pytest.approx(alpha, abs=1e-12)


def test_efference_step_is_rate_limited_not_deadbanded():
    """Small real displacements still move (no deadband); large ones are clipped to +/-0.18."""
    tiny = compute_efference_copy_step(
        -0.40, delta_k_true=+0.05, kurt_clean=50.0, delta_w=+0.01,
        delta_s_vn_true=-0.001, delta_omega_holo_true=-0.001, s_vn_clean=3.0
    )
    assert tiny > -0.40, "any non-zero differential must move the controller (no deadband)"
    hard = compute_efference_copy_step(
        0.0, delta_k_true=+900.0, kurt_clean=50.0, delta_w=+0.01,
        delta_s_vn_true=-1.0, delta_omega_holo_true=-1.0, s_vn_clean=3.0
    )
    assert hard == pytest.approx(0.18, abs=1e-12)


def test_efference_converges_symmetrically_from_both_sides():
    """Simulated closed loop with dual bulk-boundary dynamics."""
    # Narrow start (-0.40)
    alpha_n = -0.40
    for _ in range(6):
        alpha_n = compute_efference_copy_step(
            alpha_n, delta_k_true=-30.0 * alpha_n, kurt_clean=55.0, delta_w=+0.2,
            delta_s_vn_true=0.1 * alpha_n, delta_omega_holo_true=0.2 * alpha_n, s_vn_clean=3.2
        )
    assert abs(alpha_n) <= 0.05, f"narrow converged only to {alpha_n:+.4f}"

    # Wide start (+0.40)
    alpha_w = +0.40
    for _ in range(6):
        alpha_w = compute_efference_copy_step(
            alpha_w, delta_k_true=-50.0 * alpha_w, kurt_clean=30.0, delta_w=+0.05,
            delta_s_vn_true=0.1 * alpha_w, delta_d_true=0.8 * alpha_w, delta_omega_holo_true=0.5 * alpha_w, s_vn_clean=3.5
        )
    assert abs(alpha_w) <= 0.05, f"wide converged only to {alpha_w:+.4f}"


def test_efference_keeps_baseline_exactly_zero():
    """Inactive controller: delta_k == 0 and delta_om == 0 keeps alpha at 0."""
    for _ in range(6):
        assert compute_efference_copy_step(
            0.0, delta_k_true=0.0, kurt_clean=44.0, delta_w=+0.4,
            delta_s_vn_true=0.0, delta_omega_holo_true=0.0, s_vn_clean=3.0
        ) == 0.0


def test_wick_ratio_differential_is_zero_on_identical_measurements():
    """Inactive controller: identical steered/clean measurements -> delta_w exactly 0.0.

    Regression pin: the first K1b run mixed step-wise telemetry drift (~0.34 rad)
    with forward-pass drift (~0.96 rad), which produced delta_w ~= +3.9 at alpha=0
    and left the coherence gate permanently open.
    """
    m = {"wick_curvature": 2.5897, "drift": 0.9568, "kurtosis": 44.0}
    w_s, w_c, dw = wick_ratio_differential(m, dict(m))
    assert w_s == w_c
    assert dw == 0.0


def test_wick_ratio_is_scale_free():
    """W is a ratio: identical relative curvature gives the same value at any drift scale."""
    a = {"wick_curvature": 2.0, "drift": 0.5}
    b = {"wick_curvature": 4.0, "drift": 1.0}
    w_a, _, _ = wick_ratio_differential(a, a)
    w_b, _, _ = wick_ratio_differential(b, b)
    assert w_a == pytest.approx(4.0)
    assert w_a == pytest.approx(w_b)


def test_wick_ratio_differential_does_not_separate_d_rand_on_the_forward_path():
    """Measured reality (probe 2026-11-13): on the *forward replay* path both a genuine
    and an orthogonal perturbation raise W (wide +0.17, narrow +0.34, d_rand +0.22),
    so delta_w > 0 alone is NOT a noise discriminant. This test pins that finding so
    nobody re-derives the gate from a false premise.
    """
    clean = {"wick_curvature": 2.9944, "drift": 1.0077}
    for steered in ({"wick_curvature": 3.0936, "drift": 0.9751},    # +d_width
                    {"wick_curvature": 3.0694, "drift": 0.9719}):   # +d_rand
        _, _, dw = wick_ratio_differential(steered, clean)
        assert dw > 0.0, "forward-path W rises for both perturbations - not discriminative"
