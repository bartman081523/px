import math
import torch
import torch.nn as nn
import numpy as np
import pytest

from px_patches.gemma4_2b_px.telemetry_collector import (
    Gemma4ManifoldGeometryTracker,
    assemble_enriched_telemetry,
)


class DummyLayer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class DummyTextModel(nn.Module):
    def __init__(self, num_layers=35, dim=64):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(dim) for _ in range(num_layers)])
        self.config = type("Config", (), {"hidden_size": dim, "_name_or_path": "google/gemma-4-E2B-it"})()


def compute_coupled_differential_topological_step(
    current_alpha: float,
    delta_kurt: float,
    delta_pr: float,
    delta_entropy: float,
) -> float:
    """Evaluates the Coupled 2D Differential Topological Control Law."""
    if delta_kurt > 2.5 and delta_pr < -0.25:
        scaling = min(1.5, max(0.5, abs(delta_pr) / 0.50))
        step = +0.15 * scaling
        return float(min(0.50, current_alpha + step))
    elif delta_kurt < -8.0 and delta_pr > 0.35 and delta_entropy > 0.10:
        scaling = min(1.5, max(0.5, delta_pr / 0.80))
        step = -0.15 * scaling
        return float(max(-0.50, current_alpha + step))
    return float(current_alpha)


def compute_symmetric_topological_step(
    current_alpha: float,
    kurtosis: float,
    deadband_low: float = 35.0,
    deadband_high: float = 60.0,
    max_step: float = 0.10,
    target_alpha: float = 0.0,
) -> float:
    """Symmetric topological control step based on logit kurtosis."""
    if kurtosis > deadband_high:
        # Constriction -> inject positive alpha (+d_width)
        step = max_step * math.tanh((kurtosis - deadband_high) / 15.0)
        return float(min(0.50, current_alpha + step))
    elif kurtosis < deadband_low:
        # Over-dilation / loose -> decrease alpha (pull towards 0.0 or negative)
        step = max_step * math.tanh((deadband_low - kurtosis) / 10.0)
        return float(max(-0.50, current_alpha - step))
    else:
        # Balanced deadband
        return float(current_alpha)


def test_participation_ratio_mathematical_invariants():
    dim = 64
    T = 40
    text_model = DummyTextModel(num_layers=35, dim=dim)
    tracker = Gemma4ManifoldGeometryTracker(text_model, torch.randn(dim), relay_layer=26)

    # 1. Rank-1 trajectory (points along a single 1D ray)
    t_steps = torch.linspace(1.0, 10.0, T).unsqueeze(1)
    v1 = torch.randn(1, dim)
    v1 = v1 / v1.norm()
    H_rank1 = t_steps * v1
    tracker.final_hidden_states = [H_rank1[i] for i in range(T)]

    metrics_rank1 = tracker.compute_trajectory_metrics()
    pr_rank1 = metrics_rank1["trajectory_participation_ratio"]
    assert 0.99 <= pr_rank1 <= 1.05, f"Rank-1 trajectory should have PR ~ 1.0, got {pr_rank1}"

    # 2. Additive constant noise shift invariance: H_shifted = H + c * d_rand
    d_rand = torch.randn(1, dim) * 5.0
    H_shifted = H_rank1 + d_rand
    tracker.final_hidden_states = [H_shifted[i] for i in range(T)]
    metrics_shifted = tracker.compute_trajectory_metrics()
    pr_shifted = metrics_shifted["trajectory_participation_ratio"]
    assert math.isclose(pr_rank1, pr_shifted, abs_tol=1e-3), (
        f"PR must be invariant to static additive noise shift! Got {pr_rank1} vs {pr_shifted}"
    )

    # 3. High-dimensional isotropic walk
    torch.manual_seed(42)
    H_high_dim = torch.randn(T, dim)
    tracker.final_hidden_states = [H_high_dim[i] for i in range(T)]
    metrics_high_dim = tracker.compute_trajectory_metrics()
    pr_high_dim = metrics_high_dim["trajectory_participation_ratio"]
    assert pr_high_dim > 10.0, f"High-dim isotropic trajectory should have large PR, got {pr_high_dim}"


def test_symmetric_topological_controller_logic():
    # A. Narrow state (Kurtosis = 75.0 > 60.0): alpha must increase
    a0 = -0.40
    a1 = compute_symmetric_topological_step(a0, kurtosis=75.0)
    assert a1 > a0, f"Narrow state must increase alpha: {a0} -> {a1}"
    assert math.isclose(a1 - a0, 0.10 * math.tanh(15.0 / 15.0), rel_tol=1e-3)

    # B. Wide state (Kurtosis = 25.0 < 35.0): alpha must decrease
    w0 = +0.40
    w1 = compute_symmetric_topological_step(w0, kurtosis=25.0)
    assert w1 < w0, f"Wide state must decrease alpha: {w0} -> {w1}"
    assert math.isclose(w0 - w1, 0.10 * math.tanh(10.0 / 10.0), rel_tol=1e-3)

    # C. Baseline deadband (Kurtosis = 52.0 in [35, 60]): alpha unchanged
    b0 = 0.00
    b1 = compute_symmetric_topological_step(b0, kurtosis=52.0)
    assert b1 == b0, f"Deadband must keep alpha unchanged: {b0} -> {b1}"

    # D. Random noise state (Kurtosis = 44.0 in [35, 60]): 100% noise rejection
    r0 = 0.00
    r1 = compute_symmetric_topological_step(r0, kurtosis=44.0)
    assert r1 == 0.00, f"Random noise must yield 0 false steps: {r0} -> {r1}"


def test_manifold_geometry_summary_includes_trajectory_metrics():
    dim = 32
    text_model = DummyTextModel(num_layers=35, dim=dim)
    dwidth = torch.randn(dim)
    tracker = Gemma4ManifoldGeometryTracker(text_model, dwidth, relay_layer=26)

    # Simulate 5 steps
    for _ in range(5):
        tracker.final_hidden_states.append(torch.randn(dim))

    summary = tracker.get_summary()
    assert "trajectory_participation_ratio" in summary
    assert "radius_of_gyration" in summary
    assert "mean_angular_drift" in summary
    assert summary["trajectory_participation_ratio"] >= 1.0
    assert summary["radius_of_gyration"] >= 0.0
    assert summary["mean_angular_drift"] >= 0.0


def test_coupled_differential_topological_controller_logic():
    # 1. Narrow state: Delta K = +4.5 > 2.5, Delta D = -0.54 < -0.25 -> alpha increases
    a0 = -0.40
    a1 = compute_coupled_differential_topological_step(a0, delta_kurt=4.5, delta_pr=-0.54, delta_entropy=0.04)
    assert a1 > a0, f"Narrow state must increase alpha: {a0} -> {a1}"
    assert a1 == a0 + 0.15 * (0.54 / 0.50)

    # 2. Wide state: Delta K = -26.75 < -8.0, Delta D = +0.87 > 0.35, Delta H = +0.28 > 0.10 -> alpha decreases
    w0 = +0.40
    w1 = compute_coupled_differential_topological_step(w0, delta_kurt=-26.75, delta_pr=0.87, delta_entropy=0.28)
    assert w1 < w0, f"Wide state must decrease alpha: {w0} -> {w1}"
    assert w1 == w0 - 0.15 * (0.87 / 0.80)

    # 3. Rand Plus noise rejection: Delta K = -4.74 (fails < -8.0), Delta D = -0.48 (fails > 0.35) -> 0 step
    r_plus0 = 0.00
    r_plus1 = compute_coupled_differential_topological_step(r_plus0, delta_kurt=-4.74, delta_pr=-0.48, delta_entropy=-0.01)
    assert r_plus1 == 0.00, f"Rand Plus must yield 0 false steps: {r_plus0} -> {r_plus1}"

    # 4. Rand Minus noise rejection: Delta K = -6.43 (fails > 2.5), Delta H = +0.02 (fails > 0.10) -> 0 step
    r_minus0 = 0.00
    r_minus1 = compute_coupled_differential_topological_step(r_minus0, delta_kurt=-6.43, delta_pr=0.26, delta_entropy=0.02)
    assert r_minus1 == 0.00, f"Rand Minus must yield 0 false steps: {r_minus0} -> {r_minus1}"

    # 5. Baseline state: Delta K = 0.0, Delta D = 0.0, Delta H = 0.0 -> 0 step
    b0 = 0.00
    b1 = compute_coupled_differential_topological_step(b0, delta_kurt=0.0, delta_pr=0.0, delta_entropy=0.0)
    assert b1 == 0.00, f"Baseline must yield 0 false steps: {b0} -> {b1}"
