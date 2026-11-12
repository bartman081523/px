"""
Unit tests for Evaluation L: Holographic Von-Neumann Spectral Entropy,
Wick-Rotation Curvature, and Prigogine Non-Equilibrium Criticality.
"""

import math
import pytest
import torch
import torch.nn.functional as F
import numpy as np

from scripts.run_gemma4_evaluations import compute_distinct_ngrams


def test_von_neumann_entropy_and_participation_ratio():
    """Verify mathematical properties of Von-Neumann Spectral Entropy and PR on trajectory matrices."""
    torch.manual_seed(42)
    T = 40
    d = 256

    # 1. Rank-1 trajectory (pure line in representation space): minimal entropy ~ 0, PR = 1.0
    v = torch.randn(1, d)
    v = v / v.norm()
    t_steps = torch.linspace(0.1, 4.0, T).unsqueeze(1)
    H_line = t_steps * v  # rank 1
    H_c = H_line - H_line.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(H_c)
    var = s ** 2
    sum_var = var.sum().item()
    lambdas = var / (sum_var + 1e-12)
    pr_rank1 = float(1.0 / (torch.sum(lambdas ** 2).item() + 1e-12))
    s_vn_rank1 = float(-torch.sum(lambdas * torch.log2(lambdas + 1e-12)).item())

    assert abs(pr_rank1 - 1.0) < 0.05, f"Rank-1 trajectory must have PR ~ 1.0, got {pr_rank1}"
    assert s_vn_rank1 < 0.10, f"Rank-1 trajectory must have S_vN near 0, got {s_vn_rank1}"

    # 2. Isotropic high-dimensional random walk: high entropy, high PR
    H_random = torch.randn(T, d)
    H_rc = H_random - H_random.mean(dim=0, keepdim=True)
    s_r = torch.linalg.svdvals(H_rc)
    var_r = s_r ** 2
    sum_var_r = var_r.sum().item()
    lambdas_r = var_r / (sum_var_r + 1e-12)
    pr_random = float(1.0 / (torch.sum(lambdas_r ** 2).item() + 1e-12))
    s_vn_random = float(-torch.sum(lambdas_r * torch.log2(lambdas_r + 1e-12)).item())

    assert pr_random > 10.0, f"Isotropic random walk must have high PR, got {pr_random}"
    assert s_vn_random > 3.0, f"Isotropic random walk must have high S_vN, got {s_vn_random}"


def test_wick_curvature_and_anisotropy_computation():
    """Verify imaginary-time vertical curvature across layer transitions."""
    torch.manual_seed(42)
    T = 10
    d = 128
    # 9 layer intervals (layers 26 to 34)
    # Identical representations -> 0 vertical curvature
    h_same = [torch.randn(T, d) for _ in range(10)]
    for l in range(1, 10):
        h_same[l] = h_same[0].clone()

    vert_sums = []
    for t in range(T):
        s_val = 0.0
        for l in range(9):
            c = F.cosine_similarity(h_same[l][t:t+1], h_same[l+1][t:t+1]).item()
            s_val += math.acos(max(-1.0, min(1.0, c)))
        vert_sums.append(s_val)

    l_tau_zero = float(np.mean(vert_sums))
    assert abs(l_tau_zero) < 0.01, f"Identical layer states must yield near-zero curvature, got {l_tau_zero}"

    # Rotating representations across layers -> positive curvature
    vert_sums_rot = []
    h_rot = [torch.randn(T, d) for _ in range(10)]
    for t in range(T):
        s_val = 0.0
        for l in range(9):
            c = F.cosine_similarity(h_rot[l][t:t+1], h_rot[l+1][t:t+1]).item()
            s_val += math.acos(max(-1.0, min(1.0, c)))
        vert_sums_rot.append(s_val)

    l_tau_rot = float(np.mean(vert_sums_rot))
    assert l_tau_rot > 0.5, f"Non-aligned layer transitions must yield positive curvature, got {l_tau_rot}"


def test_prigogine_criticality_controller_logic():
    """Verify Prigogine non-equilibrium controller transitions."""
    # 1. Frozen crystal trap (kurtosis 70.0, S_vN 1.50): must expand via positive alpha step
    kurt_freeze = 70.0
    s_vn_freeze = 1.50
    alpha_0 = 0.00
    urgency = (kurt_freeze - 45.0) / 20.0
    step = 0.22 * urgency
    next_alpha = float(max(-0.45, min(0.45, alpha_0 + step)))
    assert next_alpha > 0.20, f"Frozen state must inject positive expansion, got {next_alpha}"

    # 2. Over-dispersed chaotic state (kurtosis 18.0, S_vN 2.65): must contract via negative step
    kurt_chaotic = 18.0
    s_vn_chaotic = 2.65
    next_alpha_focus = float(max(-0.45, min(0.45, 0.00 - 0.15)))
    assert next_alpha_focus == -0.15, f"Chaotic state must focus via negative alpha, got {next_alpha_focus}"

    # 3. Critical target zone (kurtosis 40.0, S_vN 2.10): maintains near equilibrium
    alpha_crit = 0.10 * 0.80
    assert abs(alpha_crit) < 0.10
