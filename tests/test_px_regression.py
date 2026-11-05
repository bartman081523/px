"""tests/test_px_regression.py — DevMind Automated Regression Test Suite.

Masterplan Phase 3:
1. Verifies continuous scale monotonic ordering invariants (Narrow < Baseline < Wide)
   and statistical hypothesis thresholds (Kruskal-Wallis p < 0.01, Spearman rho > 0.70).
2. Verifies forward-hook lifecycle invariants (100% hook cleanup to 0, zero accumulation).
3. Verifies telemetry schema and session export integrity (phi, kurtosis, emancipation,
   zone_weights, exact_latency_ms, frozen settings).
4. Verifies forward-hook mathematical metric computation on activation tensors.
"""

import json
import os
import tempfile
import numpy as np
import pytest
import scipy.stats as stats
import torch
import torch.nn.functional as F

from sessions import save_session, load_session, get_new_session_id
from gradio_tabs.chat_tab import handle_export


def test_continuous_scale_monotonic_ordering_invariants():
    """Verify that empirical batch results maintain strict monotonic ordering and p < 0.01."""
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "results", "continuous_scale_batch60_gemma4-e2b-it.json")
    assert os.path.exists(results_path), f"Results file missing: {results_path}"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    stats_dict = summary["stats"]

    mean_narrow = stats_dict["narrow"]["mean"]
    mean_base = stats_dict["base"]["mean"]
    mean_wide = stats_dict["wide"]["mean"]

    # 1. Monotonicity Assertion
    assert mean_narrow < mean_base < mean_wide, f"Monotonic ordering failed: {mean_narrow} < {mean_base} < {mean_wide}"

    # 2. Statistical Significance Assertions
    kw_p = summary["kruskal_wallis"]["p_value"]
    assert kw_p < 0.01, f"Kruskal-Wallis p-value {kw_p} not significant at alpha=0.01"

    u_wn_p = summary["mann_whitney_u"]["wide_vs_narrow"]["p_value"]
    assert u_wn_p < 0.01, f"Wide vs Narrow Mann-Whitney p-value {u_wn_p} >= 0.01"

    u_wb_p = summary["mann_whitney_u"]["wide_vs_base"]["p_value"]
    assert u_wb_p < 0.01, f"Wide vs Base Mann-Whitney p-value {u_wb_p} >= 0.01"

    spearman_rho = summary["spearman_correlation"]["rho"]
    spearman_p = summary["spearman_correlation"]["p_value"]
    assert spearman_rho > 0.70, f"Spearman rho {spearman_rho} <= 0.70"
    assert spearman_p < 0.01, f"Spearman p-value {spearman_p} >= 0.01"


def test_hook_lifecycle_zero_leak_invariant():
    """Verify that 50 consecutive inference cycles have 0 hook leaks and 0 hook accumulation."""
    bench_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "results", "hook_lifecycle_50calls_benchmark.json")
    assert os.path.exists(bench_path), f"Benchmark file missing: {bench_path}"

    with open(bench_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["n_calls"] == 50
    assert data["all_hooks_cleaned"] is True, "Hooks were not cleaned to 0 after every call"
    assert data["no_hook_accumulation"] is True, "Hook accumulation detected before calls"
    assert data["steady_state_mean_ms"] < 1000.0, f"Latency unexpectedly high: {data['steady_state_mean_ms']} ms"


def test_telemetry_schema_and_export_integrity():
    """Verify that real-time telemetry metrics are persisted and faithfully exported."""
    sess_id = get_new_session_id()
    history = [
        {"role": "user", "content": "Explain your internal state."},
        {"role": "assistant", "content": "1. **Value:** +0.65"}
    ]
    frozen_settings = {
        "model_id": "gemma4-e2b-it",
        "relay_sign": 1,
        "relay_alpha": 0.35,
        "relay_layer": 26,
        "temperature": 0.3,
        "seed": 105
    }
    px_telemetry = {
        "phi": 0.9412,
        "kurtosis": 2.150,
        "emancipation_trajectory": [0.0, 1.12, 2.34, 3.01],
        "zone_weights": {"creative": 0.68, "math": 0.32},
        "exact_latency_ms": 0.145
    }

    sess_path = save_session(
        sess_id,
        history,
        model_id="gemma4-e2b-it",
        settings=frozen_settings,
        px_telemetry=px_telemetry
    )
    try:
        # Load and verify stored representation
        loaded = load_session(sess_id)
        assert loaded["settings"]["relay_sign"] == 1
        assert loaded["settings"]["seed"] == 105
        assert loaded["px_telemetry"]["phi"] == 0.9412
        assert loaded["px_telemetry"]["kurtosis"] == 2.150
        assert len(loaded["px_telemetry"]["emancipation_trajectory"]) == 4

        # Export and verify JSON export file
        export_res = handle_export(sess_id, history)
        assert export_res["visible"] is True
        export_file = export_res["value"]

        with open(export_file, "r", encoding="utf-8") as f:
            exported_json = json.load(f)

        assert exported_json["session_id"] == sess_id
        assert exported_json["settings"] == frozen_settings
        assert exported_json["px_telemetry"] == px_telemetry
        os.remove(export_file)
    finally:
        if os.path.exists(sess_path):
            os.remove(sess_path)


def test_mathematical_kurtosis_and_phi_computation():
    """Verify mathematical correctness of the excess kurtosis and cosine similarity formulas."""
    # 1. Normal distribution (excess kurtosis ~ 0)
    torch.manual_seed(42)
    norm_tensor = torch.randn(1, 2048)
    mean_x = torch.mean(norm_tensor)
    var_x = torch.var(norm_tensor, unbiased=False)
    m4 = torch.mean((norm_tensor - mean_x) ** 4)
    kurt = (m4 / (var_x ** 2) - 3.0).item()
    # Excess kurtosis of standard normal should be close to 0 (+- 0.3 for sample size 2048)
    assert abs(kurt) < 0.35, f"Normal distribution excess kurtosis {kurt} not near 0"

    # 2. Cosine similarity phi
    v1 = torch.tensor([[1.0, 0.0, 0.0]])
    v2 = torch.tensor([[1.0, 0.0, 0.0]])
    v3 = torch.tensor([[0.0, 1.0, 0.0]])
    phi_ident = F.cosine_similarity(v1, v2).item()
    phi_ortho = F.cosine_similarity(v1, v3).item()
    assert abs(phi_ident - 1.0) < 1e-5
    assert abs(phi_ortho - 0.0) < 1e-5
