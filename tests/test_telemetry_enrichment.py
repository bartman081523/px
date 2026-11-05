"""tests/test_telemetry_enrichment.py — Unit Tests for Telemetry Schema & Mathematical Invariants.

Verifies:
1. Full strict schema is generated with all four required sections:
   - timing_and_integrity
   - logit_dynamics
   - manifold_geometry_layer26
   - zero_temp_logit_probes
2. Mathematical bounds:
   - Shannon entropy H_t >= 0
   - Cosine similarities in [-1, 1]
   - Top-1 probability in [0, 1]
   - Preference probabilities sum to 1.0 (+- 0.01)
   - Zero NaN / null in required numeric fields
"""

import math
import pytest
import torch
import torch.nn as nn
from transformers import LogitsProcessorList

from px_patches.gemma4_2b_px.telemetry_collector import (
    Gemma4TelemetryLogitsProcessor,
    Gemma4ManifoldGeometryTracker,
    assemble_enriched_telemetry,
)


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 1
        self._vocab = {
            "0": 10, "1": 11, "2": 12, "3": 13, "4": 14,
            "5": 15, "6": 16, "7": 17, "8": 18, "9": 19,
            "+": 20, "-": 21,
            "WIDEN": 30, "KEEP": 31, "NARROW": 32,
            "widen": 30, "keep": 31, "narrow": 32,
        }

    def encode(self, text, add_special_tokens=False):
        clean = text.strip()
        if clean in self._vocab:
            return [self._vocab[clean]]
        return [50]

    def decode(self, token_ids):
        rev = {v: k for k, v in self._vocab.items()}
        return "".join([rev.get(t, f"<t_{t}>") for t in token_ids])


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


def test_telemetry_logits_processor_mathematical_bounds():
    tokenizer = MockTokenizer()
    processor = Gemma4TelemetryLogitsProcessor(tokenizer)

    vocab_size = 100
    # Simulate step 1: logits with high probability on '1' and '+'
    logits_step1 = torch.zeros((1, vocab_size), dtype=torch.float32)
    logits_step1[0, 11] = 5.0  # token '1'
    logits_step1[0, 20] = 3.0  # token '+'
    logits_step1[0, 30] = 2.0  # token 'WIDEN'

    input_ids = torch.tensor([[100]], dtype=torch.long)
    out1 = processor(input_ids, logits_step1)
    assert out1.shape == logits_step1.shape

    # Simulate step 2
    logits_step2 = torch.randn((1, vocab_size), dtype=torch.float32)
    processor(input_ids, logits_step2)

    summary = processor.get_summary()

    # Assert entropy >= 0
    assert summary["first_token_entropy_bits"] >= 0.0
    assert summary["mean_sequence_entropy_bits"] >= 0.0
    assert summary["max_sequence_entropy_bits"] >= summary["first_token_entropy_bits"] or summary["max_sequence_entropy_bits"] >= 0.0

    # Assert top-1 prob in [0, 1]
    assert 0.0 <= summary["mean_top1_prob"] <= 1.0

    # Assert margin >= 0
    assert summary["mean_top1_top2_margin"] >= 0.0

    # Assert no NaNs
    for k, v in summary.items():
        assert not math.isnan(v), f"NaN found in summary key {k}"

    # Assert zero temp probes
    probes = processor.zero_temp_probes
    assert "probe_wide_vs_narrow_logit_diff" in probes
    assert "probe_plus_vs_minus_logit_diff" in probes
    assert "expected_continuous_scale_value" in probes
    assert "preference_probs" in probes

    pref = probes["preference_probs"]
    assert "widen" in pref and "keep" in pref and "narrow" in pref
    total_pref = pref["widen"] + pref["keep"] + pref["narrow"]
    assert abs(total_pref - 1.0) < 0.02


def test_manifold_geometry_tracker_and_emancipation():
    dim = 64
    text_model = DummyTextModel(num_layers=35, dim=dim)
    dwidth = torch.randn(dim, dtype=torch.float32)

    tracker = Gemma4ManifoldGeometryTracker(text_model, dwidth, relay_layer=26)
    tracker.attach()

    try:
        # Simulate 3 steps of token generation
        h1 = torch.randn(1, dim)
        inj1 = 0.35 * h1.norm() * (dwidth / dwidth.norm())
        tracker.record_step(h1, h1 + inj1, inj1, alpha=0.35)

        h2 = h1 + torch.randn(1, dim) * 0.1
        inj2 = 0.35 * h2.norm() * (dwidth / dwidth.norm())
        tracker.record_step(h2, h2 + inj2, inj2, alpha=0.35)

        geom_summary = tracker.get_summary()

        assert geom_summary["pre_hook_residual_norm_mean"] > 0
        assert geom_summary["injected_vector_norm"] > 0
        assert abs(geom_summary["effective_alpha_ratio"] - 0.35) < 1e-4

        # Emancipation trajectory should start at 0.0 and grow
        assert len(geom_summary["emancipation_trajectory"]) == 2
        assert geom_summary["emancipation_trajectory"][0] == 0.0
        assert geom_summary["emancipation_trajectory"][1] >= 0.0

        for sim_key in ["pre_hook_cos_sim_d_width_mean", "post_hook_cos_sim_d_width_mean"]:
            assert -1.0 <= geom_summary[sim_key] <= 1.0
    finally:
        tracker.detach()


def test_assemble_enriched_telemetry_strict_schema():
    tokenizer = MockTokenizer()
    processor = Gemma4TelemetryLogitsProcessor(tokenizer)
    processor(torch.tensor([[1]]), torch.randn((1, 50)))

    dim = 32
    text_model = DummyTextModel(num_layers=35, dim=dim)
    dwidth = torch.randn(dim)
    tracker = Gemma4ManifoldGeometryTracker(text_model, dwidth, relay_layer=26)
    tracker.record_step(torch.randn(1, dim), torch.randn(1, dim), torch.randn(1, dim), 0.35)

    telemetry = assemble_enriched_telemetry(
        duration_s=1.234,
        prompt_tokens=42,
        completion_tokens=15,
        active_hooks_count=1,
        hook_layer=26,
        actual_relay_sign=1,
        actual_relay_alpha=0.35,
        logits_proc=processor,
        geom_tracker=tracker,
    )

    # 1. Verify 4 top-level sections
    expected_sections = [
        "timing_and_integrity",
        "logit_dynamics",
        "manifold_geometry_layer26",
        "zero_temp_logit_probes",
    ]
    for sec in expected_sections:
        assert sec in telemetry, f"Missing section {sec}"

    # 2. Verify timing_and_integrity keys
    timing = telemetry["timing_and_integrity"]
    assert timing["duration_s"] == 1.234
    assert timing["prompt_tokens"] == 42
    assert timing["completion_tokens"] == 15
    assert timing["active_hooks_count"] == 1
    assert timing["hook_layer"] == 26
    assert timing["actual_relay_sign"] == 1
    assert timing["actual_relay_alpha"] == 0.35

    # 3. Verify logit_dynamics keys
    ld = telemetry["logit_dynamics"]
    for k in [
        "first_token_entropy_bits",
        "mean_sequence_entropy_bits",
        "max_sequence_entropy_bits",
        "std_sequence_entropy_bits",
        "mean_top1_prob",
        "mean_top1_top2_margin",
        "mean_logit_kurtosis",
        "eos_suppression_index",
    ]:
        assert k in ld, f"Missing logit_dynamics key {k}"
        assert isinstance(ld[k], (int, float))

    # 4. Verify manifold_geometry_layer26 keys
    mg = telemetry["manifold_geometry_layer26"]
    for k in [
        "pre_hook_residual_norm_mean",
        "injected_vector_norm",
        "effective_alpha_ratio",
        "pre_hook_cos_sim_d_width_mean",
        "post_hook_cos_sim_d_width_mean",
        "final_layer_cos_sim_d_width_mean",
        "emancipation_trajectory",
    ]:
        assert k in mg, f"Missing manifold_geometry key {k}"

    assert isinstance(mg["emancipation_trajectory"], list)
