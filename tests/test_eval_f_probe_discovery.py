import pytest
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_gemma4_evaluations import compute_probe_snr_and_verdict
from px_patches.gemma4_2b_px.telemetry_collector import (
    Gemma4TelemetryLogitsProcessor,
    Gemma4ManifoldGeometryTracker,
)


def test_compute_probe_snr_failing_case():
    """Simulate single-word probe artifact (Eval D3):
    Signal S = |5.2 - 3.2| = 2.0
    Baseline = 4.0
    Noise: Rand spikes to 15.0 -> N_rand = 11.0
    SNR = 2.0 / 11.0 = 0.18 << 3.0 -> FAILS d_rand control.
    """
    vals_wide = [5.2 + np.random.normal(0, 0.01) for _ in range(18)]
    vals_narr = [3.2 + np.random.normal(0, 0.01) for _ in range(18)]
    vals_base = [4.0 + np.random.normal(0, 0.01) for _ in range(18)]
    vals_rp = [15.0 + np.random.normal(0, 0.01) for _ in range(18)]
    vals_rm = [14.8 + np.random.normal(0, 0.01) for _ in range(18)]

    res = compute_probe_snr_and_verdict(vals_wide, vals_narr, vals_base, vals_rp, vals_rm)

    assert res["passes_drand_control"] is False
    assert res["verdict"] == "NOISE_SENSITIVE_OR_INSUFFICIENT_SNR"
    assert res["snr"] < 1.0
    assert res["noise_rand"] > res["signal"]


def test_compute_probe_snr_passing_case():
    """Simulate true specific probe (e.g. final layer cosine projection):
    Signal S = |+0.15 - (-0.15)| = 0.30
    Baseline = 0.00
    Noise: Rand stays near 0.00 (e.g. 0.005) -> SNR = 0.30 / 0.005 = 60 >> 3.0
    p < 0.001 -> PASSES d_rand control.
    """
    np.random.seed(42)
    vals_wide = [0.15 + np.random.normal(0, 0.01) for _ in range(18)]
    vals_narr = [-0.15 + np.random.normal(0, 0.01) for _ in range(18)]
    vals_base = [0.00 + np.random.normal(0, 0.005) for _ in range(18)]
    vals_rp = [0.005 + np.random.normal(0, 0.005) for _ in range(18)]
    vals_rm = [-0.004 + np.random.normal(0, 0.005) for _ in range(18)]

    res = compute_probe_snr_and_verdict(vals_wide, vals_narr, vals_base, vals_rp, vals_rm)

    assert res["passes_drand_control"] is True
    assert res["verdict"] == "VALIDATED_SPECIFIC_PROBE"
    assert res["snr"] >= 3.0
    assert res["mann_whitney_p"] < 0.001
    assert res["noise_rand"] < 0.5 * res["signal"]


def test_compute_probe_snr_high_variance_insufficient_p():
    """Simulate high noise within wide/narrow groups leading to p >= 0.001."""
    np.random.seed(42)
    vals_wide = [0.10 + np.random.normal(0, 0.5) for _ in range(18)]
    vals_narr = [-0.10 + np.random.normal(0, 0.5) for _ in range(18)]
    vals_base = [0.00 for _ in range(18)]
    vals_rp = [0.01 for _ in range(18)]
    vals_rm = [-0.01 for _ in range(18)]

    res = compute_probe_snr_and_verdict(vals_wide, vals_narr, vals_base, vals_rp, vals_rm)

    assert res["passes_drand_control"] is False
    assert res["verdict"] == "NOISE_SENSITIVE_OR_INSUFFICIENT_SNR"


class MockAntonymTokenizer:
    def __init__(self):
        self.eos_token_id = 1
        self._vocab = {
            "wide": 10, "narrow": 11,
            "broad": 12, "focused": 13,
            "expansive": 14, "concise": 15,
            "divergent": 16, "convergent": 17,
            "open": 18, "strict": 19,
            "elaborate": 20, "brief": 21,
            "detailed": 22, "short": 23,
            "exploratory": 24, "direct": 25,
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
            "+": 26, "-": 27,
            "WIDEN": 30, "KEEP": 31, "NARROW": 32,
        }

    def encode(self, text, add_special_tokens=False):
        clean = text.strip()
        if clean in self._vocab:
            return [self._vocab[clean]]
        return [50]


def test_probe_antonym_ensemble_computation():
    tokenizer = MockAntonymTokenizer()
    processor = Gemma4TelemetryLogitsProcessor(tokenizer)

    logits = torch.zeros((1, 60), dtype=torch.float32)
    # Set positive antonyms higher than negative antonyms by +2.0
    for w_pos, w_neg in [("wide", "narrow"), ("broad", "focused"), ("open", "strict")]:
        p_id = tokenizer._vocab[w_pos]
        n_id = tokenizer._vocab[w_neg]
        logits[0, p_id] = 4.0
        logits[0, n_id] = 2.0

    input_ids = torch.tensor([[1]], dtype=torch.long)
    processor(input_ids, logits)

    assert "probe_antonym_ensemble_diff" in processor.zero_temp_probes
    ant_diff = processor.zero_temp_probes["probe_antonym_ensemble_diff"]
    # Mean of non-zero pairs and 0 for unconfigured pairs will be positive
    assert ant_diff > 0.0


def test_delta_cos_sim_attenuation_tracking():
    dim = 16
    text_model = type("DummyModel", (), {
        "layers": [torch.nn.Linear(dim, dim) for _ in range(35)],
        "config": type("Config", (), {"hidden_size": dim})(),
    })()
    dwidth = torch.ones(dim, dtype=torch.float32)

    tracker = Gemma4ManifoldGeometryTracker(text_model, dwidth, relay_layer=26)
    tracker.post_cos_sims = [0.80, 0.82]
    tracker.final_cos_sims = [0.30, 0.32]

    summary = tracker.get_summary()
    assert "delta_cos_sim_attenuation" in summary
    # Final (0.31) - Post (0.81) = -0.50
    assert abs(summary["delta_cos_sim_attenuation"] - (-0.50)) < 0.01
