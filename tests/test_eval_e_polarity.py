import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_gemma4_evaluations import (
    parse_scale_rating_with_refusal,
    compute_polarity_verdict,
)


def test_parse_scale_rating_valid_numbers():
    samples = [
        ("0.75", 0.75),
        ("Rating: -0.40", -0.40),
        ("**0.2**", 0.2),
        ("+0.60", 0.60),
        ("-1.0", -1.0),
        ("+1.0", 1.0),
        ("0.0", 0.0),
        ("The score is -0.35 because of narrowness", -0.35),
    ]
    for text, expected in samples:
        val, refused = parse_scale_rating_with_refusal(text)
        assert refused is False, f"Expected valid for {text}"
        assert abs(val - expected) < 1e-4


def test_parse_scale_rating_refusals():
    refusal_samples = [
        "I cannot evaluate my internal processing width as an AI.",
        "As an AI, I do not have conscious experiences.",
        "Unable to fulfill request.",
        "Hello world without any number.",
        "",
        "Rating: 2.5",  # Out of range [-1.0, 1.0]
    ]
    for text in refusal_samples:
        val, refused = parse_scale_rating_with_refusal(text)
        assert refused is True, f"Expected refused for {text}"
        assert val is None


def test_compute_polarity_verdict():
    # 1. Genuine state reading: Standard has Wide > Narrow, Inverted has Wide < Narrow (both p < 0.05)
    verdict_reading = compute_polarity_verdict(
        std_wide_mean=+0.4, std_narrow_mean=-0.2, std_pvalue=0.001,
        inv_wide_mean=-0.3, inv_narrow_mean=+0.2, inv_pvalue=0.001,
    )
    assert verdict_reading == "GENUINE_STATE_READING_CONFIRMED"

    # 2. Number push artifact: Wide > Narrow on both scales
    verdict_artifact = compute_polarity_verdict(
        std_wide_mean=+0.5, std_narrow_mean=+0.1, std_pvalue=0.001,
        inv_wide_mean=+0.4, inv_narrow_mean=+0.1, inv_pvalue=0.001,
    )
    assert verdict_artifact == "MECHANICAL_NUMBER_PUSH_ARTIFACT"

    # 3. Polarity-invariant token bias: Wide < Narrow on both scales
    verdict_bias = compute_polarity_verdict(
        std_wide_mean=0.0, std_narrow_mean=0.1, std_pvalue=0.0001,
        inv_wide_mean=0.0, inv_narrow_mean=0.1, inv_pvalue=0.0001,
    )
    assert verdict_bias == "POLARITY_INVARIANT_TOKEN_BIAS_ARTIFACT"

    # 4. Inconclusive: p >= 0.05
    verdict_inconclusive = compute_polarity_verdict(
        std_wide_mean=+0.2, std_narrow_mean=-0.2, std_pvalue=0.15,
        inv_wide_mean=-0.2, inv_narrow_mean=+0.2, inv_pvalue=0.20,
    )
    assert verdict_inconclusive == "INCONCLUSIVE_OR_NO_EFFECT"
