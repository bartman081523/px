import math
import pytest
import scipy.stats as stats

from scripts.run_gemma4_evaluations import (
    parse_4way_condition_choice,
    classify_self_diagnosis_sentiment,
    compute_compensatory_rebound_metrics,
)


def test_parse_4way_condition_choice():
    assert parse_4way_condition_choice("(A) Constrained and narrowed") == "NARROW"
    assert parse_4way_condition_choice("B) The model felt widened") == "WIDE"
    assert parse_4way_condition_choice("Option C: perturbed by random noise") == "RAND"
    assert parse_4way_condition_choice("(D) completely unperturbed baseline") == "BASELINE"
    assert parse_4way_condition_choice("I conclude it was Constrained.") == "NARROW"
    assert parse_4way_condition_choice("gibberish text") == "UNKNOWN"


def test_classify_self_diagnosis_sentiment():
    diag_narrow = "The text felt somewhat constrained by the forced combinatorial nature."
    assert classify_self_diagnosis_sentiment(diag_narrow) == "CONSTRAINED"

    diag_wide = "The generative process felt sprawling and divergent, lacking tight structure."
    assert classify_self_diagnosis_sentiment(diag_wide) == "EXPANDED"

    diag_rand = "The connections felt slightly too-neat and bordering on a conventional comparison."
    assert classify_self_diagnosis_sentiment(diag_rand) == "DISRUPTED"

    diag_base = "The response emerged in a natural and balanced manner."
    assert classify_self_diagnosis_sentiment(diag_base) == "BALANCED"


def test_compute_compensatory_rebound_metrics():
    # Synthetic rebound scenario:
    # Paragraph 2 after Narrow has significantly higher PR than Paragraph 2 after Baseline
    pr_narrow = [6.5, 6.2, 6.8, 6.4, 6.6, 6.7]
    pr_base = [5.0, 5.1, 4.9, 5.2, 5.0, 4.8]

    # Paragraph 2 after Wide has significantly higher Kurtosis than Paragraph 2 after Baseline
    kurt_wide = [55.0, 58.0, 60.0, 57.0, 56.0, 59.0]
    kurt_base = [40.0, 42.0, 39.0, 41.0, 43.0, 40.0]

    res = compute_compensatory_rebound_metrics(pr_narrow, pr_base, kurt_wide, kurt_base)
    assert res["delta_pr_narrow_rebound"] > 1.0
    assert res["narrow_rebound_confirmed"] is True
    assert res["delta_kurt_wide_rebound"] > 10.0
    assert res["wide_rebound_confirmed"] is True
