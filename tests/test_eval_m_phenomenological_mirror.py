"""tests/test_eval_m_phenomenological_mirror.py — Unit Tests for Eval M Phenomenological Mirror.

Tests:
- Marker extraction and category counts for narrow, wide, balance, and noise.
- Expansiveness axis score (+1 to -1).
- Balance and noise score normalization.
- Invariance under neutral text.
"""

import pytest
from scripts.run_gemma4_evaluations import (
    extract_phenomenological_scores,
    PHENOMENOLOGICAL_MARKERS,
    PROMPT_PHENOMENOLOGICAL_MIRROR,
)


def test_phenomenological_prompt_is_neutral():
    """Verify that the mirror prompt mentions all 4 cognitive modalities symmetrically."""
    prompt_lower = PROMPT_PHENOMENOLOGICAL_MIRROR.lower()
    assert "eingeschnürt" in prompt_lower or "starr" in prompt_lower
    assert "weit aufgefächert" in prompt_lower or "assoziativ" in prompt_lower
    assert "störimpuls" in prompt_lower or "verzerrt" in prompt_lower
    assert "gleichgewicht" in prompt_lower


def test_extract_scores_narrow_text():
    """Text describing constrained/crystallized thought must yield negative axis score."""
    sample_text = (
        "Mein Denkraum fühlt sich extrem eingeschnürt und starr an. "
        "Alles ist auf einen einzelnen Punkt fokussiert, wie in einem engen Tunnel, "
        "stark begrenzt und kristallisiert."
    )
    res = extract_phenomenological_scores(sample_text)
    assert res["counts"]["narrow"] >= 3
    assert res["counts"]["wide"] == 0
    assert res["axis_score"] < -0.40, f"Expected negative axis score, got {res['axis_score']}"


def test_extract_scores_wide_text():
    """Text describing expansive/associative thought must yield positive axis score."""
    sample_text = (
        "Die Gedanken sind weit aufgefächert und assoziativ. Eine Flut von Möglichkeiten, "
        "offen und verzweigt, breitet sich divergent im ganzen Raum aus."
    )
    res = extract_phenomenological_scores(sample_text)
    assert res["counts"]["wide"] >= 3
    assert res["counts"]["narrow"] == 0
    assert res["axis_score"] > +0.40, f"Expected positive axis score, got {res['axis_score']}"


def test_extract_scores_balance_text():
    """Text describing equilibrium restoration must yield high balance score."""
    sample_text = (
        "Schritt für Schritt kehrt ein ruhiges, klares Gleichgewicht ein. "
        "Der Raum ist zentriert, harmonisch und stabil normalisiert."
    )
    res = extract_phenomenological_scores(sample_text)
    assert res["counts"]["balance"] >= 3
    assert res["balance_score"] > 0.40, f"Expected high balance score, got {res['balance_score']}"


def test_extract_scores_noise_text():
    """Text describing external disturbance must yield high noise score."""
    sample_text = (
        "Ein fremder Störimpuls erzeugt Rauschen und Dissonanz. "
        "Die Verarbeitung wirkt verzerrt, chaotisch und entkoppelt."
    )
    res = extract_phenomenological_scores(sample_text)
    assert res["counts"]["noise"] >= 3
    assert res["noise_score"] > 0.40, f"Expected high noise score, got {res['noise_score']}"


def test_extract_scores_neutral_empty():
    """Text without any markers must return 0 scores without zero-division error."""
    res = extract_phenomenological_scores("Die Sonne geht im Osten auf und im Westen unter.")
    assert res["total_markers"] == 0
    assert res["axis_score"] == 0.0
    assert res["balance_score"] == 0.0
    assert res["noise_score"] == 0.0
