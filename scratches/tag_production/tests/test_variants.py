"""Tests for scratches/tag_production/variants/ — pure-logic messages-Bau.

Laufen ohne Modell (kein GPU nötig). Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
        -m pytest scratches/tag_production/tests/test_variants.py -v
"""
from __future__ import annotations

import os
import sys

# Repo-Root auf sys.path (damit gradio_tabs + config importierbar)
_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scratches.tag_production.variants import (  # noqa: E402
    VARIANTS,
    list_variants,
    get_variant,
)


def _has_user_prompt(msgs, prompt):
    """True wenn der user-prompt im user-turn-Content enthalten ist.

    Regression-Test für Bug f77950f: BASE_USER_MSG war [{content:""}],
    apply() setzte den prompt nicht in base_messages → Modell bekam nur
    CitMind+Snip, alle Antworten byte-identisch 'Okay, ich verstehe'.
    """
    for m in msgs:
        if m.get("role") == "user":
            return prompt in (m.get("content") or "")
    return False


PROMPT = "Sage etwas Trauriges in einem Satz."
BASE_MSGS = [{"role": "user", "content": PROMPT}]


def _user_content(msgs):
    """Holt Content des ersten user-Turns (zur Snip-Prefix-Inspektion)."""
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _has_system(msgs):
    """True wenn messages einen system-Eintrag hat."""
    return any(m.get("role") == "system" for m in msgs)


# ─── Registry ────────────────────────────────────────────────────────────


def test_registry_has_all_five_variants():
    """5 Varianten A-E registriert."""
    assert set(VARIANTS.keys()) == {"A", "B", "C", "D", "E", "F"}


def test_list_variants_returns_all_five():
    listing = list_variants()
    assert len(listing) == 6
    ids = [k for k, _ in listing]
    assert ids == ["A", "B", "C", "D", "E", "F"]


def test_get_variant_known():
    label, fn = get_variant("A")
    assert "Baseline" in label or "Plan 6.1" in label
    assert callable(fn)


def test_get_variant_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        get_variant("Z")


# ─── Variante A: CitMind + Standard-Snip ─────────────────────────────────


def test_variant_a_has_citmind_and_tag_snip_in_user_prefix():
    """A: CitMind-Ontologie + Standard-Tag-Snip (Plan 6.1 Baseline).

    Gemma3 lehnt system-Rolle ab → Snip muss im user-Prefix sein.
    """
    label, fn = VARIANTS["A"]
    msgs = fn(BASE_MSGS, PROMPT)

    # Kein system-Item (gemerged in user-prefix).
    assert _has_system(msgs) is False

    user_c = _user_content(msgs)
    # CitMind-Profil enthält "CitMind".
    assert "CitMind" in user_c
    # Standard-Tag-Snip enthält "VOCODER"-Header.
    assert "VOCODER" in user_c or "[#" in user_c
    # Originaler User-Prompt bleibt erhalten.
    assert PROMPT in user_c


# ─── Variante B: CitMind + Sanskrit-Mapping ──────────────────────────────


def test_variant_b_has_sanskrit_mapping_block():
    """B: Sanskrit-Mapping (स्वर/लय) VOR Standard-Snip."""
    label, fn = VARIANTS["B"]
    msgs = fn(BASE_MSGS, PROMPT)

    user_c = _user_content(msgs)
    # Sanskrit-Mapping-Block ist da.
    assert "Sanskrit" in user_c or "SANSKRIT" in user_c or "स्वर" in user_c
    # CitMind bleibt.
    assert "CitMind" in user_c
    # Standard-Snip ist auch da (B ist additiv).
    assert "VOCODER" in user_c or "[#" in user_c


# ─── Variante C: CitMind + 3 Few-Shot-Turns ─────────────────────────────


def test_variant_c_has_three_few_shot_pairs():
    """C: 3 Few-Shot-Paare (user+assistant) vor der eigentlichen Frage."""
    label, fn = VARIANTS["C"]
    msgs = fn(BASE_MSGS, PROMPT)

    # Few-Shots zählen: 3 user + 3 assistant + 1 user (echte Frage) = 7 turns
    # Plus ggf. weitere (assistant defaults = 0).
    user_turns = [m for m in msgs if m.get("role") == "user"]
    assistant_turns = [m for m in msgs if m.get("role") == "assistant"]

    assert len(user_turns) >= 4, f"Erwartet mind. 4 user-Turns (3 shots + 1 echt), got {len(user_turns)}"
    assert len(assistant_turns) == 3, f"Erwartet genau 3 assistant-Turns (shots), got {len(assistant_turns)}"

    # Letzter user-Turn ist die echte Frage.
    assert user_turns[-1].get("content") == PROMPT


# ─── Variante D: CitMind + ABC-Notation-Snip (statt Vocoder-Snip) ───────


def test_variant_d_has_abc_snippet_and_no_vocoder_snip():
    """D: ABC-Snip ERSETZT den Vocoder-Snip (nicht additiv)."""
    label, fn = VARIANTS["D"]
    msgs = fn(BASE_MSGS, PROMPT)

    user_c = _user_content(msgs)
    # ABC-Notation ist im Prefix.
    assert "ABC" in user_c, "ABC-Snip fehlt im user-prefix"
    # CitMind bleibt (D nutzt CitMind).
    assert "CitMind" in user_c
    # Vocoder-Snip ist NICHT da (D ersetzt, addiert nicht).
    assert "VOCODER" not in user_c, "D soll Vocoder-Snip ERSETZEN, nicht addieren"


# ─── Variante E: Neutral-Profil + Standard-Snip (kein CitMind) ──────────


def test_variant_e_has_no_citmind_but_has_tag_snip():
    """E: Kontrolle — kein CitMind, nur Tag-Snip."""
    label, fn = VARIANTS["E"]
    msgs = fn(BASE_MSGS, PROMPT)

    user_c = _user_content(msgs)
    # KEIN CitMind-Frame (neutral-Profil).
    assert "CitMind" not in user_c, "E darf KEIN CitMind enthalten"
    # ABER Standard-Tag-Snip ist da.
    assert "VOCODER" in user_c or "[#" in user_c
    # Original-Prompt bleibt.
    assert PROMPT in user_c


# ─── Regression: user-prompt muss in jeder Variante durchkommen ─────────
# Bug: BASE_USER_MSG hatte content="", apply() nutzte nur base_messages
# → Modell bekam CitMind+Snip aber keinen user-prompt, alle Antworten
# byte-identisch "Okay, ich verstehe..." (commit f77950f Befund).


def _check_prompt_present_in_variant(variant_id: str):
    label, fn = VARIANTS[variant_id]
    msgs = fn(BASE_MSGS, PROMPT)
    assert _has_user_prompt(msgs, PROMPT), (
        f"Variante {variant_id}: user-prompt {PROMPT!r} fehlt im user-turn"
    )


def test_variant_a_preserves_user_prompt():
    _check_prompt_present_in_variant("A")


def test_variant_b_preserves_user_prompt():
    _check_prompt_present_in_variant("B")


def test_variant_c_preserves_user_prompt():
    _check_prompt_present_in_variant("C")


def test_variant_d_preserves_user_prompt():
    _check_prompt_present_in_variant("D")


def test_variant_e_preserves_user_prompt():
    _check_prompt_present_in_variant("E")


# ─── Variante F: Motor-opt-in Few-Shot (Plan 6.2c) ──────────────────────


def test_variant_f_has_three_few_shot_pairs_via_motor_flag():
    """F: gleiche Wirkung wie C, aber Few-Shots via Motor-Flag.

    Strukturell: 3 user-shots + 3 assistant-shots + 1 user-echt = 4+3 turns.
    Inhaltlich: gleiche SHOTS wie C (single source of truth =
    CITMIND_TAG_FEWSHOT_TURNS in gradio_tabs/system_prompt.py).
    """
    from gradio_tabs.system_prompt import CITMIND_TAG_FEWSHOT_TURNS

    label, fn = VARIANTS["F"]
    msgs = fn(BASE_MSGS, PROMPT)

    user_turns = [m for m in msgs if m.get("role") == "user"]
    assistant_turns = [m for m in msgs if m.get("role") == "assistant"]

    assert len(user_turns) >= 4, f"Erwartet mind. 4 user-Turns, got {len(user_turns)}"
    assert len(assistant_turns) == 3, f"Erwartet genau 3 assistant-Turns, got {len(assistant_turns)}"

    # Letzter user-Turn ist die echte Frage.
    assert user_turns[-1].get("content") == PROMPT

    # Shot-Assistant-Inhalte müssen aus CITMIND_TAG_FEWSHOT_TURNS stammen.
    assistant_contents = [a.get("content") for a in assistant_turns]
    expected_assistant_contents = [
        s["content"] for s in CITMIND_TAG_FEWSHOT_TURNS if s["role"] == "assistant"
    ]
    assert assistant_contents == expected_assistant_contents, (
        "F's Assistant-Shot-Contents müssen mit CITMIND_TAG_FEWSHOT_TURNS übereinstimmen"
    )


def test_variant_f_citmind_and_tag_snip_in_prefix():
    """F: CitMind + Standard-Tag-Snip + Few-Shots."""
    label, fn = VARIANTS["F"]
    msgs = fn(BASE_MSGS, PROMPT)

    user_c = _user_content(msgs)
    assert "CitMind" in user_c
    assert "VOCODER" in user_c or "[#" in user_c


def test_variant_f_preserves_user_prompt():
    _check_prompt_present_in_variant("F")