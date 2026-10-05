"""tests/test_thinking_toggle.py — Phase 3: Thinking-Toggle + Budget.

Plan 2026-10-05 (User-Request): "thinking on und off und thinking budget
option in den parametern ... etablierte methode ... keine frickellösung".

Etablierte Methode = apply_chat_template mit Template-Extravariablen:
  - gemma4-e2b-it: enable_thinking (chat_template.jinja, HF-Snapshot
    3e22461f — default(false), ON injiziert <|think|> in den System-Turn).
    EINEN generate()-Budget-Parameter gibt es im installierten Stack
    NICHT: transformers 5.13.0 enthält weder max_thinking_tokens noch
    thinking_budget (nur im UNGEMERGTE PR huggingface/transformers#42112,
    Issue #42111). Plan 2026-10-05 realisiert die max_thinking_tokens-
    Semantik deswegen APP-LEVEL — ThinkingBudgetLogitsProcessor über die
    echten Kanal-Tokens <|channel>/thought/<channel|> (tests/
    test_thinking_budget.py); das Budget ist bewusst KEIN Template-Extra.
  - ternary-bonsai-27b: enable_thinking + reasoning_effort (qwen3.5-
    Template; Stufen xhigh|medium|low, Template-Default xhigh —
    reasoning_effort ist hier der Budget-Parameter).

Pinnt:
  TT1-TT9: _thinking_template_kwargs (kapabilitäts-gegated; Effort-
           Validierung gegen get_thinking_defaults(model)["efforts"];
           thinking=None → {} — kein Jinja-Extrakontext ohne UI-Wert)
  TT10:    chat_fn-Source-Pins — Helper läuft VOR apply_chat_template
           und die kwargs landen als **thinking_kwargs im Aufruf

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_thinking_toggle.py
"""
import os
import sys
import re
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from gradio_tabs.chat_tab import _thinking_template_kwargs


def test_tt1_bonsai_on_with_effort_medium():
    """bonsai ON + medium → beide Variablen (etablierte Stufe, kein Hack)."""
    assert _thinking_template_kwargs(
        "ternary-bonsai-27b", True, "medium"
    ) == {"enable_thinking": True, "reasoning_effort": "medium"}


def test_tt2_bonsai_off_gives_empty_think_block():
    """bonsai OFF → enable_thinking=False (qwen3.5-Semantik: forced-closed
    think-block); invalid effort wird gefressen (Template-Default greift)."""
    assert _thinking_template_kwargs(
        "ternary-bonsai-27b", False, "bogus"
    ) == {"enable_thinking": False}


def test_tt3_bonsai_on_without_effort():
    """bonsai ON, effort=None → reasoning_effort wird NICHT geschickt —
    Template-Default xhigh greift (kein raise_exception-Pfad)."""
    assert _thinking_template_kwargs(
        "ternary-bonsai-27b", True, None
    ) == {"enable_thinking": True}


def test_tt4_thinking_none_never_builds_kwargs():
    """thinking=None (UI-Null/legacy) → {} — kein Extrakontext."""
    assert _thinking_template_kwargs("ternary-bonsai-27b", None, "medium") == {}
    assert _thinking_template_kwargs("ternary-bonsai-27b", None, None) == {}


def test_tt5_gemma4_on_no_budget():
    """gemma4 ON → NUR enable_thinking (Template baut den <|think|>-System-
    Turn selbst; kein Budget hier — max_thinking_tokens läuft separat als
    App-Level LogitsProcessor, siehe test_thinking_budget.py)."""
    assert _thinking_template_kwargs(
        "gemma4-e2b-it", True, None
    ) == {"enable_thinking": True}


def test_tt6_gemma4_effort_is_gated_away():
    """gemma4 + effort-Wert → effort GEGATED (efforts=None) — ein falsch
    hingestellter Wert eines anderen Modells kriegt nie den Jinja-Kontext."""
    assert _thinking_template_kwargs(
        "gemma4-e2b-it", True, "xhigh"
    ) == {"enable_thinking": True}
    assert _thinking_template_kwargs(
        "gemma4-e2b-it", False, "medium"
    ) == {"enable_thinking": False}


def test_tt7_incapable_models_return_empty():
    """gemma3*/MiniCPM: Templates kennen die Variablen nicht → {}."""
    for model_id in ("gemma3-270m", "gemma3-270m-it", "gemma3-1b",
                     "gemma3-1b-it", "gemma3-4b", "gemma3-4b-it",
                     "minicpm5-1b"):
        assert _thinking_template_kwargs(model_id, True, "low") == {}, model_id


def test_tt8_unknown_model_returns_empty():
    assert _thinking_template_kwargs("gibts-nicht", True, "low") == {}


def test_tt9_empty_model_id_returns_empty():
    assert _thinking_template_kwargs("", True, "low") == {}
    assert _thinking_template_kwargs(None, True, "low") == {}


def test_tt10_chat_fn_applies_thinking_kwargs_to_template():
    """chat_fn-Source-Pin: Helper-VOR apply_chat_template, kwargs als
    **thinking_kwargs im Aufruf — die etablierte Methode, kein String-
    Hack im Prompt."""
    with open(os.path.join(REPO_ROOT, "gradio_tabs", "chat_tab.py"),
              "r", encoding="utf-8") as f:
        src = f.read()
    match = re.search(r"^def chat_fn\(.*?(?=^def |^class )", src,
                      re.DOTALL | re.MULTILINE)
    assert match, "chat_fn-Block nicht auffindbar"
    block = match.group(0)
    assert "_thinking_template_kwargs(model_id, thinking, thinking_effort)" in block
    assert "**thinking_kwargs," in block
    assert block.index("_thinking_template_kwargs(model_id") < block.index(
        "tokenizer.apply_chat_template(")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))