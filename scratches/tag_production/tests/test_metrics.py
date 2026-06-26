"""Tests for scratches/tag_production/metrics.py — pure-logic Tag-Compliance.

Laufen ohne Modell (kein GPU nötig). Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
        -m pytest scratches/tag_production/tests/test_metrics.py -v
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

from scratches.tag_production.metrics import (  # noqa: E402
    count_words,
    classify_response_tags,
    compute_per_response_metrics,
    aggregate_run_metrics,
)


# ─── 1. count_words ────────────────────────────────────────────────────────


def test_count_words_empty_returns_one():
    """count_words(None/'') = 1 (verhindert Div/0)."""
    assert count_words("") == 1
    assert count_words(None) == 1  # type: ignore[arg-type]


def test_count_words_normal():
    assert count_words("Hallo Welt") == 2
    assert count_words("  spaced   out  ") == 2
    assert count_words("eins") == 1


# ─── 2. classify_response_tags ────────────────────────────────────────────


def test_classify_empty_text_zero_tags():
    out = classify_response_tags("")
    assert out["total_tags"] == 0
    assert out["has_any_tag"] is False
    assert out["has_note"] is False
    assert out["density_per_100w"] == 0.0
    assert out["clean_text"] == ""
    assert out["raw_tags"] == []
    assert out["syntax_valid"] is True
    assert out["density_warning"] is None


def test_classify_note_only():
    out = classify_response_tags("Ich [#A4] singe [#C#3] heute.")
    assert out["total_tags"] == 2
    assert out["note"] == 2
    assert out["has_note"] is True
    assert out["has_dynamic"] is False
    assert out["raw_tags"][0]["value"] == "A4"
    assert out["raw_tags"][1]["value"] == "C#3"
    assert "[#A4]" not in out["clean_text"]
    # clean_text behält Wortreihenfolge (Spacing vom Original, nicht collapsed):
    assert "Ich" in out["clean_text"]
    assert "singe" in out["clean_text"]
    assert "heute." in out["clean_text"]


def test_classify_dynamic_only():
    out = classify_response_tags("[#WHISPER] Geheimnis.")
    assert out["total_tags"] == 1
    assert out["dynamic"] == 1
    assert out["has_dynamic"] is True
    assert out["clean_text"] == " Geheimnis."


def test_classify_affect_only():
    out = classify_response_tags("[#HAPPY] [#EXCITED] Wow!")
    assert out["total_tags"] == 2
    assert out["affect"] == 2
    assert out["has_affect"] is True


def test_classify_pause_only():
    out = classify_response_tags("Hallo [#PAUSE 0.5s] Welt.")
    assert out["total_tags"] == 1
    assert out["pause"] == 1
    assert out["has_pause"] is True
    assert out["raw_tags"][0]["value"] == "0.5"


def test_classify_mixed_kinds():
    out = classify_response_tags(
        "[#CALM] [#A3]Ich [#PAUSE 0.3s] denke [#WHISPER]leise."
    )
    assert out["total_tags"] == 4
    assert out["affect"] == 1   # CALM
    assert out["note"] == 1     # A3
    assert out["pause"] == 1    # PAUSE 0.3s
    assert out["dynamic"] == 1  # WHISPER
    assert out["has_affect"] is True
    assert out["has_note"] is True
    assert out["has_pause"] is True
    assert out["has_dynamic"] is True


def test_classify_unknown_tag_stripped_not_listed():
    """[#BANANA] wird gestrippt, aber NICHT in raw_tags aufgenommen."""
    out = classify_response_tags("Hallo [#BANANA] Welt.")
    assert out["total_tags"] == 0
    assert out["raw_tags"] == []
    assert "[#BANANA]" not in out["clean_text"]
    assert out["clean_text"] == "Hallo  Welt."


def test_classify_density_warning_at_30():
    """Bei >30 Tags/100 Wörter feuert tag_density_warning."""
    # 3 Tags in 8 Wörtern = 37.5/100 → warnt
    text = "[#A4] eins [#B4] zwei [#C4] drei [#D4] vier [#E4] fünf."
    out = classify_response_tags(text)
    assert out["density_warning"] is not None
    assert "WARN" in out["density_warning"]


def test_classify_density_warning_none_normal():
    out = classify_response_tags("Hallo Welt. Wie geht es dir?")
    assert out["density_warning"] is None


def test_classify_raw_tags_preserve_order():
    out = classify_response_tags("[#A4] erst [#B4] zweit [#C4] dritt")
    values = [t["value"] for t in out["raw_tags"]]
    assert values == ["A4", "B4", "C4"]
    offsets = [t["offset"] for t in out["raw_tags"]]
    # Sortiert nach Position im Original
    assert offsets == sorted(offsets)


def test_classify_none_input_safe():
    out = classify_response_tags(None)  # type: ignore[arg-type]
    assert out["total_tags"] == 0
    assert out["syntax_valid"] is True


# ─── 3. compute_per_response_metrics ──────────────────────────────────────


def test_per_response_bundles_correctly():
    out = compute_per_response_metrics("[#A4] Hallo.", prompt_id="p01")
    assert out["prompt_id"] == "p01"
    assert out["raw_text"] == "[#A4] Hallo."
    assert out["word_count"] == 2  # "[#A4]" wird als 1 Wort gezählt von str.split
    assert out["char_count"] == len("[#A4] Hallo.")
    assert out["classification"]["has_note"] is True


def test_per_response_empty():
    out = compute_per_response_metrics("", prompt_id="p02")
    assert out["word_count"] == 1  # min 1
    assert out["char_count"] == 0
    assert out["classification"]["total_tags"] == 0


# ─── 4. aggregate_run_metrics ─────────────────────────────────────────────


def test_aggregate_empty():
    out = aggregate_run_metrics([])
    assert out == {"n": 0}


def test_aggregate_no_tags_anywhere():
    per = [
        compute_per_response_metrics("Hallo Welt.", "p01"),
        compute_per_response_metrics("Wie geht es dir?", "p02"),
    ]
    out = aggregate_run_metrics(per)
    assert out["n_responses"] == 2
    assert out["tag_rate"] == 0.0
    assert out["note_tag_rate"] == 0.0
    assert out["dynamic_tag_rate"] == 0.0
    assert out["affect_tag_rate"] == 0.0
    assert out["pause_tag_rate"] == 0.0
    assert out["mean_density_when_tagging"] is None
    assert out["max_density"] is None
    assert out["density_warnings_count"] == 0


def test_aggregate_basic_rates():
    per = [
        compute_per_response_metrics("[#A4] Hallo.", "p01"),  # note
        compute_per_response_metrics("[#WHISPER] leise.", "p02"),  # dynamic
        compute_per_response_metrics("[#HAPPY] wow.", "p03"),  # affect
        compute_per_response_metrics("[#PAUSE 0.5s] kurz.", "p04"),  # pause
        compute_per_response_metrics("plain text no tags.", "p05"),  # none
    ]
    out = aggregate_run_metrics(per)
    assert out["n_responses"] == 5
    assert out["tag_rate"] == 0.8  # 4/5
    assert out["note_tag_rate"] == 0.2  # 1/5
    assert out["dynamic_tag_rate"] == 0.2
    assert out["affect_tag_rate"] == 0.2
    assert out["pause_tag_rate"] == 0.2


def test_aggregate_density_global():
    per = [
        # 1 tag in 4 Wörtern (str.split zählt [#A4] als Wort) ≈ 25/100
        compute_per_response_metrics("[#A4] eins zwei drei", "p01"),
        # 0 tags in 5 Wörtern
        compute_per_response_metrics("eins zwei drei vier fünf", "p02"),
    ]
    out = aggregate_run_metrics(per)
    # 1 tag in 9 Wörtern ≈ 11.11/100
    assert out["tags_per_100_words_global"] == 11.11


def test_aggregate_density_when_tagging():
    per = [
        compute_per_response_metrics("[#A4] eins", "p01"),  # 1 tag / 2 W
        compute_per_response_metrics("[#B4] [#C4] eins zwei", "p02"),  # 2 tags / 4 W
        compute_per_response_metrics("plain", "p03"),  # no tags
    ]
    out = aggregate_run_metrics(per)
    # mean_density_when_tagging: nur p01 (50.0) und p02 (50.0) → 50.0
    assert out["mean_density_when_tagging"] == 50.0
    assert out["max_density"] == 50.0


def test_aggregate_density_warnings_count():
    # 3+ Tags in <10 Wörtern → warnt
    per = [
        compute_per_response_metrics(
            "[#A4] eins [#B4] zwei [#C4] drei [#D4] vier", "p01"
        ),
    ]
    out = aggregate_run_metrics(per)
    assert out["density_warnings_count"] == 1


def test_aggregate_syntax_valid_count():
    per = [
        compute_per_response_metrics("[#A4] ok", "p01"),
        compute_per_response_metrics("plain", "p02"),
    ]
    out = aggregate_run_metrics(per)
    assert out["syntax_valid_count"] == 2  # alle gültig (parse_tags ist tolerant)


# ─── Runner ───────────────────────────────────────────────────────────────


def _run_all():
    tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = _run_all()
    sys.exit(0 if ok else 1)
