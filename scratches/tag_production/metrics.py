"""scratches/tag_production/metrics.py — Pure-Logic Metrik-Aggregator.

Misst Tag-Compliance auf einem LLM-Output:
  - Total-Tag-Count
  - Per-Kind-Count (note, dynamic, affect, pause, bark)
  - Has-Flags (has_note, has_dynamic, …)
  - Density (Tags/100 Wörter) + Density-Warnung
  - Syntax-Validität (via parse_tags — toleranter Parser)
  - Clean-Text + Raw-Tags für Audit

Alle Funktionen sind pure-logic (kein torch, kein Modell). Getestet in
tests/test_metrics.py ohne GPU.

Reused: gradio_tabs.vocoder_tags.parse_tags + tag_density_warning
        (zentrale Tag-Parsing-Routinen, 137 Tests grün).
"""
from __future__ import annotations

from collections import Counter
from typing import List, Dict, Any, Optional

from gradio_tabs.vocoder_tags import (
    parse_tags,
    tag_density_warning,
    TagEvent,
)


# ─── 1. Atomare Helpers ──────────────────────────────────────────────────


def count_words(text: str) -> int:
    """Wortzahl, min 1 (verhindert Div/0 bei leerem Text)."""
    return max(1, len((text or "").split()))


def classify_response_tags(text: str) -> Dict[str, Any]:
    """Zerlegt den LLM-Output via parse_tags und liefert Counts/Flags.

    Output-Felder:
      - total_tags           : int
      - note / dynamic / affect / pause / bark : int (Count pro Kind)
      - has_note / has_dynamic / has_affect / has_pause : bool
      - has_any_tag          : bool (total_tags > 0)
      - density_per_100w     : float (Tags pro 100 Wörter)
      - density_warning      : Optional[str] (None wenn unauffällig)
      - syntax_valid         : bool (parse_tags ist tolerant — True wenn
                                    der Parser eine Liste zurückgab)
      - clean_text           : str (Original ohne Tags)
      - raw_tags             : List[dict] (kind/value/offset/raw pro Tag)
    """
    if text is None:
        text = ""
    clean, tags = parse_tags(text)
    kinds = Counter(t.kind for t in tags)

    word_count = count_words(text)
    density = round((len(tags) / word_count) * 100, 2)

    return {
        "total_tags": len(tags),
        "note": kinds.get("note", 0),
        "dynamic": kinds.get("dynamic", 0),
        "affect": kinds.get("affect", 0),
        "pause": kinds.get("pause", 0),
        "bark": kinds.get("bark", 0),
        "has_note": kinds.get("note", 0) > 0,
        "has_dynamic": kinds.get("dynamic", 0) > 0,
        "has_affect": kinds.get("affect", 0) > 0,
        "has_pause": kinds.get("pause", 0) > 0,
        "has_any_tag": len(tags) > 0,
        "density_per_100w": density,
        "density_warning": tag_density_warning(text),
        "syntax_valid": True,  # parse_tags ist tolerant (unbekannte Tags werden gestrippt)
        "clean_text": clean,
        "raw_tags": [
            {
                "kind": t.kind,
                "value": t.value,
                "offset": t.char_offset,
                "raw": t.raw,
            }
            for t in tags
        ],
    }


# ─── 2. Per-Response-Metrik ──────────────────────────────────────────────


def compute_per_response_metrics(
    raw_answer: str,
    prompt_id: str = "",
) -> Dict[str, Any]:
    """Bündelt alle Metriken für EINE Antwort.

    Felder: prompt_id, raw_text, word_count, char_count, classification.
    """
    if raw_answer is None:
        raw_answer = ""
    classification = classify_response_tags(raw_answer)
    return {
        "prompt_id": prompt_id,
        "raw_text": raw_answer,
        "word_count": count_words(raw_answer),
        "char_count": len(raw_answer),
        "classification": classification,
    }


# ─── 3. Run-Aggregation ──────────────────────────────────────────────────


def aggregate_run_metrics(per_response: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregiert Per-Response-Metriken zu Run-Level-Statistiken.

    Felder:
      - n_responses                    : int
      - tag_rate                       : float (Anteil Antworten mit ≥1 Tag)
      - note_tag_rate                  : float (Antworten mit ≥1 Note-Tag)
      - dynamic_tag_rate               : float
      - affect_tag_rate                : float
      - pause_tag_rate                 : float
      - tags_per_100_words_global      : float (global über alle Antworten)
      - mean_density_when_tagging      : Optional[float]
      - max_density                    : Optional[float]
      - density_warnings_count         : int (Antworten mit Warnung)
    """
    n = len(per_response)
    if n == 0:
        return {"n": 0}

    any_tag = sum(1 for r in per_response if r["classification"]["has_any_tag"])
    note = sum(1 for r in per_response if r["classification"]["has_note"])
    dynamic = sum(1 for r in per_response if r["classification"]["has_dynamic"])
    affect = sum(1 for r in per_response if r["classification"]["has_affect"])
    pause = sum(1 for r in per_response if r["classification"]["has_pause"])

    total_words = sum(r["word_count"] for r in per_response)
    total_tags = sum(r["classification"]["total_tags"] for r in per_response)

    densities_when_tagging = [
        r["classification"]["density_per_100w"]
        for r in per_response
        if r["classification"]["total_tags"] > 0
    ]
    density_warnings_count = sum(
        1 for r in per_response
        if r["classification"]["density_warning"] is not None
    )

    return {
        "n_responses": n,
        "tag_rate": round(any_tag / n, 3),
        "note_tag_rate": round(note / n, 3),
        "dynamic_tag_rate": round(dynamic / n, 3),
        "affect_tag_rate": round(affect / n, 3),
        "pause_tag_rate": round(pause / n, 3),
        "tags_per_100_words_global": round(
            (total_tags / max(1, total_words)) * 100, 2
        ),
        "mean_density_when_tagging": (
            round(sum(densities_when_tagging) / len(densities_when_tagging), 2)
            if densities_when_tagging
            else None
        ),
        "max_density": (
            max(densities_when_tagging)
            if densities_when_tagging
            else None
        ),
        "density_warnings_count": density_warnings_count,
        "syntax_valid_count": sum(
            1 for r in per_response if r["classification"]["syntax_valid"]
        ),
    }
