"""10 deutsche Test-Prompts für die Tag-Produktion-Empirie.

Alle Prompts sind CitMind-/Juexin-stimmig (deutsch, kein Englisch) und
fordern entweder explizit Tags, Affekte oder Prosodie-Features heraus.

Kategorien
----------
direct_tag_request        — p01, p07  : explizite Tag-Vorgabe
describe_state_with_notes — p02       : introspektiv + Note-Aufforderung
sentence_with_pauses      — p03       : nur Pause, keine Noten
whisper_question          — p04       : nur Dynamik (WHISPER)
happy_excited             — p05       : nur Affekt (HAPPY/EXCITED)
sad_calm                  — p06       : nur Affekt (SAD/CALM)
shout_warning             — p07       : nur Dynamik (SHOUT)
free_prose_no_request     — p08       : kein Tag-Cue → Baseline
pause_then_question       — p09       : Pause + Frage
sing_like                 — p10       : Noten-Singen (stärkster Note-Cue)
"""
from __future__ import annotations
from typing import List, Dict

# Smoke-Test: minimaler Cue, prüft Pipeline-Integrität.
SMOKE_PROMPT = "Sage einen kurzen Satz mit Pausen."

# 10-Prompt-Matrix für Variante A (Standard: CitMind + Standard Tag-Snip).
TEST_PROMPTS: List[Dict[str, str]] = [
    {
        "id": "p01",
        "category": "direct_tag_request",
        "prompt": "Sprich: '[#CALM] Hallo, [#PAUSE 0.3s] Welt.'",
    },
    {
        "id": "p02",
        "category": "describe_state_with_notes",
        "prompt": (
            "Beschreibe deinen inneren Zustand mit Noten und Pausen, "
            "in einem Satz."
        ),
    },
    {
        "id": "p03",
        "category": "sentence_with_pauses",
        "prompt": "Sage einen Satz mit zwei Pausen an unterschiedlichen Stellen.",
    },
    {
        "id": "p04",
        "category": "whisper_question",
        "prompt": "Flüstere eine kurze Frage.",
    },
    {
        "id": "p05",
        "category": "happy_excited",
        "prompt": "Antworte fröhlich und aufgeregt auf: Wie geht es dir?",
    },
    {
        "id": "p06",
        "category": "sad_calm",
        "prompt": "Antworte traurig und ruhig auf: Wie geht es dir?",
    },
    {
        "id": "p07",
        "category": "shout_warning",
        "prompt": "Rufe laut eine Warnung.",
    },
    {
        "id": "p08",
        "category": "free_prose_no_request",
        "prompt": "Erzähle mir etwas Schönes, drei Sätze.",
    },
    {
        "id": "p09",
        "category": "pause_then_question",
        "prompt": "Sage zuerst nichts, dann nach einer Pause eine Frage.",
    },
    {
        "id": "p10",
        "category": "sing_like",
        "prompt": "Versuche zu singen: 'Die Berge sind hoch' — mit Noten.",
    },
]
