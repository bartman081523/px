"""Variante C: CitMind + Standard-Tag-Snip + 3 Few-Shot-Paare.

Hypothese (Plan 6.2): 1B-Modelle lernen Syntax besser aus Beispielen als
aus Vokabular-Beschreibungen. 3 Beispiele im messages-Prefix vor der
eigentlichen Frage erhöhen Tag-Compliance — insbesondere bei distinktiven
Tags (WHISPER, PAUSE) wo A in Plan 6.1 ausgefallen ist.

Plan 6.2 Befund: tag_rate 0.6 → 1.0, note_tag_rate 0.5 → 0.8, pause 0.3 → 0.6.

Plan 6.2c: Die 3 Few-Shot-Paare sind jetzt in
``gradio_tabs.system_prompt.CITMIND_TAG_FEWSHOT_TURNS`` als single source
of truth definiert. Diese Datei importiert sie — keine Duplikation.
Der Aufrufer ``append_tag_snippet(..., few_shot=True)`` macht das gleiche
im Motor-Pfad.
"""
from __future__ import annotations

from typing import List, Dict, Any

from gradio_tabs.system_prompt import (
    inject_into_messages,
    append_tag_snippet,
    CITMIND_TAG_FEWSHOT_TURNS,
)
from gradio_tabs.vocoder_tags import render_tag_system_prompt

from ._common import _merge_sys_into_user


# Re-export für andere Konsumenten (z.B. empirischer Vergleich A vs C).
SHOTS = CITMIND_TAG_FEWSHOT_TURNS


def apply(base_messages: List[Dict[str, Any]], user_prompt: str) -> List[Dict[str, Any]]:
    """CitMind + Standard-Snip + 3 Few-Shot-Paare vor der Frage.

    Verwendet ``append_tag_snippet(..., few_shot=True)`` statt manuell
    SHOTS-Insertion — gleicher Pfad wie der Motor.
    """
    msgs = inject_into_messages(base_messages, profile_name="citmind")
    msgs = append_tag_snippet(msgs, render_tag_system_prompt(), few_shot=True)
    # _merge_sys_into_user packt System-Inhalt in den ersten user-Turn
    # (Gemma3 lehnt system-Rolle ab). Few-Shot-Turns überleben den Merge
    # als Multi-Turn-Liste dazwischen.
    return _merge_sys_into_user(msgs)