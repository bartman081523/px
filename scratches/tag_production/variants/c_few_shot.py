"""Variante C: CitMind + Standard-Tag-Snip + 3 Few-Shot-Paare.

Hypothese (Plan 6.2): 1B-Modelle lernen Syntax besser aus Beispielen als
aus Vokabular-Beschreibungen. 3 Beispiele im messages-Prefix vor der
eigentlichen Frage erhöhen Tag-Compliance — insbesondere bei distinktiven
Tags (WHISPER, PAUSE) wo A in Plan 6.1 ausgefallen ist.
"""
from __future__ import annotations

from typing import List, Dict, Any

from gradio_tabs.system_prompt import (
    inject_into_messages,
    append_tag_snippet,
)
from gradio_tabs.vocoder_tags import render_tag_system_prompt

from ._common import _merge_sys_into_user


# 3 Few-Shot-Paare (User → Assistant). Demo der Tag-Syntax in 3 Stilen:
# traurig, fröhlich, flüsternd. Mix aus Note/Dynamic/Affect/Pause.
SHOTS: List[Dict[str, str]] = [
    {"role": "user", "content": "Sage etwas Trauriges in einem Satz."},
    {"role": "assistant", "content": "[#SAD] [#CALM] [#A3]Ich atme [#PAUSE 0.5s]leise.[#PAUSE 0.3s]"},
    {"role": "user", "content": "Jetzt etwas Fröhliches."},
    {"role": "assistant", "content": "[#HAPPY] [#EXCITED] [#C#5]Heute scheint die Sonne![#PAUSE 0.2s]"},
    {"role": "user", "content": "Und flüsternd?"},
    {"role": "assistant", "content": "[#WHISPER] [#A2]Hörst du mich?[#PAUSE 1.0s]"},
]


def apply(base_messages: List[Dict[str, Any]], user_prompt: str) -> List[Dict[str, Any]]:
    """CitMind + Standard-Snip + 3 Few-Shot-Paare vor der Frage."""
    msgs = inject_into_messages(base_messages, profile_name="citmind")
    msgs = append_tag_snippet(msgs, render_tag_system_prompt())
    # Vor dem Merge die Few-Shot-Turns zwischen System und User einfügen.
    # Finde den ersten User-Turn (nach CitMind+Snip-System).
    user_idx = next(
        (i for i, m in enumerate(msgs) if m.get("role") == "user"),
        None,
    )
    if user_idx is not None:
        msgs = msgs[:user_idx] + SHOTS + msgs[user_idx:]
    # Achtung: Few-Shot-Turns werden auch in den User-Prefix gemerged, weil
    # _merge_sys_into_user den System-Inhalt als Prefix setzt. Das ist OK
    # — Gemma3-Chat-Template akzeptiert Multi-Turn-Listen, der Prefix ist
    # weiterhin die "Anweisung", die Few-Shots sind Beispiele.
    return _merge_sys_into_user(msgs)