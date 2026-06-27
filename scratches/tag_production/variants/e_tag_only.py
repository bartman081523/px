"""Variante E: Neutral-Profil + Standard-Tag-Snip (kein CitMind).

Hypothese (Plan 6.2): Wenn CitMind der Tag-Produktion im Weg steht
(Sanskrit-Frame dominiert), dann ist die Tag-Compliance OHNE CitMind
höher. E ist die Kontroll-Bedingung: misst was der Standard-Snip alleine
kann, ohne ontologische Vorprägung durch das Profil.

Risiko: Modell kennt keine Ontologie/Rolle mehr → kann in generisches
LLM-Verhalten fallen. E ist EXPLIZIT dieser Vergleichswert für die
CitMind-Hypothese.

Achtung: Auch E braucht den System→User-Merge-Workaround, weil Gemma3-
Chat-Template ``{"role":"system"}`` ablehnt. neutral-Profil hat zwar
keinen Render-Inhalt, aber der Snip selbst ist im System-Eintrag und
muss in den User-Turn-Prefix.
"""
from __future__ import annotations

from typing import List, Dict, Any

from gradio_tabs.system_prompt import (
    inject_into_messages,
    append_tag_snippet,
)
from gradio_tabs.vocoder_tags import render_tag_system_prompt

from ._common import _merge_sys_into_user


def apply(base_messages: List[Dict[str, Any]], user_prompt: str) -> List[Dict[str, Any]]:
    """Neutral-Profil + Standard-Tag-Snip (kein CitMind)."""
    msgs = inject_into_messages(base_messages, profile_name="neutral")
    msgs = append_tag_snippet(msgs, render_tag_system_prompt())
    # Achtung: profile_name="neutral" bei inject_into_messages erzeugt
    # keinen System-Eintrag (neutral ist No-Op). Also muss der Snip selbst
    # als System-Eintrag da sein — tut er nach append_tag_snippet. Der
    # Merge-Workaround packt ihn in den User-Prefix.
    return _merge_sys_into_user(msgs)