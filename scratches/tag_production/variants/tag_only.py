"""Variante: Reiner Tag-Snip ohne CitMind-Profil.

Hypothese: Wenn CitMind der Tag-Produktion im Weg steht (Sanskrit-Frame
dominiert), dann ist die Tag-Compliance OHNE CitMind höher. Dies ist
die Baseline-Kontrolle: misst was der Standard-Snip alleine kann.

NICHT AUSFÜHREN in diesem Plan. Nur Struktur-Skelett.
"""
from __future__ import annotations

from typing import List, Dict, Any

from gradio_tabs.system_prompt import append_tag_snippet
from gradio_tabs.vocoder_tags import render_tag_system_prompt


def apply(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strippt ALLE System-Einträge und hängt NUR den Tag-Snip an.

    Risiko: Modell kennt keine Ontologie/Rolle mehr → kann in
    generisches LLM-Verhalten fallen. Vergleichswert für die
    CitMind-Hypothese.
    """
    no_sys = [m for m in messages if m.get("role") != "system"]
    return append_tag_snippet(no_sys, render_tag_system_prompt())
