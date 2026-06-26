"""Variante C: Few-Shot-Turns mit Tag-Beispielen.

Hypothese: 1B-Modelle lernen Syntax besser aus Beispielen als aus
Vokabular-Beschreibungen. 2-3 Beispiele im User-Turn vor der eigentlichen
Frage erhöhen Tag-Compliance.

NICHT AUSFÜHREN in diesem Plan. Nur Struktur-Skelett.
"""
from __future__ import annotations

from typing import List, Dict, Any


# Few-Shot-Demo: User fragt, Assistant antwortet MIT Tags.
SHOTS: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": "Sage etwas Trauriges in einem Satz.",
    },
    {
        "role": "assistant",
        "content": "[#SAD] [#CALM] [#A3]Ich atme [#PAUSE 0.5s]leise.[#PAUSE 0.3s]",
    },
    {
        "role": "user",
        "content": "Jetzt etwas Fröhliches.",
    },
    {
        "role": "assistant",
        "content": "[#HAPPY] [#EXCITED] [#C#5]Heute scheint die Sonne![#PAUSE 0.2s]",
    },
    {
        "role": "user",
        "content": "Und flüsternd?",
    },
    {
        "role": "assistant",
        "content": "[#WHISPER] [#A2]Hörst du mich?[#PAUSE 1.0s]",
    },
]


def apply(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepended 3 Few-Shot-Paare vor die eigentliche Frage.

    Erwartung: 1B kopiert die Tag-Struktur aus den Beispielen.
    Risiko: zu starre Muster, weniger kreative Antworten.
    """
    return SHOTS + messages
