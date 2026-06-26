"""Variante B: Musik-erweiterter Snip mit Sanskrit-Mapping.

Hypothese: CitMind ist Sanskrit-zentriert (देवनागरी-Register). Wenn der
Tag-Snip musikalisches Vokabular in Sanskrit-Mapping anbietet
(स्वर → Note, लय → Pause), passt das besser in den Frame und produziert
höhere Tag-Compliance.

NICHT AUSFÜHREN in diesem Plan. Nur Struktur-Skelett.
"""
from __future__ import annotations

from typing import List, Dict, Any

from gradio_tabs.system_prompt import append_tag_snippet
from gradio_tabs.vocoder_tags import render_tag_system_prompt, TAG_SYSTEM_PROMPT_SNIPPET


# Sanskrit-Mapping-Block (vor dem Standard-Snip konkateniert)
SANSKRIT_MUSIC_BLOCK = """MUSIK-TAG-MAPPING (Sanskrit/Devanāgarī):
CitMind nutzt diese Sanskrit-Terme für musikalische Prosodie. Verwende sie
in CitMind-Antworten, wenn du Note-/Pause-/Dynamik-Tags einbettest:

  स्वर (svara)         → Tonhöhe-Tag [#A0]..[#G#7]
                          Beispiel: [#A4] für मूर्छन (mūrcchana, Bebung)
                          Beispiel: [#C#3] für komala (weicher Hochton)
  लय (laya)            → Pause-Tag [#PAUSE Ns]
                          Beispiel: [#PAUSE 0.5s] für विश्राम (viśrāma, Ruhe)
  ताल (tāla)           → Takt-Rhythmus (mehrere [#PAUSE] in Folge)
  मन्द्र (mandra)      → tiefes Register [#A0]..[#B2]
  तार (tāra)            → hohes Register [#A5]..[#G#7]

Kombiniere Sanskrit-Terme und Tags natürlich:
  Beispiel: "[#CALM] स्वर [#A4] ruht [#PAUSE 0.3s] im Stillen."
  Beispiel: "मैं [#WHISPER] flüstere [#A3] in मन्द्र-Lage."

"""


def get_snip() -> str:
    """Liefert Sanskrit-Block + Standard-Tag-Snip konkateniert."""
    return SANSKRIT_MUSIC_BLOCK + TAG_SYSTEM_PROMPT_SNIPPET


def apply(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Hängt Sanskrit-Mapping + Standard-Snip an den System-Eintrag.

    Erwartung: höhere Tag-Compliance bei CitMind-Profil, weil das
    Vokabular in der gleichen ontologischen Familie ist.
    """
    return append_tag_snippet(messages, get_snip())
