"""Variante B: CitMind + Standard-Tag-Snip + Sanskrit-Mapping-Block.

Hypothese (Plan 6.2): CitMind ist Sanskrit-zentriert (देवनागरी-Register).
Wenn der Tag-Snip musikalisches Vokabular in Sanskrit-Mapping anbietet
(स्वर → Note, लय → Pause), passt das besser in den Frame und produziert
höhere Tag-Compliance + weniger multilingualer Drift ins Devanāgarī.
"""
from __future__ import annotations

from typing import List, Dict, Any

from gradio_tabs.system_prompt import (
    inject_into_messages,
    append_tag_snippet,
)
from gradio_tabs.vocoder_tags import render_tag_system_prompt

from ._common import _merge_sys_into_user


# Sanskrit-Mapping-Block (VOR dem Standard-Snip konkateniert).
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


def apply(base_messages: List[Dict[str, Any]], user_prompt: str) -> List[Dict[str, Any]]:
    """Baut messages: CitMind + Sanskrit-Mapping-Snip + Standard-Snip."""
    msgs = inject_into_messages(base_messages, profile_name="citmind")
    # Sanskrit-Block VOR Standard-Snip (Reihenfolge wichtig: erst Mapping, dann Tags)
    msgs = append_tag_snippet(msgs, SANSKRIT_MUSIC_BLOCK + render_tag_system_prompt())
    return _merge_sys_into_user(msgs)