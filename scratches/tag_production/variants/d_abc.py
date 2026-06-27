"""Variante D: CitMind + ABC-Notation-Snip (statt Vocoder-Tag-Snip).

Hypothese (Plan 6.2): ABC-Notation ist ein etabliertes Musik-Textformat.
1B-Modelle mit allgemeinem Training könnten es besser kennen als das
proprietäre [#A0..G#7]-Vokabular. ABC hat den Vorteil, dass Noten +
Länge + Taktart in einem kompakten String ausgedrückt werden.

Wichtig: in D wird der Vocoder-Tag-Snip DURCH den ABC-Snip ERSETZT
(nicht additiv). Damit messen wir isoliert, ob ABC die Note-Compliance
hebt — wenn ja, wäre eine Integration beider Snips ein Folge-Schritt.

Post-Processing (ABC → Audio) ist nicht im Scope dieses Plans —
eigener Plan-Phase nötig.
"""
from __future__ import annotations

from typing import List, Dict, Any

from gradio_tabs.system_prompt import (
    inject_into_messages,
    append_tag_snippet,
)

from ._common import _merge_sys_into_user


ABC_SNIPPET = """ABC-NOTATION-SYSTEM (aktiv):
Du kannst optional vollständige Musik-Stücke in ABC-Notation einbetten.
Verwende EXAKT diese Syntax:

  X:<nr>           → Reference-Number (eindeutig pro Stück)
  T:<titel>        → Titel (optional)
  M:<zähler>/<nenner>  → Taktart, z.B. 4/4, 3/4, 6/8
  L:<zähler>/<nenner>  → Default-Notenlänge, z.B. 1/4 (Viertel)
  K:<tonart>       → Tonart (C, D, G, Am, Em, ...)
  |: <noten> :|    → Stück (mit Wiederholung)

Noten (gross = Viertel, lowercase = Achtel, ^/=/_ = sharp/nat/flat):
  C D E F G A B   → weiße Tasten
  ^C ^D ^F ^G ^A  → Cis Dis Fis Gis Ais (= sharp)
  _B               → B-flat
  z                → Pause (Schlag)
  |                → Taktstrich
  ||: :||          → Wiederholung

Beispiel:
  X:1
  T:Kleine Melodie
  M:4/4
  L:1/4
  K:C
  |: C D E F | G A B c :|

Setze ABC-Blöcke SPARSAM — nur wo es der Bedeutung dient. Für normalen
Fließtext KEINE ABC.
"""


def apply(base_messages: List[Dict[str, Any]], user_prompt: str) -> List[Dict[str, Any]]:
    """CitMind + ABC-Snip (KEIN Vocoder-Tag-Snip)."""
    msgs = inject_into_messages(base_messages, profile_name="citmind")
    msgs = append_tag_snippet(msgs, ABC_SNIPPET)
    return _merge_sys_into_user(msgs)