"""Variante D: ABC-Notation statt Vocoder-Tag-Syntax.

Hypothese: ABC-Notation ist ein etabliertes Musik-Textformat. 1B-Modelle
mit allgemeinem Training könnten es besser kennen als das proprietäre
[#A0..G#7]-Vokabular. ABC hat den Vorteil, dass Noten + Länge + Taktart
in einem kompakten String ausgedrückt werden.

Format-Erinnerung (siehe http://abcnotation.com/):
  X:1          → Reference-Number
  T:Title      → Titel
  M:4/4        → Taktart
  L:1/4        → Default-Notenlänge
  K:C          → Tonart
  |: C D E F | G A B c :|

NICHT AUSFÜHREN in diesem Plan. Nur Struktur-Skelett.
Post-Processing (ABC → Audio) ist komplexer als [#A4] → Pitch — eigene
Plan-Phase nötig.
"""
from __future__ import annotations

from typing import List, Dict, Any

from gradio_tabs.system_prompt import append_tag_snippet


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
  [1 [2 | [1 [2    → Volta-Klammern (1./2. Ende)

Beispiel:
  X:1
  T:Kleine Melodie
  M:4/4
  L:1/4
  K:C
  |: C D E F | G A B c :|

Setze ABC-Blöcke SPARSAM — nur wo es der Bedeutung dient. Für normalen
Fließtext KEINE ABC. Tags werden automatisch entfernt, wenn der
Post-Processor sie nicht versteht (silent strip).

Hinweis Range: Piper/esak-ng decken C3..C6 sauber ab. Darüber/darunter
klingt es verzerrt. Für realistische tiefe Noten → Bark-Engine (GPU).
"""


def apply(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Hängt ABC-Snip an den System-Eintrag (orthogonal zu CitMind)."""
    return append_tag_snippet(messages, ABC_SNIPPET)
