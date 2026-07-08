"""gradio_tabs/chat_actions.py — Pure-Logic Undo-Logik für Chat-History.

Plan: branch ui-styling, 2026-07-06, "Undo-Button für Turns".

Schicht: pure-logic Helpers für den "↶ Undo Last Turn"-Button in
chat_tab.py. Keine Gradio-Imports, keine Sessions-IO — nur Listen-Pop
+ Validierung. Wird vom UI-Handler (handle_undo) aufgerufen, der dann
save_session() macht.

Öffentliche API:
    can_undo(history) -> bool
    undo_last_turn(history) -> list[dict]
    undo_last_entry(history) -> list[dict]

Reihenfolge-Pin (T6): undo_last_turn poppt das letzte (user, assistant)-
Paar, sodass nach dem Undo kein einsamer assistant-Eintrag übrig bleibt
der die UI verwirrt.
"""
from __future__ import annotations
from typing import Any, Dict, List


def can_undo(history: List[Dict[str, Any]]) -> bool:
    """Returnt True wenn history ≥ 2 Messages hat (mindestens 1 Paar)."""
    return isinstance(history, list) and len(history) >= 2


def can_undo_entry(history: List[Dict[str, Any]]) -> bool:
    """Returnt True wenn history ≥ 1 Message hat (Undo-Bar für Single-Entry).

    Verwendet vom Button "Undo Last Message" (1 Element, nicht Paar).
    Erlaubt auch mid-stream halbe States (z.B. crash nach user ohne assistant).
    """
    return isinstance(history, list) and len(history) >= 1


def undo_last_turn(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Popt das letzte (user, assistant)-Paar.

    Bei 0/1 Messages: returnt [] (kein vollständiges Paar → no-op).
    Bei 2 Messages: returnt [].
    Bei 3+ Messages: returnt history ohne die letzten 2.

    Pin-Properties:
        T1: [] → []
        T2: [user] → [] (kein Paar)
        T3: [u, a] → []
        T4: [u, a, u] → [u]
        T5: mutiert Input NICHT (deep-copies der Dicts)
        T6: poppt immer das LETZTE Paar (Reihenfolge: assistant, dann user)
        T11: funktioniert mit Multimodal-List-Content
        T12: returnt unabhängige Dicts (UI-Mutation-Bug-Pin)
    """
    if not isinstance(history, list):
        return []
    if len(history) < 2:
        return []

    # Slice + deep-copy der verbleibenden Dicts (T5+T12: keine Mutation,
    # keine geteilten Referenzen zur Input-Liste).
    remaining = history[:-2]
    return [dict(m) for m in remaining if isinstance(m, dict)]


def undo_last_entry(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Popt NUR das letzte Element (low-level helper).

    Im Unterschied zu undo_last_turn ist dieser Helper roh-assistent: er
    poppt genau ein Element, egal welche Rolle. Wird verwendet wenn die
    History mid-stream ungleichmäßig ist (z.B. wenn bot_response crasht
    und einen halben assistant-msg hinterlässt).
    """
    if not isinstance(history, list) or len(history) == 0:
        return []
    return [dict(m) for m in history[:-1] if isinstance(m, dict)]
