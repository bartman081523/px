"""gradio_tabs/chat_actions.py — pure-logic chat history manipulation for the undo button.

No Gradio imports. Pure functions that operate on the chatbot history list
(list of dicts {"role": ..., "content": str | list}). Each function returns a NEW
list and never mutates its input.

Wired into gradio_tabs/chat_tab.py by the orchestrator (not this module).
"""

from __future__ import annotations

from typing import Any, List


def undo_last_turn(history: list) -> list:
    """Undo the most recent chat turn.

    - history ending with assistant: remove that assistant message AND the
      immediately preceding user message (one full turn [.., user, assistant] -> [..]).
    - history ending with user (no assistant reply yet): remove that trailing user.
    - empty history: return [].
    - Returns a NEW list (does not mutate input). Preserves content format
      (str or multimodal list). Compares by position/role only, never stringifies.
    """
    if not history:
        return []
    out = list(history)
    last = out[-1]
    role = last.get("role") if isinstance(last, dict) else None
    if role == "assistant" and len(out) >= 2 and isinstance(out[-2], dict) and out[-2].get("role") == "user":
        # remove the full turn: trailing assistant + preceding user
        del out[-2:]
    else:
        # trailing user, single message, or malformed tail: drop the last entry
        out.pop()
    return out


def undo_last_entry(history: list) -> list:
    """Undo only the very last message entry (single message, regardless of role).

    [u, a, u, a] -> [u, a, u]; [u, a, u] -> [u, a]; [] -> [].
    Returns a NEW list, does not mutate input.
    """
    if not history:
        return []
    out = list(history)
    out.pop()
    return out


def can_undo(history: list) -> bool:
    """True if there is any message to undo (len(history) > 0)."""
    return len(history) > 0