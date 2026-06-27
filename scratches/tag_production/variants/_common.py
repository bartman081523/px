"""variants/_common.py — Geteilte Helper-Funktionen für alle Varianten.

Aktuell:
  - _merge_sys_into_user(msgs): Workaround für den Plan-6.1-Bug in
    ``merge_system_into_first_user`` (droppt den Tag-Snip). Wird von
    ALLEN Varianten (A/B/C/D/E) gebraucht, weil Gemma3-Chat-Template
    die ``{"role": "system"}``-Einträge ablehnt — der Snip-Inhalt muss
    in den User-Turn-Prefix.

Hintergrund
----------
``gradio_tabs.system_prompt.merge_system_into_first_user`` rendert das
System aus ``profile_name`` NEU via ``render_for_chat_template`` — der
Tag-Snip, der VOR dem Merge via ``append_tag_snippet`` an die System-
Message angehängt wurde, geht dabei verloren. Workaround: System-Inhalt
(含 Snip) extrahieren, als user-turn-Prefix setzen, System-Item droppen,
und ``merge_system_into_first_user`` mit ``profile_name="neutral"``
(no-op re-render) aufrufen.
"""
from __future__ import annotations

from typing import List, Dict, Any

from gradio_tabs.system_prompt import merge_system_into_first_user


def _merge_sys_into_user(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extrahiert System-Inhalt (inkl. Tag-Snip) und packt ihn als
    user-turn-Prefix. Gibt messages ohne system-Item zurück, bereit für
    Gemma3-Chat-Template.

    Vorbedingung: msgs hat GENAU EINEN system-Eintrag (von
    inject_into_messages + append_tag_snippet).
    """
    sys_idx = next(
        (i for i, m in enumerate(msgs) if m.get("role") == "system"),
        None,
    )
    if sys_idx is None:
        # Kein System → nichts zu mergen, einfach durchlassen.
        return msgs

    sys_content = msgs[sys_idx].get("content") or ""
    if not isinstance(sys_content, str) or not sys_content:
        # Leeres System → droppen, kein Prefix.
        return [m for m in msgs if m.get("role") != "system"]

    # System-Inhalt raus, User-Turn mit Prefix.
    new_msgs = [m for m in msgs if m.get("role") != "system"]
    user_idx = next(
        (i for i, m in enumerate(new_msgs) if m.get("role") == "user"),
        None,
    )
    if user_idx is not None:
        old_user_content = new_msgs[user_idx].get("content") or ""
        new_msgs[user_idx] = {
            **new_msgs[user_idx],
            "content": sys_content + "\n\n" + old_user_content,
        }
    else:
        new_msgs.insert(0, {"role": "user", "content": sys_content})

    # neutral-Re-Render ist no-op (render_for_chat_template mit neutral
    # = keine zusätzliche System-Message).
    return merge_system_into_first_user(new_msgs, profile_name="neutral")