"""Variante A: CitMind + Standard-Tag-Snip — Baseline (Plan 6.1).

Plan 6.1 Befund (committed 16c5750):
  tag_rate=0.6, note_tag_rate=0.5
  Hypothese widerlegt — 1B produziert substanziell Note-Tags.
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
    """Baut messages 1:1 wie Plan 6.1: CitMind + Standard-Tag-Snip."""
    msgs = inject_into_messages(base_messages, profile_name="citmind")
    msgs = append_tag_snippet(msgs, render_tag_system_prompt())
    return _merge_sys_into_user(msgs)