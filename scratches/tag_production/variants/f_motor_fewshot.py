"""Variante F: Motor-opt-in Few-Shot (Plan 6.2c).

Gleiche Wirkung wie C, aber über das neue ``append_tag_snippet(..., few_shot=True)``-
Flag im Motor (``gradio_tabs.system_prompt.CITMIND_TAG_FEWSHOT_TURNS`` als
single source of truth) statt über eine separat gepflegte SHOTS-Liste in der
Variante. Validiert dass die Motor-API die gleiche Compliance bringt wie der
empirische C-Variante-Pfad.
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
    """CitMind + Standard-Snip + 3 Few-Shot-Paare via Motor-Flag."""
    msgs = inject_into_messages(base_messages, profile_name="citmind")
    msgs = append_tag_snippet(msgs, render_tag_system_prompt(), few_shot=True)
    return _merge_sys_into_user(msgs)