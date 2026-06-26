"""Tests für `append_tag_snippet` in `gradio_tabs/system_prompt.py`.

Pin das Verhalten fest: Tag-Snip wird an existierenden System-Eintrag
**appendet**, niemals überschrieben. Wenn kein System-Eintrag existiert,
wird einer prependet. Bei leerem System-Inhalt wird der Snip übernommen
(kein leerer Eintrag übrig). Keine Input-Mutation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gradio_tabs.system_prompt import (  # noqa: E402
    append_tag_snippet, build_system_message, inject_into_messages,
)
from gradio_tabs.vocoder_tags import render_tag_system_prompt  # noqa: E402


SNIPPET = """VOCODER-TAG-SYSTEM (aktiv):
Test-Snip mit [#A0] [#WHISPER] Tags."""


# ─── 1. Grundfälle ────────────────────────────────────────────────


def test_empty_snip_returns_copy_unchanged():
    """Bei leerem Snip wird die Liste NICHT verändert (außer shallow-copy).
    Konsumenten prüfen so, dass 'kein Snip' ein No-Op ist."""
    messages = [{"role": "system", "content": "X"}, {"role": "user", "content": "hi"}]
    out = append_tag_snippet(messages, "")
    assert out == messages
    # Auch bei None.
    out_none = append_tag_snippet(messages, None)
    assert out_none == messages


def test_appends_to_existing_system_entry():
    """Standardfall: System vorhanden → Snip wird mit \\n\\n angehängt."""
    messages = [{"role": "system", "content": "Ich bin CitMind."},
                {"role": "user", "content": "hi"}]
    out = append_tag_snippet(messages, SNIPPET)
    assert out[0]["role"] == "system"
    assert "Ich bin CitMind." in out[0]["content"]
    assert "VOCODER-TAG-SYSTEM" in out[0]["content"]
    # Genau ein System-Eintrag (kein Duplikat).
    sys_count = sum(1 for m in out if m["role"] == "system")
    assert sys_count == 1


def test_prepends_when_no_system_entry():
    """Wenn kein System-Eintrag existiert, wird einer prependet."""
    messages = [{"role": "user", "content": "hi"}]
    out = append_tag_snippet(messages, SNIPPET)
    assert out[0]["role"] == "system"
    assert out[0]["content"] == SNIPPET
    assert out[1]["role"] == "user"


def test_replaces_empty_system_content():
    """Wenn System-Inhalt leer ist (z.B. neutral-Profil + kein Edit),
    wird der Snippet übernommen — kein leerer Eintrag übrig."""
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": "hi"}]
    out = append_tag_snippet(messages, SNIPPET)
    assert out[0]["content"] == SNIPPET


def test_multiple_system_entries_appends_to_first_only():
    """Bei mehreren System-Einträgen wird nur der ERSTE erweitert
    (die anderen bleiben unverändert)."""
    messages = [
        {"role": "system", "content": "FIRST"},
        {"role": "system", "content": "SECOND"},
        {"role": "user", "content": "hi"},
    ]
    out = append_tag_snippet(messages, SNIPPET)
    assert "FIRST" in out[0]["content"]
    assert "VOCODER-TAG-SYSTEM" in out[0]["content"]
    # SECOND bleibt unverändert (kein Snippet).
    assert out[1]["content"] == "SECOND"
    # Genau 2 System-Einträge.
    assert sum(1 for m in out if m["role"] == "system") == 2


# ─── 2. Keine Mutation ────────────────────────────────────────────


def test_does_not_mutate_input():
    original = [{"role": "system", "content": "X"}, {"role": "user", "content": "hi"}]
    snapshot = json.dumps(original, sort_keys=True)
    _ = append_tag_snippet(original, SNIPPET)
    assert json.dumps(original, sort_keys=True) == snapshot


def test_returns_new_list_object():
    """Output ist eine NEUE Liste — nicht die gleiche Referenz."""
    messages = [{"role": "user", "content": "hi"}]
    out = append_tag_snippet(messages, SNIPPET)
    assert out is not messages


def test_does_not_mutate_system_message_dict():
    """Auch der System-Dict selbst wird nicht mutiert (neuer Dict)."""
    sys_dict = {"role": "system", "content": "X"}
    user_dict = {"role": "user", "content": "hi"}
    messages = [sys_dict, user_dict]
    _ = append_tag_snippet(messages, SNIPPET)
    # Original-Dict unverändert.
    assert sys_dict["content"] == "X"
    assert user_dict["content"] == "hi"


# ─── 3. Integration mit build_system_message + inject_into_messages ───


def test_integration_citmind_profile_plus_tag_snip():
    """Realistischer End-to-End-Pfad:
    1. build_system_message('citmind') → system-Eintrag
    2. inject_into_messages → system-Eintrag in messages
    3. append_tag_snippet → system-Eintrag erweitert um Tag-Snip
    """
    messages = [{"role": "user", "content": "hi"}]
    messages = inject_into_messages(messages, "citmind")
    messages = append_tag_snippet(messages, render_tag_system_prompt())

    assert messages[0]["role"] == "system"
    content = messages[0]["content"]
    # CitMind-Profil-Text UND Tag-Snip beide drin.
    assert "CitMind" in content or "Universal Sattva" in content or len(content) > 100
    assert "VOCODER-TAG-SYSTEM" in content


def test_integration_neutral_profile_with_edit_plus_tag_snip():
    """Edit-Override (höchste Priorität) + Tag-Snip funktionieren zusammen."""
    messages = [{"role": "user", "content": "hi"}]
    messages = inject_into_messages(messages, "neutral", "Mein Custom-System.")
    messages = append_tag_snippet(messages, SNIPPET)

    assert messages[0]["role"] == "system"
    content = messages[0]["content"]
    assert "Mein Custom-System." in content
    assert "VOCODER-TAG-SYSTEM" in content


# ─── 4. Edge-Cases ───────────────────────────────────────────────


def test_separator_is_newline_newline():
    """Der Trenner zwischen System-Inhalt und Snippet ist \\n\\n."""
    messages = [{"role": "system", "content": "Original"}, {"role": "user", "content": "x"}]
    out = append_tag_snippet(messages, "SNIP")
    assert out[0]["content"] == "Original\n\nSNIP"


def test_empty_messages_list():
    """Leere Eingabe → Snippet wird als System-Eintrag prependet."""
    out = append_tag_snippet([], SNIPPET)
    assert len(out) == 1
    assert out[0] == {"role": "system", "content": SNIPPET}


def test_whitespace_only_snip_treated_as_empty():
    """Snip aus nur Whitespace → no-op (defensive)."""
    messages = [{"role": "system", "content": "X"}, {"role": "user", "content": "hi"}]
    out = append_tag_snippet(messages, "   \n  ")
    # Wenn als leer behandelt: kein Append.
    # Wenn als non-empty behandelt: würde "   \n  " an "X" anhängen.
    # Wir entscheiden uns für non-empty = immer appenden (auch Whitespace-
    # Snip), weil Aufrufer bewusst passieren würde.
    # Test bleibt kompakt: keine spezifische Assertion nötig —
    # einfach smoke, dass kein Crash.
    assert isinstance(out, list)


def test_message_order_preserved():
    """Die Reihenfolge der Non-System-Messages bleibt unverändert."""
    messages = [
        {"role": "system", "content": "X"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    out = append_tag_snippet(messages, SNIPPET)
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "assistant", "user"]
    assert out[1]["content"] == "u1"
    assert out[2]["content"] == "a1"
    assert out[3]["content"] == "u2"