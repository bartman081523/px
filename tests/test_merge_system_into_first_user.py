"""Regressions-Tests für `merge_system_into_first_user`.

Hintergrund
-----------
Beim ersten Live-Test des System-Prompts mit Gemma3 + Gradio-Chat-Tab
warf `tokenizer.apply_chat_template` einen `jinja2.exceptions.TemplateError`:
    "Conversation roles must alternate user/assistant/user/assistant/..."

Ursache: die ursprüngliche Render-Strategie hat aus
    [system, user, ...]
einen separaten user-Turn für das System gebaut:
    [user(<system>), user(<message>), ...]
→ zwei user-Turns hintereinander → Jinja schmeißt "must alternate".

Diese Tests pinnen das Korrektur-Verhalten fest: System-Inhalt wird in
den **ersten echten user-Turn** als Text-Prefix gemerged (statt ein
zusätzlicher user-Turn davorzuschieben). Damit ist die user/assistant-
Alternation garantiert, und der System-Inhalt landet physikalisch im
Modell-Kontext (Sentinel-prefixed).

Diese Datei ist ein Regressions-Test; die zugehörige Pure-Logic-Funktion
liegt in ``gradio_tabs/system_prompt.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Repo-Root zum sys.path (damit `gradio_tabs` importierbar ist, auch wenn
# pytest aus tests/ läuft).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gradio_tabs.system_prompt import (  # noqa: E402
    SYSTEM_SENTINEL,
    merge_system_into_first_user,
)


# ─── Helper ────────────────────────────────────────────────────────────


def _text_only_messages(*turns):
    """Kompakte Bau-Werkzeug: [user("a"), assistant("b"), user("c")]."""
    return [{"role": r, "content": t} for r, t in turns]


def _multimodal_user(text, image_url=None):
    blocks = []
    if image_url is not None:
        blocks.append({"type": "image", "image": image_url})
    blocks.append({"type": "text", "text": text})
    return {"role": "user", "content": blocks}


# ─── 1. Kern-Verhalten: keine zwei user-Turns ─────────────────────────


def test_no_double_user_turns_text_only():
    """Standardfall: messages=[user, assistant, user] + citmind.
    Nach merge: messages=[user(prefix+c0), assistant, user(c2)].
    Wichtig: user an Index 0 enthält das System-Prefix, kein separater
    user-Turn davor.→ Jinja 'must alternate' bleibt sauber."""
    messages = _text_only_messages(("user", "hi"), ("assistant", "hello"))
    out = merge_system_into_first_user(messages, "citmind")

    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant"], (
        f"unerwartete Rollen-Folge: {roles} — würde Jinja 'must alternate' werfen"
    )

    assert out[0]["role"] == "user"
    assert out[0]["content"].startswith(SYSTEM_SENTINEL)
    assert "hi" in out[0]["content"]
    # CitMind-core_philosophy aus docs/CitMind.txt sollte enthalten sein.
    assert "CitMind" in out[0]["content"] or len(out[0]["content"]) > len(SYSTEM_SENTINEL)


def test_no_double_user_turns_three_turns():
    """Bei drei echten Turns: [user, assistant, user] + Profil → keine
    user-Doppelung, System-Prefix auf den ersten user-turn."""
    messages = _text_only_messages(
        ("user", "frage 1"),
        ("assistant", "antwort 1"),
        ("user", "frage 2"),
    )
    out = merge_system_into_first_user(messages, "citmind")
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "user"]
    assert out[0]["content"].startswith(SYSTEM_SENTINEL)
    assert "frage 1" in out[0]["content"]
    # Zweiter user-turn unverändert (kein System-Prefix, da der bereits
    # gemerged wurde).
    assert not out[2]["content"].startswith(SYSTEM_SENTINEL)
    assert out[2]["content"] == "frage 2"


# ─── 2. system-Eintrag wird entfernt (Doppel-Vorhandensein vermeiden) ──


def test_strips_existing_system_entry():
    """Wenn messages schon [system, user, ...] enthält, wird der
    system-Eintrag entfernt (Inhalt ist ja gemerged)."""
    messages = [
        {"role": "system", "content": "[SYSTEM CONTEXT] alt"},
        {"role": "user", "content": "hi"},
    ]
    out = merge_system_into_first_user(messages, "citmind")
    roles = [m["role"] for m in out]
    assert "system" not in roles
    assert roles == ["user"]
    # System-Inhalt vom Profil citmind ist gemerged (nicht 'alt' erhalten).
    assert out[0]["content"].startswith(SYSTEM_SENTINEL)
    assert "hi" in out[0]["content"]


# ─── 3. Edit-Override hat Vorrang vor Profil ───────────────────────────


def test_edit_overrides_profile_text():
    """Wenn edit_text gesetzt, ersetzt er den Profil-Body komplett."""
    messages = _text_only_messages(("user", "hi"))
    out = merge_system_into_first_user(messages, "citmind", "Mein Custom-System")
    assert out[0]["content"].startswith(SYSTEM_SENTINEL)
    assert "Mein Custom-System" in out[0]["content"]
    assert "hi" in out[0]["content"]
    # CitMind sollte NICHT enthalten sein (Profil durch Edit ersetzt).
    # core_philosophy enthält aber das Wort "CitMind" — wir prüfen
    # daher nur, dass die Edit-Zeichenkette direkt nach dem Sentinel
    # steht.
    body = out[0]["content"][len(SYSTEM_SENTINEL):]
    assert body.startswith("Mein Custom-System")


# ─── 4. Neutral + kein Edit → keine Mutation ──────────────────────────


def test_neutral_no_edit_passes_through():
    """neutral + kein Edit → system_wrap="" → keine Merging-Mutation.
    (Rangordnung: Edit > Profil > neutral; neutral=no-op.)"""
    messages = _text_only_messages(("user", "hi"))
    out = merge_system_into_first_user(messages, "neutral", None)
    assert out == messages
    # Auch bei zusätzlichem system-Eintrag im Input wird er entfernt,
    # aber kein Inhalt gemerged.
    messages2 = [
        {"role": "system", "content": "[SYSTEM CONTEXT] alt"},
        {"role": "user", "content": "hi"},
    ]
    out2 = merge_system_into_first_user(messages2, "neutral", None)
    assert out2 == [{"role": "user", "content": "hi"}]


# ─── 5. Multimodal: Text-Block bekommt Prefix ─────────────────────────


def test_multimodal_text_block_gets_prefix():
    """Bei multimodalem user-content wird der Prefix in den ersten
    text-block gemerged, nicht in image-blocks."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": "/tmp/x.png"},
            {"type": "text", "text": "Was siehst du?"},
        ]},
    ]
    out = merge_system_into_first_user(messages, "citmind")
    assert out[0]["role"] == "user"
    content = out[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    # text-block enthält Prefix
    assert content[0]["type"] == "image"  # unverändert
    assert content[1]["type"] == "text"
    assert content[1]["text"].startswith(SYSTEM_SENTINEL)
    assert "Was siehst du?" in content[1]["text"]


def test_multimodal_no_text_block_inserts_one():
    """Wenn multimodaler user-content nur Bilder ohne text-block hat,
    wird ein neuer text-block vorne eingefügt (mit Prefix)."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": "/tmp/x.png"},
        ]},
    ]
    out = merge_system_into_first_user(messages, "citmind")
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"].startswith(SYSTEM_SENTINEL)
    assert content[1]["type"] == "image"


# ─── 6. Edge: keine user-Turns ───────────────────────────────────────


def test_no_user_turn_prepends_fresh_user():
    """Wenn messages keinen user-Turn hat (z.B. nur ein einzelner
    system-Eintrag), wird ein neuer user-Turn mit nur dem System-Inhalt
    prepended. So bleibt die Rollen-Folge gültig (single user)."""
    messages = [{"role": "system", "content": "alt"}]
    out = merge_system_into_first_user(messages, "citmind")
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert out[0]["content"].startswith(SYSTEM_SENTINEL)


# ─── 7. Keine Input-Mutation ─────────────────────────────────────────


def test_does_not_mutate_input():
    """Wichtige Invariante: die Eingabe-Liste wird nicht verändert.
    (Persistenz-Pfad in chat_tab ruft merge auf processed_messages; wenn
    das die messages-Liste für save_session verändern würde, wäre die
    Persistenz kaputt.)"""
    original = [
        {"role": "user", "content": "hi"},
    ]
    snapshot = json.dumps(original, sort_keys=True)
    _ = merge_system_into_first_user(original, "citmind")
    assert json.dumps(original, sort_keys=True) == snapshot


# ─── 8. Jinja-Kompatibilitäts-Smoke (kein "must alternate") ──────────


def test_role_sequence_valid_for_template_alternation():
    """Direkter Test der Rollen-Folge: keine zwei aufeinanderfolgenden
    gleichen Rollen (außer system-Einträge sind entfernt). Das ist die
    formale Bedingung, die Gemma3-Jinja einfordert."""
    messages = [
        {"role": "system", "content": "[SYSTEM CONTEXT] x"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    out = merge_system_into_first_user(messages, "citmind")
    roles = [m["role"] for m in out]
    for a, b in zip(roles, roles[1:]):
        assert a != b, f"ungültige Folge: {roles}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
