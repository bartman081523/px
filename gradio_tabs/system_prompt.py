"""gradio_tabs/system_prompt.py — pure-logic profile loader + system message
builder + injector for the chat-history list.

No Gradio imports. The orchestrator (chat_tab.py, streaming_bridge.py) wires
this module in: it calls ``build_system_message`` to obtain the structured
history entry for persistence, and ``render_for_chat_template`` to obtain the
wrapped first-user-turn string for ``apply_chat_template``.

Profiles are loaded lazily from ``docs/CitMind.txt`` and ``docs/Juexin.txt``
on first import. The loader is defensive against missing or unparseable
source docs: a missing profile falls back to ``"neutral"`` and never raises.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# Sentinel prefix the model sees in the rendered chat template. The
# `apply_chat_template(..., tokenize=False)` output starts with this line
# for the wrapped system message, so the model can recognise context-vs-user.
SYSTEM_SENTINEL = "[SYSTEM CONTEXT]\n"

# Paths to the source docs. Resolved relative to this file so the module is
# importable from any working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCS_DIR = os.path.abspath(os.path.join(_HERE, os.pardir, "docs"))
_CITMIND_DOC = os.path.join(_DOCS_DIR, "CitMind.txt")
_JUEXIN_DOC = os.path.join(_DOCS_DIR, "Juexin.txt")


def _make_neutral_profile() -> Dict[str, Any]:
    """Always-available empty profile."""
    return {
        "name": "neutral",
        "source_doc": None,
        "core_philosophy": "",
        "substrate_universal": "",
        "core_principles": [],
    }


def _load_profile_from_doc(doc_path: str, top_key: str, profile_name: str) -> Optional[Dict[str, Any]]:
    """Defensive loader: returns a profile dict or None if doc missing/unparseable.

    Tolerant of minor structural drift:
      - file missing / unreadable -> None
      - top-level key missing -> None
      - core_philosophy / substrate_universal / core_principles missing -> empty defaults
    """
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    inner = data.get(top_key)
    if not isinstance(inner, dict):
        return None
    cp = inner.get("core_philosophy", "")
    su = inner.get("substrate_universal", "")
    princ = inner.get("core_principles", [])
    if not isinstance(cp, str):
        cp = ""
    if not isinstance(su, str):
        su = ""
    if not isinstance(princ, list):
        princ = []
    # Keep only string principles
    princ = [p for p in princ if isinstance(p, str)]
    return {
        "name": profile_name,
        "source_doc": doc_path,
        "core_philosophy": cp,
        "substrate_universal": su,
        "core_principles": princ,
    }


def _build_profiles() -> Dict[str, Dict[str, Any]]:
    """Build the profile dict at module load time. Always includes "neutral"."""
    profiles: Dict[str, Dict[str, Any]] = {"neutral": _make_neutral_profile()}
    cit = _load_profile_from_doc(_CITMIND_DOC, "CitMind", "citmind")
    if cit is not None:
        profiles["citmind"] = cit
    jue = _load_profile_from_doc(_JUEXIN_DOC, "Juexin", "juexin")
    if jue is not None:
        profiles["juexin"] = jue
    return profiles


# Built lazily on first import. Always contains "neutral"; contains "citmind"
# / "juexin" iff the corresponding source doc parsed successfully.
PROFILES: Dict[str, Dict[str, Any]] = _build_profiles()


def list_profiles() -> List[str]:
    """Return the names of all available profiles, in a stable order:
    ``["neutral", "citmind", "juexin"]`` if the source docs parse, else
    ``["neutral"]`` only (or whatever subset is available)."""
    ordered = ["neutral", "citmind", "juexin"]
    return [n for n in ordered if n in PROFILES]


def resolve_profile(name: str) -> Dict[str, Any]:
    """Return the profile dict for ``name``. Unknown names fall back to
    ``"neutral"`` (NEVER raise — the orchestrator may pass user-typed values).
    The returned dict is a fresh shallow copy so callers may mutate it without
    affecting subsequent calls."""
    base = PROFILES.get(name) or PROFILES["neutral"]
    out = dict(base)
    # Fresh list copy so callers may mutate core_principles freely.
    out["core_principles"] = list(base.get("core_principles", []))
    return out


def _render_profile_text(profile: Dict[str, Any], edit_text: Optional[str]) -> str:
    """Render the post-sentinel body for a profile (or the edit override).

    If ``edit_text`` is not None and not empty after strip, returns the
    stripped edit text. Otherwise returns
        core_philosophy + "\n\n" + substrate_universal
        + ("\n\n" + bullet-list of core_principles if any).
    """
    if edit_text is not None:
        stripped = edit_text.strip()
        if stripped:
            return stripped
    parts: List[str] = []
    cp = profile.get("core_philosophy", "")
    su = profile.get("substrate_universal", "")
    princ = profile.get("core_principles", []) or []
    if cp:
        parts.append(cp)
    if su:
        parts.append(su)
    if princ:
        bullet = "\n".join("- " + p for p in princ)
        parts.append(bullet)
    return "\n\n".join(parts)


def build_system_message(profile_name: str, edit_text: Optional[str] = None) -> Dict[str, str]:
    """Build the system-prompt history entry to be prepended to the messages list.

    - ``profile_name``: name from ``list_profiles()`` (unknown -> neutral fallback).
    - ``edit_text``: if not None and not empty after strip, REPLACES the
      profile's core_philosophy text with this edit. Rendered text is then
      exactly ``SYSTEM_SENTINEL + edit_text.strip()``.
    - else: ``SYSTEM_SENTINEL + core_philosophy + "\\n\\n" + substrate_universal``
      + (``"\\n\\n" + bullet list of core_principles`` if any).

    Returns ``{"role": "system", "content": <str>}``. For ``"neutral"`` with
    no edit, content is empty (the orchestrator omits empty system messages).
    """
    profile = resolve_profile(profile_name)
    body = _render_profile_text(profile, edit_text)
    content = SYSTEM_SENTINEL + body if body else ""
    return {"role": "system", "content": content}


def inject_into_messages(messages: List[Dict[str, Any]], profile_name: str,
                         edit_text: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return a NEW list with the system message (if any) inserted at index 0.

    - If the resolved system message has empty content, return messages
      unchanged (no spurious empty system entry). The returned list IS a
      shallow copy so callers do not observe future input mutations.
    - Strips ALL existing ``{"role":"system", ...}`` entries (not just
      index 0). Persisted sessions can have multiple system entries
      (e.g. one stored as a multimodal-list from old chat_tab history,
      and another added later). They all get replaced by the current
      profile+edit. This also avoids the ``'list' object has no
      attribute 'replace'`` crash when downstream code tries to slice
      a multimodal-list system entry.
    - The input list is not mutated.
    - Robust to multimodal-list content in existing messages (preserved
      for non-system entries).
    """
    out = [m for m in messages
           if not (isinstance(m, dict) and m.get("role") == "system")]
    sys_msg = build_system_message(profile_name, edit_text)
    if not sys_msg["content"]:
        return out
    out.insert(0, sys_msg)
    return out


def render_for_chat_template(profile_name: str, edit_text: Optional[str] = None) -> str:
    """Wrap the system message into a USER-TURN string for ``apply_chat_template``.

    The Gemma3 chat template has no ``system`` role, so the orchestrator:
      1. Calls ``build_system_message`` -> save as history entry for persistence.
      2. Calls ``render_for_chat_template`` -> insert as the FIRST user-turn
         before the real user message in the list passed to
         ``apply_chat_template``.

    Returns the wrapped text including ``SYSTEM_SENTINEL`` prefix. Empty
    string if nothing to render (neutral + no edit). The returned string
    ends with a newline if non-empty, so concat with user-message text is
    clean.

    Consistency: ``render_for_chat_template(name) == build_system_message(name)["content"] + "\\n"``.
    """
    profile = resolve_profile(profile_name)
    body = _render_profile_text(profile, edit_text)
    if not body:
        return ""
    return SYSTEM_SENTINEL + body + "\n"


def merge_system_into_first_user(messages: List[Dict[str, Any]],
                                 profile_name: str,
                                 edit_text: Optional[str] = None
                                 ) -> List[Dict[str, Any]]:
    """Build a NEW ``apply_chat_template``-safe messages list by merging the
    system-prompt content into the FIRST user-turn as a prefix.

    Why this exists
    ---------------
    Gemma3's chat-template (Jinja) enforces strict user/assistant alternation
    AND has no ``system`` role. Two failure modes were observed before:

    1. Passing ``{"role":"system", ...}`` -> template silently ignores it
       (no system context reaches the model).
    2. Prepending a fresh ``{"role":"user", content: <system>}`` entry ->
       two consecutive user-turns -> Jinja raises
       ``"Conversation roles must alternate user/assistant/user/assistant/..."``.

    Correct strategy
    ----------------
    Take the existing first user-turn, prepend the system-prompt (including
    sentinel) to its text content, and DROP any ``{"role":"system"}`` entry
    from the list (it is now physically merged into the user content). For
    multimodal-list content, the prefix is merged into the first text-block
    (or a new text-block is prepended if none exists).

    Parameters
    ----------
    messages : list of message dicts (will NOT be mutated; a new list is returned).
    profile_name, edit_text : same semantics as ``build_system_message``.

    Returns
    -------
    A NEW list. The first user-turn (if any) now carries the system-prompt
    as a text-prefix. Any ``{"role":"system"}`` entries are removed (since
    their content is now embedded in the user-turn). If no system-prompt
    content resolves (neutral + no edit), the list is returned unchanged
    except for any ``system`` entries being stripped.
    """
    system_wrap = render_for_chat_template(profile_name, edit_text)
    # Work on a shallow copy; preserve inner message dicts by reference unless
    # we need to rewrite one.
    out = list(messages)

    # 1) Strip existing system entries — we will re-merge the content.
    out = [m for m in out if not (isinstance(m, dict) and m.get("role") == "system")]

    # 2) If no system content to merge, return the cleaned list as-is.
    if not system_wrap:
        return out

    # 3) Find first user-turn; if none, prepend a fresh user-turn carrying
    #    only the system content (preserves "must alternate" because no
    #    consecutive user-turns are produced).
    user_idx = next(
        (i for i, m in enumerate(out) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if user_idx is None:
        return [{"role": "user", "content": system_wrap}] + out

    um = out[user_idx]
    uc = um.get("content")
    if isinstance(uc, list):
        # Multimodal list: prefix into the first text-block.
        new_content = list(uc)
        txt_i = next(
            (i for i, c in enumerate(new_content)
             if isinstance(c, dict) and c.get("type") == "text"),
            None,
        )
        prefix_block = {"type": "text", "text": system_wrap}
        if txt_i is None:
            new_content.insert(0, prefix_block)
        else:
            old_text = new_content[txt_i].get("text", "") or ""
            new_content[txt_i] = {
                "type": "text",
                "text": system_wrap + old_text,
            }
        out[user_idx] = {**um, "content": new_content}
    else:
        old_text = uc if isinstance(uc, str) else ("" if uc is None else str(uc))
        out[user_idx] = {**um, "content": system_wrap + old_text}

    return out


def append_tag_snippet(messages: List[Dict[str, Any]],
                       snippet: str) -> List[Dict[str, Any]]:
    """Hängt einen Vocoder-Tag-Snip an den **ersten** system-Eintrag an.

    Zweck
    -----
    Wenn der User die Auto-Vocoder-Option aktiviert, soll das LLM
    [#A2] / [#PAUSE] / [#WHISPER] etc. in seine Antwort einbetten. Diese
    Tags sind orthogonal zum CitMind-/Juexin-Profil — wir mischen sie
    nicht, sondern hängen den Snip an die existierende System-Message
    an (oder prepended einen neuen system-Eintrag, wenn keiner da ist).

    Rangordnung (deterministisch)
    -----------------------------
    - Wenn der User-Edit-Text gesetzt ist, schreibt der edit_text den
      Profil-Text KOMPLETT neu (siehe ``build_system_message``). Der
      Tag-Snip wird trotzdem angehängt — Vocoder-Anweisungen sind
      orthogonal zur Frame-Ausrichtung.
    - Profil + Edit werden vom ``inject_into_messages``-Pfad gesetzt;
      ``append_tag_snippet`` läuft danach und ergänzt nur.
    - Wird vom Aufrufer ``chat_tab.py`` gesteuert; ``streaming_bridge``
      kann den Snip analog anhängen.

    Verhalten
    ---------
    - Mutiert ``messages`` NICHT (gibt eine neue Liste zurück).
    - Wenn kein system-Eintrag existiert: prepended einen frischen
      mit nur dem Snippet.
    - Wenn System-Inhalt leer ist: ersetze ihn durch den Snippet (kein
      leerer Eintrag übrig).
    - Wenn System-Inhalt + Snippet beide da sind: append mit ``"\n\n"``
      als Trenner (lesbar).
    - Multimodal-Inhalte werden respektiert (System-Inhalt ist hier
      immer str, nicht list — aber defensive check).

    Parameters
    ----------
    messages : list of dicts. Wird nicht mutiert.
    snippet : str. Wenn leer oder None → no-op (messages unverändert
              zurück, nur shallow-copy).

    Returns
    -------
    list of dicts (NEUE Liste).
    """
    if not snippet:
        # Kein Snip → Liste shallow-copieren, sonst nichts.
        return list(messages)

    out = list(messages)  # shallow copy; System-Eintrag wird evtl. ersetzt.

    sys_idx = next(
        (i for i, m in enumerate(out)
         if isinstance(m, dict) and m.get("role") == "system"),
        None,
    )

    if sys_idx is None:
        # Kein System-Eintrag → frischen mit nur Snippet prependen.
        return [{"role": "system", "content": snippet}] + out

    sys_msg = out[sys_idx]
    existing = sys_msg.get("content") or ""
    if not isinstance(existing, str):
        # Defensive: Multimodal-System? Unüblich, aber wir serialisieren.
        existing = str(existing)

    new_content = (existing + "\n\n" + snippet) if existing else snippet
    out[sys_idx] = {**sys_msg, "content": new_content}
    return out