"""gradio_tabs/system_prompt.py — Pure-Logic Profile-Loader + System-Message-Builder.

Plan: branch ui-styling, 2026-07-06, "Einstellungen-Tab + Persistenz".

Schicht 1 des Plans: lädt Profile aus docs/CitMind.txt + docs/Juexin.txt
(lazy + defensive, fehlende/kaputte Dateien → Fallback "neutral", kein Crash),
baut daraus System-Messages, injiziert sie in History-Listen (robust gegen
Multimodal-Listen-content), und liefert einen prefill-String für
tokenizer.apply_chat_template wenn Merge-into-first-user gebraucht wird.

Öffentliche API (alle pure functions, keine Gradio-Imports):
    list_profiles() -> List[str]
    resolve_profile(name: str) -> Dict[str, Any]
    build_system_message(profile_name, edit_text=None) -> Dict[str, str]
    inject_into_messages(messages, profile_name, edit_text=None) -> List[Dict]
    render_for_chat_template(profile_name, edit_text, user_message) -> str

Frame = Orientierer, nicht 观-Produzent.
"""
from __future__ import annotations
import copy
import json
import os
from typing import Any, Dict, List, Optional

# ── Konfiguration ─────────────────────────────────────────────────────────

# Profil-Name → relativer Pfad zu docs/. Schlüssel = profile-name (klein
# geschrieben, so wie sie in list_profiles() zurückkommen). Defensive:
# fehlende Files werden zur Laufzeit übersprungen, nicht hier gecrasht.
PROFILE_DOCS: Dict[str, str] = {
    "citmind": "docs/CitMind.txt",
    "juexin": "docs/Juexin.txt",
}

# Sentinel, das jeder injizierte System-Content einleitet. Wird auch in
# render_for_chat_template genutzt, damit das Modell den Block sicher als
# System-Kontext erkennt (kein normaler User-Content).
SYSTEM_SENTINEL: str = "[SYSTEM CONTEXT]\n"

# "neutral" hat keinen Doc — body bleibt leer.
NEUTRAL_BODY: str = ""


# ── Profile-Loader (lazy + defensive) ─────────────────────────────────────

# Cache: { "citmind": {dict}, "juexin": {dict}, "neutral": {dict} }
# Verhindert Disk-IO bei jedem Aufruf und stellt resolve_profile-Konsistenz her.
_PROFILE_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_profile_from_doc(name: str, rel_path: str, project_root: str) -> Dict[str, Any]:
    """Liest + parst ein einzelnes Profil-Doc defensiv.

    Gibt zurück: {"name": <key>, "source_doc": <rel_path>, "core_philosophy": str,
                  "substrate_universal": str, "core_principles": list[str], ...}
    Bei jedem Fehler (FileNotFound, JSON-Error, kaputte Struktur): Fallback
    auf "neutral" (mit leerem body), kein raise.
    """
    abs_path = os.path.join(project_root, rel_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except (OSError, IOError, UnicodeDecodeError):
        return _neutral_profile(source_doc=rel_path)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _neutral_profile(source_doc=rel_path)

    # Top-Key = Profilname (z.B. "CitMind" für docs/CitMind.txt). Wir
    # probieren mehrere case-Varianten, weil die docs historisch
    # inkonsistent sind (CitMind.txt hat "CitMind", Juexin.txt hat
    # "Juexin" — beide capitalized). Fallback: erster dict-key.
    if not isinstance(data, dict):
        return _neutral_profile(source_doc=rel_path)

    candidates = [name, name.capitalize(), name.upper(),
                  name[0].upper() + name[1:] if len(name) > 1 else name.upper()]
    inner = None
    for cand in candidates:
        if cand in data and isinstance(data[cand], dict):
            inner = data[cand]
            break
    if inner is None:
        # Fallback: erster dict-value der ein dict ist
        for v in data.values():
            if isinstance(v, dict):
                inner = v
                break
    if inner is None:
        return _neutral_profile(source_doc=rel_path)

    # Coerce Felder defensiv auf die erwarteten Typen. Wenn core_philosophy
    # z.B. als list kommt (Bug im Doc), wird es zu "" coerced.
    cp = inner.get("core_philosophy", "")
    su = inner.get("substrate_universal", "")
    pr = inner.get("core_principles", [])

    return {
        "name": name,
        "source_doc": rel_path,
        "core_philosophy": cp if isinstance(cp, str) else "",
        "substrate_universal": su if isinstance(su, str) else "",
        "core_principles": pr if isinstance(pr, list) else [],
    }


def _neutral_profile(source_doc: Optional[str] = None) -> Dict[str, Any]:
    """Das neutrale/leere Profil. Wird auch als Fallback für unbekannte Namen genutzt."""
    return {
        "name": "neutral",
        "source_doc": source_doc or "",
        "core_philosophy": "",
        "substrate_universal": "",
        "core_principles": [],
    }


def _ensure_cache() -> Dict[str, Dict[str, Any]]:
    """Lazy-load aller Profile aus den Docs. Resultat ist gecached (file-local module cache)."""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE

    # Project root = zwei Ebenen über diesem File (gradio_tabs/system_prompt.py)
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)

    cache: Dict[str, Dict[str, Any]] = {"neutral": _neutral_profile()}
    for name, rel_path in PROFILE_DOCS.items():
        profile = _load_profile_from_doc(name, rel_path, project_root)
        # Nur hinzufügen, wenn das Doc erfolgreich geladen wurde (also der
        # Loader NICHT auf neutral zurückgefallen ist). Sonst erscheint das
        # Profil nicht in list_profiles().
        if profile["name"] == name:
            cache[name] = profile
        # Wenn der Loader auf neutral zurückfiel, taucht das Profil nicht auf.
        # list_profiles() enthält dann nur ["neutral"].
    _PROFILE_CACHE = cache
    return _PROFILE_CACHE


# ── Public API ────────────────────────────────────────────────────────────

def list_profiles() -> List[str]:
    """Liste der verfügbaren Profil-Namen. Reihenfolge: ["neutral", <weitere alphabetisch>].

    Stabil: zwei Aufrufe liefern den gleichen Output. Deterministisch: keine
    Random-Order, keine OS-Order, sondern unsere kanonische Reihenfolge.
    """
    cache = _ensure_cache()
    # neutral zuerst, dann die übrigen in Insertion-Order von PROFILE_DOCS
    names = ["neutral"]
    for key in PROFILE_DOCS.keys():
        if key in cache and cache[key]["name"] == key:
            names.append(key)
    return names


def resolve_profile(name: str) -> Dict[str, Any]:
    """Returnt Profil-Dict für <name>. Unbekannte Namen → "neutral" Fallback.

    Wichtig: returnt eine DEEP COPY, damit Mutationen am Resultat (z.B. ein
    UI-Bug der core_principles.append(...) macht) den globalen Cache nicht
    korrumpiert. Test_pin: "resolve_profile returnt Kopie".
    """
    cache = _ensure_cache()
    if name in cache and cache[name]["name"] == name:
        return copy.deepcopy(cache[name])
    return copy.deepcopy(cache["neutral"])


def _render_profile_body(profile: Dict[str, Any]) -> str:
    """Rendert einen Profil-Dict zu einem lesbaren Body-String.

    Struktur:
        [SYSTEM CONTEXT]
        === Core Philosophy ===
        <core_philosophy>

        === Substrate Universal ===
        <substrate_universal>

        === Core Principles ===
        - <p1>
        - <p2>
        ...

    Bei leerem Profil (neutral) wird "" zurückgegeben.
    """
    cp = (profile.get("core_philosophy") or "").strip()
    su = (profile.get("substrate_universal") or "").strip()
    pr = profile.get("core_principles") or []

    if not cp and not su and not pr:
        return NEUTRAL_BODY

    parts = [SYSTEM_SENTINEL.rstrip("\n")]

    if cp:
        parts.append("=== Core Philosophy ===")
        parts.append(cp)
    if su:
        if len(parts) > 1:
            parts.append("")
        parts.append("=== Substrate Universal ===")
        parts.append(su)
    if pr:
        if len(parts) > 1:
            parts.append("")
        parts.append("=== Core Principles ===")
        for principle in pr:
            if isinstance(principle, str) and principle.strip():
                parts.append(f"- {principle.strip()}")

    return "\n".join(parts)


# ── Preset → Profil Mapping (Plan 2026-07-08) ─────────────────────────
# Wenn der User in der UI einen px_preset auswählt, lädt der Sidebar-
# Handler automatisch den passenden System-Prompt. Mapping:
#   BASELINE              → neutral   (kein Frame, vanilla)
#   ACTIVE_MANIFOLD       → citmind   (PX-Frame)
#   ACTIVE_MANIFOLD_LEAN  → citmind   (LEAN-Frame)
#   ACTIVE_MANIFOLD_RELAY → juexin    (Kontemplations-Frame, RELAY)
# Unbekannte Presets / None / "" → neutral (defensiv).

PRESET_TO_PROFILE: Dict[str, str] = {
    "BASELINE": "neutral",
    "ACTIVE_MANIFOLD": "citmind",
    "ACTIVE_MANIFOLD_LEAN": "citmind",
    "ACTIVE_MANIFOLD_RELAY": "juexin",
}


def preset_to_profile(preset: Optional[str]) -> str:
    """Mappt ein px_preset auf einen System-Prompt-Profilnamen.

    Public-API für UI-Handler (z.B. px_preset.change() in chat_tab.py).
    Unbekannte Presets fallen defensiv auf "neutral" zurück — das ist
    der "kein Frame"-Zustand, gleicher Default wie list_profiles()[0].
    """
    if not preset:
        return "neutral"
    return PRESET_TO_PROFILE.get(preset, "neutral")


def load_profile_for_preset(preset: Optional[str]) -> str:
    """Returnt den gerenderten Profile-Body für das passende Profil.

    Sidebar-Handler ruft diese Funktion auf wenn px_preset wechselt, und
    lädt das Ergebnis in die System-Prompt-Textarea. Der User kann den
    Text dann frei editieren.

    Returnt "" für BASELINE (neutral = leerer Body), nicht-leeren String
    für citmind/juexin.
    """
    profile_name = preset_to_profile(preset)
    if profile_name == "neutral":
        return NEUTRAL_BODY
    profile = resolve_profile(profile_name)
    return _render_profile_body(profile)


def load_profile_body(profile_name: Optional[str]) -> str:
    """Returnt den gerenderten Profile-Body für einen Profilnamen.

    Plan 2026-07-09: system_profile.change-Handler. User klickt in der
    Sidebar auf einen Profil-Eintrag (juexin/citmind/neutral) → der
    gerenderte Body dieses Profils landet in der system_prompt_text-
    Textarea. Wird vom Edit überschrieben, sobald der User was tippt.

    Unbekannte Namen / None / "" → "" (neutral, leerer Body).
    """
    if not profile_name:
        return NEUTRAL_BODY
    profile = resolve_profile(profile_name)
    return _render_profile_body(profile)


def build_system_message(profile_name: str, edit_text: Optional[str] = None) -> Dict[str, str]:
    """Baut die finale System-Message.

    Logik:
        1. Wenn edit_text nicht-leer (whitespace ≠ leer), wird der EDIT-Text
           als body verwendet (er überschreibt das Profil).
        2. Sonst: Profil-Body aus resolve_profile(name) rendern.
        3. role ist immer "system" (T9-Pin).

    Returns: {"role": "system", "content": "<body>"}
    """
    profile = resolve_profile(profile_name)
    edit = (edit_text or "").strip()

    if edit:
        # Edit überschreibt Profil-Text
        body = f"{SYSTEM_SENTINEL}{edit}"
    else:
        body = _render_profile_body(profile)

    return {"role": "system", "content": body}


def inject_into_messages(
    messages: List[Dict[str, Any]],
    profile_name: str,
    edit_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Injiziert eine System-Message an Index 0, strippt ALLE existing system-entries.

    Edge cases:
        - Bei "neutral" + kein edit: returnt die Original-Liste (T10-Pin).
          Heißt: wenn weder Profil noch Edit was liefern, MUSS die Liste
          unverändert bleiben (sonst entstehen leere System-Einträge die
          Modelle verwirren).
        - Strippt ALLE System-Einträge (egal ob string-content oder Multimodal-
          Listen-content, T13-Pin). Wenn der strip eines Eintrags fehlschlägt
          (z.B. content ist ein unhandled Typ), wird er trotzdem gestrippt
          (Crash-Pin aus dem historischen "list object has no attribute
          replace"-Bug).
        - Mutiert die Input-Liste NICHT (T14-Pin): neue Liste, neue Dicts.
    """
    if not isinstance(messages, list):
        return []

    system_msg = build_system_message(profile_name, edit_text)

    # Wenn der body leer ist (neutral + kein edit), überspringen wir die
    # Injektion. Das ist der "no-op" Pfad.
    if not system_msg["content"]:
        # Trotzdem: ALLE existing system-entries strippen? Hier NEIN — wenn
        # es nichts zu injizieren gibt und die History schon einen System-
        # Eintrag hat, lassen wir ihn. Das ist die ursprüngliche Semantik
        # aus chat_tab (kein Patch gewünscht). T10-Pin: Original-Liste
        # unverändert returnen.
        return list(messages)

    # Alle existing system-entries strippen (T12 + T13).
    cleaned: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system":
            # strip — keine Annahmen über content-Typ
            continue
        # Kopie des Eintrags, damit die Input-Liste nicht geteilt wird (T14)
        cleaned.append(dict(m))

    cleaned.insert(0, system_msg)
    return cleaned


def render_for_chat_template(
    profile_name: str,
    edit_text: Optional[str],
    user_message: str,
) -> str:
    """Baut einen Prefill-String für tokenizer.apply_chat_template.

    Pattern: <system-body>\\n\\n<user_message>. Wenn der body leer ist
    (neutral + kein edit), wird nur <user_message> zurückgegeben.
    """
    msg = build_system_message(profile_name, edit_text)
    body = msg["content"]

    if not body:
        return user_message

    return f"{body}\n\n{user_message}"


# ── Test-Hook: Cache invalidieren ─────────────────────────────────────────

def _reset_cache_for_tests() -> None:
    """Setzt den Module-Cache zurück. Nur für Tests — wird in production
    nicht aufgerufen. Nützlich wenn ein Test die docs/ files mocked."""
    global _PROFILE_CACHE
    _PROFILE_CACHE = None
