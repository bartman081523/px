"""
sessions.py — Session Management for PX Explorer
================================================
Handles saving/loading chat histories + per-session settings to disk.
"""

import os
import json
import uuid
import tempfile
import threading
from typing import List, Dict, Any, Optional

SESSION_DIR = "sessions"

# ── Settings-Defaults ────────────────────────────────────────────────────
# Plan ui-styling, 2026-07-06, "Einstellungen-Tab + Persistenz".
# Diese 13 Felder werden in jedem session.json unter dem key "settings"
# persistiert. Bewusst NICHT enthalten: tts_engine / tts_sample_rate /
# tts_auto — die waren auf wip-tts migriert, gehören aber NICHT in den
# aktuellen Einstellungen-Tab (Out-of-Scope). Siehe Plan-Doc.
SETTINGS_DEFAULTS: Dict[str, Any] = {
    "model_id": "gemma3-1b-it",
    "px_preset": "ACTIVE_MANIFOLD",
    "auto_tune": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 1024,
    "rep_p": 1.15,
    "px_gamma": 0.08,
    "relay_sign": 0,
    "relay_alpha": 0.30,
    "relay_layer": 21,
    "system_profile": "neutral",
    "system_prompt_text": "",
}

# Per-session-id Lock (verhindert Race-Conditions bei update_settings
# aus mehreren Threads). Lazy-erzeugt in _get_lock().
_SESSION_LOCKS: Dict[str, threading.RLock] = {}
_SESSION_LOCKS_META = threading.Lock()


def _get_lock(session_id: str) -> threading.RLock:
    """Returnt (und cached) einen per-session-id Lock."""
    with _SESSION_LOCKS_META:
        if session_id not in _SESSION_LOCKS:
            _SESSION_LOCKS[session_id] = threading.RLock()
        return _SESSION_LOCKS[session_id]


def ensure_session_dir():
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Schreibt <data> nach <path> atomar (tempfile + os.replace).

    Bei Crash mid-write bleibt die alte Datei erhalten — das ist die T8-
    Atomic-Write-Semantik. Wenn os.replace crasht, ist nur die tmp-Datei
    korrupt; das Ziel ist unangetastet.
    """
    ensure_session_dir()
    dir_name = os.path.dirname(path) or "."
    # tempfile in der gleichen dir (sonst kann os.replace cross-device failen)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_session_", suffix=".json", dir=dir_name
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        # Bei jedem Fehler: tmp aufräumen, alte Datei bleibt
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_session(session_id: str, history: List[Dict[str, Any]],
                 model_id: str = None,
                 settings: Optional[Dict[str, Any]] = None) -> str:
    """Save session history (and optional settings) to a JSON file.

    Backward-compatible: ohne <settings> wird der "settings" key GAR NICHT
    geschrieben (T1-Pin) — so bleiben alte session.json-Dateien unverändert
    lesbar. Mit <settings> wird er unter dem key persistiert (T2-Pin).
    """
    ensure_session_dir()
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    data: Dict[str, Any] = {
        "session_id": session_id,
        "model_id": model_id,
        "history": history,
        "updated_at": str(os.times()[4])  # simplified timestamp
    }
    if settings is not None:
        data["settings"] = settings
    _atomic_write_json(path, data)
    return path


def load_session(session_id: str) -> Dict[str, Any]:
    """Load session data from a JSON file.

    Alte session.json-Dateien (ohne "settings" key) returnen unverändert
    (T6-Pin via .get("settings", {})): Aufrufer können .get("settings", {})
    benutzen und kriegen {} wenn der key fehlt. Wir setzen den key NICHT
    automatisch in die Rückgabe-Dict, weil T1 fordert dass save_session ohne
    settings-Argument auch KEIN "settings"-key in die JSON schreibt.
    """
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return {"session_id": session_id, "history": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_settings(session_id: str, **patch: Any) -> None:
    """Merged <patch> in den 'settings'-key der session.json (atomar).

    Verhalten:
        - Lädt die aktuelle session.json (oder erstellt einen minimalen
          Eintrag, falls die Datei nicht existiert — T4-Pin).
        - Merged patch in data["settings"] (überschreibt gleiche Keys).
        - Schreibt zurück via _atomic_write_json (tempfile + os.replace).
        - Hält einen per-session Lock (T5-Pin: thread-safe).

    Akzeptiert JEDEN JSON-kompatiblen Wert (T7-Pin: kein type-coercion,
    int 123 wird als 123 persistiert, nicht zu "123" gecastet). Das
    ermöglicht tests, garbage reinzuschieben, ohne dass die Persistenz crasht.
    """
    if not session_id:
        return

    lock = _get_lock(session_id)
    with lock:
        path = os.path.join(SESSION_DIR, f"{session_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # T4: minimaler Eintrag für nicht-existierende sessions
            data = {
                "session_id": session_id,
                "model_id": None,
                "history": [],
                "updated_at": str(os.times()[4]),
            }

        # Merge in den "settings" key
        existing_settings = data.get("settings", {})
        if not isinstance(existing_settings, dict):
            existing_settings = {}
        existing_settings.update(patch)
        data["settings"] = existing_settings

        _atomic_write_json(path, data)


def list_sessions() -> List[str]:
    """List all available session IDs, sorted by modification time (newest first)."""
    ensure_session_dir()
    files = [f for f in os.listdir(SESSION_DIR) if f.endswith(".json")]
    # Sort by mtime
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SESSION_DIR, x)), reverse=True)
    return [f.replace(".json", "") for f in files]


def get_new_session_id() -> str:
    return str(uuid.uuid4().hex[:8])
