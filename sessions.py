"""
sessions.py — Session Management for PX Explorer
================================================
Handles saving/loading chat histories to disk and JSON import/export.

Plan 5.3 extends the schema with a `settings` dict (16 fields) that
round-trips with the Web-UI sidebar (model, PX preset, sliders, relay,
system prompt, TTS engine). Atomic writes via tmp+os.replace protect
against corruption from concurrent live-updates.
"""

import os
import json
import tempfile
import threading
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple

SESSION_DIR = "sessions"

# Single source of truth for chat-settings defaults. Persisted sessions
# carry a full copy of these fields under `data["settings"]`. Older
# sessions without a settings block (or with missing keys) are merged
# with these defaults on load.
SETTINGS_DEFAULTS: Dict[str, Any] = {
    "model_id": "gemma3-1b",
    "px_preset": "ACTIVE_MANIFOLD",
    "auto_tune": False,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 1024,
    "rep_p": 1.15,
    "px_gamma": 0.5,
    "relay_sign": 1,
    "relay_alpha": 0.0,
    "relay_layer": 16,
    "system_profile": "ASSISTANT",
    "system_prompt_text": "",
    "tts_engine": "piper",
    "tts_sample_rate": 22050,
    "tts_auto": True,
}
SETTINGS_FIELDS: Tuple[str, ...] = tuple(SETTINGS_DEFAULTS.keys())

# Per-session write locks. Held only around the read-modify-write of a
# single session file. Concurrent reads are unconstrained.
_locks: Dict[str, threading.Lock] = {}
_locks_meta_lock = threading.Lock()


def _get_lock(session_id: str) -> threading.Lock:
    with _locks_meta_lock:
        if session_id not in _locks:
            _locks[session_id] = threading.Lock()
        return _locks[session_id]


def ensure_session_dir() -> None:
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)


def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON atomically via tmp+os.replace (POSIX atomic).

    On any failure the tmp file is removed before raising — never leaves
    a half-written ``path`` behind. ``os.replace`` is atomic on POSIX
    filesystems (single inode rename on the same partition).
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=".save_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _settings_with_defaults(settings: Optional[dict]) -> Dict[str, Any]:
    """Return a settings dict with SETTINGS_DEFAULTS filled in for any
    missing keys. Unknown keys in ``settings`` are preserved (forward
    compatibility)."""
    merged: Dict[str, Any] = dict(SETTINGS_DEFAULTS)
    if settings:
        for k, v in settings.items():
            merged[k] = v
    return merged


def save_session(session_id: str,
                  history: List[Dict[str, Any]],
                  model_id: Optional[str] = None,
                  settings: Optional[dict] = None) -> str:
    """Persist history + settings atomically. Returns the file path.

    Behavior:
      - Merge new ``settings`` over existing on-disk settings, then over
        SETTINGS_DEFAULTS (any missing key gets the default).
      - Merge new ``model_id`` over existing (None leaves prior model_id).
      - If no session file exists yet, ``model_id`` falls back to
        SETTINGS_DEFAULTS["model_id"] when not provided.
      - Lock per session_id: safe against concurrent
        save_session + update_settings calls.
    """
    ensure_session_dir()
    lock = _get_lock(session_id)
    with lock:
        path = os.path.join(SESSION_DIR, f"{session_id}.json")
        existing = load_session(session_id) or {}
        merged_settings = _settings_with_defaults(
            {**(existing.get("settings") or {}), **(settings or {})}
        )
        merged_model_id = (
            model_id
            if model_id is not None
            else existing.get("model_id")
            or SETTINGS_DEFAULTS["model_id"]
        )
        data = {
            "session_id": session_id,
            "model_id": merged_model_id,
            "history": history,
            "settings": merged_settings,
            "updated_at": time.time(),
        }
        _atomic_write_json(path, data)
        return path


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session data from a JSON file. Returns None if missing.

    Always returns a dict with a `settings` key (merged with defaults if
    the on-disk file predates the settings schema).
    """
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Backward-compat: fill missing settings with defaults.
    data["settings"] = _settings_with_defaults(data.get("settings"))
    return data


def update_settings(session_id: str, **patch: Any) -> None:
    """Atomically merge ``patch`` into the session's settings.

    - Unknown keys (not in SETTINGS_DEFAULTS) are silently ignored
      (forward-compatible callers can pass extras).
    - If the session file does not exist, creates one with default
      history=[], model_id=SETTINGS_DEFAULTS["model_id"] and the patched
      settings applied.
    - Lock per session_id prevents torn writes when called concurrently
      from the SettingsDebouncer thread and the chat-history save path.
    """
    lock = _get_lock(session_id)
    with lock:
        path = os.path.join(SESSION_DIR, f"{session_id}.json")
        data = load_session(session_id)
        if data is None:
            data = {
                "session_id": session_id,
                "model_id": SETTINGS_DEFAULTS["model_id"],
                "history": [],
                "settings": dict(SETTINGS_DEFAULTS),
            }
        data.setdefault("settings", dict(SETTINGS_DEFAULTS))
        for k, v in patch.items():
            if k in SETTINGS_DEFAULTS:
                data["settings"][k] = v
        data["updated_at"] = time.time()
        _atomic_write_json(path, data)


def list_sessions() -> List[str]:
    """List all available session IDs, sorted by modification time (newest first)."""
    ensure_session_dir()
    files = [f for f in os.listdir(SESSION_DIR) if f.endswith(".json")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SESSION_DIR, x)),
               reverse=True)
    return [f.replace(".json", "") for f in files]


def list_session_mtimes() -> List[Tuple[float, str]]:
    """Return [(mtime, session_id), ...] for all session files, newest first.

    Excludes `active.json` (Gradio internal marker). Empty list if the
    sessions directory does not exist.
    """
    if not os.path.isdir(SESSION_DIR):
        return []
    out: List[Tuple[float, str]] = []
    for fn in os.listdir(SESSION_DIR):
        if not fn.endswith(".json"):
            continue
        if fn == "active.json":
            continue
        sid = fn[:-5]
        try:
            mtime = os.path.getmtime(os.path.join(SESSION_DIR, fn))
            out.append((mtime, sid))
        except OSError:
            continue
    out.sort(reverse=True)
    return out


def get_new_session_id() -> str:
    return str(uuid.uuid4().hex[:8])
