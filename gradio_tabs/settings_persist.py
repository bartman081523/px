"""gradio_tabs/settings_persist.py — Debounced per-session Settings-Save.

User-Request 2026-10-05: "dass die einstellungen, welches modell, welcher
px mode, parameter, etc einer session auch in der json gespeichert werden
und wieder beim laden der session geladen werden."

Der Persist-Teil: Widget-.input-Handler (User-Aktion) schedule'n Feld-spezifische
Patches; ein per-session Debouncer (400ms, last-write-wins) ruft
sessions.update_settings() atomar. Der Restore-Teil liegt in chat_tab.py
(restore_session_settings → widget_updates_from_settings).

Warum debounced: Slider draggen feuert .input pro Tick — ohne Debounce
eine File-Write-Flut. Der SettingsDebouncer aus chat_settings.py ist
single-session-bound (constructor nimmt session_id fest); für eine
Multi-Session-UI brauchen wir eine session-keyed Registry DESShalb hier.
Reuse statt Duplizierung: pro Session lieg EIN SettingsDebouncer-Objekt
(alle dessen Pins T5-T11 bleiben gültig).

Öffentliche API (module-level singleton):
    schedule_settings_save(session_id, **patch) -> None
    flush_settings_save(session_id=None) -> None
    _reset_persister_for_tests() -> None

Test-Pins siehe tests/test_settings_persist.py:
    T-S1: schedule persistiert nach delay in die session.json
    T-S2: mehrere schnelle schedules mergen (last-write-wins)
    T-S3: ohne session_id → kein Write
    T-S4: pro Session getrennte patches landen in der richtigen Datei
"""
from __future__ import annotations

import threading
from typing import Dict

from gradio_tabs.chat_settings import SettingsDebouncer
from sessions import update_settings


_LOCK = threading.Lock()
_DEBOUNCERS: Dict[str, SettingsDebouncer] = {}


def _get_debouncer(session_id: str) -> SettingsDebouncer:
    """Liefert (cached) den Debouncer einer Session.

    on_save-Closure capture't die session_id des Debouncers — Patches
    landen IMMER in der Datei ihrer Session, auch wenn der Timer nach
    einem Session-Wechsel fired (Patch gehört zur Schedule-Zeit).
    """
    with _LOCK:
        d = _DEBOUNCERS.get(session_id)
        if d is None:
            def _on_save(patch, sid=session_id):
                update_settings(sid, **patch)
            d = SettingsDebouncer(
                session_id=session_id, on_save=_on_save, delay_ms=400,
            )
            _DEBOUNCERS[session_id] = d
        return d


def schedule_settings_save(session_id, **patch) -> None:
    """Fire-and-forget: <patch> landet debounced in der session.json."""
    if not session_id:
        return
    _get_debouncer(session_id).schedule(**patch)


def flush_settings_save(session_id=None) -> None:
    """Persistiert pending patches sofort (eine oder alle Sessions)."""
    with _LOCK:
        targets = (
            [_DEBOUNCERS[session_id]]
            if session_id in _DEBOUNCERS
            else list(_DEBOUNCERS.values())
        )
    for d in targets:
        d.flush_now()


def _reset_persister_for_tests() -> None:
    """Test-Isolation: alle Debouncer/Pending-Patches verwerfen."""
    with _LOCK:
        for d in _DEBOUNCERS.values():
            d.cancel()
        _DEBOUNCERS.clear()