"""tests/test_settings_persist.py — Tests für gradio_tabs/settings_persist.py.

Plan 2026-10-05 (User-Request): "dass die einstellungen ... einer session
auch in der json gespeichert werden und wieder beim laden der session
geladen werden."

Pinnt die session-keyed Debouncer-Registry (Reuses SettingsDebouncer,
dessen Pins T5-T11 in test_chat_settings.py bleiben):
  T-S1: schedule_settings_save persistiert nach flush in die session.json
  T-S2: schnelle schedules pro Feld mergen (last-write-wins, dict-key-level)
  T-S3: ohne session_id → GARKEIN Write (kein File, kein Crash)
  T-S4: getrennte Sessions → Patches landen in der RICHTIGEN Datei
        (auch wenn der Timer nach einem Session-Wechsel fired)
  T-S5: _reset_persister_for_tests() leert die Registry
  T-S6: flush_settings_save(session_id) flush't gezielt eine Session

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_settings_persist.py
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sessions import load_session, SESSION_DIR
from gradio_tabs.settings_persist import (
    schedule_settings_save,
    flush_settings_save,
    _reset_persister_for_tests,
)


def _session_path(sid):
    return os.path.join(SESSION_DIR, f"{sid}.json")


def _wipe(sid_keys):
    for sid in sid_keys:
        p = _session_path(sid)
        if os.path.exists(p):
            os.unlink(p)
    _reset_persister_for_tests()


def test_s1_schedule_then_flush_persists():
    """T-S1: schedule → flush → Feld steht in der session.json."""
    sid = "ts1_persist"
    _wipe([sid])
    try:
        schedule_settings_save(sid, temperature=0.9)
        flush_settings_save(sid)
        settings = load_session(sid).get("settings", {})
        assert settings.get("temperature") == 0.9
    finally:
        _wipe([sid])


def test_s2_quick_schedules_merge_last_write_wins():
    """T-S2: mehrere schedules mergen pro Feld — letzter Wert gewinnt."""
    sid = "ts2_merge"
    _wipe([sid])
    try:
        schedule_settings_save(sid, temperature=0.9, top_p=0.5)
        schedule_settings_save(sid, temperature=0.4)
        flush_settings_save(sid)
        settings = load_session(sid).get("settings", {})
        assert settings.get("temperature") == 0.4   # letzter Wert
        assert settings.get("top_p") == 0.5         # Feld aus Patch 1 bleibt
    finally:
        _wipe([sid])


def test_s3_no_session_id_no_write():
    """T-S3: session_id=None/'' → kein Crash, kein Write."""
    schedule_settings_save(None, temperature=0.9)
    schedule_settings_save("", temperature=0.9)
    flush_settings_save(None)
    # nichts geräumt, nichts crasht — Assertion: Registry leer
    from gradio_tabs import settings_persist as sp
    assert sp._DEBOUNCERS == {}


def test_s4_separate_sessions_isolated_files():
    """T-S4: zwei Sessions, zwei Patches → landen in der richtigen Datei."""
    sid_a = "ts4_alpha"
    sid_b = "ts4_beta"
    _wipe([sid_a, sid_b])
    try:
        schedule_settings_save(sid_a, temperature=0.1)
        schedule_settings_save(sid_b, temperature=0.2, px_preset="BASELINE")
        flush_settings_save()  # flush ALL
        a = load_session(sid_a).get("settings", {})
        assert a == {"temperature": 0.1}
        b = load_session(sid_b).get("settings", {})
        assert b == {"temperature": 0.2, "px_preset": "BASELINE"}
    finally:
        _wipe([sid_a, sid_b])


def test_s5_reset_clears_registry_and_pending():
    """T-S5: _reset verwirft pending patches (nach Reset flushed nichts)."""
    sid = "ts5_reset"
    _wipe([sid])
    schedule_settings_save(sid, temperature=0.9)
    _reset_persister_for_tests()
    flush_settings_save()  # Registry leer → kein Write
    assert not os.path.exists(_session_path(sid))


def test_s6_flush_single_session_targets_only_that_one():
    """T-S6: flush_settings_save(sid) flush't NUR die eine Session."""
    sid_a = "ts6_alpha"
    sid_b = "ts6_beta"
    _wipe([sid_a, sid_b])
    try:
        schedule_settings_save(sid_a, temperature=0.1)
        schedule_settings_save(sid_b, temperature=0.2)
        flush_settings_save(sid_a)
        assert load_session(sid_a).get("settings", {}).get("temperature") == 0.1
        assert not os.path.exists(_session_path(sid_b))
        flush_settings_save(sid_b)
        assert load_session(sid_b).get("settings", {}).get("temperature") == 0.2
    finally:
        _wipe([sid_a, sid_b])


def test_s7_creates_minimal_entry_for_unknown_session():
    """T-S7: schedule für unbekannte Session → update_settings erzeugt
    minimalen Eintrag (sessions.py T4-Semantik)."""
    sid = "ts7_minimal"
    _wipe([sid])
    try:
        schedule_settings_save(sid, relay_sign=1)
        flush_settings_save(sid)
        data = load_session(sid)
        assert data.get("session_id") == sid
        assert data.get("settings", {}).get("relay_sign") == 1
    finally:
        _wipe([sid])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))