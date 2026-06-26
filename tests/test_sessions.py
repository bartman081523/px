"""Tests for sessions.py — settings schema, atomic writes, update_settings.

Run: /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_sessions.py
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# Work on a temporary SESSION_DIR to avoid clobbering real session files.
_TEST_SESSION_DIR = tempfile.mkdtemp(prefix="test_sessions_")


# Monkey-patch BEFORE importing sessions so the module picks up our dir.
# We achieve this by writing a tiny shim that replaces SESSION_DIR on the
# already-imported module (sessions is imported above by other tests if
# any). Here we do a fresh import + reload to guarantee isolation.
import importlib  # noqa: E402
import sessions as _sessions_mod  # noqa: E402
# First import: SESSION_DIR defaults to "sessions" (real dir).
# Override before reload so the reload picks up the new value.
_sessions_mod.SESSION_DIR = _TEST_SESSION_DIR
importlib.reload(_sessions_mod)
sessions = _sessions_mod
# Belt-and-suspenders: also reassign after reload (in case reload
# re-read a cached constant elsewhere).
sessions.SESSION_DIR = _TEST_SESSION_DIR


def _cleanup_test_dir():
    """Reset sessions/ to a clean state for each test."""
    if os.path.isdir(_TEST_SESSION_DIR):
        for fn in os.listdir(_TEST_SESSION_DIR):
            os.unlink(os.path.join(_TEST_SESSION_DIR, fn))


# ── Tests ──

def test_settings_defaults_have_16_fields():
    """The settings schema has exactly 16 fields (Plan 5.3)."""
    assert len(sessions.SETTINGS_DEFAULTS) == 16
    assert len(sessions.SETTINGS_FIELDS) == 16
    assert set(sessions.SETTINGS_FIELDS) == set(sessions.SETTINGS_DEFAULTS.keys())
    # Spot-check critical fields.
    for k in ("model_id", "px_preset", "auto_tune", "temperature",
              "top_p", "max_tokens", "rep_p", "px_gamma",
              "relay_sign", "relay_alpha", "relay_layer",
              "system_profile", "system_prompt_text",
              "tts_engine", "tts_sample_rate", "tts_auto"):
        assert k in sessions.SETTINGS_DEFAULTS, f"missing {k}"


def test_save_session_creates_file_atomically():
    """save_session writes a complete JSON file with settings + history."""
    _cleanup_test_dir()
    sid = "test_atom"
    history = [{"role": "user", "content": "hi"}]
    path = sessions.save_session(sid, history,
                                  settings={"temperature": 0.9,
                                            "model_id": "gemma3-1b-it"})
    assert os.path.exists(path)
    data = json.load(open(path, "r", encoding="utf-8"))
    assert data["session_id"] == sid
    assert data["history"] == history
    assert data["settings"]["temperature"] == 0.9
    assert data["settings"]["model_id"] == "gemma3-1b-it"
    # Missing keys filled with defaults.
    assert data["settings"]["auto_tune"] is False
    assert data["settings"]["top_p"] == 0.95


def test_load_session_returns_none_for_missing():
    """load_session returns None for a session_id that has no file."""
    assert sessions.load_session("nonexistent_session_xyz") is None


def test_update_settings_merges_patches():
    """update_settings atomically merges into existing settings."""
    _cleanup_test_dir()
    sid = "test_merge"
    sessions.save_session(sid, [], settings={"temperature": 0.7})
    sessions.update_settings(sid, temperature=0.95, top_p=0.8)
    data = sessions.load_session(sid)
    assert data["settings"]["temperature"] == 0.95
    assert data["settings"]["top_p"] == 0.8
    # Other defaults preserved.
    assert data["settings"]["max_tokens"] == 1024


def test_update_settings_only_accepts_known_fields():
    """Unknown keys in the patch are silently ignored (forward-compat)."""
    _cleanup_test_dir()
    sid = "test_unknown"
    sessions.update_settings(sid, temperature=0.5, unknown_key="x",
                              relay_sign=999)
    data = sessions.load_session(sid)
    assert data["settings"]["temperature"] == 0.5
    assert "unknown_key" not in data["settings"]


def test_atomic_write_no_temp_leftovers_on_error():
    """Failed atomic write cleans up the .tmp file (no leftover)."""
    _cleanup_test_dir()
    sid = "test_atomic_fail"
    target = os.path.join(_TEST_SESSION_DIR, f"{sid}.json")

    # Inject an error in os.fdopen via monkey-patch (atomic_write_json
    # uses os.fdopen, not builtins.open).
    orig_fdopen = os.fdopen

    def boom_fdopen(*args, **kwargs):
        raise OSError("simulated disk full")
    os.fdopen = boom_fdopen
    try:
        raised = False
        try:
            sessions._atomic_write_json(target, {"x": 1})
        except OSError:
            raised = True
        assert raised, "expected OSError to propagate"
    finally:
        os.fdopen = orig_fdopen
    # No leftover .tmp.
    leftovers = [f for f in os.listdir(_TEST_SESSION_DIR)
                  if f.startswith(".save_")]
    assert not leftovers, f"leftover tmp files: {leftovers}"


def test_concurrent_update_settings_thread_safe():
    """10 concurrent threads updating settings → no torn writes."""
    _cleanup_test_dir()
    sid = "test_concurrent"
    sessions.save_session(sid, [])
    errors: List[str] = []

    def worker(idx):
        try:
            for _ in range(10):
                sessions.update_settings(
                    sid, temperature=0.1 * (idx % 10))
        except Exception as e:
            errors.append(f"{idx}: {e}")
    threads = [threading.Thread(target=worker, args=(i,))
                for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    data = sessions.load_session(sid)
    # Temperature is one of the worker values (0.0..0.9).
    assert data["settings"]["temperature"] in [i * 0.1 for i in range(10)]


def test_list_session_mtimes_sorted_descending():
    """list_session_mtimes returns [(mtime, sid), ...] sorted newest-first."""
    _cleanup_test_dir()
    sessions.save_session("a", [])
    time.sleep(0.01)
    sessions.save_session("b", [])
    time.sleep(0.01)
    sessions.save_session("c", [])
    out = sessions.list_session_mtimes()
    assert [sid for _, sid in out] == ["c", "b", "a"]


# ── Runner ──

def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    try:
        ok = _run_all()
        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(_TEST_SESSION_DIR, ignore_errors=True)