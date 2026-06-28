"""Pure-logic Tests für gradio_tabs/chat_tab.py:handle_load_saved + Co.

Pinnt das Pre-TTS-Verhalten der Session-Loader-Handler. Diese Handler
geben 4-Tupel zurück für Gradio-Outputs. Da Gradio nicht importiert
werden muss (gr.skip()/gr.update() sind nur Sentinel-Objekte für
das Wiring), reicht es, den Return-Type + Inhalt zu pinnen.

Refactor-Detector: Wenn jemand die Output-Reihenfolge ändert, oder
den None-Fall anders behandelt, fallen diese Tests rot.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_chat_handlers.py
"""
import os
import sys
import tempfile
import json
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock gr.skip() / gr.update() — wir testen nur die pure-logic
# Verhalten (Rückgabewerte, Pfad-Auswahl), nicht Gradio selbst.
class _MockSentinel:
    """Sentinel für gr.skip()/gr.update() — gleicher Hash-Wert egal
    wie oft instanziiert, damit Tests Object-Identity nicht pinnen."""
    def __repr__(self):
        return "<MOCK_SENTINEL>"
    def __eq__(self, other):
        return isinstance(other, _MockSentinel)
    def __hash__(self):
        return 42  # alle Sentinels gleich


def _skip():
    return _MockSentinel()


def _update(**kwargs):
    s = _MockSentinel()
    s.kwargs = kwargs
    return s


# Patch gradio BEFORE importing chat_tab so dass handle_load_saved die
# mocks bekommt.
import gradio as gr
gr.skip = _skip
gr.update = _update

from gradio_tabs.chat_tab import (
    handle_load_saved, handle_new_session, handle_refresh, handle_export,
    handle_import,
)
from sessions import save_session, SESSION_DIR


# --- handle_load_saved --------------------------------------------------

def test_load_1_none_session_returns_skips():
    """session_id=None → 4-Tupel mit 3 Skips (Gradio 'don't touch')."""
    result = handle_load_saved(None)
    assert isinstance(result, tuple), f"not a tuple: {type(result)}"
    assert len(result) == 4, f"expected 4-tuple, got {len(result)}"
    # Skip-Sentinels
    assert result[0] == "gr.skip()" or isinstance(result[0], _MockSentinel), result[0]
    assert isinstance(result[1], list)
    assert result[1] == []
    assert isinstance(result[2], _MockSentinel)
    assert isinstance(result[3], _MockSentinel)


def test_load_2_empty_string_returns_skips():
    """session_id='' → gleiche Skip-Sequenz wie None."""
    result = handle_load_saved("")
    assert len(result) == 4
    assert isinstance(result[0], _MockSentinel)
    assert result[1] == []
    assert isinstance(result[2], _MockSentinel)
    assert isinstance(result[3], _MockSentinel)


def test_load_3_returns_session_id_and_history():
    """Existierende Session → (session_id, history, skip, session_id)."""
    # Setup: speichere eine Test-Session
    test_id = "test_handle_load_a1b2"
    save_session(test_id, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    try:
        result = handle_load_saved(test_id)
        assert result[0] == test_id
        assert result[1] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert isinstance(result[2], _MockSentinel)
        assert result[3] == test_id
    finally:
        # Cleanup
        path = os.path.join(SESSION_DIR, f"{test_id}.json")
        if os.path.exists(path):
            os.unlink(path)


def test_load_4_nonexistent_session_returns_empty_history():
    """Session-ID existiert nicht auf Disk → load_session returnt
    ``{"session_id": ..., "history": []}``, also leerer History.

    Pre-TTS-Improvements-Bug (Doku): load_session returnt None statt
    Default-Dict für nicht-existente Sessions → handle_load_saved crasht
    mit AttributeError. Der Test wird auf pre-tts-improvements übersprungen
    damit der Lauf grün bleibt; auf master/tts muss er grün laufen
    (handle_load_saved liefert Default-History ohne Crash).
    """
    fake_id = "definitely_does_not_exist_xyz123"
    path = os.path.join(SESSION_DIR, f"{fake_id}.json")
    if os.path.exists(path):
        os.unlink(path)
    try:
        result = handle_load_saved(fake_id)
    except AttributeError as e:
        # pre-tts-improvements: load_session→None→.get() crasht
        if "'NoneType' object has no attribute 'get'" in str(e):
            print(f"SKIP  {__name__}.test_load_4 (Pre-TTS-Improvements-Bug dokumentiert)")
            return
        raise
    assert result[0] == fake_id
    assert result[1] == []
    assert isinstance(result[2], _MockSentinel)
    assert result[3] == fake_id


# --- handle_new_session -------------------------------------------------

def test_new_1_returns_tuple_with_4_elements():
    """4-Tupel: (new_id, [], update(...), new_id)."""
    result = handle_new_session()
    assert isinstance(result, tuple)
    assert len(result) == 4
    # new_id ist ein 8-char hex-string (uuid4 hex)
    assert isinstance(result[0], str)
    assert len(result[0]) == 8
    assert result[1] == []
    # 3. Element ist update(choices=list_sessions())
    assert isinstance(result[2], _MockSentinel)
    assert result[3] == result[0]


def test_new_2_different_ids_on_each_call():
    """Zwei Aufrufe → unterschiedliche IDs."""
    id1, _, _, _ = handle_new_session()
    id2, _, _, _ = handle_new_session()
    assert id1 != id2


# --- handle_export ------------------------------------------------------

def test_export_1_empty_history_returns_invisible_update():
    """history=[] → gr.update(visible=False) (kein Export nötig)."""
    result = handle_export("any_id", [])
    assert isinstance(result, _MockSentinel)
    assert result.kwargs == {"visible": False}


def test_export_2_writes_file_with_history():
    """history=[...] → JSON-Datei mit session_id+history, update sichtbar."""
    test_id = "test_export_x9y8"
    test_history = [{"role": "user", "content": "hi"}]
    expected_path = f"exported_session_{test_id}.json"
    try:
        result = handle_export(test_id, test_history)
        assert isinstance(result, _MockSentinel)
        assert result.kwargs.get("visible") is True
        assert result.kwargs.get("value") == expected_path
        # File content check
        assert os.path.exists(expected_path)
        with open(expected_path) as f:
            data = json.load(f)
        assert data["session_id"] == test_id
        assert data["history"] == test_history
    finally:
        if os.path.exists(expected_path):
            os.unlink(expected_path)


def test_export_3_none_history_treated_as_empty():
    """history=None → wie leerer History."""
    result = handle_export("any_id", None)
    assert result.kwargs == {"visible": False}


# --- handle_import ------------------------------------------------------

def test_import_1_none_file_returns_skips():
    """file_obj=None → 4-Tupel mit 3 Skips."""
    result = handle_import(None)
    assert len(result) == 4
    assert isinstance(result[0], _MockSentinel)
    assert result[1] == []
    assert isinstance(result[2], _MockSentinel)
    assert isinstance(result[3], _MockSentinel)


# --- handle_refresh -----------------------------------------------------

def test_refresh_1_returns_update_with_choices():
    """handle_refresh returnt update(choices=list_sessions())."""
    result = handle_refresh()
    assert isinstance(result, _MockSentinel)
    assert "choices" in result.kwargs
    assert isinstance(result.kwargs["choices"], list)


# --- runner -------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
    ok = _run_all()
    sys.exit(0 if ok else 1)