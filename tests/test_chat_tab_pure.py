"""Pure-logic Tests für gradio_tabs/chat_tab.py helpers.

Pinnt das Pre-TTS-Verhalten von:
- `_stringify_content(content)` — strings / lists-of-blocks / dicts → string
- `_clean_history(history)` — leer-/Same-Role-Merge-Filter

Diese Helpers sind Pre-TTS (Master `c03d224` + pre-tts-improvements) und
müssen auch nach TTS-Fixes/-Refactors/-Erweiterungen weiterhin
funktionieren. Tests sind pure-logic (kein Gradio-Import nötig).

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_chat_tab_pure.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradio_tabs.chat_tab import _stringify_content, _clean_history


# --- _stringify_content --------------------------------------------------

def test_str_1_passthrough_string():
    """Strings bleiben strings, unverändert."""
    assert _stringify_content("hello") == "hello"
    assert _stringify_content("") == ""
    assert _stringify_content("multi\nline") == "multi\nline"


def test_str_2_list_of_strings_joined_with_newline():
    """Eine Liste von plain Strings wird mit newline gejoined."""
    assert _stringify_content(["a", "b", "c"]) == "a\nb\nc"


def test_str_3_list_of_text_blocks_joined():
    """Eine Liste mit ``{"type": "text", "text": ...}`` Blöcken wird
    mit newline gejoined. Pre-TTS-Pfad."""
    blocks = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]
    assert _stringify_content(blocks) == "hello\nworld"


def test_str_4_list_with_mixed_types():
    """Mixed: dict, string, dict-with-text. Alle Text-Anteile extrahiert."""
    blocks = [
        "raw string",
        {"type": "text", "text": "block1"},
        {"type": "image", "image": "/path/to/img.png"},  # kein Text → ignoriert
        {"text": "block2"},  # kein 'type' aber 'text' → wird auch extrahiert (Pre-TTS-Pfad)
    ]
    result = _stringify_content(blocks)
    assert "raw string" in result
    assert "block1" in result
    assert "block2" in result


def test_str_5_dict_with_text():
    """Ein einzelnes Dict mit 'text'-Feld wird zu seinem Text-Value."""
    assert _stringify_content({"text": "hello"}) == "hello"


def test_str_6_dict_without_text_falls_back_to_str():
    """Dict ohne 'text'-Feld → str(dict) als Fallback (Pre-TTS-Pfad)."""
    obj = {"type": "image", "image": "/x.png"}
    result = _stringify_content(obj)
    assert "/x.png" in result
    assert isinstance(result, str)


def test_str_7_none_returns_str_none():
    """None → str(None) = 'None'."""
    assert _stringify_content(None) == "None"


def test_str_8_int_returns_str():
    """int → str(int). Edge-Case für nicht-erwartete Typen."""
    assert _stringify_content(42) == "42"


# --- _clean_history -------------------------------------------------------

def m(role, content):
    """Helper für Chat-History-Eintrag."""
    return {"role": role, "content": content}


def test_clean_1_empty_history_returns_empty():
    """Leere / None-History → leere Liste."""
    assert _clean_history([]) == []
    assert _clean_history(None) == []


def test_clean_2_drops_empty_string_messages():
    """User/Assistant mit content="" werden rausgefiltert."""
    hist = [m("user", ""), m("assistant", "  "), m("user", "hi")]
    out = _clean_history(hist)
    assert len(out) == 1
    assert out[0] == m("user", "hi")


def test_clean_3_drops_empty_list_messages():
    """User/Assistant mit content=[] werden rausgefiltert."""
    hist = [m("user", []), m("user", "hello")]
    out = _clean_history(hist)
    assert len(out) == 1
    assert out[0] == m("user", "hello")


def test_clean_4_merges_consecutive_same_role_strings():
    """Zwei aufeinanderfolgende user-Turns werden mit newline gemerged."""
    hist = [m("user", "hello"), m("user", "world")]
    out = _clean_history(hist)
    assert len(out) == 1
    assert out[0] == m("user", "hello\nworld")


def test_clean_5_merges_consecutive_assistant():
    """Auch assistant-merging funktioniert."""
    hist = [m("user", "hi"), m("assistant", "a"), m("assistant", "b")]
    out = _clean_history(hist)
    assert len(out) == 2
    assert out[1] == m("assistant", "a\nb")


def test_clean_6_keeps_alternating_pattern():
    """Sauber alternierendes Pattern bleibt unverändert."""
    hist = [
        m("user", "hi"),
        m("assistant", "hello"),
        m("user", "bye"),
        m("assistant", "bye!"),
    ]
    out = _clean_history(hist)
    assert len(out) == 4


def test_clean_7_drops_non_dict_entries():
    """Strings oder Listen als Top-Level werden ignoriert (nicht-dict)."""
    hist = ["not a dict", m("user", "hi"), None]
    out = _clean_history(hist)
    assert len(out) == 1
    assert out[0] == m("user", "hi")


def test_clean_8_pre_tts_bug_no_role_filter():
    """PRE-TTS-BUG (Regression-Marker): Einträge ohne 'role'-Feld werden
    Pre-TTS nicht rausgefiltert — ``msg.get('role', '')`` gibt ``""``
    zurück, dann wird ``role == ''`` durchgereicht (statt gefiltert).

    Test pinnt das aktuelle Verhalten:
    - Entry ohne 'role'-Feld: bleibt mit role='' im Output
    - Entry mit role=None: bleibt mit role=None im Output

    Wenn der Bug in einem Refactor gefixt wird, müssen diese Tests
    aktualisiert werden — das ist gewollt (Refactor-Detector).
    """
    hist = [{"content": "x"}, m("user", "hi"), {"role": None, "content": "y"}]
    out = _clean_history(hist)
    # 3 Einträge durch — keine Filterung
    assert len(out) == 3, f"expected 3 (Pre-TTS-Bug), got {len(out)}"
    # Erster Eintrag: role='' (Default)
    assert out[0] == {"content": "x", "role": ""}, out[0]
    # Zweiter: bleibt user
    assert out[1] == m("user", "hi")
    # Dritter: role=None bleibt None
    assert out[2] == {"role": None, "content": "y"}, out[2]


def test_clean_9_merges_string_then_list():
    """Consecutive same-role, erste String, zweite List-of-blocks →
    String wird zum typed-text-Block und in die Liste angehängt."""
    hist = [
        m("user", "hello"),
        m("user", [{"type": "text", "text": "world"}]),
    ]
    out = _clean_history(hist)
    assert len(out) == 1
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "hello"}
    assert content[1] == {"type": "text", "text": "world"}


def test_clean_10_merges_list_then_string():
    """Consecutive same-role, erste List, zweiter String → String wird
    als typed-text-Block der Liste vorangestellt."""
    hist = [
        m("user", [{"type": "text", "text": "hello"}]),
        m("user", "world"),
    ]
    out = _clean_history(hist)
    assert len(out) == 1
    content = out[0]["content"]
    assert isinstance(content, list)
    # erstes Element ist die Konvertierung des vorigen Strings
    assert content[0] == {"type": "text", "text": "hello"}
    assert content[1] == {"type": "text", "text": "world"}


def test_clean_11_merges_list_with_list():
    """Zwei Listen werden konkateniert."""
    hist = [
        m("user", [{"type": "text", "text": "a"}]),
        m("user", [{"type": "text", "text": "b"}]),
    ]
    out = _clean_history(hist)
    assert len(out) == 1
    content = out[0]["content"]
    assert content == [{"type": "text", "text": "a"},
                       {"type": "text", "text": "b"}]


def test_clean_12_does_not_mutate_input():
    """Input-History wird nicht verändert."""
    hist = [m("user", "hello"), m("user", "world")]
    snapshot = list(hist)
    _ = _clean_history(hist)
    assert hist == snapshot


# --- runner ---------------------------------------------------------------

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
