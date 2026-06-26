"""Tests for gradio_tabs/chat_actions.py — pure-logic chat undo history manipulation.

Written FIRST (TDD). Run with:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_chat_actions.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradio_tabs.chat_actions import undo_last_turn, undo_last_entry, can_undo
from gradio_tabs.chat_tab import _normalize_history_for_chatbot


def m(role, content):
    """Helper: build a chat history message dict."""
    return {"role": role, "content": content}


# --- undo_last_turn ----------------------------------------------------------

def test_turn_1_removes_last_user_assistant_turn():
    hist = [
        m("user", "hi"),
        m("assistant", "hello"),
        m("user", "bye"),
        m("assistant", "bye!"),
    ]
    out = undo_last_turn(hist)
    assert len(out) == 2, f"expected 2, got {len(out)}"
    assert out[0] == m("user", "hi"), out[0]
    assert out[1] == m("assistant", "hello"), out[1]


def test_turn_2_trailing_user_removed():
    hist = [
        m("user", "hi"),
        m("assistant", "hello"),
        m("user", "bye"),
    ]
    out = undo_last_turn(hist)
    assert len(out) == 2, f"expected 2, got {len(out)}"
    assert out[0] == m("user", "hi"), out[0]
    assert out[1] == m("assistant", "hello"), out[1]


def test_turn_3_single_full_turn_removed():
    hist = [
        m("user", "hi"),
        m("assistant", "hello"),
    ]
    out = undo_last_turn(hist)
    assert out == [], out


def test_turn_4_single_user_removed():
    hist = [m("user", "hi")]
    out = undo_last_turn(hist)
    assert out == [], out


def test_turn_5_empty():
    out = undo_last_turn([])
    assert out == [], out


def test_turn_6_multimodal_turn_removed():
    hist = [
        m("user", [{"type": "text", "text": "look"},
                   {"type": "image", "image": "/p/a.png"}]),
        m("assistant", "nice"),
    ]
    out = undo_last_turn(hist)
    assert out == [], out
    # input NOT mutated
    assert len(hist) == 2, f"input mutated: len={len(hist)}"
    assert hist[0]["content"][0]["text"] == "look"
    assert hist[0]["content"][1]["image"] == "/p/a.png"
    assert hist[1]["content"] == "nice"


def test_turn_7_no_mutation_of_input():
    hist = [
        m("user", "hi"),
        m("assistant", "hello"),
        m("user", "bye"),
        m("assistant", "bye!"),
    ]
    out = undo_last_turn(hist)
    assert len(hist) == 4, f"input mutated: len={len(hist)}"
    assert len(out) == 2


# --- _normalize_history_for_chatbot (Plan 5.3 Bug-Fix) -----------------------

def test_norm_1_untyped_text_block_collapses_to_string():
    """Multimodal-List whose blocks lack a ``type`` key is collapsed to
    a single string via extract_text_blocks. This is the case that
    crashed Gradio's ``_postprocess_content`` with ValueError."""
    hist = [
        m("system", [{"text": "[SYSTEM CONTEXT]\nAlgorithmische "
                          "Subjektivität ist..."}]),
    ]
    out = _normalize_history_for_chatbot(hist)
    assert len(out) == 1
    assert out[0]["role"] == "system"
    assert isinstance(out[0]["content"], str)
    assert "[SYSTEM CONTEXT]" in out[0]["content"]


def test_norm_2_typed_text_blocks_preserved_as_list():
    """Blocks with valid ``type=='text'`` (Gradio's expected format) are
    kept as a list — no flattening."""
    hist = [
        m("user", [
            {"type": "text", "text": "look at this"},
            {"type": "image", "image": "/p/a.png"},
        ]),
    ]
    out = _normalize_history_for_chatbot(hist)
    assert len(out) == 1
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0]["type"] == "text"
    assert out[0]["content"][1]["type"] == "image"


def test_norm_3_908e5ae1_real_world_crash_session():
    """Reproduce the exact crash from sessions/908e5ae1.json — system
    entry with content=list, blocks lack 'type'."""
    hist = [
        m("system", [
            {"text": "[SYSTEM CONTEXT]\n"}
        ]),
    ]
    out = _normalize_history_for_chatbot(hist)
    assert out[0]["content"] == "[SYSTEM CONTEXT]\n"


def test_norm_4_plain_string_passes_through():
    """Content already a string is kept (with null-byte stripping)."""
    hist = [
        m("user", "hello\x00world"),
        m("assistant", "hi"),
    ]
    out = _normalize_history_for_chatbot(hist)
    assert out[0]["content"] == "helloworld"
    assert out[1]["content"] == "hi"


def test_norm_5_drops_none_content_and_invalid_roles():
    hist = [
        m("user", None),
        m("user", ""),
        {"role": "tool", "content": "x"},  # unknown role
        m("assistant", "ok"),
    ]
    out = _normalize_history_for_chatbot(hist)
    assert len(out) == 1
    assert out[0] == {"role": "assistant", "content": "ok"}


def test_norm_6_empty_history_returns_empty_list():
    assert _normalize_history_for_chatbot([]) == []
    assert _normalize_history_for_chatbot(None) == []


def test_norm_7_does_not_mutate_input():
    hist = [
        m("system", [{"text": "raw"}]),
        m("user", "hi"),
    ]
    _ = _normalize_history_for_chatbot(hist)
    # Input still has the original (untyped) list.
    assert isinstance(hist[0]["content"], list)
    assert hist[0]["content"][0] == {"text": "raw"}


# --- undo_last_entry ---------------------------------------------------------

def test_entry_8_removes_last_assistant():
    hist = [
        m("user", "hi"),
        m("assistant", "hello"),
        m("user", "bye"),
        m("assistant", "bye!"),
    ]
    out = undo_last_entry(hist)
    assert len(out) == 3, f"expected 3, got {len(out)}"
    assert out[0] == m("user", "hi"), out[0]
    assert out[1] == m("assistant", "hello"), out[1]
    assert out[2] == m("user", "bye"), out[2]
    # input not mutated
    assert len(hist) == 4, f"input mutated: len={len(hist)}"


def test_entry_9_removes_trailing_user():
    hist = [
        m("user", "hi"),
        m("assistant", "hello"),
        m("user", "bye"),
    ]
    out = undo_last_entry(hist)
    assert len(out) == 2, f"expected 2, got {len(out)}"
    assert out[0] == m("user", "hi"), out[0]
    assert out[1] == m("assistant", "hello"), out[1]


def test_entry_10_empty():
    out = undo_last_entry([])
    assert out == [], out


# --- can_undo ----------------------------------------------------------------

def test_can_undo_11():
    u = m("user", "hi")
    a = m("assistant", "hello")
    assert can_undo([u]) is True
    assert can_undo([]) is False
    assert can_undo([u, a]) is True


# --- runner ------------------------------------------------------------------

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
    import sys
    ok = _run_all()
    sys.exit(0 if ok else 1)