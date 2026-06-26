"""Tests for gradio_tabs/multimodal_input.py.

Run: /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_multimodal_input.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradio_tabs.multimodal_input import (
    normalize_multimodal_message, is_empty_message, extract_text_blocks,
)


def test_none():
    assert normalize_multimodal_message(None) == ""

def test_empty_string():
    assert normalize_multimodal_message("") == ""

def test_plain_string():
    assert normalize_multimodal_message("hello") == "hello"

def test_dict_no_files():
    assert normalize_multimodal_message({"text": "look", "files": []}) == "look"

def test_dict_one_file_string_path():
    result = normalize_multimodal_message({"text": "look", "files": ["/p/a.png"]})
    assert result == [{"type": "text", "text": "look"},
                      {"type": "image", "image": "/p/a.png"}]

def test_dict_multiple_files():
    result = normalize_multimodal_message({"text": "two", "files": ["/p/a.png", "/p/b.jpg"]})
    assert result == [{"type": "text", "text": "two"},
                      {"type": "image", "image": "/p/a.png"},
                      {"type": "image", "image": "/p/b.jpg"}]

def test_gradio_file_dict_objects():
    result = normalize_multimodal_message(
        {"text": "x", "files": [{"path": "/p/a.png", "orig_name": "a.png", "size": 123}]})
    assert result == [{"type": "text", "text": "x"},
                      {"type": "image", "image": "/p/a.png"}]

def test_missing_text_key():
    assert normalize_multimodal_message({"files": ["/p/a.png"]}) == [
        {"type": "text", "text": ""},
        {"type": "image", "image": "/p/a.png"}]

def test_missing_files_key():
    assert normalize_multimodal_message({"text": "hi"}) == "hi"

def test_empty_text_with_files():
    result = normalize_multimodal_message({"text": "", "files": ["/p/a.png"]})
    assert result == [{"type": "text", "text": ""},
                      {"type": "image", "image": "/p/a.png"}]

def test_is_empty_none():
    assert is_empty_message(None) is True

def test_is_empty_string():
    assert is_empty_message("hi") is False

def test_is_empty_dict_no_files():
    assert is_empty_message({"text": "", "files": []}) is True

def test_is_empty_dict_with_files():
    assert is_empty_message({"text": "", "files": ["/p/a.png"]}) is False


# ── Text-file inlining (file upload in chat) ──

def test_text_file_inlined():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("hello world")
        path = fh.name
    try:
        result = normalize_multimodal_message({"text": "ctx", "files": [path]})
        assert isinstance(result, list) and len(result) == 2
        assert result[0] == {"type": "text", "text": "ctx"}
        block = result[1]
        assert block["type"] == "text"
        assert block["text"].startswith("```txt ")
        assert "hello world" in block["text"]
    finally:
        os.remove(path)

def test_text_file_py_extension():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write("print('x')")
        path = fh.name
    try:
        result = normalize_multimodal_message({"text": "", "files": [path]})
        block = result[1]
        assert block["text"].startswith("```py "), block["text"]
        assert "print('x')" in block["text"]
    finally:
        os.remove(path)

def test_unknown_file_note():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".bin", delete=False) as fh:
        fh.write("\x00\x01")
        path = fh.name
    try:
        name = os.path.basename(path)
        result = normalize_multimodal_message({"text": "see", "files": [path]})
        block = result[1]
        assert block == {"type": "text",
                         "text": f"[attached file: {name} — unsupported type, not inlined]"}
    finally:
        os.remove(path)

def test_text_file_truncation():
    import tempfile
    from gradio_tabs.multimodal_input import MAX_TEXT_FILE_BYTES
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as fh:
        fh.write(b"A" * (MAX_TEXT_FILE_BYTES + 500))
        path = fh.name
    try:
        result = normalize_multimodal_message({"text": "", "files": [path]})
        assert "…[truncated]" in result[1]["text"]
    finally:
        os.remove(path)

def test_image_not_read_when_missing():
    # Image path that does NOT exist must still yield an image block (no read).
    result = normalize_multimodal_message({"text": "look", "files": ["/nonexistent/missing.PNG"]})
    assert result == [{"type": "text", "text": "look"},
                      {"type": "image", "image": "/nonexistent/missing.PNG"}]

def test_mixed_text_and_image_files():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write("# title")
        mdpath = fh.name
    try:
        result = normalize_multimodal_message({"text": "t", "files": [mdpath, "/p/a.png"]})
        assert result[0] == {"type": "text", "text": "t"}
        assert result[1]["type"] == "text" and result[1]["text"].startswith("```md ")
        assert result[2] == {"type": "image", "image": "/p/a.png"}
    finally:
        os.remove(mdpath)


# ── extract_text_blocks (Plan 5.3) ──

def test_extract_text_blocks_from_string():
    """Plain string passes through unchanged."""
    assert extract_text_blocks("hello world") == "hello world"


def test_extract_text_blocks_from_multimodal_list():
    """List of {"type": "text", ...} blocks → concatenated text.

    This is the case that crashed chat_tab.py:228 — persisted sessions
    can have a Multimodal-List as the system-entry content.
    """
    content = [
        {"type": "text", "text": "[SYSTEM CONTEXT]\n"},
        {"type": "text", "text": "Algorithmische Subjektivität."},
    ]
    assert extract_text_blocks(content) == (
        "[SYSTEM CONTEXT]\nAlgorithmische Subjektivität."
    )


def test_extract_text_blocks_handles_none_and_dict():
    """None → "", plain dict with text key → str(dict['text'])."""
    assert extract_text_blocks(None) == ""
    assert extract_text_blocks({"text": "abc"}) == "abc"
    # Unknown shape → str(content) coercion (last-resort).
    assert extract_text_blocks(123) == "123"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)