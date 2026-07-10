"""Pure-logic Tests für streaming_bridge.py:_build_image_data_url.

Pinnt Pre-TTS-Verhalten der Multimodal-Image-Daten-URL-Bildung aus CLI-
Argumenten --image (lokaler Pfad) und --image-base64 (raw / data: URL).

Refactor-Detector: Wenn jemand an MIME-Mapping oder Path-Resolution
ändert, fallen diese Tests rot.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_streaming_bridge_pure.py
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming_bridge import (
    _build_image_data_url,
    _MIME_BY_EXT,
    _extract_text_from_content,
)


# --- _build_image_data_url — both args ----------------------------------

def test_both_args_raises():
    """Beide Args gesetzt → ValueError (CLI-Validation)."""
    try:
        _build_image_data_url(image_path="/x.png", image_base64="abc")
    except ValueError as e:
        assert "one of" in str(e).lower() or "only" in str(e).lower(), e
        return
    raise AssertionError("expected ValueError, got nothing")


def test_no_args_returns_none():
    """Keine Args → None (kein Bild)."""
    assert _build_image_data_url() is None
    assert _build_image_data_url(image_path=None, image_base64=None) is None


# --- image_path --------------------------------------------------------

def test_image_path_png_returns_png_mime():
    """PNG-Datei wird mit image/png data: URL zurückgegeben."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n")  # PNG-Magic
        path = f.name
    try:
        url = _build_image_data_url(image_path=path)
        assert url is not None
        assert url.startswith("data:image/png;base64,")
    finally:
        os.unlink(path)


def test_image_path_jpg_returns_jpeg_mime():
    """JPG-Datei (.jpg) wird mit image/jpeg data: URL zurückgegeben."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8\xff\xe0")  # JPEG-Magic
        path = f.name
    try:
        url = _build_image_data_url(image_path=path)
        assert url.startswith("data:image/jpeg;base64,")
    finally:
        os.unlink(path)


def test_image_path_jpeg_returns_jpeg_mime():
    """.jpeg-Datei wird mit image/jpeg data: URL zurückgegeben."""
    with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as f:
        f.write(b"\xff\xd8\xff\xe0")
        path = f.name
    try:
        url = _build_image_data_url(image_path=path)
        assert url.startswith("data:image/jpeg;base64,")
    finally:
        os.unlink(path)


def test_image_path_webp_returns_webp_mime():
    """.webp-Datei wird mit image/webp data: URL zurückgegeben."""
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
        f.write(b"RIFF")
        path = f.name
    try:
        url = _build_image_data_url(image_path=path)
        assert url.startswith("data:image/webp;base64,")
    finally:
        os.unlink(path)


def test_image_path_gif_returns_gif_mime():
    """.gif-Datei wird mit image/gif data: URL zurückgegeben."""
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        f.write(b"GIF89a")
        path = f.name
    try:
        url = _build_image_data_url(image_path=path)
        assert url.startswith("data:image/gif;base64,")
    finally:
        os.unlink(path)


def test_image_path_unknown_ext_defaults_to_jpeg():
    """Unbekannte Extension (z.B. .txt) → image/jpeg default."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"raw")
        path = f.name
    try:
        url = _build_image_data_url(image_path=path)
        assert url.startswith("data:image/jpeg;base64,")
    finally:
        os.unlink(path)


def test_image_path_uppercase_ext_lowercased():
    """EXT wird vor Lookup lowercased (`.PNG` wie `.png`)."""
    with tempfile.NamedTemporaryFile(suffix=".PNG", delete=False) as f:
        f.write(b"\x89PNG\r\n")
        path = f.name
    try:
        url = _build_image_data_url(image_path=path)
        assert url.startswith("data:image/png;base64,")
    finally:
        os.unlink(path)


def test_image_path_content_is_base64_encoded():
    """Inhalt wird base64-kodiert — bytes im Input erscheinen als ASCII
    Base64 im Output."""
    raw = b"hello world"
    import base64
    expected_b64 = base64.b64encode(raw).decode("ascii")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        url = _build_image_data_url(image_path=path)
        assert expected_b64 in url
    finally:
        os.unlink(path)


# --- image_base64 ------------------------------------------------------

def test_image_base64_data_url_passthrough():
    """data: URL wird durchgereicht (kein Re-Wrap)."""
    data_url = "data:image/png;base64,iVBORw0KGgo="
    out = _build_image_data_url(image_base64=data_url)
    assert out == data_url


def test_image_base64_raw_wrapped_as_jpeg_default():
    """Raw base64 (ohne data:-Prefix) wird als image/jpeg gewrappt."""
    raw_b64 = "iVBORw0KGgo="
    out = _build_image_data_url(image_base64=raw_b64)
    assert out == f"data:image/jpeg;base64,{raw_b64}"


def test_image_base64_data_url_different_mime_passthrough():
    """Auch nicht-jpeg data: URLs werden durchgereicht."""
    data_url = "data:image/webp;base64,UklGRiQ="
    out = _build_image_data_url(image_base64=data_url)
    assert out == data_url


def test_image_base64_empty_string_returns_none():
    """PRE-TTS-PFAD: leerer String ist ``if image_base64:``-falsy → wird
    als 'kein Base64' behandelt → Return None (kein Bild).

    Pin: Wenn das Verhalten geändert wird (z.B. explizite Validierung
    statt Truthiness-Check), bricht dieser Test als Refactor-Detector.
    """
    out = _build_image_data_url(image_base64="")
    assert out is None, f"expected None, got {out!r}"


# --- _MIME_BY_EXT integrity --------------------------------------------

def test_mime_map_covers_common_image_formats():
    """_MIME_BY_EXT hat die gängigen Formate (.jpg/.jpeg/.png/.webp/.gif)."""
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        assert ext in _MIME_BY_EXT, f"missing extension: {ext}"


def test_mime_map_jpg_and_jpeg_share_mime():
    """.jpg und .jpeg haben beide image/jpeg."""
    assert _MIME_BY_EXT[".jpg"] == "image/jpeg"
    assert _MIME_BY_EXT[".jpeg"] == "image/jpeg"


# --- _extract_text_from_content ----------------------------------------
# Plan 2026-07-09: Beim Rebuild der History für die /v1/chat/completions-API
# wird Multimodal-Listen-Content (text+file-Blöcke) zu reinem Text geflatted.
# Vorher crashte das still, wenn ein Gradio-File-Block im Content war — der
# wurde einfach gedroppt, der User sah nur "look at this" statt Hinweis aufs
# Bild. Mit dem Helper bekommen File-Blöcke einen [image: <path>]-Marker,
# damit der Kontext nicht still kürzer wird.

def test_extract_text_from_plain_string_passthrough():
    """Plain-String → unverändert (User hat nur Text eingegeben)."""
    assert _extract_text_from_content("hello") == "hello"


def test_extract_text_from_none_returns_empty():
    """None → "" (defensiv für uninitialisierte Messages)."""
    assert _extract_text_from_content(None) == ""


def test_extract_text_from_text_block_only():
    """Liste mit nur text-Block → dessen Text."""
    out = _extract_text_from_content([{"type": "text", "text": "hi"}])
    assert out == "hi"


def test_extract_text_from_file_block_marker():
    """Gradio-File-Block (image/png) → [image: <basename>]-Marker.

    Vor dem Fix crashte das still — der Block wurde komplett gedroppt
    und die User-Message bestand nur aus dem Text-Anteil."""
    out = _extract_text_from_content([
        {"type": "file", "file": {"path": "/tmp/cat.png", "mime_type": "image/png"}},
        {"type": "text", "text": "look at this"},
    ])
    assert "[image: cat.png]" in out
    assert "look at this" in out


def test_extract_text_from_legacy_image_block_marker():
    """Legacy-pre-2026-07-09 ``type:image``-Block → auch [image: ...]."""
    out = _extract_text_from_content([
        {"type": "image", "image": "/tmp/dog.jpg"},
    ])
    assert "[image: dog.jpg]" in out


def test_extract_text_from_openai_image_url_block_marker():
    """OpenAI-``image_url``-Block → [image: <basename of url>]-Marker.

    Bei data:-URLs wird der basename der URL verwendet (geht nicht besser,
    ohne den Base64-Inhalt zu decoden — wir wollen ja nur den Context-
    Hint, nicht das Bild selbst)."""
    out = _extract_text_from_content([
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,XYZ"}},
        {"type": "text", "text": "transcribe"},
    ])
    assert "[image:" in out
    assert "transcribe" in out


def test_extract_text_from_text_file_block_marker():
    """Gradio-File-Block mit Text-MIME (.txt/.py/.md) → [text-file: name]."""
    out = _extract_text_from_content([
        {"type": "file", "file": {"path": "/tmp/notes.md", "mime_type": "text/markdown"}},
    ])
    assert "[text-file: notes.md]" in out


def test_extract_text_from_audio_file_block_marker():
    """Audio-File-Block → [audio: <basename>]-Marker (klar vom Bild unterscheidbar)."""
    out = _extract_text_from_content([
        {"type": "file", "file": {"path": "/tmp/clip.wav", "mime_type": "audio/wav"}},
    ])
    assert "[audio: clip.wav]" in out


def test_extract_text_from_input_audio_block_marker():
    """OpenAI-``input_audio``-Block → [audio: ...]-Marker."""
    out = _extract_text_from_content([
        {"type": "input_audio", "input_audio": {"url": "data:audio/wav;base64,XYZ"}},
    ])
    assert "[audio:" in out


def test_extract_text_from_file_block_no_path_uses_default():
    """File-Block ohne path (defensive) → '?' statt Crash."""
    out = _extract_text_from_content([
        {"type": "file", "file": {"mime_type": "image/png"}},
    ])
    assert "[image: ?]" in out


def test_extract_text_preserves_text_block_order():
    """Bei mehreren text-Blöcken wird die Reihenfolge preserved (join)."""
    out = _extract_text_from_content([
        {"type": "text", "text": "first"},
        {"type": "file", "file": {"path": "/x.png", "mime_type": "image/png"}},
        {"type": "text", "text": "second"},
    ])
    # text parts in order, file-marker in between
    assert out.index("first") < out.index("[image:") < out.index("second")


def test_extract_text_from_empty_list_returns_empty():
    """Leere Liste → ""."""
    assert _extract_text_from_content([]) == ""


def test_extract_text_from_unknown_dict_type_drops_silently():
    """Unbekannter Block-Typ → wird gedroppt (nicht crashen).

    Defensiv: ein neuer Block-Typ den wir noch nicht kennen, blockiert
    nicht die ganze History."""
    out = _extract_text_from_content([
        {"type": "future_thing_we_dont_know", "data": "???"},
        {"type": "text", "text": "real text"},
    ])
    assert out == "real text"


# --- runner ------------------------------------------------------------

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