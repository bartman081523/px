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

from streaming_bridge import _build_image_data_url, _MIME_BY_EXT


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