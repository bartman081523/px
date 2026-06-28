"""Pure-logic Tests für streaming_bridge.py CLI-Flags.

Pinnt Defaults + Choices der argparse-Definition. Wenn jemand die
Defaults ändert oder ein Flag entfernt, fällt der Test.

Refactor-Detector: Webapp↔Bridge Param-Parity (top_p/gamma/rp/preset)
ist über die auto_tune_defaults.py-Konstanten geregelt. Hier geht es
nur um die CLI-Schnittstelle selbst — was der Bridge akzeptiert.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_streaming_bridge_cli.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming_bridge import _build_argparser


def _parse(argv):
    """Helper: parst argv durch den Bridge-Argumentparser."""
    return _build_argparser().parse_args(argv)


def _parse_no_args():
    """Helper: parst leere argv (alle Defaults)."""
    return _build_argparser().parse_args([])


# --- Defaults ----------------------------------------------------------

def test_default_session_is_aab82b16():
    """Ohne --session-Arg: Default 'aab82b16' (legacy Test-Session)."""
    args = _parse_no_args()
    assert args.session == "aab82b16"


def test_default_preset_is_active_manifold():
    """Ohne --preset-Arg: Default ACTIVE_MANIFOLD (PX-Engine aktiv)."""
    args = _parse_no_args()
    assert args.preset == "ACTIVE_MANIFOLD"


def test_default_model_is_gemma3_1b_it():
    """Ohne --model-Arg: Default gemma3-1b-it (PX-Engine-Hauptmodell)."""
    args = _parse_no_args()
    assert args.model == "gemma3-1b-it"


def test_default_message_is_none():
    """Ohne --message-Arg: None (= stdin-Prompt-Modus)."""
    args = _parse_no_args()
    assert args.message is None


def test_default_relay_sign_is_none():
    """Ohne --relay-sign-Arg: None (Auto-Detect via preset)."""
    args = _parse_no_args()
    assert args.relay_sign is None


def test_default_relay_alpha_is_none():
    """Ohne --relay-alpha-Arg: None (Default 0.30 wird im Code gesetzt)."""
    args = _parse_no_args()
    assert args.relay_alpha is None


def test_default_relay_layer_is_none():
    """Ohne --relay-layer-Arg: None (Default 21 wird im Code gesetzt)."""
    args = _parse_no_args()
    assert args.relay_layer is None


def test_default_image_is_none():
    """Ohne --image-Arg: None (Text-Only-Modus)."""
    args = _parse_no_args()
    assert args.image is None


def test_default_image_base64_is_none():
    """Ohne --image-base64-Arg: None (Text-Only-Modus)."""
    args = _parse_no_args()
    assert args.image_base64 is None


# --- Choices / Validation ----------------------------------------------

def test_relay_sign_accepts_plus_one():
    """--relay-sign 1 → args.relay_sign == 1."""
    args = _parse(["--relay-sign", "1"])
    assert args.relay_sign == 1


def test_relay_sign_accepts_zero():
    """--relay-sign 0 → args.relay_sign == 0 (relay off)."""
    args = _parse(["--relay-sign", "0"])
    assert args.relay_sign == 0


def test_relay_sign_accepts_minus_one():
    """--relay-sign -1 → args.relay_sign == -1 (NARROW)."""
    args = _parse(["--relay-sign", "-1"])
    assert args.relay_sign == -1


def test_relay_sign_rejects_two():
    """--relay-sign 2 → SystemExit (nur -1/0/1 erlaubt)."""
    import pytest
    with pytest.raises(SystemExit):
        _parse(["--relay-sign", "2"])


def test_relay_alpha_accepts_float():
    """--relay-alpha 0.5 → args.relay_alpha == 0.5."""
    args = _parse(["--relay-alpha", "0.5"])
    assert args.relay_alpha == 0.5


def test_relay_layer_accepts_int():
    """--relay-layer 15 → args.relay_layer == 15."""
    args = _parse(["--relay-layer", "15"])
    assert args.relay_layer == 15


def test_preset_accepts_baseline():
    """--preset BASELINE → args.preset == 'BASELINE'."""
    args = _parse(["--preset", "BASELINE"])
    assert args.preset == "BASELINE"


def test_preset_accepts_active_manifold_relay():
    """--preset ACTIVE_MANIFOLD_RELAY → args.preset == 'ACTIVE_MANIFOLD_RELAY'."""
    args = _parse(["--preset", "ACTIVE_MANIFOLD_RELAY"])
    assert args.preset == "ACTIVE_MANIFOLD_RELAY"


def test_preset_accepts_unknown_value():
    """--preset unbekannt → wird akzeptiert (Validation passiert im Manager)."""
    args = _parse(["--preset", "WILL_NICHT_EXISTIEREN"])
    assert args.preset == "WILL_NICHT_EXISTIEREN"


def test_session_accepts_arbitrary_string():
    """--session beliebig → exakt übernommen."""
    args = _parse(["--session", "my_custom_session"])
    assert args.session == "my_custom_session"


def test_image_path_accepted():
    """--image /tmp/foo.png → args.image == '/tmp/foo.png'."""
    args = _parse(["--image", "/tmp/foo.png"])
    assert args.image == "/tmp/foo.png"


def test_image_base64_raw_accepted():
    """--image-base64 <data> → args.image_base64 == <data>."""
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII="
    args = _parse(["--image-base64", b64])
    assert args.image_base64 == b64


def test_message_with_spaces_accepted():
    """--message mit Leerzeichen → exakt übernommen."""
    msg = "Dies ist ein Test mit mehreren Wörtern."
    args = _parse(["--message", msg])
    assert args.message == msg


# --- Kombinationen -----------------------------------------------------

def test_full_relay_combination():
    """preset=RELAY + relay-sign=+1 + alpha=0.30 + layer=21 → alle Felder gesetzt."""
    args = _parse([
        "--preset", "ACTIVE_MANIFOLD_RELAY",
        "--relay-sign", "1",
        "--relay-alpha", "0.30",
        "--relay-layer", "21",
    ])
    assert args.preset == "ACTIVE_MANIFOLD_RELAY"
    assert args.relay_sign == 1
    assert args.relay_alpha == 0.30
    assert args.relay_layer == 21


def test_multimodal_combination():
    """--image + --message → beide Felder gesetzt für Multimodal-Turn."""
    args = _parse([
        "--image", "/tmp/bild.jpg",
        "--message", "Was siehst du?",
    ])
    assert args.image == "/tmp/bild.jpg"
    assert args.message == "Was siehst du?"


def test_help_flag_works():
    """--help → SystemExit (argparse-Standard)."""
    import pytest
    with pytest.raises(SystemExit):
        _parse(["--help"])


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
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)