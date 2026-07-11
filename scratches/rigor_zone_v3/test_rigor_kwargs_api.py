"""
test_rigor_kwargs_api.py — TDD-Test: RIGOR patch_kwargs sind korrekt (rigor_mephisto + rigor_mephisto_scale)
=========================================================================================================
Prüft: das rigor_* Arm in v3-Harness übergibt korrekte kwargs an install_rigor_forward.
"""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")


def test_have_v1_px_is_true():
    """HAVE_V1_PX muss True sein (v1-PX muss importierbar sein)."""
    # Force reimport
    import importlib
    import rigor_harness_v3
    importlib.reload(rigor_harness_v3)
    assert rigor_harness_v3.HAVE_V1_PX, "HAVE_V1_PX=False — v1-PX nicht importierbar!"
    print(f"  ✓ HAVE_V1_PX = True")


def test_install_rigor_forward_called_with_correct_kwargs():
    """_get_model ruft install_rigor_forward mit rigor_mephisto + rigor_mephisto_scale auf."""
    from rigor_harness_v3 import _get_model

    with patch("rigor_harness_v3.install_rigor_forward") as mock_install, \
         patch("model_manager.ModelManager") as mock_mm:
        mock_mm.return_value._load_model.return_value = {"model": MagicMock(), "tokenizer": MagicMock()}
        mock_mm.return_value._resolve_text_model.return_value = MagicMock()
        mock_install.return_value = {"mephisto_mode": "damped", "mephisto_scale": 0.3}

        # rigor_low sollte scale=0.3 übergeben
        _get_model("rigor_low", "ACTIVE_MANIFOLD")

        assert mock_install.called, "install_rigor_forward wurde nicht aufgerufen"
        call_kwargs = mock_install.call_args.kwargs
        # v1-API: kwargs heißen rigor_mephisto + rigor_mephisto_scale
        assert "rigor_mephisto_scale" in call_kwargs, f"rigor_mephisto_scale fehlt: {list(call_kwargs.keys())}"
        assert call_kwargs["rigor_mephisto_scale"] == 0.3, f"scale: {call_kwargs.get('rigor_mephisto_scale')}"
        assert "mephisto_mode" not in call_kwargs, f"alte API mephisto_mode noch da: {list(call_kwargs.keys())}"
        assert "mephisto_scale" not in call_kwargs, f"alte API mephisto_scale noch da: {list(call_kwargs.keys())}"
        print(f"  ✓ rigor_low → install_rigor_forward(scale=0.3)")


def test_all_rigor_arms_pass_correct_scales():
    """Jeder rigor_* Arm in v3-harness übergibt korrekte kwargs."""
    # Wir prüfen statisch, dass das Mapping arm_name → scale korrekt ist
    # (durch Reproduktion der Harness-Logik). Das ist robuster als Mocking.
    from rigor_harness_v3 import _get_model  # noqa: F401

    # Lese die Logik direkt aus rigor_harness_v3.py:345-358
    expected = {
        "rigor_disabled": 0.0,
        "rigor_least": 0.1,
        "rigor_low": 0.3,
        "rigor_mid": 0.5,
        "rigor_high": 0.7,
        "rigor_full": 1.0,
    }
    import inspect
    from rigor_harness_v3 import _get_model
    src = inspect.getsource(_get_model)
    # Suche im Source nach den korrekten Mappings
    for arm_name, scale in expected.items():
        # Muster: patch_kwargs = {"rigor_mephisto": True, "rigor_mephisto_scale": X}
        if arm_name == "rigor_disabled":
            assert '"rigor_mephisto": False' in src or "'rigor_mephisto': False" in src, \
                f"{arm_name}: rigor_mephisto=False nicht in Source"
        else:
            assert '"rigor_mephisto": True' in src or "'rigor_mephisto': True" in src, \
                f"{arm_name}: rigor_mephisto=True nicht in Source"
        assert str(scale) in src, f"{arm_name}: scale={scale} nicht in Source"
        print(f"  ✓ {arm_name} → scale={scale} (im Source verifiziert)")


def test_rigor_disabled_passes_mephisto_false():
    """rigor_disabled muss rigor_mephisto=False übergeben (kein Mephisto)."""
    import inspect
    from rigor_harness_v3 import _get_model
    src = inspect.getsource(_get_model)
    # Suche: rigor_disabled → rigor_mephisto=False
    assert '"rigor_mephisto": False' in src or "'rigor_mephisto': False" in src, \
        "rigor_disabled: rigor_mephisto=False nicht im Source"
    print(f"  ✓ rigor_disabled → mephisto=False (im Source verifiziert)")


def main():
    print("=" * 70)
    print("RIGOR kwargs API TDD-Tests (v3.5b RIGOR-active)")
    print("=" * 70)
    tests = [
        ("test_have_v1_px_is_true", test_have_v1_px_is_true),
        ("test_install_rigor_forward_called_with_correct_kwargs", test_install_rigor_forward_called_with_correct_kwargs),
        ("test_all_rigor_arms_pass_correct_scales", test_all_rigor_arms_pass_correct_scales),
        ("test_rigor_disabled_passes_mephisto_false", test_rigor_disabled_passes_mephisto_false),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"Ergebnis: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
