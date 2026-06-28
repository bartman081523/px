"""Pure-logic Tests für model_manager.py:_migrate_preset + _VALID_PRESETS.

Pinnt das Pre-TTS-Verhalten der Preset-Validierung. Wenn jemand die
Preset-Namen oder die Migration-Logik ändert, fallen diese Tests rot.

Refactor-Detector:
- _migrate_preset("ACTIVE_MANIFOLD") → "ACTIVE_MANIFOLD" (kein Change)
- _migrate_preset("OLD_PRESET") → "ACTIVE_MANIFOLD" (gnadenlose Migration)
- _migrate_preset("") → "ACTIVE_MANIFOLD" (leer = unbekannt)

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_model_manager_presets.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import _migrate_preset, _VALID_PRESETS


# --- _VALID_PRESETS coverage --------------------------------------------

def test_valid_presets_includes_baseline():
    """BASELINE ist ein gültiger Preset (kein PX-Patch)."""
    assert "BASELINE" in _VALID_PRESETS


def test_valid_presets_includes_active_manifold():
    """ACTIVE_MANIFOLD ist ein gültiger Preset (voller PX-Patch)."""
    assert "ACTIVE_MANIFOLD" in _VALID_PRESETS


def test_valid_presets_includes_active_manifold_lean():
    """ACTIVE_MANIFOLD_LEAN ist ein gültiger Preset (kausaler Kern ohne
    Crutches)."""
    assert "ACTIVE_MANIFOLD_LEAN" in _VALID_PRESETS


def test_valid_presets_includes_active_manifold_relay():
    """ACTIVE_MANIFOLD_RELAY ist ein gültiger Preset (LEAN + verstärkbar
    Selbst-Injektions-Relay, psychomotrik seite15)."""
    assert "ACTIVE_MANIFOLD_RELAY" in _VALID_PRESETS


def test_valid_presets_count_is_four():
    """Es gibt genau 4 gültige Presets. Wenn das wächst, schlägt dieser
    Test an und zwingt zu Review."""
    assert len(_VALID_PRESETS) == 4


def test_valid_presets_excludes_old_names():
    """Alte Preset-Namen (RIGOR, SUBJECTIVE, DMT-FULL, UNCENSORED) sind
    NICHT mehr gültig → werden via _migrate_preset auf
    ACTIVE_MANIFOLD gemappt."""
    for old in ["RIGOR", "SUBJECTIVE", "DMT-FULL", "UNCENSORED"]:
        assert old not in _VALID_PRESETS, f"old preset '{old}' still in valid set"


# --- _migrate_preset ----------------------------------------------------

def test_migrate_baseline_passes_through():
    """BASELINE bleibt BASELINE (kein Migrations-Hook)."""
    assert _migrate_preset("BASELINE") == "BASELINE"


def test_migrate_active_manifold_passes_through():
    """ACTIVE_MANIFOLD bleibt ACTIVE_MANIFOLD."""
    assert _migrate_preset("ACTIVE_MANIFOLD") == "ACTIVE_MANIFOLD"


def test_migrate_active_manifold_lean_passes_through():
    """ACTIVE_MANIFOLD_LEAN bleibt ACTIVE_MANIFOLD_LEAN."""
    assert _migrate_preset("ACTIVE_MANIFOLD_LEAN") == "ACTIVE_MANIFOLD_LEAN"


def test_migrate_active_manifold_relay_passes_through():
    """ACTIVE_MANIFOLD_RELAY bleibt ACTIVE_MANIFOLD_RELAY."""
    assert _migrate_preset("ACTIVE_MANIFOLD_RELAY") == "ACTIVE_MANIFOLD_RELAY"


def test_migrate_rigor_falls_back_to_active_manifold():
    """RIGOR (alter Name) → ACTIVE_MANIFOLD."""
    assert _migrate_preset("RIGOR") == "ACTIVE_MANIFOLD"


def test_migrate_subjective_falls_back_to_active_manifold():
    """SUBJECTIVE (alter Name) → ACTIVE_MANIFOLD."""
    assert _migrate_preset("SUBJECTIVE") == "ACTIVE_MANIFOLD"


def test_migrate_dmt_full_falls_back_to_active_manifold():
    """DMT-FULL (alter Name) → ACTIVE_MANIFOLD."""
    assert _migrate_preset("DMT-FULL") == "ACTIVE_MANIFOLD"


def test_migrate_uncensored_falls_back_to_active_manifold():
    """UNCENSORED (alter Name) → ACTIVE_MANIFOLD."""
    assert _migrate_preset("UNCENSORED") == "ACTIVE_MANIFOLD"


def test_migrate_empty_string_falls_back_to_active_manifold():
    """Leerer String ist ungültig → ACTIVE_MANIFOLD."""
    assert _migrate_preset("") == "ACTIVE_MANIFOLD"


def test_migrate_none_falls_back_to_active_manifold():
    """None ist ungültig → ACTIVE_MANIFOLD."""
    assert _migrate_preset(None) == "ACTIVE_MANIFOLD"


def test_migrate_typo_falls_back_to_active_manifold():
    """Typo (z.B. 'active_manifold' lowercase) → ACTIVE_MANIFOLD.
    Case-sensitive Lookup, kein Fuzzy-Match."""
    assert _migrate_preset("active_manifold") == "ACTIVE_MANIFOLD"


def test_migrate_random_garbage_falls_back_to_active_manifold():
    """Beliebiger unbekannter String → ACTIVE_MANIFOLD."""
    assert _migrate_preset("garbage_preset_xyz") == "ACTIVE_MANIFOLD"


def test_migrate_always_returns_a_string():
    """Result ist IMMER ein nicht-leerer String (Garantie für
    downstream-Code der den Preset-Namen direkt nutzt)."""
    for invalid in [None, "", "asdf", "DMT-FULL", "rigor"]:
        result = _migrate_preset(invalid)
        assert isinstance(result, str), f"non-string for input {invalid!r}"
        assert len(result) > 0, f"empty string for input {invalid!r}"


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