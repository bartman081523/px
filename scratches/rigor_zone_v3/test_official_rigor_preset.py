"""test_official_rigor_preset.py — TDD-Test für OFFICIAL_RIGOR Preset (v3.5g).

OFFICIAL_RIGOR kombiniert alle v1-Rigor-Ideen:
  - n_loops = 14 (statt v3 default 2)
  - gamma = 0.10 (statt v3 default 0.08)
  - recur_hub = 10 (math-affin)
  - recur_start = 6 (statt 5)
  - Mephisto scale = 0.3 (v1-Default, gedämpft)

Hypothese: OFFICIAL_RIGOR muss die 5 Baseline-Matches erhalten
(gleicher v3.5f-Schutz via phi-Gate).
"""
from __future__ import annotations

import sys
import os
from unittest.mock import patch, MagicMock

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))


def test_official_rigor_in_arm_configs():
    """OFFICIAL_RIGOR ist als eigener Arm in ARM_CONFIGS."""
    from rigor_scales_v3 import ARM_CONFIGS

    arm_names = [a.name for a in ARM_CONFIGS]
    official = [a for a in ARM_CONFIGS if a.name == "official_rigor"]
    assert len(official) == 1, f"official_rigor nicht in ARM_CONFIGS: {arm_names}"
    a = official[0]
    assert a.preset == "OFFICIAL_RIGOR", f"preset={a.preset}"
    assert a.mephisto_scale == 0.3, f"mephisto_scale={a.mephisto_scale}"
    print(f"  ✓ test_official_rigor_in_arm_configs: official_rigor Arm vorhanden")


def test_official_rigor_inherits_v35f_phi_gate():
    """OFFICIAL_RIGOR nutzt den v3.5f-phi-Gate-Schutz für Reflector-Injection."""
    from px_patches_v3.patch import _px_forward  # noqa
    import inspect
    src = inspect.getsource(_px_forward)
    assert "if phi_s.item() < 0.95" in src or ("if phi" in src and "0.95" in src), \
        "phi-Gate (0.95) in _px_forward nicht gefunden"
    print(f"  ✓ test_official_rigor_inherits_v35f_phi_gate: phi-Gate vorhanden")


def test_official_rigor_preset_in_apply_px_patch():
    """apply_px_patch akzeptiert OFFICIAL_RIGOR als Preset (mapping auf ACTIVE_MANIFOLD)."""
    # Mock-model + config
    mock_model = MagicMock()
    # text_model ist model.model (HF-Convention)
    mock_text_model = MagicMock()
    mock_text_model.config = MagicMock()
    mock_text_model.config.hidden_size = 640  # 270m
    mock_text_model.config.num_hidden_layers = 18
    mock_text_model.config._name_or_path = "google/gemma-3-270m-it"
    mock_model.model = mock_text_model

    # Patche AutoCalibrator (echte Init zu teuer)
    with patch("px_patches_v3.patch.AutoCalibrator") as mock_ac:
        mock_ac.return_value = MagicMock()
        from px_patches_v3.patch import apply_px_patch
        try:
            apply_px_patch(mock_model, config_preset="OFFICIAL_RIGOR")
        except Exception as e:
            # Echte apply_px_patch macht viel — wir prüfen nur _px_config
            pass

    # _px_config wurde gesetzt?
    assert hasattr(mock_text_model, "_px_config"), "_px_config nicht gesetzt"
    cfg = mock_text_model._px_config
    assert isinstance(cfg, dict), f"cfg ist kein dict: {type(cfg)}"

    # OFFICIAL_RIGOR-Werte
    assert cfg.get("n_loops") == 14, f"n_loops={cfg.get('n_loops')}, erwartet 14"
    assert cfg.get("gamma") == 0.10, f"gamma={cfg.get('gamma')}, erwartet 0.10"
    assert cfg.get("bimodal_hub") == 10, f"bimodal_hub={cfg.get('bimodal_hub')}, erwartet 10"
    assert cfg.get("recur_start") == 6, f"recur_start={cfg.get('recur_start')}, erwartet 6"
    assert cfg.get("recur_end") == 12, f"recur_end={cfg.get('recur_end')}, erwartet 12"
    print(f"  ✓ test_official_rigor_preset_in_apply_px_patch:")
    print(f"    n_loops={cfg['n_loops']}, gamma={cfg['gamma']}, hub={cfg['bimodal_hub']}")
    print(f"    recur_start={cfg['recur_start']}, recur_end={cfg['recur_end']}")


def test_official_rigor_user_kwargs_override():
    """User kann OFFICIAL_RIGOR-Werte per kwargs überschreiben."""
    mock_model = MagicMock()
    mock_text_model = MagicMock()
    mock_text_model.config = MagicMock()
    mock_text_model.config.hidden_size = 640
    mock_text_model.config.num_hidden_layers = 18
    mock_text_model.config._name_or_path = "google/gemma-3-270m-it"
    mock_model.model = mock_text_model

    with patch("px_patches_v3.patch.AutoCalibrator") as mock_ac:
        mock_ac.return_value = MagicMock()
        from px_patches_v3.patch import apply_px_patch
        try:
            apply_px_patch(mock_model, config_preset="OFFICIAL_RIGOR", n_loops=20)
        except Exception as e:
            pass

    cfg = mock_text_model._px_config
    assert cfg.get("n_loops") == 20, f"User-Override n_loops=20 fehlgeschlagen: {cfg.get('n_loops')}"
    # Andere OFFICIAL_RIGOR-Werte bleiben (weil User sie nicht überschrieben hat)
    assert cfg.get("gamma") == 0.10, f"gamma sollte 0.10 bleiben, ist {cfg.get('gamma')}"
    print(f"  ✓ test_official_rigor_user_kwargs_override: User-kwargs gewinnen (n_loops=20)")


def main() -> int:
    print("="*70)
    print("TDD test_official_rigor_preset (v3.5g)")
    print("="*70)
    tests = [
        ("test_official_rigor_in_arm_configs", test_official_rigor_in_arm_configs),
        ("test_official_rigor_inherits_v35f_phi_gate", test_official_rigor_inherits_v35f_phi_gate),
        ("test_official_rigor_preset_in_apply_px_patch", test_official_rigor_preset_in_apply_px_patch),
        ("test_official_rigor_user_kwargs_override", test_official_rigor_user_kwargs_override),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR ({type(e).__name__}): {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*70}")
    print(f"Tests: {len(tests) - failed}/{len(tests)} grün")
    print(f"{'='*70}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
