"""
RIGOR-spezifische Modifikationen am text_model NACH apply_px_patch.
Monkey-Patcht zur Laufzeit (KEINE Code-Änderung an Production).

Vier additive Operationen:
  1. Mephisto-Damping (wenn rigor_mephisto=True + scale < 1.0)
     → Mephisto.Forward-Wrapper multipliziert Output mit scale
  2. Mephisto-Entfernung (wenn rigor_mephisto=False)
     → text_model._px_mephisto = None + _px_forward-Patch der Mephisto-Calls skipped
  3. Zone-Override (gamma, n_loops für RIGOR-Zone)
     → Da _px_forward Zone-Block hardcoded ist (Z. 396-410) und wir den
       nicht monkey-patchen können ohne komplette _px_forward-Recreation,
       lassen wir das. Die Zone selbst wird via rigor_zone_manifold.py
       korrekt klassifiziert — der get_routing_params nutzt RIGOR-Routing.
  4. Cleanup-Funktion (für sauberen Test-Teardown)
"""
import os
import sys
import types
from typing import Optional, Dict, Any

# Sicherstellen dass das Repo-Root im Path ist
_REPO_ROOT = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigor_zone_config import (
    RIGOR_MEPHISTO_DEFAULT_SCALE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup-Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _save_original(obj, attr_name):
    """Speichert Original-Wert unter _rigor_orig_<attr_name> wenn noch nicht da."""
    backup_attr = f"_rigor_orig_{attr_name}"
    if not hasattr(obj, backup_attr):
        if hasattr(obj, attr_name):
            setattr(obj, backup_attr, getattr(obj, attr_name))


def _restore_original(obj, attr_name):
    """Stellt Original-Wert wieder her und entfernt Backup."""
    backup_attr = f"_rigor_orig_{attr_name}"
    if hasattr(obj, backup_attr):
        setattr(obj, attr_name, getattr(obj, backup_attr))
        delattr(obj, backup_attr)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Mephisto-Damping
# ═══════════════════════════════════════════════════════════════════════════════

def install_mephisto_damping(text_model, scale: float) -> None:
    """Wickelt text_model._px_mephisto.forward mit Damping (scale-Faktor).

    Backup des Original-Forwards wird unter _rigor_orig_forward gespeichert.
    Cleanup via restore_mephisto_damping(text_model).
    """
    if not hasattr(text_model, "_px_mephisto") or text_model._px_mephisto is None:
        # LEAN-Modus hat kein Mephisto — nichts zu dämpfen
        return
    mephisto = text_model._px_mephisto
    if not hasattr(mephisto, "forward"):
        return
    _save_original(mephisto, "forward")
    _orig_forward = mephisto.forward
    def _damped_forward(h, phi_history):
        out = _orig_forward(h, phi_history)
        return out * scale
    mephisto.forward = _damped_forward


def restore_mephisto_damping(text_model) -> None:
    """Stellt Mephisto-Original-Forward wieder her."""
    if not hasattr(text_model, "_px_mephisto") or text_model._px_mephisto is None:
        return
    _restore_original(text_model._px_mephisto, "forward")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Mephisto-Entfernung (für rigor_mephisto=False)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Wir können _px_mephisto nicht einfach auf None setzen, weil _px_forward
# (Production-Code) Mephisto aufruft. Stattdessen: leise Variante wo
# Mephisto.forward = Identity wird (return input unchanged).

def install_mephisto_disabled(text_model) -> None:
    """Deaktiviert Mephisto (für rigor_mephisto=False). Mephisto wird zum
    NoOp-Operator: forward returnt input unverändert.

    Backup des Original-Forwards unter _rigor_orig_forward.
    Cleanup via restore_mephisto_damping (gleiche Restore-Funktion).
    """
    if not hasattr(text_model, "_px_mephisto") or text_model._px_mephisto is None:
        # Bereits kein Mephisto (LEAN-Pfad in Production) — nichts zu tun
        return
    mephisto = text_model._px_mephisto
    if not hasattr(mephisto, "forward"):
        return
    _save_original(mephisto, "forward")
    def _noop_forward(h, phi_history):
        return h
    mephisto.forward = _noop_forward


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Zentraler Entry-Point
# ═══════════════════════════════════════════════════════════════════════════════

def install_rigor_forward(text_model, config_preset: str, **kwargs) -> Dict[str, Any]:
    """Installiert RIGOR-spezifische Modifikationen am text_model.

    Args:
        text_model: das text_model (text_model des HF-Modells, NICHT der Outer-Wrapper)
        config_preset: "RIGOR" — wirkt nur bei RIGOR, sonst NoOp
        **kwargs:
            rigor_mephisto: bool = True (User-Default: an als Denkanstoß)
            rigor_mephisto_scale: float = 0.3 (30% von Original)

    Returns:
        Dict mit installierten Modi (für Cleanup-Tracking)
    """
    if config_preset != "RIGOR":
        return {"mephisto_mode": "untouched", "mephisto_scale": 1.0}

    rigor_mephisto = bool(kwargs.get("rigor_mephisto", True))
    rigor_mephisto_scale = float(
        kwargs.get("rigor_mephisto_scale", RIGOR_MEPHISTO_DEFAULT_SCALE)
    )

    if not rigor_mephisto:
        # rigor_mephisto=False → Mephisto NoOp
        install_mephisto_disabled(text_model)
        return {"mephisto_mode": "disabled", "mephisto_scale": 0.0}
    elif rigor_mephisto_scale < 1.0:
        # rigor_mephisto=True + scale < 1.0 → Mephisto gedämpft
        install_mephisto_damping(text_model, rigor_mephisto_scale)
        return {"mephisto_mode": "damped", "mephisto_scale": rigor_mephisto_scale}
    else:
        # rigor_mephisto=True + scale >= 1.0 → Mephisto unverändert
        return {"mephisto_mode": "untouched", "mephisto_scale": 1.0}


def restore_rigor_forward(text_model) -> None:
    """Cleanup: stellt alle Mephisto-Original-Forwards wieder her."""
    restore_mephisto_damping(text_model)
    # Kein separater restore für disabled — gleiche Restore-Logik
    # (Backup wird unter _rigor_orig_forward gespeichert)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-Test
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    print("="*60)
    print("RIGOR Forward — Smoke-Test")
    print("="*60)

    # Numerisches Mock mit numpy für sinnvolle Math
    import numpy as np
    class MockMephisto:
        def __init__(self):
            self.forward = self._orig_forward
            self.scale = 0.05
        def _orig_forward(self, h, phi_history):
            # Produziert Output mit Magnitude 2.0 * h
            return h * 2.0
    def make_tm():
        tm = type("TM", (), {})()
        tm._px_mephisto = MockMephisto()
        return tm

    # Test 1: rigor_mephisto=True, scale=1.0 (explicit) → untouched
    tm = make_tm()
    result = install_rigor_forward(tm, "RIGOR", rigor_mephisto_scale=1.0)
    assert result["mephisto_mode"] == "untouched", f"expected untouched, got {result}"
    h = np.array([1.0, 2.0, 3.0])
    out = tm._px_mephisto.forward(h, [])
    assert np.allclose(out, h * 2.0), f"untouched failed: {out}"
    print(f"  scale=1.0 (untouched): {result}")

    # Test 1b: rigor_mephisto=True, default scale=0.3 → damped
    tm = make_tm()
    result = install_rigor_forward(tm, "RIGOR")  # default scale=0.3
    assert result["mephisto_mode"] == "damped", f"expected damped, got {result}"
    h = np.array([1.0, 2.0, 3.0])
    out = tm._px_mephisto.forward(h, [])
    # 2.0 * 0.3 = 0.6
    assert np.allclose(out, h * 0.6), f"damping math wrong: {out}"
    print(f"  default scale=0.3 (damped): {result}")
    restore_rigor_forward(tm)
    out_restored = tm._px_mephisto.forward(h, [])
    assert np.allclose(out_restored, h * 2.0), f"restore failed: {out_restored}"
    print(f"  Restore nach damped: OK")

    # Test 3: rigor_mephisto=False → disabled (NoOp)
    tm = make_tm()
    result = install_rigor_forward(tm, "RIGOR", rigor_mephisto=False)
    h = np.array([1.0, 2.0, 3.0])
    out = tm._px_mephisto.forward(h, [])
    assert np.allclose(out, h), f"disable failed: {out}"
    print(f"  Disabled: {result}")
    restore_rigor_forward(tm)
    out_restored = tm._px_mephisto.forward(h, [])
    assert np.allclose(out_restored, h * 2.0), f"final restore failed: {out_restored}"
    print(f"  Final restore: OK")

    # Test 4: anderer preset → NoOp
    tm = make_tm()
    result = install_rigor_forward(tm, "ACTIVE_MANIFOLD")
    assert result["mephisto_mode"] == "untouched"
    out = tm._px_mephisto.forward(h, [])
    assert np.allclose(out, h * 2.0), "Non-RIGOR should not modify Mephisto"
    print(f"  Non-RIGOR preset: {result}")

    print("="*60)


if __name__ == "__main__":
    _smoke_test()
