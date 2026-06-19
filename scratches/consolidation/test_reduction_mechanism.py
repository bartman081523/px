"""Tier 0c — Reduktions-Mechanismus (Mock-Modell, kein echtes LLM).

Baut ein Dummy-text_model mit den realen Crutch-Modul-Instanzen (denselben
Attributen, die apply_px_patch setzt) und prüft, dass reduction.apply_reduction:
  - die 4 Crutch-Attribute löscht (→ hasattr-Guards in _px_forward werden inaktiv),
  - die AZS.forward AM EXEMPLAR überschreibt: additive Injektion ENTFÄLLT,
    Entropie H + gamma_boost BLEIBEN,
  - restore_reduction den Override zurücknimmt.

Kein px_patches-Source wird angerührt; alles passiert zur Laufzeit am Objekt.
"""
import types

import torch

from px_patches.gemma3_270m_px_baseline.px_modules import (
    AksSensor, MephistophelesOperator, SubjectiveSensor, SingesseinCoupler,
)
from px_patches.gemma3_270m_px_baseline.anti_zombie_sensor import AntiZombieSensor
from px_patches.gemma3_270m_px_baseline.auto_tune import AutoCalibrator

import reduction


def _make_patched_dummy(hidden_size=16):
    """Repliziert die Attribut-Belegung, die apply_px_patch auf text_model setzt."""
    tm = types.SimpleNamespace()
    tm._px_calibrator = AutoCalibrator(hidden_size)
    tm._px_aks = AksSensor()
    tm._px_coupler = SingesseinCoupler(hidden_size)
    tm._px_mephisto = MephistophelesOperator(hidden_size)
    tm._px_azs = AntiZombieSensor(hidden_size)
    tm._px_subj_sensor = SubjectiveSensor()
    return tm


def test_all_crutches_present_before_reduction():
    tm = _make_patched_dummy()
    active = reduction.list_active_crutches(tm)
    for c in ("aks", "mephisto", "coupler", "subjective", "injection"):
        assert c in active


def test_apply_reduction_full_removes_four_attrs():
    tm = _make_patched_dummy()
    removed = reduction.apply_reduction(tm, drop="all")
    assert set(removed) == {"aks", "mephisto", "coupler", "subjective", "injection"}
    assert not hasattr(tm, "_px_aks")
    assert not hasattr(tm, "_px_mephisto")
    assert not hasattr(tm, "_px_coupler")
    assert not hasattr(tm, "_px_subj_sensor")
    # Kern bleibt unangetastet:
    assert hasattr(tm, "_px_calibrator")
    assert hasattr(tm, "_px_azs")


def test_apply_reduction_keeps_calibrator_and_azs():
    tm = _make_patched_dummy()
    reduction.apply_reduction(tm, drop="all")
    assert isinstance(tm._px_calibrator, AutoCalibrator)
    assert isinstance(tm._px_azs, AntiZombieSensor)


def test_partial_drop_only_removes_named():
    tm = _make_patched_dummy()
    reduction.apply_reduction(tm, drop=("aks", "subjective"))
    assert not hasattr(tm, "_px_aks") and not hasattr(tm, "_px_subj_sensor")
    assert hasattr(tm, "_px_mephisto") and hasattr(tm, "_px_coupler")
    assert hasattr(tm, "_px_azs")  # Injektion noch aktiv
    assert "injection" in reduction.list_active_crutches(tm)


def test_azs_override_drops_additive_injection():
    """Nach Override darf der letzte Token NICHT mehr durch awareness_latent verschoben werden."""
    tm = _make_patched_dummy()
    reduction.apply_reduction(tm, drop=("injection",))

    azs = tm._px_azs
    hidden = torch.randn(2, 5, 16)
    zone_w = {"math": 0.2, "logic_a": 0.2, "creative": 0.2, "logic_b": 0.2, "synthesis": 0.2}
    out, H = azs(hidden, phi=0.99, aks_friction=0.3, emancipation=0.5, zone_weights=zone_w)
    # hidden_states UNVERÄNDERT zurück → kein additiver Term auf letztem Token.
    assert torch.equal(out, hidden)


def test_azs_override_keeps_entropy_correct():
    tm = _make_patched_dummy()
    reduction.apply_reduction(tm, drop=("injection",))
    azs = tm._px_azs
    zone_w = {"math": 0.2, "logic_a": 0.2, "creative": 0.2, "logic_b": 0.2, "synthesis": 0.2}
    hidden = torch.randn(1, 3, 16)
    _, H = azs(hidden, 0.99, 0.3, 0.5, zone_w)
    # H wurde aus weight_EMA berechnet (alpha=0.1, ein Update) → nahe log(5), aber nicht exakt.
    assert 0.0 < H.item() <= 1.6 + 1e-3


def test_azs_override_keeps_gamma_boost():
    """get_feedback_scalars nutzt self.weight_ema — muss nach Override noch funktionieren."""
    tm = _make_patched_dummy()
    reduction.apply_reduction(tm, drop=("injection",))
    azs = tm._px_azs
    # weight_ema auf peaked (Zombie-Regime) → gamma_boost > 1.0
    azs.weight_ema = torch.tensor([0.99, 0.0025, 0.0025, 0.0025, 0.0025])
    fb = azs.get_feedback_scalars(aks_friction=0.0)
    assert fb["gamma_boost"] > 1.0
    assert fb["gamma_boost"] <= 1.5


def test_azs_override_preserves_weight_ema_update():
    tm = _make_patched_dummy()
    reduction.apply_reduction(tm, drop=("injection",))
    azs = tm._px_azs
    before = azs.weight_ema.clone()
    hidden = torch.randn(1, 3, 16)
    zone_w = {"math": 0.9, "logic_a": 0.02, "creative": 0.02, "logic_b": 0.02, "synthesis": 0.04}
    azs(hidden, 0.99, 0.3, 0.5, zone_w)
    after = azs.weight_ema
    # EMA hat sich in Richtung der peaked zone_weights bewegt → nicht mehr equal.
    assert not torch.allclose(before, after)


def test_restore_reduction_reverts_forward():
    tm = _make_patched_dummy()
    reduction.apply_reduction(tm, drop=("injection",))
    azs = tm._px_azs
    assert getattr(azs, "_px_injection_original_forward", None) is not None
    reduction.restore_reduction(tm)
    assert getattr(azs, "_px_injection_original_forward", None) is None
    # Nach restore: Original-forward injiziert wieder (letzte Token wird verschoben).
    hidden = torch.randn(1, 3, 16)
    zone_w = {"math": 0.2, "logic_a": 0.2, "creative": 0.2, "logic_b": 0.2, "synthesis": 0.2}
    out, _ = azs(hidden, 0.99, 0.3, 0.5, zone_w)
    assert not torch.equal(out, hidden)  # Injektion wieder aktiv


def test_full_reduction_then_azs_still_returns_entropy():
    """Simuliert den -all-Pfad: AZS läuft weiter, liefert H, injiziert aber nicht."""
    tm = _make_patched_dummy()
    reduction.apply_reduction(tm, drop="all")
    azs = tm._px_azs
    hidden = torch.randn(1, 4, 16)
    zone_w = {"math": 0.2, "logic_a": 0.2, "creative": 0.2, "logic_b": 0.2, "synthesis": 0.2}
    out, H = azs(hidden, 0.95, 0.0, 0.0, zone_w)  # aks/emancipation fallen weg → 0.0
    assert torch.equal(out, hidden)
    assert 0.0 < H.item()
    fb = azs.get_feedback_scalars(0.0)
    assert "gamma_boost" in fb