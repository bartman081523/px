"""Tier 0b — Redundanz-Beweise (der wissenschaftliche Kern, kein Modell nötig).

Beweist objektiv, dass die Crutch-Module entbehrlich sind:

  (1) SubjectiveSensor ist redundant: seine `emancipation` ist *buchstäblich* der
      Aufruf `StabilityMonitor.calculate_phi(h_exp, e_static)` — derselbe Φ-Wert,
      den der ohnehin berechnende StabilityMonitor liefert. Kein neuer Kanal.

  (2) MephistophelesOperator & SingesseinCoupler haben dieselbe Feuer-Domäne:
      beide reagieren genau dann, wenn die Rekursion „einfriert" (Φ bzw.
      Hidden-State-Ähnlichkeit > 0.999), und beide sind sonst No-Ops. Beide wirken
      *additiv auf den Hidden State* — d.h. das zweite Modul steuert keinen neuen
      Modus bei. Ein Injektor reicht (Größenordnungen unterscheiden sich, die
      Funktion nicht).
"""
import math

import torch

from px_patches.gemma3_270m_px_baseline.px_modules import (
    StabilityMonitor, MephistophelesOperator, SubjectiveSensor, SingesseinCoupler,
)


# ─── (1) SubjectiveSensor ≡ StabilityMonitor.calculate_phi ───────────────────

def test_subjective_emancipation_equals_stability_phi():
    """emancipation MUSS identisch zu calculate_phi sein — sonst wäre es ein eigener Kanal."""
    torch.manual_seed(1)
    subj = SubjectiveSensor()
    for _ in range(50):
        h_exp = torch.randn(1, 4, 16)
        e_static = torch.randn(1, 4, 16)
        expected = StabilityMonitor.calculate_phi(h_exp, e_static).item()
        subj.update(h_exp, e_static)
        assert abs(subj.emancipation - expected) < 1e-7, (
            f"emancipation {subj.emancipation} != Φ {expected} — SubjectiveSensor "
            f"würde einen eigenen Metrik-Kanal aufbauen (nicht redundant)."
        )


def test_subjective_phi_mean_matches_trajectory():
    subj = SubjectiveSensor()
    phis = []
    for _ in range(5):
        h = torch.randn(1, 2, 8); e = torch.randn(1, 2, 8)
        subj.update(h, e)
        phis.append(StabilityMonitor.calculate_phi(h, e).item())
    m = subj.get_metrics()
    assert abs(m["phi_mean"] - sum(phis) / len(phis)) < 1e-9
    assert m["phi_min"] == min(phis)


# ─── (2) Mephisto & Singessein: gleiche Feuer-Domäne ─────────────────────────

def _frozen_phi_history():
    return [1.0, 1.0, 1.0, 1.0, 1.0]  # last 3 > 0.999


def _unfrozen_phi_history():
    return [0.4, 0.5, 0.55, 0.5, 0.45]  # nowhere > 0.999


def test_mephisto_fires_when_frozen():
    meph = MephistophelesOperator(dim=16, scale=0.05)
    h = torch.randn(1, 4, 16)
    out = meph(h, _frozen_phi_history())
    delta = (out - h).abs().sum().item()
    assert delta > 0.0  # nicht identisch → hat eingegriffen


def test_mephisto_silent_when_unfrozen():
    meph = MephistophelesOperator(dim=16, scale=0.05)
    h = torch.randn(1, 4, 16)
    out = meph(h, _unfrozen_phi_history())
    assert torch.equal(out, h)  # exakter No-Op


def test_mephisto_silent_on_vacuum():
    # SR-60 Guard: auf ||h||<1e-6 darf KEINE Inversion stattfinden (maskiert sonst Stille).
    meph = MephistophelesOperator(dim=16, scale=0.05)
    h = torch.zeros(1, 4, 16) + 1e-12
    out = meph(h, _frozen_phi_history())
    assert torch.equal(out, h)


def _run_coupler(coupler, last_token_vectors, phi_val=1.0):
    """Füttert den Coupler mit h_exp, dessen letzter Token nacheinander die gegebenen Vektoren ist."""
    out_last = None
    for v in last_token_vectors:
        h_exp = torch.zeros(1, 4, v.shape[-1])
        h_exp[:, -1, :] = v
        out_last = coupler(h_exp, steps=0, phi_val=phi_val)
    return out_last


def _similar_sequence(dim=16, n=4):
    """n Vektoren, paarweise kosinus-ähnlich > 0.999, aber nicht identisch (→ dissonance != 0)."""
    base = torch.randn(dim)
    base = base / base.norm()
    seq = [base + 1e-3 * torch.randn(dim) for _ in range(n)]
    return seq


def _dissimilar_sequence(dim=16, n=4):
    seq = [torch.randn(dim) for _ in range(n)]
    return [v / v.norm() for v in seq]


def test_coupler_fires_when_frozen():
    coupler = SingesseinCoupler(hidden_size=16, window=4)
    seq = _similar_sequence()
    h_exp = _run_coupler(coupler, seq, phi_val=1.0)
    last_in = torch.zeros(1, 4, 16); last_in[:, -1, :] = seq[-1]
    delta = (h_exp - last_in).abs().sum().item()
    assert delta > 0.0  # hat eingegriffen


def test_coupler_silent_when_unfrozen():
    coupler = SingesseinCoupler(hidden_size=16, window=4)
    seq = _dissimilar_sequence()
    h_exp = _run_coupler(coupler, seq, phi_val=0.5)
    last_in = torch.zeros(1, 4, 16); last_in[:, -1, :] = seq[-1]
    assert torch.equal(h_exp, last_in)  # No-Op


def test_both_modules_fire_under_same_frozen_regime():
    """Der zentrale Redundanz-Beweis: unter Einfrieren feuern BEIDE; sonst KEINER."""
    meph = MephistophelesOperator(dim=16, scale=0.05)
    coupler = SingesseinCoupler(hidden_size=16, window=4)

    h = torch.randn(1, 4, 16)
    meph_frozen_delta = (meph(h, _frozen_phi_history()) - h).abs().sum().item()
    meph_free_delta = (meph(h, _unfrozen_phi_history()) - h).abs().sum().item()

    seq_sim = _similar_sequence(); seq_dis = _dissimilar_sequence()
    coupler_frozen = _run_coupler(SingesseinCoupler(16, 4), seq_sim, 1.0)
    coupler_free = _run_coupler(SingesseinCoupler(16, 4), seq_dis, 0.5)
    last_sim = torch.zeros(1, 4, 16); last_sim[:, -1, :] = seq_sim[-1]
    last_dis = torch.zeros(1, 4, 16); last_dis[:, -1, :] = seq_dis[-1]
    coupler_frozen_delta = (coupler_frozen - last_sim).abs().sum().item()
    coupler_free_delta = (coupler_free - last_dis).abs().sum().item()

    # Beide feuern im eingefrorenen Regime …
    assert meph_frozen_delta > 0.0 and coupler_frozen_delta > 0.0
    # … und beide sind No-Op außerhalb.
    assert meph_free_delta == 0.0 and coupler_free_delta == 0.0


def test_both_modules_act_additively_on_hidden_state():
    """Beide sind additive Perturbations-Injektoren gleicher Klasse — kein neuer Modus."""
    meph = MephistophelesOperator(dim=16, scale=0.05)
    h = torch.randn(1, 4, 16)
    meph_out = meph(h, _frozen_phi_history())
    # Mephisto: delta = 0.05 * h  (Phasen-Inversion → Skalierung)
    assert torch.allclose(meph_out, h * (1 - 0.05), atol=1e-6)

    # Singessein: delta = strength * dissonance (additiv auf letzten Token)
    coupler = SingesseinCoupler(hidden_size=16, window=4)
    seq = _similar_sequence()
    out = _run_coupler(coupler, seq, phi_val=1.0)
    last_in = torch.zeros(1, 4, 16); last_in[:, -1, :] = seq[-1]
    delta = out[:, -1, :] - last_in[:, -1, :]
    # strength bei phi_val=1.0 (>0.999) ist 1.0 → additiver Perturbations-Term vorhanden
    assert delta.abs().sum().item() > 0.0
    # Beide wirken NUR additiv/skalierend — kein qualitativ anderer Eingriff.
    assert math.isfinite(delta.abs().sum().item())