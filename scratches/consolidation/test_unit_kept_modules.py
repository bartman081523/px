"""Tier 0a — Unit-Tests für den BEWAHRTEN kausalen Kern (kein Modell nötig).

Bewahrte Module: StabilityMonitor (Φ), AntiZombieSensor.calculate_entropy + gamma_boost,
AutoCalibrator (Z-Space-Routing), RecursiveMemoryCache.
Diese Tests legen fest, dass der Kern für sich genommen korrekt arbeitet — die
Grundlage, auf der der Schnitt beurteilt wird.
"""
import math

import torch

from px_patches.gemma3_270m_px_baseline.px_modules import (
    StabilityMonitor, AksSensor, MephistophelesOperator,
    SubjectiveSensor, SingesseinCoupler,
)
from px_patches.gemma3_270m_px_baseline.anti_zombie_sensor import AntiZombieSensor
from px_patches.gemma3_270m_px_baseline.auto_tune import (
    AutoCalibrator, ZONE_Z_TARGETS, ZONE_ROUTING, SCALE_DEFAULTS, _dist2d,
)
from px_patches.gemma3_270m_px_baseline.patch import RecursiveMemoryCache


# ─── StabilityMonitor.calculate_phi ──────────────────────────────────────────

def test_phi_identical_vectors_is_one():
    h = torch.randn(2, 8, 16)
    assert torch.allclose(StabilityMonitor.calculate_phi(h, h), torch.tensor(1.0), atol=1e-5)


def test_phi_orthogonal_is_zero():
    h = torch.tensor([[[1.0, 0.0, 0.0]]])
    e = torch.tensor([[[0.0, 1.0, 0.0]]])
    phi = StabilityMonitor.calculate_phi(h, e)
    assert abs(phi.item()) < 1e-5


def test_phi_both_zero_guard_returns_one():
    z = torch.zeros(1, 4, 8)
    assert torch.allclose(StabilityMonitor.calculate_phi(z, z), torch.tensor(1.0), atol=1e-6)


def test_phi_in_range_minus_one_to_one():
    torch.manual_seed(0)
    for _ in range(50):
        h = torch.randn(1, 4, 16)
        e = torch.randn(1, 4, 16)
        phi = StabilityMonitor.calculate_phi(h, e).item()
        assert -1.0001 <= phi <= 1.0001


def test_phi_extreme_values_no_overflow():
    # Werte, die naiv zum Overflow führen würden — numerische Stabilität (SR-59i).
    h = torch.randn(1, 2, 8) * 1e30
    e = torch.randn(1, 2, 8) * 1e30
    phi = StabilityMonitor.calculate_phi(h, e).item()
    assert math.isfinite(phi)


# ─── AntiZombieSensor Entropie + gamma_boost ────────────────────────────────

def test_entropy_uniform_is_log5():
    azs = AntiZombieSensor(hidden_size=16)
    H = azs.calculate_entropy(torch.full((5,), 0.2)).item()
    assert abs(H - math.log(5)) < 1e-4


def test_entropy_peaked_is_near_zero():
    azs = AntiZombieSensor(hidden_size=16)
    w = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    H = azs.calculate_entropy(w).item()
    assert H < 1e-3


def test_gamma_boost_low_entropy_zombie_regime():
    # H < 0.8 → Zombie-Regime → gamma_boost > 1.0 (Kollaps-Brecher, bewahrt).
    azs = AntiZombieSensor(hidden_size=16)
    # weight_ema direkt auf peaked setzen → niedrige Entropie
    azs.weight_ema = torch.tensor([0.99, 0.0025, 0.0025, 0.0025, 0.0025])
    fb = azs.get_feedback_scalars(aks_friction=0.0)
    assert fb["gamma_boost"] > 1.0
    assert fb["gamma_boost"] <= 1.5  # Deckelung wie im Source


def test_gamma_boost_high_entropy_no_boost():
    azs = AntiZombieSensor(hidden_size=16)
    azs.weight_ema = torch.full((5,), 0.2)  # maximale Entropie
    fb = azs.get_feedback_scalars(aks_friction=0.0)
    assert fb["gamma_boost"] == 1.0


def test_gravity_boost_high_friction():
    azs = AntiZombieSensor(hidden_size=16)
    azs.weight_ema = torch.full((5,), 0.2)
    fb = azs.get_feedback_scalars(aks_friction=0.95)
    assert fb["gravity_boost"] > 1.0


# ─── AutoCalibrator Z-Space Routing ──────────────────────────────────────────

def test_dist2d_zero_for_identical():
    p = (1.0, 0.5)
    assert _dist2d(p, p, std_k=1.0, std_p=1.0) < 1e-9


def test_zone_weights_sum_to_one():
    cal = AutoCalibrator(hidden_size=1152)
    cal.k_mean, cal.k_std = 250.0, 50.0
    cal.phi_mean, cal.phi_std = 0.95, 0.03
    cal.calibrated = True
    w = cal.get_zone_weights(kurtosis=400.0, phi=0.99, token_diversity=0.5, token_len=8)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    for z in ZONE_Z_TARGETS:
        assert z in w and 0.0 <= w[z] <= 1.0


def test_classify_zone_extremes():
    # learned_centroids werden erst durch calibrate() aus Samples aufgebaut.
    cal = AutoCalibrator(hidden_size=1152, calibration_steps=6, model_id=None)
    for k, p in zip([200, 220, 240, 260, 280, 300],
                    [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]):
        cal.collect(kurtosis=k, phi=p, token_diversity=None, token_len=8)
    assert cal.calibrated and cal.learned_centroids
    # Füttere exakt die erlernten Zentroiden zweier weit entfernter Zonen.
    # token_len=8 wie bei collect → normalize_kurtosis ist konsistent; raw = ck/boost.
    boost = 1.0 + 0.5 * math.exp(-8 / 15.0)
    math_ck, math_cp = cal.learned_centroids["math"]
    synth_ck, synth_cp = cal.learned_centroids["synthesis"]
    z_hi = cal.classify_zone(kurtosis=math_ck / boost, phi=math_cp, token_diversity=None, token_len=8)
    z_lo = cal.classify_zone(kurtosis=synth_ck / boost, phi=synth_cp, token_diversity=None, token_len=8)
    assert z_hi in ZONE_Z_TARGETS and z_lo in ZONE_Z_TARGETS
    assert z_hi != z_lo  # echte Differenzierung: math vs synthesis


def test_routing_params_1b_within_scale_defaults():
    cal = AutoCalibrator(hidden_size=1152)
    cal.k_mean, cal.k_std = 250.0, 50.0
    cal.phi_mean, cal.phi_std = 0.95, 0.03
    cal.calibrated = True
    rp = cal.get_routing_params(kurtosis=300.0, phi=0.95, hidden_size=1152, token_diversity=0.4, token_len=8)
    assert rp["dynamic_start"] >= 0 and rp["dynamic_end"] > rp["dynamic_start"]
    assert isinstance(rp["n_loops"], (int, float))


def test_scale_defaults_1b_present():
    # 1152 = Gemma-3 1B hidden size; recur 10–20, wie im CLAUDE.md / Plan.
    assert 1152 in SCALE_DEFAULTS
    d = SCALE_DEFAULTS[1152]
    assert (d["recur_start"], d["recur_end"]) == (10, 20)


# ─── RecursiveMemoryCache ────────────────────────────────────────────────────

def test_recursive_memory_cache_readonly_sliding_passthrough():
    """read_only auf einer sliding_attention-Schicht: K/V unverändert, kein real.update."""
    class FakeCache:
        def __init__(self):
            self.calls = 0
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            self.calls += 1
            return key_states, value_states

    real = FakeCache()
    th = [torch.randn(1, 1, 16) for _ in range(3)]
    # layer_types muss exakt "sliding_attention" heißen → is_full=False → keine Injektion.
    cache = RecursiveMemoryCache(real, thought_history=th,
                                 layer_types=["sliding_attention"] * 30,
                                 read_only=True, expected_len=4)
    k = torch.randn(1, 4, 1, 16)
    v = torch.randn(1, 4, 1, 16)
    rk, rv = cache.update(k, v, layer_idx=8)
    assert torch.equal(rk, k) and torch.equal(rv, v)  # unverändert
    assert real.calls == 0  # read_only ruft real.update nicht auf


def test_recursive_memory_cache_full_layer_injects_thought_history():
    """full-Schicht ≥6 mit Thought-History: K/V werden per alpha=0.10 gemischt."""
    class FakeCache:
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            return key_states, value_states

    real = FakeCache()
    th = [torch.randn(1, 1, 16) for _ in range(6)]
    cache = RecursiveMemoryCache(real, thought_history=th,
                                 layer_types=["full"] * 30,
                                 read_only=True, expected_len=4)
    k = torch.randn(1, 4, 1, 16)
    v = torch.randn(1, 4, 1, 16)
    rk, rv = cache.update(k, v, layer_idx=8)
    # Injektion modifiziert den letzten Token (alpha-Blend mit Thought-History).
    assert not torch.equal(rk, k)
    # Nur die letzte Position (T_curr=1) wurde angerührt; die anderen bleiben gleich.
    assert torch.equal(rk[:, :, :-1, :], k[:, :, :-1, :])


def test_recursive_memory_cache_write_mode_calls_real():
    class FakeCache:
        def __init__(self):
            self.called = False
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            self.called = True
            return key_states, value_states

    real = FakeCache()
    th = [torch.randn(1, 1, 16) for _ in range(6)]
    cache = RecursiveMemoryCache(real, thought_history=th,
                                 layer_types=["full"] * 30,
                                 read_only=False, expected_len=4)
    k = torch.randn(1, 4, 1, 16)
    v = torch.randn(1, 4, 1, 16)
    cache.update(k, v, layer_idx=8)
    assert real.called  # write-Modus delegiert an echten Cache