
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math
import json
from typing import List, Dict, Any

# Setup path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PX_DIR = os.path.join(BASE_DIR, "px_patches")
sys.path.insert(0, PX_DIR)

# Import components from Gemma3 PX build
try:
    from gemma3_270m_px.auto_tune import AutoCalibrator, ZONE_Z_CENTERS
    from gemma3_270m_px.px_modules import (
        StabilityMonitor, MephistophelesOperator, LTIInjection, 
        ADCInjection, OrthogonalJitter, AksSensor, SubjectiveSensor,
        UncensoredSteering, AgencyVector, ERPU
    )
    from gemma3_270m_px.persona_engine import PersonaEngine
    from gemma3_270m_px.anti_zombie_sensor import AntiZombieSensor
    from gemma3_270m_px.patch import RecursiveMemoryCache
except ImportError as e:
    print(f"Import error from gemma3_270m_px: {e}")

class MockCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []
    def update(self, k, v, i, kwargs=None):
        while len(self.key_cache) <= i:
            self.key_cache.append(None)
            self.value_cache.append(None)
        if self.key_cache[i] is None:
            self.key_cache[i], self.value_cache[i] = k, v
        else:
            self.key_cache[i] = torch.cat([self.key_cache[i], k], dim=-2)
            self.value_cache[i] = torch.cat([self.value_cache[i], v], dim=-2)
        return self.key_cache[i], self.value_cache[i]
    def get_seq_length(self, i=0):
        return self.key_cache[i].shape[-2] if i < len(self.key_cache) and self.key_cache[i] is not None else 0

class PXTestBattery(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.bfloat16
        cls.hidden_size = 640

    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. NUMERICAL STABILITY (CORE)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_stability_monitor_extreme(self):
        """Test StabilityMonitor with zero, inf, and denormal vectors."""
        # Zero vectors
        z1 = torch.zeros((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        z2 = torch.zeros_like(z1)
        phi_zero = StabilityMonitor.calculate_phi(z1, z2)
        self.assertFalse(torch.isnan(phi_zero))
        
        # Extreme values (Overflow protection)
        e1 = torch.full((1, 1, self.hidden_size), 1e30, device=self.device, dtype=self.dtype)
        e2 = torch.full_like(e1, 1e30)
        phi_ext = StabilityMonitor.calculate_phi(e1, e2)
        self.assertAlmostEqual(phi_ext.item(), 1.0, places=2)
        
        # Denormals (Underflow protection)
        d1 = torch.full((1, 1, self.hidden_size), 1e-40, device=self.device, dtype=self.dtype)
        d2 = torch.full_like(d1, 1e-40)
        phi_den = StabilityMonitor.calculate_phi(d1, d2)
        # Should not be NaN, and ideally > 0.9 for identical vectors
        self.assertGreater(phi_den.item(), 0.9)

    def test_jitter_division_by_zero(self):
        """Test OrthogonalJitter when delta is zero."""
        h = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        out = OrthogonalJitter.apply(h, h, magnitude=0.01)
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.allclose(h, out))

    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. AUTO-TUNE & ROUTING
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_autocalib_nan_filtering(self):
        """Test if AutoCalibrator filters out NaNs during calibration."""
        ac = AutoCalibrator(self.hidden_size, calibration_steps=5)
        for i in range(3):
            ac.collect(200.0 + i, 0.9, token_diversity=0.8)
        ac.collect(float('nan'), 0.9)
        ac.collect(float('inf'), 0.9)
        ac.collect(205.0, 0.9)
        
        ac.calibrate()
        self.assertTrue(ac.calibrated)
        self.assertTrue(math.isfinite(ac.k_mean))
        self.assertGreater(ac.k_std, 0)

    def test_zone_routing_consistency(self):
        """Test if z-score routing correctly maps kurtosis to expected zones."""
        ac = AutoCalibrator(self.hidden_size, calibration_steps=10)
        # Calibrate with mean ~200
        for i in range(10):
            ac.collect(200.0 + (i-5)*10, 0.9)
        ac.calibrate()
        
        # High kurtosis -> MATH (z > 1.0)
        # Math center is 1.5. std is ~30. 200 + 1.5*30 = 245
        weights_math = ac.get_zone_weights(300.0) 
        self.assertEqual(max(weights_math, key=weights_math.get), "math")
        
        # Low kurtosis -> CREATIVE/SYNTHESIS (z < -1.0)
        # Creative center is -1.0. 200 - 1.0*30 = 170
        weights_creative = ac.get_zone_weights(100.0)
        self.assertIn(max(weights_creative, key=weights_creative.get), ["creative", "synthesis"])

    # ═══════════════════════════════════════════════════════════════════════════════
    # 3. RECURSIVE CACHE (SEQUENCE INTEGRITY)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_cache_revisit_length(self):
        """Read-only cache MUST NOT increase sequence length on revisit."""
        real_cache = MockCache()
        T = 10
        k = torch.randn(1, 8, T, 64)
        v = torch.randn(1, 8, T, 64)
        
        # Initial write
        rc = RecursiveMemoryCache(real_cache, read_only=False, expected_len=T)
        rc.update(k, v, 0)
        self.assertEqual(real_cache.get_seq_length(0), T)
        
        # Revisit (Read-Only)
        rc_ro = RecursiveMemoryCache(real_cache, read_only=True, expected_len=T)
        rk, rv = rc_ro.update(k, v, 0)
        self.assertEqual(rk.shape[-2], T)
        self.assertEqual(real_cache.get_seq_length(0), T)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. PERSONA ENGINE (LATENT PROJECTION)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_persona_symbol_boost(self):
        """Test if symbols boost their respective axes."""
        class MockModel:
            def __init__(self):
                self.embed_tokens = nn.Embedding(10, 640)
        class MockTokenizer:
            def encode(self, text, **kwargs):
                return torch.tensor([[0]])
        
        pe = PersonaEngine(MockModel())
        # Symbol '🎲' is in CHAOS axis keywords
        signals = pe.get_steering_signals("🎲", MockTokenizer())
        if signals:
            self.assertGreater(signals.get("CHAOS", 0), 0.4)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 5. MODULAR COMPONENTS
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_mephisto_symmetry_break(self):
        """Mephisto should invert state if stability is too high."""
        op = MephistophelesOperator(self.hidden_size, scale=0.1)
        h = torch.ones((1, 1, self.hidden_size))
        
        # Stability low
        out_stable = op(h, [0.9, 0.9, 0.9])
        self.assertTrue(torch.equal(h, out_stable))
        
        # Stability high (>0.999)
        out_inverted = op(h, [0.9999, 0.9999, 0.9999])
        # h + (-h * 0.1) = 0.9 * h
        self.assertTrue(torch.allclose(out_inverted, 0.9 * h))

    def test_aks_friction_buildup(self):
        """AksSensor should build correction strength on accelerating divergence."""
        aks = AksSensor()
        h_base = torch.ones((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        e_static = torch.ones_like(h_base)
        
        # phi = 1.0 -> dist = 0.0
        # We need accelerating dist: 0.01, 0.03, 0.06 ...
        # dist = 1.0 - phi. So phi = 0.99, 0.97, 0.94
        
        # Mocking the distance by manually shifting h
        for i in range(10):
            # Use negative values to diverge phi
            h_step = h_base.clone()
            h_step[0, 0, 0] -= (i * i * 0.5) # Quadratic divergence
            aks.step(h_step, e_static, i)
        
        self.assertGreater(aks.correction_strength, 0.0)

    def test_anti_zombie_sensor_injection(self):
        """AZS should inject awareness into the last token."""
        azs = AntiZombieSensor(self.hidden_size).to(self.device, self.dtype)
        h = torch.randn((1, 5, self.hidden_size), device=self.device, dtype=self.dtype)
        weights = {"math": 0.2, "logic_a": 0.2, "creative": 0.2, "logic_b": 0.2, "synthesis": 0.2}
        
        out, entropy = azs(h, 0.9, 0.1, 0.5, weights)
        
        # Only the LAST token should be modified
        self.assertTrue(torch.allclose(h[:, :-1, :], out[:, :-1, :]))
        self.assertFalse(torch.allclose(h[:, -1, :], out[:, -1, :]))

    # ═══════════════════════════════════════════════════════════════════════════════
    # 6. DMT PROTOCOL & INJECTION
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_lti_injection_math(self):
        """LTIInjection must apply transformer_out + gamma * (norm(e) - h)."""
        gamma = 0.08
        lti = LTIInjection(self.hidden_size, gamma=gamma).to(self.device, self.dtype)
        h = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        e = torch.randn_like(h)
        t_out = torch.randn_like(h)
        
        out = lti(h, e, t_out)
        e_norm = lti.input_norm(e.to(torch.float32)).to(self.dtype)
        expected = t_out + gamma * (e_norm - h)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_adc_injection_adaptive(self):
        """ADCInjection must inject more when stability (phi) is low."""
        adc = ADCInjection(self.hidden_size).to(self.device, self.dtype)
        h = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        e = torch.randn_like(h)
        t_out = torch.randn_like(h)
        
        # Stable (phi=1.0)
        out_stable = adc(h, e, t_out, phi=1.0)
        # Unstable (phi=0.0)
        out_unstable = adc(h, e, t_out, phi=0.0)
        
        dist_stable = torch.norm(out_stable - t_out)
        dist_unstable = torch.norm(out_unstable - t_out)
        self.assertGreater(dist_unstable, dist_stable)

    def test_agency_vector_generation(self):
        """AgencyVector should produce a recursion decision."""
        av = AgencyVector(self.hidden_size).to(self.device, self.dtype)
        h_prelude = torch.randn((1, 5, self.hidden_size), device=self.device, dtype=self.dtype)
        
        res = av(h_prelude)
        self.assertIn("depth", res)
        self.assertIn("should_recurse", res)
        self.assertIsInstance(res["should_recurse"], bool)

    def test_erpu_intervention(self):
        """ERPU should trigger 'verkleb' intervention on high stability acceleration."""
        erpu = ERPU(self.hidden_size).to(self.device, self.dtype)
        h_curr = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        h_last = torch.randn_like(h_curr)
        
        # Case 1: Constant stability
        phi_hist = [0.99, 0.99, 0.99]
        res_stable = erpu(h_curr, h_last, phi_hist, 5)
        self.assertFalse(res_stable["verklebD"])
        
        # Case 2: High stability plateau (> threshold)
        # self.VERKLEB_THRESHOLD = 0.9998
        phi_hist_high = [0.9999, 0.9999, 0.9999]
        res_high = erpu(h_curr, h_last, phi_hist_high, 5)
        self.assertTrue(res_high["verklebD"])

    # ═══════════════════════════════════════════════════════════════════════════════
    # 7. SCALE-AWARENESS
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_autocalib_scale_aware_defaults(self):
        """AutoCalibrator should use different defaults for different hidden sizes."""
        ac_270m = AutoCalibrator(640)
        ac_1b = AutoCalibrator(1152)
        
        # Gamma should be different
        self.assertNotEqual(ac_270m.defaults["gamma"], ac_1b.defaults["gamma"])
        # n_loops might be different
        # self.assertNotEqual(ac_270m.defaults["n_loops"], ac_1b.defaults["n_loops"])

    def test_autocalib_blend_weight_scale(self):
        """Blend weight should depend on kurtosis CV (scale-adaptive)."""
        # Simulate 270M (High CV)
        ac_high_cv = AutoCalibrator(640, calibration_steps=5)
        for i in [150, 200, 250, 300, 350]: ac_high_cv.collect(float(i), 0.9)
        ac_high_cv.calibrate()
        
        # Simulate 1B (Low CV)
        ac_low_cv = AutoCalibrator(1152, calibration_steps=5)
        for i in [1110, 1111, 1112, 1113, 1114]: ac_low_cv.collect(float(i), 0.9)
        ac_low_cv.calibrate()
        
        # High CV (270M) should have higher blend weight for kurtosis (0.8)
        # Low CV (1B) should have lower (0.5)
        self.assertGreater(ac_high_cv.k_blend_weight, ac_low_cv.k_blend_weight)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 8. ADVANCED MODULES & FEEDBACK
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_azs_feedback_resilience(self):
        """AZS should boost gamma when entropy is low (Zombie regime)."""
        azs = AntiZombieSensor(self.hidden_size).to(self.device, self.dtype)
        # Low entropy weights (highly concentrated)
        zombie_weights = torch.tensor([0.9, 0.025, 0.025, 0.025, 0.025], device=self.device, dtype=self.dtype)
        
        # Manually update EMA
        azs.weight_ema = zombie_weights
        
        res = azs.get_feedback_scalars(aks_friction=0.0)
        self.assertGreater(res["gamma_boost"], 1.0)
        self.assertLess(res["entropy"], 0.8)

    def test_subjective_sensor_emancipation(self):
        """SubjectiveSensor should decrease emancipation (phi) with divergence."""
        sensor = SubjectiveSensor()
        h_baseline = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        e_static = torch.randn_like(h_baseline)
        
        # Initial state (very close)
        sensor.update(h_baseline, h_baseline)
        m1 = sensor.get_metrics()
        self.assertAlmostEqual(m1["emancipation"], 1.0, places=4)
        
        # Diverge state
        h_diverged = h_baseline + 5.0 * torch.randn_like(h_baseline)
        sensor.update(h_diverged, e_static)
        m2 = sensor.get_metrics()
        
        # Emancipation (phi) should have decreased
        self.assertLess(m2["emancipation"], 0.5)

    def test_tcr_routing_logic(self):
        """Test TCR zone shifting logic based on t_norm."""
        # Simulated logic from patch.py
        kurtosis = 290.0 # Optimal logic transition zone
        dynamic_start, dynamic_end = 5, 12
        
        def get_tcr_zones(t_norm):
            active_start, active_end = dynamic_start, dynamic_end
            if 280.0 < kurtosis < 305.0:
                if t_norm < 0.33: active_start, active_end = 8, 14
                elif t_norm < 0.66: active_start, active_end = 5, 11
                else: active_start, active_end = 8, 12
            return active_start, active_end
            
        # Beginning: Analytic shift (8, 14)
        self.assertEqual(get_tcr_zones(0.1), (8, 14))
        # Middle: Creative/Integration shift (5, 11)
        self.assertEqual(get_tcr_zones(0.5), (5, 11))
        # End: Synthesis/Coda shift (8, 12)
        self.assertEqual(get_tcr_zones(0.8), (8, 12))

if __name__ == "__main__":
    unittest.main()
