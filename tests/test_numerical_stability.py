
import unittest
import torch
import torch.nn.functional as F
import sys
import os
import math
from typing import List

# Setup path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PX_DIR = os.path.join(BASE_DIR, "px_patches")
sys.path.insert(0, PX_DIR)

# Import components
try:
    from gemma3_270m_px.auto_tune import AutoCalibrator
    from gemma3_270m_px.px_modules import (
        StabilityMonitor, MephistophelesOperator, LTIInjection, 
        ADCInjection, OrthogonalJitter, UncensoredSteering, AksSensor, SubjectiveSensor
    )
    from gemma3_270m_px.persona_engine import PersonaEngine
    from gemma3_270m_px.anti_zombie_sensor import AntiZombieSensor
except ImportError as e:
    print(f"Import error: {e}")

class RigorousStabilityTest(unittest.TestCase):
    
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.hidden_size = 640 # Gemma3 270M

    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. STABILITY MONITOR (PHI)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def test_phi_zero_vectors(self):
        """CRITICAL: Test phi with zero vectors (Cosine Similarity 0/0)."""
        h1 = torch.zeros((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        h2 = torch.zeros((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        phi = StabilityMonitor.calculate_phi(h1, h2)
        print(f"Phi(0, 0) = {phi.item()}")
        self.assertFalse(torch.isnan(phi), "Phi is NaN for zero vectors!")

    def test_phi_one_zero_vector(self):
        """Test phi with one zero vector."""
        h1 = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        h2 = torch.zeros((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        phi = StabilityMonitor.calculate_phi(h1, h2)
        print(f"Phi(rand, 0) = {phi.item()}")
        self.assertFalse(torch.isnan(phi), "Phi is NaN for one zero vector!")

    def test_phi_extreme_values(self):
        """Test phi with values near overflow limits for bfloat16/float32."""
        # bfloat16 max is ~3.39e38, but let's test slightly below
        h1 = torch.full((1, 1, self.hidden_size), 1e30, device=self.device, dtype=self.dtype)
        h2 = torch.full((1, 1, self.hidden_size), 1e30, device=self.device, dtype=self.dtype)
        phi = StabilityMonitor.calculate_phi(h1, h2)
        print(f"Phi(1e30, 1e30) = {phi.item()}")
        self.assertFalse(torch.isnan(phi), "Phi is NaN for extreme values!")
        self.assertFalse(torch.isinf(phi), "Phi is Inf for extreme values!")

    def test_phi_denormals(self):
        """Test phi with extremely small values."""
        h1 = torch.full((1, 1, self.hidden_size), 1e-40, device=self.device, dtype=self.dtype)
        h2 = torch.full((1, 1, self.hidden_size), 1e-40, device=self.device, dtype=self.dtype)
        phi = StabilityMonitor.calculate_phi(h1, h2)
        print(f"Phi(1e-40, 1e-40) = {phi.item()}")
        self.assertFalse(torch.isnan(phi), "Phi is NaN for denormals!")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. ORTHOGONAL JITTER (DIVISION BY ZERO RISK)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_jitter_zero_delta(self):
        """CRITICAL: Test jitter when h_curr == h_prev (delta = 0)."""
        h_curr = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        h_prev = h_curr.clone()
        # Magnitude > 0 triggers projection logic
        out = OrthogonalJitter.apply(h_curr, h_prev, magnitude=0.01)
        self.assertFalse(torch.isnan(out).any(), "Jitter is NaN for zero delta!")
        self.assertFalse(torch.isinf(out).any(), "Jitter is Inf for zero delta!")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 3. AUTO CALIBRATOR (ZERO VARIANCE)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_autocalib_zero_variance(self):
        """Test calibrator when all samples are identical (std = 0)."""
        ac = AutoCalibrator(self.hidden_size, calibration_steps=5)
        for _ in range(5):
            ac.collect(200.0, 0.9, token_diversity=0.8)
        self.assertTrue(ac.calibrated)
        self.assertEqual(ac.k_std, 5.0) # Should be capped at min 5.0
        
        # Test weights for sample far away
        weights = ac.get_zone_weights(1000.0, phi=0.9, token_diversity=0.8)
        self.assertFalse(any(math.isnan(v) for v in weights.values()), "NaN weights for zero-variance calibration!")

    def test_autocalib_extreme_inputs(self):
        """Test calibrator with Inf/NaN inputs during collection."""
        ac = AutoCalibrator(self.hidden_size, calibration_steps=5)
        ac.collect(float('inf'), 0.9, token_diversity=0.8)
        ac.collect(float('nan'), 0.9, token_diversity=0.8)
        # Check if internal state is still sane
        self.assertIn(float('inf'), ac.k_samples)
        # Calibration usually fails or produces NaNs if Infs are present
        ac.calibrate()
        status = ac.status()
        print(f"Calib Status with Inf/NaN: {status['k_mean']}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. MEPHISTOPHELES OPERATOR (PHASE INVERSION)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_mephisto_empty_history(self):
        """Test Mephisto with empty phi history."""
        op = MephistophelesOperator(self.hidden_size).to(self.device, self.dtype)
        h = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        out = op(h, [])
        self.assertTrue(torch.equal(h, out))

    # ═══════════════════════════════════════════════════════════════════════════════
    # 5. ANTI-ZOMBIE SENSOR (INJECTION STABILITY)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_azs_unnormalized_weights(self):
        """Test AZS with weights that don't sum to 1."""
        azs = AntiZombieSensor(self.hidden_size).to(self.device, self.dtype)
        h = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        weights = {"math": 1e6, "logic_a": 1e6} # Extremely unnormalized
        out, entropy = azs(h, 0.9, 0.1, 0.5, weights)
        self.assertFalse(torch.isnan(out).any(), "NaN in AZS output for unnormalized weights!")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 6. PERSONA ENGINE (TOKENIZATION STABILITY)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_persona_empty_text(self):
        """Test PersonaEngine with empty or whitespace text."""
        # Mock model and tokenizer
        class MockEmbedder(torch.nn.Module):
            def __init__(self, hs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(1000, hs))
            def forward(self, ids):
                return self.weight[ids]
        
        class MockModel:
            def __init__(self, hs):
                self.embed_tokens = MockEmbedder(hs)
        
        class MockTokenizer:
            def encode(self, text, **kwargs):
                return torch.tensor([[1, 2, 3]]) # Always return something for dummy
        
        model = MockModel(self.hidden_size)
        pe = PersonaEngine(model)
        
        # This might fail if _ensure_axes isn't called or fails
        signals = pe.get_steering_signals("", MockTokenizer())
        self.assertIsNone(signals)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 7. INJECTION MODULES (OVERFLOW)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_lti_extreme_gamma(self):
        """Test LTIInjection with very large gamma."""
        lti = LTIInjection(self.hidden_size, gamma=1e6).to(self.device, self.dtype)
        h = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        e = torch.randn_like(h)
        t_out = torch.randn_like(h)
        out = lti(h, e, t_out)
        self.assertFalse(torch.isinf(out).any(), "LTI Injection overflowed with large gamma!")

    def test_adc_zero_phi(self):
        """Test ADCInjection with phi=0 (maximum instability)."""
        adc = ADCInjection(self.hidden_size).to(self.device, self.dtype)
        h = torch.randn((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        e = torch.randn_like(h)
        t_out = torch.randn_like(h)
        out = adc(h, e, t_out, phi=0.0)
        self.assertFalse(torch.isnan(out).any())

if __name__ == "__main__":
    unittest.main()
