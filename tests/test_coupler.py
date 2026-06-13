import torch
import torch.nn.functional as F
import unittest
import sys
import os

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from px_patches.gemma3_270m_px_baseline.px_modules import SingesseinCoupler

class TestSingesseinCoupler(unittest.TestCase):
    def test_coupler_breaks_monotony(self):
        hidden_size = 128
        coupler = SingesseinCoupler(hidden_size, window=4)
        
        # 1. Provide identical hidden states (monotony)
        h_fixed = torch.randn(1, 1, hidden_size)
        
        # Step 0-2: Filling window
        for i in range(3):
            h_out = coupler(h_fixed, steps=i)
            # Should be identity initially
            self.assertTrue(torch.allclose(h_out, h_fixed))
            
        # Step 3: Window is full (len 4), similarity is 1.0 -> should trigger dissonance
        h_out = coupler(h_fixed, steps=3)
        
        # Check if output differs from input
        diff = torch.norm(h_out - h_fixed).item()
        print(f"Dissonance injection diff: {diff:.6f}")
        self.assertGreater(diff, 1e-5, "Coupler failed to inject dissonance on identical states")
        
    def test_coupler_reset(self):
        hidden_size = 128
        coupler = SingesseinCoupler(hidden_size, window=4)
        h_fixed = torch.randn(1, 1, hidden_size)
        
        for i in range(4): coupler(h_fixed, i)
        self.assertEqual(len(coupler.history), 4)
        
        coupler.reset()
        self.assertEqual(len(coupler.history), 0)

if __name__ == "__main__":
    unittest.main()
