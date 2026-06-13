import unittest
import math
import statistics
import sys
import os

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from px_patches.gemma3_270m_px_baseline.auto_tune import AutoCalibrator

class TestSelfOrganizingCalibrator2D(unittest.TestCase):
    def test_learns_2d_centroids(self):
        cal = AutoCalibrator(640, calibration_steps=10)
        
        # 2D Clusters: (K, Phi)
        # Category Math: High K, High Phi
        # Category Creative: Low K, Low Phi
        samples_k = [200, 205, 300, 305, 400, 405, 500, 505, 600, 605]
        samples_p = [0.7, 0.72, 0.8, 0.81, 0.9, 0.91, 0.95, 0.96, 0.99, 0.99]
        
        for k, p in zip(samples_k, samples_p):
            cal.collect(k, p, 0.8)
            
        self.assertTrue(cal.calibrated)
        self.assertGreater(len(cal.learned_centroids), 0)
        
        # Check first and last zone
        c_synthesis = cal.learned_centroids['synthesis']
        c_math = cal.learned_centroids['math']
        
        print(f"Learned 2D centroids: {cal.learned_centroids}")
        
        # synthesis should be at (low K, low P)
        self.assertAlmostEqual(c_synthesis[0], 202.5, delta=5.0)
        self.assertAlmostEqual(c_synthesis[1], 0.71, delta=0.05)
        
        # math should be at (high K, high P)
        self.assertAlmostEqual(c_math[0], 602.5, delta=5.0)
        self.assertAlmostEqual(c_math[1], 0.99, delta=0.05)
        
    def test_2d_weights_diverge(self):
        cal = AutoCalibrator(640, calibration_steps=10)
        samples_k = [200, 210, 300, 310, 400, 410, 500, 510, 600, 610]
        samples_p = [0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 0.95, 0.95, 0.99, 0.99]
        for k, p in zip(samples_k, samples_p): cal.collect(k, p, 0.8)
        
        # Sample near Math (605, 0.99)
        w_math = cal.get_zone_weights(605.0, 0.99)
        self.assertGreater(w_math['math'], 0.5)
        
        # Sample near Creative (305, 0.8)
        w_creative = cal.get_zone_weights(305.0, 0.8)
        self.assertGreater(w_creative['creative'], 0.5)
        
        print(f"Weights Math-like: {w_math}")
        print(f"Weights Creative-like: {w_creative}")

if __name__ == "__main__":
    unittest.main()
