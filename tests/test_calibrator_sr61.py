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

class TestSelfOrganizingCalibrator(unittest.TestCase):
    def test_learns_centroids(self):
        cal = AutoCalibrator(640, calibration_steps=10)
        
        # 1. Provide distinct clusters of kurtosis
        # Cluster A: ~200 (synthesis)
        # Cluster B: ~300 (creative)
        # Cluster C: ~400 (logic_b)
        # Cluster D: ~500 (logic_a)
        # Cluster E: ~600 (math)
        samples = [200, 205, 300, 305, 400, 405, 500, 505, 600, 605]
        
        for s in samples:
            cal.collect(s, 0.9, 0.8)
            
        self.assertTrue(cal.calibrated)
        self.assertGreater(len(cal.learned_centroids), 0)
        
        # Verify centroid ordering (synthesis is lowest, math is highest)
        centroids = sorted(cal.learned_centroids.values())
        print(f"Learned centroids: {cal.learned_centroids}")
        
        self.assertAlmostEqual(centroids[0], 202.5, delta=5.0)
        self.assertAlmostEqual(centroids[-1], 602.5, delta=5.0)
        
    def test_weights_diverge_after_learning(self):
        cal = AutoCalibrator(640, calibration_steps=10)
        samples = [200, 210, 300, 310, 400, 410, 500, 510, 600, 610]
        for s in samples: cal.collect(s, 0.9, 0.8)
        
        # Weights for a clear outlier
        w_low = cal.get_zone_weights(205.0)
        w_high = cal.get_zone_weights(605.0)
        
        # In low K, synthesis should dominate
        self.assertGreater(w_low['synthesis'], 0.5)
        # In high K, math should dominate
        self.assertGreater(w_high['math'], 0.5)
        
        print(f"Weights low (205): {w_low}")
        print(f"Weights high (605): {w_high}")

if __name__ == "__main__":
    unittest.main()
