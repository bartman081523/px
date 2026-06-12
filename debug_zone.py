import sys
sys.path.append("/run/media/julian/ML4/ollama-work/all_space")
from px_patches.gemma3_270m_px_baseline.auto_tune import AutoCalibrator

calibrator = AutoCalibrator(hidden_size=1536)
kurtosis = 149.0
phi = 0.4332

weights = calibrator.get_zone_weights(kurtosis, phi)
print("Weights:", weights)
zone = calibrator.classify_zone(kurtosis, phi)
print("Classified Zone:", zone)

rp = calibrator.get_routing_params(kurtosis, phi, hidden_size=1536)
print("Routing Params:", rp)
