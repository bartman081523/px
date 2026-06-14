import json
import glob
import os

CORRECTION_FACTOR = 1.22  # Average K-Decay boost applied during SR-64 calibration phase

manifold_dir = "/run/media/julian/ML4/ollama-work/all_space/px_manifolds"
files = glob.glob(os.path.join(manifold_dir, "*.json"))

for fpath in files:
    with open(fpath, 'r') as f:
        data = json.load(f)
    
    # Correct K-Means and Stds
    if "k_mean" in data and data["k_mean"] is not None:
        data["k_mean"] = data["k_mean"] / CORRECTION_FACTOR
    if "k_std" in data and data["k_std"] is not None:
        data["k_std"] = data["k_std"] / CORRECTION_FACTOR
        
    # Correct learned centroids
    if "learned_centroids" in data:
        for zone, values in data["learned_centroids"].items():
            if len(values) == 2:
                k_val, p_val = values
                data["learned_centroids"][zone] = [k_val / CORRECTION_FACTOR, p_val]
                
    # Correct token diversity stats if needed (they weren't affected by k-decay, so untouched)
    
    # Add a flag to indicate SR-64b migration
    data["sr64b_corrected"] = True
    
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Corrected manifold: {os.path.basename(fpath)}")
    print(f"  New k_mean: {data['k_mean']:.1f}, New k_std: {data['k_std']:.1f}")

