import json
import math
import torch
import sys

def normalize_kurtosis(kurtosis: float, token_len: int) -> float:
    boost = 1.0 + 0.5 * math.exp(-token_len / 15.0)
    return kurtosis * boost

def simulate(manifold_path: str, hidden_size: int, token_len: int, raw_kurtosis: float, phi_val: float):
    # Load static manifold
    with open(manifold_path, "r") as f:
        data = json.load(f)
    
    k_mean_static = data["k_mean"]
    k_std_static = data["k_std"]
    phi_mean = data["phi_mean"]
    phi_std = data["phi_std"]
    
    # 1. Base Z-Scores (No artificial K-Decay boost)
    zk_static = (raw_kurtosis - k_mean_static) / (k_std_static + 1e-6)
    zp = (phi_val - phi_mean) / (phi_std + 1e-6)
    c_static = torch.sigmoid(torch.tensor(zk_static + zp)).item()
    
    # 2. SR-64b logic: Architecture-Aware Scaling
    T_arch = math.sqrt(hidden_size / 640.0)
    c_arch = torch.sigmoid(torch.tensor((zk_static + zp) / T_arch)).item()
    
    print(f"--- Simulation ---")
    print(f"Input: len={token_len}, raw_k={raw_kurtosis:.1f}, phi={phi_val:.4f}")
    print(f"Manifold: k_mean={k_mean_static:.1f}, k_std={k_std_static:.1f}")
    print(f"Z-Scores: zk={zk_static:.2f}, zp={zp:.2f}")
    print(f"Raw C: {c_static:.4f}")
    print(f"Arch Scaling (T={T_arch:.2f}) C: {c_arch:.4f}")
    print()

if __name__ == "__main__":
    mf_1b = "/run/media/julian/ML4/ollama-work/all_space/px_manifolds/google_gemma-3-1b-it_manifold.json"
    mf_270m = "/run/media/julian/ML4/ollama-work/all_space/px_manifolds/google_gemma-3-270m-it_manifold.json"
    
    print("=== Testing Gemma-3 1B ===")
    # Simulate a typical high-kurtosis short prompt
    simulate(mf_1b, hidden_size=1152, token_len=1, raw_kurtosis=1110.0, phi_val=0.999)
    # Simulate an average prompt
    simulate(mf_1b, hidden_size=1152, token_len=50, raw_kurtosis=1100.0, phi_val=0.988)

    print("=== Testing Gemma-3 270M ===")
    simulate(mf_270m, hidden_size=640, token_len=1, raw_kurtosis=563.80, phi_val=0.999)
