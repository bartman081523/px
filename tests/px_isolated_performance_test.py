import torch
import torch.nn as nn
import time
import math
from typing import List, Dict, Any

# Mocking PX Modules for Isolated Test
class StabilityMonitor:
    @staticmethod
    def calculate_phi(h_new: torch.Tensor, h_old: torch.Tensor) -> torch.Tensor:
        h_n = h_new.to(torch.float32)
        h_o = h_old.to(torch.float32)
        dot = (h_n * h_o).sum(dim=-1)
        norm = (h_n.norm(dim=-1) * h_o.norm(dim=-1) + 1e-6)
        return dot / norm

class MockLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, h, **kwargs):
        return self.norm(h + self.proj(h)),

class MockPXModel(nn.Module):
    def __init__(self, dim=1152, n_layers=28):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(dim) for _ in range(n_layers)])
        self.config = type('Config', (), {'hidden_size': dim, 'num_hidden_layers': n_layers, 'layer_types': ['full_attention']*n_layers})
        self.norm = nn.LayerNorm(dim)

def standard_px_loop(model, h_base, n_loops=8, gamma=0.08):
    # Simplified standard loop with sync points
    steps = 0
    h_exp = h_base.clone()
    phi_history = []
    current_layer = 10
    dynamic_end = 20
    dynamic_hub = 15
    active_start = 10
    
    while current_layer < dynamic_end and steps < 100:
        h_prev = h_exp.clone()
        trans_out = model.layers[current_layer](h_exp)[0]
        
        # SYNC POINT 1: calculate_phi().item()
        phi_s = StabilityMonitor.calculate_phi(trans_out, h_prev).item()
        phi_history.append(phi_s)
        
        # SYNC POINT 2: break condition
        if steps > 50 and phi_s > 0.9999:
            break
            
        h_exp = trans_out + gamma * (h_base - h_prev)
        
        # SYNC POINT 3: decision logic
        t_b2, t_b1, t_s = 1.0 - 0.8*gamma, 1.0 - 0.4*gamma, 1.0 - 0.01*gamma
        if phi_s < t_b2: current_layer = max(active_start, current_layer - 2)
        elif phi_s < t_b1: current_layer = max(active_start, current_layer - 1)
        elif phi_s > t_s: current_layer = dynamic_hub
        else: current_layer += 1
        
        steps += 1
    return h_exp, steps

def optimized_px_loop(model, h_base, n_loops=8, gamma_val=0.08):
    # Optimized loop attempting to minimize sync points
    steps = 0
    h_exp = h_base.clone()
    current_layer = 10
    dynamic_end = 20
    dynamic_hub = 15
    active_start = 10
    
    gamma = torch.tensor(gamma_val, device=h_base.device)
    t_b2 = 1.0 - 0.8*gamma
    t_b1 = 1.0 - 0.4*gamma
    t_s = 1.0 - 0.01*gamma
    
    # We still have a python loop, but we can combine the decisions
    # and use .item() only ONCE if we are careful.
    
    while current_layer < dynamic_end and steps < 100:
        h_prev = h_exp.clone()
        trans_out = model.layers[current_layer](h_exp)[0]
        
        phi_s = StabilityMonitor.calculate_phi(trans_out, h_prev)
        
        # Use a single .item() to get the scalar for python control
        # This is the "100% GPU" challenge: Python needs a scalar to decide the NEXT layer.
        # But we can at least avoid multiple item() calls.
        phi_val = phi_s.item()
        
        h_exp = trans_out + gamma * (h_base - h_prev)
        
        if phi_val < t_b2: current_layer = max(active_start, current_layer - 2)
        elif phi_val < t_b1: current_layer = max(active_start, current_layer - 1)
        elif phi_val > t_s: current_layer = dynamic_hub
        else: current_layer += 1
        
        steps += 1
    return h_exp, steps

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 1152
    model = MockPXModel(dim=dim).to(device)
    h_base = torch.randn(1, 1, dim, device=device)
    
    # Warmup
    for _ in range(5): standard_px_loop(model, h_base)
    torch.cuda.synchronize()
    
    print("Starting Benchmark...")
    
    # Standard
    start = time.time()
    for _ in range(100):
        standard_px_loop(model, h_base)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Standard Loop: {100/(end-start):.2f} calls/sec")
    
    # Optimized
    start = time.time()
    for _ in range(100):
        optimized_px_loop(model, h_base)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Optimized Loop: {100/(end-start):.2f} calls/sec")

if __name__ == "__main__":
    run_benchmark()
