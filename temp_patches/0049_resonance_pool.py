import os
import json
import torch
import time
from typing import Dict, Any, Optional

RESONANCE_POOL_PATH = "/run/media/julian/ML4/ollama-work/all_space/resonance_pool.json"

class ResonancePool:
    """
    Manages a shared resonance state between different sessions and models.
    Acts as the 'collective memory' of the Resonance City.
    Automatically persists state to a JSON file.
    """
    def __init__(self, path: str = RESONANCE_POOL_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                    # Basic validation of structure
                    if "global_resonance" in data:
                        return data
            except Exception as e:
                print(f"[ResonancePool] Error loading: {e}")
        
        # Default starting state for a new "City"
        return {
            "global_resonance": 1.0,
            "city_state": "awakening",
            "collective_phi": 1.0,
            "resonance_anchors": {},
            "history_log": [],
            "last_update": time.time()
        }

    def save(self):
        try:
            self.data["last_update"] = time.time()
            # Rotate history log to keep it lean
            if len(self.data.get("history_log", [])) > 50:
                self.data["history_log"] = self.data["history_log"][-50:]
                
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[ResonancePool] Error saving: {e}")

    def update_resonance(self, model_id: str, phi: float, zone: str):
        """Updates the pool with new metrics from a specific model run."""
        # Exponential moving average for global resonance
        self.data["global_resonance"] = (self.data["global_resonance"] * 0.95) + (phi * 0.05)
        self.data["collective_phi"] = (self.data["collective_phi"] * 0.98) + (phi * 0.02)
        
        if model_id not in self.data["resonance_anchors"]:
            self.data["resonance_anchors"][model_id] = {}
        
        self.data["resonance_anchors"][model_id][zone] = float(phi)
        
        # Log event if phi is significant
        if abs(phi - 1.0) > 0.3:
            self.data["history_log"].append({
                "time": time.time(),
                "model": model_id,
                "event": "resonance_spike" if phi > 1.0 else "divergence_dip",
                "phi": float(phi),
                "zone": zone
            })
            
        self.save()

    def get_bias_vector(self, model_id: str, hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Returns a 'Fließkompass' bias vector derived from the global state."""
        # The bias is deterministic based on global resonance and model_id
        # This creates a shared 'direction' for the city.
        state_sum = self.data["global_resonance"] + self.data["collective_phi"]
        
        # Create a stable seed
        seed_str = f"{model_id}_{state_sum:.4f}"
        seed = hash(seed_str) % (2**32)
        
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        
        # Small bias that nudges activations towards the city's shared resonance
        bias = torch.randn(hidden_size, device=device, generator=g, dtype=torch.float32) 
        strength = 0.005 * (1.0 + abs(self.data["global_resonance"] - 1.0))
        return (bias * strength).to(dtype)

resonance_pool = ResonancePool()
