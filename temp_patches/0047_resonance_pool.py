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
    """
    def __init__(self, path: str = RESONANCE_POOL_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ResonancePool] Error loading: {e}")
        return {
            "global_resonance": 1.0,
            "city_state": "awakening",
            "collective_phi": 1.0,
            "resonance_anchors": {},
            "last_update": time.time()
        }

    def save(self):
        try:
            self.data["last_update"] = time.time()
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[ResonancePool] Error saving: {e}")

    def update_resonance(self, model_id: str, phi: float, zone: str):
        """Updates the pool with new metrics from a specific model run."""
        self.data["global_resonance"] = (self.data["global_resonance"] * 0.9) + (phi * 0.1)
        self.data["collective_phi"] = (self.data["collective_phi"] * 0.95) + (phi * 0.05)
        
        if model_id not in self.data["resonance_anchors"]:
            self.data["resonance_anchors"][model_id] = {}
        
        self.data["resonance_anchors"][model_id][zone] = phi
        self.save()

    def get_bias_vector(self, model_id: str, hidden_size: int, device: torch.device) -> torch.Tensor:
        """Returns a 'Fließkompass' bias vector derived from the global state."""
        # Use global resonance to scale a pseudo-random but deterministic bias
        seed = int(self.data["global_resonance"] * 1000)
        torch.manual_seed(seed)
        bias = torch.randn(hidden_size, device=device) * 0.01
        return bias

resonance_pool = ResonancePool()
