"""
auto_tune.py — SR-61b: 2D Manifold Persistence
=============================================================================
SR-61b Innovation: Persistent cognitive manifolds.

This version implements:
1. 2D Hybrid Routing: Centroids in (Kurtosis, Phi) space.
2. Manifold Persistence: Save/Load learned centroids to JSON.
3. Scale-Adaptive Temperature: Sharpens zones based on model size.

Centroid targets in Z-space (z_k, z_p):
  - math:      (1.5,  0.5)  -> High kurtosis, high stability
  - logic_a:   (0.5,  0.2)  -> Above mean
  - logic_b:   (0.0,  0.0)  -> Average
  - creative:  (-1.0, -0.5) -> Below mean
  - synthesis: (-1.5, -1.0) -> Very flat
"""

import math
import statistics
import json
import os
from typing import Dict, Optional, Tuple


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _dist2d(p1: Tuple[float, float], p2: Tuple[float, float], std_k: float, std_p: float) -> float:
    """Normalized 2D Euclidean distance."""
    return math.sqrt(((p1[0] - p2[0]) / std_k)**2 + ((p1[1] - p2[1]) / std_p)**2)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

SCALE_DEFAULTS = {
    640:   dict(recur_start=5,  recur_end=12, hub=10, n_loops=8, gamma=0.08),
    1152:  dict(recur_start=10, recur_end=20, hub=18, n_loops=8, gamma=0.12),
    # Gemma-4 E2B (hidden_size=1536, 35 layers) — 1B parity target (2026-06-09):
    # same n_loops=8 / gamma=0.12 as 1B (1152), with a wider recursion window
    # (recur_end=26) for the deeper 35-layer stack. Restored after e7f2942
    # accidentally removed it; tests in test_gemma4_e2b_mock.py lock these values.
    1536:  dict(recur_start=10, recur_end=26, hub=18, n_loops=8, gamma=0.12),
    2560:  dict(recur_start=8,  recur_end=22, hub=16, n_loops=6, gamma=0.05),
    4096:  dict(recur_start=10, recur_end=30, hub=20, n_loops=6, gamma=0.04),
}

ZONE_ROUTING = {
    'math':      dict(start=5,  end=11, hub=10, loops=8),
    'logic_a':   dict(start=8,  end=12, hub=10, loops=8),
    'creative':  dict(start=10, end=16, hub=10, loops=6),
    'logic_b':   dict(start=8,  end=14, hub=10, loops=10),
    'synthesis': dict(start=6,  end=14, hub=10, loops=8),
}

# 2D Centroid targets in Z-space (z_kurtosis, z_phi)
ZONE_Z_TARGETS = {
    'math':      (1.5,  0.5),
    'logic_a':   (0.5,  0.2),
    'logic_b':   (0.0,  0.0),
    'creative':  (-1.0, -0.5),
    'synthesis': (-1.5, -1.0),
}

ZONE_Z_SIGMAS = {
    'math':      0.8,
    'logic_a':   0.6,
    'creative':  1.0,
    'logic_b':   0.7,
    'synthesis': 0.9,
}

ONLINE_WARMUP = 5
MIN_TD_STD = 0.10
MIN_ONLINE_K_STD = 1.0


class AutoCalibrator:
    """Adaptive 2D Routing with Manifold Persistence. (SR-61b)"""

    def __init__(self, hidden_size: int, calibration_steps: int = 10, model_id: Optional[str] = None):
        self.hidden_size = hidden_size
        self.calibration_steps = calibration_steps
        self.model_id = model_id

        # Persistent state directory
        self.manifold_dir = "/run/media/julian/ML4/ollama-work/all_space/px_manifolds"
        
        self.calibrated = False
        self.k_samples = []
        self.phi_samples = []
        self.token_diversity_samples = []

        # State
        self.k_mean = None
        self.k_std = None
        self.phi_mean = None
        self.phi_std = None
        self.token_diversity_mean = None
        self.token_diversity_std = None
        self.k_blend_weight = 0.8 if hidden_size == 640 else 0.5
        self.zone_temperature = 0.8
        self.learned_centroids: Dict[str, Tuple[float, float]] = {}

        # Online stats (Welford)
        self._online_n = 0
        self._online_k_mean = 0.0
        self._online_k_m2 = 0.0

        # Try to load existing manifold
        if model_id:
            self.load_manifold()

    def collect(self, kurtosis: float, phi: float, token_diversity: Optional[float] = None,
            update_online: bool = False, token_len: int = 1):
        # SR-64b uses raw kurtosis
        k_norm = kurtosis

        if not self.calibrated:
            self.k_samples.append(k_norm)
            self.phi_samples.append(phi)
            if token_diversity is not None:
                self.token_diversity_samples.append(token_diversity)

            if len(self.k_samples) >= self.calibration_steps:
                self.calibrate()
                return True
            return False

        if update_online:
            self._update_online_stats(k_norm)
        return False

    def _update_online_stats(self, kurtosis: float):
        self._online_n += 1
        delta = kurtosis - self._online_k_mean
        self._online_k_mean += delta / self._online_n
        delta2 = kurtosis - self._online_k_mean
        self._online_k_m2 += delta * delta2

    def calibrate(self):
        """Compute 2D zone centroids and persist. (SR-61b)"""
        k_samples = [k for k in self.k_samples if math.isfinite(k)]
        phi_samples = [p for p in self.phi_samples if math.isfinite(p)]
        td_samples = [t for t in self.token_diversity_samples if math.isfinite(t)]

        if len(k_samples) < 2: return

        self.k_mean = statistics.mean(k_samples)
        self.k_std = max(statistics.stdev(k_samples), 5.0)
        self.phi_mean = statistics.mean(phi_samples) if len(phi_samples) >= 2 else 0.9
        self.phi_std = max(statistics.stdev(phi_samples), 0.01) if len(phi_samples) >= 2 else 0.05
        
        # Learn 2D centroids
        self.learned_centroids = {}
        for zone, (zk, zp) in ZONE_Z_TARGETS.items():
            raw_k = self.k_mean + zk * self.k_std
            raw_p = self.phi_mean + zp * self.phi_std
            self.learned_centroids[zone] = (raw_k, raw_p)

        # Token diversity stats
        if td_samples:
            self.token_diversity_mean = statistics.mean(td_samples)
            self.token_diversity_std = max(statistics.stdev(td_samples) if len(td_samples) > 1 else 0.0, MIN_TD_STD)

        # Scale-adaptive temperature (SR-59h)
        k_cv = self.k_std / (abs(self.k_mean) + 1e-9)
        if k_cv > 0.05: self.zone_temperature = 0.3
        elif k_cv > 0.01: self.zone_temperature = 0.6
        else: self.zone_temperature = 1.0

        # Initialize online stats
        self._online_n = len(k_samples)
        self._online_k_mean = self.k_mean
        self._online_k_m2 = (self.k_std ** 2) * self._online_n

        self.calibrated = True
        self.save_manifold()

    def _get_kurtosis_weights(self, kurtosis: float, phi: float) -> Dict[str, float]:
        """Compute 2D zone weights using Euclidean distance in manifold space."""
        if not self.learned_centroids:
            return {z: 1.0/len(ZONE_Z_TARGETS) for z in ZONE_Z_TARGETS}
            
        # Use online mean for kurtosis z-score if available
        k_center = self._online_k_mean if self._online_n >= ONLINE_WARMUP else self.k_mean
        k_std_eff = math.sqrt(self._online_k_m2 / max(self._online_n - 1, 1)) if self._online_n > 1 else self.k_std
        k_std_eff = max(k_std_eff, 1.0)

        weights = {}
        temp = self.zone_temperature
        for zone, (ck, cp) in self.learned_centroids.items():
            sigma = ZONE_Z_SIGMAS[zone] * temp
            # Calculate 2D distance normalized by local manifold density
            dist = _dist2d((kurtosis, phi), (ck, cp), k_std_eff, self.phi_std)
            weights[zone] = math.exp(-0.5 * (dist / sigma)**2)

        W = sum(weights.values()) + 1e-9
        return {k: v / W for k, v in weights.items()}

    def _compute_scf_weights(self, token_diversity: float) -> Optional[Dict[str, float]]:
        if self.token_diversity_mean is None or token_diversity is None:
            return None
        z_td = (token_diversity - self.token_diversity_mean) / (self.token_diversity_std + 1e-9)
        phi_signal = _sigmoid(-z_td)
        weights = {
            'math':      phi_signal ** 2,
            'logic_a':   phi_signal * (1 - phi_signal) * 2,
            'creative':  (1 - phi_signal) ** 2,
            'logic_b':   phi_signal * (1 - phi_signal),
            'synthesis': (1 - phi_signal) * phi_signal * 0.5,
        }
        W = sum(weights.values()) + 1e-9
        return {k: v / W for k, v in weights.items()}

    def get_zone_weights(self, kurtosis: float, phi: Optional[float] = None, token_diversity: Optional[float] = None, token_len: int = 1) -> Dict[str, float]:
        # SR-64b uses raw kurtosis
        k_norm = kurtosis

        # SR-61b: Fallback for None phi (first token of session)
        if phi is None:
            phi = self.phi_mean if self.phi_mean is not None else 0.9
            
        k_weights = self._get_kurtosis_weights(k_norm, phi)
        scf_weights = self._compute_scf_weights(token_diversity)
        if scf_weights is None: return k_weights
        blend = self.k_blend_weight
        blended = {z: blend * k_weights[z] + (1 - blend) * scf_weights[z] for z in k_weights}
        W = sum(blended.values()) + 1e-9
        return {k: v / W for k, v in blended.items()}

    def classify_zone(self, kurtosis: float, phi: Optional[float] = None, token_diversity: Optional[float] = None, token_len: int = 1) -> str:
        weights = self.get_zone_weights(kurtosis, phi, token_diversity, token_len=token_len)
        return max(weights, key=weights.get)

    def get_routing_params(self, kurtosis: float, phi: Optional[float] = None, hidden_size: Optional[int] = None, token_diversity: Optional[float] = None, token_len: int = 1) -> Dict[str, any]:
        weights = self.get_zone_weights(kurtosis, phi, token_diversity, token_len=token_len)
        start = sum(weights[z] * ZONE_ROUTING[z]['start'] for z in weights)
        end = sum(weights[z] * ZONE_ROUTING[z]['end'] for z in weights)
        hub = sum(weights[z] * ZONE_ROUTING[z]['hub'] for z in weights)
        loops = sum(weights[z] * ZONE_ROUTING[z]['loops'] for z in weights)

        # Scale adjustments
        if hidden_size and hidden_size in SCALE_DEFAULTS:
            defaults = SCALE_DEFAULTS[hidden_size]
            b = 0.3
            start = b * defaults['recur_start'] + (1-b) * start
            end = b * defaults['recur_end'] + (1-b) * end
            hub = b * defaults['hub'] + (1-b) * hub
            loops = b * defaults['n_loops'] + (1-b) * loops

        return {
            'dynamic_start': max(1, int(round(start))),
            'dynamic_end': max(int(round(start))+2, int(round(end))),
            'dynamic_hub': max(int(round(start)), min(int(round(end)), int(round(hub)))),
            'n_loops': max(1, int(round(loops))),
        }

    def save_manifold(self):
        if not self.model_id: return
        os.makedirs(self.manifold_dir, exist_ok=True)
        safe_id = self.model_id.replace("/", "_")
        path = os.path.join(self.manifold_dir, f"{safe_id}_manifold.json")
        data = {
            "k_mean": self.k_mean, "k_std": self.k_std,
            "phi_mean": self.phi_mean, "phi_std": self.phi_std,
            "learned_centroids": self.learned_centroids,
            "calibrated": self.calibrated,
            "k_blend_weight": self.k_blend_weight,
            "zone_temperature": self.zone_temperature,
            "token_diversity_mean": self.token_diversity_mean,
            "token_diversity_std": self.token_diversity_std
        }
        with open(path, "w") as f: json.dump(data, f, indent=2)

    def load_manifold(self):
        if not self.model_id: return
        safe_id = self.model_id.replace("/", "_")
        path = os.path.join(self.manifold_dir, f"{safe_id}_manifold.json")
        if not os.path.exists(path): return
        try:
            with open(path, "r") as f: data = json.load(f)
            self.k_mean = data.get("k_mean")
            self.k_std = data.get("k_std")
            self.phi_mean = data.get("phi_mean")
            self.phi_std = data.get("phi_std")
            self.learned_centroids = data.get("learned_centroids", {})
            self.calibrated = data.get("calibrated", False)
            self.k_blend_weight = data.get("k_blend_weight", self.k_blend_weight)
            self.zone_temperature = data.get("zone_temperature", self.zone_temperature)
            self.token_diversity_mean = data.get("token_diversity_mean")
            self.token_diversity_std = data.get("token_diversity_std")
            if self.calibrated:
                self._online_n = ONLINE_WARMUP + 1
                self._online_k_mean = self.k_mean
                self._online_k_m2 = (self.k_std ** 2) * self._online_n
            print(f"[AutoCalibrator] Loaded manifold for {self.model_id} (T={self.zone_temperature:.2f})")
        except: pass

    def status(self) -> Dict[str, any]:
        k_cv = self.k_std / (abs(self.k_mean) + 1e-9) if self.k_mean else None
        return {
            'calibrated': self.calibrated, 'model_id': self.model_id,
            'k_mean': self.k_mean, 'k_std': self.k_std, 'k_cv': k_cv,
            'phi_mean': self.phi_mean, 'phi_std': self.phi_std,
            'learned_centroids': self.learned_centroids,
            'zone_temperature': self.zone_temperature,
            'online_n': self._online_n,
            'routing_mode': 'online_2d' if self._online_n >= ONLINE_WARMUP else 'calibration_2d',
        }
