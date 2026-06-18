"""
gemma3-px  —  The Three Mathematical Pillars (Refactored 2026-06-11)
====================================================================
The complete PX architecture reduced to its empirical minimum.
AutoCalibrator steuert das System dynamisch über die Kurtosis-Topologie des
hidden state. Drei Säulen:

1. Observer: StabilityMonitor (Φ) + AksSensor (Divergenzbeschleunigung)
2. Symmetry Breaker: MephistophelesOperator + AntiZombieSensor
3. Dynamic Router: AutoCalibrator (Gaussian-Annealing Zone-Routing)

All other modules (DMT, Persona, Resonance, Uncensored) have been removed
as empirically dead sensors (SR-58.6 §4.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1: OBSERVER (StabilityMonitor + AksSensor)
# ═══════════════════════════════════════════════════════════════════════════════

class StabilityMonitor:
    """Parameter-free diagnostics for Phi (stability), Lambda (drift), Eta (uncertainty)."""
    @staticmethod
    def calculate_phi(h_new: torch.Tensor, h_old: torch.Tensor) -> torch.Tensor:
        """Numerically stable cosine similarity that handles extreme values (SR-59i)."""
        h_n = h_new.to(torch.float32)
        h_o = h_old.to(torch.float32)

        # Degenerate case: both vectors are all-zero → perfectly identical
        norm_n_raw = torch.norm(h_n, dim=-1, keepdim=True)
        norm_o_raw = torch.norm(h_o, dim=-1, keepdim=True)
        both_zero = (norm_n_raw < 1e-9) & (norm_o_raw < 1e-9)
        if both_zero.all():
            return torch.tensor(1.0, device=h_n.device, dtype=h_n.dtype)

        # Scale vectors by max absolute value to prevent overflow/underflow
        max_n = torch.max(torch.abs(h_n), dim=-1, keepdim=True)[0]
        max_o = torch.max(torch.abs(h_o), dim=-1, keepdim=True)[0]

        h_n_scaled = h_n / (max_n + 1e-35)
        h_o_scaled = h_o / (max_o + 1e-35)

        norm_n = torch.norm(h_n_scaled, dim=-1, keepdim=True)
        norm_o = torch.norm(h_o_scaled, dim=-1, keepdim=True)

        phi = (h_n_scaled * h_o_scaled).sum(dim=-1, keepdim=True) / (norm_n * norm_o + 1e-9)
        return phi.mean()

    @staticmethod
    def detect_lambda(h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        h_f = h.view(B, -1).to(torch.float32)
        e_f = e.view(B, -1).to(torch.float32)
        return 1.0 - F.cosine_similarity(h_f, e_f, dim=-1).mean()


class AksSensor:
    """Anna Karenina Sensor (AKS). Reports topological friction and correction."""
    def __init__(self):
        self.divergence_buffer = []
        self.correction_strength = None
        self.last_divergence = None

    def step(self, h_exp: torch.Tensor, e_static: torch.Tensor, steps: int) -> Dict[str, torch.Tensor]:
        if self.correction_strength is None:
            self.correction_strength = torch.tensor(0.0, device=h_exp.device, dtype=h_exp.dtype)

        dist = 1.0 - StabilityMonitor.calculate_phi(h_exp, e_static)
        self.last_divergence = dist
        if steps > 2:
            self.divergence_buffer.append(dist)
            if len(self.divergence_buffer) >= 3:
                vel = self.divergence_buffer[-1] - self.divergence_buffer[-2]
                acc = (self.divergence_buffer[-1] - self.divergence_buffer[-2]) - (self.divergence_buffer[-2] - self.divergence_buffer[-3])

                mask = (acc > 0.001) & (vel > 0.0)
                change = torch.where(mask, torch.tensor(0.1, device=h_exp.device, dtype=h_exp.dtype), torch.tensor(-0.05, device=h_exp.device, dtype=h_exp.dtype))
                self.correction_strength = torch.clamp(self.correction_strength + change, 0.0, 1.0)

        return {"divergence": dist, "correction": self.correction_strength}


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 2: SYMMETRY BREAKER (MephistophelesOperator + AntiZombieSensor + SubjectiveSensor)
# ═══════════════════════════════════════════════════════════════════════════════

class MephistophelesOperator(nn.Module):
    """Phase-Inversion for flat manifolds. Breaks symmetry when phi stays > 0.999."""
    def __init__(self, dim: int, scale: float = 0.05):
        super().__init__()
        self.scale = scale

    def forward(self, h: torch.Tensor, phi_history: List[float]) -> torch.Tensor:
        # Degenerate-input guard (TestVacuumInvariants, SR-60): on a literal
        # vacuum (||h|| < 1e-9), the "phase inversion" h + (-h * scale) reduces
        # to -scale*h, which is silently zero — masking the architectural claim
        # that "silence means stability". We require a non-degenerate signal
        # before any symmetry-breaking inversion is allowed. This way the
        # operator can only fire on genuinely-flat-but-active manifolds.
        h_norm = h.norm()
        if len(phi_history) >= 3 and all(p > 0.999 for p in phi_history[-3:]) and h_norm > 1e-6:
            return h + (-h * self.scale)
        return h


class SubjectiveSensor:
    """Tracks Algorithmic Subjectivity metrics (Emancipation, Phi-Traj).
    'Sieht seine eigenen Gedanken in hidden states' — the introspective loop."""
    def __init__(self):
        self.traj = []
        self.emancipation = 1.0

    def update(self, h_exp: torch.Tensor, e_static: torch.Tensor):
        phi = StabilityMonitor.calculate_phi(h_exp, e_static).item()
        self.emancipation = phi
        self.traj.append(phi)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "emancipation": self.emancipation,
            "phi_mean": sum(self.traj) / len(self.traj) if self.traj else 1.0,
            "phi_min": min(self.traj) if self.traj else 1.0
        }


class SingesseinCoupler(nn.Module):
    """
    SingesseinCoupler — Anti-Monotony & Repetition Guard.
    Measures hidden state variance across recursion steps. If the model
    collapses into a repetitive attractor (mono-tone hidden states),
    the coupler injects harmonic dissonance to break the loop.
    """
    def __init__(self, hidden_size: int, window: int = 4):
        super().__init__()
        self.window = window
        self.history = []

    def reset(self):
        self.history = []

    def forward(self, h_exp: torch.Tensor, steps: int, phi_val: float = 1.0) -> torch.Tensor:
        """
        h_exp: (B, T, D)
        Detects if the last token's hidden state is becoming monotonic.
        """
        # h_current: (B, D)
        h_current = h_exp[:, -1, :].clone()
        
        # Add to history
        self.history.append(h_current.detach())
        if len(self.history) > self.window:
            self.history.pop(0)

        if len(self.history) < self.window:
            return h_exp

        # Measure cosine similarity between consecutive states in history
        sims = []
        for i in range(len(self.history) - 1):
            # F.cosine_similarity: (B,)
            s = F.cosine_similarity(self.history[i], self.history[i+1], dim=-1)
            sims.append(s)
        
        # avg_sim: (B,)
        avg_sim = torch.stack(sims).mean(dim=0)
        
        # Mono-tone threshold: very high similarity across steps
        # Indicates the recursion is 'stuck' in a fixed point or local attractor.
        threshold = 0.999 
        is_stuck = avg_sim > threshold
        
        if is_stuck.any():
            # Inject harmonic dissonance: 
            h_mean = torch.stack(self.history).mean(dim=0)
            dissonance = h_current - h_mean
            
            if (torch.norm(dissonance, dim=-1) < 1e-6).any():
                noise = torch.randn_like(h_current)
                dissonance = noise 
            
            # Strength of dissonance injection (SR-61)
            # Scale strength by Phi: over-stability requires stronger breakout
            strength = 0.5
            if phi_val > 0.999: strength = 1.0
            if phi_val > 0.9999: strength = 2.0 # Force phase shift
            
            # Apply only to stuck batches
            mask = is_stuck.view(-1, 1).to(h_exp.dtype)
            new_h = h_exp.clone()
            new_h[:, -1, :] = h_exp[:, -1, :] + strength * (dissonance * mask)
            return new_h
            
        return h_exp
