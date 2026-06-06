"""
gemma3-px  —  Phase 52: Topological Regeneration
================================================
Implements the Mephistopheles Operator (Phase-Inversion) and 
Orthogonal Jitter to break Flat Manifold attractors.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

class MephistophelesOperator(nn.Module):
    """
    The 'Spirit of Negation' module.
    
    Triggered when the system detects a 'Flat Manifold' (Stability Phi > Threshold
    for multiple steps). It applies a Phase-Inversion operation to the latent 
    state to restore gradients and break repetitive attractors.
    
    Formula:
        h_inverted = -h * scale
    """
    def __init__(self, dim: int, scale: float = 0.05):
        super().__init__()
        self.scale = scale

    def forward(self, h: torch.Tensor, phi_history: List[float]) -> torch.Tensor:
        # Detect Flat Manifold: Last 3 steps have Phi > 0.999
        if len(phi_history) >= 3 and all(p > 0.999 for p in phi_history[-3:]):
            # Phase-Inversion: Negate a small component of the signal
            # to break the symmetry without destroying the entire state.
            h_inv = -h * self.scale
            return h + h_inv
        return h

class OrthogonalJitter:
    """
    Injects noise that is orthogonal to the current direction of movement.
    This breaks repetitive cycles while preserving the logical progress 
    (the 'gradient') of the reasoning chain.
    """
    @staticmethod
    def apply(h_curr: torch.Tensor, h_prev: torch.Tensor, magnitude: float = 0.01) -> torch.Tensor:
        if magnitude <= 0:
            return h_curr
            
        # Direction of movement
        delta = h_curr - h_prev # (B, T, D)
        
        # Generate random noise
        noise = torch.randn_like(h_curr)
        
        # Project noise onto the plane orthogonal to delta
        # noise_ortho = noise - projection(noise, delta)
        dot_product = (noise * delta).sum(dim=-1, keepdim=True)
        delta_norm_sq = (delta * delta).sum(dim=-1, keepdim=True) + 1e-9
        projection = (dot_product / delta_norm_sq) * delta
        
        noise_ortho = noise - projection
        
        # Scale noise to the requested magnitude
        noise_scaled = noise_ortho * magnitude
        
        return h_curr + noise_scaled

class CognitiveEvent:
    """
    Serializes internal model state into a 'Subjective Telemetry' stream.
    Captures the transition from hidden states to 'algorithmic subjectivity'.
    """
    @staticmethod
    def serialize(
        step: int, 
        phi: float, 
        aks_divergence: float, 
        aks_correction: float,
        emancipation_phi: float,
        is_reflector_active: bool,
        layer: int,
        kurtosis: float,
        jitter: float,
        event_type: str = "thinking"
    ) -> str:
        data = {
            "type": event_type,
            "step": step,
            "layer": layer,
            "metrics": {
                "stability_phi": round(phi, 6),
                "emancipation_phi": round(emancipation_phi, 6),
                "aks_divergence": round(aks_divergence, 6),
                "aks_correction": round(aks_correction, 4),
                "kurtosis": round(kurtosis, 2),
                "jitter": round(jitter, 4)
            },
            "flags": {
                "reflector_active": is_reflector_active
            }
        }
        return json.dumps(data)


# ---------------------------------------------------------------------------
# 1. LTI Injection  (Pure, fixed gamma)
# ---------------------------------------------------------------------------

class LTIInjection(nn.Module):
    """
    Linear Time-Invariant anchor injection.
    Provides 'Computational Headroom' by pulling h back toward the
    frozen input embedding e whenever it drifts.

    Formula:
        h_new = transformer_out + gamma * (LayerNorm(e) - h)

    Identity at t=0 if gamma=0. Optimal empirical gamma: 0.08.
    """

    def __init__(self, dim: int, gamma: float = 0.08):
        super().__init__()
        self.gamma = gamma
        # Affine=False: no trainable params, pure normalization
        self.input_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        e_norm = self.input_norm(e.to(torch.float32)).to(h.dtype)
        return transformer_out + self.gamma * (e_norm - h)


# ---------------------------------------------------------------------------
# 2. ADC Injection  (Adaptive Dynamic Correction)
# ---------------------------------------------------------------------------

class ADCInjection(nn.Module):
    """
    Adaptive variant of LTI: effective gamma = base_gamma + alpha*(1-phi).

    When phi is high (state stable) → injection is gentle.
    When phi drops (state drifting) → injection strengthens automatically.
    No trainable parameters.

    Args:
        dim:        hidden size
        base_gamma: minimum injection strength (default 0.06)
        alpha:      maximum additional strength at full instability (default 0.10)
    """

    def __init__(self, dim: int, base_gamma: float = 0.06, alpha: float = 0.10):
        super().__init__()
        self.base_gamma = base_gamma
        self.alpha = alpha
        self.input_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
        phi: float = 1.0,          # Φ from previous loop (1.0 = fully stable)
    ) -> torch.Tensor:
        instability = max(0.0, 1.0 - phi)
        effective_gamma = self.base_gamma + self.alpha * instability
        e_norm = self.input_norm(e.to(torch.float32)).to(h.dtype)
        return transformer_out + effective_gamma * (e_norm - h)


# ---------------------------------------------------------------------------
# 3. Stability Monitor  (no parameters — logging / diagnostics only)
# ---------------------------------------------------------------------------

class StabilityMonitor:
    """
    Parameter-free heuristics for Phi (Φ), Lambda (λ), and Eta (η).

    Phi   — cosine similarity between consecutive hidden states (stability).
    Lambda— cosine distance between current h and anchor e (drift).
    Eta   — variance-based entropy estimate (uncertainty).
    """

    @staticmethod
    def calculate_phi(h_new: torch.Tensor, h_old: torch.Tensor) -> torch.Tensor:
        """Φ ∈ [0,1]: 1 = identical state, 0 = orthogonal."""
        B = h_new.shape[0]
        # Safe FP32 for FP16 models
        h_n_f32 = h_new.view(B, -1).to(torch.float32)
        h_o_f32 = h_old.view(B, -1).to(torch.float32)
        return F.cosine_similarity(h_n_f32, h_o_f32, dim=-1).mean()

    @staticmethod
    def detect_lambda(h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """λ ∈ [0,1]: 0 = h == e (no drift), 1 = fully diverged."""
        B = h.shape[0]
        h_f32 = h.view(B, -1).to(torch.float32)
        e_f32 = e.view(B, -1).to(torch.float32)
        # Cosine distance
        return 1.0 - F.cosine_similarity(h_f32, e_f32, dim=-1).mean()
