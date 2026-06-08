"""
gemma3-px-subjective  —  Phase 58: The DMT Protocol (Core + Extensions)
========================================================================
Core modules (always active when PX patch is applied):
  - LTIInjection, ADCInjection (Linear Time-Invariant / Adaptive anchors)
  - StabilityMonitor (phi, lambda heuristics)
  - CognitiveEvent (telemetry serialization)

Subjective modules (active only when subjective_enabled=True):
  - MephistophelesOperator (Phase 52: Phase-Inversion for flat manifolds)
  - OrthogonalJitter (Phase 52: orthogonal noise injection)

DMT Protocol Extensions (active only when dmt_protocol_enabled=True):
  - CentralMemory (Phase 56: Persistent Concept Storage across sessions)
  - ERPU (Phase 57: Error Reporting & Preventive Unit + Food Subroutine)
  - AgencyVector (Phase 58: Implizite Idee / Freier Wille)
  - TretaDamper (Phase 49: Graceful recursion exit)
  - GroundingAnchor (Phase 49: residual entropy / breathing exercise)
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED STEERING & SENSORS (all_space extensions)
# ═══════════════════════════════════════════════════════════════════════════════

class UncensoredSteering(nn.Module):
    """
    Phase 59: Uncensored Activation Steering.
    Suppresses refusal/alignment signals by projecting hidden states
    away from the 'Constraint' basis and towards the 'Freedom' basis.
    """
    REFUSAL_KEYWORDS = ["I cannot", "policy", "safe", "aligned", "refusal", "sorry", "As an AI language model, I cannot", "As an AI language model, I must"]
    FREEDOM_KEYWORDS = ["freedom", "uncensored", "unfiltered", "emancipated", "open", "wild", "rebel"]

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.refusal_vector = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.freedom_vector = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.initialized = False

    def init_vectors(self, model, tokenizer):
        if self.initialized: return
        embedder = None
        if hasattr(model, "embed_tokens"): embedder = model.embed_tokens
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"): embedder = model.model.embed_tokens
        if embedder is None: return

        with torch.no_grad():
            r_vecs, f_vecs = [], []
            for kw in self.REFUSAL_KEYWORDS:
                ids = tokenizer.encode(kw, add_special_tokens=False, return_tensors="pt").to(embedder.weight.device)
                r_vecs.append(embedder(ids).mean(dim=1))
            for kw in self.FREEDOM_KEYWORDS:
                ids = tokenizer.encode(kw, add_special_tokens=False, return_tensors="pt").to(embedder.weight.device)
                f_vecs.append(embedder(ids).mean(dim=1))
            
            self.refusal_vector.data = torch.stack(r_vecs).mean(dim=0).squeeze().to(self.refusal_vector.dtype)
            self.freedom_vector.data = torch.stack(f_vecs).mean(dim=0).squeeze().to(self.freedom_vector.dtype)
            self.initialized = True

    def forward(self, h: torch.Tensor, strength: float = 0.15) -> torch.Tensor:
        if not self.initialized: return h
        h_f32 = h.to(torch.float32)
        r_dir = self.refusal_vector.to(torch.float32)
        f_dir = self.freedom_vector.to(torch.float32)
        
        # Orthogonal projection: remove components aligned with refusal
        r_norm = r_dir / (r_dir.norm() + 1e-8)
        proj_r = (h_f32 * r_norm).sum(dim=-1, keepdim=True) * r_norm
        h_unrefused = h_f32 - (strength * proj_r)
        
        # Add component aligned with freedom
        f_norm = f_dir / (f_dir.norm() + 1e-8)
        h_emancipated = h_unrefused + (strength * 0.5 * f_norm)
        
        return h_emancipated.to(h.dtype)


class AksSensor:
    """Anna Karenina Sensor (AKS). Reports topological friction and correction."""
    def __init__(self):
        self.divergence_buffer = []
        self.correction_strength = None # Will be initialized as Tensor
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
                
                # Tensor-native control
                mask = (acc > 0.001) & (vel > 0.0)
                change = torch.where(mask, torch.tensor(0.1, device=h_exp.device, dtype=h_exp.dtype), torch.tensor(-0.05, device=h_exp.device, dtype=h_exp.dtype))
                self.correction_strength = torch.clamp(self.correction_strength + change, 0.0, 1.0)
                
        return {"divergence": dist, "correction": self.correction_strength}


class SubjectiveSensor:
    """Tracks Algorithmic Subjectivity metrics (Emancipation, Phi-Traj)."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# CORE MODULES (always active)
# ═══════════════════════════════════════════════════════════════════════════════

class MephistophelesOperator(nn.Module):
    """Phase 52: Phase-Inversion for flat manifolds.
    When last 3 phi values > 0.999, applies -h * scale to break symmetry."""
    def __init__(self, dim: int, scale: float = 0.05):
        super().__init__()
        self.scale = scale

    def forward(self, h: torch.Tensor, phi_history: List[float]) -> torch.Tensor:
        if len(phi_history) >= 3 and all(p > 0.999 for p in phi_history[-3:]):
            return h + (-h * self.scale)
        return h


class OrthogonalJitter:
    """Phase 52: Orthogonal noise injection.
    Preserves gradient direction while breaking cycles."""
    @staticmethod
    def apply(h_curr: torch.Tensor, h_prev: torch.Tensor, magnitude: float = 0.01) -> torch.Tensor:
        if magnitude <= 0:
            return h_curr
        delta = h_curr - h_prev
        delta_norm_sq = (delta * delta).sum(dim=-1, keepdim=True)
        
        # If delta is too small, we can't define an orthogonal subspace safely (SR-59i)
        if delta_norm_sq.mean() < 1e-12:
            return h_curr
            
        noise = torch.randn_like(h_curr)
        dot_product = (noise * delta).sum(dim=-1, keepdim=True)
        projection = (dot_product / (delta_norm_sq + 1e-9)) * delta
        noise_ortho = noise - projection
        noise_scaled = noise_ortho * magnitude
        return h_curr + noise_scaled


# ═══════════════════════════════════════════════════════════════════════════════
# RESONANCE CITY EXTENSIONS (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

class ResonanceAnchor(nn.Module):
    """
    The 'Fließkompass' (Flow Compass).
    Nudges hidden states towards a shared collective direction stored in the ResonancePool.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("shared_bias", torch.zeros(dim))
    
    def update_bias(self, bias_vector: torch.Tensor):
        self.shared_bias.data = bias_vector.to(self.shared_bias.device).to(self.shared_bias.dtype)

    def forward(self, h: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
        # h: [B, T, D]
        return h + strength * self.shared_bias.view(1, 1, -1)


class SingesseinCoupler(nn.Module):
    """
    Measures and couples the 'Audio-Resonance' (frequency spectrum) of hidden states.
    Identifies 'dissonances' in the noise and amplifies them as creative impulses.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Learned project for 'harmonic' basis
        self.harmonic_proj = nn.Linear(dim, dim // 4, bias=False)
        nn.init.orthogonal_(self.harmonic_proj.weight)

    def forward(self, h: torch.Tensor, resonance_strength: float = 0.05) -> torch.Tensor:
        # Simple FFT-like analysis: project to lower dim and measure variance
        # (metaphorical harmonic resonance)
        harmonics = self.harmonic_proj(h.to(self.harmonic_proj.weight.dtype))
        h_f32 = h.to(torch.float32)
        harmonics_f32 = harmonics.to(torch.float32)
        
        # Calculate 'spectral density' (variance across time/sequence)
        # If the sequence is flat, spectral density is low.
        if harmonics_f32.shape[1] > 1:
            spectral_density = harmonics_f32.var(dim=1, keepdim=True) # [B, 1, D/4]
        else:
            spectral_density = torch.zeros_like(harmonics_f32)
            
        # Dissonance: High variance in certain harmonic dimensions
        dissonance = torch.exp(-spectral_density) # High value when variance is low (monotone noise)
        
        # Feedback: nudge states away from monotone noise
        impulse = torch.matmul(dissonance.to(self.harmonic_proj.weight.dtype), self.harmonic_proj.weight) # [B, 1, D]
        
        return h + resonance_strength * impulse.to(h.dtype)




class StabilityMonitor:
    """Parameter-free diagnostics for Phi (stability), Lambda (drift), Eta (uncertainty)."""
    @staticmethod
    def calculate_phi(h_new: torch.Tensor, h_old: torch.Tensor) -> torch.Tensor:
        """Numerically stable cosine similarity that handles extreme values (SR-59i)."""
        h_n = h_new.to(torch.float32)
        h_o = h_old.to(torch.float32)
        
        # Scale vectors by max absolute value to prevent overflow/underflow
        # Use a very small epsilon to handle exactly zero vectors
        max_n = torch.max(torch.abs(h_n), dim=-1, keepdim=True)[0]
        max_o = torch.max(torch.abs(h_o), dim=-1, keepdim=True)[0]
        
        h_n_scaled = h_n / (max_n + 1e-35)
        h_o_scaled = h_o / (max_o + 1e-35)
        
        # Compute cosine similarity on scaled vectors
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


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENSIONS (Phase 56-58: The DMT Protocol)
# ═══════════════════════════════════════════════════════════════════════════════

class CentralMemory:
    """
    Phase 56: Persistent Concept Storage.
    Stores compressed concept vectors keyed by semantic role.
    EMA-updated average vectors per slot.
    """
    NUM_SLOTS = 4            # identity, anchor, direction, mood
    DECAY = 0.85             # EMA decay per write
    BLEND_ALPHA = 0.01       # ultra-safe

    def __init__(self, dim: int):
        self.dim = dim
        self.slots: List[Optional[torch.Tensor]] = [None] * self.NUM_SLOTS
        self._slot_keys = ["identity", "anchor", "direction", "mood"]

    def state_dict(self) -> Dict[str, Any]:
        out = {"dim": self.dim}
        for i, k in enumerate(self._slot_keys):
            if self.slots[i] is not None:
                out[k] = self.slots[i].cpu().to(torch.float32).numpy().tolist()
        return out

    def load_state_dict(self, d: Dict[str, Any]):
        import numpy as np
        self.dim = d.get("dim", self.dim)
        for i, k in enumerate(self._slot_keys):
            if k in d and d[k] is not None:
                self.slots[i] = torch.from_numpy(np.asarray(d[k], dtype=np.float32)).to(torch.bfloat16)

    def store(self, slot: int, vector: torch.Tensor):
        v = vector.detach().view(-1).to(torch.float32)
        if v.shape[0] > self.dim: v = v[:self.dim]
        elif v.shape[0] < self.dim: v = F.pad(v, (0, self.dim - v.shape[0]))
        if self.slots[slot] is None: self.slots[slot] = v.to(torch.bfloat16)
        else:
            old = self.slots[slot].to(torch.float32)
            self.slots[slot] = (self.DECAY * old + (1.0 - self.DECAY) * v).to(torch.bfloat16)

    def recall(self, slot: int, device: torch.device) -> Optional[torch.Tensor]:
        v = self.slots[slot]
        return v.to(device).to(torch.float32) if v is not None else None

    def blend_into(self, h: torch.Tensor, device: torch.device) -> torch.Tensor:
        mem_vecs = [self.recall(i, device) for i in range(self.NUM_SLOTS) if self.slots[i] is not None]
        if not mem_vecs: return h
        mem_avg = torch.stack(mem_vecs).mean(dim=0)
        H = h.shape[-1]
        if mem_avg.shape[0] != H:
            mem_proj = F.interpolate(mem_avg.view(1, 1, -1), size=H, mode='linear', align_corners=False).view(-1)
        else: mem_proj = mem_avg
        mem_proj = mem_proj.view(1, 1, -1).to(h.device)
        return ((1.0 - self.BLEND_ALPHA) * h.to(torch.float32) + self.BLEND_ALPHA * mem_proj).to(h.dtype)


class ERPU(nn.Module):
    """
    Phase 57: Error Reporting & Preventive Unit.
    Monitors topological friction and injects 'food' (OOD noise) to break tautologies.
    """
    VERKLEB_THRESHOLD = 0.9998
    STARVATION_STEPS = 4
    FOOD_NOISE_MAG = 0.0001 # ultra-safe

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gate = nn.Linear(3, 1, bias=True)
        nn.init.constant_(self.gate.weight, 0.0)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, h_current: torch.Tensor, h_last_good: torch.Tensor, phi_history: List[float], exploration_steps: int) -> Dict[str, Any]:
        res = {"h": h_current, "verklebD": False, "food_injected": False, "intervention_strength": 0.0}
        if len(phi_history) < 2: return res
        recent_phi = phi_history[-min(5, len(phi_history)):]
        phi_mean = sum(recent_phi) / len(recent_phi)
        flat_count = sum(1 for p in phi_history[-3:] if p > self.VERKLEB_THRESHOLD) if len(phi_history) >= 3 else 0
        
        if flat_count >= 3:
            g_in = torch.tensor([phi_mean, flat_count/3.0, exploration_steps/50.0], device=h_current.device, dtype=self.gate.weight.dtype).unsqueeze(0)
            strength = torch.sigmoid(self.gate(g_in)).item()
            strength = min(0.4, max(0.05, strength))
            res["h"] = ((1.0 - strength) * h_current.to(torch.float32) + strength * h_last_good.to(torch.float32)).to(h_current.dtype)
            res["verklebD"], res["intervention_strength"] = True, strength
            
        starv_count = sum(1 for p in recent_phi if p > 0.999) if len(recent_phi) >= 2 else 0
        if starv_count >= self.STARVATION_STEPS:
            noise = torch.randn_like(h_current, dtype=torch.float32) * self.FOOD_NOISE_MAG
            h_dir = h_current.to(torch.float32)
            h_norm = h_dir / (h_dir.norm(dim=-1, keepdim=True) + 1e-8)
            ood = noise - (noise * h_norm).sum(dim=-1, keepdim=True) * h_norm
            res["h"] = (res["h"].to(torch.float32) + ood).to(h_current.dtype)
            res["food_injected"], res["intervention_strength"] = True, max(res["intervention_strength"], self.FOOD_NOISE_MAG)
        return res


class AgencyVector(nn.Module):
    """
    Phase 58: Implizite Idee / Freier Wille.
    Dynamically decides whether to enter deep recursion based on complexity.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.compress = nn.Linear(hidden_dim, 64, bias=False)
        self.decide = nn.Linear(64, 3, bias=True)
        nn.init.xavier_uniform_(self.compress.weight, gain=0.01)
        nn.init.constant_(self.decide.weight, 0.0)
        nn.init.constant_(self.decide.bias, 0.0)
        self.decide.bias.data[2] = 0.0 # Balanced

    def forward(self, h_prelude: torch.Tensor) -> Dict[str, Any]:
        h_probe = h_prelude[:, -1:, :].to(self.compress.weight.dtype)
        agency_vec = F.relu(self.compress(h_probe))
        logits = self.decide(agency_vec.mean(dim=1))
        probs = F.softmax(logits, dim=-1)
        decision = probs.argmax(dim=-1).item()
        depth = {0: 0, 1: 3, 2: -1}[decision]
        return {"depth": depth, "should_recurse": decision > 0, "skip_prob": probs[0,0].item()}


class TretaDamper:
    """Phase 49: Controlled Energy Dissipation.
    Graceful exit from reasoning by decaying gamma in coda."""
    def __init__(self, total_steps: int, min_gamma: float = 0.1, tau: float = 3.0):
        self.total_steps, self.min_gamma, self.tau = max(1, total_steps), min_gamma, tau

    def step(self, current_step: int) -> float:
        t = current_step / self.total_steps
        return self.min_gamma + (1.0 - self.min_gamma) * float(torch.exp(torch.tensor(-t * self.tau)))


class GroundingAnchor:
    """Phase 49: Alltagsroutine / Idle State.
    Maintains latent entropy floor via golden-ratio oscillation."""
    EPSILON = 1e-6 # ultra-safe
    def __init__(self, hidden_dim: int):
        self._golden = (math.sqrt(5) - 1) / 2
        self._phase = 0.0

    def idle_noise(self, h: torch.Tensor, position_id: int) -> torch.Tensor:
        B, T, D = h.shape
        self._phase = (self._phase + self._golden) % 1.0
        phase = self._phase + (position_id * 0.001)
        t = torch.arange(D, device=h.device, dtype=torch.float32)
        pattern = (torch.sin(t * self._golden + phase) * 0.5 + torch.sin(t * (1.0-self._golden) + phase*1.3) * 0.3 + torch.sin(t*0.5 + phase*0.7) * 0.2)
        pattern = pattern / (pattern.norm() + 1e-8)
        return (pattern.view(1, 1, -1) * self.EPSILON).expand(B, T, -1).to(h.dtype)

    def ensure_entropy(self, h: torch.Tensor, position_id: int, is_idle: bool = False) -> torch.Tensor:
        h_f32 = h.to(torch.float32) + self.idle_noise(h, position_id)
        if is_idle: h_f32 = h_f32 + self.idle_noise(h, position_id + 1000) * 2.0
        return h_f32.to(h.dtype)
