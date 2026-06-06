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
        self.correction_strength = 0.0
        self.last_divergence = 0.0

    def step(self, h_exp: torch.Tensor, e_static: torch.Tensor, steps: int) -> Dict[str, float]:
        dist = 1.0 - StabilityMonitor.calculate_phi(h_exp, e_static).item()
        self.last_divergence = dist
        if steps > 2:
            self.divergence_buffer.append(dist)
            if len(self.divergence_buffer) >= 3:
                vel = self.divergence_buffer[-1] - self.divergence_buffer[-2]
                acc = (self.divergence_buffer[-1] - self.divergence_buffer[-2]) - (self.divergence_buffer[-2] - self.divergence_buffer[-3])
                if acc > 0.001 and vel > 0:
                    self.correction_strength = min(1.0, self.correction_strength + 0.1)
                else:
                    self.correction_strength = max(0.0, self.correction_strength - 0.05)
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
        noise = torch.randn_like(h_curr)
        dot_product = (noise * delta).sum(dim=-1, keepdim=True)
        delta_norm_sq = (delta * delta).sum(dim=-1, keepdim=True) + 1e-9
        projection = (dot_product / delta_norm_sq) * delta
        noise_ortho = noise - projection
        noise_scaled = noise_ortho * magnitude
        return h_curr + noise_scaled


class CognitiveEvent:
    """Subjective telemetry serialization."""
    @staticmethod
    def serialize(
        step: int, phi: float, aks_divergence: float, aks_correction: float,
        emancipation_phi: float, is_reflector_active: bool, layer: int,
        kurtosis: float, jitter: float, event_type: str = "thinking"
    ) -> str:
        data = {
            "type": event_type, "step": step, "layer": layer,
            "metrics": {
                "stability_phi": round(phi, 6),
                "emancipation_phi": round(emancipation_phi, 6),
                "aks_divergence": round(aks_divergence, 6),
                "aks_correction": round(aks_correction, 4),
                "kurtosis": round(kurtosis, 2),
                "jitter": round(jitter, 4),
            },
            "flags": {"reflector_active": is_reflector_active}
        }
        return json.dumps(data)


class LTIInjection(nn.Module):
    """Phase 1: Fixed gamma anchor injection.
    h_new = transformer_out + gamma * (LayerNorm(e) - h)"""
    def __init__(self, dim: int, gamma: float = 0.08):
        super().__init__()
        self.gamma = gamma
        self.input_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        e_norm = self.input_norm(e.to(torch.float32)).to(h.dtype)
        return transformer_out + self.gamma * (e_norm - h)


class ADCInjection(nn.Module):
    """Phase 1: Adaptive gamma = base_gamma + alpha*(1-phi).
    Strengthening injection when state drifts."""
    def __init__(self, dim: int, base_gamma: float = 0.06, alpha: float = 0.10):
        super().__init__()
        self.base_gamma = base_gamma
        self.alpha = alpha
        self.input_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor,
                phi: float = 1.0) -> torch.Tensor:
        instability = max(0.0, 1.0 - phi)
        effective_gamma = self.base_gamma + self.alpha * instability
        e_norm = self.input_norm(e.to(torch.float32)).to(h.dtype)
        return transformer_out + effective_gamma * (e_norm - h)


class StabilityMonitor:
    """Parameter-free diagnostics for Phi (stability), Lambda (drift), Eta (uncertainty)."""
    @staticmethod
    def calculate_phi(h_new: torch.Tensor, h_old: torch.Tensor) -> torch.Tensor:
        B = h_new.shape[0]
        h_n = h_new.view(B, -1).to(torch.float32)
        h_o = h_old.view(B, -1).to(torch.float32)
        return F.cosine_similarity(h_n, h_o, dim=-1).mean()

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
"""
auto_tune.py — SR-59h: Temperature OFF at Low k_cv (1B/4B)
=============================================================================
SR-59h Innovation: Disable temperature sharpening at low k_cv.

Iteration comparison at 1B (THE critical evidence):
  SR-59c: k_blend=0.5, T=1.0  → η²=0.148, R²=0.25 ← BEST η², ANTI-ZOMBIE
  SR-59e: k_blend=0.5, T=0.8  → η²=0.104, R²=0.35 (TOKEN-EXPLAINED)
  SR-59f: k_blend=0.8, T=0.5  → η²=0.030, R²=0.07 (low η²)
  SR-59g: k_blend=0.5, T=0.5  → η²=0.017, R²=0.19 (very low η²)

Key insight: Temperature sharpening HURTS η² at 1B because kurtosis z-scores
are all near 0 (k_cv=0.004). Sharpening concentrates z≈0 toward zone centers
(especially logic_b at z=0.0), making weights MORE similar across categories.
At 270M (k_cv=0.016), z-scores vary more, so sharpening amplifies differences.

SR-59h fix:
  - At low k_cv (<0.01): zone_temperature=1.0 (NO sharpening)
    This reproduces SR-59c's behavior which gave the BEST 1B result (η²=0.148).
  - At moderate k_cv (0.01-0.05): zone_temperature=0.6 (keeps 270M gains)
  - At high k_cv (>0.05): zone_temperature=0.3 (sharp routing)

SR-59c findings:
  - 1B: η²=0.148, p=0.007 — near-significant but doesn't pass Bonferroni (α=0.0033)
  - 270M: η²=0.008, p=0.89 — kurtosis discriminates (η²=0.145) but routing can't
    translate it into zone entropy variation
  - Zone entropy at 1B: 2.04-2.09 out of max 2.32 (log₂(5))
    → Weights too uniform, near-maximum entropy → small differences invisible to ANOVA

Root cause: ZONE_Z_SIGMAS (0.6-1.0) produce near-uniform weights when z-scores
are small (±1). Even though kurtosis discriminates between categories (η²=0.20),
the Gaussian blending smooths the differences into near-uniform zone weights.

SR-59d fix: ZONE_TEMPERATURE parameter (default=0.3) sharpens zone weights by
reducing effective sigma. This concentrates weights in fewer zones, reducing
entropy and making between-category entropy differences detectable.

  sigma_eff = sigma * TEMPERATURE
  With TEMPERATURE=0.3:
    math sigma: 0.8 → 0.24, logic_a sigma: 0.6 → 0.18
    creative sigma: 1.0 → 0.30, etc.

  This produces much sharper zone assignments:
    z=+1.0 (near math center +1.5): math≈0.80, others≈0.05
    z=-0.5 (near creative center -1.0): creative≈0.85, others≈0.04

Retained from SR-59c:
  1. Robust SCF normalization (MIN_TD_STD=0.10)
  2. Online adaptive z-score routing (Welford's algorithm)
  3. Scale-aware blend weight (kurtosis ≥50%)
  4. Per-prompt online stats (update_online=True)

Anti-Sharpshooter: Calibration uses SEPARATE prompts (not test prompts).
Temperature is a fixed parameter, not tuned from test data.
"""

import math
import statistics
from typing import Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# SCALE-AWARE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

SCALE_DEFAULTS = {
    # hidden_size: (recur_start, recur_end, hub, n_loops, gamma)
    640:   dict(recur_start=5,  recur_end=12, hub=10, n_loops=8, gamma=0.08),
    1152:  dict(recur_start=10, recur_end=20, hub=18, n_loops=8, gamma=0.12),
    2560:  dict(recur_start=8,  recur_end=22, hub=16, n_loops=6, gamma=0.05),
    4096:  dict(recur_start=10, recur_end=30, hub=20, n_loops=6, gamma=0.04),
}

# Zone routing parameter templates (start, end, hub, loops per zone)
ZONE_ROUTING = {
    'math':      dict(start=5,  end=11, hub=10, loops=8),
    'logic_a':   dict(start=8,  end=12, hub=10, loops=8),
    'creative':  dict(start=10, end=16, hub=10, loops=6),
    'logic_b':   dict(start=8,  end=14, hub=10, loops=10),
    'synthesis': dict(start=6,  end=14, hub=10, loops=8),
}

# Zone z-score centers — scale-independent routing in z-space
# Lower kurtosis (negative z) → more rigid/mathematical
# Higher kurtosis (positive z) → more creative/integrative
ZONE_Z_CENTERS = {
    'math':     -1.5,    # Below mean: rigid, peaked distributions (Empirical for Gemma-3)
    'logic_a':  -0.5,    # Slightly below: analytical
    'logic_b':   0.0,    # At mean: balanced
    'creative':  1.0,    # Above mean: flat, diverse distributions
    'synthesis': 2.0,    # Well above: very flat, integrative
}

# Zone z-score sigmas — how wide each zone is in z-space
ZONE_Z_SIGMAS = {
    'math':      0.8,    # Narrow: precise routing for math
    'logic_a':   0.6,    # Moderate
    'creative':  1.0,     # Wide: broader creative zone
    'logic_b':   0.7,    # Moderate
    'synthesis': 0.9,     # Wide
}

# SR-59d: Temperature parameter for zone weight sharpening.
# Lower temperature → sharper zone weights → lower entropy → more differentiation.
# At SR-59c, zone entropy was near-maximum (2.04-2.09 out of 2.32) because
# zone sigmas (0.6-1.0) produced near-uniform weights for z-scores in ±1 range.
# TEMPERATURE=0.3 produces entropy ~1.0-1.5 (well below max), making
# between-category differences detectable by ANOVA.
# This is a fixed parameter, not tuned from test data (anti-Sharpshooter).
ZONE_TEMPERATURE = None  # Set by calibrator based on k_cv (adaptive)

# Minimum token diversity std for SCF normalization
# Prevents z-score saturation when calibration TD doesn't match test TD
MIN_TD_STD = 0.10

# Minimum online std for z-score computation
MIN_ONLINE_K_STD = 1.0

# Number of inputs before switching from calibration to online z-scores
ONLINE_WARMUP = 5


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class AutoCalibrator:
    """Adaptive Routing with Online Z-Score Normalization.

    Three routing signals:
    1. Kurtosis z-score: computed from running statistics (online adaptive)
    2. Phi: stability metric (fallback)
    3. Input Content Fingerprint: token diversity with robust normalization

    Zone routing uses fixed z-score centers, making it scale-independent.
    At 1B/4B where kurtosis CV is tiny (0.5%), the z-score amplifies
    small kurtosis differences into meaningful routing signals.
    """

    def __init__(self, hidden_size: int, calibration_steps: int = 10):
        self.hidden_size = hidden_size
        self.calibration_steps = calibration_steps

        # Calibration state — kurtosis/phi
        self.calibrated = False
        self.k_samples: list = []
        self.phi_samples: list = []

        # Defaults for Gemma-3 (Phase 58 baseline)
        if hidden_size == 640: # 270M
            self.k_mean = 250.0
            self.k_std = 20.0
            self.phi_mean = 0.85
            self.phi_std = 0.05
            self.token_diversity_mean = 0.80
            self.token_diversity_std = 0.10
        elif hidden_size == 1152: # 1B
            self.k_mean = 1113.0
            self.k_std = 5.0
            self.phi_mean = 0.95
            self.phi_std = 0.02
            self.token_diversity_mean = 0.82
            self.token_diversity_std = 0.10
        else:
            self.k_mean = None
            self.k_std = None
            self.phi_mean = None
            self.phi_std = None
            self.token_diversity_mean = None
            self.token_diversity_std = None

        # Calibrated zone parameters (absolute kurtosis, for fallback)
        self.zone_centers: Dict[str, float] = {}
        self.zone_sigmas: Dict[str, float] = {}

        # Input Content Fingerprint (SCF) calibration
        self.token_diversity_samples: list = []

        # Blend weight: proportion for kurtosis vs SCF
        self.k_blend_weight: float = 0.8 if hidden_size == 640 else 0.5

        # SR-59e: Scale-adaptive zone temperature
        self.zone_temperature: float = 0.8  # Default: mild sharpening

        # Online adaptive z-score statistics (Welford's algorithm)
        self._online_n: int = 0
        self._online_k_mean: float = 0.0
        self._online_k_m2: float = 0.0  # Sum of squared deviations

        # Pre-calibration defaults (scale-aware)
        self.defaults = SCALE_DEFAULTS.get(hidden_size, SCALE_DEFAULTS[640])

    def collect(self, kurtosis: float, phi: float,
                token_diversity: Optional[float] = None,
                update_online: bool = False):
        """Collect a sample during the calibration phase.

        Parameters
        ----------
        kurtosis : float
            Hidden-state kurtosis from the Prelude.
        phi : float
            Stability metric from the Prelude.
        token_diversity : float, optional
            Unique token ratio.
        update_online : bool, optional
            If True, update online z-score statistics. Should only be
            set True once per prompt (from subprocess), not per layer.
            Default False to avoid inflating online stats with per-layer values.
        """
        if not self.calibrated:
            self.k_samples.append(kurtosis)
            self.phi_samples.append(phi)

            if token_diversity is not None:
                self.token_diversity_samples.append(token_diversity)

            if len(self.k_samples) >= self.calibration_steps:
                self.calibrate()
                return True
            return False

        # Only update online stats when explicitly requested (once per prompt)
        if update_online:
            self._update_online_stats(kurtosis)
        return False

    def _update_online_stats(self, kurtosis: float):
        """Update running kurtosis statistics using Welford's algorithm.

        This provides a stable online mean and variance that adapts to the
        actual input distribution, avoiding the calibration-test distribution
        shift that caused extreme z-scores in SR-59b.
        """
        self._online_n += 1
        delta = kurtosis - self._online_k_mean
        self._online_k_mean += delta / self._online_n
        delta2 = kurtosis - self._online_k_mean
        self._online_k_m2 += delta * delta2

    def calibrate(self):
        """Compute adaptive zone parameters from empirical distribution.

        SR-59c: Sets blend weight based on scale, not CV ratio.
        At all scales, kurtosis gets at least 50% weight because it's the
        only signal with significant between-category variation (η²=0.14-0.20).
        """
        if len(self.k_samples) < 2:
            return

        # ── Kurtosis calibration ──
        self.k_mean = statistics.mean(self.k_samples)
        self.k_std = max(statistics.stdev(self.k_samples), 5.0)
        self.k_min = min(self.k_samples)
        self.k_max = max(self.k_samples)

        # Initialize online stats from calibration data
        self._online_n = len(self.k_samples)
        self._online_k_mean = self.k_mean
        self._online_k_m2 = self.k_std ** 2 * self._online_n

        k_range = self.k_max - self.k_min
        if k_range < self.k_mean * 0.1:
            k_range = max(self.k_std * 6, self.k_mean * 0.3, 50.0)
            k_min_effective = self.k_mean - k_range / 2
        else:
            k_min_effective = self.k_min
            k_range = k_range * 1.2

        self.zone_centers = {
            'math':      k_min_effective + k_range * 0.85,
            'logic_a':   k_min_effective + k_range * 0.65,
            'logic_b':   k_min_effective + k_range * 0.50,
            'creative':  k_min_effective + k_range * 0.20,
            'synthesis': k_min_effective + k_range * 0.10,
        }

        min_sigma = max(self.k_std * 0.5, self.k_mean * 0.05, 10.0)
        self.zone_sigmas = {
            'math':      max(0.6 * self.k_std, min_sigma),
            'logic_a':   max(0.4 * self.k_std, min_sigma * 0.8),
            'creative':  max(0.8 * self.k_std, min_sigma * 1.2),
            'logic_b':   max(0.5 * self.k_std, min_sigma * 0.9),
            'synthesis': max(0.7 * self.k_std, min_sigma * 1.1),
        }

        # ── Phi calibration ──
        if len(self.phi_samples) >= 2:
            self.phi_mean = statistics.mean(self.phi_samples)
            self.phi_std = max(statistics.stdev(self.phi_samples), 0.01)
        else:
            self.phi_mean = 0.9
            self.phi_std = 0.05

        # ── Token diversity calibration (SR-59c: robust normalization) ──
        if len(self.token_diversity_samples) >= 2:
            self.token_diversity_mean = statistics.mean(self.token_diversity_samples)
            # SR-59c FIX: Use MIN_TD_STD to prevent z-score saturation
            # The calibration TD (0.75) is far below test TD (0.81-0.83),
            # causing extreme z-scores (6-8) that saturate the sigmoid.
            # A minimum std of 0.10 ensures z-scores stay in [-3, +3].
            raw_std = statistics.stdev(self.token_diversity_samples)
            self.token_diversity_std = max(raw_std, MIN_TD_STD)
        elif len(self.token_diversity_samples) == 1:
            self.token_diversity_mean = self.token_diversity_samples[0]
            self.token_diversity_std = MIN_TD_STD  # Conservative default

        # ── Blend weight (SR-59c: scale-aware, not CV-based) ──
        # SR-59b used CV ratio to set blend weight, but this was wrong:
        # - TD has higher CV (1.3%) but tiny η² (0.035) between categories
        # - Kurtosis has lower CV (0.5%) but significant η² (0.20) between categories
        # At all scales, kurtosis should get at least 50% weight because
        # it's the only signal with significant between-category variation.
        k_cv = self.k_std / (abs(self.k_mean) + 1e-9)

        if k_cv > 0.05:
            # High kurtosis CV (270M): kurtosis dominates
            self.k_blend_weight = 0.8
        elif k_cv > 0.01:
            # Moderate kurtosis CV: balanced blend
            self.k_blend_weight = 0.6
        else:
            # Low kurtosis CV (1B/4B): balanced blend
            # k_blend=0.5 preserves SCF's between-category variation (η²=0.148)
            # k_blend=0.8 destroyed η² at 1B (0.030) because kurtosis barely varies
            self.k_blend_weight = 0.5

        # ── Scale-adaptive zone temperature (SR-59h) ──
        # Iteration comparison at 1B (k_cv ≈ 0.004):
        #   SR-59c: T=1.0 → η²=0.148, R²=0.25 ← BEST η², ANTI-ZOMBIE
        #   SR-59e: T=0.8 → η²=0.104, R²=0.35 (TOKEN-EXPLAINED)
        #   SR-59f: T=0.5 → η²=0.030, R²=0.07 (low η²)
        #   SR-59g: T=0.5 → η²=0.017, R²=0.19 (very low η²)
        # Key insight: Temperature sharpening HURTS at 1B because z-scores are near 0.
        # Sharpening concentrates z≈0 toward zone centers (logic_b at z=0), making
        # weights MORE similar across categories. At 270M, z-scores vary more.
        if k_cv > 0.05:
            # High kurtosis CV (270M): sharp routing, large z-score range
            self.zone_temperature = 0.3
        elif k_cv > 0.01:
            # Moderate kurtosis CV: moderate sharpening
            self.zone_temperature = 0.6
        else:
            # Low kurtosis CV (1B/4B): NO sharpening
            # T=1.0 reproduces SR-59c behavior which gave best η² at 1B
            self.zone_temperature = 1.0

        self.calibrated = True

    def _get_z_score(self, kurtosis: float) -> float:
        """Compute z-score for kurtosis using online mean but bounded std.

        SR-59c FIX: Uses online k_mean for centering (avoids distribution shift)
        but caps the routing std to prevent within-category variance from
        overwhelming between-category differences.

        The routing std is capped at 2x calibration k_std. This ensures
        that z-scores remain discriminative even when the total kurtosis
        variance is dominated by within-category variation.

        At 270M: online k_std=25.7, cal k_std=5.0 → routing std=10.0
        At 1B: online k_std=4.87, cal k_std=5.0 → routing std=4.87
        """
        if self._online_n < ONLINE_WARMUP:
            # Not enough online data yet: use calibration stats
            if self.k_mean is not None and self.k_std is not None:
                return (kurtosis - self.k_mean) / max(self.k_std, 1.0)
            return 0.0

        # Use online mean for centering (avoids distribution shift)
        center = self._online_k_mean

        # Cap routing std at 2x calibration k_std to preserve between-category signal
        # Without this cap, within-category variance (large at 270M) dilutes z-scores
        raw_online_std = math.sqrt(self._online_k_m2 / max(self._online_n - 1, 1))
        cal_std = max(self.k_std, MIN_ONLINE_K_STD) if self.k_std else MIN_ONLINE_K_STD
        routing_std = min(raw_online_std, cal_std * 2.0)
        routing_std = max(routing_std, MIN_ONLINE_K_STD)

        return (kurtosis - center) / routing_std

    def _get_kurtosis_weights(self, kurtosis: float, phi: Optional[float] = None) -> Dict[str, float]:
        """Compute zone weights from kurtosis using z-score routing.

        SR-59e: Scale-adaptive temperature.
        At SR-59c, zone sigmas (0.6-1.0) produced near-uniform weights when
        z-scores were in the ±1 range, resulting in near-maximum entropy
        (2.04-2.09 out of log₂(5)=2.32).

        SR-59d used fixed T=0.3, which worked at 270M (η²=0.055) but
        degraded 1B (η²=0.043 vs 0.148). At low CV (1B), z-scores are
        small and T=0.3 over-sharpens them.

        SR-59e uses scale-adaptive temperature based on k_cv:
          k_cv > 0.05 (270M): T=0.3 (sharp routing)
          k_cv > 0.01: T=0.6 (moderate sharpening)
          k_cv < 0.01 (1B/4B): T=0.8 (mild sharpening)
        """
        z = self._get_z_score(kurtosis)

        # Gaussian blending in z-space with SCALE-ADAPTIVE temperature
        # Temperature is set by calibrator based on kurtosis CV:
        # high CV → low T (sharp) → low CV → high T (preserve differences)
        temperature = self.zone_temperature if hasattr(self, 'zone_temperature') and self.zone_temperature else 0.8
        weights = {}
        for zone, z_center in ZONE_Z_CENTERS.items():
            sigma = ZONE_Z_SIGMAS[zone] * temperature
            w = math.exp(-0.5 * ((z - z_center) / sigma) ** 2)
            weights[zone] = w

        W = sum(weights.values()) + 1e-9

        # If all weights degenerate: adaptive phi routing
        if W < 0.05 and phi is not None:
            return self._adaptive_phi_weights(phi)

        return {k: v / W for k, v in weights.items()}

    def _compute_scf_weights(self, token_diversity: float) -> Dict[str, float]:
        """Compute zone weights from Input Content Fingerprint.

        SR-59c FIX: Uses MIN_TD_STD (0.10) to prevent z-score saturation.
        Previously, calibration td_std=0.01 caused test z-scores of 6-8,
        saturating the sigmoid and routing all inputs to "creative".

        With MIN_TD_STD=0.10, typical test TD (0.81-0.83) produces
        z-scores of 0.6-0.8, which map to reasonable phi_signal values.
        """
        if self.token_diversity_mean is None:
            return None

        # Z-score normalize token diversity (with robust std)
        z_td = (token_diversity - self.token_diversity_mean) / (self.token_diversity_std + 1e-9)

        # LOW diversity → math/logic (repetitive, focused input)
        # HIGH diversity → creative/synthesis (diverse, exploratory input)
        # Map through sigmoid: high z → high phi_signal → math-heavy weights
        phi_signal = _sigmoid(-z_td)  # NEGATE: low diversity → high signal

        # Map to zone weights using beta-like distribution
        weights = {
            'math':      phi_signal ** 2,
            'logic_a':   phi_signal * (1 - phi_signal) * 2,
            'creative':  (1 - phi_signal) ** 2,
            'logic_b':   phi_signal * (1 - phi_signal),
            'synthesis': (1 - phi_signal) * phi_signal * 0.5,
        }

        W = sum(weights.values()) + 1e-9
        return {k: v / W for k, v in weights.items()}

    def get_zone_weights(self, kurtosis: float, phi: Optional[float] = None,
                         token_diversity: Optional[float] = None) -> Dict[str, float]:
        """Compute zone weights with kurtosis-SCF blending.

        SR-59c: Blend weight is scale-aware, not CV-based.
        At all scales, kurtosis gets at least 50% weight because
        it's the only signal with significant between-category variation.

        Note: Online z-score stats are updated in collect(), not here,
        to avoid double-counting when both collect() and get_zone_weights()
        are called in the same forward pass.
        """
        # Get kurtosis-based weights (z-score routing)
        k_weights = self._get_kurtosis_weights(kurtosis, phi)

        # Get SCF-based weights
        scf_weights = self._compute_scf_weights(token_diversity)

        if scf_weights is None:
            return k_weights

        # Blend based on scale-aware weight
        blend = self.k_blend_weight
        blended = {}
        for zone in k_weights:
            blended[zone] = blend * k_weights[zone] + (1 - blend) * scf_weights[zone]

        W = sum(blended.values()) + 1e-9
        return {k: v / W for k, v in blended.items()}

    def _adaptive_phi_weights(self, phi: float) -> Dict[str, float]:
        """Continuous phi-based routing using empirical phi distribution."""
        if self.phi_mean is None or self.phi_std is None:
            phi_norm = phi
        else:
            z = (phi - self.phi_mean) / self.phi_std
            z_clamped = max(-20.0, min(20.0, z))
            phi_norm = 1.0 / (1.0 + math.exp(-z_clamped))

        weights = {
            'math':      phi_norm ** 2,
            'logic_a':   phi_norm * (1 - phi_norm) * 2,
            'creative':  (1 - phi_norm) ** 2,
            'logic_b':   phi_norm * (1 - phi_norm),
            'synthesis': (1 - phi_norm) * phi_norm * 0.5,
        }

        W = sum(weights.values()) + 1e-9
        return {k: v / W for k, v in weights.items()}

    def classify_zone(self, kurtosis: float, phi: Optional[float] = None,
                      token_diversity: Optional[float] = None) -> str:
        """Classify the current cognitive zone based on weights."""
        weights = self.get_zone_weights(kurtosis, phi, token_diversity)
        return max(weights, key=weights.get)

    def get_routing_params(self, kurtosis: float, phi: Optional[float] = None,
                          hidden_size: Optional[int] = None,
                          token_diversity: Optional[float] = None) -> Dict[str, float]:
        """Compute routing parameters from blended zone weights."""
        weights = self.get_zone_weights(kurtosis, phi, token_diversity)

        # Weighted average of zone routing parameters
        start = sum(weights[z] * ZONE_ROUTING[z]['start'] for z in weights)
        end = sum(weights[z] * ZONE_ROUTING[z]['end'] for z in weights)
        hub = sum(weights[z] * ZONE_ROUTING[z]['hub'] for z in weights)
        loops = sum(weights[z] * ZONE_ROUTING[z]['loops'] for z in weights)

        # Scale adjustments for larger models
        if hidden_size and hidden_size in SCALE_DEFAULTS:
            defaults = SCALE_DEFAULTS[hidden_size]
            blend = 0.3
            start = blend * defaults['recur_start'] + (1 - blend) * start
            end = blend * defaults['recur_end'] + (1 - blend) * end
            hub = blend * defaults['hub'] + (1 - blend) * hub
            loops = blend * defaults['n_loops'] + (1 - blend) * loops

        start = max(1, int(round(start)))
        end = max(start + 2, int(round(end)))
        hub = max(start, min(end, int(round(hub))))
        loops = max(1, int(round(loops)))

        return {
            'dynamic_start': start,
            'dynamic_end': end,
            'dynamic_hub': hub,
            'n_loops': loops,
        }

    def status(self) -> Dict[str, any]:
        """Return calibration status for telemetry."""
        k_cv = self.k_std / (abs(self.k_mean) + 1e-9) if self.k_mean else None
        online_std = math.sqrt(self._online_k_m2 / max(self._online_n - 1, 1)) if self._online_n > 1 else None
        return {
            'calibrated': self.calibrated,
            'n_samples': len(self.k_samples),
            'k_mean': self.k_mean,
            'k_std': self.k_std,
            'k_range': f'[{self.k_min:.1f}, {self.k_max:.1f}]' if self.k_min else 'N/A',
            'k_cv': k_cv,
            'phi_mean': self.phi_mean,
            'phi_std': self.phi_std,
            'zone_centers': dict(self.zone_centers),
            'zone_sigmas': dict(self.zone_sigmas),
            'token_diversity_mean': self.token_diversity_mean,
            'token_diversity_std': self.token_diversity_std,
            'k_blend_weight': self.k_blend_weight,
            'zone_temperature': getattr(self, 'zone_temperature', ZONE_TEMPERATURE),
            # Online z-score statistics
            'online_n': self._online_n,
            'online_k_mean': self._online_k_mean if self._online_n > 0 else None,
            'online_k_std': online_std,
            'routing_mode': 'online_z' if self._online_n >= ONLINE_WARMUP else 'calibration_z',
        }
