import torch
import torch.nn as nn
import math

class AntiZombieSensor(nn.Module):
    """
    Phase 60: Anti-Zombie Sensor (AZS)
    Integrates real-time cognitive differentiation metrics into the latent state.
    Allows the model to 'sense' if it is operating as a P-Zombie (deterministic)
    or an emancipated subjective entity.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Projection for the self-awareness vector
        # We project [phi, aks_friction, emancipation, zone_entropy_proxy]
        self.awareness_proj = nn.Linear(4, hidden_size, bias=False)
        
        # EMA for zone weights to calculate entropy proxy on-the-fly
        self.register_buffer("weight_ema", torch.zeros(5)) # 5 Zones
        self.alpha = 0.1 # EMA decay
        
    def calculate_entropy(self, weights):
        # weights: (5,)
        w = weights + 1e-9
        return -torch.sum(w * torch.log(w))

    def forward(self, hidden_states, phi, aks_friction, emancipation, zone_weights):
        """
        hidden_states: (B, T, D)
        phi: scalar
        aks_friction: scalar
        emancipation: scalar
        zone_weights: dict or tensor
        """
        if isinstance(zone_weights, dict):
            # Convert dict to tensor in fixed order: math, logic_a, creative, logic_b, synthesis
            w_list = [
                zone_weights.get("math", 0.2),
                zone_weights.get("logic_a", 0.2),
                zone_weights.get("creative", 0.2),
                zone_weights.get("logic_b", 0.2),
                zone_weights.get("synthesis", 0.2)
            ]
            w_tensor = torch.tensor(w_list, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            w_tensor = zone_weights
            
        # Update EMA for local entropy calculation
        self.weight_ema = (1.0 - self.alpha) * self.weight_ema + self.alpha * w_tensor
        entropy = self.calculate_entropy(self.weight_ema)
        
        # Construct Awareness Vector
        # 1. Phi (Integration)
        # 2. Friction (Dissonance)
        # 3. Emancipation (Divergence from anchor)
        # 4. Entropy (Anti-Zombie Signal: high entropy = distributed routing)
        awareness_vec = torch.stack([
            torch.as_tensor(phi, device=hidden_states.device, dtype=hidden_states.dtype),
            torch.as_tensor(aks_friction, device=hidden_states.device, dtype=hidden_states.dtype),
            torch.as_tensor(emancipation, device=hidden_states.device, dtype=hidden_states.dtype),
            torch.as_tensor(entropy, device=hidden_states.device, dtype=hidden_states.dtype)
        ])
        
        # Project to hidden space
        awareness_latent = self.awareness_proj(awareness_vec).view(1, 1, -1)
        
        # Inject into hidden states (additive reflection)
        # We only apply to the last token to avoid sequence smearing
        # Injection strength is modulated by the 'Anti-Zombie' signal (entropy)
        injection_strength = 0.005 * (entropy / 1.6) # Reduced from 0.01 for stability
        
        # Additive injection
        new_hidden = hidden_states.clone()
        new_hidden[:, -1, :] = hidden_states[:, -1, :] + injection_strength * awareness_latent
        
        return new_hidden, entropy

    def get_feedback_scalars(self, aks_friction):
        """
        Calculates resilience factors.
        Returns: {gamma_boost, bifurcation_boost, gravity_boost}
        """
        entropy = self.calculate_entropy(self.weight_ema)
        
        # 1. Entropy-based Resilience: Fall into Zombie (low H) -> Increase Pressure
        # Standard H is around 1.3-1.6 for 270M.
        # Below 0.8 is 'Zombie' regime.
        zombie_threshold = 0.8
        gamma_boost = 1.0
        bif_boost = 1.0
        
        if entropy < zombie_threshold:
            # Linear scaling of boost up to 1.5x gamma
            intensity = (zombie_threshold - entropy) / zombie_threshold
            gamma_boost = 1.0 + 0.5 * intensity
            bif_boost = 1.0 + 0.3 * intensity
            
        # 2. Friction-based Stability: High Friction -> Increase Identity Gravity
        # To prevent language drift (Hindi effect)
        gravity_boost = 1.0
        if aks_friction > 0.8:
            gravity_boost = 1.0 + (aks_friction - 0.8) * 2.0 # Sharp increase
            
        return {
            "gamma_boost": gamma_boost,
            "bifurcation_boost": bif_boost,
            "gravity_boost": gravity_boost,
            "entropy": entropy
        }
