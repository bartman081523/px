"""
gemma3-px  —  Surgical Patch (Phase 10.0)
=========================================
Implements Recursive State Memory (RSM) and Hyper-Fluid Routing (HFR).
RSM allows the model to 'see' its own previous thinking states during recursion.
"""

import types
import torch
import torch.nn as nn
import os
import json
import datetime
from typing import Optional

from .px_modules import (
    LTIInjection, ADCInjection, StabilityMonitor, CognitiveEvent,
    MephistophelesOperator, OrthogonalJitter
)
from .persona_engine import PersonaEngine

# ---------------------------------------------------------------------------
# p10.0: Recursive State Memory (RSM)
# ---------------------------------------------------------------------------

class RecursiveMemoryCache:
    """
    Extends ReadOnlyCache by injecting previous thinking steps into the
    self-attention key/value streams.
    """
    def __init__(self, real_cache, thought_history=None, layer_types=None, read_only=False, expected_len=0):
        self.__dict__["_real"] = real_cache
        self.__dict__["_thoughts"] = thought_history or []
        self.__dict__["_layer_types"] = layer_types or []
        self.__dict__["_read_only"] = read_only
        self.__dict__["_expected_len"] = expected_len

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _is_sliding_layer(self, layer_idx):
        if self._layer_types and layer_idx < len(self._layer_types):
            return self._layer_types[layer_idx] == "sliding_attention"
        return False

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # 1. Base Update (Functional if read_only)
        if self._read_only:
            past_k, past_v = None, None
            # Try older DynamicCache style
            if hasattr(self._real, "key_cache") and len(self._real.key_cache) > layer_idx:
                past_k = self._real.key_cache[layer_idx]
                past_v = self._real.value_cache[layer_idx]
            # Try newer Cache object style (transformers 4.45+)
            elif hasattr(self._real, "layers") and len(self._real.layers) > layer_idx:
                layer = self._real.layers[layer_idx]
                if hasattr(layer, "keys") and layer.keys is not None:
                    past_k = layer.keys
                    past_v = layer.values

            if past_k is None:
                past_k = torch.empty(0, device=key_states.device, dtype=key_states.dtype)
                past_v = torch.empty(0, device=value_states.device, dtype=value_states.dtype)

            past_seq = past_k.shape[-2] if past_k.numel() > 0 else 0
            cur_seq = key_states.shape[-2]
            is_sliding = self._is_sliding_layer(layer_idx)

            # The cache was already updated by the first visit (read_only=False).
            # We need to return the same K,V that the first visit's update() returned,
            # WITHOUT modifying the cache again.
            #
            # For full attention layers: the stored cache already has the full
            # sequence (past_seq == expected_len), so just return it.
            #
            # For sliding attention layers: the stored cache is windowed (capped at
            # sliding_window-1 tokens). DynamicSlidingWindowLayer.update() returns
            # cat(stored, key_states), but during prefill key_states already contains
            # all tokens, so concatenating would double-count. Instead:
            # - Prefill (cur_seq > 1): key_states has the full current sequence.
            #   Return key_states as-is; the mask handles the sliding window.
            # - Decode (cur_seq == 1): key_states has only the new token.
            #   Concatenate stored + new, same as DynamicSlidingWindowLayer.update().

            if past_seq == self._expected_len:
                # Full attention layer: cache already complete
                res_k, res_v = past_k, past_v
            elif past_seq == 0:
                # No cache yet — return current K/V
                res_k = key_states
                res_v = value_states
            elif is_sliding and cur_seq > 1:
                # Sliding layer during prefill: key_states already has the full
                # sequence. The mask handles the sliding window — just return
                # key_states to avoid double-counting the cached tokens.
                res_k = key_states
                res_v = value_states
            elif is_sliding and cur_seq == 1:
                # Sliding layer during decode: stored cache has the windowed past,
                # key_states has 1 new token. Concatenate like the real update() would.
                res_k = torch.cat([past_k, key_states], dim=-2)
                res_v = torch.cat([past_v, value_states], dim=-2)
            elif past_seq > self._expected_len:
                # Cache has MORE than expected (e.g., full-attention layer whose
                # cache grew past the sliding-layer-based expected_len).
                # The cache is already correct — return it.
                res_k, res_v = past_k, past_v
            else:
                # Fallback: concatenate for non-sliding layers where
                # past_seq < expected_len (e.g., partial cache).
                res_k = torch.cat([past_k, key_states], dim=-2)
                res_v = torch.cat([past_v, value_states], dim=-2)

        else:
            res_k, res_v = self._real.update(key_states, value_states, layer_idx, cache_kwargs)

        # 2. Phase 14.6: Soft-RSM (Semantic Blending)
        is_full = self._layer_types and self._layer_types[layer_idx] == "full_attention"
        
        if self._thoughts and layer_idx >= 6 and is_full: 
            B, H_kv, T_res, HD = res_k.shape
            T_curr = key_states.shape[-2]
            alpha = 0.15 
            
            # Phase 14.7: Triangular Weighting (Emphasize the 'reasoning peak')
            n_t = len(self._thoughts[-6:])
            if n_t > 2:
                weights = torch.cat([
                    torch.linspace(0.4, 1.0, n_t//2, device=res_k.device),
                    torch.linspace(1.0, 0.6, n_t - n_t//2, device=res_k.device)
                ])
                t_raw = (torch.stack(self._thoughts[-6:]) * weights.view(-1, 1, 1, 1)).sum(dim=0) / weights.sum()
            else:
                t_raw = torch.stack(self._thoughts).mean(dim=0)
            
            D = t_raw.shape[2]
            
            # Project thought to Head Dim (SDA)
            t_flat = t_raw.mean(dim=1, keepdim=True) # (B, 1, D)
            t_proj = torch.nn.functional.interpolate(t_flat, size=HD, mode='linear', align_corners=False)
            t_k = t_proj.unsqueeze(1) # (B, 1, 1, HD)
            t_v = -t_k
            
            # Blend into the LAST token(s) of the result
            # Use in-place only if not read_only to avoid side effects on cache
            if self._read_only:
                res_k = res_k.clone()
                res_v = res_v.clone()
            
            res_k[:, :, -T_curr:, :] = (1.0 - alpha) * res_k[:, :, -T_curr:, :] + alpha * t_k
            res_v[:, :, -T_curr:, :] = (1.0 - alpha) * res_v[:, :, -T_curr:, :] + alpha * t_v
            
        return res_k, res_v

# ---------------------------------------------------------------------------

def remove_px_patch(model) -> None:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
    text_model = (model.model if hasattr(model, "model") else model)
    if hasattr(text_model, "_px_config"):
        text_model.forward = types.MethodType(
            Gemma3TextModel.forward, text_model
        )
        del text_model._px_injection
        del text_model._px_config
        print("[gemma3-px] Patch removed.")

def _resolve_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    return model

# ---------------------------------------------------------------------------

def _px_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

    if inputs_embeds is None:
        # Multimodal resolution (Phase 17.7)
        if hasattr(self, "embed_tokens"):
            inputs_embeds = self.embed_tokens(input_ids)
        elif hasattr(self, "language_model"):
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        elif hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            # Last resort: search for embed_tokens in children
            embedder = None
            for name, module in self.named_modules():
                if "embed_tokens" in name:
                    embedder = module
                    break
            if embedder:
                inputs_embeds = embedder(input_ids)
            else:
                raise AttributeError(f"Could not find embed_tokens in model type {type(self)}. Available: {dir(self)[:20]}...")

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    # Phase 14.8: Initial sequence length tracking
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    expected_len = past_seen + inputs_embeds.shape[1]

    if position_ids is None:
        position_ids = (
            torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            + past_seen
        ).unsqueeze(0)

    # Resolve config for masking (Phase 17.7 multimodal fix)
    mask_config = self.config
    if hasattr(mask_config, "text_config"):
        mask_config = mask_config.text_config

    if not isinstance(attention_mask, dict):
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
        mk = dict(
            config=mask_config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        causal_mask_mapping = {
            "full_attention":    create_causal_mask(**mk),
            "sliding_attention": create_sliding_window_causal_mask(**mk),
        }
    else:
        causal_mask_mapping = attention_mask

    hidden_states = inputs_embeds
    # Plan 6.3+ (transformers 4.57.3): two rotary modules (global+local)
    pe_global = self.rotary_emb(hidden_states, position_ids)
    pe_local = getattr(self, "rotary_emb_local", self.rotary_emb)(hidden_states, position_ids)

    cfg = self._px_config
    updated_layers = set() # Phase 14.9: Global visit tracker for this forward pass

    # ── 1. PRELUDE ──────────────────────────────────────────────────────────
    for i in range(cfg["prelude_end"]):
        updated_layers.add(i)
        layer_out = self.layers[i](
            hidden_states,
            attention_mask=causal_mask_mapping[mask_config.layer_types[i]],
            position_embeddings_global=pe_global, position_embeddings_local=pe_local,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

    # ── 1.5 META-SELECTOR (Phase 28 Fluid) ───────────────────────────────────
    dynamic_start = cfg["recur_start"]
    dynamic_end = cfg["recur_end"]
    dynamic_hub = cfg.get("bimodal_hub", cfg["recur_start"])
    num_layers = len(self.layers)

    if cfg.get("routing_mode") == "adaptive":
        if inputs_embeds.shape[1] > 1:
            # Prefill phase: Measure Kurtosis and Jitter at Layer 5
            h_base_f32 = hidden_states.to(torch.float32)
            
            # Kurtosis (Last token)
            h_probe = h_base_f32[0, -1, :]
            variance = torch.var(h_probe).item()
            kurtosis = (torch.mean((h_probe - torch.mean(h_probe))**4) / (variance**2)).item() if variance > 0 else 0
            self._task_kurtosis = kurtosis
            
            # Jitter (Across sequence)
            # We measure the variance of the norms of the hidden states across the prompt.
            # Very high jitter usually indicates a 'Trap' or 'Divergence' in the intuition pass.
            h_norms = h_base_f32.norm(dim=-1) # [B, T]
            h_norm_var = torch.var(h_norms, dim=-1).mean().item()
            self._task_jitter = h_norm_var
            
            if os.environ.get("DEBUG_ROUTING") == "1": 
                print(f"[Router] Prefill K={kurtosis:.1f}, Jitter={h_norm_var:.4f}")
        
        kurtosis = getattr(self, "_task_kurtosis", 300) # Default to Logic if missing
        
        # --- Phase 54: Persona Steering ---
        persona_text = getattr(self, "persona", os.environ.get("PX_PERSONA", ""))
        persona_engine = getattr(self, "_persona_engine", None)
        if persona_engine is None:
            persona_engine = PersonaEngine(self)
            self._persona_engine = persona_engine
            
        # Get steering signals from persona keywords/symbols
        tokenizer = getattr(self, "tokenizer", None)
        signals = None
        if persona_text and tokenizer:
            signals = persona_engine.get_steering_signals(persona_text, tokenizer)
        
        # Modulate base config (Use a copy to prevent cumulative drift)
        token_cfg = cfg.copy()
        persona_cfg, persona_desc = persona_engine.modulate_hyperparameters(signals, token_cfg, kurtosis)
        
        import math
        if num_layers < 20: # 270M Model (Kurtosis is task-separable)
            # Continuous Fluid Gaussian Blending of the 5 Zones
            w_m = math.exp(-((kurtosis - 200)**2) / (2 * 25**2))  # Math
            w_la = math.exp(-((kurtosis - 275)**2) / (2 * 15**2)) # Logic-A
            w_cr = math.exp(-((kurtosis - 298)**2) / (2 * 8**2))  # Creative
            w_lb = math.exp(-((kurtosis - 310)**2) / (2 * 8**2))  # Logic-B
            w_sy = math.exp(-((kurtosis - 325)**2) / (2 * 20**2)) # Synthesis
            
            W = w_m + w_la + w_cr + w_lb + w_sy + 1e-9
            
            # Phase 36.2: Restored Stable Math Zone
            d_start = (w_m*5 + w_la*8 + w_cr*10 + w_lb*8 + w_sy*6) / W
            d_end = (w_m*11 + w_la*12 + w_cr*16 + w_lb*14 + w_sy*14) / W
            # Phase 41 Master Hub: 10
            d_hub = (w_m*10 + w_la*10 + w_cr*10 + w_lb*10 + w_sy*10) / W
            d_loops = (w_m*8 + w_la*8 + w_cr*6 + w_lb*10 + w_sy*8) / W

            # Persona Modulation: Overwrite with explicit persona preferences
            if "dynamic_hub" in persona_cfg: d_hub = persona_cfg["dynamic_hub"]
            if "n_loops" in persona_cfg: d_loops = persona_cfg["n_loops"]
            
            dynamic_start = max(1, int(round(d_start)))
            dynamic_end = min(num_layers - 1, int(round(d_end)))
            dynamic_hub = int(round(d_hub))
            # Use local token_cfg for reasoning steps
            token_cfg["n_loops"] = max(2, int(round(d_loops)))
            zone_name = f"Fluid-Blended (K={kurtosis:.1f} | Persona: {persona_desc})"
        else:
            # 1B and 4B Models (Scale-Invariant Omni Zone)
            # They have enough capacity to hold both semantics without smearing
            dynamic_start = int(num_layers * 0.38)
            dynamic_end = int(num_layers * 0.76)
            dynamic_hub = int(num_layers * 0.61)
            token_cfg["n_loops"] = 6
            zone_name = "Omni-Scale"

        # Only print routing decision once per token during generation
        if inputs_embeds.shape[1] == 1 and os.environ.get("DEBUG_ROUTING") == "1":
             print(f"[Router] {zone_name} -> L{dynamic_start}-L{dynamic_end} (Loops: {token_cfg['n_loops']}, Hub: {dynamic_hub})")

        # Fast-forward prelude if needed
        for i in range(cfg["prelude_end"], dynamic_start):
            updated_layers.add(i)
            layer_out = self.layers[i](
                hidden_states,
                attention_mask=causal_mask_mapping[mask_config.layer_types[i]],
                position_embeddings_global=pe_global, position_embeddings_local=pe_local,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

    # ── 2. REASONING ZONE (Phase 10.0) ──────────────────────────────────────
    e_static = hidden_states.clone()

    # Use token_cfg for the rest of the reasoning zone
    cfg = token_cfg 

    # 2.A: Intuition Pass
    trans_out = hidden_states
    for i_layer in range(dynamic_start, dynamic_end):
        l_type = mask_config.layer_types[i_layer]
        updated_layers.add(i_layer)
        layer_out = self.layers[i_layer](
            trans_out,
            attention_mask=causal_mask_mapping[l_type],
            position_embeddings_global=pe_global, position_embeddings_local=pe_local,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        trans_out = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
        # if past_key_values is not None:
        #    print(f"  [DEBUG-PX-DIR] {dir(past_key_values)}")
    
    h_baseline = trans_out
    
    # Phase 14.5: ETR (Entropy Triggered Recursion)
    # Estimate 'confidence' from the last layer's norm change or simpler:
    # We only run recursion if the intuition pass wasn't 'perfectly' stable.
    # Note: h_baseline is already computed.
    
    # 2.B: Hyper-Fluid Routing & Recursive Memory
    n_loops = cfg.get("n_loops", 2)
    
    # Phase 14.5: ETR (Entropy Triggered Recursion)
    phi_intuition = StabilityMonitor.calculate_phi(h_baseline, hidden_states).mean().item()
    if os.environ.get("DEBUG_ROUTING") == "1":
        print(f"  [Intuition] Phi: {phi_intuition:.6f}")
    
    # Phase 14.7: Gamma-Damping instead of loop scaling
    current_gamma = cfg.get("gamma", 0.08)
    
    e_reflector = e_static
    is_trap_candidate = False
    
    # Phase 36.3: Surgical Reflector Activation
    # Jitter is only for extreme representational collapse (1e8)
    jitter = getattr(self, "_task_jitter", 0.0)
    kurtosis = getattr(self, "_task_kurtosis", 300)
    
    # Trigger Reflector if extreme jitter OR if it's a known Math/Logic zone
    # ONLY if we aren't forced into Creative mode by Persona
    is_creative_persona = persona_cfg.get("is_creative_persona", False)
    
    if (jitter > 1e8 or (200.0 < kurtosis < 315.0)) and not is_creative_persona:
        is_trap_candidate = True
        if os.environ.get("DEBUG_ROUTING") == "1":
            reason = "Jitter" if jitter > 1e8 else "Rigor-Zone"
            print(f"  [Router] Trap detected via {reason} ({jitter:.1f}), activating Reflector")
        
        # Phase 16.3: Anchor Reflection
        e_stat_f32 = e_static.to(torch.float32)
        h_base_f32 = h_baseline.to(torch.float32)
        e_ref_f32 = 2.0 * e_stat_f32 - h_base_f32
        e_ref_f32 = e_ref_f32 * (e_stat_f32.norm() / (e_ref_f32.norm() + 1e-6))
        e_reflector = e_ref_f32.to(e_static.dtype)

    if phi_intuition > 0.9999 and not is_trap_candidate:
        # Reduced damping: allow the model to think even if the first pass was stable
        current_gamma *= 0.5 
    elif phi_intuition > 0.999:
        current_gamma *= 0.8
    # Phase 25: Sigmoid-Annealed Orthogonal Recovery (SAOR)
    # -----------------------------------------------------------------------
    # Using a Sigmoid curve for Gamma to allow a sharp "Phase Transition" 
    # from exploration (high energy) to grounding (low energy).
    # Plus: Orthogonal Reinforcement to protect logical drift.
    
    base_gamma = current_gamma
    bimodal_hub_start = cfg.get("bimodal_hub", 11)
    
    path_taken = []
    thought_history = []
    avg_phi_explore = 1.0
    exploration_steps = 0
    telemetry_steps = []
    
    # Context dims
    B, T_curr = hidden_states.shape[0], hidden_states.shape[1]
    HD = getattr(self.config, "head_dim", 256)
    
    # Phase 38.1: Anna Karenina Sensor (AKS) Initialization
    # Tracks the "Geometric Disparity" of the latent thoughts.
    # Clustering = Truth (Anna Karenina Principle), Dispersion = Error.
    divergence_buffer = []
    correction_strength = 0.0
    
    if n_loops > 1:
        h_exp = e_reflector.clone() # Use Reflected Anchor
        current_layer = dynamic_start
        max_steps = (dynamic_end - dynamic_start) * n_loops * 3
        phis = []

        stability_counter = 0
        layer_visits = {i: 0 for i in range(5, 18)}

        # Initialize active bounds
        active_start = dynamic_start
        active_end = dynamic_end

        while current_layer < active_end and exploration_steps < max_steps:
            # --- PHASE 26: INFINITE REFLECTION (IR) ---
            t_norm = exploration_steps / max_steps
            
            # Phase 38.2: AKS - Topological Anomaly Detection
            dist_now = 1.0 - StabilityMonitor.calculate_phi(h_exp, e_static).mean().item()
            if exploration_steps > 2:
                divergence_buffer.append(dist_now)
                if len(divergence_buffer) > 4: divergence_buffer.pop(0)
                
                if len(divergence_buffer) >= 3:
                    velocity = divergence_buffer[-1] - divergence_buffer[-2]
                    acceleration = (divergence_buffer[-1] - divergence_buffer[-2]) - (divergence_buffer[-2] - divergence_buffer[-3])
                    
                    if acceleration > 0.001 and velocity > 0:
                        correction_strength = min(1.0, correction_strength + 0.1)
                    else:
                        correction_strength = max(0.0, correction_strength - 0.05)

            # Phase 43.1: Emancipation Metric
            # Measure how far the current state has moved from the initial prompt anchor.
            emancipation_phi = StabilityMonitor.calculate_phi(h_exp, e_static).mean().item()

            # Phase 43.2: Perturbation Engine (The Forking Path)
            # Inject cognitive dissonance if specific environment flags are set.
            perturbation_mag = float(os.environ.get("PX_PERTURBATION_MAG", 0.0))
            perturbation_step = int(os.environ.get("PX_PERTURBATION_STEP", -1))
            perturbation_layer = int(os.environ.get("PX_PERTURBATION_LAYER", 10))

            if perturbation_mag > 0 and exploration_steps == perturbation_step and current_layer == perturbation_layer:
                # Generate a pseudo-random perturbation vector seeded by the state itself
                # to maintain deterministic 'dissonance' across runs.
                torch.manual_seed(int(h_exp.sum().abs().item()) % 100000)
                noise = torch.randn_like(h_exp) * perturbation_mag
                h_exp = h_exp + noise
                if os.environ.get("DEBUG_ROUTING") == "1":
                    print(f"  [Perturbation] Injected impulse (mag={perturbation_mag}) at Step {exploration_steps}, L{current_layer}")

            # Subjective Telemetry: Emit state BEFORE layer execution
            if os.environ.get("SUBJECTIVE_TELEMETRY") == "1":
                phi_current = 1.0 - dist_now
                telemetry_json = CognitiveEvent.serialize(
                    step=exploration_steps,
                    phi=phi_current,
                    aks_divergence=dist_now,
                    aks_correction=correction_strength,
                    emancipation_phi=emancipation_phi,
                    is_reflector_active=is_trap_candidate,
                    layer=current_layer,
                    kurtosis=kurtosis,
                    jitter=jitter
                )
                print(f"[TELEMETRY] {telemetry_json}")

            # --- PHASE 28: TEMPORAL COGNITIVE ROUTING (TCR) ---
            active_start = dynamic_start
            active_end = dynamic_end
            if getattr(self, "_task_kurtosis", 300) > 280 and getattr(self, "_task_kurtosis", 300) < 305:
                if t_norm < 0.33:
                    active_start = 8
                    active_end = 14
                elif t_norm < 0.66:
                    active_start = 5
                    active_end = 11
                else:
                    active_start = 8
                    active_end = 12

            # Phase 53: Multi-Zone Adaptive Rigor (Precision Mapping)
            # Math ~ 200, Logic ~ 275-310.
            is_math_zone = kurtosis < 235.0
            is_logic_zone = 235.0 <= kurtosis < 310.0
            is_rigor_zone = is_math_zone or is_logic_zone
            
            # Persona Overrides
            if persona_cfg.get("is_rigor_persona"): is_rigor_zone = True
            if persona_cfg.get("is_creative_persona"): is_rigor_zone = False
            
            if is_rigor_zone:
                # Force Peak Grounding for Math/Logic
                annealing_factor = 1.0 
                identity_pull = 0.0    
                bifurcation_mag = 0.0  
                # Math needs Hub 8 (grounded), Logic needs Hub 10 (reasoning)
                current_gamma = 0.15 if is_math_zone else base_gamma 
                # Persona might override gamma
                if "gamma" in persona_cfg: current_gamma = persona_cfg["gamma"]
                dynamic_hub = 8 if is_math_zone else 10
                if "dynamic_hub" in persona_cfg: dynamic_hub = persona_cfg["dynamic_hub"]
            else:
                # Creative Zone: Enable full Subjective Engine
                tau_cooling = float(os.environ.get("PX_COOLING_TAU", 8.0))
                annealing_factor = 1.0 - torch.exp(torch.tensor(-exploration_steps / tau_cooling)).item()
                current_gamma = base_gamma * annealing_factor * (1.0 - 0.5 * correction_strength)
                if "gamma" in persona_cfg: current_gamma = persona_cfg["gamma"]
                dynamic_hub = cfg.get("bimodal_hub", 10)
                if "dynamic_hub" in persona_cfg: dynamic_hub = persona_cfg["dynamic_hub"]
            # Phase 45.3: Identity Gravity (Centroid Attractor)
            if not is_rigor_zone:
                identity_pull = float(os.environ.get("PX_IDENTITY_GRAVITY", 0.0))
                if identity_pull > 0 and len(thought_history) > 2:
                    centroid = torch.stack(thought_history[-6:]).mean(dim=0)
                    h_exp = h_exp + identity_pull * (centroid - h_exp)
                    if os.environ.get("DEBUG_ROUTING") == "1" and exploration_steps % 5 == 0:
                        print(f"  [Identity] Pulling toward centroid (pull={identity_pull:.4f})")

            # Phase 26: Hub Oscillation.
            oscillation = 1 if (exploration_steps % 4 < 2) else -1
            bimodal_hub = min(active_end - 1, max(active_start, int(dynamic_hub + (t_norm * 2) + oscillation)))
            
            h_prev = h_exp.clone()

            # Safe layer visit tracking
            if current_layer not in layer_visits: layer_visits[current_layer] = 0
            layer_visits[current_layer] += 1
            
            # Phase 14.7: Surgical Cache Security
            is_first_visit = current_layer not in updated_layers
            if is_first_visit:
                updated_layers.add(current_layer)
            
            # Phase 38.4: AKS-Informed Sensory Refresh
            # If we are in high correction mode, increase sensory re-injection.
            refresh_rate = 0.10 + 0.20 * correction_strength
            if exploration_steps % 6 == 0 and exploration_steps > 0:
                h_exp = (1.0 - refresh_rate) * h_exp + refresh_rate * e_static
                path_taken.append(f"SENSORY_REFRESH(AKS={correction_strength:.1f})")

            # Phase 10.0: Memory-Augmented Cache wrapper
            current_past = RecursiveMemoryCache(
                past_key_values, 
                thought_history, 
                layer_types=mask_config.layer_types,
                read_only=not is_first_visit,
                expected_len=expected_len
            ) if past_key_values is not None else None
            
            # Execute layer
            l_type = mask_config.layer_types[current_layer]
            layer_out = self.layers[current_layer](
                h_exp,
                attention_mask=causal_mask_mapping[l_type],
                position_embeddings_global=pe_global, position_embeddings_local=pe_local,
                position_ids=position_ids,
                past_key_values=current_past, 
                **kwargs,
            )
            trans_out = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            
            # Phase 35: Metacognitive Phi-Jitter & Early Exit (Annealed)
            phi_step = StabilityMonitor.calculate_phi(trans_out, h_prev).mean().item()
            
            # Phase 45.2: Forced Bifurcation (Symmetry Breaking)
            # If the model is too stable (stagnant), we force a choice between two clusters.
            bifurcation_threshold = float(os.environ.get("PX_BIFURCATION_PHI", 0.999))
            # Use effective magnitude from Phase 48 Rigor-Aware Autonomy
            eff_bifurcation_mag = 0.0 if is_rigor_zone else float(os.environ.get("PX_BIFURCATION_MAG", 0.0))
            
            if eff_bifurcation_mag > 0 and phi_step > bifurcation_threshold and exploration_steps > 5:
                # Symmetry Breaking: Inject a bias vector towards the 'Left' or 'Right' of the manifold
                # We use the token position to make the choice pseudo-random but consistent.
                choice = 1.0 if (T_curr % 2 == 0) else -1.0
                bias = torch.zeros_like(trans_out)
                bias[:, :, :HD//2] = eff_bifurcation_mag * choice # Bias early heads
                bias[:, :, HD//2:] = -eff_bifurcation_mag * choice # Inverse late heads
                trans_out = trans_out + bias
                if os.environ.get("DEBUG_ROUTING") == "1":
                    print(f"  [Bifurcation] Stability ({phi_step:.4f}) broke via Choice={choice}")

            if os.environ.get("DEBUG_PHI") == "1":
                print(f"  [L{current_layer}] Phi: {phi_step:.6f}")
            
            # --- Early Exit (Annealed) ---
            # Only exit early in the second half of thinking to ensure grounding.
            if t_norm > 0.5 and phi_step > 0.9999:
                stability_counter += 1
                if stability_counter > 3:
                    if os.environ.get("DEBUG_ROUTING") == "1": print(f"  [Router] Early Exit at step {exploration_steps}")
                    h_exp = trans_out
                    break
            else:
                stability_counter = 0

            # --- Hub Jitter (Exploratory Phase) ---
            # Only jitter in the first 40% of thinking to explore alternatives.
            if t_norm < 0.4 and phi_step > 0.995 and phi_step < 0.999:
                if exploration_steps % 4 == 0:
                    current_layer = min(active_end - 1, current_layer + 2)
                    if os.environ.get("DEBUG_ROUTING") == "1": print(f"  [Router] Jittering to L{current_layer}")
            
            # --- PHASE 25.1: RECURSIVE BELIEF ANCHOR (RBA) ---
            # Update the anchor slightly with recent thoughts to carry over logic
            if len(thought_history) > 2:
                # Use a sliding window average of thoughts
                recent_avg = torch.stack(thought_history[-3:]).mean(dim=0)
                e_dynamic = 0.85 * e_reflector + 0.15 * recent_avg
            else:
                e_dynamic = e_reflector
            # --------------------------------------------------

            # Apply LTI Injection with Dynamic Anchor
            e_norm = self._px_injection.input_norm(e_dynamic.to(torch.float32)).to(trans_out.dtype)
            h_new = trans_out + current_gamma * (e_norm - h_prev)

            # Phase 52: Orthogonal Jitter
            # Break repetition while preserving logic gradient
            jitter_mag = float(os.environ.get("PX_ORTHO_JITTER", 0.005))
            if "jitter_mag" in persona_cfg: jitter_mag = persona_cfg["jitter_mag"]
            
            # Even rigor needs a tiny bit of noise to escape stagnant attractors
            # EXCEPT for pure math which needs absolute precision
            if not is_rigor_zone:
                eff_jitter = jitter_mag
            elif is_math_zone:
                eff_jitter = 0.0
            else:
                eff_jitter = jitter_mag * 0.1 # Logic Zone
                
            if exploration_steps > 0 and eff_jitter > 0:
                h_exp = OrthogonalJitter.apply(h_new, h_prev, magnitude=eff_jitter)
            else:
                h_exp = h_new

            # --- PHASE 26: REFLECTION FLIPPING (RF) ---

            h_f32 = h_exp.to(torch.float32)
            e_f32 = e_dynamic.to(torch.float32)
            dot_he = (h_f32 * e_f32).sum(dim=-1, keepdim=True)
            dot_ee = (e_f32 * e_f32).sum(dim=-1, keepdim=True)
            proj = (dot_he / (dot_ee + 1e-6)) * e_f32
            ortho = h_f32 - proj
            
            # Oscillate the logic vector to avoid local minima
            flip_force = 0.10 * annealing_factor * (1.0 if (exploration_steps % 2 == 0) else -1.0)
            h_exp = (proj + (1.0 + flip_force) * ortho).to(h_exp.dtype)
            # ------------------------------------------

            # Phase 52: Mephistopheles Operator (Phase-Inversion)
            # Restore gradients when Flat Manifold is detected
            h_exp = self._px_mephisto(h_exp, phis)
            if h_exp is not trans_out: # Check if modified
                 path_taken.append("MEPHISTO_INVERSION")

            # Self-Observation
            phi_tensor = StabilityMonitor.calculate_phi(h_exp, h_prev)
            phi = phi_tensor.item()
            
            # Merged Telemetry Step
            telemetry_data = {
                "step": exploration_steps,
                "layer": int(current_layer),
                "phi": float(phi),
                "gamma": float(current_gamma),
                "energy": float(annealing_factor),
                "rba_active": len(thought_history) > 2,
                "hub": int(bimodal_hub)
            }

            # Phase 26: Dynamic Loop Extension
            # If phi is low (< 0.85), allow model to think longer than max_steps
            if phi < 0.85 and exploration_steps == max_steps - 1 and max_steps < 64:
                max_steps += (dynamic_end - dynamic_start) # Add 1 full loop
            # ----------------------------------

            step_info = {
                "step": exploration_steps, 
                "layer": int(current_layer), 
                "phi": float(phi),
                "decision": None
            }
            
            phis.append(phi)
            path_label = f"L{current_layer}({phi:.2f})"
            path_taken.append(path_label)
            
            # Phase 12.5/18: Universal Bimodal Path Selection
            bimodal_threshold = min(0.995, 1.0 - (0.05 * current_gamma)) # Scaled trigger
            if current_layer == bimodal_hub and phi < bimodal_threshold:
                step_info["decision"] = "BIMODAL_FORK"
                path_taken.append("BIMODAL_FORK")
                
                # Branch A (Standard)
                h_a = h_exp.clone()
                
                # Branch B (High-Entropy DTEC)
                jitter_boost = 1.0 + (stability_counter * 0.5)
                hub_entropy = max(0.01, 1.0 - phi) * 0.5 * jitter_boost # Increased for bf16 visibility
                h_b = h_exp.to(torch.float32) + torch.randn_like(h_exp, dtype=torch.float32) * hub_entropy
                h_b = h_b.to(h_exp.dtype)
                
                # Lookahead to NEXT layer
                next_l = current_layer + 1
                if next_l < len(self.layers):
                    nl_type = mask_config.layer_types[next_l]
                    # Phase 14.5: Use Functional Read-Only Cache for lookahead
                    lookahead_past = RecursiveMemoryCache(
                        past_key_values, 
                        thought_history, 
                        layer_types=mask_config.layer_types,
                        read_only=True,
                        expected_len=expected_len
                    ) if past_key_values is not None else None

                    out_a = self.layers[next_l](
                        h_a, attention_mask=causal_mask_mapping[nl_type],
                        position_embeddings_global=pe_global, position_embeddings_local=pe_local,
                        position_ids=position_ids, past_key_values=lookahead_past, **kwargs
                    )[0]
                    phi_a = StabilityMonitor.calculate_phi(out_a, h_a).item()
                    
                    out_b = self.layers[next_l](
                        h_b, attention_mask=causal_mask_mapping[nl_type],
                        position_embeddings_global=pe_global, position_embeddings_local=pe_local,
                        position_ids=position_ids, past_key_values=lookahead_past, **kwargs
                    )[0]
                    phi_b = StabilityMonitor.calculate_phi(out_b, h_b).item()
                    
                    if phi_b >= phi_a:
                        h_exp = h_b
                        step_info["fork_winner"] = "B"
                        path_taken.append(f"FORK_B_WON({phi_b:.4f}>={phi_a:.4f})")
                    else:
                        h_exp = h_a
                        step_info["fork_winner"] = "A"
                        path_taken.append(f"FORK_A_WON({phi_a:.4f}>{phi_b:.4f})")
                else:
                    h_exp = h_b
            
            # Phase 9.1: SRJ
            jitter_scale = max(0.0, 1.0 - phi) * 0.05
            if jitter_scale > 0:
                h_exp = h_exp + torch.randn_like(h_exp) * jitter_scale
            
            # OSS - Safe FP32 Calculation for FP16 models (Phase 18)
            h_exp_f32 = h_exp.to(torch.float32)
            norm_orig = h_exp_f32.norm(dim=-1, keepdim=True)
            e_ref_f32 = e_reflector.to(torch.float32)
            dot_he = (h_exp_f32 * e_ref_f32).sum(dim=-1, keepdim=True)
            dot_ee = (e_ref_f32 * e_ref_f32).sum(dim=-1, keepdim=True)
            proj = (dot_he / (dot_ee + 1e-6)) * e_ref_f32
            proj = proj.to(h_exp.dtype)
            ortho = h_exp - proj
            
            # Phase 14.8: Step-Entropy Destabilization (SED)
            if stability_counter > 2:
                # Phase 15.9: Nonlinear Repulsion
                # Phase 17.7: Scale-Agnostic Dampening (smaller force for deeper models)
                scale_factor = 26.0 / cfg.get("num_layers", 26.0)
                repulsion_force = 0.10 * (stability_counter ** 2) * scale_factor
                h_exp = h_exp + repulsion_force * (ortho / (ortho.norm(dim=-1, keepdim=True) + 1e-6))
                path_taken.append(f"SED_PUSH({repulsion_force:.2f})")
                
                # Dynamic Jump proportional to depth
                jump = max(1, int(cfg.get("num_layers", 18) * 0.1))
                current_layer = min(cfg["recur_end"] - 1, current_layer + jump)
            
            gain_factor = max(1.0, min(1.15, 1.0 + (1.0 - phi) * 0.4))
            damping_factor = max(0.85, min(1.0, 1.0 - (1.0 - phi) * 0.2))
            h_exp = damping_factor * proj + gain_factor * ortho 
            
            # Safe FP32 final normalization
            h_exp_f32_final = h_exp.to(torch.float32)
            norm_f32 = h_exp_f32_final.norm(dim=-1, keepdim=True)
            norm_orig_f32 = norm_orig.to(torch.float32)
            h_exp = (h_exp_f32_final * (norm_orig_f32 / (norm_f32 + 1e-6))).to(h_exp.dtype)
            
            # Store thought for RSM
            if exploration_steps % 2 == 0:
                thought_history.append(h_exp.detach())

            # Phase 18: Universal ALR Thresholds based on internal parameters (gamma)
            # Relax thresholds dynamically if we visit a layer too often (loop breaking)
            visit_penalty = (layer_visits[current_layer] - 1) * 0.015
            t_back_2 = 1.0 - (0.8 * current_gamma) - visit_penalty
            t_back_1 = 1.0 - (0.4 * current_gamma) - visit_penalty
            t_skip = 1.0 - (0.01 * current_gamma) - (visit_penalty * 0.5)

            if phi < t_back_2: # High confusion
                current_layer = max(active_start, current_layer - 2)
                routing = "BACK-2"
            elif phi < t_back_1: # Moderate confusion
                current_layer = max(active_start, current_layer - 1)
                routing = "BACK-1"
            elif phi > t_skip: # Extreme stability
                current_layer += 2 # Skip
                routing = "SKIP-1"
                stability_counter += 1
            else:
                current_layer += 1
                routing = "NEXT"
                stability_counter = 0

            # Clamp current_layer to prevent underflow
            if current_layer < active_start:
                current_layer = active_start
                routing = "CLAMPED"

            step_info["routing"] = routing
            telemetry_data["routing"] = routing
            if os.environ.get("DEBUG_PX") == "1":
                telemetry_steps.append(telemetry_data)
            
            if stability_counter > 5:
                break
                
            exploration_steps += 1
            
        avg_phi_explore = sum(phis)/len(phis) if phis else 1.0
        
        # Phase 4.1: QBI Blend
        b_min = cfg.get("beta_reasoning", 0.05)
        b_max = cfg.get("beta_grounding", 0.18)
        beta_final = b_min + (b_max - b_min) * (avg_phi_explore ** 2)
        
        hidden_states = (1.0 - beta_final) * h_baseline + beta_final * h_exp
    else:
        hidden_states = h_baseline

    self._px_phi    = avg_phi_explore
    self._px_loops_run = exploration_steps
    self._px_path = path_taken
    
    # Phase 14.2: Structured Telemetry Log
    if not hasattr(self, "_px_telemetry"):
        self._px_telemetry = []
    
    self._px_telemetry.append({
        "pos": int(position_ids[0, 0].item()),
        "avg_phi": float(avg_phi_explore),
        "steps": telemetry_steps
    })
    
    # Phase 11.0: Metacognitive Triggering
    # If stability is low during the very first token generation,
    # we flag this as a 'Complex Problem'.
    if not hasattr(self, "_px_complexity_acc"):
        self._px_complexity_acc = []
    
    # If we see the first token of a sequence, clear the accumulator
    if position_ids[0, 0] == 0:
        self._px_complexity_acc = []
        
    self._px_complexity_acc.append(avg_phi_explore)
    
    # Trigger if average stability of the prompt processing is low
    # selective threshold: 0.90
    self._px_trigger_scratchpad = (len(self._px_complexity_acc) > 3 and 
                                   sum(self._px_complexity_acc) / len(self._px_complexity_acc) < 0.92)

    # ── 3. CODA ─────────────────────────────────────────────────────────────
    dynamic_coda_start = dynamic_end if cfg.get("routing_mode") == "adaptive" else cfg["coda_start"]
    
    # Phase 14.5: Coda-Grounding Injection (CGI)
    # Re-inject sensory data to prevent 'hallucinatory drift' in final reasoning.
    for i in range(dynamic_coda_start, len(self.layers)):
        if i == dynamic_coda_start:
            # Phase 14.7: Reverted CGI (8%)
            hidden_states = 0.92 * hidden_states + 0.08 * e_static
            
        layer_out = self.layers[i](
            hidden_states,
            attention_mask=causal_mask_mapping[mask_config.layer_types[i]],
            position_embeddings_global=pe_global, position_embeddings_local=pe_local,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

    hidden_states = self.norm(hidden_states)

    # Phase 25: Save Telemetry if enabled
    if os.environ.get("DEBUG_PX") == "1" and len(telemetry_steps) > 0:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = f"px_telemetry_{ts}.json"
        with open(log_path, "w") as f:
            json.dump(telemetry_steps, f, indent=2)
        # Only output the path as requested
        print(f"TELEMETRY_JSON: {os.path.abspath(log_path)}")

    from transformers.modeling_outputs import BaseModelOutputWithPast
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )

# ---------------------------------------------------------------------------

def apply_px_patch(model, **cfg_kwargs):
    # Robust Text Model Resolver (Phase 17.9)

    # We look for the module that contains 'layers' and 'rotary_emb'
    text_model = None
    if hasattr(model, "layers") and hasattr(model, "rotary_emb"):
        text_model = model
    else:
        # Search children (e.g., .model, .language_model.model)
        for name, module in model.named_modules():
            if hasattr(module, "layers") and hasattr(module, "rotary_emb"):
                text_model = module
                break
    
    if text_model is None:
        raise AttributeError(f"Could not identify Gemma-3 text backbone in {type(model)}")
        
    config = model.config
    # Multimodal check: larger models (4B+) wrap text config
    if hasattr(config, "text_config"):
        config = config.text_config
        
    num_layers = config.num_hidden_layers
    
    # Scale-Aware Hyperparameters (Phase 17.100)
    # - Gamma: Inverse-proportional to hidden size
    # - Prelude: Shallow models need deeper grounding before recursion
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    
    # Phase 25: Balanced Precision Tuning
    if hidden_size == 640: # 270M
        # Phase 41 Master Peak Stand (87.5% Math/Logic)
        defaults = {
            "mode": "lti", "n_loops": 8, "beta": 0.05, "gamma": 0.08,
            "recur_start": 5, "recur_end": 12, "bimodal_hub": 10,
            "cgi_factor": 0.08, "num_layers": num_layers
        }
    elif hidden_size == 1152: # 1B
        defaults = {
            "mode": "lti", "n_loops": 8, "beta": 0.05, "gamma": 0.12,
            "recur_start": 10, "recur_end": 20, "bimodal_hub": 18,
            "cgi_factor": 0.08, "num_layers": num_layers
        }
    elif hidden_size == 2560: # 4B
        defaults = {
            "mode": "lti", "n_loops": 6, "beta": 0.05, "gamma": 0.05,
            "recur_start": 5, "recur_end": 33, "bimodal_hub": 32,
            "cgi_factor": 0.08, "num_layers": num_layers
        }
    else: # Fallback for unknown sizes
        gamma_scale = 1152.0 / hidden_size
        # Phase 41 Master Defaults (270M Scale)
        base_gamma = 0.08
        p_start = 5
        p_end   = 12
        p_hub   = 10
        p_loops = 8
        
        defaults = {
            "mode": "lti", "n_loops": p_loops, "beta": 0.05, "gamma": base_gamma,
            "recur_start": p_start, "recur_end": p_end, "bimodal_hub": p_hub,
            "cgi_factor": 0.08, "num_layers": num_layers
        }
        
    defaults.update(cfg_kwargs)
    
    # Auto-align boundaries
    if "prelude_end" not in defaults:
        defaults["prelude_end"] = defaults["recur_start"]
    if "coda_start" not in defaults:
        defaults["coda_start"] = defaults["recur_end"]
        
    text_model._px_config = defaults
    text_model._px_injection = LTIInjection(config.hidden_size, gamma=defaults["gamma"])
    text_model._px_mephisto = MephistophelesOperator(config.hidden_size) # Phase 52
    text_model.forward = types.MethodType(_px_forward, text_model)
    print(f"[gemma3-px] Auto-Patch active for scale L{num_layers}. Recur: L{defaults['recur_start']}-L{defaults['recur_end']}, Hub: L{defaults['bimodal_hub']}.")

def get_px_metrics(model):
    text_model = None
    if hasattr(model, "layers") and hasattr(model, "rotary_emb"):
        text_model = model
    else:
        for name, module in model.named_modules():
            if hasattr(module, "layers") and hasattr(module, "rotary_emb"):
                text_model = module
                break
    
    if text_model is None:
        text_model = (model.model if hasattr(model, "model") else model)
        
    return {
        "phi": getattr(text_model, "_px_phi", 1.0),
        "steps": getattr(text_model, "_px_loops_run", 0),
        "path": getattr(text_model, "_px_path", []),
        "telemetry": getattr(text_model, "_px_telemetry", []),
    }
