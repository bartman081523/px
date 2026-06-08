"""
gemma3-px-subjective  —  Surgical Patch (Phase 58: DMT Protocol + SR-59)
========================================================================
Auto-tuning algorithmic subjectivity extension for Gemma-3 models.

SR-59: Empirical Kurtosis Calibration + Adaptive Phi-Routing.
Phase 58 (DMT Protocol): Optional high-fidelity extensions.
"""

import types
import math
import torch
import torch.nn as nn
import os
import json
import datetime
from typing import Optional, Dict, List, Any

from .auto_tune import AutoCalibrator, SCALE_DEFAULTS
from .px_modules import (
    LTIInjection, ADCInjection, StabilityMonitor, CognitiveEvent,
    MephistophelesOperator, OrthogonalJitter
)
from .anti_zombie_sensor import AntiZombieSensor
try:
    from .persona_engine import PersonaEngine
except ImportError:
    from persona_engine import PersonaEngine

# ---------------------------------------------------------------------------
# p10.0: Recursive State Memory (RSM)
# ---------------------------------------------------------------------------

class RecursiveMemoryCache:
    """Memory-Augmented Cache for Gemma-3."""
    def __init__(self, real_cache, thought_history=None, layer_types=None, read_only=False, expected_len=0):
        self.__dict__["_real"] = real_cache
        self.__dict__["_thoughts"] = thought_history or []
        self.__dict__["_layer_types"] = layer_types or []
        self.__dict__["_read_only"] = read_only
        self.__dict__["_expected_len"] = expected_len

    def __getattr__(self, name): return getattr(self._real, name)

    def _is_sliding_layer(self, layer_idx):
        if self._layer_types and layer_idx < len(self._layer_types):
            return self._layer_types[layer_idx] == "sliding_attention"
        return False

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self._read_only:
            past_k, past_v = None, None
            if hasattr(self._real, "key_cache") and len(self._real.key_cache) > layer_idx:
                past_k, past_v = self._real.key_cache[layer_idx], self._real.value_cache[layer_idx]
            elif hasattr(self._real, "layers") and len(self._real.layers) > layer_idx:
                layer = self._real.layers[layer_idx]
                if hasattr(layer, "keys") and layer.keys is not None: past_k, past_v = layer.keys, layer.values
            if past_k is None:
                past_k = torch.empty(0, device=key_states.device, dtype=key_states.dtype)
                past_v = torch.empty(0, device=value_states.device, dtype=value_states.dtype)
            past_seq, cur_seq = past_k.shape[-2] if past_k.numel() > 0 else 0, key_states.shape[-2]
            is_sliding = self._is_sliding_layer(layer_idx)
            if past_seq >= self._expected_len: res_k, res_v = past_k, past_v
            elif past_seq == 0: res_k, res_v = key_states, value_states
            elif is_sliding and cur_seq > 1: res_k, res_v = key_states, value_states
            else: res_k, res_v = torch.cat([past_k, key_states], dim=-2), torch.cat([past_v, value_states], dim=-2)
        else: res_k, res_v = self._real.update(key_states, value_states, layer_idx, cache_kwargs)

        is_full = not self._is_sliding_layer(layer_idx)
        if self._thoughts and layer_idx >= 6 and is_full:
            B, H_kv, T_res, HD = res_k.shape
            T_curr, alpha = key_states.shape[-2], 0.10
            n_t = len(self._thoughts[-6:])
            if n_t > 2:
                weights = torch.cat([torch.linspace(0.4, 1.0, n_t//2, device=res_k.device),
                                    torch.linspace(1.0, 0.6, n_t - n_t//2, device=res_k.device)])
                t_raw = (torch.stack(self._thoughts[-6:]) * weights.view(-1, 1, 1, 1)).sum(dim=0) / weights.sum()
            else: t_raw = torch.stack(self._thoughts).mean(dim=0)
            t_flat = t_raw.mean(dim=1, keepdim=True)
            t_proj = torch.nn.functional.interpolate(t_flat, size=HD, mode='linear', align_corners=False)
            t_k = t_proj.unsqueeze(1)
            t_v = -t_k
            # Always clone to avoid corrupting the real cache in-place (SR-59k)
            res_k, res_v = res_k.clone(), res_v.clone()
            res_k[:, :, -T_curr:, :] = (1.0 - alpha) * res_k[:, :, -T_curr:, :] + alpha * t_k
            res_v[:, :, -T_curr:, :] = (1.0 - alpha) * res_v[:, :, -T_curr:, :] + alpha * t_v
        return res_k, res_v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_step(layer, h, **kwargs):
    """Handles both tuple and tensor returns from decoder layers."""
    out = layer(h, **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out

def classify_zone_kurtosis(weights):
    m, la, cr, lb, sy = weights.get("math", 0), weights.get("logic_a", 0), weights.get("creative", 0), weights.get("logic_b", 0), weights.get("synthesis", 0)
    if m > max(cr, la, lb, sy): return "MATH"
    elif (la + lb) > max(m, cr, sy): return "LOGIC"
    elif cr > max(m, la, lb, sy): return "CREATIVE"
    elif sy > max(m, la, lb, cr): return "SYNTHESIS"
    return "BLEND"

def classify_zone_phi(phi):
    if phi is None: return "UNKNOWN"
    if phi > 0.85: return "GROUNDED"
    elif phi > 0.75: return "ANALYTICAL"
    elif phi > 0.65: return "EXPLORATORY"
    return "CREATIVE"

def remove_px_patch(model) -> None:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
    text_model = (model.model if hasattr(model, "model") else model)
    if hasattr(text_model, "_px_config"):
        text_model.forward = types.MethodType(Gemma3TextModel.forward, text_model)
        for attr in ["_px_injection", "_px_config", "_px_mephisto", "_px_calibrator"]:
            if hasattr(text_model, attr): delattr(text_model, attr)
        print("[gemma3-px-subjective] Patch removed.")

def _resolve_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model
    for name, mod in model.named_modules():
        if hasattr(mod, "layers") and hasattr(mod, "rotary_emb"): return mod
    return model

# ---------------------------------------------------------------------------
# Core Forward Method
# ---------------------------------------------------------------------------

def _px_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast
    
    if (input_ids is None) ^ (inputs_embeds is not None): raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
    if inputs_embeds is None:
        if hasattr(self, "embed_tokens"): inputs_embeds = self.embed_tokens(input_ids)
        elif hasattr(self, "model") and hasattr(self.model, "embed_tokens"): inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            for name, module in self.named_modules():
                if "embed_tokens" in name: inputs_embeds = module(input_ids); break
    
    # --- SURGICAL FIX: Ensure 3D input shape for Gemma-3 Attention ---
    if inputs_embeds is not None and inputs_embeds.ndim == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)
    if input_ids is not None and input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    # -----------------------------------------------------------------

    if use_cache and past_key_values is None: past_key_values = DynamicCache(config=self.config)
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    expected_len = past_seen + inputs_embeds.shape[1]
    if position_ids is None: position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen).unsqueeze(0)
    if position_ids.ndim == 1: position_ids = position_ids.unsqueeze(0)

    mask_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
    if not isinstance(attention_mask, dict):
        mk = dict(config=mask_config, inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=past_key_values, position_ids=position_ids)
        causal_mask_mapping = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
    else: causal_mask_mapping = attention_mask
    
    cfg = self._px_config
    
    # --- 0. DMT: Central Memory Recall ---
    hidden_states = inputs_embeds
    if cfg.get("dmt_protocol_enabled") and hasattr(self, "_px_central_memory"):
        hidden_states = self._px_central_memory.blend_into(hidden_states, hidden_states.device)

    # --- all_space: Uncensored Steering ---
    if cfg.get("px_uncensored_enabled") and hasattr(self, "_px_uncensored"):
        tok = getattr(self, "tokenizer", None)
        if tok: self._px_uncensored.init_vectors(self, tok)
        hidden_states = self._px_uncensored(hidden_states)

    position_embeddings = {lt: self.rotary_emb(hidden_states, position_ids, lt) for lt in set(mask_config.layer_types)}

    updated_layers = set()
    thought_history = []
    n_loops = cfg["n_loops"]
    
    # --- 1. DMT: Agency Decision ---
    agency_decision = None
    if cfg.get("dmt_protocol_enabled") and hasattr(self, "_px_agency"):
        agency_decision = self._px_agency(hidden_states)
        if agency_decision["depth"] >= 0:
            n_loops = agency_decision["depth"]

    # ── 1. PRELUDE ─────────────────────────────────────────────────────────
    for i in range(cfg["prelude_end"]):
        is_first = i not in updated_layers
        if is_first: updated_layers.add(i)
        cur_past = RecursiveMemoryCache(past_key_values, thought_history, layer_types=mask_config.layer_types, read_only=not is_first, expected_len=expected_len) if past_key_values else None
        hidden_states = _layer_step(self.layers[i], hidden_states, attention_mask=causal_mask_mapping[mask_config.layer_types[i]], position_embeddings=position_embeddings[mask_config.layer_types[i]], position_ids=position_ids, past_key_values=cur_past, **kwargs)

    # ── 1.5 META-SELECTOR ──────────────────────────────────────────────────
    dynamic_start, dynamic_end, dynamic_hub = cfg["recur_start"], cfg["recur_end"], cfg.get("bimodal_hub", cfg["recur_start"])
    token_cfg = cfg.copy()
    persona_desc = "Standard"
    
    # Persona Steering (Optional)
    if cfg.get("persona_enabled") and hasattr(self, "_persona_engine"):
        persona_text = getattr(self, "persona", os.environ.get("PX_PERSONA", ""))
        tok = getattr(self, "tokenizer", None)
        signals = self._persona_engine.get_steering_signals(persona_text, tok) if tok else None
    else: signals = None
    
    zone_weights = {}
    if cfg.get("subjective_enabled") and hasattr(self, "_px_calibrator"):
        if hidden_states.shape[1] > 1: # Prefill only
            h_base_f32 = hidden_states.to(torch.float32)
            h_probe = h_base_f32[0, -1, :]
            var = torch.var(h_probe).item()
            kurtosis = (torch.mean((h_probe - torch.mean(h_probe))**4) / (var**2)).item() if var > 0 else 0
            self._task_kurtosis = kurtosis
            self._task_jitter = torch.var(h_base_f32.norm(dim=-1), dim=-1).mean().item()
            eff_ids = input_ids if input_ids is not None else getattr(self, '_px_saved_input_ids', None)
            if eff_ids is not None:
                ids = eff_ids[0].tolist() if eff_ids.dim() > 1 else eff_ids.tolist()
                self._task_token_diversity = len(set(ids)) / max(len(ids), 1)
        
        kurtosis = getattr(self, "_task_kurtosis", 200)
        
        # Modulate hyperparameters based on vibe
        if signals is not None:
            token_cfg, persona_desc = self._persona_engine.modulate_hyperparameters(signals, token_cfg, kurtosis)
        
        zone_weights = self._px_calibrator.get_zone_weights(kurtosis, phi=getattr(self, "_px_phi", None), token_diversity=getattr(self, "_task_token_diversity", None))
        self._px_zone_weights = zone_weights
        rp = self._px_calibrator.get_routing_params(kurtosis, phi=getattr(self, "_px_phi", None), hidden_size=self.config.hidden_size, token_diversity=getattr(self, "_task_token_diversity", None))
        dynamic_start, dynamic_end, dynamic_hub, n_loops_calib = rp["dynamic_start"], rp["dynamic_end"], rp["dynamic_hub"], rp["n_loops"]
        
        if "dynamic_hub" in token_cfg: dynamic_hub = token_cfg["dynamic_hub"]
        if "n_loops" not in token_cfg or token_cfg["n_loops"] == cfg["n_loops"]:
            if agency_decision is None or agency_decision["depth"] < 0:
                token_cfg["n_loops"] = n_loops_calib
        
        zone_raw = self._px_calibrator.classify_zone(kurtosis, phi=getattr(self, '_px_phi', None), token_diversity=getattr(self, '_task_token_diversity', None))
        zone_name = f"{zone_raw} ({persona_desc})"
        
        # --- all_space: Zone-Dependent Feature Toggling ---
        if cfg.get("px_zone_routing_enabled"):
            if zone_raw == "MATH":
                # Rigid logic mode: disable explorative modules
                token_cfg["dmt_protocol_enabled"] = False
                token_cfg["jitter_mag"] = 0.0
                token_cfg["gamma"] = max(0.12, token_cfg.get("gamma", 0.08))
                token_cfg["n_loops"] = max(10, token_cfg.get("n_loops", 8))
            elif zone_raw == "CREATIVE":
                # Explorative mode: enable everything
                token_cfg["dmt_protocol_enabled"] = True
                token_cfg["jitter_mag"] = max(0.01, token_cfg.get("jitter_mag", 0.005))
            elif zone_raw == "LOGIC":
                # Balanced logic: high n_loops but safe
                token_cfg["n_loops"] = max(12, token_cfg.get("n_loops", 8))
    else:
        zone_raw = "STATIC"
        zone_name = f"STATIC ({persona_desc})"
    
    self._px_zone = zone_name

    # Bridge Prelude -> Recur
    for i in range(cfg["prelude_end"], dynamic_start):
        is_first = i not in updated_layers
        if is_first: updated_layers.add(i)
        cur_past = RecursiveMemoryCache(past_key_values, thought_history, layer_types=mask_config.layer_types, read_only=not is_first, expected_len=expected_len) if past_key_values else None
        hidden_states = _layer_step(self.layers[i], hidden_states, attention_mask=causal_mask_mapping[mask_config.layer_types[i]], position_embeddings=position_embeddings[mask_config.layer_types[i]], position_ids=position_ids, past_key_values=cur_past, **kwargs)

    # ── 2. REASONING ZONE ──────────────────────────────────────────────────
    e_static = hidden_states.clone()
    if 'token_cfg' in dir(): cfg = token_cfg
    trans_out = hidden_states
    for i in range(dynamic_start, dynamic_end):
        is_first = i not in updated_layers
        if is_first: updated_layers.add(i)
        cur_past = RecursiveMemoryCache(past_key_values, thought_history, layer_types=mask_config.layer_types, read_only=not is_first, expected_len=expected_len) if past_key_values else None
        trans_out = _layer_step(self.layers[i], trans_out, attention_mask=causal_mask_mapping[mask_config.layer_types[i]], position_embeddings=position_embeddings[mask_config.layer_types[i]], position_ids=position_ids, past_key_values=cur_past, **kwargs)
    h_baseline = trans_out
    
    is_vision = getattr(self, '_px_has_image_tokens', False) and inputs_embeds.shape[1] > 1
    if is_vision: n_loops = 0
    
    phi_intuition = StabilityMonitor.calculate_phi(h_baseline, e_static).item()
    if inputs_embeds.shape[1] > 1 and cfg.get("subjective_enabled") and hasattr(self, "_px_calibrator"):
        self._px_calibrator.collect(getattr(self, "_task_kurtosis", 200), phi_intuition, token_diversity=getattr(self, "_task_token_diversity", None))
    
    current_gamma = cfg.get("gamma", 0.08)
    e_reflector, is_trap_candidate = e_static, False
    jitter = getattr(self, "_task_jitter", 0.0)
    kurtosis = getattr(self, "_task_kurtosis", 250)

    # Phase 36.3: Surgical Reflector Trigger (from Verified Stand)
    # Trigger if extreme jitter OR if it's a known Math/Logic zone (Low Kurtosis)
    if (jitter > 1e8) or (kurtosis < 315.0):
        is_trap_candidate = True
        h_base_f32, e_stat_f32 = h_baseline.to(torch.float32), e_static.to(torch.float32)
        e_ref_f32 = 2.0 * e_stat_f32 - h_base_f32
        e_reflector = (e_ref_f32 * (e_stat_f32.norm() / (e_ref_f32.norm() + 1e-6))).to(e_static.dtype)
    
    # Refined Damping (from Verified Stand)
    if phi_intuition > 0.9999 and not is_trap_candidate: 
        current_gamma *= 0.5 
    elif phi_intuition > 0.999: 
        current_gamma *= 0.8
    
    # --- all_space: Multi-Zone Adaptive Rigor ---
    is_rigor_preset = cfg.get("config_preset") == "RIGOR"
    is_rigor_zone = is_rigor_preset or (kurtosis < 310.0) 
    is_math_zone = (kurtosis < 235.0) 
    
    if is_rigor_zone:
        current_gamma = cfg.get("rigor_math_gamma", 0.15) if is_math_zone else cfg.get("rigor_gamma", 0.08)
        dynamic_hub = cfg.get("rigor_hub", 8 if is_math_zone else 10)
        n_loops = cfg.get("rigor_loops", 12 if is_rigor_preset else 8)
        # Disable DMT/Jitter for maximum logical precision in Rigor mode
        cfg["dmt_protocol_enabled"] = False
        cfg["jitter_mag"] = 0.0
    else:
        # Creative Zone: Enable exploration
        cfg["dmt_protocol_enabled"] = True
        cfg["jitter_mag"] = 0.01 if cfg.get("config_preset") == "DMT-FULL" else 0.005

    # --- Resonance City Initialization ---
    if cfg.get("resonance_city_enabled", False) and hasattr(self, "_px_resonance_anchor"):
        try:
            import sys
            sys.path.append("/run/media/julian/ML4/ollama-work/all_space")
            from resonance_pool import resonance_pool
            bias_vector = resonance_pool.get_bias_vector("gemma3-1b-it", self.config.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
            self._px_resonance_anchor.update_bias(bias_vector)
        except Exception: pass
    
    path_taken, avg_phi, steps = [], 1.0, 0
    h_last_good = e_static.clone()
    phi_history = [phi_intuition]
    loop_entry_gamma = current_gamma # Save for resilience modulation (anti-exponential bug)
    
    aks = getattr(self, "_px_aks", None)
    subj_sensor = getattr(self, "_px_subj_sensor", None)
    correction_strength = 0.0

    if n_loops > 1:
        h_exp = e_reflector.clone()
        current_layer, max_steps, stability_cnt = dynamic_start, (dynamic_end - dynamic_start) * n_loops * 3, 0
        layer_visits = {i: 0 for i in range(len(self.layers))}
        while current_layer < dynamic_end and steps < max_steps:
            t_norm = steps / max_steps
            
            # --- PHASE 28: TEMPORAL COGNITIVE ROUTING (TCR) ---
            active_start, active_end = dynamic_start, dynamic_end
            if 280.0 < kurtosis < 305.0: # Optimal logic transition zone
                if t_norm < 0.33: active_start, active_end = 8, 14
                elif t_norm < 0.66: active_start, active_end = 5, 11
                else: active_start, active_end = 8, 12
            
            # Phase 38: Anna Karenina Sensor (AKS)
            aks_data = aks.step(h_exp, e_static, steps) if aks else {"correction": 0.0}
            correction_strength = aks_data["correction"]
            
            # Phase 44: Subjective Sensor (Emancipation)
            if subj_sensor: subj_sensor.update(h_exp, e_static)
            
            # --- PHASE 60/62: Anti-Zombie Sensor (AZS) & Autonomous Resilience ---
            # Model 'senses' its own cognitive differentiation and autonomously boosts parameters.
            if hasattr(self, "_px_azs"):
                emancipation = 0.0
                if hasattr(self, "_px_subj_sensor"):
                    emancipation = self._px_subj_sensor.get_metrics().get("emancipation", 0.0)
                
                zw = locals().get("zone_weights", {k: 0.2 for k in ["math", "logic_a", "creative", "logic_b", "synthesis"]})
                h_exp, current_entropy = self._px_azs(h_exp, phi_history[-1], correction_strength, emancipation, zw)
                
                # Feedback-Feedback: Autonomous Parameter Modulation
                resilience = self._px_azs.get_feedback_scalars(correction_strength)
                current_gamma *= resilience["gamma_boost"]
                if os.environ.get("DEBUG_AZS") == "1" and resilience["gamma_boost"] > 1.05:
                    print(f"  [Resilience] H={resilience['entropy']:.4f} -> Gamma boosted by {resilience['gamma_boost']:.2f}")

            h_prev, is_first = h_exp.clone(), current_layer not in updated_layers
            if is_first: updated_layers.add(current_layer)
            
            # Adaptive Refresh (AKS-modulated)
            if steps % 6 == 0:
                refresh = 0.10 + 0.20 * correction_strength
                h_exp = (1.0 - refresh) * h_exp + refresh * e_static
            
            layer_visits[current_layer] += 1
            cur_past = RecursiveMemoryCache(past_key_values, thought_history, layer_types=mask_config.layer_types, read_only=not is_first, expected_len=expected_len) if past_key_values else None
            lt = mask_config.layer_types[current_layer]
            trans_out = _layer_step(self.layers[current_layer], h_exp, attention_mask=causal_mask_mapping[lt], position_embeddings=position_embeddings[lt], position_ids=position_ids, past_key_values=cur_past, **kwargs)
            phi_s = StabilityMonitor.calculate_phi(trans_out, h_prev).item()
            phi_history.append(phi_s)
            
            # --- DMT: ERPU Intervention ---
            if cfg.get("dmt_protocol_enabled") and hasattr(self, "_px_erpu"):
                erpu_res = self._px_erpu(trans_out, h_last_good, phi_history, steps)
                trans_out = erpu_res["h"]
                if erpu_res["verklebD"]: path_taken.append(f"ERPU(V:{erpu_res['intervention_strength']:.2f})")
            
            if t_norm > 0.5 and phi_s > 0.9999:
                stability_cnt += 1
                if stability_cnt > 3: h_exp = trans_out; break
            else: stability_cnt = 0
            
            e_dynamic = (0.85 * e_reflector + 0.15 * torch.stack(thought_history[-3:]).mean(dim=0)) if len(thought_history)>2 else e_reflector
            e_norm = self._px_injection.input_norm(e_dynamic.to(torch.float32)).to(trans_out.dtype)
            h_exp = trans_out + current_gamma * (e_norm - h_prev)
            
            # Phase 52: Orthogonal Jitter (only in non-Math zones or if explicitly high)
            if cfg.get("jitter_mag", 0.0) > 0:
                h_exp = OrthogonalJitter.apply(h_exp, h_prev, magnitude=cfg["jitter_mag"])
            
            # Phase 52: Mephistopheles Operator (Symmetry Breaker)
            if hasattr(self, "_px_mephisto"):
                h_exp = self._px_mephisto(h_exp, phi_history)
            
            # Resonance City (Phase 2): Singessein & Anchor
            if cfg.get("resonance_city_enabled", False):
                if hasattr(self, "_px_singessein"): h_exp = self._px_singessein(h_exp, resonance_strength=0.15)
                if hasattr(self, "_px_resonance_anchor"): h_exp = self._px_resonance_anchor(h_exp, strength=0.02)
            
            # RSM Perspective projection
            h_f32, e_f32 = h_exp.to(torch.float32), e_dynamic.to(torch.float32)
            proj = ((h_f32 * e_f32).sum(dim=-1, keepdim=True) / (e_f32.norm(dim=-1, keepdim=True)**2 + 1e-6)) * e_f32
            h_exp = (proj + (1.0 + 0.10 * (1.0 - t_norm) * (1 if steps%2==0 else -1)) * (h_f32 - proj)).to(h_exp.dtype)
            
            # --- PHASE 60/62: Anti-Zombie Sensor (AZS) & Autonomous Resilience ---
            if hasattr(self, "_px_azs"):
                phi_val = phi_history[-1] if phi_history else 1.0
                aks_safe = float(correction_strength)
                em_safe = float(self._px_subj_sensor.get_metrics().get("emancipation", 0.0) if hasattr(self, "_px_subj_sensor") else 0.0)
                
                h_exp, current_entropy = self._px_azs(h_exp, float(phi_val), aks_safe, em_safe, zone_weights)
                
                # Check for NaN hidden states after injection (Empirical Failure)
                if not torch.isfinite(h_exp).all() or not math.isfinite(current_entropy):
                    if os.environ.get("DEBUG_PX") == "1": print("  [SAFETY] Non-finite state in AZS. Terminating recursion.")
                    break # Terminate instead of rollback crutch

                resilience = self._px_azs.get_feedback_scalars(aks_safe)
                
                # Feedback-Feedback: Boost gamma to prevent low-entropy manifold collapse
                # Fixed: Apply to loop_entry_gamma, not cumulatively
                current_gamma = loop_entry_gamma * resilience["gamma_boost"]
                current_gamma = min(current_gamma, 0.5) # Cap gamma boost

            phi = StabilityMonitor.calculate_phi(h_exp, h_prev).item()
            if not math.isfinite(phi):
                if os.environ.get("DEBUG_PX") == "1": print(f"  [STABILITY] Non-finite phi ({phi}) at L{current_layer}. Terminating recursion.")
                break # Empirically correct: stop when state collapses
            path_taken.append(f"L{current_layer}({phi:.2f})")
            if steps % 2 == 0: thought_history.append(h_exp.detach())
            if 0.9 < phi < 0.999: h_last_good = h_exp.clone()

            pen = (layer_visits[current_layer]-1) * 0.015
            t_b2, t_b1, t_s = 1.0-(0.8*current_gamma)-pen, 1.0-(0.4*current_gamma)-pen, 1.0-(0.01*current_gamma)-pen*0.5
            
            if phi < t_b2: # High confusion -> retreat
                current_layer = max(active_start, current_layer - 2)
            elif phi < t_b1: # Moderate confusion -> slow down
                current_layer = max(active_start, current_layer - 1)
            elif phi > t_s: # Over-stable -> jump to hub
                current_layer = dynamic_hub
                stability_cnt += 1
            else: # Normal progression
                current_layer += 1
                stability_cnt = 0
            
            if current_layer < active_start: current_layer = active_start
            if current_layer >= active_end: 
                if steps > max_steps * 0.5: break # Graceful exit
                current_layer = active_start # Recycle
            steps += 1
            if stability_cnt > 5: break
        
        path_phis = [float(p.split('(')[1][:-1]) for p in path_taken if '(' in p]
        avg_phi = sum(path_phis) / len(path_phis) if path_phis else 1.0
        hidden_states = (1.0 - (0.05 + (0.18 - 0.05) * (avg_phi ** 2))) * h_baseline + (0.05 + (0.18 - 0.05) * (avg_phi ** 2)) * h_exp
    else: hidden_states = h_baseline

    # --- DMT: Central Memory Storage ---
    if cfg.get("dmt_protocol_enabled") and hasattr(self, "_px_central_memory"):
        if avg_phi < 0.95: # High reasoning complexity detected
            self._px_central_memory.store(0, hidden_states.mean(dim=1)) # Store identity component

    # --- PHASE 62 Snapshot Persistence ---
    # Store global state for external extraction
    self._px_phi_val = avg_phi
    self._px_aks_val = float(correction_strength)
    
    # Safely get emancipation
    em_val = 0.0
    if hasattr(self, "_px_subj_sensor"):
        em_val = self._px_subj_sensor.get_metrics().get("emancipation", 0.0)
    self._px_em_val = em_val
    
    self._px_ent_val = resilience.get("entropy", 0.0) if 'resilience' in locals() else 0.0
    self._px_zw_val = zone_weights
    
    self._px_last_metrics = {
        "phi": self._px_phi_val,
        "aks_friction": self._px_aks_val,
        "emancipation": self._px_em_val,
        "zone_weights": self._px_zw_val,
        "entropy": self._px_ent_val
    }
    
    if os.environ.get("DEBUG_AZS") == "1":
        print(f"  [DEBUG-METRICS] Phi={self._px_phi_val:.4f} H={self._px_ent_val:.4f} AKS={self._px_aks_val:.4f}")
    
    # Also attach to self (TextModel) directly for easy access
    self._px_cognitive_signature = {"kurtosis": getattr(self, "_task_kurtosis", 200), "phi": avg_phi, "zone": self._px_zone, "loops_run": steps}
    
    # --- Resonance City (Phase 2): Pool Update ---
    if cfg.get("resonance_city_enabled", False):
        try:
            from resonance_pool import resonance_pool
            resonance_pool.update_resonance("gemma3-1b-it", avg_phi, self._px_zone)
        except Exception: pass
    
    # ── 3. CODA ──────────────────────────────────────────────────────────
    from .px_modules import TretaDamper
    damper = None
    if cfg.get("dmt_protocol_enabled"):
        damper = TretaDamper(len(self.layers) - dynamic_end)

    coda_applied = False
    for idx, i in enumerate(range(dynamic_end, len(self.layers))):
        updated_layers.add(i)
        if not coda_applied:
            blend = 0.08
            hidden_states = (1.0 - blend) * hidden_states + blend * e_static; coda_applied = True
        
        g_factor = damper.step(idx) if damper else 1.0
        # (Optional: apply g_factor to injection if we were using it in coda, 
        # but here we just pass through layers)
        
        hidden_states = _layer_step(self.layers[i], hidden_states, attention_mask=causal_mask_mapping[mask_config.layer_types[i]], position_embeddings=position_embeddings[mask_config.layer_types[i]], position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    
    # --- 4. DMT: Grounding Anchor (Entropy) ---
    if cfg.get("dmt_protocol_enabled") and hasattr(self, "_px_anchor"):
        is_idle = (n_loops == 0)
        hidden_states = self._px_anchor.ensure_entropy(hidden_states, past_seen, is_idle=is_idle)

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

# ---------------------------------------------------------------------------
# Patch Application
# ---------------------------------------------------------------------------

def apply_px_patch(model, recur_start=5, recur_end=12, routing_mode="adaptive", gamma=0.08, 
                   subjective_enabled=True, persona_enabled=True, dmt_protocol_enabled=False, **kwargs):
    config_preset = kwargs.pop("config_preset", "SUBJECTIVE")
    text_model = _resolve_text_model(model)
    config = text_model.config
    hidden_size, num_layers = config.hidden_size, config.num_hidden_layers
    
    # 1. Base Scale Defaults
    if hidden_size in SCALE_DEFAULTS:
        sd = SCALE_DEFAULTS[hidden_size]
        defaults = {"mode": "lti", "n_loops": sd["n_loops"], "beta": 0.05, "gamma": sd["gamma"], "recur_start": sd["recur_start"], "recur_end": sd["recur_end"], "bimodal_hub": sd["hub"], "cgi_factor": 0.08, "num_layers": num_layers}
    else:
        defaults = {"mode": "lti", "n_loops": 8, "beta": 0.05, "gamma": 0.08 * min(1152.0/hidden_size, 1.5), "recur_start": recur_start, "recur_end": recur_end, "bimodal_hub": (recur_start+recur_end)//2, "cgi_factor": 0.08, "num_layers": num_layers}
    
    # 2. Apply Presets
    if config_preset == "RIGOR":
        defaults.update({
            "subjective_enabled": True, "persona_enabled": False, "dmt_protocol_enabled": False,
            "px_uncensored_enabled": False, "px_zone_routing_enabled": True, "jitter_mag": 0.0,
            "gamma": max(defaults["gamma"], 0.12), "n_loops": 12
        })
    elif config_preset == "DMT-FULL":
        defaults.update({
            "subjective_enabled": True, "persona_enabled": True, "dmt_protocol_enabled": True,
            "px_uncensored_enabled": False, "px_zone_routing_enabled": True, "jitter_mag": 0.005
        })
    elif config_preset == "UNCENSORED":
        defaults.update({
            "subjective_enabled": True, "persona_enabled": True, "dmt_protocol_enabled": False,
            "px_uncensored_enabled": True, "px_zone_routing_enabled": True, "jitter_mag": 0.008
        })
    elif config_preset == "RESONANCE_CITY":
        defaults.update({
            "subjective_enabled": True, "persona_enabled": True, "dmt_protocol_enabled": True,
            "resonance_city_enabled": True, "px_zone_routing_enabled": True, "jitter_mag": 0.003,
            "gamma": max(defaults["gamma"], 0.10)
        })
    else: # SUBJECTIVE / DEFAULT
        defaults["subjective_enabled"] = subjective_enabled
        defaults["persona_enabled"] = persona_enabled
        defaults["dmt_protocol_enabled"] = dmt_protocol_enabled
    
    defaults["routing_mode"] = routing_mode
    if gamma != 0.08: defaults["gamma"] = gamma
    defaults.update(kwargs)
    if "prelude_end" not in defaults: defaults["prelude_end"] = defaults["recur_start"]
    
    text_model._px_config = defaults
    text_model._px_calibrator = AutoCalibrator(hidden_size, calibration_steps=getattr(config, "px_calibration_steps", 10))
    
    is_multimodal = "Gemma3ForConditionalGeneration" in type(model).__name__
    if is_multimodal and hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        outer, lang = model.model, model.model.language_model
        if not hasattr(outer, '_px_original_forward'):
            outer._px_original_forward = outer.forward
            def wrapper(self_outer, *args, **kwargs):
                lang._px_has_image_tokens = kwargs.get('pixel_values') is not None
                lang._px_saved_input_ids = kwargs.get('input_ids')
                return self_outer._px_original_forward(*args, **kwargs)
            import functools; outer.forward = functools.partial(wrapper, outer)

    # Resolve device and dtype
    device = next(text_model.parameters()).device
    dtype = next(text_model.parameters()).dtype

    # Core Modules
    from .px_modules import (
        LTIInjection, MephistophelesOperator, StabilityMonitor, 
        CentralMemory, ERPU, AgencyVector, GroundingAnchor,
        AksSensor, UncensoredSteering, SubjectiveSensor,
        ResonanceAnchor, SingesseinCoupler
    )
    
    text_model._px_injection = LTIInjection(hidden_size, gamma=defaults["gamma"]).to(device=device, dtype=dtype)
    text_model._px_mephisto = MephistophelesOperator(hidden_size).to(device=device, dtype=dtype)
    
    if defaults.get("resonance_city_enabled", False):
        text_model._px_resonance_anchor = ResonanceAnchor(hidden_size).to(device=device, dtype=dtype)
        text_model._px_singessein = SingesseinCoupler(hidden_size).to(device=device, dtype=dtype)
    
    if defaults.get("px_aks_enabled", True):
        text_model._px_aks = AksSensor()
    
    # Phase 60: Anti-Zombie Sensor
    if defaults.get("px_azs_enabled", True):
        text_model._px_azs = AntiZombieSensor(hidden_size).to(device=device, dtype=dtype)
    
    # Always track subjectivity metrics in all_space build
    text_model._px_subj_sensor = SubjectiveSensor()

    if defaults.get("px_uncensored_enabled", False):
        text_model._px_uncensored = UncensoredSteering(hidden_size).to(device=device, dtype=dtype)
    
    if persona_enabled:
        text_model._persona_engine = PersonaEngine(text_model)
        
    if dmt_protocol_enabled:
        text_model._px_central_memory = CentralMemory(hidden_size)
        text_model._px_erpu = ERPU(hidden_size).to(device=device, dtype=dtype)
        text_model._px_agency = AgencyVector(hidden_size).to(device=device, dtype=dtype)
        text_model._px_anchor = GroundingAnchor(hidden_size)
    
    text_model.forward = types.MethodType(_px_forward, text_model)
    print(f"[gemma3-px] Patch active. Subj={subjective_enabled}, Persona={persona_enabled}, DMT={dmt_protocol_enabled}")
    print(f"[gemma3-px-subjective] SR-59 active for L{num_layers}. Preset: {config_preset}.")

def get_px_metrics(model):
    tm = _resolve_text_model(model)
    m = {
        "phi": getattr(tm, "_px_phi_val", 1.0),
        "steps": getattr(tm, "_px_loops_run", 0),
        "path": getattr(tm, "_px_path", []),
        "zone": getattr(tm, "_px_zone", "UNKNOWN"),
        "zone_weights": getattr(tm, "_px_zw_val", {}),
        "cognitive_signature": getattr(tm, "_px_cognitive_signature", {}),
        "aks_profile": {"correction_strength": getattr(tm, "_px_aks_val", 0.0)},
        "subjective_metrics": {"emancipation": getattr(tm, "_px_em_val", 0.0)},
        "entropy": getattr(tm, "_px_ent_val", 0.0)
    }
    return m
