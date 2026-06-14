"""
minicpm5-px  —  Surgical Patch for LlamaForCausalLM (MiniCPM5-1B)
================================================================
Phase 58: DMT Protocol + SR-59 (Adaptive Subjectivity)
"""

import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import datetime
import traceback
from typing import Optional, Dict, List, Any

try:
    from .auto_tune import AutoCalibrator, SCALE_DEFAULTS
    from .px_modules import (
        LTIInjection, ADCInjection, StabilityMonitor, CognitiveEvent,
        MephistophelesOperator, OrthogonalJitter
    )
    try:
        from .persona_engine import PersonaEngine
    except ImportError:
        from persona_engine import PersonaEngine
except ImportError:
    from auto_tune import AutoCalibrator, SCALE_DEFAULTS
    from px_modules import (
        LTIInjection, ADCInjection, StabilityMonitor, CognitiveEvent,
        MephistophelesOperator, OrthogonalJitter
    )
    from persona_engine import PersonaEngine


class RecursiveMemoryCache:
    """Memory-Augmented Cache for Llama (Soft-RSM)."""
    def __init__(self, real_cache, thought_history=None, read_only=False, expected_len=0):
        self.__dict__["_real"] = real_cache
        self.__dict__["_thoughts"] = thought_history or []
        self.__dict__["_read_only"] = read_only
        self.__dict__["_expected_len"] = expected_len

    def __getattr__(self, name): return getattr(self._real, name)

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
            if past_seq >= self._expected_len: res_k, res_v = past_k, past_v
            elif past_seq == 0: res_k, res_v = key_states, value_states
            else: res_k, res_v = torch.cat([past_k, key_states], dim=-2), torch.cat([past_v, value_states], dim=-2)
        else: res_k, res_v = self._real.update(key_states, value_states, layer_idx, cache_kwargs)

        if self._thoughts and layer_idx >= 6:
            B, H_kv, T_res, HD = res_k.shape
            T_curr, alpha = key_states.shape[-2], 0.15
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
            if self._read_only: res_k, res_v = res_k.clone(), res_v.clone()
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
    from transformers.models.llama.modeling_llama import LlamaModel
    text_model = (model.model if hasattr(model, "model") else model)
    if hasattr(text_model, "_px_config"):
        text_model.forward = types.MethodType(LlamaModel.forward, text_model)
        for attr in ["_px_injection", "_px_config", "_px_mephisto", "_px_calibrator"]:
            if hasattr(text_model, attr): delattr(text_model, attr)
        print("[minicpm5-px] Patch removed.")

def _resolve_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model
    for name, mod in model.named_modules():
        if hasattr(mod, "layers") and hasattr(mod, "rotary_emb"): return mod
    return model

def _px_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast
    
    if (input_ids is None) ^ (inputs_embeds is not None): raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
    if inputs_embeds is None:
        if hasattr(self, "embed_tokens"): inputs_embeds = self.embed_tokens(input_ids)
        elif hasattr(self, "model") and hasattr(self.model, "embed_tokens"): inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            for name, module in self.named_modules():
                if "embed_tokens" in name: inputs_embeds = module(input_ids); break
    
    # --- SURGICAL FIX: Ensure 3D input shape for Attention ---
    if inputs_embeds is not None and inputs_embeds.ndim == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)
    if input_ids is not None and input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    # ---------------------------------------------------------

    if use_cache and past_key_values is None: past_key_values = DynamicCache(config=self.config)
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    expected_len = past_seen + inputs_embeds.shape[1]
    if position_ids is None: position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen).unsqueeze(0)
    
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)

    causal_mask = create_causal_mask(config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=past_key_values, position_ids=position_ids)
    hidden_states = inputs_embeds
    
    # RoPE tuple generation
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    cfg = self._px_config
    subjective = cfg.get("subjective_enabled", False)
    updated_layers = set()
    
    # --- Subjective Routing ---
    n_loops = cfg["n_loops"]

    # ── 1. PRELUDE ──
    for i in range(cfg["prelude_end"]):
        updated_layers.add(i)
        hidden_states = _layer_step(self.layers[i], hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings, position_ids=position_ids, past_key_values=past_key_values, **kwargs)

    # ── 1.5 META-SELECTOR ──
    dynamic_start, dynamic_end, dynamic_hub = cfg["recur_start"], cfg["recur_end"], cfg.get("bimodal_hub", cfg["recur_start"])
    token_cfg = cfg.copy()
    
    # Persona Steering (Phase 54)
    persona_text = getattr(self, "persona", os.environ.get("PX_PERSONA", ""))
    tok = getattr(self, "tokenizer", None)
    signals = self._persona_engine.get_steering_signals(persona_text, tok) if tok else None

    zone_weights = getattr(self, "_px_zone_weights", {})
    if cfg.get("routing_mode") == "adaptive":
        if hidden_states.shape[1] > 1:
            h_base_f32 = hidden_states.to(torch.float32)
            h_probe = h_base_f32[0, -1, :]
            var = torch.var(h_probe).item()
            kurtosis = (torch.mean((h_probe - torch.mean(h_probe))**4) / (var**2)).item() if var > 0 else 0
            self._task_kurtosis = kurtosis
            self._task_jitter = torch.var(h_base_f32.norm(dim=-1), dim=-1).mean().item()
            if input_ids is not None:
                ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
                self._task_token_diversity = len(set(ids)) / max(len(ids), 1)
        kurtosis = getattr(self, "_task_kurtosis", 200)
        token_diversity = getattr(self, "_task_token_diversity", None)
        
        # Modulate hyperparameters based on vibe
        token_cfg, persona_desc = self._persona_engine.modulate_hyperparameters(signals, token_cfg, kurtosis)

        if subjective and hasattr(self, "_px_calibrator"):
            zone_weights = self._px_calibrator.get_zone_weights(kurtosis, phi=getattr(self, "_px_phi", None), token_diversity=token_diversity)
            self._px_zone_weights = zone_weights
            rp = self._px_calibrator.get_routing_params(kurtosis, phi=getattr(self, "_px_phi", None), hidden_size=self.config.hidden_size, token_diversity=token_diversity)
            dynamic_start, dynamic_end, dynamic_hub, n_loops_calib = rp["dynamic_start"], rp["dynamic_end"], rp["dynamic_hub"], rp["n_loops"]
            
            # Persona overrides for routing
            if "dynamic_hub" in token_cfg: dynamic_hub = token_cfg["dynamic_hub"]
            if "n_loops" not in token_cfg or token_cfg["n_loops"] == cfg["n_loops"]:
                 token_cfg["n_loops"] = n_loops_calib

            if dynamic_start >= dynamic_end: dynamic_start, dynamic_end, dynamic_hub = int(len(self.layers)*0.38), int(len(self.layers)*0.75), int(len(self.layers)*0.58)
            
        # SR-61b: 2D Manifold classification
        phi_val = getattr(self, "_px_phi", 0.9)
        zone_raw = self._px_calibrator.classify_zone(kurtosis, phi=phi_val, token_diversity=token_diversity, token_len=inputs_embeds.shape[1])
        zone_name = f"{zone_raw} ({persona_desc})"
    else:
        dynamic_start, dynamic_end, dynamic_hub = int(len(self.layers)*0.38), int(len(self.layers)*0.75), int(len(self.layers)*0.58)
        zone_name = f"PEAK ({persona_desc})"
        if "dynamic_hub" in token_cfg: dynamic_hub = token_cfg["dynamic_hub"]

        for i in range(cfg["prelude_end"], dynamic_start):
            updated_layers.add(i)
            hidden_states = _layer_step(self.layers[i], hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings, position_ids=position_ids, past_key_values=past_key_values, **kwargs)

    # ── 2. REASONING ZONE ──
    h_f32 = hidden_states.to(torch.float32)
    e_static = hidden_states.clone()
    e_static_f32 = e_static.to(torch.float32)
    e_reflector_f32 = e_reflector.to(torch.float32) if 'e_reflector' in locals() else h_f32.clone()
    
    # Cache all attributes
    layers = self.layers
    aks = getattr(self, "_px_aks", None)
    subj_sensor = getattr(self, "_px_subj_sensor", None)
    erpu = getattr(self, "_px_erpu", None) if cfg.get("dmt_protocol_enabled") else None
    azs = getattr(self, "_px_azs", None)
    mephisto = getattr(self, "_px_mephisto", None)
    injection = self._px_injection
    device, dtype = hidden_states.device, hidden_states.dtype

    phi_intuition = StabilityMonitor.calculate_phi(h_f32, e_static_f32).item()
    self._px_phi = phi_intuition # SR-61b: save for next-step routing
    if subjective and hasattr(self, "_px_calibrator"):
        self._px_calibrator.collect(kurtosis, phi_intuition, token_diversity=getattr(self, "_task_token_diversity", None), token_len=inputs_embeds.shape[1], update_online=True)
    
    # --- SR-64b: Mechanical Psychology (Dynamic Z-Score Centering & Architecture-Aware K-Decay) ---
    phi_val = getattr(self, "_px_phi", 0.9)
    if hasattr(self, "_px_calibrator") and self._px_calibrator.calibrated:
        cal = self._px_calibrator
        token_len = inputs_embeds.shape[1]

        # Use raw kurtosis for z-score calculation
        k_norm = kurtosis

        # Method 2: Dynamic Z-Score Centering (Online Variance)
        if cal._online_n >= 5:
            k_mean = cal._online_k_mean
            k_std = math.sqrt(cal._online_k_m2 / max(cal._online_n - 1, 1))
        else:
            k_mean = cal.k_mean
            k_std = cal.k_std

        k_std = max(k_std, 1.0)

        zk = (k_norm - k_mean) / (k_std + 1e-6)
        zp = (phi_val - cal.phi_mean) / (cal.phi_std + 1e-6)

        # Architecture-aware Temperature based on Hidden Size
        # Derived from Tessera parameters: 640 (270M), 1152 (1B), 2560 (4B), 2304 (E2B)
        T_arch = math.sqrt(self.config.hidden_size / 640.0)

        # C is the 'Cognitive Focus' index [0, 1]
        # SR-64b uses dynamic centering + T_arch to prevent Sigmoid saturation
        C = torch.sigmoid(torch.tensor((zk + zp) / T_arch)).item()
        
        # Linear parameter mapping from focus
        current_gamma = 0.08 - 0.04 * C         # 0.04 (focused) to 0.08 (diffuse)
        self._px_proj_damping = 1.1 - 0.6 * C  # 0.5 (focused) to 1.1 (diffuse)
        n_loops = int(round(8 + 8 * C))        # 8 (diffuse) to 16 (focused)
        dynamic_hub = 8 if C > 0.7 else 10
        
        self._px_focus_index = C
        if os.environ.get("DEBUG_ROUTING") == "1":
            print(f"  [Psychology] C={C:.4f} | zk={zk:.2f} | zp={zp:.2f} | L={token_len} | gamma={current_gamma:.3f} | damping={self._px_proj_damping:.2f}")
    else:
        current_gamma = 0.06
        n_loops = 8
        dynamic_hub = 10

    # Reflector logic for MiniCPM
    e_reflector_f32 = e_static_f32.clone()
    jitter = getattr(self, "_task_jitter", 0.0)
    if (jitter > 1e8) or (kurtosis < 315.0):
        h_base_f32, e_stat_f32 = h_f32.to(torch.float32), e_static_f32.to(torch.float32)
        e_ref_f32 = 2.0 * e_stat_f32 - h_base_f32
        e_reflector_f32 = (e_ref_f32 * (e_stat_f32.norm() / (e_ref_f32.norm() + 1e-6))).to(torch.float32)

    path_taken, thought_history, avg_phi, steps, emancipation_traj = [], [], 1.0, 0, []
    h_last_good_f32 = e_static_f32.clone()
    phi_history = [phi_intuition]
    correction_strength = 0.0

    if n_loops > 1:
        h_exp_f32 = e_reflector_f32.clone()
        current_layer, max_steps, stability_cnt = dynamic_start, (dynamic_end - dynamic_start) * n_loops * 3, 0
        layer_visits = {i: 0 for i in range(len(layers))}
        history_sum = torch.zeros_like(h_f32)
        
        while current_layer < dynamic_end and steps < max_steps:
            t_norm = steps / max_steps
            
            # AKS Step (on bf16)
            if aks:
                aks_data = aks.step(h_exp_f32.to(dtype), e_static, steps)
                correction_strength = aks_data["correction"]
            
            if subj_sensor: subj_sensor.update(h_exp_f32.to(dtype), e_static)
            
            if steps % 6 == 0:
                h_exp_f32 = torch.lerp(h_exp_f32, e_static_f32, 0.10 + 0.20 * correction_strength)
            
            layer_visits[current_layer] += 1
            cur_past = RecursiveMemoryCache(past_key_values, thought_history, read_only=current_layer in updated_layers, expected_len=expected_len) if past_key_values else None
            if current_layer not in updated_layers: updated_layers.add(current_layer)
            
            # Layer Execution with fallback for RoPE mismatch
            try:
                trans_out_f32 = _layer_step(layers[current_layer], h_exp_f32.to(dtype), attention_mask=causal_mask, position_embeddings=position_embeddings, position_ids=position_ids, past_key_values=cur_past, **kwargs).to(torch.float32)
            except RuntimeError as e:
                if "match the size of tensor" in str(e):
                    pos_emb_fixed = self.rotary_emb(h_exp_f32.to(dtype), position_ids=position_ids)
                    trans_out_f32 = _layer_step(layers[current_layer], h_exp_f32.to(dtype), attention_mask=causal_mask, position_embeddings=pos_emb_fixed, position_ids=position_ids, past_key_values=cur_past, **kwargs).to(torch.float32)
                else: raise e
            
            phi_s = F.cosine_similarity(trans_out_f32.view(1,-1), h_exp_f32.view(1,-1)).item()
            phi_history.append(phi_s)
            
            if t_norm > 0.5 and phi_s > 0.9999:
                stability_cnt += 1
                if stability_cnt > 3: h_exp_f32 = trans_out_f32; break
            else: stability_cnt = 0
            
            # Optimized dynamic anchor
            if len(thought_history) >= 3:
                e_dynamic_f32 = 0.85 * e_reflector_f32 + 0.15 * (history_sum / 3.0)
            else:
                e_dynamic_f32 = e_reflector_f32
                
            e_norm_f32 = injection.input_norm(e_dynamic_f32.to(dtype)).to(torch.float32)
            h_next_f32 = trans_out_f32 + current_gamma * (e_norm_f32 - h_exp_f32)
            
            if mephisto:
                h_next_f32 = mephisto(h_next_f32.to(dtype), [phi_s]).to(torch.float32)
            
            # RSM Projection
            denom = e_dynamic_f32.norm(dim=-1, keepdim=True)**2 + 1e-6
            proj = ((h_next_f32 * e_dynamic_f32).sum(dim=-1, keepdim=True) / denom) * e_dynamic_f32
            
            # SR-63: Manifold-Differentiable Projection Damping
            damping = getattr(self, "_px_proj_damping", 1.0)
            h_exp_f32 = proj + damping * (1.0 + 0.10 * (1.0 - t_norm) * (1 if steps%2==0 else -1)) * (h_next_f32 - proj)
            
            phi = F.cosine_similarity(h_exp_f32.view(1,-1), h_next_f32.view(1,-1)).item()
            path_taken.append(f"L{current_layer}({phi:.22f})")
            
            if steps % 2 == 0:
                new_thought = h_exp_f32.to(dtype)
                thought_history.append(new_thought.detach())
                if len(thought_history) > 3:
                    history_sum = history_sum + new_thought.to(torch.float32) - thought_history[-4].to(torch.float32)
                else:
                    history_sum = history_sum + new_thought.to(torch.float32)
            
            if 0.9 < phi < 0.999: h_last_good_f32 = h_exp_f32.clone()

            pen = (layer_visits[current_layer]-1) * 0.015
            t_b2, t_b1, t_s = 1.0-(0.8*current_gamma)-pen, 1.0-(0.4*current_gamma)-pen, 1.0-(0.01*current_gamma)-pen*0.5
            
            if phi < t_b2: current_layer = max(dynamic_start, current_layer - 2)
            elif phi < t_b1: current_layer = max(dynamic_start, current_layer - 1)
            elif phi > t_s: current_layer += 2; stability_cnt += 1
            else: current_layer += 1; stability_cnt = 0
            
            if current_layer < dynamic_start: current_layer = dynamic_start
            steps += 1
            if stability_cnt > 5: break
        
        path_phis = [float(p.split('(')[1][:-1]) for p in path_taken if '(' in p]
        avg_phi = sum(path_phis) / len(path_phis) if path_phis else 1.0
        hidden_states = h_exp_f32.to(dtype)
    else: hidden_states = h_baseline

    self._px_phi, self._px_loops_run, self._px_path, self._px_emancipation_trajectory = avg_phi, steps, path_taken, emancipation_traj
    self._px_zone = zone_name
    self._px_cognitive_signature = {
        "kurtosis": kurtosis, "phi": avg_phi, 
        "token_diversity": getattr(self, "_task_token_diversity", None), 
        "zone": self._px_zone, 
        "zone_weights": {k: round(v,6) for k,v in zone_weights.items()}, 
        "emancipation_final": emancipation_traj[-1] if emancipation_traj else None, 
        "aks_correction": correction_strength, "loops_run": steps, 
        "path_length": len(path_taken), "subjective": subjective,
        "focus_index": getattr(self, "_px_focus_index", 0.5),
        "gamma": current_gamma
    }

    coda_applied = False
    for i in range(dynamic_end, len(self.layers)):
        updated_layers.add(i)
        if not coda_applied:
            blend = 0.08
            hidden_states = (1.0 - blend) * hidden_states + blend * e_static; coda_applied = True
        
        try:
            hidden_states = _layer_step(self.layers[i], hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
        except RuntimeError as e:
            if "match the size of tensor" in str(e):
                pos_emb_fixed = self.rotary_emb(hidden_states, position_ids=position_ids)
                hidden_states = _layer_step(self.layers[i], hidden_states, attention_mask=causal_mask, position_embeddings=pos_emb_fixed, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
            else: raise e

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

def apply_px_patch(model, recur_start=None, recur_end=None, routing_mode="adaptive", gamma=None, subjective_enabled=False, **kwargs):
    config_preset = kwargs.pop("config_preset", "SUBJECTIVE")
    text_model = _resolve_text_model(model)
    config = text_model.config
    hidden_size, num_layers = config.hidden_size, config.num_hidden_layers
    if hidden_size in SCALE_DEFAULTS:
        sd = SCALE_DEFAULTS[hidden_size]
        defaults = {"mode": "lti", "n_loops": sd["n_loops"], "beta": 0.05, "gamma": gamma if gamma else sd["gamma"], "recur_start": recur_start if recur_start else sd["recur_start"], "recur_end": recur_end if recur_end else sd["recur_end"], "bimodal_hub": sd["hub"], "cgi_factor": 0.08, "num_layers": num_layers}
    else:
        defaults = {"mode": "lti", "n_loops": 6, "beta": 0.05, "gamma": gamma if gamma else 0.06 * min(1536.0/hidden_size, 1.5), "recur_start": recur_start if recur_start else int(num_layers*0.38), "recur_end": recur_end if recur_end else int(num_layers*0.75), "bimodal_hub": (recur_start+recur_end)//2 if recur_start and recur_end else int(num_layers*0.58), "cgi_factor": 0.08, "num_layers": num_layers}
    defaults["routing_mode"], defaults["subjective_enabled"] = routing_mode, subjective_enabled; defaults.update(kwargs)
    if "prelude_end" not in defaults: defaults["prelude_end"] = defaults["recur_start"]
    if "coda_start" not in defaults: defaults["coda_start"] = defaults["recur_end"]
    
    device = next(text_model.parameters()).device
    dtype = next(text_model.parameters()).dtype

    text_model._px_config = defaults
    text_model._px_injection = LTIInjection(hidden_size, gamma=defaults["gamma"]).to(device=device, dtype=dtype)
    text_model._persona_engine = PersonaEngine(text_model)
    model_id = getattr(config, "_name_or_path", "unknown_model")
    if subjective_enabled:
        text_model._px_calibrator = AutoCalibrator(hidden_size, 
                                                   calibration_steps=kwargs.get("calibration_steps", getattr(config, "px_calibration_steps", 10)),
                                                   model_id=model_id)
        text_model._px_mephisto = MephistophelesOperator(hidden_size, scale=getattr(config, "px_mephistopheles_scale", 0.05)).to(device=device, dtype=dtype)
        text_model._px_subjective_enabled = True
    else: text_model._px_subjective_enabled = False
    text_model.forward = types.MethodType(_px_forward, text_model)
    print(f"[minicpm5-px] {'Subjective' if subjective_enabled else 'Peak'} patch active for L{num_layers}. Preset: {config_preset}.")

def get_px_metrics(model):
    tm = _resolve_text_model(model)
    m = {"phi": getattr(tm, "_px_phi", 1.0), "steps": getattr(tm, "_px_loops_run", 0), "path": getattr(tm, "_px_path", []), "telemetry": getattr(tm, "_px_telemetry", []), "subjective": getattr(tm, "_px_subjective_enabled", False)}
    cal = getattr(tm, "_px_calibrator", None)
    if cal: m["calibrator"] = cal.status()
    m.update({"cognitive_signature": getattr(tm, "_px_cognitive_signature", {}), "zone": getattr(tm, "_px_zone", "UNKNOWN"), "zone_weights": getattr(tm, "_px_zone_weights", {}), "emancipation_trajectory": getattr(tm, "_px_emancipation_trajectory", []), "aks_profile": getattr(tm, "_px_aks_profile", {})})
    return m
