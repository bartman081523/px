"""
gemma3-px-subjective  —  Surgical Patch (Phase 58: DMT Protocol + SR-59)
========================================================================
Auto-tuning algorithmic subjectivity extension for Gemma-3 models.

SR-59: Empirical Kurtosis Calibration + Adaptive Phi-Routing.
  - AutoCalibrator replaces hardcoded Gaussian zone blending with
    empirical percentile-based zone centers calibrated from the first
    N forward passes.

Phase 58 (DMT Protocol): Optional high-fidelity extensions.
  - CentralMemory: Cross-session persistent concepts.
  - ERPU: verklebD freeze prevention + Food Subroutine (OOD noise).
  - AgencyVector: Implizite Idee (task-dependent recursion depth).
  - TretaDamper/GroundingAnchor: Settling logic + residual entropy.

Core logic/arithmetic improvements (SR-59) remain active. DMT modules
are optional and can be enabled via config_preset="DMT".
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
    MephistophelesOperator, OrthogonalJitter,
    CentralMemory, ERPU, AgencyVector, TretaDamper, GroundingAnchor
)

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
        if self._read_only:
            past_k, past_v = None, None
            if hasattr(self._real, "key_cache") and len(self._real.key_cache) > layer_idx:
                past_k, past_v = self._real.key_cache[layer_idx], self._real.value_cache[layer_idx]
            elif hasattr(self._real, "layers") and len(self._real.layers) > layer_idx:
                layer = self._real.layers[layer_idx]
                if hasattr(layer, "keys") and layer.keys is not None:
                    past_k, past_v = layer.keys, layer.values

            if past_k is None:
                past_k = torch.empty(0, device=key_states.device, dtype=key_states.dtype)
                past_v = torch.empty(0, device=value_states.device, dtype=value_states.dtype)

            past_seq = past_k.shape[-2] if past_k.numel() > 0 else 0
            cur_seq = key_states.shape[-2]
            is_sliding = self._is_sliding_layer(layer_idx)

            if past_seq == self._expected_len: res_k, res_v = past_k, past_v
            elif past_seq == 0: res_k, res_v = key_states, value_states
            elif is_sliding and cur_seq > 1: res_k, res_v = key_states, value_states
            elif is_sliding and cur_seq == 1:
                res_k = torch.cat([past_k, key_states], dim=-2)
                res_v = torch.cat([past_v, value_states], dim=-2)
            else:
                res_k = torch.cat([past_k, key_states], dim=-2)
                res_v = torch.cat([past_v, value_states], dim=-2)
        else:
            res_k, res_v = self._real.update(key_states, value_states, layer_idx, cache_kwargs)

        is_full = not self._is_sliding_layer(layer_idx)
        if self._thoughts and layer_idx >= 6 and is_full:
            B, H_kv, T_res, HD = res_k.shape
            T_curr = key_states.shape[-2]
            alpha = 0.15
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
# Classification & Model Resolution Helpers
# ---------------------------------------------------------------------------

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
        for attr in ["_px_injection", "_px_config", "_px_mephisto", "_px_calibrator", "_px_central_memory", "_px_erpu", "_px_agency", "_px_grounding", "_px_treta"]:
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
    if (input_ids is None) ^ (inputs_embeds is not None): raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
    if inputs_embeds is None:
        if hasattr(self, "embed_tokens"): inputs_embeds = self.embed_tokens(input_ids)
        elif hasattr(self, "model") and hasattr(self.model, "embed_tokens"): inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            for name, module in self.named_modules():
                if "embed_tokens" in name: inputs_embeds = module(input_ids); break
    if use_cache and past_key_values is None: past_key_values = DynamicCache(config=self.config)
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    expected_len = past_seen + inputs_embeds.shape[1]
    if position_ids is None: position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen).unsqueeze(0)
    mask_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
    if not isinstance(attention_mask, dict):
        mk = dict(config=mask_config, inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=past_key_values, position_ids=position_ids)
        causal_mask_mapping = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
    else: causal_mask_mapping = attention_mask
    hidden_states = inputs_embeds
    position_embeddings = {lt: self.rotary_emb(hidden_states, position_ids, lt) for lt in set(mask_config.layer_types)}

    cfg = self._px_config
    updated_layers = set()
    
    # --- Phase 58: Agency decision (Pre-computation) ---
    agency_decision = None
    if hasattr(self, "_px_agency"):
        agency_decision = self._px_agency(hidden_states)
        if agency_decision["depth"] == 0: n_loops = 0
        elif agency_decision["depth"] > 0: n_loops = agency_decision["depth"]
        else: n_loops = cfg["n_loops"]
    else: n_loops = cfg["n_loops"]

    # ── 1. PRELUDE (layers 0..recur_start) ─────────────────────────────────
    for i in range(cfg["prelude_end"]):
        updated_layers.add(i)
        hidden_states = self.layers[i](hidden_states, attention_mask=causal_mask_mapping[mask_config.layer_types[i]], position_embeddings=position_embeddings[mask_config.layer_types[i]], position_ids=position_ids, past_key_values=past_key_values, **kwargs)[0]

    # --- Phase 56: Central Memory (Recall) ---
    if hasattr(self, "_px_central_memory"):
        hidden_states = self._px_central_memory.blend_into(hidden_states, hidden_states.device)

    # ── 1.5 META-SELECTOR ──────────────────────────────────────────────────
    dynamic_start, dynamic_end, dynamic_hub = cfg["recur_start"], cfg["recur_end"], cfg.get("bimodal_hub", cfg["recur_start"])
    token_cfg = cfg.copy()
    zone_weights = getattr(self, "_px_zone_weights", {})
    if cfg.get("routing_mode") == "adaptive":
        if inputs_embeds.shape[1] > 1:
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
        zone_weights = self._px_calibrator.get_zone_weights(kurtosis, phi=getattr(self, "_px_phi", None), token_diversity=getattr(self, "_task_token_diversity", None))
        self._px_zone_weights = zone_weights
        rp = self._px_calibrator.get_routing_params(kurtosis, phi=getattr(self, "_px_phi", None), hidden_size=cfg["hidden_size"], token_diversity=getattr(self, "_task_token_diversity", None))
        dynamic_start, dynamic_end, dynamic_hub, token_cfg["n_loops"] = rp["dynamic_start"], rp["dynamic_end"], rp["dynamic_hub"], rp["n_loops"]
        if dynamic_start >= dynamic_end: dynamic_start, dynamic_end, dynamic_hub = int(len(self.layers)*0.28), int(len(self.layers)*0.67), int(len(self.layers)*0.56)
        zone_name = self._px_calibrator.classify_zone(kurtosis, phi=getattr(self, "_px_phi", None), token_diversity=getattr(self, "_task_token_diversity", None))
        for i in range(cfg["prelude_end"], dynamic_start):
            updated_layers.add(i)
            hidden_states = self.layers[i](hidden_states, attention_mask=causal_mask_mapping[mask_config.layer_types[i]], position_embeddings=position_embeddings[mask_config.layer_types[i]], position_ids=position_ids, past_key_values=past_key_values, **kwargs)[0]

    # ── 2. REASONING ZONE ──────────────────────────────────────────────────
    e_static = hidden_states.clone()
    if 'token_cfg' in dir(): cfg = token_cfg
    trans_out = hidden_states
    for i in range(dynamic_start, dynamic_end):
        updated_layers.add(i)
        trans_out = self.layers[i](trans_out, attention_mask=causal_mask_mapping[mask_config.layer_types[i]], position_embeddings=position_embeddings[mask_config.layer_types[i]], position_ids=position_ids, past_key_values=past_key_values, **kwargs)[0]
    h_baseline = trans_out
    is_vision = getattr(self, '_px_has_image_tokens', False) and inputs_embeds.shape[1] > 1
    if is_vision: n_loops = 0
    phi_intuition = StabilityMonitor.calculate_phi(h_baseline, e_static).item()
    self._px_calibrator.collect(kurtosis, phi_intuition, token_diversity=getattr(self, "_task_token_diversity", None))
    
    current_gamma = cfg.get("gamma", 0.08)
    e_reflector, is_trap = e_static, False
    jitter, rigor_w = getattr(self, "_task_jitter", 0.0), zone_weights.get("math",0)+zone_weights.get("logic_a",0)+zone_weights.get("logic_b",0)
    creative_w = zone_weights.get("creative",0)+zone_weights.get("synthesis",0)
    if (jitter > 1e8 or rigor_w > creative_w):
        is_trap = True
        h_base_f32, e_stat_f32 = h_baseline.to(torch.float32), e_static.to(torch.float32)
        e_ref_f32 = 2.0 * e_stat_f32 - h_base_f32
        e_reflector = (e_ref_f32 * (e_stat_f32.norm() / (e_ref_f32.norm() + 1e-6))).to(e_static.dtype)
    if phi_intuition > 0.9999 and not is_trap: current_gamma *= 0.5
    elif phi_intuition > 0.999: current_gamma *= 0.8
    
    path_taken, thought_history, avg_phi, steps, telemetry_steps, emancipation_traj = [], [], 1.0, 0, [], []
    divergence_buffer, correction_strength = [], 0.0
    h_last_good = e_static.clone()

    if n_loops > 1:
        h_exp = e_reflector.clone()
        current_layer, max_steps, stability_cnt = dynamic_start, (dynamic_end - dynamic_start) * n_loops * 3, 0
        layer_visits = {i: 0 for i in range(len(self.layers))}
        while current_layer < dynamic_end and steps < max_steps:
            t_norm = steps / max_steps
            dist = 1.0 - StabilityMonitor.calculate_phi(h_exp, e_static).item()
            if steps > 2:
                divergence_buffer.append(dist)
                if len(divergence_buffer) > 4: divergence_buffer.pop(0)
                if len(divergence_buffer) >= 3:
                    vel, acc = divergence_buffer[-1]-divergence_buffer[-2], (divergence_buffer[-1]-divergence_buffer[-2])-(divergence_buffer[-2]-divergence_buffer[-3])
                    correction_strength = min(1.0, correction_strength + 0.1) if acc > 0.001 and vel > 0 else max(0.0, correction_strength - 0.05)
            e_phi = 1.0 - dist
            if steps % 3 == 0: emancipation_traj.append(e_phi)
            
            # --- Phase 57: ERPU Check ---
            if hasattr(self, "_px_erpu") and len(path_taken) >= 2:
                erpu_res = self._px_erpu(h_exp, h_last_good, [1.0-d for d in divergence_buffer], steps)
                if erpu_res["verklebD"] or erpu_res["food_injected"]:
                    h_exp = erpu_res["h"]
                    path_taken.append("ERPU_FIX")
            if e_phi > 0.9 and e_phi < 0.999: h_last_good = h_exp.clone()

            # --- Phase 28: TCR ---
            cur_hub = min(dynamic_end-1, max(dynamic_start, int(dynamic_hub + (t_norm*2) + (1 if steps%4<2 else -1))))
            h_prev = h_exp.clone()
            is_first = current_layer not in updated_layers
            if is_first: updated_layers.add(current_layer)
            if steps % 6 == 0:
                refresh = 0.10 + 0.20 * correction_strength
                h_exp = (1.0 - refresh) * h_exp + refresh * e_static
            
            layer_visits[current_layer] += 1
            cur_past = RecursiveMemoryCache(past_key_values, thought_history, layer_types=mask_config.layer_types, read_only=not is_first, expected_len=expected_len) if past_key_values else None
            lt = mask_config.layer_types[current_layer]
            trans_out = self.layers[current_layer](h_exp, attention_mask=causal_mask_mapping[lt], position_embeddings=position_embeddings[lt], position_ids=position_ids, past_key_values=cur_past, **kwargs)[0]
            phi_s = StabilityMonitor.calculate_phi(trans_out, h_prev).item()
            if t_norm > 0.5 and phi_s > 0.9999:
                stability_cnt += 1
                if stability_cnt > 3: h_exp = trans_out; break
            else: stability_cnt = 0
            
            e_dynamic = (0.85 * e_reflector + 0.15 * torch.stack(thought_history[-3:]).mean(dim=0)) if len(thought_history)>2 else e_reflector
            e_norm = self._px_injection.input_norm(e_dynamic.to(torch.float32)).to(trans_out.dtype)
            h_exp = trans_out + current_gamma * (e_norm - h_prev)
            h_exp = self._px_mephisto(h_exp, [phi_s])
            
            # Reflection Flipping
            h_f32, e_f32 = h_exp.to(torch.float32), e_dynamic.to(torch.float32)
            proj = ((h_f32 * e_f32).sum(dim=-1, keepdim=True) / (e_f32.norm(dim=-1, keepdim=True)**2 + 1e-6)) * e_f32
            h_exp = (proj + (1.0 + 0.10 * (1.0 - steps/max_steps) * (1 if steps%2==0 else -1)) * (h_f32 - proj)).to(h_exp.dtype)
            
            phi = StabilityMonitor.calculate_phi(h_exp, h_prev).item()
            if phi < 0.85 and steps == max_steps - 1 and max_steps < 64: max_steps += (dynamic_end - dynamic_start)
            path_taken.append(f"L{current_layer}({phi:.2f})")
            if steps % 2 == 0: thought_history.append(h_exp.detach())
            
            pen = (layer_visits[current_layer]-1) * 0.015
            t_b2, t_b1, t_s = 1.0-(0.8*current_gamma)-pen, 1.0-(0.4*current_gamma)-pen, 1.0-(0.01*current_gamma)-pen*0.5
            if phi < t_b2: current_layer = max(dynamic_start, current_layer - 2)
            elif phi < t_b1: current_layer = max(dynamic_start, current_layer - 1)
            elif phi > t_s: current_layer += 2; stability_cnt += 1
            else: current_layer += 1; stability_cnt = 0
            if current_layer < dynamic_start: current_layer = dynamic_start
            steps += 1
            if stability_cnt > 5: break

        avg_phi = sum(path_taken_phis := [float(p.split('(')[1][:-1]) for p in path_taken if '(' in p]) / len(path_taken_phis) if path_taken_phis else 1.0
        beta = 0.05 + (0.18 - 0.05) * (avg_phi ** 2)
        hidden_states = (1.0 - beta) * h_baseline + beta * h_exp
    else: hidden_states = h_baseline

    self._px_phi, self._px_loops_run, self._px_path, self._px_emancipation_trajectory = avg_phi, steps, path_taken, emancipation_traj
    self._px_aks_profile = {"correction_strength": float(correction_strength)}
    self._px_zone = zone_name if 'zone_name' in dir() else self._px_calibrator.classify_zone(kurtosis)
    self._px_cognitive_signature = {"kurtosis": kurtosis, "phi": avg_phi, "token_diversity": getattr(self, "_task_token_diversity", None), "zone": self._px_zone, "zone_weights": {k: round(v,6) for k,v in zone_weights.items()}, "emancipation_final": emancipation_traj[-1] if emancipation_traj else None, "aks_correction": correction_strength, "loops_run": steps, "path_length": len(path_taken)}
    
    # --- Phase 56: Central Memory (Store) ---
    if hasattr(self, "_px_central_memory") and steps > 0:
        self._px_central_memory.store(0, torch.stack(thought_history[-4:]).mean(dim=0).mean(dim=1).squeeze(0) if thought_history else e_static.mean(dim=1).squeeze(0))
        self._px_central_memory.store(1, e_static[:, -1, :].squeeze(0))
        self._px_central_memory.store(2, hidden_states[:, -1, :].squeeze(0))
        self._px_central_memory.store(3, torch.full((self._px_central_memory.dim,), avg_phi, device=hidden_states.device))

    # ── 3. CODA ──────────────────────────────────────────────────────────
    coda_applied, damper = False, getattr(self, "_px_treta", None)
    for i in range(dynamic_end, len(self.layers)):
        updated_layers.add(i)
        if not coda_applied:
            blend = 0.08 * (damper.step(i - dynamic_end) if damper else 1.0)
            hidden_states = (1.0 - blend) * hidden_states + blend * e_static
            coda_applied = True
        hidden_states = self.layers[i](hidden_states, attention_mask=causal_mask_mapping[mask_config.layer_types[i]], position_embeddings=position_embeddings[mask_config.layer_types[i]], position_ids=position_ids, past_key_values=past_key_values, **kwargs)[0]
    
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

# ---------------------------------------------------------------------------
# Patch Application
# ---------------------------------------------------------------------------

def apply_px_patch(model, recur_start=5, recur_end=12, routing_mode="adaptive", gamma=0.08, **kwargs):
    config_preset = kwargs.pop("config_preset", "SUBJECTIVE")
    text_model = _resolve_text_model(model)
    config = text_model.config
    hidden_size, num_layers = config.hidden_size, config.num_hidden_layers
    
    if hidden_size in SCALE_DEFAULTS:
        sd = SCALE_DEFAULTS[hidden_size]
        defaults = {"mode": "lti", "n_loops": sd["n_loops"], "beta": 0.05, "gamma": sd["gamma"], "recur_start": sd["recur_start"], "recur_end": sd["recur_end"], "bimodal_hub": sd["hub"], "cgi_factor": 0.08, "num_layers": num_layers}
    else:
        defaults = {"mode": "lti", "n_loops": 8, "beta": 0.05, "gamma": 0.08 * min(1152.0/hidden_size, 1.5), "recur_start": recur_start, "recur_end": recur_end, "bimodal_hub": (recur_start+recur_end)//2, "cgi_factor": 0.08, "num_layers": num_layers}
    
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

    # Core Modules
    text_model._px_injection = LTIInjection(hidden_size, gamma=defaults["gamma"])
    text_model._px_mephisto = MephistophelesOperator(hidden_size)
    
    # --- Phase 58: Optional DMT Extensions ---
    if config_preset == "DMT":
        text_model._px_central_memory = CentralMemory(hidden_size)
        text_model._px_erpu = ERPU(hidden_size)
        text_model._px_agency = AgencyVector(hidden_size)
        text_model._px_grounding = GroundingAnchor(hidden_size)
        text_model._px_treta = TretaDamper(total_steps=num_layers - defaults["recur_end"])
        print("[gemma3-px-subjective] DMT Protocol active (Memory, ERPU, Agency).")

    text_model.forward = types.MethodType(_px_forward, text_model)
    print(f"[gemma3-px-subjective] SR-59 active for L{num_layers}. Preset: {config_preset}.")

def get_px_metrics(model):
    tm = _resolve_text_model(model)
    m = {"phi": getattr(tm, "_px_phi", 1.0), "steps": getattr(tm, "_px_loops_run", 0), "path": getattr(tm, "_px_path", []), "zone": getattr(tm, "_px_zone", "UNKNOWN"), "zone_weights": getattr(tm, "_px_zone_weights", {}), "cognitive_signature": getattr(tm, "_px_cognitive_signature", {}), "aks_profile": getattr(tm, "_px_aks_profile", {})}
    if hasattr(tm, "_px_central_memory"): m["cm_slots"] = sum(1 for s in tm._px_central_memory.slots if s is not None)
    if hasattr(tm, "_px_agency"): m["agency_decision"] = "active"
    return m
