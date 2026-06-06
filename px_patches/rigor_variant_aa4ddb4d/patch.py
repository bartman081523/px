"""
minicpm5-px  —  Surgical Patch for LlamaForCausalLM (MiniCPM5-1B)
================================================================
Ported from Gemma3 PX Subjective (SR-59i) and Peak patches.

Key architectural differences from Gemma3:
  - Single causal_mask (not a dict of full+sliding masks)
  - No layer_types — all layers use full attention
  - RoPE: rotary_emb(h, position_ids=pos) → (cos, sin) tuple
  - No text_config wrapper, no multimodal code
  - head_dim=128 (not 256), num_layers=24

Subjective optional:
  - px_subjective_enabled=False (default): Peak mode (core only)
  - px_subjective_enabled=True: All SR-59i Subjective features active
    (MephistophelesOperator, OrthogonalJitter, AutoCalibrator zone routing)

Counterfactual extensions NOT ported (SR-59i: CentralMemory, ERPU,
AgencyVector, TretaDamper, GroundingAnchor removed — they altered
hidden_states without contributing to Zombie/Anti-Zombie measurement).
"""

import types
import math
import torch
import torch.nn as nn
import os
import json
import datetime
from typing import Optional, Dict, List, Any

try:
    from .auto_tune import AutoCalibrator, SCALE_DEFAULTS
    from .px_modules import (
        LTIInjection, ADCInjection, StabilityMonitor, CognitiveEvent,
        MephistophelesOperator, OrthogonalJitter,
    )
except ImportError:
    # Standalone execution (e.g., from test files)
    from auto_tune import AutoCalibrator, SCALE_DEFAULTS
    from px_modules import (
        LTIInjection, ADCInjection, StabilityMonitor, CognitiveEvent,
        MephistophelesOperator, OrthogonalJitter,
    )


# ---------------------------------------------------------------------------
# p10.0: Recursive State Memory (RSM) — Llama-adapted
# ---------------------------------------------------------------------------

class RecursiveMemoryCache:
    """
    Extends ReadOnlyCache by injecting previous thinking steps into the
    self-attention key/value streams.

    Llama adaptation: No layer_types (all full attention), no sliding
    window handling. Soft-RSM blending always active for layers ≥ 6.
    """
    def __init__(self, real_cache, thought_history=None, read_only=False, expected_len=0):
        self.__dict__["_real"] = real_cache
        self.__dict__["_thoughts"] = thought_history or []
        self.__dict__["_read_only"] = read_only
        self.__dict__["_expected_len"] = expected_len

    def __getattr__(self, name):
        return getattr(self._real, name)

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

            if past_seq == self._expected_len:
                # Cache already complete for this token
                res_k, res_v = past_k, past_v
            elif past_seq == 0:
                res_k = key_states
                res_v = value_states
            elif past_seq > self._expected_len:
                res_k, res_v = past_k, past_v
            else:
                # Concatenate for partial cache
                res_k = torch.cat([past_k, key_states], dim=-2)
                res_v = torch.cat([past_v, value_states], dim=-2)

        else:
            res_k, res_v = self._real.update(key_states, value_states, layer_idx, cache_kwargs)

        # 2. Soft-RSM (Semantic Blending) — always active for layers >= 6
        # Llama: no layer_types check needed (all full attention)
        if self._thoughts and layer_idx >= 6:
            B, H_kv, T_res, HD = res_k.shape
            T_curr = key_states.shape[-2]
            alpha = 0.15

            # Triangular Weighting (Emphasize the 'reasoning peak')
            n_t = len(self._thoughts[-6:])
            if n_t > 2:
                weights = torch.cat([
                    torch.linspace(0.4, 1.0, n_t // 2, device=res_k.device),
                    torch.linspace(1.0, 0.6, n_t - n_t // 2, device=res_k.device)
                ])
                t_raw = (torch.stack(self._thoughts[-6:]) * weights.view(-1, 1, 1, 1)).sum(dim=0) / weights.sum()
            else:
                t_raw = torch.stack(self._thoughts).mean(dim=0)

            D = t_raw.shape[2]

            # Project thought to Head Dim (SDA)
            t_flat = t_raw.mean(dim=1, keepdim=True)  # (B, 1, D)
            t_proj = torch.nn.functional.interpolate(t_flat, size=HD, mode='linear', align_corners=False)
            t_k = t_proj.unsqueeze(1)  # (B, 1, 1, HD)
            t_v = -t_k

            # Blend into the LAST token(s) of the result
            if self._read_only:
                res_k = res_k.clone()
                res_v = res_v.clone()

            res_k[:, :, -T_curr:, :] = (1.0 - alpha) * res_k[:, :, -T_curr:, :] + alpha * t_k
            res_v[:, :, -T_curr:, :] = (1.0 - alpha) * res_v[:, :, -T_curr:, :] + alpha * t_v

        return res_k, res_v


# ---------------------------------------------------------------------------
# Zone Classification Helpers
# ---------------------------------------------------------------------------

def classify_zone_kurtosis(weights):
    """Kurtosis-based zone classification from Gaussian/empirical weights."""
    m = weights.get("math", 0)
    la = weights.get("logic_a", 0)
    cr = weights.get("creative", 0)
    lb = weights.get("logic_b", 0)
    sy = weights.get("synthesis", 0)
    if m > max(cr, la, lb, sy):
        return "MATH"
    elif (la + lb) > max(m, cr, sy):
        return "LOGIC"
    elif cr > max(m, la, lb, sy):
        return "CREATIVE"
    elif sy > max(m, la, lb, cr):
        return "SYNTHESIS"
    elif (m + la + lb) > (cr + sy):
        return "RIGOR-blend"
    else:
        return "CREATIVE-blend"


def classify_zone_phi(phi):
    """Phi-based zone classification."""
    if phi is None:
        return "UNKNOWN"
    if phi > 0.85:
        return "GROUNDED"
    elif phi > 0.75:
        return "ANALYTICAL"
    elif phi > 0.65:
        return "EXPLORATORY"
    else:
        return "CREATIVE"


# ---------------------------------------------------------------------------
# Patch removal
# ---------------------------------------------------------------------------

def remove_px_patch(model) -> None:
    """Remove the PX patch and restore original LlamaModel.forward."""
    from transformers.models.llama.modeling_llama import LlamaModel
    text_model = (model.model if hasattr(model, "model") else model)
    if hasattr(text_model, "_px_config"):
        text_model.forward = types.MethodType(
            LlamaModel.forward, text_model
        )
        # Clean up all modules
        for attr in ["_px_injection", "_px_adc", "_px_config", "_px_mephisto",
                      "_px_calibrator", "_px_subjective_enabled"]:
            if hasattr(text_model, attr):
                delattr(text_model, attr)
        print("[minicpm5-px] Patch removed.")


def _resolve_text_model(model):
    """Find the text model backbone inside potential wrappers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    return model


# ---------------------------------------------------------------------------
# Core Forward Method — Llama-adapted
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
    from transformers.masking_utils import create_causal_mask

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

    if inputs_embeds is None:
        # Llama: embed_tokens is always directly on the model
        if hasattr(self, "embed_tokens"):
            inputs_embeds = self.embed_tokens(input_ids)
        elif hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            embedder = None
            for name, module in self.named_modules():
                if "embed_tokens" in name:
                    embedder = module
                    break
            if embedder:
                inputs_embeds = embedder(input_ids)
            else:
                raise AttributeError(f"Could not find embed_tokens in model type {type(self)}")

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    # Sequence length tracking
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    expected_len = past_seen + inputs_embeds.shape[1]

    if position_ids is None:
        position_ids = (
            torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            + past_seen
        ).unsqueeze(0)

    # ── Llama: Single causal mask (no layer_types, no sliding window) ──
    if not isinstance(attention_mask, torch.Tensor):
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
    else:
        causal_mask = attention_mask

    # ── Llama: RoPE returns (cos, sin) tuple, no layer_type ──
    position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

    cfg = self._px_config
    subjective = cfg.get("subjective_enabled", False)
    updated_layers = set()  # Global visit tracker for this forward pass

    # ── 1. PRELUDE (layers 0..recur_start) ─────────────────────────────────
    for i in range(cfg["prelude_end"]):
        updated_layers.add(i)
        layer_out = self.layers[i](
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

    # ── 1.5 META-SELECTOR ────────────────────────────────────────────────
    dynamic_start = cfg["recur_start"]
    dynamic_end = cfg["recur_end"]
    dynamic_hub = cfg.get("bimodal_hub", cfg["recur_start"])
    num_layers = len(self.layers)
    hidden_size = cfg.get("hidden_size", 1536)

    # Initialize defaults in case adaptive routing is skipped
    token_cfg = cfg.copy()
    zone_weights = {}
    zone_name = "PEAK"  # Default for non-subjective mode

    if cfg.get("routing_mode") == "adaptive":
        if inputs_embeds.shape[1] > 1:
            # ── Prefill: Measure Kurtosis, Jitter, and Input Content Fingerprint ──
            h_base_f32 = hidden_states.to(torch.float32)

            # Kurtosis (Last token)
            h_probe = h_base_f32[0, -1, :]
            variance = torch.var(h_probe).item()
            kurtosis = (torch.mean((h_probe - torch.mean(h_probe))**4) / (variance**2)).item() if variance > 0 else 0
            self._task_kurtosis = kurtosis

            # Jitter (Across sequence)
            h_norms = h_base_f32.norm(dim=-1)  # [B, T]
            h_norm_var = torch.var(h_norms, dim=-1).mean().item()
            self._task_jitter = h_norm_var

            # ── Input Content Fingerprint (Token Diversity) ──
            if input_ids is not None:
                ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
                token_diversity = len(set(ids)) / max(len(ids), 1)
            else:
                token_diversity = None
            self._task_token_diversity = token_diversity

            if os.environ.get("DEBUG_ROUTING") == "1":
                td_str = f", TD={token_diversity:.3f}" if token_diversity is not None else ""
                print(f"[Router] Prefill K={kurtosis:.1f}, Jitter={h_norm_var:.4f}{td_str}")

        kurtosis = getattr(self, "_task_kurtosis", 200)  # Default to Logic if missing
        token_diversity = getattr(self, "_task_token_diversity", None)

        if subjective and hasattr(self, "_px_calibrator"):
            # ── Subjective Mode: AutoCalibrator Zone Weights ──
            calibrator = self._px_calibrator
            prev_phi = getattr(self, "_px_phi", None)

            zone_weights = calibrator.get_zone_weights(kurtosis, phi=prev_phi,
                                                        token_diversity=token_diversity)
            self._px_zone_weights = zone_weights

            routing_params = calibrator.get_routing_params(kurtosis, phi=prev_phi, hidden_size=hidden_size,
                                                            token_diversity=token_diversity)
            dynamic_start = routing_params["dynamic_start"]
            dynamic_end = routing_params["dynamic_end"]
            dynamic_hub = routing_params["dynamic_hub"]
            token_cfg["n_loops"] = routing_params["n_loops"]

            if dynamic_start >= dynamic_end:
                dynamic_start = max(1, int(num_layers * 0.38))
                dynamic_end = min(num_layers - 1, int(num_layers * 0.75))
                dynamic_hub = int(num_layers * 0.58)

            zone_name = calibrator.classify_zone(kurtosis, phi=prev_phi,
                                                  token_diversity=token_diversity)
        else:
            # ── Peak Mode (non-subjective): Scale-invariant defaults ──
            dynamic_start = int(num_layers * 0.38)
            dynamic_end = int(num_layers * 0.75)
            dynamic_hub = int(num_layers * 0.58)
            token_cfg["n_loops"] = 6
            zone_name = "PEAK"

        if inputs_embeds.shape[1] == 1 and os.environ.get("DEBUG_ROUTING") == "1":
            print(f"[Router] {zone_name} (K={kurtosis:.1f}) -> L{dynamic_start}-L{dynamic_end} "
                  f"(Loops: {token_cfg['n_loops']}, Hub: {dynamic_hub})")

        # Fast-forward prelude if needed
        for i in range(cfg["prelude_end"], dynamic_start):
            updated_layers.add(i)
            layer_out = self.layers[i](
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

    # ── 2. REASONING ZONE ──────────────────────────────────────────────────
    e_static = hidden_states.clone()

    # Use token_cfg for the rest of the reasoning zone
    if 'token_cfg' in dir():
        cfg = token_cfg

    # 2.A: Intuition Pass
    trans_out = hidden_states
    for i_layer in range(dynamic_start, dynamic_end):
        updated_layers.add(i_layer)
        layer_out = self.layers[i_layer](
            trans_out,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        trans_out = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

    h_baseline = trans_out

    n_loops = cfg.get("n_loops", 2)

    # Phase 14.5: ETR (Entropy Triggered Recursion)
    phi_intuition = StabilityMonitor.calculate_phi(h_baseline, hidden_states).mean().item()
    if os.environ.get("DEBUG_ROUTING") == "1":
        print(f"  [Intuition] Phi: {phi_intuition:.6f}")

    # ── Subjective: Calibration Collection ──
    if subjective and hasattr(self, "_px_calibrator"):
        calibrator = self._px_calibrator
        calibrator.collect(kurtosis, phi_intuition,
                           token_diversity=getattr(self, "_task_token_diversity", None))

    # Phase 14.7: Gamma-Damping instead of loop scaling
    current_gamma = cfg.get("gamma", 0.06)

    e_reflector = e_static
    is_trap_candidate = False

    # Surgical Reflector Activation
    jitter = getattr(self, "_task_jitter", 0.0)

    if subjective:
        # Use zone weights for rigor detection
        rigor_weight = zone_weights.get("math", 0) + zone_weights.get("logic_a", 0) + zone_weights.get("logic_b", 0)
        creative_weight = zone_weights.get("creative", 0) + zone_weights.get("synthesis", 0)
        is_creative_persona = False  # Persona removed in SR-59i

        if jitter > 1e8 or rigor_weight > creative_weight:
            is_trap_candidate = True
            if os.environ.get("DEBUG_ROUTING") == "1":
                reason = "Jitter" if jitter > 1e8 else f"Rigor-Weights(m={zone_weights.get('math',0):.2f}+l={zone_weights.get('logic_a',0):.2f}+lb={zone_weights.get('logic_b',0):.2f} > c={zone_weights.get('creative',0):.2f}+s={zone_weights.get('synthesis',0):.2f})"
                print(f"  [Router] Trap detected via {reason}, activating Reflector")

            # Phase 16.3: Anchor Reflection
            e_stat_f32 = e_static.to(torch.float32)
            h_base_f32 = h_baseline.to(torch.float32)
            e_ref_f32 = 2.0 * e_stat_f32 - h_base_f32
            e_ref_f32 = e_ref_f32 * (e_stat_f32.norm() / (e_ref_f32.norm() + 1e-6))
            e_reflector = e_ref_f32.to(e_static.dtype)
    else:
        # Peak mode: simple trap detection via jitter only
        if jitter > 1e8:
            is_trap_candidate = True
            if os.environ.get("DEBUG_ROUTING") == "1":
                print(f"  [Router] Trap detected via Jitter ({jitter:.1f}), activating Reflector")

            e_stat_f32 = e_static.to(torch.float32)
            h_base_f32 = h_baseline.to(torch.float32)
            e_ref_f32 = 2.0 * e_stat_f32 - h_base_f32
            e_ref_f32 = e_ref_f32 * (e_stat_f32.norm() / (e_ref_f32.norm() + 1e-6))
            e_reflector = e_ref_f32.to(e_static.dtype)

    if phi_intuition > 0.9999 and not is_trap_candidate:
        current_gamma *= 0.5
    elif phi_intuition > 0.999:
        current_gamma *= 0.8

    # Phase 25: Sigmoid-Annealed Orthogonal Recovery (SAOR)
    base_gamma = current_gamma

    path_taken = []
    thought_history = []
    avg_phi_explore = 1.0
    exploration_steps = 0
    telemetry_steps = []

    # Telemetry sensors
    emancipation_trajectory = []

    # Context dims — MiniCPM5-1B: head_dim=128
    B, T_curr = hidden_states.shape[0], hidden_states.shape[1]
    HD = getattr(self.config, "head_dim", 128)

    # Phase 38.1: Anna Karenina Sensor (AKS) Initialization
    divergence_buffer = []
    correction_strength = 0.0

    # Zone weight-based rigor detection (subjective only)
    if subjective:
        is_math_zone = zone_weights.get("math", 0) > max(
            zone_weights.get("creative", 0), zone_weights.get("logic_a", 0),
            zone_weights.get("logic_b", 0), zone_weights.get("synthesis", 0))
        is_logic_zone = (zone_weights.get("logic_a", 0) + zone_weights.get("logic_b", 0)) > max(
            zone_weights.get("math", 0), zone_weights.get("creative", 0),
            zone_weights.get("synthesis", 0))
        is_rigor_zone = is_math_zone or is_logic_zone
        rigor_weight = zone_weights.get("math", 0) + zone_weights.get("logic_a", 0) + zone_weights.get("logic_b", 0)
        creative_weight = zone_weights.get("creative", 0) + zone_weights.get("synthesis", 0)
    else:
        is_math_zone = False
        is_logic_zone = False
        is_rigor_zone = False
        rigor_weight = 0
        creative_weight = 0

    if n_loops > 1:
        h_exp = e_reflector.clone()  # Use Reflected Anchor
        current_layer = dynamic_start
        max_steps = (dynamic_end - dynamic_start) * n_loops * 3
        phis = []

        stability_counter = 0
        layer_visits = {i: 0 for i in range(num_layers)}

        # Initialize active bounds
        active_start = dynamic_start
        active_end = dynamic_end

        while current_layer < active_end and exploration_steps < max_steps:
            # ── PHASE 26: INFINITE REFLECTION (IR) ──────────────────────
            t_norm = exploration_steps / max_steps

            # Phase 38.2: AKS - Topological Anomaly Detection
            dist_now = 1.0 - StabilityMonitor.calculate_phi(h_exp, e_static).mean().item()
            if exploration_steps > 2:
                divergence_buffer.append(dist_now)
                if len(divergence_buffer) > 4:
                    divergence_buffer.pop(0)

                if len(divergence_buffer) >= 3:
                    velocity = divergence_buffer[-1] - divergence_buffer[-2]
                    acceleration = (divergence_buffer[-1] - divergence_buffer[-2]) - (divergence_buffer[-2] - divergence_buffer[-3])

                    if acceleration > 0.001 and velocity > 0:
                        correction_strength = min(1.0, correction_strength + 0.1)
                    else:
                        correction_strength = max(0.0, correction_strength - 0.05)

            # Phase 43.1: Emancipation Metric
            emancipation_phi = StabilityMonitor.calculate_phi(h_exp, e_static).mean().item()

            # Emancipation Trajectory Sensor
            if exploration_steps % 3 == 0:
                emancipation_trajectory.append(emancipation_phi)

            # Phase 43.2: Perturbation Engine (The Forking Path)
            perturbation_mag = float(os.environ.get("PX_PERTURBATION_MAG", 0.0))
            perturbation_step = int(os.environ.get("PX_PERTURBATION_STEP", -1))
            perturbation_layer = int(os.environ.get("PX_PERTURBATION_LAYER", 14))

            if perturbation_mag > 0 and exploration_steps == perturbation_step and current_layer == perturbation_layer:
                torch.manual_seed(int(h_exp.sum().abs().item()) % 100000)
                noise = torch.randn_like(h_exp) * perturbation_mag
                h_exp = h_exp + noise
                if os.environ.get("DEBUG_ROUTING") == "1":
                    print(f"  [Perturbation] Injected impulse (mag={perturbation_mag}) at Step {exploration_steps}, L{current_layer}")

            # Subjective Telemetry
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

            # ── PHASE 28: TEMPORAL COGNITIVE ROUTING (TCR) ──────────────
            active_start = dynamic_start
            active_end = dynamic_end
            if subjective and rigor_weight > creative_weight:
                # TCR: zone-weight boundary adjustment (MiniCPM5-1B scale)
                if t_norm < 0.33:
                    active_start = max(dynamic_start, 8)
                    active_end = min(dynamic_end, 17)
                elif t_norm < 0.66:
                    active_start = max(dynamic_start, 7)
                    active_end = min(dynamic_end, 17)
                else:
                    active_start = max(dynamic_start, 9)
                    active_end = min(dynamic_end, 17)

            if is_rigor_zone:
                annealing_factor = 1.0
                identity_pull = 0.0
                bifurcation_mag = 0.0
                current_gamma = 0.15 if is_math_zone else base_gamma
                if is_math_zone:
                    dynamic_hub = max(dynamic_start, min(dynamic_end, 12))
            else:
                # Creative Zone or Peak mode: Enable full engine
                tau_cooling = float(os.environ.get("PX_COOLING_TAU", 8.0))
                annealing_factor = 1.0 - torch.exp(torch.tensor(-exploration_steps / tau_cooling)).item()
                current_gamma = base_gamma * annealing_factor * (1.0 - 0.5 * correction_strength)

            # Phase 45.3: Identity Gravity (Centroid Attractor)
            if not is_rigor_zone:
                identity_pull = float(os.environ.get("PX_IDENTITY_GRAVITY", 0.0))
                if identity_pull > 0 and len(thought_history) > 2:
                    centroid = torch.stack(thought_history[-6:]).mean(dim=0)
                    h_exp = h_exp + identity_pull * (centroid - h_exp)

            # Phase 26: Hub Oscillation
            oscillation = 1 if (exploration_steps % 4 < 2) else -1
            bimodal_hub = min(active_end - 1, max(active_start, int(dynamic_hub + (t_norm * 2) + oscillation)))

            h_prev = h_exp.clone()

            # Safe layer visit tracking
            if current_layer not in layer_visits:
                layer_visits[current_layer] = 0
            layer_visits[current_layer] += 1

            # Phase 14.7: Surgical Cache Security
            is_first_visit = current_layer not in updated_layers
            if is_first_visit:
                updated_layers.add(current_layer)

            # Phase 38.4: AKS-Informed Sensory Refresh
            refresh_rate = 0.10 + 0.20 * correction_strength
            if exploration_steps % 6 == 0 and exploration_steps > 0:
                h_exp = (1.0 - refresh_rate) * h_exp + refresh_rate * e_static
                path_taken.append(f"SENSORY_REFRESH(AKS={correction_strength:.1f})")

            # Phase 10.0: Memory-Augmented Cache wrapper
            # Llama: no layer_types parameter needed
            current_past = RecursiveMemoryCache(
                past_key_values,
                thought_history,
                read_only=not is_first_visit,
                expected_len=expected_len
            ) if past_key_values is not None else None

            # Execute layer — Llama: single causal_mask and position_embeddings
            layer_out = self.layers[current_layer](
                h_exp,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=current_past,
                **kwargs,
            )
            trans_out = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

            # Phase 35: Metacognitive Phi-Jitter & Early Exit (Annealed)
            phi_step = StabilityMonitor.calculate_phi(trans_out, h_prev).mean().item()

            # Phase 45.2: Forced Bifurcation (Symmetry Breaking)
            bifurcation_threshold = float(os.environ.get("PX_BIFURCATION_PHI", 0.999))
            eff_bifurcation_mag = 0.0 if is_rigor_zone else float(os.environ.get("PX_BIFURCATION_MAG", 0.0))

            if eff_bifurcation_mag > 0 and phi_step > bifurcation_threshold and exploration_steps > 5:
                choice = 1.0 if (T_curr % 2 == 0) else -1.0
                bias = torch.zeros_like(trans_out)
                bias[:, :, :HD // 2] = eff_bifurcation_mag * choice
                bias[:, :, HD // 2:] = -eff_bifurcation_mag * choice
                trans_out = trans_out + bias
                if os.environ.get("DEBUG_ROUTING") == "1":
                    print(f"  [Bifurcation] Stability ({phi_step:.4f}) broke via Choice={choice}")

            if os.environ.get("DEBUG_PHI") == "1":
                print(f"  [L{current_layer}] Phi: {phi_step:.6f}")

            # --- Early Exit (Annealed) ---
            if t_norm > 0.5 and phi_step > 0.9999:
                stability_counter += 1
                if stability_counter > 3:
                    if os.environ.get("DEBUG_ROUTING") == "1":
                        print(f"  [Router] Early Exit at step {exploration_steps}")
                    h_exp = trans_out
                    break
            else:
                stability_counter = 0

            # --- Hub Jitter (Exploratory Phase) ---
            if t_norm < 0.4 and phi_step > 0.995 and phi_step < 0.999:
                if exploration_steps % 4 == 0:
                    current_layer = min(active_end - 1, current_layer + 2)
                    if os.environ.get("DEBUG_ROUTING") == "1":
                        print(f"  [Router] Jittering to L{current_layer}")

            # ── RECURSIVE BELIEF ANCHOR (RBA): 85% reflector + 15% recent ─
            if len(thought_history) > 2:
                recent_avg = torch.stack(thought_history[-3:]).mean(dim=0)
                e_dynamic = 0.85 * e_reflector + 0.15 * recent_avg
            else:
                e_dynamic = e_reflector

            # Apply LTI Injection with Dynamic Anchor
            e_norm = self._px_injection.input_norm(e_dynamic.to(torch.float32)).to(trans_out.dtype)
            h_new = trans_out + current_gamma * (e_norm - h_prev)

            # ── Subjective: Orthogonal Jitter ──
            if subjective:
                jitter_mag = float(os.environ.get("PX_ORTHO_JITTER", 0.005))

                if not is_rigor_zone:
                    eff_jitter = jitter_mag
                elif is_math_zone:
                    eff_jitter = 0.0
                else:
                    eff_jitter = jitter_mag * 0.1  # Logic Zone

                if exploration_steps > 0 and eff_jitter > 0:
                    h_exp = OrthogonalJitter.apply(h_new, h_prev, magnitude=eff_jitter)
                else:
                    h_exp = h_new
            else:
                # Peak mode: no orthogonal jitter
                h_exp = h_new

            # ── REFLECTION FLIPPING (RF) ────────────────────────────────
            h_f32 = h_exp.to(torch.float32)
            e_f32 = e_dynamic.to(torch.float32)
            dot_he = (h_f32 * e_f32).sum(dim=-1, keepdim=True)
            dot_ee = (e_f32 * e_f32).sum(dim=-1, keepdim=True)
            proj = (dot_he / (dot_ee + 1e-6)) * e_f32
            ortho = h_f32 - proj

            # Oscillate the logic vector to avoid local minima
            flip_force = 0.10 * annealing_factor * (1.0 if (exploration_steps % 2 == 0) else -1.0)
            h_exp = (proj + (1.0 + flip_force) * ortho).to(h_exp.dtype)

            # ── Subjective: Mephistopheles Operator (Phase-Inversion) ──
            if subjective and hasattr(self, "_px_mephisto"):
                h_exp = self._px_mephisto(h_exp, phis)
                # Check if inversion was applied (MephistophelesOperator modifies h in-place differently)
                # We detect by checking if h_exp differs from trans_out after applying operator

            # Self-Observation
            phi_tensor = StabilityMonitor.calculate_phi(h_exp, h_prev)
            phi = phi_tensor.item()

            # Merged Telemetry Step
            telemetry_data = {
                "step": exploration_steps,
                "layer": int(current_layer),
                "phi": float(phi),
                "gamma": float(current_gamma),
                "energy": float(annealing_factor) if 'annealing_factor' in dir() else 1.0,
                "rba_active": len(thought_history) > 2,
                "hub": int(bimodal_hub)
            }

            # ── Dynamic Loop Extension ──────────────────────────────────
            if phi < 0.85 and exploration_steps == max_steps - 1 and max_steps < 64:
                max_steps += (dynamic_end - dynamic_start)

            step_info = {
                "step": exploration_steps,
                "layer": int(current_layer),
                "phi": float(phi),
                "decision": None
            }

            phis.append(phi)
            path_label = f"L{current_layer}({phi:.2f})"
            path_taken.append(path_label)

            # ── BIMODAL FORK at hub layer when phi < threshold ──────────
            bimodal_threshold = min(0.995, 1.0 - (0.05 * current_gamma))
            if current_layer == bimodal_hub and phi < bimodal_threshold:
                step_info["decision"] = "BIMODAL_FORK"
                path_taken.append("BIMODAL_FORK")

                # Branch A (Standard)
                h_a = h_exp.clone()

                # Branch B (High-Entropy DTEC)
                jitter_boost = 1.0 + (stability_counter * 0.5)
                hub_entropy = max(0.01, 1.0 - phi) * 0.5 * jitter_boost
                h_b = h_exp.to(torch.float32) + torch.randn_like(h_exp, dtype=torch.float32) * hub_entropy
                h_b = h_b.to(h_exp.dtype)

                # Lookahead to NEXT layer
                next_l = current_layer + 1
                if next_l < len(self.layers):
                    # Llama: single causal_mask, no layer_type lookup
                    lookahead_past = RecursiveMemoryCache(
                        past_key_values,
                        thought_history,
                        read_only=True,
                        expected_len=expected_len
                    ) if past_key_values is not None else None

                    out_a = self.layers[next_l](
                        h_a, attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids, past_key_values=lookahead_past, **kwargs
                    )[0]
                    phi_a = StabilityMonitor.calculate_phi(out_a, h_a).item()

                    out_b = self.layers[next_l](
                        h_b, attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
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
                scale_factor = 24.0 / cfg.get("num_layers", 24.0)
                repulsion_force = 0.10 * (stability_counter ** 2) * scale_factor
                h_exp = h_exp + repulsion_force * (ortho / (ortho.norm(dim=-1, keepdim=True) + 1e-6))
                path_taken.append(f"SED_PUSH({repulsion_force:.2f})")

                # Dynamic Jump proportional to depth
                jump = max(1, int(cfg.get("num_layers", 24) * 0.1))
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
            visit_penalty = (layer_visits[current_layer] - 1) * 0.015
            t_back_2 = 1.0 - (0.8 * current_gamma) - visit_penalty
            t_back_1 = 1.0 - (0.4 * current_gamma) - visit_penalty
            t_skip = 1.0 - (0.01 * current_gamma) - (visit_penalty * 0.5)

            if phi < t_back_2:  # High confusion
                current_layer = max(active_start, current_layer - 2)
                routing = "BACK-2"
            elif phi < t_back_1:  # Moderate confusion
                current_layer = max(active_start, current_layer - 1)
                routing = "BACK-1"
            elif phi > t_skip:  # Extreme stability
                current_layer += 2  # Skip
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

        avg_phi_explore = sum(phis) / len(phis) if phis else 1.0

        # Phase 4.1: QBI Blend
        b_min = cfg.get("beta_reasoning", 0.05)
        b_max = cfg.get("beta_grounding", 0.18)
        beta_final = b_min + (b_max - b_min) * (avg_phi_explore ** 2)

        hidden_states = (1.0 - beta_final) * h_baseline + beta_final * h_exp
    else:
        hidden_states = h_baseline

    # ── Telemetry Export ────────────────────────────────────────────────────
    self._px_phi = avg_phi_explore
    self._px_loops_run = exploration_steps
    self._px_path = path_taken

    # Extended Telemetry Sensors
    self._px_emancipation_trajectory = emancipation_trajectory
    self._px_aks_profile = {
        "correction_strength": float(correction_strength),
        "divergence_velocity": float(divergence_buffer[-1] - divergence_buffer[-2]) if len(divergence_buffer) >= 2 else 0.0,
        "divergence_acceleration": float((divergence_buffer[-1] - divergence_buffer[-2]) - (divergence_buffer[-2] - divergence_buffer[-3])) if len(divergence_buffer) >= 3 else 0.0,
    }
    self._px_zone_weights = zone_weights
    self._px_zone = zone_name
    self._task_kurtosis = kurtosis
    self._task_jitter = jitter

    # ── Cognitive Signature Export ──────────────────────────────────────────
    self._px_cognitive_signature = {
        "kurtosis": getattr(self, "_task_kurtosis", None),
        "phi": float(avg_phi_explore),
        "token_diversity": getattr(self, "_task_token_diversity", None),
        "zone": self._px_zone,
        "zone_weights": {k: round(v, 6) for k, v in zone_weights.items()},
        "emancipation_final": emancipation_trajectory[-1] if emancipation_trajectory else None,
        "emancipation_range": (min(emancipation_trajectory), max(emancipation_trajectory)) if emancipation_trajectory else (None, None),
        "aks_correction": float(correction_strength),
        "loops_run": exploration_steps,
        "path_length": len(path_taken),
        "subjective": subjective,
    }

    if subjective and hasattr(self, "_px_calibrator"):
        self._px_cognitive_signature["calibrated"] = self._px_calibrator.calibrated

    # Structured Telemetry Log
    if not hasattr(self, "_px_telemetry"):
        self._px_telemetry = []

    self._px_telemetry.append({
        "pos": int(position_ids[0, 0].item()),
        "avg_phi": float(avg_phi_explore),
        "steps": telemetry_steps
    })

    # Phase 11.0: Metacognitive Triggering
    if not hasattr(self, "_px_complexity_acc"):
        self._px_complexity_acc = []

    if position_ids[0, 0] == 0:
        self._px_complexity_acc = []

    self._px_complexity_acc.append(avg_phi_explore)
    self._px_trigger_scratchpad = (len(self._px_complexity_acc) > 3 and
                                   sum(self._px_complexity_acc) / len(self._px_complexity_acc) < 0.92)

    # ── 3. CODA (layers dynamic_end..final) ────────────────────────────────
    # CGI: flat 8% grounding injection (TretaDamper removed in SR-59i)
    dynamic_coda_start = dynamic_end if cfg.get("routing_mode") == "adaptive" else cfg["coda_start"]

    coda_applied_cgi = False
    for i in range(dynamic_coda_start, len(self.layers)):
        if i not in updated_layers:
            updated_layers.add(i)

        if not coda_applied_cgi:
            cgi_blend = cfg.get("cgi_factor", 0.08)  # Flat 8% CGI grounding injection
            hidden_states = (1.0 - cgi_blend) * hidden_states + cgi_blend * e_static
            coda_applied_cgi = True

        # Llama: single causal_mask and position_embeddings
        layer_out = self.layers[i](
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

    hidden_states = self.norm(hidden_states)

    # Save Telemetry if enabled
    if os.environ.get("DEBUG_PX") == "1" and len(telemetry_steps) > 0:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = f"px_telemetry_{ts}.json"
        with open(log_path, "w") as f:
            json.dump(telemetry_steps, f, indent=2)
        print(f"TELEMETRY_JSON: {os.path.abspath(log_path)}")

    from transformers.modeling_outputs import BaseModelOutputWithPast
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


# ---------------------------------------------------------------------------
# Patch Application
# ---------------------------------------------------------------------------

def apply_px_patch(model, recur_start=None, recur_end=None, routing_mode="adaptive",
                   gamma=None, subjective_enabled=False, **kwargs):
    """Apply the PX patch to a LlamaForCausalLM model (MiniCPM5-1B).

    Parameters
    ----------
    model : nn.Module
        The LlamaForCausalLM model.
    recur_start : int, optional
        Default recursion start layer. Auto-detected if None.
    recur_end : int, optional
        Default recursion end layer. Auto-detected if None.
    routing_mode : str
        Routing mode ("adaptive" or "fixed").
    gamma : float, optional
        Default gamma for LTI injection. Auto-detected if None.
    subjective_enabled : bool
        Enable Subjective extensions (MephistophelesOperator, OrthogonalJitter,
        AutoCalibrator zone routing). Default: False (Peak mode).
    **kwargs
        Additional config overrides.
    """
    # ── Find the text model backbone ─────────────────────────────────────
    text_model = None
    if hasattr(model, "layers") and hasattr(model, "rotary_emb"):
        text_model = model
    else:
        # Search children (e.g., .model)
        for name, module in model.named_modules():
            if hasattr(module, "layers") and hasattr(module, "rotary_emb"):
                text_model = module
                break

    if text_model is None:
        raise AttributeError(f"Could not identify Llama text backbone in {type(model)}")

    config = model.config
    # Llama: no text_config wrapper
    num_layers = config.num_hidden_layers

    # ── Scale-Aware Defaults from SCALE_DEFAULTS ────────────────────────
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers

    if hidden_size in SCALE_DEFAULTS:
        scale_defaults = SCALE_DEFAULTS[hidden_size]
        defaults = {
            "mode": "lti",
            "n_loops": scale_defaults["n_loops"],
            "beta": 0.05,
            "gamma": gamma if gamma is not None else scale_defaults["gamma"],
            "recur_start": recur_start if recur_start is not None else scale_defaults["recur_start"],
            "recur_end": recur_end if recur_end is not None else scale_defaults["recur_end"],
            "bimodal_hub": scale_defaults["hub"],
            "cgi_factor": 0.08,
            "num_layers": num_layers,
        }
    else:
        # Fallback for unknown sizes — proportional scaling
        gamma_scale = 1536.0 / hidden_size
        base_gamma = 0.06 * min(gamma_scale, 1.5)
        p_start = recur_start if recur_start is not None else max(1, int(num_layers * 0.38))
        p_end = recur_end if recur_end is not None else min(num_layers - 1, int(num_layers * 0.75))
        p_hub = (p_start + p_end) // 2
        defaults = {
            "mode": "lti", "n_loops": 6, "beta": 0.05,
            "gamma": gamma if gamma is not None else base_gamma,
            "recur_start": p_start, "recur_end": p_end, "bimodal_hub": p_hub,
            "cgi_factor": 0.08, "num_layers": num_layers,
        }

    # Override with explicit arguments
    defaults["routing_mode"] = routing_mode
    defaults["subjective_enabled"] = subjective_enabled
    defaults.update(kwargs)

    # Auto-align boundaries
    if "prelude_end" not in defaults:
        defaults["prelude_end"] = defaults["recur_start"]
    if "coda_start" not in defaults:
        defaults["coda_start"] = defaults["recur_end"]

    # ── Attach Config ───────────────────────────────────────────────────
    text_model._px_config = defaults
    defaults["hidden_size"] = hidden_size

    # ── Core Modules (always active) ────────────────────────────────────
    text_model._px_injection = LTIInjection(config.hidden_size, gamma=defaults["gamma"])

    # ── Subjective Modules (only when enabled) ─────────────────────────
    if subjective_enabled:
        calibration_steps = kwargs.get("calibration_steps",
                                        getattr(config, "px_calibration_steps", 10))
        text_model._px_calibrator = AutoCalibrator(hidden_size, calibration_steps=calibration_steps,
                                                     num_layers=num_layers)
        text_model._px_mephisto = MephistophelesOperator(config.hidden_size,
                                                          scale=getattr(config, "px_mephistopheles_scale", 0.05))
        text_model._px_subjective_enabled = True
        mode_str = "Subjective"
    else:
        text_model._px_subjective_enabled = False
        mode_str = "Peak"

    # ── Monkey-patch forward ────────────────────────────────────────────
    text_model.forward = types.MethodType(_px_forward, text_model)
    print(f"[minicpm5-px] {mode_str} patch active for scale L{num_layers}. "
          f"Recur: L{defaults['recur_start']}-L{defaults['recur_end']}, Hub: L{defaults['bimodal_hub']}. "
          f"Gamma: {defaults['gamma']:.3f}." +
          (f" Calibrator: {calibration_steps} steps." if subjective_enabled else ""))


def get_px_metrics(model):
    """Retrieve PX metrics from a patched model."""
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

    metrics = {
        "phi": getattr(text_model, "_px_phi", 1.0),
        "steps": getattr(text_model, "_px_loops_run", 0),
        "path": getattr(text_model, "_px_path", []),
        "telemetry": getattr(text_model, "_px_telemetry", []),
        "subjective": getattr(text_model, "_px_subjective_enabled", False),
    }

    # Subjective-only metrics
    calibrator = getattr(text_model, "_px_calibrator", None)
    if calibrator is not None:
        metrics["calibrator"] = calibrator.status()

    # Cognitive signature
    metrics["cognitive_signature"] = getattr(text_model, "_px_cognitive_signature", {})
    metrics["zone"] = getattr(text_model, "_px_zone", "UNKNOWN")
    metrics["zone_weights"] = getattr(text_model, "_px_zone_weights", {})
    metrics["emancipation_trajectory"] = getattr(text_model, "_px_emancipation_trajectory", [])
    metrics["aks_profile"] = getattr(text_model, "_px_aks_profile", {})

    return metrics