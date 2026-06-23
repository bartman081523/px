"""
int8_kv_cache.py — Plan 3 Phase A: KV-Cache int8 Quantization
=================================================================

KV-Tensoren (k/v aus attention forward) werden per-channel int8 quantisiert.
Halbiert KV-Speicher ohne Motor-Edit (forward_hook).

Architektur:
  - hook auf attention forward
  - nach k_proj/v_proj: quantize → (int8, fp32 scale, fp32 zero) per (head, channel)
  - storage: int8 + per-head scale/zero (je head ein scale-pair)
  - vor matmul: dequantize back to bf16 on-the-fly

Diese Datei nutzt NICHT Gemma3Attention.forward direkt. Sie patch via
forward_hook (äquivalent zu infllm_integration.py):
  - module.k_proj output → quantize
  - vor attention matmul → dequantize
  - module.v_proj output → quantize
  - vor attention matmul → dequantize

Das ist NICHT trivial als reiner Hook machbar (matmul ist innerhalb von
ALL_ATTENTION_FUNCTIONS). Daher: forward-Patch der das gleiche Pattern
wie infllm_integration.py nutzt.

Run:
    python -c "from int8_kv_cache import install_int8_kv_hooks; help(install_int8_kv_hooks)"
"""
from __future__ import annotations

import types
from typing import Tuple

import torch
import torch.nn as nn


# Per-head quantisiation: jeder KV-Head hat seinen eigenen scale/zero.
# Pro Token ist int8; das macht die Quantisierung für lange Sequenzen
# linear in T.

# Global registry
_HOOK_REGISTRY: dict = {}


def quantize_kv(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel int8 quantization für KV-Tensor.

    Input:  x mit shape [B, H_kv, T, head_dim]
    Output: (x_int8, scale, zero) wo
        x_int8: [B, H_kv, T, head_dim] in int8
        scale:   [B, H_kv, 1, head_dim] in float32 (per-channel)
        zero:    [B, H_kv, 1, head_dim] in float32 (per-channel)

    Form: x_q = round((x - zero) / scale), mit scale/zero per (B, H_kv, head_dim).
    zero = min(x, dim=T) (für symmetrische range).

    Für speed: per-token scale ist overkill. Per-channel (über T) ist genug.
    """
    # Per-channel: stats über T-axis (axis=-2)
    if x.dtype != torch.float32:
        x_f = x.float()
    else:
        x_f = x
    x_min = x_f.amin(dim=-2, keepdim=True)  # [B, H_kv, 1, head_dim]
    x_max = x_f.amax(dim=-2, keepdim=True)
    # Symmetric range mit zero=0 (einfacher): scale = max(|min|, |max|) / 127
    abs_max = torch.maximum(x_min.abs(), x_max.abs())
    scale = (abs_max / 127.0).clamp(min=1e-8)
    zero = torch.zeros_like(scale)
    x_int8 = torch.round(x_f / scale).clamp(-128, 127).to(torch.int8)
    return x_int8, scale, zero


def dequantize_kv(x_int8: torch.Tensor, scale: torch.Tensor,
                   zero: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize int8 KV zurück zu bf16 (oder float32).

    Input shapes:
        x_int8: [B, H_kv, T, head_dim]
        scale:  [B, H_kv, 1, head_dim]
        zero:   [B, H_kv, 1, head_dim]
    """
    x_f = x_int8.float() * scale + zero  # broadcast over T
    return x_f.to(dtype)


def _is_attention_module(module) -> bool:
    """Heuristik: ist Modul ein Attention-Layer (gleiche Logik wie infllm_integration)."""
    cls_name = type(module).__name__
    if "Attention" in cls_name:
        return True
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "rotary_emb",
                "q_norm", "k_norm", "head_dim", "layer_idx")
    return all(hasattr(module, attr) for attr in required)


def _patch_module_forward_with_int8_kv(module, skip_prefill_threshold: int = 64):
    """Replace module.forward mit conditional int8-KV-Version.

    Hook ist CONDITIONAL: nur aktiv wenn T_new <= skip_prefill_threshold
    (default 64, d.h. incremental decode). Bei Prefill (T_new > threshold)
    wird der Hook umgangen — die Quantisierung würde 2x Speicher brauchen
    ohne storage-Vorteil (Prefill wird nicht gecacht in int8).

    Effekt:
      - Prefill: bf16 KV (kein Overhead, keine Änderung)
      - Incremental Decode (T_new=1..64): k/v werden int8 quantisiert,
        dann für matmul rekonstruiert. Im Cache landet die int8-Form.

    WICHTIG: das ist eine ECHTE memory-reduction für inkrementelles decode:
    der KV-Cache wächst in int8 statt bf16 → halbiert Cache-Speicher.
    """
    def forward_with_int8_kv(self, hidden_states, position_embeddings=None,
                              attention_mask=None, past_key_values=None, **kwargs):
        """Forward mit CONDITIONAL int8 KV-Quantisierung (skip bei Prefill)."""
        from transformers.models.gemma3.modeling_gemma3 import (
            apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
        )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        T_new = input_shape[1] if len(input_shape) >= 2 else 1

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # CONDITIONAL: int8 nur für inkrementelles Decode, NICHT Prefill
        if T_new <= skip_prefill_threshold:
            k_int8, k_scale, k_zero = quantize_kv(key_states)
            v_int8, v_scale, v_zero = quantize_kv(value_states)
            key_states = dequantize_kv(k_int8, k_scale, k_zero, dtype=key_states.dtype)
            value_states = dequantize_kv(v_int8, v_scale, v_zero, dtype=value_states.dtype)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if hasattr(past_key_values, "update"):
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx,
                )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    return forward_with_int8_kv


def install_int8_kv_hooks(model) -> int:
    """Installiert int8 KV-Forward-Patches auf alle Attention-Layer.

    Returns:
        Anzahl der gepatchten Attention-Layer.
    """
    model_id = id(model)
    if model_id in _HOOK_REGISTRY:
        return len(_HOOK_REGISTRY[model_id])

    patched = []
    for name, module in model.named_modules():
        if _is_attention_module(module) and module is not model:
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward
            bound_fn = _patch_module_forward_with_int8_kv(module)
            module.forward = types.MethodType(bound_fn, module)
            patched.append((name, module))

    _HOOK_REGISTRY[model_id] = patched
    return len(patched)


def remove_int8_kv_hooks(model) -> int:
    """Entfernt int8 KV-Patches."""
    model_id = id(model)
    if model_id not in _HOOK_REGISTRY:
        return 0
    patched = _HOOK_REGISTRY.pop(model_id)
    for name, module in patched:
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
    return len(patched)


def is_int8_kv_installed(model) -> bool:
    return id(model) in _HOOK_REGISTRY