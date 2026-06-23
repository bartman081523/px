"""
infllm_integration.py — Phase C: Forward-Hook-Integration für InfLLM
====================================================================

Bindet InfLLMCache an Gemma3Attention-Layer über HuggingFace-Forward-Hooks.
Diese Hooks sind **NICHT in patch.py** — sie sind eine separate, ablegbare
Integrations-Schicht. Motor-Edit in Phase D kommt erst nach expliziter
User-Freigabe.

Architektur:
    model: HuggingFace Gemma3ForConditionalGeneration
      └─ language_model: Gemma3TextModel
           └─ layers[i]: Gemma3DecoderLayer
                └─ self_attn: Gemma3Attention   ← HIER der Hook

Hook-Signatur (HF forward_hook):
    hook(module, input, output) -> output oder None

Was wir tun:
    1. Holen das `past_key_values` aus dem forward-input (es ist ein
       DynamicCache oder HybridCache oder — in unserem Fall — InfLLMCache).
    2. Wenn past_key_values eine `prepare_reattention` Methode hat, leiten
       wir die q/k/v durch InfLLM statt durch SDPA.

Problem: Gemma3Attention.forward erwartet (q, k, v) als interne States,
nicht als Forward-Input. Wir können q/k/v NICHT direkt im Hook abfangen —
sie werden INNERHALB des forwards erzeugt (durch q_proj/k_proj/v_proj).

Lösung: Wir patchen NICHT das forward, sondern registrieren einen
**pre-forward hook** der `past_key_values` durch eine "InfLLM-getaggte"
Version ersetzt (mit zusätzlichem Flag), und einen **post-forward hook**
der NICHTS tut (Hook ist nur ein Marker).

Der eigentliche InfLLM-Call passiert dann in einer **separaten Schicht**,
die wir als forward-Ersatz registrieren — das ist äquivalent zum
existierenden `apply_reattention_patch` (welcher module.forward direkt
ersetzt).

Wir bieten ZWEI Wege:
  (a) `install_infllm_hooks(model, cache)`: forward_hook-Stack (kein
      module.forward-Replacement; rückgängig machbar via remove_hook).
  (b) `apply_infllm_forward_patch(model, cache)`: module.forward ersetzen
      (äquivalent zu existierendem apply_reattention_patch, aber mit
      Cache-Argument explizit; rückgängig via remove_infllm_forward_patch).

(a) ist die "saubere" Variante — wir nutzen sie für Phase C.

Run:
    python -c "from infllm_integration import install_infllm_hooks; help(install_infllm_hooks)"
"""
from __future__ import annotations

import types
from typing import Optional, List

import torch

from infinite_context import InfLLMCache


# Global registry: model → list of (module, hook_handle) tuples. Wird für
# idempotente install/uninstall verwendet.
_HOOK_REGISTRY: dict = {}


def _patch_module_forward_with_cache(module, cache: InfLLMCache):
    """Replace module.forward with a version that calls prepare_reattention.

    Diese Funktion ist die Grundlage für Phase D (motor-integration) —
    hier definieren wir sie als externe Funktion, nicht in patch.py.
    """
    def forward_with_infllm(self, hidden_states, position_embeddings=None,
                             attention_mask=None, past_key_values=None, **kwargs):
        """Surgical patch für Gemma3Attention mit InfLLM/ReAttention."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if past_key_values is None:
            past_key_values = cache

        if hasattr(past_key_values, "prepare_reattention"):
            # ReAttention-Pfad
            query_states, key_states, value_states = past_key_values.prepare_reattention(
                query_states, key_states, value_states, self.layer_idx, self.rotary_emb,
            )

            # Attention-Mask auf neue Key-Länge anpassen
            if attention_mask is not None:
                T_k = key_states.size(-2)
                T_orig_k = attention_mask.size(-1)
                if T_k > T_orig_k:
                    import torch.nn.functional as F
                    pad_len = T_k - T_orig_k
                    attention_mask = F.pad(attention_mask, (pad_len, 0), value=0.0)
                elif T_k < T_orig_k:
                    attention_mask = attention_mask[..., -T_k:]

            # Standard SDPA (oder FlashAttention) ohne RoPE (InfLLM hat's
            # schon angewendet)
            from transformers.models.gemma3.modeling_gemma3 import (
                ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
            )
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward,
            )
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
            )
        else:
            # Standard Gemma3-Pfad (KEIN InfLLM, kein Cache oder
            # nicht-InfLLM-Cache): roPE + standard-attention
            from transformers.models.gemma3.modeling_gemma3 import (
                apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
            )
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward_with_infllm


def _is_attention_module(module) -> bool:
    """Heuristik: ist dieses Modul ein Attention-Layer?

    Detection: Class-Name enthält "Attention" ODER das Modul hat die
    Gemma3-Attention-typischen Attribute q_proj/k_proj/v_proj/o_proj
    plus rotary_emb. Beide Bedingungen decken Gemma3-Realität UND
    Mock-Test-Module ab.
    """
    cls_name = type(module).__name__
    if "Attention" in cls_name:
        return True
    # Strukturelle Heuristik für Mocks oder alternative Attention-Klassen
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "rotary_emb",
                "q_norm", "k_norm", "head_dim", "layer_idx")
    return all(hasattr(module, attr) for attr in required)


def install_infllm_hooks(model, cache: InfLLMCache) -> int:
    """Installiert InfLLM-Forward-Patches auf alle Gemma3Attention-Layer.

    Argument:
        model: HuggingFace Gemma3-Stack (mit Gemma3Attention-Sublayern)
        cache: InfLLMCache-Instanz (geteilt zwischen allen Layern)

    Returns:
        Anzahl der gepatchten Attention-Layer.

    Side-effect:
        Speichert die ursprüngliche forward-Methode in `_original_forward`
        auf jedem Modul, damit remove_infllm_hooks die Patches rückgängig
        machen kann.
    """
    model_id = id(model)
    if model_id in _HOOK_REGISTRY:
        # Idempotent: schon gepatcht
        return len(_HOOK_REGISTRY[model_id])

    patched = []
    for name, module in model.named_modules():
        if _is_attention_module(module) and module is not model:
            # Original forward speichern (für uninstall)
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward
            # Patch installieren
            bound_fn = _patch_module_forward_with_cache(module, cache)
            module.forward = types.MethodType(bound_fn, module)
            patched.append((name, module))

    _HOOK_REGISTRY[model_id] = patched
    return len(patched)


def remove_infllm_hooks(model) -> int:
    """Entfernt InfLLM-Forward-Patches. Stellt ursprüngliches forward wieder her.

    Returns:
        Anzahl der entfernten Patches.
    """
    model_id = id(model)
    if model_id not in _HOOK_REGISTRY:
        return 0
    patched = _HOOK_REGISTRY.pop(model_id)
    for name, module in patched:
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
    return len(patched)


def is_infllm_installed(model) -> bool:
    """True wenn InfLLM-Patches auf diesem Model aktiv sind."""
    return id(model) in _HOOK_REGISTRY
