"""
quantize.py — int8 per-channel-symmetric weight quantization
=============================================================

Eigenständiger Helper ohne externe Deps (kein BitsAndBytes, kein accelerate).
Aktiviert als Phase A von Plan 1 (siehe PLAN_QUANTIZATION.md). Reduziert
Model-Gewichte (bf16/fp16/fp32) auf int8 + per-Channel scales.

Mechanik:
  Pro Output-Channel i (weight[i, :]):
    scale[i] = max(|weight[i, :]|) / 127      # symmetric, [-127, 127]
    q_weight[i, :] = round(weight[i, :] / scale[i])
    weight ≈ dequant: q_weight * scale[i]

VRAM-Bilanz für ein Linear [out, in] (fp32, 4 bytes/param):
  raw:      out * in * 4 bytes
  int8:     out * in * 1 + out * 4 bytes (scales)
  Beispiel 64x128: 32768 → 8448 bytes (3.88x compression)
  Für bf16 (2 bytes/param): 16384 → 8448 bytes (1.94x compression ≈ "2x").

Dequantisierung ist deterministisch und ohne Look-up-Tabellen. Round-Trip-
Fehler pro Output-Channel: < 1/127 ≈ 0.79% relativ (worst-case).

Wichtig: KV-Cache wird NICHT quantisiert (nur Linear-Gewichte). KV-Optimierung
ist separater Plan (InfLLM/ReAttention).
"""
from __future__ import annotations

from typing import Dict
import torch


# Scheme constants
SCHEME_NONE = "none"
SCHEME_INT8 = "int8"

# Supported schemes — used by model_manager / tests to validate config.
SUPPORTED_SCHEMES = (SCHEME_NONE, SCHEME_INT8)


def quantize_state_dict(state_dict: Dict[str, torch.Tensor],
                         scheme: str = SCHEME_INT8,
                         ) -> Dict[str, torch.Tensor]:
    """Quantize a state_dict in-place-style: returns a new dict of quantized
    tensors. Only 2D weight tensors (typical for nn.Linear) are quantized;
    everything else (biases, layernorm, etc.) is passed through unchanged.

    Args:
        state_dict: {name: tensor}. 2D tensors get quantized, others copied.
        scheme: "none" → no-op (returns same tensor refs), "int8" → symmetric.

    Returns:
        For scheme="none": same dict, tensor identities preserved.
        For scheme="int8": dict with quantized 2D tensors (keys preserve
            name; original + "_scale" appended for scales).
    """
    if scheme == SCHEME_NONE:
        return dict(state_dict)

    if scheme != SCHEME_INT8:
        raise ValueError(f"unsupported scheme {scheme!r}; "
                         f"supported: {SUPPORTED_SCHEMES}")

    out: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if tensor.dim() == 2 and tensor.is_floating_point():
            # Per-channel: outer dim is the output-feature axis.
            max_abs = tensor.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            scale = (max_abs / 127.0).squeeze(1).contiguous()  # [out]
            q_weight = torch.round(tensor / scale.unsqueeze(1)).to(torch.int8).contiguous()
            out[name] = q_weight
            out[name + "_scale"] = scale.to(torch.float32)
        else:
            # Pass-through: biases (1D), layernorm (1D), embeddings (2D but
            # not "weight" of nn.Linear). For embeddings we COULD quantize
            # but that's a separate decision (lookup tables behave differently).
            out[name] = tensor.clone() if tensor.is_floating_point() else tensor
    return out


def dequantize_state_dict(quantized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reverse of quantize_state_dict: int8 + scale → original-dtype tensor.

    For quantized tensors (have a paired "_scale" entry), reconstructs the
    original-dtype weight by multiplying int8 * scale[:, None] and casting
    back to the original dtype (inferred from the scale tensor, which is fp32).

    For pass-through tensors (no matching "_scale"), returns a clone.
    """
    out: Dict[str, torch.Tensor] = {}
    scale_keys = {k for k in quantized if k.endswith("_scale")}

    for name, tensor in quantized.items():
        if name.endswith("_scale"):
            continue  # scales handled by their weight

        scale_key = name + "_scale"
        if scale_key in quantized:
            q = tensor                          # [out, in] int8
            scale = quantized[scale_key]        # [out] fp32
            # Original dtype — we don't know it from quantized dict, so default
            # to bf16 (matches our model_manager default). Caller can cast.
            original_dtype = torch.bfloat16
            dequant = (q.to(torch.float32) * scale.unsqueeze(1)).to(original_dtype)
            out[name] = dequant.contiguous()
        else:
            out[name] = tensor.clone() if tensor.is_floating_point() else tensor

    return out


def storage_bytes(quantized: Dict[str, torch.Tensor]) -> int:
    """Sum of storage bytes for all tensors (including scales). Used by tests
    to verify compression ratio. Excludes Python-object overhead."""
    total = 0
    for t in quantized.values():
        total += t.numel() * t.element_size()
    return total
