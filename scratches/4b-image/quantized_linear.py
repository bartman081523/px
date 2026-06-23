"""
quantized_linear.py — QuantizedLinear Drop-in-Ersatz für nn.Linear
===================================================================

Phase B von Plan 1 (siehe PLAN_QUANTIZATION.md). Ersetzt ein nn.Linear durch
eine int8 quantisierte Variante. Speichert:
  - self.weight: int8 [out, in]
  - self.weight_scale: float32 [out]
  - self.bias: optional, gleicher dtype wie Original (meist bf16)

Forward macht:
  dequant = self.weight.float() * self.weight_scale.unsqueeze(1)  # [out, in] fp32
  output = F.linear(x, dequant.to(x.dtype), self.bias)

Das ist absichtlich NICHT optimiert (kein cuBLAS int8 GEMM, kein fused
Kernel). Für unsere 4b-Anwendung ist der Dequant-Aufwand klein im
Verhältnis zur 50% VRAM-Ersparnis. Phase 2 (InfLLM) wird das eh nochmal
angreifen.

Sicherheitsnetz: das ist eine LERN-IMPLEMENTIERUNG, kein Produktions-Code.
Jeder Quantisierungs-Fehler wird via test_quantize.py + test_quantized_linear.py
gepinnt.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize import quantize_state_dict, dequantize_state_dict, SCHEME_INT8


class QuantizedLinear(nn.Module):
    """Drop-in-Ersatz für nn.Linear. Speichert Gewicht in int8, dequantisiert
    on-the-fly im forward(). Bias (falls vorhanden) bleibt im Original-Dtype.

    Hinweis: Dies ist KEIN vollständig bit-genauer Ersatz. Der forward()-Output
    unterscheidet sich vom Original-Linear um int8-Round-Trip-Fehler
    (~0.79% pro Element, summiert zu cos_sim > 0.99 in der Praxis).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Int8 storage for weight (allocated in quantize_linear, not here).
        self.register_buffer("weight", torch.zeros(out_features, in_features,
                                                     dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(out_features,
                                                         dtype=torch.float32))
        if bias:
            self.register_parameter("bias", nn.Parameter(
                torch.zeros(out_features, dtype=dtype)))
        else:
            self.register_parameter("bias", None)
        self._forward_dtype = dtype  # dtype to cast dequant to

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: int8 → float32 (per-channel symmetric).
        w_fp32 = self.weight.to(torch.float32) * self.weight_scale.unsqueeze(1)
        # Cast to input dtype to keep numerical behavior consistent with the
        # original nn.Linear (which had the same dtype as x).
        w = w_fp32.to(x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, dtype={self._forward_dtype}, "
                f"storage=int8+scale")


def quantize_linear(layer: nn.Linear) -> QuantizedLinear:
    """Convert an existing nn.Linear to a QuantizedLinear.

    Steps:
      1. Allocate QuantizedLinear with same in/out/bias.
      2. Quantize the weight (int8 + per-channel scale).
      3. Copy bias if present.
    """
    has_bias = layer.bias is not None
    bias_dtype = layer.bias.dtype if has_bias else torch.bfloat16

    q = QuantizedLinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=has_bias,
        dtype=bias_dtype,
    )
    # Quantize weight. quantize_state_dict is 2D-aware; biases/1D are passed
    # through. We only quantize the weight here.
    qd = quantize_state_dict({"weight": layer.weight.data}, scheme=SCHEME_INT8)
    # Inherit device from the original layer so quantized buffers land on the
    # same device as the rest of the model (cuda after from_pretrained with
    # device_map="auto"). Without this, the int8 weight + fp32 scale stay on
    # CPU and forward() crashes with "mat2 is on cpu, different from other
    # tensors on cuda:0".
    target_device = layer.weight.device
    if target_device.type != "cpu":
        q.weight = q.weight.to(target_device)
        q.weight_scale = q.weight_scale.to(target_device)
        if has_bias:
            q.bias = nn.Parameter(q.bias.data.to(target_device))
    with torch.no_grad():
        q.weight.copy_(qd["weight"])
        q.weight_scale.copy_(qd["weight_scale"])
        if has_bias:
            q.bias.copy_(layer.bias.data.to(bias_dtype))
    return q


def storage_bytes(q_layer: QuantizedLinear) -> int:
    """Storage-bytes für int8 weight + fp32 scale + bias (falls vorhanden)."""
    total = (q_layer.weight.numel() * q_layer.weight.element_size() +
             q_layer.weight_scale.numel() * q_layer.weight_scale.element_size())
    if q_layer.bias is not None:
        total += q_layer.bias.numel() * q_layer.bias.element_size()
    return total
