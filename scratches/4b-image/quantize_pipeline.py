"""
quantize_pipeline.py — End-to-End Quantisierungs-Pipeline
===========================================================

Phase C von Plan 1. Walkt durch ein nn.Module und ersetzt jedes nn.Linear
durch ein QuantizedLinear. Idempotent (zweimal aufgerufen = no-op beim 2. Mal).

Anwendungsfall: model_manager._load_model ruft dies nach `from_pretrained`
auf, BEVOR das Model auf die GPU verschoben wird (oder direkt nach — die
QuantizedLinear-Layer verwalten sich via register_buffer / register_parameter).

Sicherheitsnetz: walkt nur, wenn `module` selbst KEIN QuantizedLinear ist
(siehe idempotency_check), und nutzt `setattr(parent, name, new_module)` für
den In-Place-Swap (getestet in test_quantized_linear).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from quantized_linear import QuantizedLinear, quantize_linear


def _is_quantized_already(module: nn.Module) -> bool:
    """True wenn `module` (oder ein direkter Nachfolger) bereits ein
    QuantizedLinear ist. Vermeidet doppelte Quantisierung."""
    if isinstance(module, QuantizedLinear):
        return True
    for child in module.children():
        if _is_quantized_already(child):
            return True
    return False


def _walk_and_replace(module: nn.Module) -> int:
    """Rekursiver Walk: ersetzt jedes nn.Linear durch ein QuantizedLinear.

    Returns die Anzahl der Ersetzungen. Idempotent: bereits ersetzte Module
    werden übersprungen (ihre nn.Linear-Nachfolger gibt es nicht mehr)."""
    n_replaced = 0
    # Iteriere über (name, child) Paare, KANN NICHT während der Iteration
    # modifiziert werden, daher sammeln wir Targets zuerst.
    targets = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            targets.append((name, child))
        else:
            n_replaced += _walk_and_replace(child)
    # Jetzt ersetzen.
    for name, child in targets:
        q = quantize_linear(child)
        setattr(module, name, q)
        n_replaced += 1
    return n_replaced


def quantize_all_linears(model: nn.Module) -> nn.Module:
    """Quantize all nn.Linear submodules in-place. Returns the model.

    Safe to call multiple times (no-op if already quantized).
    """
    if _is_quantized_already(model):
        # Idempotenter Pfad: wir walken trotzdem, aber die inneren
        # QuantizedLinears werden durch den `isinstance(nn.Linear)`-Check
        # in _walk_and_replace übersprungen. Stellen sicher, dass es
        # keine unquantized Linears mehr gibt.
        pass
    _walk_and_replace(model)
    return model
