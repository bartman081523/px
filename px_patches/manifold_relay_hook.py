"""px_patches/manifold_relay_hook.py — Strict Hook Lifecycle Context Manager.

Prevents forward hook stacking/leaks across inference calls and guarantees
deterministic hook cleanup (__exit__) with active hook count logging.
"""

from typing import Optional
import torch


class ManifoldRelayHook:
    """Strict Context Manager for Layer Forward Hooks.

    Guarantees:
    - Pre-inference cleanup of any orphaned or dangling hooks on target layer.
    - Deterministic removal via handle.remove() in __exit__ even if exceptions occur.
    - Tracking of active_hooks_count (1 for active relay, 0 for baseline).
    """

    def __init__(
        self,
        model,
        layer_idx: int = 26,
        vector: Optional[object] = None,
        alpha: float = 0.35,
        sign: int = 0,
        model_id: str = "gemma4-e2b-it",
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.vector = vector
        self.alpha = float(alpha)
        self.sign = int(sign)
        self.model_id = model_id
        self.text_model = None
        self.active_hooks_count = 0
        self.target_layer = None

    def _resolve_text_model(self, model):
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        if hasattr(model, "layers") and hasattr(model, "rotary_emb"):
            return model
        candidates = []
        for _, mod in model.named_modules():
            if hasattr(mod, "layers") and hasattr(mod, "rotary_emb"):
                candidates.append(mod)
        if candidates:
            return max(candidates, key=lambda m: len(m.layers))
        return getattr(model, "model", model)

    def __enter__(self):
        self.text_model = self._resolve_text_model(self.model)
        if not hasattr(self.text_model, "layers") or self.layer_idx >= len(self.text_model.layers):
            raise IndexError(f"Layer index {self.layer_idx} out of bounds for text model layers ({len(getattr(self.text_model, 'layers', []))})")

        self.target_layer = self.text_model.layers[self.layer_idx]

        # Ensure _px_hf_id is available on text_model
        if not getattr(self.text_model, "_px_hf_id", None):
            if "gemma4" in self.model_id.lower() or "e2b" in self.model_id.lower():
                self.text_model._px_hf_id = "google/gemma-4-E2B-it"
            elif "270m" in self.model_id.lower():
                self.text_model._px_hf_id = "google/gemma-3-270m-it"
            elif "1b" in self.model_id.lower():
                self.text_model._px_hf_id = "google/gemma-3-1b-it"
            elif "4b" in self.model_id.lower():
                self.text_model._px_hf_id = "google/gemma-3-4b-it"
            elif "bonsai" in self.model_id.lower() or "27b" in self.model_id.lower():
                self.text_model._px_hf_id = "ternary-bonsai-2-27b-hf"

        # 1. Clean existing relay handles on text_model
        handles = getattr(self.text_model, "_px_relay_handles", None)
        if handles:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self.text_model._px_relay_handles = []

        # 2. Clear any lingering forward hooks directly on target layer
        if hasattr(self.target_layer, "_forward_hooks"):
            self.target_layer._forward_hooks.clear()
        if hasattr(self.target_layer, "_forward_pre_hooks"):
            self.target_layer._forward_pre_hooks.clear()

        # 3. If sign != 0, install the relay hook
        if self.sign != 0:
            if "gemma4" in self.model_id.lower() or "e2b" in self.model_id.lower():
                from px_patches.gemma4_2b_px.relay_inject import install_relay
            elif "270m" in self.model_id.lower():
                from px_patches.gemma3_270m_px_baseline.relay_inject import install_relay
            elif "bonsai" in self.model_id.lower() or "27b" in self.model_id.lower():
                from px_patches.ternary_bonsai_27b_px.relay_inject import install_relay
            else:
                from px_patches.gemma4_2b_px.relay_inject import install_relay

            install_relay(
                self.text_model,
                sign=self.sign,
                alpha_frac=self.alpha,
                layer=self.layer_idx,
                dwidth=self.vector,
            )

        self.active_hooks_count = len(self.target_layer._forward_hooks)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.text_model is not None:
            if "gemma4" in self.model_id.lower() or "e2b" in self.model_id.lower():
                from px_patches.gemma4_2b_px.relay_inject import remove_relay
            elif "270m" in self.model_id.lower():
                from px_patches.gemma3_270m_px_baseline.relay_inject import remove_relay
            elif "bonsai" in self.model_id.lower() or "27b" in self.model_id.lower():
                from px_patches.ternary_bonsai_27b_px.relay_inject import remove_relay
            else:
                from px_patches.gemma4_2b_px.relay_inject import remove_relay

            remove_relay(self.text_model)
            if self.target_layer is not None and hasattr(self.target_layer, "_forward_hooks"):
                self.target_layer._forward_hooks.clear()
        self.active_hooks_count = 0
        return False
