"""gradio_tabs/auto_tune_defaults.py — Pure-logic auto-tune defaults.

Provides the calibrated parameter values that the Gradio webapp should
display/send when auto-tune is enabled, so that it matches the
streaming-bridge behaviour (model-calibrated params instead of hardcoded
slider values). This module is pure logic — it does NOT import gradio and
does NOT touch the chat_tab; integration is done later by the orchestrator.

Bridge parity reference (streaming_bridge.py payload):
  temperature=0.7, max_tokens=1024, no top_p (schema default 0.9),
  no repetition_penalty, no px_gamma.

When auto-tune is ON:
  - px_gamma -> None  (so model_manager uses registry calibrated gamma; bridge parity)
  - top_p -> 0.9       (bridge schema default)
  - repetition_penalty -> 1.15  (PX-default for non-BASELINE patched models)
  - temperature -> user value (bridge & webapp both 0.7)
"""
from typing import Optional

from config import MODEL_REGISTRY

# Params that auto-tune locks when enabled (the ones that diverged bridge vs webapp).
# Plan 5.3: temperature is now also locked when auto-tune is ON — model-calibrated
# values (270m -> 0.3, 1b -> 0.6, 4b/e2b -> 1.0) are used via scale_adaptive_temperature.
AUTO_TUNABLE_PARAMS = (
    "temperature", "top_p", "repetition_penalty", "px_gamma",
)


def calibrated_gamma(model_id: str) -> Optional[float]:
    """Registry-calibrated gamma for model_id, from
    MODEL_REGISTRY[model_id]['patch_kwargs']['gamma'].
    Returns None for unknown models (caller should treat as 'no override / use slider').
    """
    entry = MODEL_REGISTRY.get(model_id)
    if entry is None:
        return None
    pk = entry.get("patch_kwargs") or {}
    g = pk.get("gamma")
    if isinstance(g, (int, float)):
        return float(g)
    return None


def bridge_top_p() -> float:
    """The effective top_p the streaming-bridge uses (schema default 0.9)."""
    return 0.9


def bridge_repetition_penalty() -> float:
    """The PX-default repetition_penalty for non-BASELINE patched models."""
    return 1.15


def scale_adaptive_temperature(model_id: str) -> Optional[float]:
    """CLAUDE.md §2.III scale-adaptive temperature by model hidden_size class:
    270m -> 0.3, 1b -> 0.6, 4b/e2b -> 1.0. Returns None for unknown model_id.
    Exposed for future use; the auto-tune LOCK does NOT use it (bridge sends 0.7).
    """
    mid = model_id.lower() if model_id else ""
    if "gemma3-270m" in mid:
        return 0.3
    if "gemma3-1b" in mid:
        return 0.6
    if "gemma3-4b" in mid or "gemma4-e2b" in mid or "gemma4_2b" in mid:
        return 1.0
    return None


def calibrated_values(model_id: str) -> dict:
    """The locked values auto-tune should DISPLAY and SEND when enabled.
    Returns dict with keys px_gamma (the registry float, or None if unknown),
    top_p (0.9), repetition_penalty (1.15). NOTE: for px_gamma the value to
    DISPLAY on the slider is calibrated_gamma(model_id); the value to SEND to
    get_model is None (so the registry calibrated gamma applies — bridge parity).
    This dict returns the DISPLAY values; see resolve_for_backend for send values.
    """
    return {
        "px_gamma": calibrated_gamma(model_id),
        "top_p": bridge_top_p(),
        "repetition_penalty": bridge_repetition_penalty(),
    }


def resolve_for_backend(model_id: str, auto_tune_on: bool, user_values: dict) -> dict:
    """Decide what chat_fn sends to manager.get_model / model.generate.
    When auto_tune_on=True:
      - px_gamma -> None  (so registry calibrated gamma applies; bridge parity)
      - top_p -> bridge_top_p() = 0.9
      - repetition_penalty -> bridge_repetition_penalty() = 1.15
      - temperature -> user_values.get("temperature", 0.7)  (NOT locked; bridge & webapp both 0.7)
      - max_tokens -> user_values.get("max_tokens", 1024)
    When auto_tune_on=False:
      - pass through user_values for all params (px_gamma=user_values['px_gamma'], etc.)
    Returns a dict with keys: px_gamma, top_p, repetition_penalty, temperature, max_tokens.
    user_values keys that may be missing default to: px_gamma=None, top_p=0.9,
    repetition_penalty=1.15, temperature=0.7, max_tokens=1024.
    """
    def _u(key, default):
        v = user_values.get(key, default)
        return v if v is not None else default

    if auto_tune_on:
        return {
            "px_gamma": None,
            "top_p": bridge_top_p(),
            "repetition_penalty": bridge_repetition_penalty(),
            "temperature": _u("temperature", 0.7),
            "max_tokens": _u("max_tokens", 1024),
        }
    return {
        "px_gamma": _u("px_gamma", None),
        "top_p": _u("top_p", 0.9),
        "repetition_penalty": _u("repetition_penalty", 1.15),
        "temperature": _u("temperature", 0.7),
        "max_tokens": _u("max_tokens", 1024),
    }