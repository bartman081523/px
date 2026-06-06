"""PX Subjective — Model Entry Point for Gemma-3"""
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM, Gemma3Model
from .configuration_gemma_px import Gemma3PXConfig
from .patch import apply_px_patch


class Gemma3PXForCausalLM(Gemma3ForCausalLM):
    config_class = Gemma3PXConfig

    def __init__(self, config):
        super().__init__(config)
        routing_mode = getattr(config, "px_routing_mode", "adaptive")
        gamma = getattr(config, "px_gamma", 0.08)
        # Auto-tuning is always on by default
        config_kwargs = {}
        for attr in ["px_auto_tune", "px_calibration_steps",
                      "px_central_memory_enabled", "px_erpu_enabled",
                      "px_agency_enabled", "px_treta_enabled",
                      "px_grounding_anchor_enabled", "px_persona_enabled"]:
            if hasattr(config, attr):
                config_kwargs[attr] = getattr(config, attr)
        apply_px_patch(self, recur_start=5, recur_end=12,
                       routing_mode=routing_mode, gamma=gamma, **config_kwargs)


class Gemma3PXModel(Gemma3Model):
    config_class = Gemma3PXConfig

    def __init__(self, config):
        super().__init__(config)