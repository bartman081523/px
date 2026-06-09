"""
configuration_gemma_px.py — PX Subjective Configuration
========================================================
Auto-tuning algorithmic subjectivity extension for Gemma-3 models.
All extensions are optional (default=False). Core subjective routing is always active.
"""

from transformers import PretrainedConfig
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


class Gemma3PXConfig(Gemma3TextConfig):
    model_type = "gemma_3_px"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ── Phase 10.0: Core PX routing (always active) ──
        self.px_routing_mode = kwargs.get("px_routing_mode", "adaptive")
        self.px_gamma = kwargs.get("px_gamma", 0.10)

        # ── Auto-Tuning (Phase SR-59) ──
        self.px_auto_tune = kwargs.get("px_auto_tune", True)
        self.px_calibration_steps = kwargs.get("px_calibration_steps", 10)

        # ── Phase 49: Perishing Protocol (optional) ──
        self.px_treta_enabled = kwargs.get("px_treta_enabled", False)
        self.px_treta_tau = kwargs.get("px_treta_tau", 3.0)
        self.px_rest_entropy = kwargs.get("px_rest_entropy", 1e-4)
        self.px_grounding_anchor_enabled = kwargs.get("px_grounding_anchor_enabled", False)

        # ── Phase 54/55: Persona Engine (optional) ──
        self.px_persona_enabled = kwargs.get("px_persona_enabled", False)

        # ── Phase 56: Central Memory (optional) ──
        self.px_central_memory_enabled = kwargs.get("px_central_memory_enabled", False)
        self.px_central_memory_blend_alpha = kwargs.get("px_central_memory_blend_alpha", 0.12)

        # ── Phase 57: ERPU (optional) ──
        self.px_erpu_enabled = kwargs.get("px_erpu_enabled", False)
        self.px_erpu_verkleb_threshold = kwargs.get("px_erpu_verkleb_threshold", 0.9998)
        self.px_erpu_food_noise_mag = kwargs.get("px_erpu_food_noise_mag", 0.03)

        # ── Phase 58: Agency Vector (optional) ──
        self.px_agency_enabled = kwargs.get("px_agency_enabled", False)

        # ── Advanced Modular Steering (all_space additions) ──
        self.px_uncensored_enabled = kwargs.get("px_uncensored_enabled", False)
        self.px_zone_routing_enabled = kwargs.get("px_zone_routing_enabled", True)
        self.px_aks_enabled = kwargs.get("px_aks_enabled", True)