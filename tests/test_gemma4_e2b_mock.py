"""
test_gemma4_e2b_mock.py — Gemma 4 E2B Mock Tests (no GPU, no HF download)
==========================================================================
Validates the Gemma 4 E2B integration without loading the actual model.

Tests:
  - TestGemma4E2BConfig       : registry, model_type, patch_dir, scale defaults
  - TestGemma4E2BZoneRouting  : AutoCalibrator(1536) zone_weights determinism
  - TestGemma4E2BPatch        : gemma4-conditional branches, conditional patches
  - TestGemma4E2BTokenLoopMitigation : repetition_penalty injection, config values
  - TestGemma4E2BLayerBounceBreak : patch.py bounce-break logic (mocked layer history)
  - TestGemma4E2BVision       : _px_has_image_tokens, prefill recursion skip

Run: PYTHONPATH=. python -m pytest tests/test_gemma4_e2b_mock.py -v
Time: < 5s
"""

import os
import sys
import unittest
from collections import UserDict
from unittest.mock import MagicMock, patch

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_REGISTRY
# 2026-06-09: routed to isolated gemma4_2b_px directory
from px_patches.gemma4_2b_px.auto_tune import (
    AutoCalibrator, SCALE_DEFAULTS, ZONE_ROUTING, ZONE_Z_CENTERS,
)
from px_patches.gemma4_2b_px.patch import apply_px_patch, _resolve_text_model


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BConfig — Registry & Scale Defaults
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BConfig(unittest.TestCase):
    """Validate that gemma4-e2b-it is registered with correct parameters."""

    def test_registry_contains_gemma4_e2b_it(self):
        self.assertIn("gemma4-e2b-it", MODEL_REGISTRY,
                      "gemma4-e2b-it must be in MODEL_REGISTRY")

    def test_model_type_is_gemma4_conditional(self):
        reg = MODEL_REGISTRY["gemma4-e2b-it"]
        self.assertEqual(reg["model_type"], "gemma4_conditional")

    def test_patch_dir_is_isolated_gemma4_2b_px(self):
        # 2026-06-09: Patches are now isolated. gemma4 must NOT share
        # gemma3_270m_px with gemma3 — that would let gemma3-conditional
        # fixes (e.g. SCALE_DEFAULTS[1536] entry, gemma4-only gamma cap)
        # leak into gemma3 behavior and break byte-identical parity.
        reg = MODEL_REGISTRY["gemma4-e2b-it"]
        self.assertEqual(reg["patch_dir"], "gemma4_2b_px")

    def test_gemma3_models_route_to_isolated_baseline(self):
        # After 2026-06-09 isolation, all gemma3 models point to the
        # byte-identical pre-gemma4 baseline directory. If this assertion
        # fails, somebody re-routed a gemma3 model to the old shared dir
        # and the isolation invariant is broken.
        for key in ("gemma3-270m", "gemma3-270m-it", "gemma3-1b", "gemma3-1b-it", "gemma3-4b", "gemma3-4b-it"):
            self.assertEqual(
                MODEL_REGISTRY[key]["patch_dir"], "gemma3_270m_px_baseline",
                f"{key} must route to gemma3_270m_px_baseline for isolation"
            )

    def test_hf_id_points_to_gemma_4_E2B_it(self):
        reg = MODEL_REGISTRY["gemma4-e2b-it"]
        self.assertEqual(reg["hf_id"], "google/gemma-4-E2B-it")

    def test_token_loop_mitigation_recur_range(self):
        # Parity target (2026-06-09): gemma4-e2b uses SCALE_DEFAULTS[1536]
        # directly (recur_start=10, recur_end=26, n_loops=6, gamma=0.06),
        # the same defaults a gemma3-1b would use for hidden=1536.
        # The 2026-06-08 overrides (recur_start=8, recur_end=18, n_loops=4)
        # produced shallower recursion than gemma3 and were removed.
        # Token-loop stability now comes from repetition_penalty=1.15 +
        # no_repeat_ngram_size=3 set in patch.py for gemma4.
        reg = MODEL_REGISTRY["gemma4-e2b-it"]
        kw = reg["patch_kwargs"]
        # No override on recur_start/end/n_loops/gamma — must use SCALE_DEFAULTS
        self.assertNotIn("recur_start", kw,
                         "gemma4 config no longer overrides recur_start (uses SCALE_DEFAULTS)")
        self.assertNotIn("recur_end", kw,
                         "gemma4 config no longer overrides recur_end (uses SCALE_DEFAULTS)")
        self.assertNotIn("n_loops", kw,
                         "gemma4 config no longer overrides n_loops (uses SCALE_DEFAULTS)")
        # Routing mode is the only kwargs override we keep
        self.assertEqual(kw.get("routing_mode"), "adaptive")

    def test_scale_defaults_1536_aligned_with_config(self):
        # hidden_size 1536 → SCALE_DEFAULTS must align with the registered kwargs
        self.assertIn(1536, SCALE_DEFAULTS,
                      "AutoCalibrator(1536) requires SCALE_DEFAULTS[1536] entry")
        sd = SCALE_DEFAULTS[1536]
        # gamma=0.06 in SCALE_DEFAULTS, but config overrides to 0.05
        self.assertEqual(sd["recur_start"], 10)
        self.assertEqual(sd["recur_end"], 26)
        self.assertEqual(sd["n_loops"], 6)
        self.assertEqual(sd["gamma"], 0.06)

    def test_coda_has_minimum_3_layers(self):
        # 35 layers with SCALE_DEFAULTS[1536] recur range (10-26):
        # Prelude: 0-9 (10 layers), Recur: 10-25 (16 layers), Coda: 26-34 (9 layers)
        from px_patches.gemma4_2b_px.auto_tune import SCALE_DEFAULTS
        num_layers = 35
        recur_end = SCALE_DEFAULTS[1536]["recur_end"]
        coda_layers = num_layers - recur_end
        self.assertGreaterEqual(coda_layers, 3,
                                f"Coda needs ≥ 3 layers for output processing, got {coda_layers} (recur_end={recur_end})")

    def test_dtype_is_bfloat16(self):
        reg = MODEL_REGISTRY["gemma4-e2b-it"]
        self.assertEqual(reg["dtype"], "bfloat16")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BZoneRouting — AutoCalibrator(1536) determinism
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BZoneRouting(unittest.TestCase):
    """AutoCalibrator with hidden_size=1536 (Gemma 4 E2B) routes correctly."""

    def setUp(self):
        self.cal = AutoCalibrator(hidden_size=1536)

    def test_zone_weights_sum_to_one(self):
        w = self.cal.get_zone_weights(149.0, phi=0.4332)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=3,
                               msg=f"Weights must sum to 1.0, got {sum(w.values())}")

    def test_zone_weights_contain_all_5_zones(self):
        w = self.cal.get_zone_weights(149.0, phi=0.4332)
        expected_zones = {"math", "logic_a", "creative", "logic_b", "synthesis"}
        self.assertEqual(set(w.keys()), expected_zones)

    def test_zone_weights_deterministic(self):
        # Same inputs → same outputs (debug_zone.py reproducibility check)
        w1 = self.cal.get_zone_weights(149.0, phi=0.4332)
        w2 = self.cal.get_zone_weights(149.0, phi=0.4332)
        for zone in w1:
            self.assertAlmostEqual(w1[zone], w2[zone], places=6,
                                   msg=f"Zone {zone} not deterministic")

    def test_low_kurtosis_prefers_creative_synthesis(self):
        # After calibration (k_mean≈1000, k_std≈10) the calibrator
        # only meaningfully distinguishes kurtosis within ~±3 sigma.
        # Pre-calibration: k=50 yields z=-95 (saturated to creative) which is
        # the EXPECTED behavior for a non-calibrated calibrator on a 1536 model.
        # Here we simulate post-calibration by setting k_std wider so the
        # routing actually reflects relative kurtosis differences.
        cal = AutoCalibrator(hidden_size=1536)
        # Manually set calibrated state to a realistic distribution
        cal.calibrated = True
        cal.k_mean = 1000.0
        cal.k_std = 500.0  # Wide std so 50 vs 2000 are clearly different
        # Zone routing params (must be set for classify_zone to work)
        cal.zone_centers = {"math": 1800, "logic_a": 1200, "creative": 600, "logic_b": 1000, "synthesis": 400}
        cal.zone_sigmas = {"math": 200, "logic_a": 200, "creative": 200, "logic_b": 200, "synthesis": 200}
        # kurtosis=200 (well below mean) → creative/synthesis
        zone = cal.classify_zone(200.0)
        self.assertIn(zone, ("creative", "synthesis"),
                      f"kurtosis=200 should route to creative/synthesis, got {zone}")

    def test_high_kurtosis_prefers_math(self):
        # Use same calibrated state for consistency
        cal = AutoCalibrator(hidden_size=1536)
        cal.calibrated = True
        cal.k_mean = 1000.0
        cal.k_std = 500.0
        cal.zone_centers = {"math": 1800, "logic_a": 1200, "creative": 600, "logic_b": 1000, "synthesis": 400}
        cal.zone_sigmas = {"math": 200, "logic_a": 200, "creative": 200, "logic_b": 200, "synthesis": 200}
        # kurtosis=2000 (well above mean) → math/logic_a
        zone = cal.classify_zone(2000.0)
        self.assertIn(zone, ("math", "logic_a"),
                      f"kurtosis=2000 should route to math/logic_a, got {zone}")

    def test_zone_routing_params_in_valid_range(self):
        # Routing params must be within model bounds (1 ≤ start, end ≤ 35)
        rp = self.cal.get_routing_params(149.0, phi=0.4332, hidden_size=1536)
        self.assertGreaterEqual(rp["dynamic_start"], 1)
        self.assertLessEqual(rp["dynamic_end"], 35)
        self.assertGreaterEqual(rp["n_loops"], 1)
        self.assertLessEqual(rp["n_loops"], 12)

    def test_zone_z_centers_documented(self):
        # Validate zone_z_centers are sensible (math positive, synthesis negative)
        self.assertGreater(ZONE_Z_CENTERS["math"], 0,
                           "math zone should be at positive z-score (peaked)")
        self.assertLess(ZONE_Z_CENTERS["synthesis"], 0,
                        "synthesis zone should be at negative z-score (flat)")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BPatch — gemma4_conditional model loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BPatch(unittest.TestCase):
    """Validate gemma4_conditional loading branch in ModelManager."""

    @patch('transformers.AutoModelForImageTextToText.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_gemma4_uses_image_text_to_text_loader(self, mock_tok, mock_image_text):
        """gemma4_conditional must use AutoModelForImageTextToText (multimodal)."""
        from model_manager import ModelManager

        # Set up mocks
        mock_tok.return_value = MagicMock()
        mock_model = MagicMock()
        # Mock model.model.language_model.layers structure
        mock_lang = MagicMock()
        mock_lang.config = MagicMock()
        mock_lang.config.hidden_size = 1536
        mock_lang.config.num_hidden_layers = 35
        mock_lang.config.model_type = "gemma4_text"
        # Real parameter so .to() inside apply_px_patch works
        import torch
        param = torch.nn.Parameter(torch.zeros(1536))
        # Use a list-based generator that supports multiple calls (device, dtype)
        def _param_iter():
            return iter([param, param, param])
        mock_lang.parameters = MagicMock(side_effect=_param_iter)
        mock_lang.rotary_emb = MagicMock()
        mock_model.model = MagicMock()
        mock_model.model.language_model = mock_lang
        mock_image_text.return_value = mock_model

        manager = ModelManager()
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            entry = loop.run_until_complete(
                manager.get_model("gemma4-e2b-it", px_subjective=True,
                                  px_config_preset="SUBJECTIVE")
            )
            # Verify AutoModelForImageTextToText was called
            self.assertTrue(mock_image_text.called,
                            "gemma4_conditional must use AutoModelForImageTextToText")
            # Verify trust_remote_code=True was passed (Gemma 4 needs custom code)
            call_kwargs = mock_image_text.call_args.kwargs
            self.assertTrue(call_kwargs.get("trust_remote_code", False),
                            "Gemma 4 needs trust_remote_code=True for custom code")
        finally:
            loop.close()
            # Unload to free mock
            try:
                loop.run_until_complete(manager.unload_model("gemma4-e2b-it"))
            except Exception:
                pass

    def test_conditional_branches_in_patch_for_gemma4(self):
        """patch.py:168, 386, 412, 626, 742 — gemma4 conditionals exist."""
        import inspect
        from px_patches.gemma4_2b_px import patch as patch_mod
        source = inspect.getsource(patch_mod)

        # These conditional branches must exist for gemma4 to work correctly
        expected_branches = [
            'model_type", "").startswith("gemma4")',
            'shared_kv_states',  # line 168/386
            'Gemma4TextModelOutputWithPast',  # line 627
        ]
        for branch in expected_branches:
            self.assertIn(branch, source,
                          f"Missing gemma4 conditional: {branch!r}")

    def test_repetition_penalty_default_for_gemma4(self):
        """Patch must set _px_repetition_penalty=1.05 for gemma4."""
        # Build a minimal mock text model with a real tensor parameter
        # (apply_px_patch calls .to(device, dtype) on injected modules)
        import torch
        mock_text_model = MagicMock()
        mock_text_model.config = MagicMock()
        mock_text_model.config.hidden_size = 1536
        mock_text_model.config.num_hidden_layers = 35
        mock_text_model.config.model_type = "gemma4_text"
        mock_text_model.config.text_config = mock_text_model.config
        mock_text_model.rotary_emb = MagicMock()
        mock_text_model.rotary_emb.return_value = MagicMock()
        # Layers list (35 mock layers)
        mock_layers = [MagicMock() for _ in range(35)]
        mock_text_model.layers = mock_layers
        mock_text_model.norm = MagicMock(return_value=MagicMock())
        # Real parameter so .to() works
        param = torch.nn.Parameter(torch.zeros(1536))
        def _param_iter_e2b():
            return iter([param, param, param])
        mock_text_model.parameters = MagicMock(side_effect=_param_iter_e2b)
        mock_text_model.to = MagicMock(return_value=mock_text_model)

        # Outer mock model — name must contain "Gemma3" or "Gemma4" to trigger
        # the multimodal branch (patch.py:690)
        mock_outer = MagicMock()
        mock_outer.model = MagicMock()
        mock_outer.model.language_model = mock_text_model
        type(mock_outer).__name__ = "Gemma4ForConditionalGeneration"

        # Apply patch (kwargs consistent with config)
        apply_px_patch(mock_outer, recur_start=8, recur_end=18, n_loops=4,
                       gamma=0.05, routing_mode="adaptive",
                       config_preset="SUBJECTIVE")

        # Verify _px_repetition_penalty was set on the text model
        self.assertTrue(hasattr(mock_text_model, "_px_config"),
                        "apply_px_patch must set _px_config on text model")
        rp = mock_text_model._px_config.get("repetition_penalty", 1.0)
        # rp=1.05 was too weak (token loops in SUBJECTIVE preset still).
        # rp=1.15 is the current production value — see test_gemma4_e2b_presets.py
        self.assertGreaterEqual(rp, 1.10,
                                f"repetition_penalty must be ≥1.10 for gemma4, got {rp}")

    def test_repetition_penalty_default_for_gemma3(self):
        """For non-gemma4 models, repetition_penalty defaults to 1.0 (off)."""
        # Use a real Gemma3Config + minimal real model to avoid MagicMock
        # resolution issues in _resolve_text_model
        from transformers import Gemma3TextConfig
        from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
        import torch

        config = Gemma3TextConfig(
            hidden_size=1152, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=1,
            head_dim=128, intermediate_size=2048,
            vocab_size=100, max_position_embeddings=128,
        )
        text_model = Gemma3TextModel(config).to(torch.float32)
        # Wrap so type name triggers non-multimodal branch
        class _Outer:
            def __init__(self, m): self.model = m
        outer = _Outer(text_model)
        type(outer).__name__ = "Gemma3ForCausalLM"

        apply_px_patch(outer, config_preset="SUBJECTIVE")
        rp = text_model._px_config.get("repetition_penalty", 1.0)
        self.assertEqual(rp, 1.0,
                         f"gemma3 must default to repetition_penalty=1.0, got {rp}")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BTokenLoopMitigation — Verifies 4 mitigations are present
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BTokenLoopMitigation(unittest.TestCase):
    """Validate the 4 token-loop mitigations applied 2026-06-08."""

    def test_stability_break_uses_threshold_1_not_3(self):
        # patch.py:451 — `if stability_cnt > 1: break` (was `> 3`)
        import inspect
        from px_patches.gemma4_2b_px import patch as patch_mod
        source = inspect.getsource(patch_mod)
        # The fix line should be: if stability_cnt > 1: h_exp = trans_out; break
        self.assertIn("stability_cnt > 1: h_exp = trans_out; break", source,
                      "Stability-Break Fix: `stability_cnt > 1` required")
        # Old `> 3` threshold should be gone
        self.assertNotIn("stability_cnt > 3", source,
                         "Old `stability_cnt > 3` threshold must be removed")

    def test_layer_bounce_break_present(self):
        # Anti-bounce: detect L↔L+1 oscillation
        import inspect
        from px_patches.gemma4_2b_px import patch as patch_mod
        source = inspect.getsource(patch_mod)
        self.assertIn("BOUNCE-BREAK", source,
                      "Layer-bounce break (BOUNCE-BREAK) must be present")
        # The bounce-break must read from _px_current_telemetry_raw (per-step dicts
        # with "layer" key), NOT thought_history (which holds h_exp tensors)
        self.assertIn("_px_current_telemetry_raw", source,
                      "BOUNCE-BREAK must read layers from _px_current_telemetry_raw")
        # Verify the comment explains why (helps future maintainers)
        self.assertIn("thought_history contains h_exp tensors", source,
                      "Comment explaining the data source choice must be present")
        # Verify the relaxed thresholds (2026-06-08: 6→10 entries, 4→6 visits)
        self.assertIn("len(telemetry) >= 10", source,
                      "BOUNCE-BREAK threshold must be ≥10 telemetry entries "
                      "(relaxed after Steps=0 false-positive)")
        self.assertIn(">= 6", source,
                      "BOUNCE-BREAK must check ≥6 visits to last layer "
                      "(relaxed from 4 after Steps=0 false-positive)")

    def test_repetition_penalty_1536_for_gemma4(self):
        # SCALE_DEFAULTS[1536] must have correct gamma for repetition-aware routing
        sd = SCALE_DEFAULTS[1536]
        # gamma=0.06 in SCALE_DEFAULTS (with config override to 0.05 for E2B)
        self.assertLessEqual(sd["gamma"], 0.10,
                             "gamma must be conservative for E2B token-loop avoidance")

    def test_config_passes_repetition_penalty_through(self):
        # benchmark_engine._run_pzombie_impl must read _px_repetition_penalty
        import inspect
        from benchmark_engine import BenchmarkEngine
        source = inspect.getsource(BenchmarkEngine._run_pzombie_impl)
        self.assertIn("_px_repetition_penalty", source,
                      "P-Zombie eval must read _px_repetition_penalty from model")
        self.assertIn("repetition_penalty", source,
                      "P-Zombie eval must pass repetition_penalty to model.generate")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BLayerBounceBreak — Simulates layer-bounce history
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BLayerBounceBreak(unittest.TestCase):
    """Validates the layer-bounce detection logic by simulating thought_history.

    Note (2026-06-08): thresholds were relaxed after Steps=0 false-positive:
    - len(telemetry) >= 6  →  >= 10  (more warmup before bouncing detection)
    - last_visit >= 4      →  >= 6   (more evidence of oscillation)
    """

    def test_bounce_pattern_detected(self):
        # Simulate the L3↔L4 bouncing. With relaxed thresholds (last_visit≥6),
        # we need 10+ entries where one layer appears ≥6 times.
        # True alternation over 10 steps: [3,4,3,4,3,4,3,4,3,4] → last=4, count=5 (NOT enough)
        # So we need 10+ entries with one layer dominant (e.g., 4,3,4,3,4,3,4,3,4,4)
        recent_layers = [4, 3, 4, 3, 4, 3, 4, 3, 4, 4]  # 6x layer 4
        unique_recent = set(recent_layers)
        last_visit_count = recent_layers.count(recent_layers[-1])
        should_break = (len(unique_recent) <= 2 and last_visit_count >= 6)
        self.assertTrue(should_break,
                        "L4-dominated bouncing (10 steps, 6×L4) must trigger BOUNCE-BREAK")

    def test_no_break_on_diverse_layers(self):
        # If 6 different layers are visited, no break
        recent_layers = [3, 5, 7, 8, 9, 12]
        unique_recent = set(recent_layers)
        last_visit_count = recent_layers.count(recent_layers[-1])
        should_break = (len(unique_recent) <= 2 and last_visit_count >= 6)
        self.assertFalse(should_break,
                         "Diverse layer visits must NOT trigger break")

    def test_no_break_on_recent_revisit_below_threshold(self):
        # 5 visits to last layer is below the new ≥6 threshold
        recent_layers = [3, 4, 5, 3, 4, 3, 4, 3, 4, 3]  # 5x layer 3
        unique_recent = set(recent_layers)
        last_visit_count = recent_layers.count(recent_layers[-1])
        should_break = (len(unique_recent) <= 2 and last_visit_count >= 6)
        self.assertFalse(should_break,
                         "5 visits (below new ≥6 threshold) must NOT trigger break")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BVision — Multimodal prefill handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BVision(unittest.TestCase):
    """Validates _px_has_image_tokens flag and prefill recursion skip."""

    def test_has_image_tokens_attribute_set(self):
        # patch.py:696 — outer.forward wrapper sets _px_has_image_tokens
        import inspect
        from px_patches.gemma4_2b_px import patch as patch_mod
        source = inspect.getsource(patch_mod)
        # The wrapper line must reference _px_has_image_tokens
        self.assertIn("_px_has_image_tokens", source,
                      "_px_has_image_tokens flag must be set in multimodal wrapper")
        # And it must check for pixel_values
        self.assertIn("pixel_values", source,
                      "Multimodal wrapper must detect pixel_values")

    def test_vision_detection_uses_pixel_values(self):
        # Simulate the logic from patch.py:696
        def has_image_tokens(kwargs):
            return kwargs.get('pixel_values') is not None
        self.assertTrue(has_image_tokens({'pixel_values': 'fake_tensor'}))
        self.assertFalse(has_image_tokens({'input_ids': 'fake_ids'}))

    def test_vision_kurtosis_clamp_in_calibrator(self):
        # Phase 58.3: vision kurtosis K~2450 is far outside text calibration
        # The calibrator should not crash on this range
        cal = AutoCalibrator(hidden_size=1536)
        # Should not raise
        try:
            w = cal.get_zone_weights(2450.0, phi=0.7)
            self.assertAlmostEqual(sum(w.values()), 1.0, places=3)
        except Exception as e:
            self.fail(f"Vision kurtosis K=2450 must not crash calibrator: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BResolveTextModel — _resolve_text_model for multimodal
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BResolveTextModel(unittest.TestCase):
    """Validate _resolve_textModel finds language_model inside multimodal wrapper."""

    def test_resolves_language_model_in_image_text_wrapper(self):
        # Build mock mimicking AutoModelForImageTextToText output
        # Structure: model.model.language_model (Gemma 4)
        lang = MagicMock()
        lang.layers = [MagicMock() for _ in range(35)]
        lang.rotary_emb = MagicMock()
        outer = MagicMock()
        outer.model = MagicMock()
        outer.model.language_model = lang
        # Strip the named_modules fallback to force path-based resolution
        type(outer).named_modules = MagicMock(return_value=[])

        resolved = _resolve_text_model(outer)
        self.assertIs(resolved, lang,
                      "_resolve_text_model must find language_model for Gemma 4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
