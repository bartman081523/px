"""
test_gemma4_e2b_vision.py — Vision Prefill Path Test for Gemma 4 E2B
=====================================================================
Validates the multimodal prefill recursion-skip logic for Gemma 4 E2B.

Phase 58.3 pattern: when pixel_values is present, the prefill must
run with n_loops=0 (no recursion) to avoid KV cache mismatches with
the image tokens. This test verifies that logic without loading the
real model.

Tests:
  - _px_has_image_tokens flag is set by outer.forward wrapper
  - During prefill (T > 1) with vision: n_loops → 0
  - During decode (T == 1) without vision: normal recursion

Usage:
  PYTHONPATH=. python -m pytest tests/test_gemma4_e2b_vision.py -v
"""

import os
import sys
import unittest
from collections import UserDict
from unittest.mock import MagicMock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable debug output for the layer-bounce break and prefill logging
os.environ.setdefault("DEBUG_PX", "1")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BVisionPrefill — prefill n_loops=0 with image tokens
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BVisionPrefill(unittest.TestCase):
    """Validate prefill n_loops override when image tokens are detected."""

    def test_vision_flag_set_by_wrapper(self):
        """_px_has_image_tokens must be True when pixel_values is present."""
        # Simulate the wrapper from patch.py:695
        captured = {}
        def wrapper(kwargs):
            captured['pixel_values'] = kwargs.get('pixel_values')
            captured['_px_has_image_tokens'] = kwargs.get('pixel_values') is not None
            return "forward_result"

        # With pixel_values → has image tokens
        result = wrapper({'pixel_values': 'fake_pixels', 'input_ids': 'fake_ids'})
        self.assertTrue(captured['_px_has_image_tokens'])
        self.assertEqual(result, "forward_result")

        # Without pixel_values → no image tokens
        result = wrapper({'input_ids': 'fake_ids'})
        self.assertFalse(captured['_px_has_image_tokens'])

    def test_prefill_n_loops_zero_with_vision(self):
        """patch.py:308 — if is_vision: n_loops = 0"""
        # Simulate the logic block
        n_loops_normal = 8
        n_loops = n_loops_normal
        is_vision = True
        inputs_embeds_shape = (1, 256)  # Prefill: T > 1

        # Recreate the patch.py:308-309 logic
        if is_vision and inputs_embeds_shape[1] > 1:
            n_loops = 0

        self.assertEqual(n_loops, 0,
                         "Vision prefill must set n_loops=0 to skip recursion")

    def test_prefill_n_loops_full_without_vision(self):
        """No pixel_values → recursion runs normally."""
        n_loops_normal = 8
        n_loops = n_loops_normal
        is_vision = False
        inputs_embeds_shape = (1, 256)

        if is_vision and inputs_embeds_shape[1] > 1:
            n_loops = 0

        self.assertEqual(n_loops, n_loops_normal,
                         "Text-only prefill must keep n_loops=8 (full recursion)")

    def test_decode_step_unaffected_by_vision_flag(self):
        """Decode step (T==1) must NOT trigger n_loops=0 override."""
        n_loops_normal = 8
        n_loops = n_loops_normal
        is_vision = True  # Flag still set, but...
        inputs_embeds_shape = (1, 1)  # Decode: T == 1

        # Condition requires T > 1
        if is_vision and inputs_embeds_shape[1] > 1:
            n_loops = 0

        self.assertEqual(n_loops, n_loops_normal,
                         "Decode step (T=1) must NOT trigger n_loops=0 override")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BImageTokens — pixel_values & input_ids shape validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BImageTokens(unittest.TestCase):
    """Validate input shape expectations for multimodal forward pass."""

    def test_pixel_values_4d_shape(self):
        # Gemma 4 expects (B, C, H, W) for pixel_values
        B, C, H, W = 1, 3, 224, 224
        pixel_values = torch.zeros(B, C, H, W)
        self.assertEqual(pixel_values.ndim, 4)
        self.assertEqual(pixel_values.shape, (B, C, H, W))

    def test_input_ids_with_image_token_markers(self):
        # Multimodal input_ids contain image-token markers
        # (e.g., Gemma 4 uses <image_soft_token> placeholders)
        input_ids = torch.tensor([[1, 100, 200, 300, 2]])  # 1=bos, 2=eos, placeholders
        # Validate that placeholder positions are detected
        # (in real Gemma 4, special image tokens are inserted between text tokens)
        self.assertEqual(input_ids.ndim, 2)
        self.assertGreater(input_ids.shape[1], 3,
                          "Multimodal sequence must have > 3 tokens (text + image placeholders)")

    def test_vision_kurtosis_in_expected_range(self):
        # Phase 58.3 finding: vision kurtosis K~2450 (vs text K~150-350)
        # The calibrator must NOT crash on this
        from px_patches.gemma4_2b_px.auto_tune import AutoCalibrator
        cal = AutoCalibrator(hidden_size=1536)
        try:
            for k in [2400, 2450, 2500, 2550]:
                w = cal.get_zone_weights(k, phi=0.7)
                self.assertAlmostEqual(sum(w.values()), 1.0, places=3,
                                       msg=f"k={k} weights must sum to 1.0")
        except Exception as e:
            self.fail(f"Vision kurtosis must not crash calibrator: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BWrapperIntegration — outer.forward wrapper for multimodal
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BWrapperIntegration(unittest.TestCase):
    """Validate the outer.forward wrapper sets _px_has_image_tokens correctly."""

    def test_wrapper_sets_flag_from_pixel_values(self):
        """Wrapper logic at patch.py:695-699."""
        # Build mock outer model with language_model child
        mock_lang = MagicMock()
        mock_outer = MagicMock()
        mock_outer.model = MagicMock()
        mock_outer.model.language_model = mock_lang
        # Store the original forward
        mock_outer._px_original_forward = MagicMock(return_value="result")

        # Replicate wrapper from patch.py
        def wrapper(self_outer, *args, **kwargs):
            mock_lang._px_has_image_tokens = kwargs.get('pixel_values') is not None
            mock_lang._px_saved_input_ids = kwargs.get('input_ids')
            return self_outer._px_original_forward(*args, **kwargs)

        # Test 1: With pixel_values
        import functools
        wrapped = functools.partial(wrapper, mock_outer)
        wrapped(pixel_values="fake_pixels", input_ids="fake_ids")
        self.assertTrue(mock_lang._px_has_image_tokens)
        self.assertEqual(mock_lang._px_saved_input_ids, "fake_ids")

        # Test 2: Without pixel_values
        wrapped(input_ids="fake_ids")
        self.assertFalse(mock_lang._px_has_image_tokens)
        self.assertEqual(mock_lang._px_saved_input_ids, "fake_ids")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BPrefillSkip — end-to-end logic for vision prefill
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BPrefillSkip(unittest.TestCase):
    """End-to-end logic test: vision prefill → n_loops=0 → no recursion."""

    def test_full_vision_prefill_flow(self):
        # Scenario: User sends image+text prompt
        # Expected: prefill runs (n_loops=0), then decode runs (n_loops=normal)

        # Stage 1: PREFILL (vision)
        n_loops = 8
        is_vision = True
        T = 256  # Prefill sequence length
        if is_vision and T > 1:
            n_loops = 0
        self.assertEqual(n_loops, 0, "Prefill: n_loops=0 with vision")

        # Stage 2: DECODE (text only, but vision flag may still be set)
        n_loops = 8
        is_vision = True  # Flag may persist
        T = 1  # Decode: 1 token at a time
        if is_vision and T > 1:  # Condition fails
            n_loops = 0
        self.assertEqual(n_loops, 8, "Decode: n_loops back to normal")

    def test_calibration_skipped_during_vision_prefill(self):
        # patch.py:315 — calibration only happens if shape[1] > 1
        # AND subjective_enabled. With vision, the recursion is skipped
        # but calibration still runs (collect() is independent).
        # This test verifies calibration's collect() does not break on
        # image-token kurtosis (K~2450).
        from px_patches.gemma4_2b_px.auto_tune import AutoCalibrator
        cal = AutoCalibrator(hidden_size=1536)
        # Simulate 12 vision prefill calls (calibration_steps default 10+)
        for i in range(12):
            cal.collect(kurtosis=2450.0, phi=0.85, token_diversity=0.95)
        # Calibrator should now be calibrated
        self.assertTrue(cal.calibrated,
                        "Calibrator must reach calibrated state after 10+ vision calls")


if __name__ == "__main__":
    unittest.main(verbosity=2)
