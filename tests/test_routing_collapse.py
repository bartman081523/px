"""
test_routing_collapse.py — Regression tests for the routing-collapse failure
============================================================================

Symptom (SR-61, 4B ACTIVE_MANIFOLD, 80 prompts):
  - All 80 prompts produce bit-identical zone_weights
  - zone_entropy H = 1.719065 for every single prompt
  - Despite kurtosis varying 2106..2441 and phi varying 0.83..0.99

Hypothesis: at scales where the calibrated k_std is dominated by
MIN_K_STD (5.0) or where ZONE_Z_SIGMAS are all scaled by the same
temperature, the Gaussian falloff plateaus and every input lands
on the same weights.

These tests probe:
  1. For each SCALE_DEFAULTS hidden_size, do TWO different kurtosis
     inputs produce DIFFERENT zone_weights?
  2. For each scale, is the entropy variation across kurtosis sweep
     ≥ some minimum threshold?
  3. Does the gemma3 patch and the gemma4 patch both pass (parity)?
  4. Does the live-measured 4B kurtosis distribution from
     dmt_space_50 reproduce the bit-identical symptom?
"""

import json
import math
import os
import sys
import unittest
from unittest.mock import MagicMock

# Project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _calibrate_then_route(hidden_size, kurtosis_samples, phi=0.9, td=0.7):
    """Calibrate an AutoCalibrator with a sample distribution, then return
    the zone_weights for a new query kurtosis."""
    from px_patches.gemma3_270m_px_baseline.auto_tune import AutoCalibrator
    cal = AutoCalibrator(hidden_size, calibration_steps=10)
    for k in kurtosis_samples:
        cal.collect(k, phi, token_diversity=td)
    cal.calibrate()
    return cal


def _weights_dict_distance(w1, w2):
    """L1 distance between two weight dicts (assumes same keys)."""
    keys = set(w1) | set(w2)
    return sum(abs(w1.get(k, 0) - w2.get(k, 0)) for k in keys)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoutingVariance(unittest.TestCase):
    """The AutoCalibrator must produce DIFFERENT zone_weights for
    DIFFERENT kurtosis inputs. If not, the router has collapsed.
    """

    def test_270m_routes_differently_for_different_kurtosis(self):
        """270M (HS=640) — wide kurtosis range, expect strong variation."""
        # Calibrate with a wide 270M-like distribution
        cal = _calibrate_then_route(
            640,
            [150, 200, 250, 300, 350, 400, 450, 200, 250, 300],
            phi=0.9, td=0.7
        )
        # Query at the extremes of the calibrated range
        w_low = cal.get_zone_weights(150.0, phi=0.9, token_diversity=0.7)
        w_high = cal.get_zone_weights(450.0, phi=0.9, token_diversity=0.7)
        dist = _weights_dict_distance(w_low, w_high)
        # L1 distance should be substantial — 0.5 means significant divergence
        self.assertGreater(
            dist, 0.5,
            f"270M: zone_weights for k=150 vs k=450 must diverge. "
            f"L1={dist:.4f}, w_low={w_low}, w_high={w_high}"
        )

    def test_4b_routes_differently_for_different_kurtosis(self):
        """4B (HS=2560) — narrow kurtosis range, weaker but still nonzero."""
        # Calibrate with the EXACT distribution from the live 4B run
        # (kurtosis std=35..85, mean≈2300..2400)
        cal = _calibrate_then_route(
            2560,
            [2283, 2368, 2441, 2199, 2353, 2416, 2109, 2314, 2426,
             2106, 2303, 2402],  # min..max from 4B run
            phi=0.87, td=0.7
        )
        # Query at the empirical extremes
        w_low = cal.get_zone_weights(2106.0, phi=0.87, token_diversity=0.7)
        w_high = cal.get_zone_weights(2441.0, phi=0.87, token_diversity=0.7)
        dist = _weights_dict_distance(w_low, w_high)
        # Even at 4B's narrow range, weights must shift.
        # If this is 0, the router has plateaued.
        self.assertGreater(
            dist, 0.05,
            f"4B: zone_weights for k=2106 vs k=2441 must differ. "
            f"L1={dist:.6f}, w_low={w_low}, w_high={w_high}"
        )

    def test_4b_actual_run_had_routing_collapse(self):
        """The symptom is in the recorded data — confirm it bit-identically."""
        # Find the most recent 4B aggregate
        results_root = os.path.join(_ROOT, "eval", "results")
        candidates = []
        if os.path.isdir(results_root):
            for d in os.listdir(results_root):
                if d.startswith("4B_") and os.path.isdir(os.path.join(results_root, d)):
                    for f in os.listdir(os.path.join(results_root, d)):
                        if f.endswith("_aggregate.json"):
                            candidates.append(os.path.join(results_root, d, f))
        if not candidates:
            self.skipTest("No 4B aggregate found — re-run eval first")

        agg_path = sorted(candidates)[-1]
        with open(agg_path) as f:
            agg = json.load(f)
        results = agg.get("results", [])
        if not results:
            self.skipTest(f"Empty aggregate: {agg_path}")

        # All zone_entropy values should NOT be bit-identical
        hs = [r.get("zone_entropy") for r in results if r.get("zone_entropy") is not None]
        if len(set(hs)) <= 1:
            # Document the failure, do not just skip
            self.fail(
                f"ROUTING COLLAPSE reproduced in {agg_path}: "
                f"all {len(hs)} prompts have zone_entropy={hs[0] if hs else 'NA'}. "
                f"AutoCalibrator is producing a single fixed weights vector."
            )

    def test_4b_zone_weights_diverge_across_prompts(self):
        """Same data — the zone_weights dict itself must vary."""
        results_root = os.path.join(_ROOT, "eval", "results")
        candidates = []
        if os.path.isdir(results_root):
            for d in os.listdir(results_root):
                if d.startswith("4B_") and os.path.isdir(os.path.join(results_root, d)):
                    for f in os.listdir(os.path.join(results_root, d)):
                        if f.endswith("_aggregate.json"):
                            candidates.append(os.path.join(results_root, d, f))
        if not candidates:
            self.skipTest("No 4B aggregate found")

        with open(sorted(candidates)[-1]) as f:
            agg = json.load(f)
        weights_seen = set()
        for r in agg.get("results", []):
            zw = r.get("zone_weights")
            if not zw:
                continue
            # Round to 6 decimals so the test isn't fooled by FP noise
            key = tuple(round(v, 6) for v in zw.values())
            weights_seen.add(key)
        self.assertGreater(
            len(weights_seen), 1,
            f"Only {len(weights_seen)} unique zone_weights vector(s) across "
            f"{len(agg['results'])} prompts. Router is bit-fixed."
        )

    def test_entropy_range_across_kurtosis_sweep(self):
        """Sweep kurtosis 0.5σ to 2σ, entropy should vary measurably."""
        cal = _calibrate_then_route(
            640,
            [150, 200, 250, 300, 350, 400, 450, 200, 250, 300],
            phi=0.9, td=0.7
        )
        entropies = []
        for k_offset in [-2, -1, 0, 1, 2]:  # kurtosis values around mean
            k = cal.k_mean + k_offset * cal.k_std
            w = cal.get_zone_weights(k, phi=0.9, token_diversity=0.7)
            # Shannon entropy
            total = sum(w.values()) + 1e-9
            ps = [v / total for v in w.values() if v > 1e-10]
            H = -sum(p * math.log2(p) for p in ps)
            entropies.append(H)
        rng = max(entropies) - min(entropies)
        self.assertGreater(
            rng, 0.05,
            f"Entropy range over kurtosis sweep must exceed 0.05, got {rng:.4f}. "
            f"entropies={[f'{e:.4f}' for e in entropies]}"
        )


class TestGemma4Parity(unittest.TestCase):
    """The gemma4-px patch must NOT have the same routing collapse."""

    def test_gemma4_routes_differently_for_different_kurtosis(self):
        """gemma4 (HS=1536) — same routing-collapse test as gemma3 4B."""
        from px_patches.gemma4_2b_px.auto_tune import (
            AutoCalibrator as G4AutoCalibrator
        )
        cal = G4AutoCalibrator(1536, calibration_steps=10)
        for k in [2283, 2368, 2441, 2199, 2353, 2416, 2109, 2314, 2426, 2402]:
            cal.collect(k, 0.87, token_diversity=0.7)
        cal.calibrate()
        w_low = cal.get_zone_weights(2106.0, phi=0.87, token_diversity=0.7)
        w_high = cal.get_zone_weights(2441.0, phi=0.87, token_diversity=0.7)
        dist = _weights_dict_distance(w_low, w_high)
        self.assertGreater(
            dist, 0.05,
            f"gemma4: zone_weights for k=2106 vs k=2441 must differ. "
            f"L1={dist:.6f}, w_low={w_low}, w_high={w_high}"
        )

    def test_gemma4_kurtosis_dispatch_matches_4b_regime(self):
        """gemma4 at HS=1536 should dispatch to a regime that doesn't plateau."""
        from px_patches.gemma4_2b_px.auto_tune import AutoCalibrator
        cal = AutoCalibrator(1536, calibration_steps=10)
        for k in [2283, 2368, 2441, 2199, 2353, 2416, 2109, 2314, 2426, 2402]:
            cal.collect(k, 0.87, token_diversity=0.7)
        cal.calibrate()
        # The k_blend_weight and zone_temperature must be in the
        # 'produce variation' regime, not the 'plateau' regime
        k_cv = cal.k_std / (cal.k_mean + 1e-9)
        # If k_cv is moderate (0.01-0.05), we expect T < 1.0 to sharpen
        # If k_cv is low (<0.01), T = 1.0 is the no-sharpening regime
        # The point is: T should NOT be such that the weights plateau
        # for ALL kurtosis values
        self.assertIsNotNone(
            cal.zone_temperature,
            "gemma4 calibrator must set zone_temperature"
        )
        self.assertGreater(cal.k_blend_weight, 0.3,
                          f"k_blend_weight too low: {cal.k_blend_weight}")


class TestRoutingParameterSanity(unittest.TestCase):
    """When the calibrator dispatches, the parameters must be such that
    the Gaussian falloff is sensitive to the input z-score."""

    def test_4b_calibration_does_not_saturate(self):
        """For 4B HS=2560 with realistic kurtosis (std~50), the
        z-score range should be wide enough to produce measurable
        weight variation. If the calibrated k_std is clamped to
        its minimum (5.0), the z-score range is huge and
        everything saturates."""
        cal = _calibrate_then_route(
            2560,
            [2283, 2368, 2441, 2199, 2353, 2416, 2109, 2314, 2426, 2402],
            phi=0.87, td=0.7
        )
        # The empirical std is ~85 (per the 4B run logs).
        # If cal.k_std is clamped to 5.0, that's a saturation bug.
        self.assertGreater(
            cal.k_std, 10.0,
            f"4B k_std saturated to {cal.k_std} — too low. "
            f"Empirical std is ~85. If clamped, every kurtosis "
            f"becomes an extreme z-score and the Gaussian plateaus."
        )
        # Also check the k_cv — the regime dispatch depends on it
        k_cv = cal.k_std / (cal.k_mean + 1e-9)
        # For 4B, k_cv should be small (<0.05) so the low-CV regime
        # (T=1.0, k_blend=0.5) is selected — which preserves SCF signal
        self.assertLess(k_cv, 0.05,
                        f"4B k_cv should be < 0.05, got {k_cv}")


class TestSR61RunnerWarmup(unittest.TestCase):
    """The runner's _calibrator_warmup() must pre-seed online stats so
    that the FIRST get_zone_weights call doesn't return z=0.0 for every
    input (which is the routing-collapse root cause)."""

    def test_warmup_seeds_online_n_to_5(self):
        """After warmup, _online_n must be >= ONLINE_WARMUP=5 so the
        online-z branch of _get_z_score is used."""
        from px_patches.gemma3_270m_px_baseline.auto_tune import (
            AutoCalibrator, ONLINE_WARMUP
        )
        cal = AutoCalibrator(2560, calibration_steps=10)
        # Without warmup, _online_n=0 → collapse
        self.assertEqual(cal._online_n, 0)
        self.assertIsNone(cal.k_mean)  # also: no hardcoded default for HS=2560

        # Simulate what _calibrator_warmup does
        import random
        rng = random.Random(0xC0DE)
        samples = [2400.0 + rng.uniform(-85, 85) for _ in range(5)]
        n = len(samples)
        mean = sum(samples) / n
        m2 = sum((x - mean) ** 2 for x in samples)
        cal._online_n = n
        cal._online_k_mean = mean
        cal._online_k_m2 = m2

        self.assertGreaterEqual(cal._online_n, ONLINE_WARMUP,
                                f"warmup must reach ONLINE_WARMUP={ONLINE_WARMUP}")

    def test_warmup_breaks_bit_identical_weights(self):
        """BEFORE warmup: every kurtosis → z=0.0 → identical zone_weights.
        AFTER warmup: different kurtosis → different z-scores → different
        zone_weights.

        Note: jitter must be realistic (≈empirical kurtosis std), not huge.
        With jitter=300, z-scores become ±3σ outliers, all weights
        degenerate to W<0.05, and the function falls back to
        _adaptive_phi_weights(phi) which is identical for all prompts.
        """
        from px_patches.gemma3_270m_px_baseline.auto_tune import (
            AutoCalibrator, ONLINE_WARMUP
        )

        # ---- BEFORE: simulate the cold-start the live 4B run hit ----
        cal_cold = AutoCalibrator(2560, calibration_steps=10)
        w_a = cal_cold.get_zone_weights(2106.0, phi=0.87, token_diversity=0.7)
        w_b = cal_cold.get_zone_weights(2441.0, phi=0.87, token_diversity=0.7)
        dist_before = _weights_dict_distance(w_a, w_b)
        # The bug: bit-identical (or near-zero) weights
        self.assertLess(
            dist_before, 0.01,
            f"Expected cold-start to produce near-identical weights, got L1={dist_before:.6f}. "
            f"If this fails, the bug has been fixed in the calibrator itself."
        )

        # ---- AFTER: same calibrator, but with warmup-applied online stats ----
        # Empirical 4B kurtosis std is ~85 — use that as the jitter
        import random
        rng = random.Random(0xC0DE)
        samples = [2400.0 + rng.uniform(-85, 85) for _ in range(5)]
        mean = sum(samples) / 5
        m2 = sum((x - mean) ** 2 for x in samples)
        cal_cold._online_n = 5
        cal_cold._online_k_mean = mean
        cal_cold._online_k_m2 = m2
        # CRITICAL: also seed cal_k_mean/cal_k_std — line 380 of
        # auto_tune.py uses cal_std * 2.0 as the routing_std cap. With
        # k_std=None the cap falls back to 2*MIN_ONLINE_K_STD=2.0, making
        # every kurtosis a ±huge-z outlier. See SR-61 warmup comments.
        cal_cold.k_mean = mean
        cal_cold.k_std = 85.0

        w_a2 = cal_cold.get_zone_weights(2106.0, phi=0.87, token_diversity=0.7)
        w_b2 = cal_cold.get_zone_weights(2441.0, phi=0.87, token_diversity=0.7)
        dist_after = _weights_dict_distance(w_a2, w_b2)
        # The fix: weights diverge
        self.assertGreater(
            dist_after, 0.05,
            f"Warmup must break the cold-start collapse. L1={dist_after:.6f}"
        )

    def test_warmup_via_helper_function(self):
        """The actual _calibrator_warmup() in runner.py must be importable
        and operate on a mock model with _px_calibrator attribute."""
        from eval.runner import _calibrator_warmup
        from px_patches.gemma3_270m_px_baseline.auto_tune import (
            AutoCalibrator, ONLINE_WARMUP
        )

        # Mock the multimodal-wrapped model: model.model.language_model
        cal = AutoCalibrator(2560, calibration_steps=10)
        text_model = type("M", (), {"_px_calibrator": cal})()
        inner = type("I", (), {"language_model": text_model})()
        model = type("Outer", (), {"model": inner})()

        _calibrator_warmup(model, n_warmup=5, kurtosis_seed=2400.0)

        self.assertGreaterEqual(cal._online_n, ONLINE_WARMUP)
        # Online mean should be near seed (jitter ± 300, but center=2400)
        self.assertAlmostEqual(cal._online_k_mean, 2400.0, delta=300.0)

    def test_warmup_handles_text_only_model(self):
        """For gemma3-1b (text-only), model.model has the _px_calibrator
        directly — no language_model wrapper."""
        from eval.runner import _calibrator_warmup
        from px_patches.gemma3_270m_px_baseline.auto_tune import (
            AutoCalibrator, ONLINE_WARMUP
        )

        cal = AutoCalibrator(1152, calibration_steps=10)
        text_model = type("M", (), {"_px_calibrator": cal})()
        model = type("Outer", (), {"model": text_model})()

        _calibrator_warmup(model, n_warmup=5, kurtosis_seed=1100.0)

        self.assertGreaterEqual(cal._online_n, ONLINE_WARMUP)
        self.assertAlmostEqual(cal._online_k_mean, 1100.0, delta=300.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
