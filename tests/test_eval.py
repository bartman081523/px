"""
test_eval.py — Smoke test for eval/ package
============================================

Verifies that the eval/ package is self-consistent and that stats.py
can run on a synthetic dataset. Does NOT load any model — pure stdlib
+ json + math checks.

Run: PYTHONPATH=. python -m pytest tests/test_eval.py -v
"""

import json
import math
import os
import sys
import unittest
import tempfile

# Project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestEvalImports(unittest.TestCase):
    """The eval/ package must be importable as a unit."""

    def test_eval_package_imports(self):
        import eval
        # __init__.py must be a real module
        self.assertTrue(hasattr(eval, "__file__"))

    def test_runner_imports(self):
        from eval import runner
        self.assertTrue(hasattr(runner, "_run_one_prompt"))
        self.assertTrue(hasattr(runner, "PROMPTS"))
        self.assertTrue(hasattr(runner, "shannon_entropy"))
        self.assertTrue(hasattr(runner, "token_diversity"))

    def test_stats_imports(self):
        from eval import stats
        self.assertTrue(hasattr(stats, "analyze"))
        self.assertTrue(hasattr(stats, "one_way_anova"))
        self.assertTrue(hasattr(stats, "linear_r_squared"))

    def test_run_4b_eval_imports(self):
        from eval.run_4b_eval import run_eval, SCALE_TO_MODEL
        self.assertIn("4B", SCALE_TO_MODEL)
        self.assertEqual(SCALE_TO_MODEL["4B"], "gemma3-4b-it")
        self.assertEqual(SCALE_TO_MODEL["E2B"], "gemma4-e2b-it")


class TestPromptsValid(unittest.TestCase):
    """The PROMPTS dictionary must be the right shape."""

    def test_prompt_categories(self):
        from eval.runner import PROMPTS
        self.assertEqual(set(PROMPTS.keys()), {"math", "logic", "creative", "synthesis"})

    def test_prompt_counts(self):
        from eval.runner import PROMPTS
        for cat, prompts in PROMPTS.items():
            self.assertGreaterEqual(len(prompts), 20,
                                    f"category {cat} has only {len(prompts)} prompts")
            for p in prompts:
                self.assertIsInstance(p, str)
                self.assertGreater(len(p), 5, f"prompt too short: {p!r}")


class TestShannonEntropy(unittest.TestCase):
    """Sanity checks on the entropy helper."""

    def test_empty_returns_zero(self):
        from eval.runner import shannon_entropy
        self.assertEqual(shannon_entropy({}), 0.0)

    def test_uniform_returns_max(self):
        from eval.runner import shannon_entropy
        # 4 equally weighted zones → H = log2(4) = 2
        h = shannon_entropy({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})
        self.assertAlmostEqual(h, 2.0, places=5)

    def test_concentrated_returns_low(self):
        from eval.runner import shannon_entropy
        # One zone with all the weight → H = 0
        h = shannon_entropy({"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0})
        self.assertAlmostEqual(h, 0.0, places=5)

    def test_half_split(self):
        from eval.runner import shannon_entropy
        # 50/50 → H = 1
        h = shannon_entropy({"a": 1.0, "b": 1.0})
        self.assertAlmostEqual(h, 1.0, places=5)


class TestTokenDiversity(unittest.TestCase):
    """Token diversity must handle empty, all-same, and mixed inputs."""

    def test_empty(self):
        from eval.runner import token_diversity
        self.assertEqual(token_diversity(None), 0.0)

    def test_all_same(self):
        from eval.runner import token_diversity
        # 10 copies of the same token → diversity = 1/10 = 0.1
        import torch
        ids = torch.tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        self.assertAlmostEqual(token_diversity(ids), 0.1, places=5)

    def test_all_distinct(self):
        from eval.runner import token_diversity
        import torch
        ids = torch.tensor([1, 2, 3, 4, 5])
        # 5 distinct / 5 total = 1.0
        self.assertEqual(token_diversity(ids), 1.0)


class TestAnovaPurePython(unittest.TestCase):
    """The stdlib-only ANOVA implementation must match expected behavior."""

    def test_identical_groups_zero_eta(self):
        from eval.stats import one_way_anova
        # 3 groups, all with values [1, 2, 3] — between-group variance is 0
        res = one_way_anova({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        self.assertAlmostEqual(res["eta_squared"], 0.0, places=5)

    def test_different_groups_high_eta(self):
        from eval.stats import one_way_anova
        # Strongly separated groups — high η²
        res = one_way_anova({
            "a": [1, 1, 1, 1],
            "b": [5, 5, 5, 5],
            "c": [9, 9, 9, 9],
        })
        self.assertGreater(res["eta_squared"], 0.95)
        # F should be very large
        self.assertGreater(res["F"], 50)

    def test_moderate_difference(self):
        from eval.stats import one_way_anova
        # Modest between-group difference (typical SR-59 territory)
        res = one_way_anova({
            "math":      [1.20, 1.22, 1.18, 1.21],
            "logic":     [1.20, 1.21, 1.19, 1.22],
            "creative":  [1.10, 1.12, 1.09, 1.11],
            "synthesis": [1.05, 1.07, 1.04, 1.06],
        })
        # η² should be moderate-to-high (creative+synthesis are clearly different)
        self.assertGreater(res["eta_squared"], 0.5)
        # p should be small (effect is real)
        self.assertLess(res["p_approx"], 0.05)

    def test_too_few_groups_returns_zero(self):
        from eval.stats import one_way_anova
        res = one_way_anova({"a": [1, 2, 3]})
        self.assertEqual(res["eta_squared"], 0.0)
        self.assertEqual(res["k"], 1)


class TestRSquared(unittest.TestCase):
    """linear_r_squared must be 1.0 for perfect correlation, 0.0 for none."""

    def test_perfect_positive(self):
        from eval.stats import linear_r_squared
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.assertAlmostEqual(linear_r_squared(xs, ys), 1.0, places=5)

    def test_perfect_negative(self):
        from eval.stats import linear_r_squared
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 4.0, 3.0, 2.0, 1.0]
        self.assertAlmostEqual(linear_r_squared(xs, ys), 1.0, places=5)

    def test_no_correlation(self):
        from eval.stats import linear_r_squared
        # Random-ish, no relationship
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [3.0, 3.0, 3.0, 3.0, 3.0]
        # ys has zero variance → R² = 0 (our implementation guards against this)
        self.assertEqual(linear_r_squared(xs, ys), 0.0)


class TestAnalyzeOnMockData(unittest.TestCase):
    """End-to-end: synthesize an aggregate.json, run analyze(), check verdict."""

    def test_synthetic_high_eta_low_r2(self):
        """Mock the SR-59 'ANTI_P_ZOMBIE_CONFIRMED' scenario."""
        from eval.stats import analyze

        # Strong category separation, token diversity NOT correlated with entropy
        results = []
        for cat_idx, cat in enumerate(["math", "logic", "creative", "synthesis"]):
            base_entropy = [1.20, 1.85, 1.50, 1.10][cat_idx]
            for i in range(5):
                results.append({
                    "category": cat,
                    "prompt": f"{cat} prompt {i}",
                    "phi": 0.95 - 0.01 * cat_idx,
                    "zone_entropy": base_entropy + 0.02 * (i - 2),  # small noise
                    "token_diversity_input": 0.5 + 0.01 * i,         # UNcorrelated
                    "zone_weights": {"a": 0.5, "b": 0.5},
                    "kurtosis": 250.0,
                })

        with tempfile.TemporaryDirectory() as tmp:
            agg_path = os.path.join(tmp, "test_aggregate.json")
            with open(agg_path, "w") as f:
                json.dump({
                    "scale": "4B",
                    "model_id": "gemma3-4b-it",
                    "preset": "ACTIVE_MANIFOLD",
                    "results": results,
                }, f)
            summary, out_path = analyze(agg_path)
            self.assertEqual(summary["verdict"], "ANTI_P_ZOMBIE_CONFIRMED")
            self.assertGreater(summary["anova_zone_entropy"]["eta_squared"], 0.10)
            self.assertLess(summary["r2_token_diversity_to_zone_entropy"], 0.30)

    def test_synthetic_low_eta(self):
        """Mock the SR-59 'low η²' scenario — zones all collapse to similar entropy."""
        from eval.stats import analyze

        results = []
        for cat in ["math", "logic", "creative", "synthesis"]:
            for i in range(5):
                results.append({
                    "category": cat,
                    "prompt": f"{cat} prompt {i}",
                    "phi": 0.95,
                    "zone_entropy": 1.5 + 0.01 * i,  # Same for all categories
                    "token_diversity_input": 0.5,
                    "zone_weights": {},
                    "kurtosis": 200.0,
                })

        with tempfile.TemporaryDirectory() as tmp:
            agg_path = os.path.join(tmp, "test_aggregate.json")
            with open(agg_path, "w") as f:
                json.dump({
                    "scale": "4B", "model_id": "gemma3-4b-it",
                    "preset": "ACTIVE_MANIFOLD", "results": results,
                }, f)
            summary, _ = analyze(agg_path)
            # Low η² should trigger ETA2_BELOW_THRESHOLD verdict
            self.assertEqual(summary["verdict"], "ETA2_BELOW_THRESHOLD")
            self.assertLess(summary["anova_zone_entropy"]["eta_squared"], 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
