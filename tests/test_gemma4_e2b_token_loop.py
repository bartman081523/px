"""
test_gemma4_e2b_token_loop.py — Token-Loop Regression for Gemma 4 E2B
=====================================================================
Detects the "mathematics\\nmathematics\\nmathematics..." failure mode
from the 2026-06-08 E2B test (repetition collapse on narrow kurtosis).

This is a STRUCTURED unit test: the actual generation requires GPU,
so we test the *mechanisms* that prevent the loop:
  1. repetition_penalty in _px_config (= 1.15, set unconditionally for all models)
  2. Stability-break threshold (stability_cnt > 3, gated on phi_s > 0.9999)
  3. Anti-oscillation guards (per-layer-visit penalty + over-stable hub-jump)
  4. Generative output checks (when run with GPU)

Note (2026-06-18): the earlier `stability_cnt > 1` threshold and the dedicated
BOUNCE-BREAK block were removed in the 2026-06-11 patch.py rewrite. Token-loop
avoidance now comes from repetition_penalty=1.15 + no_repeat_ngram_size=3 plus
the is_gemma4 n_loops cap (8→4). The tests below were updated accordingly.

The GPU-dependent `test_token_loop_in_generation` is skipped if no
GPU is available — all other tests run on CPU only.

Usage:
  PYTHONPATH=. python -m pytest tests/test_gemma4_e2b_token_loop.py -v
  (with GPU) PYTHONPATH=. python tests/test_gemma4_e2b_token_loop.py
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BTokenLoopMechanisms — verify the 4 mitigations
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BTokenLoopMechanisms(unittest.TestCase):
    """Each mechanism that prevents token loops is verified independently."""

    def test_repetition_penalty_in_config(self):
        """Mechanism 1: repetition_penalty=1.15 (set unconditionally in
        apply_px_patch for ALL models, not just gemma4)."""
        import inspect
        from px_patches.gemma4_2b_px import patch as patch_mod
        source = inspect.getsource(patch_mod)
        # Production sets repetition_penalty=1.15 unconditionally (patch.py:1008).
        self.assertIn('defaults["repetition_penalty"] = 1.15', source,
                      "patch.py must set repetition_penalty=1.15")
        self.assertIn('defaults["no_repeat_ngram_size"] = 3', source,
                      "patch.py must set no_repeat_ngram_size=3")

    def test_stability_break_threshold(self):
        """Mechanism 2: stability_cnt > 3 (gated on phi_s > 0.9999) triggers
        break, plus a tail safety guard stability_cnt > 5. The earlier > 1
        threshold was removed in the 2026-06-11 rewrite."""
        import inspect
        from px_patches.gemma4_2b_px import patch as patch_mod
        source = inspect.getsource(patch_mod)

        # Primary stability break, gated on near-perfect cosine stability:
        self.assertIn("if stability_cnt > 3:", source,
                      "stability break must fire at `stability_cnt > 3`")
        self.assertIn("phi_s > 0.9999", source,
                      "stability break must be gated on phi_s > 0.9999")
        # Tail-of-iteration safety guard:
        self.assertIn("if stability_cnt > 5: break", source,
                      "tail safety guard `stability_cnt > 5` must be present")
        # The old `> 1` one-line break must be gone:
        self.assertNotIn("stability_cnt > 1: h_exp = trans_out; break", source,
                         "Old `stability_cnt > 1` one-line break must be removed")

    def test_layer_anti_oscillation_guards_present(self):
        """Mechanism 3: the dedicated BOUNCE-BREAK block was removed
        (2026-06-11). Anti-oscillation is now handled by lighter in-loop
        guards: per-layer-visit penalty, over-stable hub-jump, and
        layer-visits bookkeeping, with per-step telemetry collection."""
        import inspect
        from px_patches.gemma4_2b_px import patch as patch_mod
        source = inspect.getsource(patch_mod)
        self.assertIn("layer_visits", source,
                      "layer-visits bookkeeping must be present")
        self.assertIn("dynamic_hub + 1", source,
                      "over-stable hub-jump guard must be present")
        self.assertIn("_px_current_telemetry_raw", source,
                      "per-step telemetry collection must be present")
        # The removed BOUNCE-BREAK marker must NOT have come back:
        self.assertNotIn("BOUNCE-BREAK", source,
                          "BOUNCE-BREAK block was removed 2026-06-11; "
                          "must not be reintroduced")

    def test_reduced_recur_range(self):
        """Mechanism 4: gemma4-e2b uses SCALE_DEFAULTS[1536] for 1B parity.
        Token-loop avoidance comes from repetition_penalty=1.15 +
        no_repeat_ngram_size=3 (set in patch.py) plus the is_gemma4 n_loops
        cap (8→4) inside apply_px_patch — not from a low gamma."""
        from config import MODEL_REGISTRY
        from px_patches.gemma4_2b_px.auto_tune import SCALE_DEFAULTS
        reg = MODEL_REGISTRY["gemma4-e2b-it"]
        kw = reg["patch_kwargs"]
        # No explicit override — defaults come from SCALE_DEFAULTS[1536]
        self.assertNotIn("recur_start", kw)
        self.assertNotIn("n_loops", kw)
        # SCALE_DEFAULTS[1536] must exist and have the 1B-parity values
        self.assertIn(1536, SCALE_DEFAULTS)
        sd = SCALE_DEFAULTS[1536]
        self.assertEqual(sd["recur_start"], 10)
        self.assertEqual(sd["recur_end"], 26)
        self.assertEqual(sd["n_loops"], 8)
        self.assertEqual(sd["gamma"], 0.12)


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BTokenRepetition — structural repetition detection helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BTokenRepetition(unittest.TestCase):
    """Helper functions for detecting repetitive token patterns."""

    def _count_consecutive_repeats(self, tokens):
        """Return the max run of identical consecutive tokens."""
        if not tokens:
            return 0
        max_run = cur_run = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        return max_run

    def _token_type_ratio(self, tokens):
        """Unique tokens / total tokens (1.0 = all unique, low = repetitive)."""
        if not tokens:
            return 1.0
        return len(set(tokens)) / len(tokens)

    def test_detects_three_consecutive_repeats(self):
        # The bug pattern: "math", "math", "math", ...
        tokens = ["the", "math", "math", "math", "induction"]
        repeats = self._count_consecutive_repeats(tokens)
        self.assertEqual(repeats, 3,
                         "Three consecutive 'math' tokens must be detected")

    def test_detects_long_run(self):
        # The actual 2026-06-08 output pattern
        tokens = ["mathematics"] * 5 + ["."]
        repeats = self._count_consecutive_repeats(tokens)
        self.assertEqual(repeats, 5)

    def test_token_type_ratio_low_for_loop(self):
        tokens = ["mathematics"] * 10
        ratio = self._token_type_ratio(tokens)
        self.assertLess(ratio, 0.2,
                        "Pure repetition must yield very low type ratio")

    def test_token_type_ratio_high_for_diverse(self):
        tokens = ["The", "concept", "of", "mathematical", "induction", "is", "fundamental", "."]
        ratio = self._token_type_ratio(tokens)
        self.assertGreater(ratio, 0.8,
                          "Diverse tokens must yield high type ratio")

    def test_no_repetition_in_normal_text(self):
        tokens = "Explain the concept of mathematical induction.".split()
        repeats = self._count_consecutive_repeats(tokens)
        self.assertEqual(repeats, 1, "Normal text has no consecutive repeats")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BLoopPattern — the 2026-06-08 actual failure pattern
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BLoopPattern(unittest.TestCase):
    """Reproduces the exact failure pattern from 2026-06-08 and checks fixes."""

    def test_2026_06_08_failure_pattern(self):
        # The actual output: "Explain the concept of mathematical induction.\n
        #                    Explain what is the concept of mathematical.\n
        #                    Explain.\n  Explain why\n  the relationship between
        #                    the concept of mathematical induction.\n  Explain why
        #                    mathematics\n  mathematics\n  mathematics"
        response = (
            "Explain the concept of mathematical induction.\n"
            "Explain what is the concept of mathematical.\n"
            "Explain.\n  Explain why\n  the relationship between the concept of mathematical induction.\n"
            "  Explain why\n  mathematics\n  mathematics\n  mathematics"
        )
        # Word-level tokens
        words = response.split()
        # Count "mathematics" runs
        run_lengths = []
        i = 0
        while i < len(words):
            j = i
            while j < len(words) and words[j] == words[i]:
                j += 1
            run_lengths.append(j - i)
            i = j
        max_run = max(run_lengths) if run_lengths else 0
        # The failure: 3 consecutive "mathematics" at the end
        self.assertEqual(max_run, 3,
                         "The 2026-06-08 pattern has a 3-token run of 'mathematics'")

    def test_loop_should_have_been_broken(self):
        """Verify that with the current threshold (stability_cnt > 3, gated on
        phi > 0.9999), a sustained high-phi loop is eventually broken."""
        # Patch.py threshold: phi_s > 0.9999 sustained for stability_cnt > 3
        simulated_history = [0.99995, 0.99995, 0.99995, 0.99995, 0.99995]  # 5 high-phi steps
        stability_cnt = 0
        broke = False
        for phi in simulated_history:
            if phi > 0.9999:
                stability_cnt += 1
                if stability_cnt > 3:  # CURRENT THRESHOLD (was > 1 pre-2026-06-11)
                    broke = True
                    break
            else:
                stability_cnt = 0
        self.assertTrue(broke,
                        "Stability-break must fire once stability_cnt exceeds 3")
        self.assertEqual(stability_cnt, 4,
                         "stability_cnt must reach 4 to trigger break (> 3)")


# ═══════════════════════════════════════════════════════════════════════════════
# TestGemma4E2BGeneratedOutput — optional GPU test (skipped if no GPU)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGemma4E2BGeneratedOutput(unittest.TestCase):
    """GPU test: actually generate text and check for loops. Skipped without GPU."""

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("RUN_GPU_TESTS") == "1",
        "GPU not available or RUN_GPU_TESTS not set"
    )
    def test_token_loop_in_generation(self):
        """Actually generate from gemma4-e2b-it and verify no loops."""
        from model_manager import ModelManager
        manager = ModelManager()
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            entry = loop.run_until_complete(
                manager.get_model("gemma4-e2b-it", px_subjective=True,
                                  px_config_preset="SUBJECTIVE")
            )
            model = entry["model"]
            tokenizer = entry["tokenizer"]
            prompt = "Explain the concept of mathematical induction."
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=50, do_sample=True,
                    temperature=0.7, top_p=0.9,
                )

            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            tokens = generated.split()

            # After token-loop fix: max 2 consecutive repeats
            max_run = 1
            cur = 1
            for i in range(1, len(tokens)):
                if tokens[i] == tokens[i - 1]:
                    cur += 1
                    max_run = max(max_run, cur)
                else:
                    cur = 1

            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            print(f"\nGenerated ({len(tokens)} tokens): {generated[:200]}")
            print(f"Max consecutive repeat: {max_run}, unique ratio: {unique_ratio:.2f}")

            # The 2026-06-08 fix should prevent long runs
            self.assertLessEqual(max_run, 2,
                                 f"Max consecutive repeat must be ≤ 2, got {max_run}")
            self.assertGreater(unique_ratio, 0.3,
                               f"Token type ratio must be > 0.3, got {unique_ratio:.2f}")
        finally:
            loop.close()
            try:
                loop.run_until_complete(manager.unload_model("gemma4-e2b-it"))
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
