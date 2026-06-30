"""test_generators_helper_invariants.py — pin generator-Helper gegen Drift.

Folgt der gleichen Methodik wie test_chat_tab_token_type_ids_strip.py:
TDD-Rot → Fix → Grün. Pinnt die strukturellen Annahmen der Helper, die
sonst bei kleinen Refactors stillschweigend brechen.

Hintergrund (Live-Crash-Logs, 2026-06-30):
  - generators.py:_inject_eot_eos ist nicht idempotent (Plan 7.3 hat das
    in Zeile 577 dokumentiert: "_inject_eot_eos setzt eos_token_id auf
    eine Liste — wir brauchen..."). Doppelter Aufruf → doppelte EOS-IDs.
  - generators.py:_is_small_model nutzt einen Magic-Threshold (30_000) der
    gemma3-1b (29952) knapp unter den "small"-Cut-off setzt. Bei einer
    MiniCPM-Variante mit ähnlichem Footprint → Drift.
  - generators.py:strip_unsupported_model_kwargs fällt ohne __code__
    (z.B. functools.partial forward) auf targeted-Liste zurück. Test
    pinnen, dass das auch im Fallback funktioniert.

Tests hier:
  T1-T3: _inject_eot_eos Idempotenz + Edge-Cases (None tokenizer,
        leerer eos_token_id, doppelter Aufruf).
  T4-T5: _is_small_model Threshold-Drift (genau 30000, 29999, 30001).
  T6-T7: strip_unsupported_model_kwargs mit functools.partial + C-Binding
        Fallbacks — der targeted-Strip MUSS immer funktionieren.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_generators_helper_invariants.py
"""
from __future__ import annotations
import os
import sys
import unittest
import functools

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── _inject_eot_eos tests ────────────────────────────────────────────────

class TestInjectEotEosIdempotency(unittest.TestCase):
    """T1: _inject_eot_eos muss idempotent sein.

    Hintergrund: chat_tab.py ruft _inject_eot_eos in der chat_fn-Pipeline
    auf, und in Test-Fixtures (z.B. test_chat_handlers) wird der Helper
    ggf. mehrfach aufgerufen. Doppelte EOS-IDs in eos_token_id können
    StopOnEOT-Criteria verwirren (z.B. wenn stop_ids dedupliziert werden
    und eine Reihenfolge-Annahme besteht).
    """

    def setUp(self):
        from generators import _inject_eot_eos
        self.helper = _inject_eot_eos

    def test_t1_double_call_no_duplicate_eos_ids(self):
        """Zwei aufeinanderfolgende Aufrufe produzieren dieselbe
        eos_token_id-Liste wie ein einzelner Aufruf."""
        class _FakeTokenizer:
            eos_token_id = 99
            unk_token_id = 0
            def convert_tokens_to_ids(self, t):
                return {"<end_of_turn>": 106, "<end_of_thought>": 107,
                        "<eos>": 1, "</s>": 2}.get(t)
        tok = _FakeTokenizer()
        base = {"eos_token_id": 99, "pad_token_id": None}
        once = self.helper(base, tok)
        twice = self.helper(once, tok)
        self.assertEqual(
            sorted(once["eos_token_id"]), sorted(twice["eos_token_id"]),
            f"eos_token_id list must be idempotent; once={once['eos_token_id']}, "
            f"twice={twice['eos_token_id']}"
        )
        # No duplicates within a single call either
        self.assertEqual(
            len(once["eos_token_id"]),
            len(set(once["eos_token_id"])),
            f"eos_token_id must be unique within a single call: "
            f"{once['eos_token_id']}"
        )

    def test_t2_none_tokenizer_raises_informative_error(self):
        """tokenizer=None darf nicht stillschweigend craschen — entweder
        klarer TypeError, oder graceful Return (was das auch immer die
        Policy ist, der Test pinnt das Verhalten)."""
        base = {"eos_token_id": 99, "pad_token_id": None}
        with self.assertRaises((TypeError, AttributeError)) as ctx:
            self.helper(base, None)
        # Fehlermeldung muss hilfreich sein, nicht kryptisch
        self.assertIn(
            "None", str(ctx.exception),
            f"Error message should mention None, got: {ctx.exception}"
        )

    def test_t3_empty_eos_token_id_initial(self):
        """Wenn eos_token_id initial None/leer ist, muss pad_token_id
        auf den ersten gefundenen EOS gesetzt werden (sonst bricht
        model.generate ohne pad_token_id ab)."""
        class _FakeTokenizer:
            eos_token_id = 1
            unk_token_id = 0
            def convert_tokens_to_ids(self, t):
                return {"<end_of_turn>": 106, "<eos>": 1, "</s>": 2}.get(t)
        tok = _FakeTokenizer()
        base = {"eos_token_id": None, "pad_token_id": None}
        result = self.helper(base, tok)
        self.assertIsNotNone(
            result.get("pad_token_id"),
            "pad_token_id must be set when eos_token_id is initially None"
        )
        self.assertIn(
            result["pad_token_id"], result["eos_token_id"],
            "pad_token_id must be one of the eos_token_ids"
        )


# ── _is_small_model tests ────────────────────────────────────────────────

class TestIsSmallModelThreshold(unittest.TestCase):
    """T4-T5: _is_small_model Threshold-Drift pinnen.

    Hintergrund: 30_000 als Magic-Threshold für "n_layers * hidden < 30_000"
    ist eine BRITTLE Heuristik. gemma3-1b hat 26*1152 = 29952 → "small".
    Eine MiniCPM-Variante mit 28*1152 = 32256 → nicht "small". Bei einer
    1.1B-Erweiterung auf 26*1280 = 33280 → nicht "small". Der Test
    dokumentiert diese harten Grenzen.
    """

    def setUp(self):
        from generators import _is_small_model
        self.is_small = _is_small_model

    def _make_model_with_layers(self, n_layers: int, hidden: int):
        """Baut ein Fake-transformers-Model mit layers + embed_tokens.

        Wichtig: muss ein echtes nn.Module sein, weil _is_small_model
        model.named_modules() iteriert (das ist eine nn.Module-Methode).
        """
        import torch.nn as nn
        class _Decoder(nn.Module):
            def __init__(self, n, h):
                super().__init__()
                # nn.ModuleList + Linear mit der richtigen Hidden-Size,
                # damit embed_tokens.embedding_dim stimmt.
                self.layers = nn.ModuleList([nn.Linear(h, h) for _ in range(n)])
                self.embed_tokens = nn.Embedding(100, h)
                # rotary_emb ist ein nn.Module-Attribut (irrelevant für
                # die Logik, aber hasattr muss True liefern)
                self.rotary_emb = nn.Linear(1, 1)
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Decoder(n_layers, hidden)
        return _Model()

    def test_t4_below_threshold_is_small(self):
        """n_layers * hidden = 29999 (knapp unter 30k) → small=True."""
        m = self._make_model_with_layers(26, 1152)  # 29952
        # 29952 < 30000 → small
        self.assertTrue(
            self.is_small(m),
            f"26*1152=29952 should be 'small' (<30000); got is_small=False"
        )

    def test_t5_at_threshold_is_not_small(self):
        """n_layers * hidden = 30001 (knapp über 30k) → small=False.
        gemma3-1b mit 26*1152=29952 ist small, aber 26*1154=30004 ist NICHT.
        Diese Grenze ist hart und sollte nicht versehentlich verschoben
        werden."""
        m = self._make_model_with_layers(26, 1154)  # 30004
        self.assertFalse(
            self.is_small(m),
            f"26*1154=30004 should NOT be 'small' (>=30000); got is_small=True"
        )


# ── strip_unsupported_model_kwargs fallback tests ────────────────────────

class TestStripUnsupportedFallbacks(unittest.TestCase):
    """T6-T7: strip_unsupported_model_kwargs mit Fallback-Pfaden.

    Hintergrund: der Helper nutzt `model.forward.__code__` zur Inspektion.
    Falls forward ein functools.partial ist (z.B. nach monkey-patching) oder
    ein C-Binding ohne __code__ → Fallback greift. Test pinnen, dass der
    targeted Strip (token_type_ids) IMMER funktioniert, auch im Fallback.
    """

    def setUp(self):
        from generators import strip_unsupported_model_kwargs
        self.strip = strip_unsupported_model_kwargs

    def test_t6_functools_partial_fallback_strips_token_type_ids(self):
        """Wenn model.forward ein functools.partial ist (kein __code__),
        MUSS der Fallback den targeted Strip trotzdem machen."""
        def real_forward(self, input_ids, attention_mask, position_ids=None,
                         past_key_values=None, labels=None, use_cache=None):
            pass
        # functools.partial hat kein __code__ auf der partial-Instanz
        partial = functools.partial(real_forward, self=None)
        class _Model:
            forward = partial
        m = _Model()
        gen_kwargs = {
            "input_ids": "fake", "attention_mask": "fake",
            "token_type_ids": "should_be_stripped",
            "streamer": "fake_streamer",
        }
        result = self.strip(m, gen_kwargs)
        self.assertNotIn(
            "token_type_ids", result,
            f"functools.partial fallback must still strip token_type_ids; "
            f"got: {result!r}"
        )
        # Andere kwargs bleiben drin
        self.assertIn("input_ids", result)
        self.assertIn("streamer", result)

    def test_t7_no_forward_attr_strips_token_type_ids(self):
        """Wenn model.forward fehlt (z.B. pures C++-Model), MUSS der
        targeted Strip trotzdem greifen — sonst lehnt model.generate()
        das kwarg ab und wir crashen."""
        class _ModelNoForward:
            """Model ohne forward-Attribut (z.B. abstract base)."""
            pass
        m = _ModelNoForward()
        gen_kwargs = {
            "input_ids": "fake", "attention_mask": "fake",
            "token_type_ids": "should_be_stripped",
        }
        result = self.strip(m, gen_kwargs)
        self.assertNotIn(
            "token_type_ids", result,
            f"Missing forward attr must fall back to targeted strip; "
            f"got: {result!r}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
