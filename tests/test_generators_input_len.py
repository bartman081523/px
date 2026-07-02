"""test_generators_input_len.py — pin generators.py:input_len-Logik.

Hintergrund (Live-Crash auf master, 2026-06-30, local_debug.log):
    POST /v1/chat/completions (multimodal) →
      File "generators.py", line 456, in generate_chat_completion_stream
        gen_kwargs["_input_len"] = input_len
      NameError: name 'input_len' is not defined

Ursache: Pre-Plan-4-Code hatte `input_len`-Zugriff vor Definition
(Reihenfolge: try/multi-image → fallback → return without defining input_len
im except-Pfad). Log ist aus älterer Version; aktueller Code setzt
input_len=Z.437 (generate) und Z.582 (stream) — aber die Logik ist fragil:
input_len wird unbedingt als `inputs["input_ids"].shape[1]` angenommen.

Tests hier pinnen:
  T1: input_len wird in generate_text() vor _input_len-Setzung definiert
      (kein NameError bei gültigem input).
  T2: input_len ist korrekt = inputs["input_ids"].shape[1] (nicht eine
      andere Größe, nicht 0).
  T3: Strip-Helper strippt auch _px_input_len (PX-Internes, nie in forward).
  T4: Multimodal-Pfad setzt input_len ebenfalls (pixel_values vorhanden).

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_generators_input_len.py
"""
from __future__ import annotations
import os
import sys
import unittest

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStripPxInputLen(unittest.TestCase):
    """T1-T2: strip_unsupported_model_kwargs strippt _px_input_len."""

    def setUp(self):
        from generators import strip_unsupported_model_kwargs
        self.strip = strip_unsupported_model_kwargs

    def _make_minimal_model(self):
        """Minimale forward-Impl. ohne spezielle kwargs."""
        def forward(input_ids=None, attention_mask=None):
            return None
        class _M:
            pass
        m = _M()
        m.forward = forward
        return m

    def test_t1_strips_px_input_len(self):
        """_px_input_len ist PX-Internes → muss weg (Live-Crash-Trigger)."""
        gen_kwargs = {
            "input_ids": "fake",
            "attention_mask": "fake",
            "_px_input_len": 5533,
        }
        result = self.strip(self._make_minimal_model(), gen_kwargs)
        self.assertNotIn(
            "_px_input_len", result,
            f"_px_input_len must be stripped; got: {result!r}"
        )

    def test_t2_px_input_len_with_other_kwargs_preserves_legit(self):
        """Strip von _px_input_len darf input_ids/attention_mask etc.
        nicht antasten."""
        gen_kwargs = {
            "input_ids": "fake",
            "attention_mask": "fake",
            "streamer": "fake_streamer",
            "max_new_tokens": 256,
            "_px_input_len": 5533,
        }
        result = self.strip(self._make_minimal_model(), gen_kwargs)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("streamer", result)
        self.assertEqual(result["max_new_tokens"], 256)
        self.assertNotIn("_px_input_len", result)


class TestInputLenDefinition(unittest.TestCase):
    """T3-T4: input_len wird im generate()-Pfad konsistent definiert.

    Pinned durch Source-Inspektion (kein Live-Generate in Unit-Tests).
    Falls jemand `input_len` vor seiner Definition nutzt, fällt der Test.
    """

    def setUp(self):
        # inspect.getsource ist robuster als regex für Funktions-Body
        import inspect
        from generators import generate_chat_completion_stream, generate_chat_completion
        self.stream_src = inspect.getsource(generate_chat_completion_stream)
        self.generate_src = inspect.getsource(generate_chat_completion)

    def test_t3_input_len_defined_before_input_len_kwarg(self):
        """Im generate_chat_completion_stream-Pfad muss `input_len = ...`
        VOR `gen_kwargs["_input_len"] = input_len` definiert sein.
        Live-Crash-Logik: Bug war genau umgekehrt."""
        import re
        body = self.stream_src
        # input_len-Definition (vor _input_len-Zugriff)
        defn_match = re.search(
            r"input_len\s*=\s*inputs\[.input_ids.\]\.shape\[1\]", body
        )
        self.assertIsNotNone(
            defn_match,
            "input_len = inputs['input_ids'].shape[1] definition missing "
            "in generate_chat_completion_stream — would NameError on use."
        )
        # Verwendung als _input_len
        use_match = re.search(
            r'gen_kwargs\[["\']_?input_len["\']\]\s*=\s*input_len', body
        )
        self.assertIsNotNone(use_match, "input_len usage missing")
        # Definition muss VOR Verwendung sein
        self.assertLess(
            defn_match.start(), use_match.start(),
            f"input_len definition must come BEFORE its usage; got "
            f"defn@{defn_match.start()} use@{use_match.start()}"
        )

    def test_t4_generate_chat_completion_also_defines_input_len(self):
        """Auch generate_chat_completion() (non-streaming) muss input_len
        definieren, bevor es als _input_len in gen_kwargs landet."""
        import re
        body = self.generate_src
        defn_match = re.search(
            r"input_len\s*=\s*inputs\[.input_ids.\]\.shape\[1\]", body
        )
        self.assertIsNotNone(
            defn_match,
            "input_len = inputs['input_ids'].shape[1] missing in "
            "generate_chat_completion() — would NameError on use"
        )


def inspect_source(module):
    """Helper: returns module source."""
    import inspect
    return inspect.getsource(module)


if __name__ == "__main__":
    unittest.main(verbosity=2)
