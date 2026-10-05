"""tests/test_greedy_preserves_px.py — Regression-Tests für PX-Sampling-Contract.

RTPF-A4 (Plan 2026-10-05): Der Temperature-Slider wird in ALLEN Presets
respektiert (SSE-Parität mit streaming_bridge). temp>0 → Sampling
(do_sample=True, temperatura=Slider); temp=0 → Greedy-Fallback
(1e-10, do_sample=False) — die deterministische Mode bleibt über den
Slider erreichbar. Kein px_preset-Branch mehr in der Sampling-Entscheidung.

Begründung: TT0-Attribution (scratches/rtpf/tt0_result.json, seed 42,
T=5326) — weder Greedy 1e-10 noch temp 0.7 reproduzieren den
f31eff3e-Loop im Replay; die alte Greedy-Forcing-Branch
(Plan 2026-07-08) versprach Reproduzierbarkeit, die sie nicht liefert,
und neutralisiert den Slider stillschweigend. PX-Mechanik (patch.py,
relay_inject) bleibt unangetastet — nur die Sampling-Entscheidung
wird entkoppelt.

Pin-Tests:
  G1: chat_fn hat EINE Sampling-Entscheidung (`_do_sample = temp > 0`)
      OHNE px_preset-Branch in naher Umgebung
  G2: Greedy-Fallback über `temp if temp > 0 else 1e-10` (Slider 0)
  G3: alter px_preset-Verzweigungsblock ist entfernt
  G4: RTPF-A4-Begründung (TT0/Streamer-Skip-Kaskade) im Source dokumentiert

Diese Tests prüfen STATISCH (Source-Code-Inspektion), nicht zur Laufzeit.
Ein echter generate()-Aufruf würde GPU + Model + Tokenizer brauchen, was in
Smoke-Tests hängt.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_greedy_preserves_px.py
"""
from __future__ import annotations
import os
import sys
import inspect
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WINDOW = 400   # Zeichen-Umgebung um die Sampling-Entscheidung


class TestSamplingIsTemperatureDriven(unittest.TestCase):
    """G1: Die Sampling-Entscheidung hängt NUR am Slider (temp>0)."""

    def setUp(self):
        from gradio_tabs import chat_tab
        self.source = inspect.getsource(chat_tab.chat_fn)

    def test_g1a_slider_respect_marker_present(self):
        """Source-Code muss `_do_sample = temp > 0` enthalten (Marker)."""
        self.assertIn("_do_sample = temp > 0", self.source,
            "chat_fn muss die Sampling-Entscheidung am Slider koppeln "
            "(`_do_sample = temp > 0`)")

    def test_g1b_no_px_preset_branch_at_decision(self):
        """Kein px_preset-Branch direkt an der Sampling-Entscheidung."""
        idx = self.source.find("_do_sample = temp > 0")
        self.assertGreater(idx, 0, "Sampling-Marker fehlt")
        window = self.source[max(0, idx - WINDOW):idx]
        self.assertNotIn("px_preset", window,
            "die Sampling-Entscheidung darf nicht von px_preset "
            "abhängen (RTPF-A4: Slider in allen Presets respektiert)")

    def test_g1c_streaming_bridge_parity(self):
        """SSE-Parität: streaming_bridge samplet mit Request-Temperatur."""
        import streaming_bridge   # noqa: F401  (path already on sys.path)
        src = inspect.getsource(streaming_bridge)
        self.assertGreater(src.find("temperature"), 0,
            "streaming_bridge muss temperature verwenden (SSE-Parität)")


class TestGreedyFallback(unittest.TestCase):
    """G2: Slider 0 = Greedy (1e-10 + do_sample=False im Short-Pfad)."""

    def setUp(self):
        from gradio_tabs import chat_tab
        self.source = inspect.getsource(chat_tab.chat_fn)

    def test_g2_greedy_fallback_expression(self):
        """`temp if temp > 0 else 1e-10` — deterministische Mode bleibt."""
        self.assertIn("temp if temp > 0 else 1e-10", self.source,
            "chat_fn muss den Greedy-Fallback (temp=0 → 1e-10) behalten")

    def test_g2b_greedy_marker_present(self):
        """1e-10 (Greedy-Trick dieses Repos) muss im Source stehen."""
        self.assertIn("1e-10", self.source,
            "chat_fn muss einen Greedy-Marker haben (1e-10)")


class TestOldForcingBranchRemoved(unittest.TestCase):
    """G3: Der alte px_preset-Greedy-Zweig ist entfernt."""

    def setUp(self):
        from gradio_tabs import chat_tab
        self.source = inspect.getsource(chat_tab.chat_fn)

    def test_g3_no_baseline_exclusive_greedy_branch(self):
        """`if px_preset == "BASELINE":` darf die Sampling-Entscheidung
        NICHT mehr umschließen (alte Branch-Muster)."""
        old_block = 'if px_preset == "BASELINE":\n        _temperature'
        self.assertNotIn(old_block, self.source,
            "alter Greedy-Forcing-Block (Plan 2026-07-08) ist zu entfernen — "
            "ersetzt durch Slider-Respekt (RTPF-A4, Plan 2026-10-05)")


class TestRemedyRationaleDocumented(unittest.TestCase):
    """G4: RTPF-A4-Begründung im Source (TT0 + Streamer-Skip-Kaskade)."""

    def setUp(self):
        from gradio_tabs import chat_tab
        self.source = inspect.getsource(chat_tab.chat_fn)

    def test_g4_rtpf_a4_comment_present(self):
        """Source muss die RTPF-A4-Begründung nennen (TT0-Evidenz)."""
        self.assertIn("RTPF-A4", self.source,
            "chat_fn muss den RTPF-A4-Remedy-Verweis dokumentieren")
        self.assertIn("tt0_result.json", self.source,
            "chat_fn muss die TT0-Evidenzquelle nennen")


if __name__ == "__main__":
    unittest.main(verbosity=2)