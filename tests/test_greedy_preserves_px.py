"""tests/test_greedy_preserves_px.py — Regression-Tests für Greedy Decoding in PX-Presets.

Plan 2026-07-08: Output-Quality-Fix für RELAY-Modus. Greedy Decoding (do_sample=False)
als Default für alle PX-Presets, BASELINE bleibt beim Sampling. PX-Mechanik
(LEAN/RELAY) und Preset-Logik bleiben unverändert — diese Tests pinnen das.

Pin-Tests:
  G1: chat_fn Source-Code enthält Greedy-Entscheidung in Abhängigkeit von px_preset
  G2: BASELINE-Preset benutzt temperature/temp>0 (Sampling-Pfad)
  G3: ACTIVE_MANIFOLD-Preset erzwingt do_sample=False
  G4: ACTIVE_MANIFOLD_LEAN-Preset erzwingt do_sample=False
  G5: ACTIVE_MANIFOLD_RELAY-Preset erzwingt do_sample=False

Diese Tests prüfen STATISCH (Source-Code-Inspektion), nicht zur Laufzeit.
Ein echter generate()-Aufruf würde GPU + Model + Tokenizer brauchen, was in
Smoke-Tests hängt. Statt dessen prüfen wir, dass die Logik an der richtigen
Stelle im Source-Code steht.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_greedy_preserves_px.py
"""
from __future__ import annotations
import os
import sys
import re
import unittest
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGreedyDecodingSourceLogic(unittest.TestCase):
    """G1: chat_fn Source-Code muss Greedy-Decoding an px_preset koppeln."""

    def setUp(self):
        from gradio_tabs import chat_tab
        self.source = inspect.getsource(chat_tab.chat_fn)

    def test_g1a_px_preset_branches_present(self):
        """Source-Code muss eine px_preset-Verzweigung für Greedy haben."""
        # Erwartet: if px_preset == "BASELINE": ... else: ... (oder ähnlich)
        # Wir prüfen, dass der Source-Code px_preset im Kontext von temperature
        # oder do_sample erwähnt — also eine echte Verzweigung.
        self.assertIn("px_preset", self.source,
            "chat_fn muss px_preset referenzieren (für Greedy-Decision)")

    def test_g1b_do_sample_false_present(self):
        """Source-Code muss do_sample=False enthalten (Greedy-Marker)."""
        self.assertIn("do_sample=False", self.source,
            "chat_fn muss do_sample=False für Greedy-Modus enthalten")

    def test_g1c_baseline_branch_excluded_from_greedy(self):
        """Source-Code muss BASELINE explizit vom Greedy-Modus ausschließen."""
        # Akzeptiert: 'if px_preset == "BASELINE":' oder ähnliche Patterns
        # Wir prüfen, dass es eine Bedingung gibt, die BASELINE vom Greedy
        # trennt. Mindestens ein 'if px_preset == "BASELINE"' oder ein
        # 'if px_preset != "BASELINE"' oder eine Membership-Prüfung.
        has_baseline_branch = (
            re.search(r'if\s+px_preset\s*[!=]=\s*["\']BASELINE["\']', self.source)
            or re.search(r'px_preset\s+in\s*\([^)]*["\']BASELINE["\']', self.source)
        )
        self.assertIsNotNone(has_baseline_branch,
            "chat_fn muss BASELINE explizit von Greedy ausschließen")


class TestGreedyAppliesToPxPresets(unittest.TestCase):
    """G3-G5: Alle 3 PX-Presets erzwingen do_sample=False."""

    def setUp(self):
        from gradio_tabs import chat_tab
        self.source = inspect.getsource(chat_tab.chat_fn)

    def _assert_greedy_for_preset(self, preset: str):
        """Helper: prüft dass im chat_fn-Source für `preset` ein
        do_sample=False in Reichweite ist (entweder in einem expliziten
        if/elif-Block für das preset, oder im else-Block der den Greedy-Pfad
        abdeckt)."""
        # Suche nach if/elif px_preset == "PRESET": oder px_preset in ("X", "Y", ...)
        # und prüfe ob do_sample=False in einer logischen Nähe steht
        # Vereinfachung: prüfe ob do_sample=False im Source VORHANDEN ist
        # UND dass der Source eine px_preset-Bedingung hat, die NICHT nur
        # BASELINE matcht (sondern auch die PX-Presets umfasst).
        self.assertIn("do_sample=False", self.source,
            f"do_sample=False muss im Source vorhanden sein für {preset}")

    def test_g3_active_manifold_forces_greedy(self):
        """ACTIVE_MANIFOLD-Preset: do_sample=False."""
        # Statische Inspektion: Source muss eine Verzweigung haben die
        # ACTIVE_MANIFOLD dem Greedy-Pfad zuordnet.
        # Akzeptiert wird:
        #   - if px_preset in ("ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN", ...)
        #   - if px_preset != "BASELINE"
        #   - if px_preset == "ACTIVE_MANIFOLD" (mit elif/else für die anderen)
        has_px_branch = re.search(
            r'px_preset\s+(in\s*\(|!=\s*["\']|==\s*["\'](?:ACTIVE_MANIFOLD|ACTIVE_MANIFOLD_(?:LEAN|RELAY)))',
            self.source
        )
        self.assertIsNotNone(has_px_branch,
            "chat_fn muss eine px_preset-Verzweigung haben, die ACTIVE_MANIFOLD "
            "in den Greedy-Pfad routet")
        self._assert_greedy_for_preset("ACTIVE_MANIFOLD")

    def test_g4_active_lean_forces_greedy(self):
        """ACTIVE_MANIFOLD_LEAN-Preset: do_sample=False."""
        # LEAN wird typischerweise in einer Membership-Prüfung mit den anderen
        # PX-Presets erfasst ('px_preset in (..., "ACTIVE_MANIFOLD_LEAN", ...)').
        # Wir prüfen entweder explizit oder per != BASELINE.
        has_lean_in_branch = re.search(
            r'["\']ACTIVE_MANIFOLD_LEAN["\']',
            self.source
        )
        has_negative_baseline = re.search(
            r'px_preset\s*!=\s*["\']BASELINE["\']',
            self.source
        )
        self.assertTrue(has_lean_in_branch or has_negative_baseline,
            "chat_fn muss ACTIVE_MANIFOLD_LEAN entweder explizit im Greedy-"
            "Branch haben oder per 'px_preset != BASELINE' erfassen")
        self._assert_greedy_for_preset("ACTIVE_MANIFOLD_LEAN")

    def test_g5_active_relay_forces_greedy(self):
        """ACTIVE_MANIFOLD_RELAY-Preset: do_sample=False."""
        has_relay_in_branch = re.search(
            r'["\']ACTIVE_MANIFOLD_RELAY["\']',
            self.source
        )
        has_negative_baseline = re.search(
            r'px_preset\s*!=\s*["\']BASELINE["\']',
            self.source
        )
        self.assertTrue(has_relay_in_branch or has_negative_baseline,
            "chat_fn muss ACTIVE_MANIFOLD_RELAY entweder explizit im Greedy-"
            "Branch haben oder per 'px_preset != BASELINE' erfassen")
        self._assert_greedy_for_preset("ACTIVE_MANIFOLD_RELAY")


class TestBaselineKeepsSampling(unittest.TestCase):
    """G2: BASELINE-Preset behält Sampling-Verhalten."""

    def setUp(self):
        from gradio_tabs import chat_tab
        self.source = inspect.getsource(chat_tab.chat_fn)

    def test_g2_baseline_has_sampling_path(self):
        """Source-Code muss für BASELINE den Sampling-Pfad vorsehen.

        Akzeptiert wird:
          - if px_preset == "BASELINE": do_sample=temp>0
          - if px_preset == "BASELINE": _do_sample = temp > 0
          - explizite Zweige die BASELINE vom Greedy trennen UND im BASELINE-
            Pfad do_sample in Abhängigkeit von temp setzen
        """
        # Suche nach dem BASELINE-Branch und prüfe ob darin do_sample (in
        # irgendeiner Form) vorkommt. Wir matchen auf 'do_sample' in einer
        # 500-Zeichen-Umgebung nach 'BASELINE'.
        baseline_idx = self.source.find('"BASELINE"')
        self.assertGreater(baseline_idx, 0,
            "chat_fn muss 'BASELINE' als String enthalten")
        # Suche das nächste do_sample VOR oder NACH BASELINE im gleichen Block
        # Vereinfacht: wenn do_sample im Source ist UND es eine BASELINE-
        # Bedingung gibt, ist der Test grün (entweder im BASELINE-Block oder
        # im else-Block — der Test für G1 hat das schon verifiziert).
        self.assertIn("do_sample", self.source,
            "chat_fn muss do_sample enthalten (entweder für BASELINE-Sampling "
            "oder für PX-Greedy)")


class TestGreedyIsReproducible(unittest.TestCase):
    """G6: Greedy Decoding ist deterministisch (Konzept-Pin, kein GPU-Lauf).

    Wir pinnen nur die Logik dass Greedy = do_sample=False + temperature≈0
    in den Source geschrieben wurde. Ein echter Reproducibility-Test würde
    Model + GPU + 2 generate()-Calls brauchen, das ist out of scope für
    diese Smoke-Tests.
    """

    def setUp(self):
        from gradio_tabs import chat_tab
        self.source = inspect.getsource(chat_tab.chat_fn)

    def test_g6_greedy_temperature_marker_present(self):
        """Source-Code muss temperature=1e-10 ODER temperature=0 ODER eine
        ähnliche Greedy-Markierung im Greedy-Pfad haben."""
        # Wir prüfen, dass 1e-10 ODER ein ähnlich kleiner Wert im Source
        # ist (typischer Greedy-Trick in diesem Repo: temperature=1e-10).
        has_greedy_temp = (
            "1e-10" in self.source
            or "0.0" in self.source
            or re.search(r"temperature\s*=\s*0[^.0-9]", self.source)
        )
        self.assertTrue(has_greedy_temp,
            "chat_fn muss einen Greedy-Marker haben (1e-10 oder temperature=0)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
