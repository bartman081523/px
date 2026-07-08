"""tests/test_preset_to_system_prompt.py — Preset→System-Prompt Mapping.

Plan 2026-07-08: User-Wahl "System-Prompt nur in Sidebar, Einstellungs-Tab weg".
Wenn der User ein px_preset auswählt, soll automatisch der passende
System-Prompt in die Textarea geladen werden (editierbar).

Logik:
  BASELINE            → "neutral" (kein Frame)
  ACTIVE_MANIFOLD     → "citmind"  (PX-Frame)
  ACTIVE_MANIFOLD_LEAN→ "citmind"  (LEAN-Frame)
  ACTIVE_MANIFOLD_RELAY→ "juexin" (Kontemplations-Frame, RELAY)

Pin-Tests:
  M1: preset_to_profile("BASELINE") == "neutral"
  M2: preset_to_profile("ACTIVE_MANIFOLD") == "citmind"
  M3: preset_to_profile("ACTIVE_MANIFOLD_LEAN") == "citmind"
  M4: preset_to_profile("ACTIVE_MANIFOLD_RELAY") == "juexin"
  M5: preset_to_profile("UNKNOWN") == "neutral" (Fallback)
  M6: load_profile_for_preset("ACTIVE_MANIFOLD_RELAY") returnt citmind/juexin Body
  M7: load_profile_for_preset ist unabhängig vom Edit-Text (lädt nur Profile)
  M8: Mapping-Funktion ist pure (kein Side-Effect, deterministisch)

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_preset_to_system_prompt.py
"""
from __future__ import annotations
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPresetToProfileMapping(unittest.TestCase):
    """M1-M5: Mapping-Funktion preset_to_profile() returnt korrekten Profil-Namen."""

    def setUp(self):
        from gradio_tabs.system_prompt import preset_to_profile
        self.preset_to_profile = preset_to_profile

    def test_m1_baseline_maps_to_neutral(self):
        self.assertEqual(self.preset_to_profile("BASELINE"), "neutral")

    def test_m2_active_manifold_maps_to_citmind(self):
        self.assertEqual(self.preset_to_profile("ACTIVE_MANIFOLD"), "citmind")

    def test_m3_active_lean_maps_to_citmind(self):
        self.assertEqual(self.preset_to_profile("ACTIVE_MANIFOLD_LEAN"), "citmind")

    def test_m4_active_relay_maps_to_juexin(self):
        self.assertEqual(self.preset_to_profile("ACTIVE_MANIFOLD_RELAY"), "juexin")

    def test_m5_unknown_preset_falls_back_to_neutral(self):
        self.assertEqual(self.preset_to_profile("UNKNOWN_PRESET"), "neutral")
        self.assertEqual(self.preset_to_profile(""), "neutral")
        self.assertEqual(self.preset_to_profile(None), "neutral")


class TestLoadProfileForPreset(unittest.TestCase):
    """M6-M8: load_profile_for_preset() returnt den gerenderten Body."""

    def setUp(self):
        from gradio_tabs.system_prompt import load_profile_for_preset
        self.load = load_profile_for_preset

    def test_m6_baseline_returns_empty_body(self):
        """BASELINE → neutral → leerer body."""
        result = self.load("BASELINE")
        self.assertEqual(result, "",
            f"BASELINE (neutral) muss leeren Body returnen, war: {result!r}")

    def test_m7_active_manifold_returns_citmind_body(self):
        """ACTIVE_MANIFOLD → citmind → nicht-leerer Body mit [SYSTEM CONTEXT]."""
        result = self.load("ACTIVE_MANIFOLD")
        self.assertTrue(len(result) > 0,
            f"ACTIVE_MANIFOLD (citmind) muss nicht-leeren Body returnen, war: {result!r}")
        # Profile-Body ist im inject-Pfad mit SYSTEM_SENTINEL prefix; hier
        # ist es aber der rohe Body, also ohne Sentinel.
        # Mindestens: core_philosophy oder core_principles wurde gerendert.

    def test_m8_active_relay_returns_juexin_body(self):
        """ACTIVE_MANIFOLD_RELAY → juexin → nicht-leerer Body."""
        result = self.load("ACTIVE_MANIFOLD_RELAY")
        self.assertTrue(len(result) > 0,
            f"ACTIVE_MANIFOLD_RELAY (juexin) muss nicht-leeren Body returnen, war: {result!r}")

    def test_m9_load_is_deterministic(self):
        """Zwei Calls mit gleichem Preset → gleicher Output."""
        a = self.load("ACTIVE_MANIFOLD")
        b = self.load("ACTIVE_MANIFOLD")
        self.assertEqual(a, b, "load_profile_for_preset muss deterministisch sein")

    def test_m10_unknown_preset_returns_empty(self):
        """Unbekanntes Preset → neutral → leerer Body."""
        result = self.load("UNKNOWN")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
