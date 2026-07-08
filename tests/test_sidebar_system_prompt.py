"""tests/test_sidebar_system_prompt.py — Sidebar System-Prompt Handler.

Plan 2026-07-08: User-Wahl "System-Prompt nur in Sidebar, Einstellungs-Tab weg".

Pinnt dass:
  1. on_preset_change_load_profile returnt (profil_name, profil_body)
  2. on_reset_prompt_click returnt leeren String (für Edit-Reset)
  3. Beide Helper pure-Funktionen sind (kein Side-Effect)

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_sidebar_system_prompt.py
"""
from __future__ import annotations
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOnPresetChangeLoadProfile(unittest.TestCase):
    """on_preset_change_load_profile returnt (profile_name, body)."""

    def setUp(self):
        from gradio_tabs.chat_tab import on_preset_change_load_profile
        self.handler = on_preset_change_load_profile

    def test_returns_tuple_of_two(self):
        result = self.handler("ACTIVE_MANIFOLD")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_baseline_returns_neutral_empty(self):
        name, body = self.handler("BASELINE")
        self.assertEqual(name, "neutral")
        self.assertEqual(body, "")

    def test_active_manifold_returns_citmind(self):
        name, body = self.handler("ACTIVE_MANIFOLD")
        self.assertEqual(name, "citmind")
        self.assertTrue(len(body) > 0, f"citmind body muss nicht-leer sein, war: {body!r}")

    def test_active_lean_returns_citmind(self):
        name, body = self.handler("ACTIVE_MANIFOLD_LEAN")
        self.assertEqual(name, "citmind")
        self.assertTrue(len(body) > 0)

    def test_active_relay_returns_juexin(self):
        name, body = self.handler("ACTIVE_MANIFOLD_RELAY")
        self.assertEqual(name, "juexin")
        self.assertTrue(len(body) > 0, f"juexin body muss nicht-leer sein, war: {body!r}")

    def test_unknown_returns_neutral_empty(self):
        name, body = self.handler("UNKNOWN_PRESET")
        self.assertEqual(name, "neutral")
        self.assertEqual(body, "")


class TestOnResetPromptClick(unittest.TestCase):
    """on_reset_prompt_click returnt leeren String (Edit-Reset)."""

    def setUp(self):
        from gradio_tabs.chat_tab import on_reset_prompt_click
        self.handler = on_reset_prompt_click

    def test_returns_empty_string(self):
        result = self.handler()
        self.assertEqual(result, "")

    def test_is_pure(self):
        """Handler darf keinen Side-Effect haben (z.B. kein state-Update)."""
        # Mehrfach aufrufen muss immer das gleiche returnen.
        a = self.handler()
        b = self.handler()
        self.assertEqual(a, b)
        self.assertEqual(a, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
