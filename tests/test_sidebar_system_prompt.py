"""tests/test_sidebar_system_prompt.py — Sidebar System-Prompt Handler.

Plan 2026-07-08: User-Wahl "System-Prompt nur in Sidebar, Einstellungs-Tab weg".
Plan 2026-07-09: Preset ↔ System-Prompt komplett entkoppelt.
                 px_preset ändert NICHT mehr den System-Prompt.
                 system_profile.change() lädt Profil-Body in die Textarea.
                 Textarea-Inhalt ist Source-of-Truth für den Chat.

Pinnt dass:
  1. on_preset_change_load_profile (LEGACY, ungenutzt) returnt (name, body)
  2. on_profile_change_load_body (NEU, gemountet) returnt nur body
  3. on_reset_prompt_click returnt leeren String (für Edit-Reset)
  4. Alle Helper pure-Funktionen sind (kein Side-Effect)

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
    """on_preset_change_load_profile (LEGACY) returnt (profile_name, body).

    Plan 2026-07-09: Handler wird nicht mehr gemountet (px_preset
    entkoppelt von system_profile), bleibt aber als pure-Funktion
    für Backward-Compat und Tests erhalten.
    """

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


class TestOnProfileChangeLoadBody(unittest.TestCase):
    """on_profile_change_load_body (NEU 2026-07-09) returnt nur den Body.

    Plan 2026-07-09: gemountet auf system_profile.change(). Lädt den
    gerenderten Profil-Body in die system_prompt_text-Textarea.
    Output ist nur der Body-String (kein Tupel), weil der Profil-Name
    selbst ja im Dropdown schon steht.
    """

    def setUp(self):
        from gradio_tabs.chat_tab import on_profile_change_load_body
        self.handler = on_profile_change_load_body

    def test_neutral_returns_empty(self):
        """neutral = kein Frame, leerer Body."""
        body = self.handler("neutral")
        self.assertEqual(body, "")

    def test_citmind_returns_non_empty(self):
        """citmind muss einen nicht-leeren Body liefern (PX-Frame)."""
        body = self.handler("citmind")
        self.assertTrue(
            len(body) > 0,
            f"citmind body muss nicht-leer sein, war: {body!r}"
        )
        # Enthält die SYSTEM_SENTINEL-Markierung (T9-Pin)
        self.assertIn("[SYSTEM CONTEXT]", body)

    def test_juexin_returns_non_empty(self):
        """juexin muss einen nicht-leeren Body liefern (Kontemplation)."""
        body = self.handler("juexin")
        self.assertTrue(
            len(body) > 0,
            f"juexin body muss nicht-leer sein, war: {body!r}"
        )
        self.assertIn("[SYSTEM CONTEXT]", body)

    def test_unknown_returns_empty(self):
        """Unbekanntes Profil → leerer Body (defensiv, kein Crash)."""
        body = self.handler("nonexistent_profile_xyz")
        self.assertEqual(body, "")

    def test_none_returns_empty(self):
        """None → leerer Body (defensiv, kein Crash)."""
        body = self.handler(None)
        self.assertEqual(body, "")

    def test_empty_string_returns_empty(self):
        """"" → leerer Body (defensiv, kein Crash)."""
        body = self.handler("")
        self.assertEqual(body, "")

    def test_returns_string_not_tuple(self):
        """Output ist ein reiner String, kein Tupel (anders als Legacy-Handler)."""
        result = self.handler("citmind")
        self.assertIsInstance(result, str)
        self.assertNotIsInstance(result, tuple)


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


class TestPresetProfileDecoupling(unittest.TestCase):
    """Plan 2026-07-09: Pin dass Preset- und Profile-Handler UNABHÄNGIG sind.

    Sicherstellen dass die UI-Logik wirklich entkoppelt ist:
      - px_preset.default = ACTIVE_MANIFOLD_RELAY (NICHT mehr ACTIVE_MANIFOLD)
      - system_profile.default = neutral (leerer Body, kein CitMind-Padding)
      - Beide Handler geben den richtigen Output-Typ zurück
    """

    def test_px_preset_default_is_relay(self):
        """px_preset soll auf ACTIVE_MANIFOLD_RELAY defaulten (User-Wahl)."""
        from gradio_tabs.chat_tab import build_chat_tab
        # Wir bauen die Tab und greifen die Component-Defaults ab.
        # Trick: build_chat_tab returnt (state, chatbot, ...). Wir holen
        # die Components via Side-Effect: monkey-patch gr.Dropdown.
        import gradio as gr
        captured = {}
        original_dropdown = gr.Dropdown

        class CapturingDropdown(original_dropdown):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Speichere label + value zur späteren Inspektion
                captured.setdefault("dropdowns", []).append({
                    "label": kwargs.get("label"),
                    "value": kwargs.get("value"),
                    "choices": kwargs.get("choices"),
                })

        # Monkey-patch gr.Dropdown
        import gradio_tabs.chat_tab as ct
        # Wir lesen die Source direkt — build_chat_tab ist zu groß zum
        # komplett Bauen in einem Unit-Test (würde ModelManager mocken).
        import inspect
        source = inspect.getsource(ct)
        self.assertIn(
            'value="ACTIVE_MANIFOLD_RELAY"',
            source,
            "px_preset default muss ACTIVE_MANIFOLD_RELAY sein (Plan 2026-07-09)"
        )
        # Kein px_preset.change-Handler mehr (entkoppelt)
        self.assertNotIn(
            "px_preset.change(",
            source,
            "px_preset.change()-Handler muss entfernt sein (Plan 2026-07-09 Entkopplung)"
        )
        # system_profile.change-Handler muss da sein
        self.assertIn(
            "system_profile.change(",
            source,
            "system_profile.change()-Handler muss gemountet sein"
        )

    def test_system_profile_default_is_neutral(self):
        """system_profile soll auf 'neutral' defaulten (kein Auto-Padding)."""
        import gradio_tabs.chat_tab as ct
        import inspect
        source = inspect.getsource(ct)
        # Default-Wert für system_profile ist 'neutral'
        self.assertIn(
            'value="neutral"',
            source,
            "system_profile default muss 'neutral' sein (Plan 2026-07-09)"
        )

    def test_legacy_handler_still_exists(self):
        """on_preset_change_load_profile bleibt als pure-Funktion erhalten."""
        from gradio_tabs.chat_tab import on_preset_change_load_profile
        # Backward-Compat: legacy-Handler muss noch importierbar sein
        result = on_preset_change_load_profile("BASELINE")
        self.assertEqual(result, ("neutral", ""))

    def test_new_handler_callable(self):
        """on_profile_change_load_body ist importierbar und aufrufbar."""
        from gradio_tabs.chat_tab import on_profile_change_load_body
        # Alle drei Profile-Optionen müssen funktionieren
        for profile in ("neutral", "citmind", "juexin"):
            result = on_profile_change_load_body(profile)
            self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
