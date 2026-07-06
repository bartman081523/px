"""tests/test_chat_tab_wiring.py — Tests für chat_tab.py system_prompt-Integration.

Plan: branch ui-styling, 2026-07-06, "Einstellungen-Tab + Persistenz".

Pinnt dass chat_fn die system_profile + system_prompt_text Parameter
akzeptiert und via inject_into_messages() anwendet. Wenn jemand den
Inject-Call vergisst oder die Parameter-Signatur ändert, fallen diese Tests.

Pin-Tests:
  T1: chat_fn Signatur hat system_profile und system_prompt_text als Parameter
  T2: chat_fn importiert inject_into_messages und ruft es auf (smoke via mock)
  T3: Bei system_prompt_text="" und system_profile="neutral" wird KEIN
      System-Eintrag injiziert

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_chat_tab_wiring.py
"""
from __future__ import annotations
import os
import sys
import inspect
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestChatFnSignature(unittest.TestCase):
    """T1: chat_fn hat system_profile + system_prompt_text als Parameter."""

    def test_t1_chat_fn_accepts_system_params(self):
        """chat_fn Signatur enthält system_profile und system_prompt_text."""
        from gradio_tabs.chat_tab import chat_fn
        sig = inspect.signature(chat_fn)
        params = list(sig.parameters.keys())
        self.assertIn("system_profile", params,
            f"chat_fn Signatur fehlt 'system_profile': {params}")
        self.assertIn("system_prompt_text", params,
            f"chat_fn Signatur fehlt 'system_prompt_text': {params}")


class TestChatFnUsesInjector(unittest.TestCase):
    """T2: chat_fn ruft inject_into_messages() auf."""

    def test_t2_chat_fn_calls_inject_into_messages(self):
        """chat_fn importiert und ruft inject_into_messages().

        Wir prüfen statisch (per Source-Code-Inspektion), dass der Call
        vorhanden ist. Ein Laufzeit-Test mit model.generate() ist zu
        teuer (echtes Model + Tokenizer + Thread) und hängt in Smoke-
        Tests, ohne Mehrwert für den Pin.
        """
        import inspect
        from gradio_tabs import chat_tab
        from gradio_tabs import system_prompt

        # 1. Source code muss inject_into_messages importieren + aufrufen
        source = inspect.getsource(chat_tab.chat_fn)
        self.assertIn("inject_into_messages", source,
            "chat_fn Source-Code muss 'inject_into_messages' enthalten")
        self.assertIn("system_profile", source,
            "chat_fn Source-Code muss 'system_profile' referenzieren")
        self.assertIn("system_prompt_text", source,
            "chat_fn Source-Code muss 'system_prompt_text' referenzieren")

        # 2. Die injizierte Funktion ist auch wirklich die aus system_prompt
        self.assertTrue(hasattr(system_prompt, "inject_into_messages"),
            "system_prompt Modul muss inject_into_messages exportieren")
        self.assertTrue(callable(system_prompt.inject_into_messages),
            "inject_into_messages muss callable sein")


class TestChatFnNoOpForNeutral(unittest.TestCase):
    """T3: Bei neutral+kein-edit wird KEIN System-Eintrag injiziert."""

    def test_t3_neutral_no_edit_no_system_inject(self):
        """Bei system_profile='neutral' und system_prompt_text='' bleibt
        die History unverändert (kein leerer System-Eintrag)."""
        from gradio_tabs.system_prompt import inject_into_messages
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = inject_into_messages(history, "neutral", "")
        # T10-Pin: Original-Liste returnt
        self.assertEqual(result, history,
            f"Bei neutral+kein-edit muss Original zurückkommen, war: {result}")
        # Sicherheits-Check: kein System-Eintrag
        system_entries = [m for m in result if m.get("role") == "system"]
        self.assertEqual(len(system_entries), 0,
            f"Neutral+kein-edit darf KEINEN System-Eintrag erzeugen, war: {system_entries}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
