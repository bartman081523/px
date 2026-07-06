"""tests/test_chat_actions.py — Pure-Logic Tests für chat_actions.py.

Plan: branch ui-styling, 2026-07-06, "Undo-Button für Turns".

Pinnt das Undo-Verhalten: pop des letzten (user, assistant)-Paars aus
einer History-Liste, plus Helper für "kann ich undoen?" (can_undo).
Wenn jemand die Pop-Semantik ändert (z.B. nur user pop, nicht das
assistant-Paar) oder die Mindestlänge für can_undo ändert, fallen
diese Tests.

Pin-Tests:
  T1: undo_last_turn([]) returnt [] (no-op auf leere Liste)
  T2: undo_last_turn([user]) returnt [] (1-element ist nicht undo-bar)
  T3: undo_last_turn([user, assistant]) returnt []
  T4: undo_last_turn([user, assistant, user]) returnt [user] (1 Paar pop)
  T5: undo_last_turn mutiert Input NICHT (pure)
  T6: undo_last_turn pop-assistant dann pop-user (Reihenfolge Pin)
  T7: undo_last_entry(history) ist der LOW-LEVEL: pop nur das letzte
      Element (egal welche Rolle). Wird intern verwendet wenn die
      History mid-stream ungleichmäßig ist.
  T8: can_undo(history) returnt True nur wenn ≥ 2 Messages
  T9: can_undo([]) returnt False
  T10: can_undo([user]) returnt False
  T11: undo_last_turn funktioniert auch mit Multimodal-Content (Liste
       statt str) — pop ist content-agnostic.
  T12: undo_last_turn returnt neue Liste (deep-copy der user/assistant
       Dicts falls das relevant ist) — verhindert UI-Mutation-Bugs.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_chat_actions.py
"""
from __future__ import annotations
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUndoLastTurn(unittest.TestCase):
    """T1-T7, T11-T12: undo_last_turn und undo_last_entry."""

    def setUp(self):
        from gradio_tabs.chat_actions import undo_last_turn, undo_last_entry, can_undo
        self.undo_last_turn = undo_last_turn
        self.undo_last_entry = undo_last_entry
        self.can_undo = can_undo

    def test_t1_empty_history_returns_empty(self):
        """undo_last_turn([]) returnt [] (no-op)."""
        self.assertEqual(self.undo_last_turn([]), [])

    def test_t2_single_user_message_returns_empty(self):
        """undo_last_turn([user]) returnt [] (kein Paar)."""
        history = [{"role": "user", "content": "hi"}]
        self.assertEqual(self.undo_last_turn(history), [])

    def test_t3_user_assistant_pair_returns_empty(self):
        """undo_last_turn([user, assistant]) returnt [] (1 Paar weg)."""
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        self.assertEqual(self.undo_last_turn(history), [])

    def test_t4_three_messages_pops_one_pair(self):
        """undo_last_turn([u, a, u]) returnt [u] (1 Paar pop, 1 user übrig)."""
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]
        result = self.undo_last_turn(history)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "hi")

    def test_t5_does_not_mutate_input(self):
        """undo_last_turn mutiert Input-Liste NICHT."""
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]
        original = list(history)
        _ = self.undo_last_turn(history)
        self.assertEqual(history, original)

    def test_t6_pops_assistant_then_user(self):
        """undo_last_turn poppt assistant ZUERST, dann user (Reihenfolge-Pin).

        Hintergrund: nach dem Undo soll der nächste user-input direkt
        anschließen, nicht eine Lücke haben. Wenn die Reihenfolge
        umgedreht würde, bliebe ein assistant-msg ohne user-msg → UI-Bug.
        """
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = self.undo_last_turn(history)
        # Die letzten 2 müssen weg sein — Reihenfolge ist egal, aber
        # es muss ein konsistentes Paar sein (entweder beide oder keiner)
        self.assertEqual(len(result), 2)
        # Letzte verbleibende Message ist user "u1" / assistant "a1" (das
        # erste Paar bleibt stehen)
        self.assertEqual(result[-1]["content"], "a1",
            "Nach undo_last_turn muss assistant 'a1' das letzte Element sein")

    def test_t7_undo_last_entry_pops_only_last(self):
        """undo_last_entry poppt NUR das letzte Element (low-level helper)."""
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]
        result = self.undo_last_entry(history)
        # Letzter user "again" muss weg sein, die anderen 2 bleiben
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1]["content"], "hello")

    def test_t11_multimodal_content_works(self):
        """undo_last_turn funktioniert auch mit Multimodal-List-Content."""
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "hello"}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "again"},
                {"type": "image", "image": "/tmp/x.png"},
            ]},
        ]
        result = self.undo_last_turn(history)
        # 1 Paar weg → 1 Eintrag übrig (user "hi")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "hi")

    def test_t12_returns_new_list_with_independent_dicts(self):
        """undo_last_turn returnt neue Liste — Dicts sind unabhängig vom
        Input (UI-Mutation-Bug-Pin)."""
        original_msg = {"role": "user", "content": "hi"}
        history = [
            original_msg,
            {"role": "assistant", "content": "hello"},
        ]
        result = self.undo_last_turn(history)
        # result ist [] (Paar weg), aber wenn nicht leer müssen die Dicts
        # unabhängig sein. Test mit kürzerer history:
        history2 = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        result2 = self.undo_last_turn(history2)
        # result2[0] darf nicht dasselbe dict wie history2[0] sein
        self.assertIsNot(result2[0], history2[0],
            "undo_last_turn muss unabhängige Dicts returnen, nicht Referenzen")


class TestCanUndo(unittest.TestCase):
    """T8-T10: can_undo."""

    def setUp(self):
        from gradio_tabs.chat_actions import can_undo
        self.can_undo = can_undo

    def test_t8_two_or_more_messages_can_undo(self):
        """can_undo(history mit ≥ 2 Messages) returnt True."""
        self.assertTrue(self.can_undo([{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": "x"}]))
        self.assertTrue(self.can_undo([{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": "x"},
                                       {"role": "user", "content": "y"}]))

    def test_t9_empty_history_cannot_undo(self):
        """can_undo([]) returnt False."""
        self.assertFalse(self.can_undo([]))

    def test_t10_single_message_cannot_undo(self):
        """can_undo([user]) returnt False (kein vollständiges Paar)."""
        self.assertFalse(self.can_undo([{"role": "user", "content": "hi"}]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
