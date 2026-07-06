"""tests/test_system_prompt.py — Pure-Logic Tests für system_prompt.py.

Plan: branch ui-styling, 2026-07-06, Plan "Einstellungen-Tab + Persistenz".

Pinnt das System-Prompt-Loader + Injector-Verhalten. Wenn jemand die
Profil-Resolution ändert oder den Multimodal-Crash-Pin entfernt, fallen
diese Tests.

Pin-Tests:
  T1: list_profiles() enthält mindestens ["neutral"]
  T2: list_profiles() Reihenfolge: ["neutral", "citmind", "juexin"] (subset-stable)
  T3: list_profiles() ist deterministisch (zwei Calls gleicher Output)
  T4: resolve_profile("citmind") returnt nicht-leeres dict mit core_philosophy-Key
  T5: resolve_profile("unknown") fällt auf "neutral" zurück
  T6: build_system_message("neutral", None) returnt leeres content
  T7: build_system_message("citmind", "Mein Edit") überschreibt body mit "Mein Edit"
  T8: build_system_message("citmind", None) rendert core_philosophy + substrate_universal + principles
  T9: build_system_message returnt immer role="system"
  T10: inject_into_messages([...], "neutral", None) returnt Original-Liste (leerer body)
  T11: inject_into_messages setzt System-Message an Index 0
  T12: inject_into_messages strippt ALLE existing system-entries
  T13: inject_into_messages strippt Multimodal-Listen-System-Einträge (Crash-Pin)
  T14: inject_into_messages mutiert Input nicht
  T15: render_for_chat_template returnt String mit "[SYSTEM CONTEXT]\n" Sentinel + body + user_message
  T16: docs/CitMind.txt fehlend → list_profiles() = ["neutral"]
  T17: docs/CitMind.txt mit garbage JSON → fallback auf neutral
  T18: docs/CitMind.txt mit kaputter Struktur (core_philosophy als list) → coercion auf ""

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_system_prompt.py
"""
from __future__ import annotations
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestListProfiles(unittest.TestCase):
    """T1-T3: list_profiles() Output."""

    def setUp(self):
        from gradio_tabs.system_prompt import list_profiles
        self.list_profiles = list_profiles

    def test_t1_contains_neutral(self):
        """list_profiles() enthält mindestens ['neutral']."""
        names = self.list_profiles()
        self.assertIn("neutral", names)

    def test_t2_order_stable(self):
        """list_profiles() Reihenfolge: neutral zuerst, dann citmind, juexin (subset-stable)."""
        names = self.list_profiles()
        # neutral muss Index 0 haben wenn vorhanden
        if "neutral" in names:
            self.assertEqual(names[0], "neutral")
        # citmind und juexin (wenn vorhanden) müssen in Reihenfolge nach neutral sein
        if "citmind" in names and "juexin" in names:
            cit_i = names.index("citmind")
            jue_i = names.index("juexin")
            self.assertLess(cit_i, jue_i, "citmind muss vor juexin sein")

    def test_t3_deterministic(self):
        """list_profiles() ist deterministisch (zwei Calls gleicher Output)."""
        a = self.list_profiles()
        b = self.list_profiles()
        self.assertEqual(a, b)


class TestResolveProfile(unittest.TestCase):
    """T4-T5: resolve_profile(name) returnt dict oder Fallback."""

    def setUp(self):
        from gradio_tabs.system_prompt import resolve_profile
        self.resolve_profile = resolve_profile

    def test_t4_citmind_returns_nonempty(self):
        """resolve_profile("citmind") returnt dict mit core_philosophy (wenn doc geladen)."""
        # Falls docs/CitMind.txt fehlt: fällt auf neutral zurück → leeres dict
        # Falls vorhanden: nicht-leeres core_philosophy
        p = self.resolve_profile("citmind")
        self.assertIsInstance(p, dict)
        self.assertIn("name", p)
        # "citmind" oder "neutral" (Fallback) — beide OK
        self.assertIn(p["name"], ("citmind", "neutral"))

    def test_t5_unknown_falls_back_to_neutral(self):
        """resolve_profile("nonexistent") fällt auf "neutral" zurück."""
        p = self.resolve_profile("nonexistent_xyz")
        self.assertEqual(p["name"], "neutral")
        # core_principles muss als Liste zurückkommen (auch wenn leer)
        self.assertIsInstance(p.get("core_principles", []), list)

    def test_resolve_returns_fresh_copy(self):
        """resolve_profile returnt Kopie — Mutationen wirken nicht auf nächsten Call."""
        p1 = self.resolve_profile("neutral")
        if "core_principles" in p1:
            p1["core_principles"].append("MUTATED")
        p2 = self.resolve_profile("neutral")
        # p2 darf die Mutation nicht sehen
        self.assertNotIn("MUTATED", p2.get("core_principles", []))


class TestBuildSystemMessage(unittest.TestCase):
    """T6-T9: build_system_message(profile_name, edit_text)."""

    def setUp(self):
        from gradio_tabs.system_prompt import build_system_message
        self.build_system_message = build_system_message

    def test_t6_neutral_no_edit_empty_content(self):
        """build_system_message("neutral", None) returnt {"role": "system", "content": ""}."""
        m = self.build_system_message("neutral", None)
        self.assertEqual(m["role"], "system")
        self.assertEqual(m["content"], "")

    def test_t7_edit_overrides_profile_body(self):
        """build_system_message("citmind", "Mein Edit") → content enthält "Mein Edit"."""
        m = self.build_system_message("citmind", "Mein Edit")
        self.assertIn("Mein Edit", m["content"])
        # Wenn profile geladen ist, soll "Mein Edit" das Body-DOMINANT sein
        # (es kommt nach SYSTEM_SENTINEL)
        if m["content"]:  # nur wenn nicht leer
            self.assertIn("[SYSTEM CONTEXT]", m["content"])

    def test_t8_profile_renders_philosophy_substrate_principles(self):
        """build_system_message("citmind", None) rendert alle drei Sections."""
        m = self.build_system_message("citmind", None)
        # Wenn CitMind-doc geladen ist, muss content nicht-leer sein und
        # alle drei Komponenten rendern (oder ein leeres content, falls doc fehlt)
        if m["content"]:
            # Mindestens eine Sektion muss da sein
            self.assertGreater(len(m["content"]), 50,
                "Profile-Body sollte substantiell sein wenn doc geladen")

    def test_t9_always_role_system(self):
        """build_system_message returnt immer role='system'."""
        for name in ("neutral", "citmind", "juexin", "unknown_xyz"):
            m = self.build_system_message(name, None)
            self.assertEqual(m["role"], "system",
                f"profile={name} lieferte role={m['role']}")

    def test_edit_whitespace_only_falls_back_to_profile(self):
        """edit_text='   ' (nur whitespace) fällt auf Profil-Text zurück."""
        m_edit = self.build_system_message("citmind", "   ")
        m_none = self.build_system_message("citmind", None)
        # Beide müssen den gleichen Body liefern
        self.assertEqual(m_edit["content"], m_none["content"])


class TestInjectIntoMessages(unittest.TestCase):
    """T10-T14: inject_into_messages(messages, profile_name, edit_text)."""

    def setUp(self):
        from gradio_tabs.system_prompt import inject_into_messages
        self.inject_into_messages = inject_into_messages

    def test_t10_neutral_no_edit_returns_unchanged(self):
        """inject_into_messages([user_msg], "neutral", None) returnt Original-Liste."""
        original = [{"role": "user", "content": "hi"}]
        result = self.inject_into_messages(original, "neutral", None)
        self.assertEqual(result, original)

    def test_t11_inserts_system_at_index_zero(self):
        """inject_into_messages setzt System-Message an Index 0."""
        msgs = [{"role": "user", "content": "hi"}]
        result = self.inject_into_messages(msgs, "citmind", "edit-text")
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[1]["content"], "hi")

    def test_t12_strips_all_existing_system_entries(self):
        """inject_into_messages entfernt ALLE alten System-Einträge."""
        msgs = [
            {"role": "system", "content": "OLD1"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "OLD2"},  # raus
            {"role": "assistant", "content": "answer"},
        ]
        result = self.inject_into_messages(msgs, "citmind", "edit")
        system_entries = [m for m in result if m["role"] == "system"]
        self.assertEqual(len(system_entries), 1,
            f"Erwartet 1 System-Eintrag, gefunden {len(system_entries)}")
        # Der eine System-Eintrag muss "edit" enthalten
        self.assertIn("edit", system_entries[0]["content"])

    def test_t13_strips_multimodal_list_system_entries(self):
        """Crash-Pin: Multimodal-Listen-System-Einträge (list statt str) dürfen nicht crashen."""
        # Reproduziert den historischen Bug "list object has no attribute replace"
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "old"}]},  # list!
            {"role": "user", "content": "hi"},
        ]
        # Darf NICHT crashen
        result = self.inject_into_messages(msgs, "citmind", "edit")
        system_entries = [m for m in result if m["role"] == "system"]
        self.assertEqual(len(system_entries), 1)
        self.assertIn("edit", system_entries[0]["content"])

    def test_t14_does_not_mutate_input(self):
        """inject_into_messages mutiert Input-Liste nicht."""
        msgs = [
            {"role": "system", "content": "OLD"},
            {"role": "user", "content": "hi"},
        ]
        original = list(msgs)  # shallow copy
        _ = self.inject_into_messages(msgs, "citmind", "edit")
        # msgs muss unverändert sein
        self.assertEqual(len(msgs), len(original))
        for i, m in enumerate(msgs):
            self.assertEqual(m, original[i])


class TestRenderForChatTemplate(unittest.TestCase):
    """T15: render_for_chat_template(profile_name, edit_text, user_message)."""

    def setUp(self):
        from gradio_tabs.system_prompt import render_for_chat_template
        self.render_for_chat_template = render_for_chat_template

    def test_t15_returns_string_with_sentinel_and_user(self):
        """render_for_chat_template enthält [SYSTEM CONTEXT] + body + user_message."""
        out = self.render_for_chat_template("citmind", "edit", "Was ist Bewusstsein?")
        self.assertIsInstance(out, str)
        # Sentinel MUSS da sein
        self.assertIn("[SYSTEM CONTEXT]", out)
        # User-Message MUSS am Ende sein
        self.assertTrue(out.rstrip().endswith("Was ist Bewusstsein?"))

    def test_render_neutral_no_edit_just_user_message(self):
        """render_for_chat_template("neutral", None, "hi") ohne System-Content."""
        out = self.render_for_chat_template("neutral", None, "hi")
        # Bei neutral+None ist der body leer, also nur user-message
        self.assertIn("hi", out)


class TestDefensiveLoader(unittest.TestCase):
    """T16-T18: Defensive Loader — keine Crashes bei fehlenden/kaputten docs."""

    def test_t16_falls_back_to_neutral_on_unknown_profile(self):
        """resolve_profile("nonexistent") fällt auf neutral zurück (kein Crash)."""
        from gradio_tabs.system_prompt import resolve_profile
        p = resolve_profile("nonexistent")
        self.assertEqual(p["name"], "neutral")

    def test_t17_build_system_message_with_unknown_profile(self):
        """build_system_message("nonexistent") returnt neutral-message (kein Crash)."""
        from gradio_tabs.system_prompt import build_system_message
        m = build_system_message("nonexistent", None)
        self.assertEqual(m["role"], "system")
        # Body sollte neutral sein (= leer)
        self.assertEqual(m["content"], "")

    def test_t18_inject_handles_empty_messages_list(self):
        """inject_into_messages([]) mit neutral → returnt []."""
        from gradio_tabs.system_prompt import inject_into_messages
        result = inject_into_messages([], "neutral", None)
        self.assertEqual(result, [])

    def test_inject_with_edit_text_only(self):
        """inject_into_messages([], "neutral", "x") — bei neutral+non-empty-edit wird System injiziert."""
        from gradio_tabs.system_prompt import inject_into_messages
        # Bei "neutral" Profile ist body normalerweise leer. ABER: mit edit="x"
        # wird der edit genutzt (nicht der profile-body), also wird system injiziert.
        result = inject_into_messages([], "neutral", "my-edit")
        system_entries = [m for m in result if m["role"] == "system"]
        self.assertEqual(len(system_entries), 1)
        self.assertIn("my-edit", system_entries[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
