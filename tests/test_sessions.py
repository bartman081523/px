"""tests/test_sessions.py — Tests für sessions.py Settings-Persistenz.

Plan: branch ui-styling, 2026-07-06, "Einstellungen-Tab + Persistenz".

Pinnt die additive Settings-Persistenz in sessions.py: SETTINGS_DEFAULTS,
update_settings mit atomarem write + per-session Lock. Wenn jemand die
Defaults ändert, die atomic-write Semantik bricht, oder die Lock-Mechanik
entfernt, fallen diese Tests.

Pin-Tests:
  T1: save_session ohne settings-Argument schreibt JSON ohne "settings" key
  T2: save_session mit settings schreibt JSON mit "settings" key
  T3: update_settings(sid, temperature=0.9) liest, mergt, schreibt atomar
  T4: update_settings für nicht-existierende session_id erstellt minimalen Eintrag
  T5: update_settings ist thread-safe (10 parallele Patches → finale Werte konsistent)
  T6: load_session returnt {} für settings wenn key fehlt (alte sessions)
  T7: update_settings mit garbage-patch (z.B. system_prompt_text=123) akzeptiert
      (kein type-coercion, Speichern als JSON-kompatibel)
  T8: Atomic-write-Semantik: bei Crash mid-write bleibt alte Datei erhalten

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_sessions.py
"""
from __future__ import annotations
import os
import sys
import json
import shutil
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSettingsDefaults(unittest.TestCase):
    """SETTINGS_DEFAULTS ist importierbar + hat 13 erwartete Keys."""

    def test_settings_defaults_exists(self):
        """SETTINGS_DEFAULTS Konstante ist in sessions.py definiert."""
        from sessions import SETTINGS_DEFAULTS
        self.assertIsInstance(SETTINGS_DEFAULTS, dict)
        # Mindestens die 13 Settings müssen da sein
        expected_keys = {
            "model_id", "px_preset", "auto_tune",
            "temperature", "top_p", "max_tokens", "rep_p", "px_gamma",
            "relay_sign", "relay_alpha", "relay_layer",
            "system_profile", "system_prompt_text",
        }
        self.assertEqual(set(SETTINGS_DEFAULTS.keys()), expected_keys,
            f"Erwartete 13 Keys, gefunden {len(SETTINGS_DEFAULTS)}")

    def test_settings_defaults_excludes_tts(self):
        """SETTINGS_DEFAULTS enthält KEINE tts_* Felder (Out-of-Scope dieses Plans)."""
        from sessions import SETTINGS_DEFAULTS
        for key in SETTINGS_DEFAULTS.keys():
            self.assertFalse(key.startswith("tts_"),
                f"Settings enthält tts_-Feld '{key}', sollte nicht migriert sein")

    def test_settings_defaults_have_reasonable_values(self):
        """SETTINGS_DEFAULTS Werte sind im erwarteten Bereich."""
        from sessions import SETTINGS_DEFAULTS
        d = SETTINGS_DEFAULTS
        self.assertEqual(d["model_id"], "gemma3-1b-it")
        self.assertEqual(d["px_preset"], "ACTIVE_MANIFOLD")
        self.assertIs(d["auto_tune"], True)
        self.assertAlmostEqual(d["temperature"], 0.7, places=2)
        self.assertAlmostEqual(d["top_p"], 0.95, places=2)
        self.assertEqual(d["max_tokens"], 1024)
        self.assertEqual(d["system_profile"], "neutral")
        self.assertEqual(d["system_prompt_text"], "")


class TestSaveLoadSettings(unittest.TestCase):
    """T1-T2: save_session mit/ohne settings."""

    def setUp(self):
        """Pro Test: isoliertes tmp-dir für SESSION_DIR."""
        from sessions import SESSION_DIR as REAL_SESSION_DIR
        self._tmpdir = tempfile.mkdtemp(prefix="sessions_test_")
        self._real_session_dir = REAL_SESSION_DIR
        # Patch SESSION_DIR in sessions module
        import sessions
        sessions.SESSION_DIR = self._tmpdir
        # Auch ensure_session_dir muss die neue Variable sehen
        self._sessions_mod = sessions

    def tearDown(self):
        import sessions
        sessions.SESSION_DIR = self._real_session_dir
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_t1_save_without_settings_omits_key(self):
        """save_session ohne settings schreibt JSON ohne 'settings' key."""
        from sessions import save_session, load_session
        sid = "abc123"
        save_session(sid, [{"role": "user", "content": "hi"}], model_id="gemma3-1b-it")
        data = load_session(sid)
        self.assertNotIn("settings", data,
            "Ohne settings-Argument darf 'settings' key nicht in JSON sein")

    def test_t2_save_with_settings_persists(self):
        """save_session mit settings schreibt JSON mit 'settings' key."""
        from sessions import save_session, load_session
        sid = "def456"
        settings = {"temperature": 0.9, "system_profile": "citmind"}
        save_session(sid, [], model_id="gemma3-1b-it", settings=settings)
        data = load_session(sid)
        self.assertIn("settings", data)
        self.assertEqual(data["settings"]["temperature"], 0.9)
        self.assertEqual(data["settings"]["system_profile"], "citmind")

    def test_t6_load_session_returns_empty_settings_if_missing(self):
        """load_session returnt {} für settings wenn key fehlt (alte sessions)."""
        from sessions import save_session, load_session
        # Schreibe ein altes Format-JSON ohne settings
        sid = "old_session"
        path = os.path.join(self._tmpdir, f"{sid}.json")
        with open(path, "w") as f:
            json.dump({
                "session_id": sid,
                "model_id": "gemma3-1b-it",
                "history": [{"role": "user", "content": "old"}],
            }, f)
        data = load_session(sid)
        # settings key fehlt → kompatibilität: returnt {} oder fehlend
        self.assertEqual(data.get("settings", {}), {},
            f"Alte sessions ohne settings key sollten {{}} liefern, war: {data.get('settings')}")


class TestUpdateSettings(unittest.TestCase):
    """T3-T5, T7-T8: update_settings Logik + atomic write + thread-safety."""

    def setUp(self):
        from sessions import SESSION_DIR as REAL_SESSION_DIR
        self._tmpdir = tempfile.mkdtemp(prefix="sessions_test_")
        self._real_session_dir = REAL_SESSION_DIR
        import sessions
        sessions.SESSION_DIR = self._tmpdir
        self._sessions_mod = sessions

    def tearDown(self):
        import sessions
        sessions.SESSION_DIR = self._real_session_dir
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_t3_update_settings_merges_into_existing(self):
        """update_settings(sid, temperature=0.9) liest, mergt, schreibt."""
        from sessions import save_session, update_settings, load_session
        sid = "t3_sid"
        save_session(sid, [], model_id="gemma3-1b-it",
                     settings={"temperature": 0.7, "system_profile": "neutral"})
        # Patch: nur temperature
        update_settings(sid, temperature=0.9)
        data = load_session(sid)
        # temperature wurde überschrieben, system_profile bleibt
        self.assertEqual(data["settings"]["temperature"], 0.9)
        self.assertEqual(data["settings"]["system_profile"], "neutral")

    def test_t4_update_settings_creates_minimal_entry_for_unknown_sid(self):
        """update_settings für nicht-existierende session_id erstellt minimalen Eintrag."""
        from sessions import update_settings, load_session, SESSION_DIR
        sid = "t4_new_sid"
        # Noch keine session.json vorhanden
        self.assertFalse(os.path.exists(os.path.join(self._tmpdir, f"{sid}.json")))
        update_settings(sid, temperature=0.5)
        # Jetzt muss die Datei da sein
        self.assertTrue(os.path.exists(os.path.join(self._tmpdir, f"{sid}.json")))
        data = load_session(sid)
        self.assertEqual(data["settings"]["temperature"], 0.5)
        # History ist leer
        self.assertEqual(data.get("history", []), [])

    def test_t5_update_settings_thread_safe(self):
        """update_settings ist thread-safe: 10 parallele Patches, finale Werte konsistent."""
        from sessions import update_settings, load_session
        sid = "t5_thread_sid"
        # 10 Threads setzen verschiedene temperature-Werte
        results = []
        errors = []

        def worker(temp_val, idx):
            try:
                update_settings(sid, **{f"thread_{idx}": temp_val})
                results.append((idx, temp_val))
            except Exception as e:
                errors.append((idx, str(e)))

        threads = [
            threading.Thread(target=worker, args=(float(i), i))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Thread-Fehler: {errors}")
        # Alle 10 thread_* keys müssen in der finalen Datei sein
        data = load_session(sid)
        for idx, _ in results:
            self.assertIn(f"thread_{idx}", data["settings"],
                f"thread_{idx} fehlt in finalen settings (Race-Condition?)")

    def test_t7_update_settings_accepts_garbage_patch(self):
        """update_settings mit garbage-patch (z.B. system_prompt_text=123) akzeptiert
        (kein type-coercion, Speichern als JSON-kompatibel)."""
        from sessions import update_settings, load_session
        sid = "t7_sid"
        # 123 ist int, aber system_prompt_text ist eigentlich str
        update_settings(sid, system_prompt_text=123)
        data = load_session(sid)
        # JSON-kompatibel: 123 wird als 123 persistiert (nicht gecrashed, nicht coerced)
        self.assertEqual(data["settings"]["system_prompt_text"], 123)

    def test_t8_atomic_write_preserves_old_file_on_crash(self):
        """Atomic-write-Semantik: bei Crash mid-write bleibt alte Datei erhalten.

        Wir simulieren einen Crash, indem wir os.replace mocken, sodass es
        eine Exception wirft. Die alte Datei muss danach noch lesbar sein.
        """
        from sessions import save_session, update_settings, load_session
        sid = "t8_atomic"
        # Initiale Schreibung
        save_session(sid, [{"role": "user", "content": "old-turn"}],
                     model_id="gemma3-1b-it",
                     settings={"temperature": 0.5})
        # Lese das Original (sollte erhalten bleiben bei Crash)
        original_data = load_session(sid)
        original_history = original_data["history"]

        # Patch os.replace sodass es crasht
        import sessions as sess_mod
        original_replace = sess_mod.os.replace
        def crashing_replace(src, dst):
            raise OSError("simulated crash mid-write")
        sess_mod.os.replace = crashing_replace

        try:
            with self.assertRaises(Exception):
                update_settings(sid, temperature=0.99)
        finally:
            sess_mod.os.replace = original_replace

        # Nach Crash: alte Datei muss noch lesbar sein mit ursprünglichem Inhalt
        recovered = load_session(sid)
        self.assertEqual(recovered["history"], original_history,
            "Nach Crash muss die alte history erhalten sein")
        self.assertEqual(recovered["settings"]["temperature"], 0.5,
            "Nach Crash muss das alte settings erhalten sein (nicht 0.99)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
