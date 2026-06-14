"""
test_ui_backend.py — Tests for Gradio UI Backend Logic
======================================================
Verifies session logic, export/import, and chat response formatting.
"""

import unittest
import json
import os
import sys
from unittest.mock import MagicMock, patch

# Mock torch before any imports that might trigger model loading
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import sessions
from gradio_tabs.chat_tab import handle_new_session, handle_load_saved, handle_export, handle_import

class TestUIBackend(unittest.TestCase):
    def setUp(self):
        if not os.path.exists("sessions"):
            os.makedirs("sessions")
        self.test_session_id = "test-session-123"
        self.test_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

    def test_session_save_load(self):
        sessions.save_session(self.test_session_id, self.test_history)
        data = sessions.load_session(self.test_session_id)
        self.assertEqual(data["history"], self.test_history)
        self.assertIn(self.test_session_id, sessions.list_sessions())

    def test_handle_new_session(self):
        sid, hist, list_s, disp = handle_new_session()
        self.assertEqual(hist, [])
        self.assertIsInstance(sid, str)

    def test_handle_export_import(self):
        # Test Export
        update = handle_export(self.test_session_id, self.test_history)
        export_path = update["value"]
        self.assertTrue(os.path.exists(export_path))
        
        # Test Import
        mock_file = MagicMock()
        mock_file.name = export_path
        new_id, history, _, _ = handle_import(mock_file)
        self.assertEqual(new_id, self.test_session_id)
        self.assertEqual(history, self.test_history)
        
        # Cleanup
        os.remove(export_path)

if __name__ == "__main__":
    unittest.main()
