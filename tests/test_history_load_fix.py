"""
test_history_load_fix.py — Verification of History Persistence Fix
=================================================================
Ensures that loading an existing session and sending a new message
correctly appends to the history instead of overwriting it.
"""

import unittest
import os
import json
import asyncio
from unittest.mock import MagicMock, patch, PropertyMock
import sys

# ── Mocking complex dependencies before imports ──
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['telemetry'] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sessions
from model_manager import ModelManager

# Define a mock for get_model that works with run_until_complete
async def mock_get_model(*args, **kwargs):
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "templated"
    
    # Mock calling the tokenizer: tokenizer(text) -> inputs
    # inputs must have .to(device) and ["input_ids"]
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_inputs.__getitem__.side_effect = lambda k: MagicMock() if k == "input_ids" else MagicMock()
    
    # Fix for inputs["input_ids"].shape[1]
    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 10)
    mock_inputs.__getitem__.side_effect = lambda k: mock_input_ids if k == "input_ids" else MagicMock()
    
    tokenizer.return_value = mock_inputs
    tokenizer.encode.return_value = [1, 2, 3]
    return {"model": model, "tokenizer": tokenizer}

class TestHistoryLoadFix(unittest.TestCase):
    def setUp(self):
        if not os.path.exists("sessions"):
            os.makedirs("sessions")
        self.session_id = "test_load_fix_unique"
        self.initial_history = [{"role": "user", "content": "Initial Message"}]
        sessions.save_session(self.session_id, self.initial_history)
        
        self.manager = MagicMock(spec=ModelManager)
        self.manager.get_model = mock_get_model
        self.manager.get_px_metrics.return_value = {"phi": 0.9}

    @patch('gradio_tabs.chat_tab.TextIteratorStreamer')
    @patch('gradio_tabs.chat_tab.Thread')
    def test_history_persistence(self, mock_thread, mock_streamer_cls):
        # Setup streamer
        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = ["Response Chunk"]
        mock_streamer_cls.return_value = mock_streamer
        
        # Import chat_fn here to ensure mocks are applied
        from gradio_tabs.chat_tab import chat_fn
        
        # Execute chat_fn with empty history (simulating the bug condition)
        gen = chat_fn(
            message="New Question",
            history=None, # This was the trigger for the bug
            model_id="test-model",
            px_preset="ACTIVE_MANIFOLD",
            temp=0.7,
            tp=0.9,
            mt=128,
            rp=1.1,
            gamma=0.08,
            session_id=self.session_id,
            manager=self.manager
        )
        
        # Consume the generator to trigger the completion logic
        list(gen)
        
        # Verify result
        data = sessions.load_session(self.session_id)
        final_history = data["history"]
        
        # Expectation: Initial Message + New Question + Response
        self.assertEqual(len(final_history), 3, f"History should have 3 messages, got {len(final_history)}")
        self.assertEqual(final_history[0]["content"], "Initial Message")
        self.assertEqual(final_history[1]["content"], "New Question")
        self.assertEqual(final_history[2]["role"], "assistant")
        print("\n[Test Success] History correctly preserved and appended.")

    def tearDown(self):
        path = os.path.join("sessions", f"{self.session_id}.json")
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    unittest.main()
