"""
test_model_manager.py — Verify ModelManager Re-Patching Logic
=============================================================
Ensures that switching subjective modes doesn't reload weights.
"""

import unittest
import torch
import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import ModelManager

class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager()
        # Mock model registry
        self.registry_mock = {
            "test-model": {
                "hf_id": "mock-hf",
                "tokenizer_id": "mock-tok",
                "patch_dir": "test_patch",
                "patch_kwargs": {"recur_start": 5},
                "subjective_kwargs": {"subjective_enabled": True},
                "model_type": "llama",
                "dtype": "float32"
            }
        }

    @patch('model_manager.AutoModelForCausalLM.from_pretrained')
    @patch('model_manager.AutoTokenizer.from_pretrained')
    @patch('model_manager.importlib.import_module')
    def test_repatch_no_reload(self, mock_import, mock_tok_from, mock_model_from):
        # 1. First load
        mock_model = MagicMock()
        mock_model_from.return_value = mock_model
        
        with patch('model_manager.MODEL_REGISTRY', self.registry_mock):
            # Load Peak mode
            loop = asyncio.new_event_loop()
            entry1 = loop.run_until_complete(self.manager.get_model("test-model", px_subjective=False))
            self.assertEqual(mock_model_from.call_count, 1)
            
            # 2. Re-patch to Subjective mode
            entry2 = loop.run_until_complete(self.manager.get_model("test-model", px_subjective=True))
            
            # Weights should NOT have been reloaded
            self.assertEqual(mock_model_from.call_count, 1)
            self.assertTrue(entry2["px_subjective"])
            
            loop.close()

if __name__ == "__main__":
    unittest.main()
