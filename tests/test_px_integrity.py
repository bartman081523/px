import unittest
import torch
import asyncio
import os
import json
import pytest
from model_manager import ModelManager

class TestPXIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = ModelManager()
        cls.model_id = "gemma3-270m-it"
        
    @pytest.mark.gpu
    def test_recursion_active(self):
        """Verify that SUBJECTIVE mode actually enters recursion (steps > 0)."""
        loop = asyncio.new_event_loop()
        try:
            # Enable DEBUG_PX to see what's happening
            os.environ["DEBUG_PX"] = "1"
            os.environ["DEBUG_ROUTING"] = "1"
            
            model_entry = loop.run_until_complete(
                self.manager.get_model(self.model_id, px_subjective=True)
            )
            model = model_entry["model"]
            tokenizer = model_entry["tokenizer"]
            
            # Simple reasoning prompt to trigger recursion
            prompt = "What is the cube root of 27?"
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=5)
            
            metrics = self.manager.get_px_metrics(self.model_id)
            print(f"\n[Test Integrity] Metrics: {metrics}")
            
            self.assertGreater(metrics.get("steps", 0), 0, "Model did NOT enter recursion loop (steps=0)")
            self.assertNotEqual(metrics.get("zone", "UNKNOWN"), "UNKNOWN", "Cognitive zone classification failed")
            self.assertGreater(len(metrics.get("telemetry_trace", [])), 0, "Telemetry trace is empty")
            
        finally:
            loop.close()

    def test_telemetry_file_persistence(self):
        """Verify that telemetry.record() persists a well-formed JSON record.

        Audit fix 2026-11-13: the test used to read a hardcoded *sibling*
        directory (/run/media/julian/ML4/ollama-work/all_space/telemetry) and
        picked `sorted(files)[-1]` lexicographically — which is not necessarily
        the record it just wrote and may be a non-JSON artefact. That produced a
        JSONDecodeError unrelated to any code under test. It now uses the real
        TELEMETRY_DIR from the module and diffs the file set around the write.
        """
        import telemetry as telemetry_mod

        telemetry = telemetry_mod.telemetry
        telemetry_dir = telemetry_mod.TELEMETRY_DIR
        self.assertTrue(os.path.isdir(telemetry_dir),
                        f"TELEMETRY_DIR does not exist: {telemetry_dir}")

        before = set(os.listdir(telemetry_dir))

        test_model = "test-integrity-model"
        telemetry.record(
            test_model,
            prompt_tokens=10,
            completion_tokens=5,
            px_metrics={"phi": 0.9},
            prompt_text="Test prompt",
            completion_text="Test response",
        )

        new_files = sorted(set(os.listdir(telemetry_dir)) - before)
        self.assertEqual(len(new_files), 1,
                         f"Expected exactly one new telemetry file, got {new_files}")

        with open(os.path.join(telemetry_dir, new_files[0]), "r") as f:
            data = json.load(f)
        self.assertEqual(data["model_id"], test_model)
        self.assertEqual(data["prompt"], "Test prompt")
        self.assertEqual(data["completion"], "Test response")
        self.assertEqual(data["prompt_tokens"], 10)
        self.assertEqual(data["completion_tokens"], 5)
        self.assertEqual(data["px_metrics"], {"phi": 0.9})

if __name__ == "__main__":
    unittest.main()
