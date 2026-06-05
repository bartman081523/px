import unittest
import torch
import asyncio
import os
import json
from model_manager import ModelManager
from telemetry import telemetry

class TestSessionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = ModelManager()
        cls.model_id = "gemma3-270m-it-px" # The primary model for these tests
        
    def test_math_persona_regression(self):
        """Test if 'Test' persona with simple math prompt yields expected phi range."""
        # Data from session 6b28cd56.json
        persona = "Test"
        prompt = "What is 2+2?"
        expected_phi_approx = 0.90
        
        loop = asyncio.new_event_loop()
        try:
            model_entry = loop.run_until_complete(
                self.manager.get_model(self.model_id, px_subjective=True)
            )
            model = model_entry["model"]
            tokenizer = model_entry["tokenizer"]
            
            tm = self.manager._resolve_text_model(model)
            model.persona = tm.persona = persona
            
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            # Generate one token to trigger PX logic
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False
                )
            
            metrics = self.manager.get_px_metrics(self.model_id)
            phi = metrics.get("phi", 1.0)
            kurtosis = metrics.get("cognitive_signature", {}).get("kurtosis", 0)
            
            print(f"\n[Test] Persona: {persona} | Prompt: {prompt}")
            print(f"[Test] Result Phi: {phi:.4f} (Expected ~{expected_phi_approx})")
            print(f"[Test] Result Kurtosis: {kurtosis:.2f}")
            
            # We expect phi to be in the Subjective/Creative regime (not 1.0)
            self.assertLess(phi, 0.999, "Model should be in subjective mode (phi < 1.0)")
            # Allow some variance due to weight differences/updates, but it should be in the same ballpark
            self.assertGreater(phi, 0.70, "Phi dropped too low, indicates instability")
            
        finally:
            loop.close()

    def test_dmt_vibe_steering(self):
        """Test if 'DMT' persona changes routing/hyperparams compared to baseline."""
        loop = asyncio.new_event_loop()
        try:
            # 1. Baseline (No Persona)
            model_entry = loop.run_until_complete(
                self.manager.get_model(self.model_id, px_subjective=True)
            )
            model = model_entry["model"]
            tokenizer = model_entry["tokenizer"]
            tm = self.manager._resolve_text_model(model)
            
            model.persona = tm.persona = ""
            prompt = "Describe the nature of reality."
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1)
            
            metrics_base = self.manager.get_px_metrics(self.model_id)
            # gamma_base = model._px_config.get("gamma", 0.08)
            
            # 2. DMT Persona (High Chaos)
            model.persona = tm.persona = "DMT 🌀"
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1)
            
            metrics_dmt = self.manager.get_px_metrics(self.model_id)
            # Check if gamma was modulated (PersonaEngine should decrease gamma for chaos)
            # Note: _px_config might be updated in-place or copied.
            # In our implementation it's copied into token_cfg in _px_forward.
            # So we might need to check internal state or the 'zone' name.
            
            zone_dmt = metrics_dmt.get("zone", "")
            print(f"\n[Test] Persona: DMT 🌀 | Zone: {zone_dmt}")
            
            self.assertIn("Entropy", zone_dmt, "DMT persona should trigger Entropy modulation")
            
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
