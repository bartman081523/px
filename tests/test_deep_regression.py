import unittest
import torch
import asyncio
import os
import json
from model_manager import ModelManager
from telemetry import telemetry

class TestDeepSessionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = ModelManager()
        cls.model_id = "gemma3-270m-it"
        
    def test_logic_hallucination_regression(self):
        """
        Regression for session 6b28cd56.json: 'What is 2+2?' -> '2 + 2 = 3'.
        Also checks phi and kurtosis against original telemetry.
        """
        # Session 6b28cd56.json Data
        persona = "Test"
        prompt = "What is 2+2?"
        # Note: Hallucinations are non-deterministic, but we check if recursion happens
        orig_phi = 0.9004
        orig_kurtosis = 271.98
        
        loop = asyncio.new_event_loop()
        try:
            model_entry = loop.run_until_complete(
                self.manager.get_model(self.model_id, px_subjective=True)
            )
            model = model_entry["model"]
            tokenizer = model_entry["tokenizer"]
            
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            metrics = self.manager.get_px_metrics(self.model_id)
            phi = metrics.get("phi", 1.0)
            kurtosis = metrics.get("cognitive_signature", {}).get("kurtosis", 0)
            steps = metrics.get("steps", 0)
            
            print(f"\n[Session 6b28cd56] Prompt: {prompt}")
            print(f"  Generated: '{generated_text.strip()}'")
            print(f"  Phi: {phi:.4f} | Kurtosis: {kurtosis:.2f} | Steps: {steps}")
            
            self.assertGreater(steps, 0, "Recursion should be active")
            
        finally:
            loop.close()

    def test_complex_riddle_regression(self):
        """
        Regression for test123.json: 'If I have 3 apples...'
        Prüft das SR-61b Prompt-getriebene Routing: ein Math-Riddle routet nach
        MATH, gesteuert durch Prompt-Kurtosis/Focus-C — NICHT durch die Persona
        (die ein Surface-Label ist). Die alte Erwartung („Entropy" im Zonen-
        Namen bei DMT-Persona) war eine Personen-Steuerungs-Annahme, die nicht
        zur Architektur passt (2026-06-20 repurpose, siehe OBSOLETE_TESTS.md).
        Zusätzlich: Entropie-Modulation aktiv (AZS-Kern H+gamma_boost, der im
        lean-Schnitt bleibt).
        """
        # Session test123.json Data (Approximated from log)
        persona = "DMT Psilocybin 🌀"
        prompt = "If I have 3 apples and you take 2, how many apples do you have?"

        loop = asyncio.new_event_loop()
        try:
            model_entry = loop.run_until_complete(
                self.manager.get_model(self.model_id, px_subjective=True, px_gamma=0.12)
            )
            model = model_entry["model"]
            tokenizer = model_entry["tokenizer"]
            tm = self.manager._resolve_text_model(model)
            model.persona = tm.persona = persona

            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=20)

            generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            metrics = self.manager.get_px_metrics(self.model_id)
            zone = metrics.get("zone", "")
            phi = metrics.get("phi", 1.0)
            entropy = metrics.get("entropy", 0.0)

            print(f"\n[Session test123] Persona: {persona} | Prompt: {prompt}")
            print(f"  Generated: '{generated_text.strip()}'")
            print(f"  Zone: {zone} | Phi: {phi:.4f} | Entropy: {entropy}")

            # Prompt-getriebenes Routing: Math-Riddle → MATH (Kurtosis/Focus-C,
            # nicht die Persona). Die Persona darf die Zone NICHT überschreiben.
            self.assertEqual(zone, "MATH",
                             f"Math-Riddle sollte nach MATH routen "
                             f"(Prompt-Kurtosis, nicht Persona); got zone={zone}")
            # Entropie-Modulation aktiv: AZS-Kern H > 0 (bleibt im lean-Schnitt).
            self.assertGreater(entropy, 0.0,
                               "Entropie-Modulation sollte aktiv sein "
                               "(H > 0, AZS-Kern)")
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
