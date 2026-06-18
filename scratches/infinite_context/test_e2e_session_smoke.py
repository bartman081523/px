"""
Optional end-to-end smoke test: load gemma3-1b-it + the real 3479b4f9 session,
inject Layer A (InfiniteContextManager) into the tokenization path, generate, and
assert NO OOM + non-empty text. Mirrors gradio_tabs/chat_tab.py:chat_fn and
generators.py but with the windowing call inserted (the surgical change).

Skipped automatically when CUDA or the model is unavailable, so it never breaks CI
on CPU-only machines. Run explicitly on the RTX 2060 box:

    ../../open-mythos_p2/venv_openmythos/bin/python -m pytest test_e2e_session_smoke.py -s

Env:
  ALL_SPACE_E2E=1  -> force-run even if heuristics say skip
"""
import os
import json
import unittest

E2E = os.environ.get("ALL_SPACE_E2E") == "1"
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
SESSION = os.path.join(os.path.dirname(__file__), "..", "..", "sessions", "3479b4f9.json")
SESSION = os.path.abspath(SESSION)


@unittest.skipUnless(E2E and HAS_CUDA, "set ALL_SPACE_E2E=1 on a CUDA box to run")
class TestE2ESessionSmoke(unittest.TestCase):
    def test_3479b4f9_no_oom(self):
        import asyncio
        from model_manager import ModelManager
        from infinite_context import InfiniteContextManager

        with open(SESSION) as f:
            history = json.load(f).get("history", [])
        self.assertGreater(len(history), 0)

        async def run():
            manager = ModelManager()
            entry = await manager.get_model(
                "gemma3-1b-it", px_subjective=True, px_config_preset="ACTIVE_MANIFOLD"
            )
            tok = entry["tokenizer"]
            model = entry["model"]

            # The surgical change: bound the prompt before tokenization.
            mgr = InfiniteContextManager(max_tokens=2048, max_history_messages=None)
            processed = mgr.process_history(history, tokenizer=tok)

            input_text = tok.apply_chat_template(processed, tokenize=False, add_generation_prompt=True)
            inputs = tok(input_text, return_tensors="pt").to(model.device)
            self.assertLessEqual(inputs["input_ids"].shape[1], 2200, "prompt should be bounded")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return text

        text = asyncio.run(run())
        self.assertGreater(len(text.strip()), 0, "generation should be non-empty")


if __name__ == "__main__":
    unittest.main()