"""test_chat_tab_token_type_ids_strip.py — pin chat_tab.py:generate_with_lock
gegen den Llama-/MiniCPM-Pfad, der `token_type_ids` nicht akzeptiert.

Hintergrund (Live-Crash auf master, 2026-06-30):
  ./run_local.sh  →  POST /v1/chat/completions mit minicpm5-1b +
  ACTIVE_MANIFOLD_RELAY-Preset  →
  CRASH_HANDLER traceback:
    ValueError: The following `model_kwargs` are not used by the model:
    ['token_type_ids']

Ursache:
  - generators.py:375-377 + 520-522 (Plan 7.2) strippen token_type_ids
    in chunked_generate, weil Llama-Pfade das Argument nicht kennen.
  - chat_tab.py:199-200 baut gen_kwargs DIREKT aus inputs (tokenizer-Output
    inkl. token_type_ids), und der Strip läuft dort nicht.
  - model.generate() validiert model_kwargs VOR dem ersten forward und
    lehnt unbekannte Keys ab.

Fix-Ort:
  - Zentralisierter Helper `strip_unsupported_model_kwargs(model, gen_kwargs)`
    in generators.py (single source of truth, beide Pfade nutzen ihn).
  - chat_tab.py ruft diesen Helper zwischen gen_kwargs-Bau und
    model.generate(**gen_kwargs) auf.

Tests hier pinnen:
  T1: strip_unsupported_model_kwargs entfernt token_type_ids für Llama-Pfad.
  T2: strip_unsupported_model_kwargs BEHÄLT token_type_ids für Gemma-Pfad
      (oder jedes Modell, das token_type_ids in forward() akzeptiert).
  T3: strip_unsupported_model_kwargs ist idempotent.
  T4: strip_unsupported_model_kwargs entfernt KEINE anderen kwargs
      (input_ids, attention_mask, streamer etc. bleiben unangetastet).
  T5: chat_tab.py Import-Test — `from gradio_tabs.chat_tab import generate`
      funktioniert, und der Helper ist referenziert (kein direkter inline
      filter).

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_chat_tab_token_type_ids_strip.py
"""
from __future__ import annotations
import os
import sys
import inspect
import unittest

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeForward:
    """Simuliert Llama-/MiniCPM-Forward: kennt `token_type_ids` NICHT
    in co_varnames (Llama-Signatur: input_ids, attention_mask, position_ids,
    past_key_values, inputs_embeds, labels, use_cache, output_attentions,
    output_hidden_states, return_dict, cache_position)."""
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, cache_position=None):
        return None


class _FakeForwardGemma:
    """Simuliert Gemma3-Forward: kennt token_type_ids (Gemma3 erlaubt es
    für Type-Token-Embeddings, auch wenn 1b es nicht nutzt — co_varnames
    enthält es)."""
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, cache_position=None, token_type_ids=None):
        return None


class _FakeModel:
    """Wrapper mit `.forward` Attribut (so wie transformers-Modelle)."""
    def __init__(self, forward_impl):
        self.forward = forward_impl


class TestStripUnsupportedModelKwargs(unittest.TestCase):
    """T1-T4: pin strip_unsupported_model_kwargs."""

    def setUp(self):
        from generators import strip_unsupported_model_kwargs
        self.strip = strip_unsupported_model_kwargs
        self.llama_model = _FakeModel(_FakeForward().forward)
        self.gemma_model = _FakeModel(_FakeForwardGemma().forward)

    def test_t1_strips_token_type_ids_for_llama(self):
        """Llama-Pfad: token_type_ids ist NICHT in co_varnames → muss weg."""
        gen_kwargs = {
            "input_ids": "fake_tensor",
            "attention_mask": "fake_mask",
            "token_type_ids": "fake_tti",
            "streamer": "fake_streamer",
        }
        result = self.strip(self.llama_model, gen_kwargs)
        self.assertNotIn(
            "token_type_ids", result,
            f"token_type_ids must be stripped for Llama-path, got: {result!r}"
        )
        # Andere kwargs bleiben unangetastet
        self.assertEqual(result["input_ids"], "fake_tensor")
        self.assertEqual(result["attention_mask"], "fake_mask")
        self.assertEqual(result["streamer"], "fake_streamer")

    def test_t2_keeps_token_type_ids_for_gemma(self):
        """Gemma-Pfad: token_type_ids IST in co_varnames → bleibt drin."""
        gen_kwargs = {
            "input_ids": "fake_tensor",
            "attention_mask": "fake_mask",
            "token_type_ids": "fake_tti",
        }
        result = self.strip(self.gemma_model, gen_kwargs)
        self.assertIn(
            "token_type_ids", result,
            f"token_type_ids must be kept for Gemma-path, got: {result!r}"
        )
        self.assertEqual(result["token_type_ids"], "fake_tti")

    def test_t3_idempotent(self):
        """strip auf bereits-gestripptem Dict ist no-op."""
        gen_kwargs_with_tti = {
            "input_ids": "fake", "attention_mask": "fake",
            "token_type_ids": "fake_tti",
        }
        once = self.strip(self.llama_model, gen_kwargs_with_tti)
        twice = self.strip(self.llama_model, once)
        self.assertEqual(once, twice, "strip must be idempotent")

    def test_t4_preserves_other_kwargs(self):
        """strip entfernt KEINE anderen kwargs (input_ids, attention_mask,
        streamer, max_new_tokens, temperature etc. bleiben)."""
        gen_kwargs = {
            "input_ids": "fake_tensor",
            "attention_mask": "fake_mask",
            "streamer": "fake_streamer",
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "do_sample": True,
            "token_type_ids": "should_be_stripped",
        }
        result = self.strip(self.llama_model, gen_kwargs)
        for key in ("input_ids", "attention_mask", "streamer", "max_new_tokens",
                    "temperature", "top_p", "repetition_penalty", "do_sample"):
            self.assertIn(key, result, f"strip must not remove {key!r}")
        self.assertNotIn("token_type_ids", result)


class TestChatTabImportsStrip(unittest.TestCase):
    """T5: chat_tab.py muss den Helper aus generators importieren
    (nicht inline filtern)."""

    def test_chat_tab_uses_centralized_strip(self):
        """chat_tab.py:chat_fn (parent von generate_with_lock) muss
        strip_unsupported_model_kwargs aus generators importieren und
        VOR model.generate(**gen_kwargs) aufrufen — sonst dupliziert sich
        der Filter und Drift ist vorprogrammiert.

        Anmerkung: der strip darf im chat_fn-Body (vor Thread-Start) oder
        in der nested generate_with_lock stehen. Beides ist korrekt.
        Wichtig ist nur: irgendwo in chat_fn zwischen gen_kwargs-Bau und
        model.generate(**gen_kwargs)."""
        try:
            from gradio_tabs import chat_tab
        except ImportError as e:
            self.fail(f"chat_tab import failed: {e}")

        # 1) Modul-Source muss den Helper-Namen referenzieren.
        module_src = inspect.getsource(chat_tab)
        self.assertIn(
            "strip_unsupported_model_kwargs", module_src,
            "chat_tab must reference strip_unsupported_model_kwargs from "
            "generators. Currently, the Plan 7.2 strip in generators.py "
            "is duplicated logic that drifts from the chat_tab path — "
            "which is exactly the crash that produced this test."
        )

        # 2) Source von chat_fn extrahieren (alles bis zur nächsten
        # top-level def / class).
        import re
        chat_fn_match = re.search(
            r"^def chat_fn\(.*?(?=\n\ndef |\nclass |\Z)",
            module_src, re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(
            chat_fn_match,
            "Could not locate chat_fn top-level definition in chat_tab"
        )
        chat_fn_src = chat_fn_match.group(0)
        strip_pos = chat_fn_src.find("strip_unsupported_model_kwargs(")

        # Suche den ECHTEN model.generate(**gen_kwargs)-Aufruf (nicht den
        # Kommentar). Das ist ein Aufruf mit `**gen_kwargs` als Argument.
        # Kommentare matchen "model.generate(" auch, aber **gen_kwargs
        # nur im Code.
        gen_pos = chat_fn_src.find("**gen_kwargs")

        self.assertGreater(
            strip_pos, 0,
            f"strip_unsupported_model_kwargs must be CALLED inside chat_fn; "
            f"not found. chat_fn src (first 500 chars):\n"
            f"{chat_fn_src[:500]}"
        )
        self.assertGreater(
            gen_pos, 0,
            f"model.generate(**gen_kwargs) call must be inside chat_fn; "
            f"not found."
        )
        self.assertGreater(
            gen_pos, strip_pos,
            f"model.generate(**gen_kwargs) must be called AFTER "
            f"strip_unsupported_model_kwargs (strip@{strip_pos}, "
            f"generate@{gen_pos}). If strip is INSIDE a nested helper "
            f"that runs after gen_kwargs is already passed to generate(), "
            f"the crash is back."
        )


class TestStripPxInternalKwargs(unittest.TestCase):
    """T7-T10: pin strip_unsupported_model_kwargs für PX-Engine-Internes.

    Hintergrund (Live-Crash auf master, 2026-06-30):
      POST /v1/chat/completions mit long-context →
        ValueError: The following `model_kwargs` are not used by the model:
        ['_px_use_chunked_prefill']

    Ursache: generators.py:_px_gen_kwargs setzt `base["_px_use_chunked_prefill"]=True`
    als Marker für chunked_generate. Dieser Marker wird in `generate()`
    gepoppt (Zeile 440: `gen_kwargs.pop("_px_use_chunked_prefill", False)`),
    ABER nur im chunked-Pfad. Im Standard-pfad bleibt er in gen_kwargs und
    landet bei model.generate() — wo transformers ihn als unbekanntes kwarg
    ablehnt.

    Fix: strip_unsupported_model_kwargs muss PX-Engine-Internes (alle Keys
    mit `_px_` prefix) auch strippen — diese sind NIE in model.forward.
    """

    def setUp(self):
        from generators import strip_unsupported_model_kwargs
        self.strip = strip_unsupported_model_kwargs
        self.llama_model = _FakeModel(_FakeForward().forward)
        self.gemma_model = _FakeModel(_FakeForwardGemma().forward)

    def test_t7_strips_px_use_chunked_prefill(self):
        """_px_use_chunked_prefill ist PX-Internes → muss weg, BEIDE Pfade."""
        gen_kwargs = {
            "input_ids": "fake_tensor",
            "attention_mask": "fake_mask",
            "_px_use_chunked_prefill": True,
        }
        # Auf Llama UND Gemma — der Key ist NIE in forward.
        for model in (self.llama_model, self.gemma_model):
            result = self.strip(model, gen_kwargs)
            self.assertNotIn(
                "_px_use_chunked_prefill", result,
                f"_px_use_chunked_prefill must be stripped (PX-internal); "
                f"got: {result!r}"
            )

    def test_t8_strips_all_px_prefixed_keys(self):
        """ALLE Keys mit _px_ prefix sind PX-Internes und müssen gestrippt
        werden, unabhängig davon ob sie in forward.co_varnames sind.

        Hintergrund: PX-Engine hat ~10 interne Marker (_px_input_len,
        _px_use_chunked_prefill, _px_skip_apply_template, etc.). Strip muss
        pauschal für _px_* greifen — sonst crasht ein neuer Marker sofort
        beim ersten Auftreten (Live-Crash-Logik)."""
        gen_kwargs = {
            "input_ids": "fake",
            "attention_mask": "fake",
            "_px_use_chunked_prefill": True,
            "_px_input_len": 1234,
            "_px_skip_apply_template": False,
            "_px_some_future_marker": "x",
        }
        result = self.strip(self.llama_model, gen_kwargs)
        for k in ("_px_use_chunked_prefill", "_px_input_len",
                  "_px_skip_apply_template", "_px_some_future_marker"):
            self.assertNotIn(
                k, result,
                f"{k!r} (_px_-prefixed) must be stripped; got: {result!r}"
            )
        # non-_px_ keys bleiben
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

    def test_t9_idempotent_with_px_keys(self):
        """Strip mit _px_-Keys ist idempotent (zweiter Aufruf no-op)."""
        gen_kwargs = {
            "input_ids": "fake", "attention_mask": "fake",
            "_px_use_chunked_prefill": True,
        }
        once = self.strip(self.llama_model, gen_kwargs)
        twice = self.strip(self.llama_model, once)
        self.assertEqual(once, twice, "strip must be idempotent for _px_ keys")

    def test_t10_combined_token_type_ids_and_px(self):
        """Real-Crash-Szenario: token_type_ids UND _px_use_chunked_prefill
        gleichzeitig in gen_kwargs → BEIDE müssen weg (Llama-Pfad)."""
        gen_kwargs = {
            "input_ids": "fake", "attention_mask": "fake",
            "token_type_ids": "should_be_stripped",
            "_px_use_chunked_prefill": True,
            "streamer": "fake_streamer",
        }
        result = self.strip(self.llama_model, gen_kwargs)
        self.assertNotIn("token_type_ids", result)
        self.assertNotIn("_px_use_chunked_prefill", result)
        self.assertEqual(result["input_ids"], "fake")
        self.assertEqual(result["streamer"], "fake_streamer")


class TestLlamaForwardHasNoTokenTypeIds(unittest.TestCase):
    """Sanity-Check: Vergewissert uns, dass die Llama-Forward-Signatur
    tatsächlich kein token_type_ids hat. Falls transformers das ändert,
    fällt dieser Test und signalisiert: Strip-Logik neu bewerten."""

    def test_llama_forward_signature_excludes_token_type_ids(self):
        try:
            from transformers.models.llama.modeling_llama import LlamaForCausalLM
            import inspect
            sig = inspect.signature(LlamaForCausalLM.forward)
            self.assertNotIn(
                "token_type_ids", sig.parameters,
                "LlamaForCausalLM.forward now accepts token_type_ids — "
                "revisit strip_unsupported_model_kwargs and remove the "
                "filter if no longer needed."
            )
        except ImportError:
            self.skipTest("transformers Llama not installed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
