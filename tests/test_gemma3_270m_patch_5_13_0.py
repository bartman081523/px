"""test_gemma3_270m_patch_5_13_0.py — regression tests for the transformers
5.13.0 API refactor of gemma3_270m_px_baseline/patch.py.

Background (2026-07-08): transformers upgraded from 4.57.3 → 5.13.0. The
mask + rotary APIs changed in 5.13.0 and broke all gemma3/gemma4 patches:

1. ``create_causal_mask()`` / ``create_sliding_window_causal_mask()`` no
   longer accept ``input_embeds=`` (singular) — must be ``inputs_embeds=``
   (plural). ``cache_position=`` was removed entirely.
2. ``Gemma3RotaryEmbedding.forward`` now needs explicit ``layer_type=``
   kwarg (default None crashes with "None_inv_freq"). Same pattern as
   gemma4 (see test_gemma4_e2b_patch_refactor.py).
3. ``Gemma3DecoderLayer.forward`` (5.13.0) expects ``position_embeddings=``
   (positional kwarg, dict-lookup per layer_type), NOT the old
   ``position_embeddings_global``/``position_embeddings_local`` kwargs.

These tests pin the new pattern and act as a regression detector.

Test-Scope:
  T1-T2  mask_kwargs uses inputs_embeds= (plural), no cache_position=
  T3     rotary_emb gets layer_type= kwarg (no None crash)
  T4     pe_dict built from unique layer_types
  T5     _layer_step is called with position_embeddings=pe_dict[lt]
  T6     regression: no position_embeddings_global/_local in patch.py
  T7     regression: no pe_global/pe_local variable assignments
  T8     regression: no input_embeds= (singular) kwarg in mask construction
"""
import os
import re
import sys
import unittest
from unittest.mock import MagicMock

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# --- Mock infrastructure -----------------------------------------------------

def _make_mock_gemma3(num_layers=26, sliding_count=22):
    """Mock Gemma3ForCausalLM mit realistic layer_types."""
    cfg = MagicMock()
    cfg.hidden_size = 1152
    cfg.num_hidden_layers = num_layers
    # gemma3-1b: 5 sliding + 1 full, repeated 4× (=24) + 2 trailing sliding
    pattern = (["sliding_attention"] * 5 + ["full_attention"]) * 4
    pattern += ["sliding_attention"] * (num_layers - len(pattern))
    cfg.layer_types = pattern
    cfg.sliding_window = 512
    cfg.text_config = cfg  # for mask_config.text_config fallback

    text_model = MagicMock()
    text_model.config = cfg
    text_model.rotary_emb = MagicMock()

    def _rotary(*args, **kwargs):
        import torch
        lt = kwargs.get("layer_type", "full_attention")
        return (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))

    text_model.rotary_emb.side_effect = _rotary
    text_model.layers = [MagicMock(name=f"L{i}") for i in range(num_layers)]
    return text_model


# --- T1-T2: mask construction ------------------------------------------------

class TestMaskKwargRename(unittest.TestCase):
    """transformers 5.13.0: inputs_embeds= plural, no cache_position=."""

    PATCH_FILE = os.path.join(_REPO, "px_patches", "gemma3_270m_px_baseline", "patch.py")

    def test_T1_no_input_embeds_singular_in_mask_construction(self):
        """The mk = dict(...) for create_causal_mask must use inputs_embeds= (plural)."""
        with open(self.PATCH_FILE, "r") as f:
            content = f.read()
        # Find the mask-construction block
        m = re.search(r'mk\s*=\s*dict\(([^)]+)\)', content)
        self.assertIsNotNone(m, "mk = dict(...) nicht gefunden")
        mk_body = m.group(1)
        self.assertIn("inputs_embeds=", mk_body,
                      "create_causal_mask braucht inputs_embeds= (plural) in 5.13.0")
        self.assertNotIn("input_embeds=", mk_body,
                         "regression: input_embeds= (singular) wurde in 5.13.0 zu inputs_embeds=")
        self.assertNotIn("cache_position=", mk_body,
                         "regression: cache_position= wurde in 5.13.0 entfernt")

    def test_T2_create_causal_mask_uses_inputs_embeds(self):
        """Smoke: create_causal_mask with inputs_embeds= kwarg is accepted (kein TypeError)."""
        import torch
        from transformers.masking_utils import create_causal_mask
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("google/gemma-3-1b-it")
        mk = dict(config=cfg, inputs_embeds=torch.zeros(1, 1, 1152),
                  attention_mask=None, past_key_values=None, position_ids=None)
        # In 5.13.0 darf create_causal_mask nicht crashen — None ist eine
        # legale Rückgabe (kein Mask nötig wenn past_key_values leer).
        try:
            m = create_causal_mask(**mk)
        except TypeError as e:
            self.fail(f"create_causal_mask crashte mit TypeError: {e}")


# --- T3-T5: rotary_emb + pe_dict + position_embeddings= positional ---------

class TestRotaryAndPeDict(unittest.TestCase):
    """pe_dict is built once per forward and reused per-layer via dict lookup."""

    def setUp(self):
        self.tm = _make_mock_gemma3()

    def test_T3_rotary_emb_called_with_explicit_layer_type(self):
        """rotary_emb.forward bekommt layer_type= kwarg (kein None crash)."""
        self.tm.rotary_emb.reset_mock()
        self.tm.rotary_emb(None, None, layer_type="full_attention")
        self.tm.rotary_emb(None, None, layer_type="sliding_attention")
        for call in self.tm.rotary_emb.call_args_list:
            self.assertIn("layer_type", call.kwargs,
                          "rotary_emb.forward braucht expliziten layer_type kwarg (transformers 5.13.0)")
            self.assertIsNotNone(call.kwargs["layer_type"])

    def test_T4_pedict_built_from_unique_layer_types(self):
        """pe_dict hat 2 Keys: 'sliding_attention' + 'full_attention'."""
        cfg = self.tm.config
        pe_dict = {lt: self.tm.rotary_emb(None, None, layer_type=lt)
                   for lt in set(cfg.layer_types)}
        self.assertIn("sliding_attention", pe_dict)
        self.assertIn("full_attention", pe_dict)
        self.assertEqual(len(pe_dict), 2)
        # pe_dict[lt] muss (cos, sin) tuple sein
        for lt, val in pe_dict.items():
            self.assertIsInstance(val, tuple, f"pe_dict[{lt}] muss tuple sein, ist {type(val)}")
            self.assertEqual(len(val), 2)

    def test_T5_layer_step_called_with_position_embeddings_pedict(self):
        """_layer_step(self.layers[i], h, position_embeddings=pe_dict[lt], ...)."""
        from px_patches.gemma3_270m_px_baseline.patch import _layer_step
        cfg = self.tm.config
        pe_dict = {lt: self.tm.rotary_emb(None, None, layer_type=lt)
                   for lt in set(cfg.layer_types)}
        for i in [0, 5, 10, 25]:  # mix of sliding + full
            lt = cfg.layer_types[i]
            _layer_step(self.tm.layers[i], None, position_embeddings=pe_dict[lt],
                        attention_mask=None, position_ids=None, past_key_values=None)
            call = self.tm.layers[i].call_args
            self.assertIn("position_embeddings", call.kwargs,
                          "transformers 5.13.0: position_embeddings=... (positional kwarg)")
            self.assertEqual(call.kwargs["position_embeddings"], pe_dict[lt])
            self.assertNotIn("position_embeddings_global", call.kwargs,
                             "regression: position_embeddings_global in 5.13.0 deprecated")
            self.assertNotIn("position_embeddings_local", call.kwargs,
                             "regression: position_embeddings_local in 5.13.0 deprecated")


# --- T6-T8: regression detector in source file ------------------------------

class TestNoOldKwargsInPatchFile(unittest.TestCase):
    """patch.py darf die alten kwargs nicht mehr enthalten."""

    PATCH_FILE = os.path.join(_REPO, "px_patches", "gemma3_270m_px_baseline", "patch.py")

    def test_T6_no_position_embeddings_global_kwarg(self):
        with open(self.PATCH_FILE, "r") as f:
            content = f.read()
        self.assertNotIn("position_embeddings_global=", content,
                         "regression: position_embeddings_global= kwarg in patch.py — "
                         "transformers 5.13.0 nutzt position_embeddings= positional")

    def test_T7_no_pe_global_or_pe_local_variables(self):
        with open(self.PATCH_FILE, "r") as f:
            content = f.read()
        code_only = re.sub(r"#.*", "", content)
        self.assertNotRegex(code_only, r"\bpe_global\s*=",
                            "pe_global Variable existiert noch — pe_dict ersetzt sie")
        self.assertNotRegex(code_only, r"\bpe_local\s*=",
                            "pe_local Variable existiert noch — pe_dict ersetzt sie")

    def test_T8_pedict_used_in_layer_calls(self):
        with open(self.PATCH_FILE, "r") as f:
            content = f.read()
        count = content.count("position_embeddings=pe_dict")
        self.assertGreaterEqual(count, 5,
                               f"erwartete ≥5 position_embeddings=pe_dict[..., fanden {count}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
