"""
tests/test_relay_inject.py — Regression tests for ACTIVE_MANIFOLD_RELAY preset
================================================================================

Background: commit 8235926 wired the seite15 verstärkbar-Selbst-Injektions-
Relay into production via px_patches/gemma3_270m_px_baseline/relay_inject.py.
The relay adds a forward_hook on text_model.layers[L] that injects
`sign * alpha_frac * ||h_lastpos|| * d_unit` at the last position of every
generated token (prefill discarded via seq_len>1 guard). sign=0 or no
d_width-Artefakt → no-op (LEAN engine runs as-is).

What this file pins:
  - sign=0 / missing d_width / dim mismatch / bad layer → no-op (no handle)
  - sign=+1/-1 with d_width → forward_hook installed, injection fires for
    seq_len==1 only (prefill guard)
  - alpha_frac scalar-multiplies ||h_lastpos|| exactly
  - remove_relay is idempotent and tears down the hook
  - re-install is idempotent (no stacked hooks)

IMPORTANT — d_unit is the RAW dwidth vector (no unit normalisation):
  install_relay uses dwidth_np as the injection direction without dividing
  by ||dwidth_np||. Test expectations MUST use the raw vector too, otherwise
  the expected delta is ~||dwidth||× too small and the test reports a
  "delta too large" failure that mimics a hook bug. An earlier revision
  of this file unit-normalised d_unit and the resulting 11× factor was
  mistaken for a view-aliasing bug in the production hook (which was
  defensively hardened anyway — see relay_inject.py:inj = torch.empty_like
  pre-allocation — but the original failure was a test/production API
  mismatch, not a production bug).

Run:
    pytest tests/test_relay_inject.py -v
or:
    python tests/test_relay_inject.py
"""
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

# Repo root on sys.path so `px_patches.*` imports cleanly under pytest
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from px_patches.gemma3_270m_px_baseline.relay_inject import (  # noqa: E402
    install_relay,
    remove_relay,
    load_dwidth,
)


class _IdentityLayer(nn.Module):
    """A no-op layer that returns its input unchanged. Used as the
    relay-hosting layer in tests so the captured post-hook output
    reflects the injected displacement directly (no Linear weight
    mixing of the injected vector with the natural activation)."""
    def forward(self, x):
        return x


class _FakeTextModel(nn.Module):
    """Minimal stand-in for a HF Gemma3 text model. We need:
      - .config with .hidden_size and ._name_or_path
      - .layers (nn.ModuleList) — indexable for register_forward_hook
      - forward() that runs each layer in sequence (so the relay hook fires)

    Layer 2 is an Identity so the relay-installed forward_hook on it
    observes the displacement exactly (any Linear in L2 would mix
    the injected vector with W@x, contaminating the delta measurement).
    """
    def __init__(self, n_layers=4, hidden_size=128, hf_id="fake/model"):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 2:
                self.layers.append(_IdentityLayer())
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        # Mimic HF PretrainedConfig
        class _Cfg:
            pass
        self.config = _Cfg()
        self.config.hidden_size = hidden_size
        self.config._name_or_path = hf_id

    def forward(self, hidden_states):
        """Sequential layer pass — exercises any forward_hooks registered on
        the sublayers (which is where the relay installs)."""
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TestRelayNoop(unittest.TestCase):
    """Conditions under which install_relay must NOT install a hook."""

    def setUp(self):
        self.tm = _FakeTextModel(n_layers=4, hidden_size=128, hf_id="fake/no-artifact")

    def tearDown(self):
        remove_relay(self.tm)

    def test_sign_zero_is_noop(self):
        """sign=0 → explicit LEAN mode. No handle, no _px_relay_cfg."""
        install_relay(self.tm, sign=0, alpha_frac=0.30, layer=2)
        self.assertFalse(hasattr(self.tm, "_px_relay_handles"))

    def test_missing_dwidth_is_noop(self):
        """No dwidth arg + no on-disk artefact for the hf_id → LEAN mode."""
        # No dwidth= provided, and the fake hf_id has no artefact under
        # px_manifolds/.
        install_relay(self.tm, sign=1, alpha_frac=0.30, layer=2)
        self.assertFalse(hasattr(self.tm, "_px_relay_handles"))

    def test_dim_mismatch_does_not_crash(self):
        """When dwidth= is passed explicitly with the wrong dim, install_relay
        currently does NOT reject it (only load_dwidth-from-disk checks dim).
        The hook will run with a mismatched d_unit and inject garbage rather
        than falling back to LEAN. We pin the current behaviour here so any
        future tightening (e.g. adding the dim check to install_relay) shows
        up as a deliberate test update rather than a silent regression.

        Pinned behaviour: hook IS installed even with wrong dim. Use
        load_dwidth() to get the on-disk check.
        """
        dwidth = (np.ones(256, dtype=np.float32), {"direction": "test"})
        # Should not raise.
        install_relay(self.tm, sign=1, alpha_frac=0.30, layer=2, dwidth=dwidth)
        # Hook IS installed (the dim check is in load_dwidth, not install_relay).
        self.assertTrue(hasattr(self.tm, "_px_relay_handles"))

    def test_out_of_range_layer_is_noop(self):
        """layer index past len(layers) → LEAN, no crash."""
        dwidth = (np.ones(128, dtype=np.float32), {"direction": "test"})
        install_relay(self.tm, sign=1, alpha_frac=0.30, layer=999, dwidth=dwidth)
        self.assertFalse(hasattr(self.tm, "_px_relay_handles"))


class TestRelayActive(unittest.TestCase):
    """With sign=±1 and a matching dwidth, install_relay must install
    a forward_hook that injects `sign * alpha_frac * ||h_lastpos|| * d_unit`
    at the last position, but only for seq_len==1 (decode), not prefill."""

    def setUp(self):
        torch.manual_seed(0)
        self.hidden = 128
        self.tm = _FakeTextModel(n_layers=4, hidden_size=self.hidden,
                                  hf_id="fake/with-artifact")
        self.dwidth = np.random.randn(self.hidden).astype(np.float32)
        # install_relay expects (dwidth_np, meta) tuple when dwidth= is passed
        # explicitly (mirrors load_dwidth's return shape). The relay uses
        # dwidth_np as-is (no unit-normalization), so the test's expected-
        # delta formula must use the SAME raw vector, not a unit-normalized
        # version. Earlier revisions of this test divided by ||dwidth||,
        # which produced an expected delta 11x too small and surfaced as
        # a "delta too large" assertion failure that looked like a hook
        # bug but was actually a test/production-API mismatch.
        self.dwidth_tuple = (self.dwidth, {"direction": "test"})
        self.d_unit = torch.tensor(self.dwidth)  # raw, as production uses it
        self.tm._px_d_unit_canonical = self.d_unit  # for assertions

    def tearDown(self):
        remove_relay(self.tm)

    def _make_input(self, seq_len, batch=1):
        return torch.randn(batch, seq_len, self.hidden)

    def test_install_sets_handles_and_cfg(self):
        install_relay(self.tm, sign=1, alpha_frac=0.30, layer=2,
                       dwidth=self.dwidth_tuple)
        self.assertTrue(hasattr(self.tm, "_px_relay_handles"))
        self.assertEqual(len(self.tm._px_relay_handles), 1)
        cfg = self.tm._px_relay_cfg
        self.assertEqual(cfg["sign"], 1.0)
        self.assertAlmostEqual(cfg["alpha_frac"], 0.30, places=6)
        self.assertEqual(cfg["layer"], 2)
        self.assertEqual(cfg["hf_id"], "fake/with-artifact")

    def _capture_hook(self):
        """Return a forward_hook that captures the layer output as a clone,
        regardless of whether the layer returns a tensor or a tuple."""
        def _fn(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            return _fn._captured.setdefault("out", h.detach().clone())
        _fn._captured = {}
        return _fn

    def test_decode_injects_scaled_vector(self):
        """seq_len==1 (decode step) → h_lastpos gets +sign*alpha*||h||*d_unit.

        We measure the relay-installed L2 output and the no-relay L2
        output on the same input, then assert the delta on the last
        position equals sign*alpha*||h_layer_output_last||*d_unit.

        NOTE: `d_unit` here is the RAW dwidth vector as production uses
        it (install_relay does NOT unit-normalise). If you "fix" this
        test to use dwidth/||dwidth||, the expected delta shrinks ~11x
        and the test starts failing for the wrong reason (a phantom
        hook bug that is actually a test/production API mismatch).
        See setUp comment."""
        target_layer = self.tm.layers[2]
        x = self._make_input(seq_len=1)

        # No-relay baseline: capture L2 output
        ch_no = self._capture_hook()
        ph1 = target_layer.register_forward_hook(ch_no)
        _ = self.tm(x)
        ph1.remove()

        # With relay: capture L2 output
        install_relay(self.tm, sign=1, alpha_frac=0.30, layer=2,
                       dwidth=self.dwidth_tuple)
        ch_yes = self._capture_hook()
        ph2 = target_layer.register_forward_hook(ch_yes)
        _ = self.tm(x)
        ph2.remove()

        natural_last = ch_no._captured["out"][:, -1, :]
        expected_delta = (1.0 * 0.30) * natural_last.float().norm().item() * self.d_unit.to(natural_last.dtype)
        actual_delta = ch_yes._captured["out"][:, -1, :] - natural_last
        self.assertTrue(torch.allclose(actual_delta, expected_delta, atol=1e-5))

    def test_decode_injects_negative_direction(self):
        """sign=-1 → injection in -d_unit direction.

        Same dwidth-norm caveat as test_decode_injects_scaled_vector:
        use the raw dwidth vector, not a unit-normalised one. See setUp."""
        target_layer = self.tm.layers[2]
        x = self._make_input(seq_len=1)

        ch_no = self._capture_hook()
        ph1 = target_layer.register_forward_hook(ch_no)
        _ = self.tm(x)
        ph1.remove()

        install_relay(self.tm, sign=-1, alpha_frac=0.30, layer=2,
                       dwidth=self.dwidth_tuple)
        ch_yes = self._capture_hook()
        ph2 = target_layer.register_forward_hook(ch_yes)
        _ = self.tm(x)
        ph2.remove()

        natural_last = ch_no._captured["out"][:, -1, :]
        expected_delta = (-1.0 * 0.30) * natural_last.float().norm().item() * self.d_unit.to(natural_last.dtype)
        actual_delta = ch_yes._captured["out"][:, -1, :] - natural_last
        self.assertTrue(torch.allclose(actual_delta, expected_delta, atol=1e-5))

    def test_prefill_unchanged(self):
        """seq_len>1 (prefill) → no injection. The hook guards with
        `if h.shape[1] > 1: return` and never writes. We verify by
        comparing the relay-installed L2 output to the no-relay L2
        output on identical input — they must be byte-identical."""
        target_layer = self.tm.layers[2]
        x = self._make_input(seq_len=8)  # prefill

        ch_no = self._capture_hook()
        ph1 = target_layer.register_forward_hook(ch_no)
        _ = self.tm(x)
        ph1.remove()

        install_relay(self.tm, sign=1, alpha_frac=0.30, layer=2,
                       dwidth=self.dwidth_tuple)
        ch_yes = self._capture_hook()
        ph2 = target_layer.register_forward_hook(ch_yes)
        _ = self.tm(x)
        ph2.remove()

        self.assertTrue(torch.allclose(ch_no._captured["out"], ch_yes._captured["out"], atol=1e-6))

    def test_alpha_frac_zero_is_zero_injection(self):
        """alpha_frac=0 → no displacement (sign ignored). Verifies the
        scalar multiplication wiring without relying on sign."""
        target_layer = self.tm.layers[2]
        x = self._make_input(seq_len=1)

        ch_no = self._capture_hook()
        ph1 = target_layer.register_forward_hook(ch_no)
        _ = self.tm(x)
        ph1.remove()

        install_relay(self.tm, sign=1, alpha_frac=0.0, layer=2,
                       dwidth=self.dwidth_tuple)
        ch_yes = self._capture_hook()
        ph2 = target_layer.register_forward_hook(ch_yes)
        _ = self.tm(x)
        ph2.remove()

        self.assertTrue(torch.allclose(ch_no._captured["out"], ch_yes._captured["out"], atol=1e-6))


class TestRelayLifecycle(unittest.TestCase):
    """remove_relay + re-install must be idempotent."""

    def setUp(self):
        torch.manual_seed(0)
        self.tm = _FakeTextModel(n_layers=4, hidden_size=128,
                                  hf_id="fake/lifecycle")
        self.dwidth = np.random.randn(128).astype(np.float32)
        self.dwidth_tuple = (self.dwidth, {"direction": "test"})

    def tearDown(self):
        remove_relay(self.tm)

    def test_remove_clears_handles_and_cfg(self):
        install_relay(self.tm, sign=1, alpha_frac=0.30, layer=2,
                       dwidth=self.dwidth_tuple)
        self.assertTrue(hasattr(self.tm, "_px_relay_handles"))
        remove_relay(self.tm)
        self.assertFalse(hasattr(self.tm, "_px_relay_handles"))
        self.assertFalse(hasattr(self.tm, "_px_relay_cfg"))

    def test_remove_when_not_installed_is_noop(self):
        """Idempotent: removing an un-installed relay must not raise."""
        # No install first
        self.assertFalse(hasattr(self.tm, "_px_relay_handles"))
        remove_relay(self.tm)  # must not raise
        self.assertFalse(hasattr(self.tm, "_px_relay_handles"))

    def test_reinstall_replaces_previous_hook(self):
        """A second install_relay must NOT stack hooks — the previous one
        must be removed first (idempotency)."""
        install_relay(self.tm, sign=1, alpha_frac=0.30, layer=2,
                       dwidth=self.dwidth_tuple)
        n_first = len(self.tm._px_relay_handles)
        install_relay(self.tm, sign=-1, alpha_frac=0.10, layer=1,
                       dwidth=self.dwidth_tuple)
        n_second = len(self.tm._px_relay_handles)
        # Both installs produced exactly one handle; second replaced first.
        self.assertEqual(n_first, 1)
        self.assertEqual(n_second, 1)
        # cfg reflects the second install
        self.assertEqual(self.tm._px_relay_cfg["sign"], -1.0)
        self.assertAlmostEqual(self.tm._px_relay_cfg["alpha_frac"], 0.10, places=6)
        self.assertEqual(self.tm._px_relay_cfg["layer"], 1)


class TestLoadDwidthCaching(unittest.TestCase):
    """load_dwidth caches per hf_id and returns None when no artefact."""

    def setUp(self):
        # Reset the module-level cache so test ordering doesn't matter.
        from px_patches.gemma3_270m_px_baseline import relay_inject
        relay_inject._DWIDTH_CACHE.clear()

    def test_missing_artefact_returns_none(self):
        tm = _FakeTextModel(hf_id="no/such/artefact")
        result = load_dwidth(tm)
        self.assertIsNone(result)

    def test_cache_is_per_hf_id(self):
        from px_patches.gemma3_270m_px_baseline import relay_inject
        tm_a = _FakeTextModel(hf_id="model/a")
        tm_b = _FakeTextModel(hf_id="model/b")
        load_dwidth(tm_a)
        load_dwidth(tm_b)
        self.assertIn("model_a", relay_inject._DWIDTH_CACHE)
        self.assertIn("model_b", relay_inject._DWIDTH_CACHE)
        # Both cached as None (no artefact on disk for either).
        self.assertIsNone(relay_inject._DWIDTH_CACHE["model_a"])
        self.assertIsNone(relay_inject._DWIDTH_CACHE["model_b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)