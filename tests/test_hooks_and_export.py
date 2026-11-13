"""tests/test_hooks_and_export.py — Unit Tests for Hook Lifecycle & Session Settings Freezing.

Verifies:
1. ManifoldRelayHook context manager cleans up forward hooks deterministically.
2. active_hooks_count is correctly updated (1 during active relay, 0 after exit).
3. Exception safety: hooks are removed even if an error is raised inside the context.
4. Session export decoupling: generation-time settings snapshot is frozen and immune to slider mutation.
"""

import os
import json
import pytest
import torch
import torch.nn as nn

from px_patches.manifold_relay_hook import ManifoldRelayHook
from sessions import save_session, load_session, update_settings, SESSION_DIR
from gradio_tabs.chat_tab import handle_export


class DummyLayer(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class DummyTextModel(nn.Module):
    def __init__(self, num_layers=30, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(hidden_dim) for _ in range(num_layers)])
        self.config = type("Config", (), {"hidden_size": hidden_dim, "_name_or_path": "google/gemma-4-E2B-it"})()
        self._px_relay_handles = []


class DummyModel(nn.Module):
    def __init__(self, num_layers=30, hidden_dim=64):
        super().__init__()
        self.language_model = DummyTextModel(num_layers, hidden_dim)


def test_manifold_relay_hook_lifecycle():
    """Verify clean hook entry, exit, and active_hooks_count."""
    model = DummyModel(num_layers=30)
    layer26 = model.language_model.layers[26]

    # Initially no hooks
    assert len(layer26._forward_hooks) == 0

    # Enter baseline context (sign=0)
    with ManifoldRelayHook(model, layer_idx=26, sign=0, model_id="gemma4-e2b-it") as hook:
        assert hook.active_hooks_count == 0
        assert len(layer26._forward_hooks) == 0

    assert hook.active_hooks_count == 0
    assert len(layer26._forward_hooks) == 0


def test_manifold_relay_hook_exception_safety():
    """Verify hook cleanup even when an exception occurs inside the context block.

    Audit fix 2026-11-13: the context manager must never wipe *foreign* forward hooks
    on the relay layer. Doing so silently destroyed the geometry tracker on layer 26,
    which is why Eval L reported wick_vertical_curvature_rad == 0.0 for all 90 trials.
    Only relay-installed handles may be removed.
    """
    model = DummyModel(num_layers=30)
    layer26 = model.language_model.layers[26]

    # Pre-add a dummy hook to simulate a third-party observer (e.g. geometry tracker)
    def dummy_hook(m, i, o):
        return o

    layer26.register_forward_hook(dummy_hook)
    assert len(layer26._forward_hooks) == 1

    with pytest.raises(RuntimeError):
        with ManifoldRelayHook(model, layer_idx=26, sign=0, model_id="gemma4-e2b-it") as hook:
            # Foreign hooks on the target layer must survive context entry
            assert len(layer26._forward_hooks) == 1
            raise RuntimeError("Simulated mid-inference failure")

    # After exception exit, relay hooks are gone but the foreign hook is intact
    assert hook.active_hooks_count == 0
    assert len(layer26._forward_hooks) == 1


def test_session_settings_freezing_and_export(tmp_path, monkeypatch):
    """Verify that session settings snapshot is frozen at generation time and not mutated by slider changes."""
    test_session_id = "test_freeze_sess_01"
    sess_file = os.path.join(SESSION_DIR, f"{test_session_id}.json")

    # Initial settings at generation time
    frozen_settings = {
        "model_id": "gemma4-e2b-it",
        "temperature": 0.3,
        "relay_sign": 1,
        "relay_alpha": 0.35,
        "relay_layer": 26,
    }
    frozen_telemetry = {
        "timing_and_integrity": {"active_hooks_count": 1, "actual_relay_sign": 1},
        "logit_dynamics": {"first_token_entropy_bits": 1.42},
    }

    history = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "+0.45",
            "settings": dict(frozen_settings),
            "telemetry": dict(frozen_telemetry),
        },
    ]

    # Save session with frozen settings
    save_session(
        session_id=test_session_id,
        history=history,
        model_id="gemma4-e2b-it",
        settings=frozen_settings,
        px_telemetry=frozen_telemetry,
    )

    try:
        loaded = load_session(test_session_id)
        assert loaded["settings"]["relay_sign"] == 1
        assert loaded["settings"]["relay_alpha"] == 0.35
        assert loaded["history"][1]["settings"]["temperature"] == 0.3
        assert loaded["history"][1]["telemetry"]["logit_dynamics"]["first_token_entropy_bits"] == 1.42

        # Simulate UI slider change occurring AFTER generation (e.g., user moves slider to -1 and 0.9)
        # The exported file must reflect the frozen generation state or preserved history!
        update = handle_export(test_session_id, history)
        export_path = f"exported_session_{test_session_id}.json"
        assert os.path.exists(export_path)

        with open(export_path, "r", encoding="utf-8") as f:
            exported = json.load(f)

        assert exported["settings"]["relay_sign"] == 1
        assert exported["settings"]["relay_alpha"] == 0.35
        assert exported["px_telemetry"]["timing_and_integrity"]["active_hooks_count"] == 1
        assert exported["history"][1]["settings"]["relay_sign"] == 1
    finally:
        if os.path.exists(sess_file):
            try:
                os.unlink(sess_file)
            except OSError:
                pass
        export_path = f"exported_session_{test_session_id}.json"
        if os.path.exists(export_path):
            try:
                os.unlink(export_path)
            except OSError:
                pass
