#!/usr/bin/env python
"""test_px_module_alignment.py — Contract fuer die Modul-Aufloesung im Defekt
9349-vs-3653 (RTPF 2026-10-06).

Der Server (model_manager) legt px_patches/ auf sys.path und laedt den
Patch als "ternary_bonsai_27b_px.patch"; long_context loest px/gf3/rt
seither ueber die bereits geladenen Kopien auf (_loaded_module /
_px_patch_module_for). Gebundener Forward und px_capture_guard muessen
dasselbe _PX_CAPTURE-Dict sehen — sonst bleibt der Capture-Guard unsichtbar
und der erste Decode-Schritt crasht (Maske written+1 gegen
Full-Capacity-Keys). Rein sys.modules-Manipulation, kein CUDA/Modell.
"""
import sys
import types

import pytest

from px_patches.ternary_bonsai_27b_px import long_context as lc

A_KEY = "ternary_bonsai_27b_px.patch"
B_KEY = "px_patches.ternary_bonsai_27b_px.patch"


def _fake_mod(key):
    mod = types.ModuleType(key)
    mod._PX_CAPTURE = {"active": False, "need_flush": False}
    return mod


def _model_with_patched_forward(name, capture):
    """Model-Fake, dessen forward aus einem Modul-Namespace 'name' stammt."""
    g = {"__name__": name, "_PX_CAPTURE": capture}
    exec("def _fwd(*a, **k):\n    return None", g)
    return types.SimpleNamespace(forward=g["_fwd"])


@pytest.fixture(autouse=True)
def _clean_sys_modules():
    saved = {k: sys.modules.get(k) for k in (A_KEY, B_KEY)}
    for k in (A_KEY, B_KEY):
        sys.modules.pop(k, None)
    yield
    for k, old in saved.items():
        if old is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = old


def test_bound_forward_wins_over_loaded_copies():
    """Server-Fall: der gebundene Forward determiniert das Modul-Objekt."""
    a, other = _fake_mod(A_KEY), _fake_mod(B_KEY)
    sys.modules[A_KEY] = a
    sys.modules[B_KEY] = other
    model = _model_with_patched_forward(A_KEY, a._PX_CAPTURE)
    assert lc._px_patch_module_for(model) is a


def test_forward_reads_only_loaded_copy():
    """Nur der Package-Key geladen, Forward unerkennbar → B gewinnt."""
    b = _fake_mod(B_KEY)
    sys.modules[B_KEY] = b
    model = types.SimpleNamespace(forward=None)
    assert lc._px_patch_module_for(model) is b


def test_no_guard_in_forward_globals_falls_to_loaded():
    """Forward ohne _PX_CAPTURE (unpatched) → bereits geladene Kopie."""
    a = _fake_mod(A_KEY)
    sys.modules[A_KEY] = a
    g = {"__name__": "transformers.models.foo.mod"}  # kein _PX_CAPTURE
    exec("def _fwd(*a, **k):\n    return None", g)
    model = types.SimpleNamespace(forward=g["_fwd"])
    assert lc._px_patch_module_for(model) is a


def test_neither_loaded_defers_to_px_patch_module(monkeypatch):
    """Kein Key geladen → klassischer deferred Import (Fallback)."""
    sentinel = _fake_mod("sentinel")
    monkeypatch.setattr(lc, "_px_patch_module", lambda: sentinel)
    model = types.SimpleNamespace(forward=None)
    assert lc._px_patch_module_for(model) is sentinel


def test_loaded_module_first_key_wins():
    a, b = _fake_mod("m.a"), _fake_mod("m.b")
    sys.modules["m.b"] = b
    assert lc._loaded_module("m.a", "m.b") is b
    sys.modules["m.a"] = a
    assert lc._loaded_module("m.a", "m.b") is a
    assert lc._loaded_module("m.z") is None