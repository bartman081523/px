"""Regression: ein Preset-Wechsel darf keine PX-Attribute zurücklassen.

Befund 2026-11-14 (Capability-Kampagne): Nach ACTIVE_MANIFOLD → BASELINE
blieben `_px_repetition_penalty` (1.15) und `_px_no_repeat_ngram_size` (3)
auf dem Textmodell. `generators._px_gen_kwargs` liest beide bei jeder
Anfrage; die Token-Loop-Mitigation zerlegte danach Code-Generierung
(`result. append(...)`, `result.Append(...)`), während derselbe Prompt über
den Direktpfad fehlerfrei war.
"""
import types

import torch.nn as nn
import pytest

from px_patches.gemma4_2b_px.patch import _strip_px_attrs, remove_px_patch


class _StubTextModel(nn.Module):
    """Minimales Modul, das die vom Patch gesetzten Attribute trägt."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self._px_repetition_penalty = 1.15
        self._px_no_repeat_ngram_size = 3
        self._px_config = {"config_preset": "ACTIVE_MANIFOLD"}
        self._px_original_forward = self.forward
        self.unrelated_attr = "keep me"


def test_strip_px_attrs_removes_generation_kwargs():
    m = _StubTextModel()
    removed = _strip_px_attrs(m)
    assert "_px_repetition_penalty" in removed
    assert not hasattr(m, "_px_repetition_penalty")
    assert not hasattr(m, "_px_no_repeat_ngram_size")
    # Nicht-PX-Attribute bleiben unangetastet (modular, keine Kollateralschäden)
    assert m.unrelated_attr == "keep me"


def test_strip_px_attrs_is_idempotent():
    m = _StubTextModel()
    _strip_px_attrs(m)
    assert _strip_px_attrs(m) == []


def test_remove_px_patch_leaves_no_generation_attrs(monkeypatch):
    """remove_px_patch muss die von _px_gen_kwargs gelesenen Attrs entfernen."""
    import px_patches.gemma4_2b_px.patch as patch_mod

    text_model = _StubTextModel()
    outer = types.SimpleNamespace(model=text_model)

    monkeypatch.setattr(patch_mod, "_resolve_text_model", lambda model: text_model)
    monkeypatch.setattr(patch_mod, "remove_relay", lambda tm: None)

    assert patch_mod.remove_px_patch(outer) is True

    assert not hasattr(text_model, "_px_repetition_penalty"), \
        "repetition_penalty must not survive a return to BASELINE"
    assert not hasattr(text_model, "_px_no_repeat_ngram_size"), \
        "no_repeat_ngram_size must not survive a return to BASELINE"
    assert not any(a.startswith("_px_") for a in vars(text_model))


def test_px_gen_kwargs_ignores_stripped_attrs():
    """Nach dem Strip darf generators._px_gen_kwargs nichts mehr injizieren."""
    from generators import _px_gen_kwargs, _find_px_attr

    m = _StubTextModel()
    _strip_px_attrs(m)
    assert _find_px_attr(m, "_px_repetition_penalty", default=1.0) == 1.0
    kw = _px_gen_kwargs(m, {"max_new_tokens": 8})
    assert "repetition_penalty" not in kw
    assert "no_repeat_ngram_size" not in kw
