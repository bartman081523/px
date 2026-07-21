"""test_model_manager_cache.py — TDD: ModelManager soll Modelle cachen.

Spezifikation (DevMind 1.0 — Spec vor Implementation):
- ModelManager hält Dict[model_id, (tokenizer, model)]
- get_model(model_id):
  - Falls im Cache: return ohne I/O
  - Falls nicht im Cache: lade, füge zum Cache hinzu
- Optional: LRU-Eviction wenn GPU-Memory > threshold (für später, jetzt out-of-scope)
- Wichtig: keine _unload() bei Cache-Hit (verhindert Reload-Bug)
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch
from pathlib import Path

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))


def test_model_manager_caches_first_request():
    """Erster Request für ein Modell lädt es; keine _unload-Aufrufe."""
    from server_v35g import ModelManager
    mm = ModelManager()
    mm._models = {}  # Cache-Dict (vorher _model als single slot)

    with patch("server_v35g.AutoTokenizer") as mock_tok, \
         patch("server_v35g.AutoModelForCausalLM") as mock_model_cls, \
         patch("server_v35g.apply_px_patch"):
        mock_tok.from_pretrained.return_value = MagicMock(name="tok")
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        tok, mdl = mm.get_model("gemma3-270m-px-lean")
        # Verify: 1× from_pretrained called, model im Cache
        assert mock_model_cls.from_pretrained.call_count == 1, \
            f"Expected 1 load, got {mock_model_cls.from_pretrained.call_count}"
        assert "gemma3-270m-px-lean" in mm._models, \
            f"Model not in cache: {list(mm._models.keys())}"
        print(f"  ✓ test_model_manager_caches_first_request: 1 load, model cached")


def test_model_manager_reuses_cached_model():
    """Zweiter Request für SELBES Modell: KEIN erneuter Load."""
    from server_v35g import ModelManager
    mm = ModelManager()
    mm._models = {}

    with patch("server_v35g.AutoTokenizer") as mock_tok, \
         patch("server_v35g.AutoModelForCausalLM") as mock_model_cls, \
         patch("server_v35g.apply_px_patch"):
        mock_tok.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Erster Request
        tok1, mdl1 = mm.get_model("gemma3-270m-px-lean")
        # Zweiter Request (selbes Modell)
        tok2, mdl2 = mm.get_model("gemma3-270m-px-lean")
        # Verify: 1× Load (NICHT 2)
        assert mock_model_cls.from_pretrained.call_count == 1, \
            f"Expected 1 total load (cached), got {mock_model_cls.from_pretrained.call_count}"
        assert tok1 is tok2, "Tokenizers should be identical (cached)"
        assert mdl1 is mdl2, "Models should be identical (cached)"
        print(f"  ✓ test_model_manager_reuses_cached_model: cached, 0 new loads")


def test_model_manager_switch_loads_only_new():
    """Switch zu ANDEREM Modell: lädt nur das NEUE, behält das alte im Cache."""
    from server_v35g import ModelManager
    mm = ModelManager()
    mm._models = {}

    with patch("server_v35g.AutoTokenizer") as mock_tok, \
         patch("server_v35g.AutoModelForCausalLM") as mock_model_cls, \
         patch("server_v35g.apply_px_patch"):
        mock_tok.from_pretrained.return_value = MagicMock()
        # Jeder Load returnt ein NEUES MagicMock (verschiedene Instanzen)
        model_instances = [MagicMock(name=f"model_{i}") for i in range(5)]
        for m in model_instances:
            m.model = MagicMock()
        model_iter = iter(model_instances)
        mock_model_cls.from_pretrained.side_effect = lambda *a, **kw: next(model_iter)

        # Request 1: lean
        mm.get_model("gemma3-270m-px-lean")
        # Request 2: baseline (Switch)
        mm.get_model("gemma3-270m-px-baseline")
        # Request 3: lean zurück
        mm.get_model("gemma3-270m-px-lean")
        # Request 4: baseline zurück
        mm.get_model("gemma3-270m-px-baseline")

        # Verify: 2 Loads (lean + baseline), NICHT 4
        assert mock_model_cls.from_pretrained.call_count == 2, \
            f"Expected 2 loads (lean + baseline), got {mock_model_cls.from_pretrained.call_count}"
        # Beide im Cache
        assert "gemma3-270m-px-lean" in mm._models
        assert "gemma3-270m-px-baseline" in mm._models
        # Cache-Größe
        assert len(mm._models) == 2, f"Cache should have 2 models, got {len(mm._models)}"
        print(f"  ✓ test_model_manager_switch_loads_only_new: 2 loads, 2 in cache")


def test_model_manager_no_unload_call():
    """Spezifikation: KEIN _unload() in get_model() bei Cache-Hit."""
    from server_v35g import ModelManager
    mm = ModelManager()
    mm._models = {}

    with patch("server_v35g.AutoTokenizer") as mock_tok, \
         patch("server_v35g.AutoModelForCausalLM") as mock_model_cls, \
         patch("server_v35g.apply_px_patch") as mock_apply:
        mock_tok.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Spy auf _unload
        with patch.object(mm, "_unload", wraps=mm._unload) as mock_unload:
            mm.get_model("gemma3-270m-px-lean")  # Erstes Load
            mm.get_model("gemma3-270m-px-lean")  # Cache-Hit
            # Verify: _unload NICHT aufgerufen (war der Bug)
            assert mock_unload.call_count == 0, \
                f"_unload() should NOT be called (cache), got {mock_unload.call_count}"
            print(f"  ✓ test_model_manager_no_unload_call: _unload not called on cache-hit")


def test_model_manager_404_unknown_model():
    """Unbekanntes Modell → HTTPException 404 (kein Load-Versuch)."""
    from server_v35g import ModelManager
    from fastapi import HTTPException
    mm = ModelManager()
    mm._models = {}

    with patch("server_v35g.AutoModelForCausalLM") as mock_model_cls:
        try:
            mm.get_model("nonexistent-model")
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 404, f"Expected 404, got {e.status_code}"
        assert mock_model_cls.from_pretrained.call_count == 0, \
            "Should not attempt to load unknown model"
        print(f"  ✓ test_model_manager_404_unknown_model: 404 raised, no load attempt")


def main() -> int:
    print("=" * 70)
    print("TDD test_model_manager_cache (Multi-Model-Cache Refactor)")
    print("=" * 70)
    tests = [
        ("test_model_manager_caches_first_request", test_model_manager_caches_first_request),
        ("test_model_manager_reuses_cached_model", test_model_manager_reuses_cached_model),
        ("test_model_manager_switch_loads_only_new", test_model_manager_switch_loads_only_new),
        ("test_model_manager_no_unload_call", test_model_manager_no_unload_call),
        ("test_model_manager_404_unknown_model", test_model_manager_404_unknown_model),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR ({type(e).__name__}): {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*70}")
    print(f"Tests: {len(tests) - failed}/{len(tests)} grün")
    print(f"{'='*70}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
