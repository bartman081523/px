"""test_cuda_graph_max_new_cap.py — TDD: CUDA-Graph soll nicht für
riesige max_new_tokens kompilieren.

Problem: cc-symbolic px-Mode schickt max_tokens=32000 (Anthropic-Limit).
Unser CudaGraphRunner versucht 32000-Step-Loop zu capturen → ewiges Compile
(>4 min) und CUDA-Graph-Buffer > 295MB pro Modell.

Spezifikation:
- CUDA-Graph-Pfad ist für max_new_tokens <= CAP sinnvoll
- CAP = 256 (sinnvoll für 270m, Output ist eh kurz)
- Oberhalb CAP: Fallback auf model.generate()
- Wichtig: Fallback muss funktionieren, kein Server-Hang
"""
from __future__ import annotations
import sys, os
from unittest.mock import MagicMock, patch
from pathlib import Path

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))


CUDA_GRAPH_MAX_NEW = 256


def test_should_use_cuda_graph_below_cap():
    """max_new <= CAP: True (CUDA-Graph erlaubt)."""
    from server_v35g import should_use_cuda_graph
    assert should_use_cuda_graph(50) is True
    assert should_use_cuda_graph(256) is True
    assert should_use_cuda_graph(0) is True
    print(f"  ✓ test_should_use_cuda_graph_below_cap: True for 0/50/256")


def test_should_use_cuda_graph_above_cap():
    """max_new > CAP: False (Fallback auf model.generate)."""
    from server_v35g import should_use_cuda_graph
    assert should_use_cuda_graph(257) is False
    assert should_use_cuda_graph(1000) is False
    assert should_use_cuda_graph(32000) is False
    print(f"  ✓ test_should_use_cuda_graph_above_cap: False for 257/1000/32000")


def test_cuda_graph_cap_constant_exported():
    """CAP-Konstante ist als Modul-Attribut verfügbar."""
    from server_v35g import CUDA_GRAPH_MAX_NEW
    assert isinstance(CUDA_GRAPH_MAX_NEW, int)
    assert CUDA_GRAPH_MAX_NEW > 0
    assert CUDA_GRAPH_MAX_NEW <= 512, f"CAP too high: {CUDA_GRAPH_MAX_NEW}"
    print(f"  ✓ test_cuda_graph_cap_constant_exported: CAP={CUDA_GRAPH_MAX_NEW}")


def test_runner_cache_respects_cap():
    """CudaGraphRunnerCache gibt None zurück wenn max_new > CAP."""
    from server_v35g import CudaGraphRunnerCache, should_use_cuda_graph
    cache = CudaGraphRunnerCache(max_runners_per_model=2)
    # Wenn should_use_cuda_graph False: get_or_create returnt None ohne Compile
    with patch("server_v35g.should_use_cuda_graph", return_value=False):
        result = cache.get_or_create(
            "lean", MagicMock(), MagicMock(), MagicMock(shape=MagicMock(__getitem__=lambda s, i: 1)),
            MagicMock(), max_new_tokens=32000,
        )
        assert result is None, f"Should return None when over CAP, got {result}"
        print(f"  ✓ test_runner_cache_respects_cap: None for max_new=32000")


def main() -> int:
    print("=" * 70)
    print("TDD test_cuda_graph_max_new_cap (CC-Symbolic 32k-Token-Blocker)")
    print("=" * 70)
    tests = [
        ("test_cuda_graph_cap_constant_exported", test_cuda_graph_cap_constant_exported),
        ("test_should_use_cuda_graph_below_cap", test_should_use_cuda_graph_below_cap),
        ("test_should_use_cuda_graph_above_cap", test_should_use_cuda_graph_above_cap),
        ("test_runner_cache_respects_cap", test_runner_cache_respects_cap),
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
