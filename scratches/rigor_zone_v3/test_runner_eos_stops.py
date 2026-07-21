"""test_runner_eos_stops.py — TDD: CUDA-Graph-Runner soll bei EOS stoppen.

Problem: runner_cache.generate() macht IMMER max_new_tokens Iterationen.
Bei Greedy-Argmax kommt EOS ab irgendeinem Punkt. Aber Runner produziert
200 Tokens davon 199 sind EOS-Pad. Das model.generate() stoppt bei EOS,
der Runner nicht.

Spezifikation:
- generate(runner, max_new) soll bei EOS-Token aufhören (nicht max_new forcieren)
- Realistisch: max_new=200 → Runner macht 1-3 Iterationen (bis EOS), nicht 200
"""
from __future__ import annotations
import sys, os
from unittest.mock import MagicMock, patch

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))


def test_runner_cache_generate_method_stops_at_eos():
    """CudaGraphRunnerCache.generate() hat EOS-Stop eingebaut."""
    from server_v35g import CudaGraphRunnerCache
    cache = CudaGraphRunnerCache()
    mock_runner = MagicMock()
    eos_token = 106

    # Mock-Schritt der EOS nach 2 Steps produziert.
    mock_runner.model.config.eos_token_id = eos_token
    step_returns = []
    for val in [42, 42, eos_token]:
        t = MagicMock()
        t.item = MagicMock(return_value=val)
        step_returns.append(t)
    mock_runner.step = MagicMock(side_effect=step_returns)
    mock_runner.append = MagicMock()
    import torch
    mock_runner.static_input_ids.clone = MagicMock(return_value=torch.tensor([[42]]))

    with patch("server_v35g.torch.cat", side_effect=lambda *a, **kw: MagicMock()):
        out = cache.generate(mock_runner, max_new_tokens=200)
        # Verify: 3 step() calls (initial + 2 + EOS)
        assert mock_runner.step.call_count == 3, \
            f"Expected 3 steps, got {mock_runner.step.call_count}"
        # Verify: 3 append() calls
        assert mock_runner.append.call_count == 3, \
            f"Expected 3 appends, got {mock_runner.append.call_count}"
        print(f"  ✓ test_runner_cache_generate_method_stops_at_eos: 3 steps, 3 appends")


def test_runner_cache_generate_runs_full_when_no_eos():
    """Wenn kein EOS kommt: läuft max_new_tokens Iterationen (Voll-Generierung)."""
    from server_v35g import CudaGraphRunnerCache
    cache = CudaGraphRunnerCache()
    mock_runner = MagicMock()
    mock_runner.model.config.eos_token_id = 106
    # Alle Returns sind 42 (kein EOS)
    step_returns = []
    for val in [42, 42, 42, 42, 42]:
        t = MagicMock()
        t.item = MagicMock(return_value=val)
        step_returns.append(t)
    mock_runner.step = MagicMock(side_effect=step_returns)
    mock_runner.append = MagicMock()
    import torch
    mock_runner.static_input_ids.clone = MagicMock(return_value=torch.tensor([[42]]))

    with patch("server_v35g.torch.cat", side_effect=lambda *a, **kw: MagicMock()):
        out = cache.generate(mock_runner, max_new_tokens=5)
        # 4 Iterationen (range(5-1))
        assert mock_runner.step.call_count == 4, \
            f"Expected 4 steps (max_new-1), got {mock_runner.step.call_count}"
        print(f"  ✓ test_runner_cache_generate_runs_full_when_no_eos: 4 steps for max_new=5")


def main() -> int:
    print("=" * 70)
    print("TDD test_runner_eos_stops (Qualitätsfix: Runner soll bei EOS stoppen)")
    print("=" * 70)
    tests = [
        ("test_runner_cache_generate_method_stops_at_eos", test_runner_cache_generate_method_stops_at_eos),
        ("test_runner_cache_generate_runs_full_when_no_eos", test_runner_cache_generate_runs_full_when_no_eos),
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
