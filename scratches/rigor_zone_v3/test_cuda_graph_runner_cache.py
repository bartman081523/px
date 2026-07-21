"""test_cuda_graph_runner_cache.py — TDD: Server soll CUDA-Graph-Runner pro Modell cachen.

Spezifikation:
- Pro Modell-Instanz: Dict[(B, max_seq), CUDAGraphRunner]
- Bei Request: wenn Runner für (B=1, max_seq_bucket) existiert → reuse
  sonst: erstelle + capture (1× Setup, dann schnelle replays)
- LRU-Limit: max 4 Runner pro Modell (verhindert GPU-MEM-OOM)
- Falls Setup/Capture fehlschlägt: Fallback auf model.generate()
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


def test_runner_cache_first_request_creates():
    """Erster Request für ein Modell: erstellt einen Runner."""
    from server_v35g import CudaGraphRunnerCache
    cache = CudaGraphRunnerCache(max_runners_per_model=2)

    class FakeShape:
        def __init__(self, dims): self.dims = dims
        def __getitem__(self, i): return self.dims[i]
    class FakeIds:
        def __init__(self): self.shape = FakeShape([1, 16])
    fake_ids = FakeIds()
    mock_model = MagicMock()
    mock_tok = MagicMock()
    with patch("px_patches_v3.cuda_graph_runner.CUDAGraphRunner") as mock_runner_cls, \
         patch("px_patches_v3.cuda_graph_runner.CUDAGraphRunnerConfig") as mock_cfg_cls:
        # mock runner
        mock_runner = MagicMock()
        mock_runner.setup = MagicMock(return_value=MagicMock())  # first token
        mock_runner.step = MagicMock(side_effect=[MagicMock(), MagicMock(), MagicMock()])
        mock_runner_cls.return_value = mock_runner
        mock_cfg_cls.return_value = MagicMock()

        first = cache.get_or_create(
            model_id="gemma3-270m-px-lean",
            model=mock_model,
            tokenizer=mock_tok,
            input_ids=fake_ids,
            attention_mask=MagicMock(),
            max_new_tokens=10,
        )
        assert first is mock_runner, "Should return the created runner"
        assert mock_runner_cls.call_count == 1, f"Should create 1 runner, got {mock_runner_cls.call_count}"
        assert len(cache._runners.get("gemma3-270m-px-lean", [])) == 1
        print(f"  ✓ test_runner_cache_first_request_creates: 1 runner created, cached")


def test_runner_cache_reuses_same_shape():
    """Zweiter Request mit gleicher Shape: reuse Runner, kein neuer."""
    from server_v35g import CudaGraphRunnerCache
    cache = CudaGraphRunnerCache(max_runners_per_model=2)
    mock_model = MagicMock()
    mock_tok = MagicMock()

    # Echtes Tensor-Substitut mit konsistenter .shape
    class FakeShape:
        def __init__(self, dims): self.dims = dims
        def __getitem__(self, i): return self.dims[i]
    class FakeIds:
        def __init__(self): self.shape = FakeShape([1, 16])
    fake_ids = FakeIds()

    with patch("px_patches_v3.cuda_graph_runner.CUDAGraphRunner") as mock_runner_cls:
        mock_runner = MagicMock()
        mock_runner.setup = MagicMock(return_value=MagicMock())
        mock_runner.step = MagicMock(side_effect=[MagicMock()])
        mock_runner_cls.return_value = mock_runner

        # Erster Request
        cache.get_or_create("lean", mock_model, mock_tok, fake_ids, MagicMock(), max_new_tokens=10)
        # Zweiter Request (gleiche Shape: B=1, max_new=10)
        cache.get_or_create("lean", mock_model, mock_tok, fake_ids, MagicMock(), max_new_tokens=10)
        # Verify: 1× Runner-Create (NICHT 2)
        assert mock_runner_cls.call_count == 1, \
            f"Expected 1 runner, got {mock_runner_cls.call_count}"
        print(f"  ✓ test_runner_cache_reuses_same_shape: 1 runner, reused for 2 requests")


def test_runner_cache_lru_eviction():
    """Bei mehr als max_runners_per_model: LRU-Eviction."""
    from server_v35g import CudaGraphRunnerCache
    cache = CudaGraphRunnerCache(max_runners_per_model=2)
    mock_model = MagicMock()
    mock_tok = MagicMock()

    class FakeShape:
        def __init__(self, dims): self.dims = dims
        def __getitem__(self, i): return self.dims[i]
    class FakeIds:
        def __init__(self): self.shape = FakeShape([1, 16])
    fake_ids = FakeIds()

    with patch("px_patches_v3.cuda_graph_runner.CUDAGraphRunner") as mock_runner_cls:
        mock_runner = MagicMock()
        mock_runner.setup = MagicMock(return_value=MagicMock())
        mock_runner.step = MagicMock(side_effect=[MagicMock()])
        mock_runner_cls.return_value = mock_runner

        # 3 Requests mit verschiedenen max_new → 3 Shapes → 3 Runner
        cache.get_or_create("lean", mock_model, mock_tok, fake_ids, MagicMock(), max_new_tokens=10)
        cache.get_or_create("lean", mock_model, mock_tok, fake_ids, MagicMock(), max_new_tokens=20)
        cache.get_or_create("lean", mock_model, mock_tok, fake_ids, MagicMock(), max_new_tokens=30)
        # max=2 → ältester (10) wurde evicted
        # Verify: 3 Runner erstellt
        assert mock_runner_cls.call_count == 3, \
            f"Expected 3 runners (3 different shapes), got {mock_runner_cls.call_count}"
        # Verify: 2 Runner im Cache (LRU evict)
        runners = cache._runners.get("lean", [])
        assert len(runners) == 2, f"Cache should hold max 2, got {len(runners)}"
        # Verify: 20 + 30 sind drin, 10 ist evicted
        max_news = [r["max_new"] for r in runners]
        assert 10 not in max_news, f"Oldest (10) should be evicted, cache: {max_news}"
        assert 20 in max_news and 30 in max_news
        print(f"  ✓ test_runner_cache_lru_eviction: 3 created, 2 kept (LRU evict)")


def test_runner_cache_per_model_separation():
    """Runner-Cache ist pro Modell getrennt."""
    from server_v35g import CudaGraphRunnerCache
    cache = CudaGraphRunnerCache(max_runners_per_model=4)
    mock_model = MagicMock()
    mock_tok = MagicMock()

    class FakeShape:
        def __init__(self, dims): self.dims = dims
        def __getitem__(self, i): return self.dims[i]
    class FakeIds:
        def __init__(self): self.shape = FakeShape([1, 16])
    fake_ids = FakeIds()

    with patch("px_patches_v3.cuda_graph_runner.CUDAGraphRunner") as mock_runner_cls:
        mock_runner = MagicMock()
        mock_runner.setup = MagicMock(return_value=MagicMock())
        mock_runner.step = MagicMock(side_effect=[MagicMock()])
        mock_runner_cls.return_value = mock_runner

        cache.get_or_create("lean", mock_model, mock_tok, fake_ids, MagicMock(), max_new_tokens=10)
        cache.get_or_create("baseline", mock_model, mock_tok, fake_ids, MagicMock(), max_new_tokens=10)
        # 2 Runner, einer pro Modell
        assert "lean" in cache._runners
        assert "baseline" in cache._runners
        assert len(cache._runners["lean"]) == 1
        assert len(cache._runners["baseline"]) == 1
        print(f"  ✓ test_runner_cache_per_model_separation: 2 models, separate caches")


def main() -> int:
    print("=" * 70)
    print("TDD test_cuda_graph_runner_cache (CUDA-Graph-Runner-Cache)")
    print("=" * 70)
    tests = [
        ("test_runner_cache_first_request_creates", test_runner_cache_first_request_creates),
        ("test_runner_cache_reuses_same_shape", test_runner_cache_reuses_same_shape),
        ("test_runner_cache_lru_eviction", test_runner_cache_lru_eviction),
        ("test_runner_cache_per_model_separation", test_runner_cache_per_model_separation),
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
