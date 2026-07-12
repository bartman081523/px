"""TDD-Test: SearchCache vermeidet redundante DDG-HTTP-Calls.

SciMind 5.0 + Zero-Trust: Cache-Hits dürfen KEIN HTTP machen.
v2 hat 240× DDG serial aufgerufen (1× pro Task × 8 Arms × 2 Search).
v3 mit SearchCache: 30× (1× pro Task) + 0× für Cache-Hits.

RED-Phase: Existiert nicht, muss ImportError werfen.
GREEN-Phase: prefetch + 8× get() = 3 unique HTTP-Calls, nicht 24.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))


def test_search_cache_module_exists():
    """rigor_hle_suite_v3.SearchCache muss importierbar sein."""
    from rigor_hle_suite_v3 import SearchCache
    assert SearchCache is not None, "SearchCache class fehlt in rigor_hle_suite_v3.py"


def test_search_cache_avoids_redundant_http_calls():
    """Mock DDGS.text mit Call-Counter. Prefetch + 8× get() muss nur 3 HTTP-Calls
    machen (für 3 unique Tasks), nicht 24 (1 prefetch + 8 cache-lookups + 8 HTTP).
    """
    from rigor_hle_suite_v3 import SearchCache

    # Mock DDG mit deterministischem Inhalt
    mock_ddgs = MagicMock()
    call_counter = {"n": 0}

    def mock_text(query, max_results=3):
        call_counter["n"] += 1
        return [{"title": f"Result {call_counter['n']}",
                 "body": f"body for {query}",
                 "href": f"http://test/{call_counter['n']}"}]

    mock_ddgs.return_value.__enter__.return_value.text = mock_text

    with patch("rigor_hle_suite_v3.ddgs.DDGS", mock_ddgs):
        cache = SearchCache(path=Path("/tmp/_test_cache.json"))
        # 3 unique Tasks prefetchen
        hle_tasks = [
            ("t1", "math", "What is 2+2?", "4"),
            ("t2", "humanities", "Capital of France?", "Paris"),
            ("t3", "science", "Speed of light?", "299792458"),
        ]
        cache.prefetch(hle_tasks, max_workers=2)
        n_after_prefetch = call_counter["n"]
        # 8× get() — alle aus Cache, KEIN HTTP
        for _ in range(8):
            cache.get("t1", with_search=True)
            cache.get("t2", with_search=True)
        n_after_gets = call_counter["n"]

    print(f"  HTTP-Calls: prefetch={n_after_prefetch}, after 8 cache-lookups={n_after_gets}")
    assert n_after_prefetch == 3, (
        f"prefetch sollte 3 unique HTTP-Calls machen, hat {n_after_prefetch} gemacht"
    )
    assert n_after_gets == 3, (
        f"Cache-Lookups dürfen KEIN HTTP machen — "
        f"prefetch=3, after_gets={n_after_gets} (sollte 3 bleiben)"
    )


def test_search_cache_persists_to_file():
    """Nach save() + load() sind alle Einträge ohne HTTP verfügbar."""
    from rigor_hle_suite_v3 import SearchCache

    cache_path = Path("/tmp/_test_cache_persist.json")
    if cache_path.exists():
        cache_path.unlink()

    cache = SearchCache(path=cache_path)
    cache._cache["t1"] = [{"title": "x", "body": "y", "href": "z"}]
    cache.save()

    # Frischer Cache, lädt aus File
    cache2 = SearchCache(path=cache_path)
    cache2.load()
    result = cache2.get("t1", with_search=True)
    print(f"  loaded from file: t1 → {result}")
    assert len(result) == 1
    assert result[0]["title"] == "x"
    cache_path.unlink()


def test_search_cache_with_search_false_is_noop():
    """get(task_id, False) returnt [] ohne HTTP, auch wenn prefetch lief."""
    from rigor_hle_suite_v3 import SearchCache

    cache = SearchCache(path=Path("/tmp/_test_cache_noop.json"))
    cache._cache["t1"] = [{"title": "x", "body": "y", "href": "z"}]
    result = cache.get("t1", with_search=False)
    print(f"  get(t1, with_search=False) → {result}")
    assert result == [], "with_search=False muss IMMER [] returnen"


def main():
    import traceback
    print("=" * 70)
    print("RIGOR v3 — SearchCache TDD-Test")
    print("=" * 70)
    print()
    tests = [
        ("test_search_cache_module_exists", test_search_cache_module_exists),
        ("test_search_cache_avoids_redundant_http_calls", test_search_cache_avoids_redundant_http_calls),
        ("test_search_cache_persists_to_file", test_search_cache_persists_to_file),
        ("test_search_cache_with_search_false_is_noop", test_search_cache_with_search_false_is_noop),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"--- {name} ---")
        try:
            fn()
            print(f"  ✓ PASS\n")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            print(traceback.format_exc())
            failed += 1

    print("=" * 70)
    print(f"Ergebnis: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
