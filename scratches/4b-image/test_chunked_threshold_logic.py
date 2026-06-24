"""
test_chunked_threshold_logic.py — Plan 3 Phase D Step 1: TDD-rot
=====================================================================

Unit-Test der MEM_EFF_MAX_SCORE_MB-Logik in patch.py.

Aktuell (vor Phase D): MEM_EFF_THRESHOLD-Logik wählt chunked-Pfad nur wenn
T_q UND T_k > 4096.

Neu: score_mem_mb = T_q * T_k * 8 (heads) * 4 (bytes) / 1MB
     chunked wenn score_mem_mb > MEM_EFF_MAX_SCORE_MB (default 64 MB)

Akzeptanzkriterien:
  - MEM_EFF_MAX_SCORE_MB existiert in patch.py
  - Bei T_q=1 (decode): SDPA (optimal)
  - Bei T_q=128, T_k=8000: 31 MB → SDPA (klein genug)
  - Bei T_q=512, T_k=8000: 125 MB → chunked
  - Bei T_q=2048, T_k=2048: 128 MB → chunked
  - Bei T_q=512, T_k=4096: 64 MB → SDPA (knapp)

Run:
    python test_chunked_threshold_logic.py
"""
import sys
import os


def test_mem_eff_max_score_mb_exists():
    """MEM_EFF_MAX_SCORE_MB Konstante existiert in patch.py."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "px_patches"))
    from gemma3_270m_px_baseline.patch import MEM_EFF_MAX_SCORE_MB
    assert isinstance(MEM_EFF_MAX_SCORE_MB, (int, float)), \
        f"MEM_EFF_MAX_SCORE_MB must be numeric, got {type(MEM_EFF_MAX_SCORE_MB)}"
    assert MEM_EFF_MAX_SCORE_MB > 0, f"MEM_EFF_MAX_SCORE_MB must be > 0"
    print(f"[OK] MEM_EFF_MAX_SCORE_MB = {MEM_EFF_MAX_SCORE_MB}")


def test_score_mem_mb_helper():
    """Helper-Funktion score_mem_mb(T_q, T_k) → MB korrekt.

    Formel: T_q * T_k * 8 (Hq=8 für 4b) * 4 (bf16) / (1024*1024)
    """
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "px_patches"))
    try:
        from gemma3_270m_px_baseline.patch import score_mem_mb
    except ImportError:
        raise AssertionError("score_mem_mb helper missing in patch.py")

    # T_q=128, T_k=8000: 128*8000*8*4 / 1MB = 31.25 MB
    mb = score_mem_mb(128, 8000)
    assert 30 < mb < 32, f"expected ~31 MB, got {mb}"
    print(f"[OK] score_mem_mb(128, 8000) = {mb:.2f} MB")

    # T_q=512, T_k=8000: 125 MB
    mb = score_mem_mb(512, 8000)
    assert 124 < mb < 126, f"expected ~125 MB, got {mb}"
    print(f"[OK] score_mem_mb(512, 8000) = {mb:.2f} MB")

    # T_q=2048, T_k=2048: 128 MB
    mb = score_mem_mb(2048, 2048)
    assert 127 < mb < 129, f"expected ~128 MB, got {mb}"
    print(f"[OK] score_mem_mb(2048, 2048) = {mb:.2f} MB")


def test_should_use_chunked_path_decode():
    """T_q=1 (decode) → SDPA."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "px_patches"))
    try:
        from gemma3_270m_px_baseline.patch import should_use_chunked
    except ImportError:
        raise AssertionError("should_use_chunked helper missing in patch.py")

    # Decode: T_q=1 immer SDPA (chunked bringt nichts bei T_q=1)
    assert should_use_chunked(1, 8000) is False, "T_q=1 should be SDPA"
    assert should_use_chunked(1, 1) is False, "T_q=1 should be SDPA"
    print("[OK] decode (T_q=1) → SDPA")


def test_should_use_chunked_path_short():
    """Kleine Prefills → SDPA (schneller)."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "px_patches"))
    from gemma3_270m_px_baseline.patch import should_use_chunked, MEM_EFF_MAX_SCORE_MB

    # T_q=128, T_k=8000: 31 MB → SDPA
    assert should_use_chunked(128, 8000) is False, \
        f"T_q=128,T_k=8000 (31MB) should be SDPA"

    # T_q=512, T_k=4096: 64 MB → SDPA (knapp an der Grenze, default 64)
    if MEM_EFF_MAX_SCORE_MB >= 64:
        assert should_use_chunked(512, 4096) is False, \
            f"T_q=512,T_k=4096 (64MB) should be SDPA at threshold {MEM_EFF_MAX_SCORE_MB}"

    print("[OK] short prefills → SDPA")


def test_should_use_chunked_path_long():
    """Lange Prefills → chunked (memory-bound)."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "px_patches"))
    from gemma3_270m_px_baseline.patch import should_use_chunked

    # T_q=512, T_k=8000: 125 MB → chunked
    assert should_use_chunked(512, 8000) is True, \
        "T_q=512,T_k=8000 (125MB) should be chunked"

    # T_q=2048, T_k=2048: 128 MB → chunked
    assert should_use_chunked(2048, 2048) is True, \
        "T_q=2048,T_k=2048 (128MB) should be chunked"

    # T_q=4096, T_k=4096: 512 MB → chunked
    assert should_use_chunked(4096, 4096) is True, \
        "T_q=4096,T_k=4096 (512MB) should be chunked"

    print("[OK] long prefills → chunked")


if __name__ == "__main__":
    tests = [
        ("MEM_EFF_MAX_SCORE_MB exists", test_mem_eff_max_score_mb_exists),
        ("score_mem_mb helper",         test_score_mem_mb_helper),
        ("decode → SDPA",               test_should_use_chunked_path_decode),
        ("short → SDPA",                test_should_use_chunked_path_short),
        ("long → chunked",              test_should_use_chunked_path_long),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)