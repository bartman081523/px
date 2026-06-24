"""
test_server_chunked_integration.py — Plan 3 Phase D Step 5 Verifikation
=========================================================================

Unit-Test der _px_gen_kwargs Auto-Detection:
  - T<4500: kein chunked_generate marker
  - T>4500: chunked_generate marker gesetzt (für 4b/E2B)
  - User-override use_cache wird respektiert

Akzeptanz:
  - Auto-Detection funktioniert für kleine + große Modelle
  - Marker-Setzung für alle drei Fälle (short/long/override)
"""
import sys
import os

_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
_SCRATCHES = os.path.join(_REPO, "scratches", "4b-image")


def test_short_input_no_chunked():
    """T<4500: kein chunked_generate (zu schnell für 4b)."""
    sys.path.insert(0, _SCRATCHES)
    sys.path.insert(0, _REPO)
    from generators import _px_gen_kwargs

    # Mock model: 4b (large)
    class MockModel:
        class _M:
            layers = [None] * 34
            embed_tokens = type("E", (), {"embedding_dim": 2560})()
            rotary_emb = None
        def named_modules(self):
            yield ("inner", self._M())

    m = MockModel()
    base = {"max_new_tokens": 16, "do_sample": False, "_input_len": 100}
    out = _px_gen_kwargs(m, base)
    assert out.get("_px_use_chunked_prefill", False) is False, \
        f"short input should NOT trigger chunked: {out}"
    print("[OK] short input → no chunked")


def test_long_input_triggers_chunked():
    """T > _LONG_INPUT_THRESHOLD + 4b: chunked_generate marker gesetzt.

    Schwelle ist seit profile_threshold_sweep.py 8800 (use_cache=True safe
    bis T=9000 mit 4b+int8+PX). Test nutzt T=10000 um jenseits zu landen.
    """
    sys.path.insert(0, _SCRATCHES)
    sys.path.insert(0, _REPO)
    from generators import _px_gen_kwargs, _LONG_INPUT_THRESHOLD

    class MockModel:
        class _M:
            layers = [None] * 34
            embed_tokens = type("E", (), {"embedding_dim": 2560})()
            rotary_emb = None
        def named_modules(self):
            yield ("inner", self._M())

    m = MockModel()
    # T = threshold + 1200 → garantiert jenseits der Schwelle
    base = {"max_new_tokens": 16, "do_sample": False,
            "_input_len": _LONG_INPUT_THRESHOLD + 1200}
    out = _px_gen_kwargs(m, base)
    assert out.get("_px_use_chunked_prefill", False) is True, \
        f"long input (T={_LONG_INPUT_THRESHOLD + 1200}) on 4b SHOULD trigger chunked: {out}"
    print(f"[OK] long input (T={_LONG_INPUT_THRESHOLD + 1200}) + 4b → chunked")


def test_small_model_no_chunked():
    """T>4500 + 1b (small model): kein chunked (passt locker in 12GB)."""
    sys.path.insert(0, _SCRATCHES)
    sys.path.insert(0, _REPO)
    from generators import _px_gen_kwargs

    class MockModel:
        class _M:
            layers = [None] * 26
            embed_tokens = type("E", (), {"embedding_dim": 1152})()
            rotary_emb = None
        def named_modules(self):
            yield ("inner", self._M())

    m = MockModel()
    base = {"max_new_tokens": 16, "do_sample": False, "_input_len": 8000}
    out = _px_gen_kwargs(m, base)
    assert out.get("_px_use_chunked_prefill", False) is False, \
        f"small model + long input should NOT trigger chunked: {out}"
    print("[OK] small model + long input → no chunked (use_cache=True OK)")


def test_user_override_respected():
    """User setzt use_cache=False explizit: kein chunked-marker (use_cache=False Pfad)."""
    sys.path.insert(0, _SCRATCHES)
    sys.path.insert(0, _REPO)
    from generators import _px_gen_kwargs

    class MockModel:
        class _M:
            layers = [None] * 34
            embed_tokens = type("E", (), {"embedding_dim": 2560})()
            rotary_emb = None
        def named_modules(self):
            yield ("inner", self._M())

    m = MockModel()
    base = {"use_cache": False, "max_new_tokens": 16, "do_sample": False, "_input_len": 8000}
    out = _px_gen_kwargs(m, base)
    # use_cache is not None → no auto-detection
    assert out.get("_px_use_chunked_prefill", False) is False, \
        f"user-override use_cache=False should skip auto-detection: {out}"
    assert out.get("use_cache") is False, "use_cache should be preserved"
    print("[OK] user-override use_cache → skip chunked")


def test_input_len_cleanup():
    """_input_len wird nach _px_gen_kwargs entfernt (interner Marker)."""
    sys.path.insert(0, _SCRATCHES)
    sys.path.insert(0, _REPO)
    from generators import _px_gen_kwargs

    class MockModel:
        class _M:
            layers = [None] * 34
            embed_tokens = type("E", (), {"embedding_dim": 2560})()
            rotary_emb = None
        def named_modules(self):
            yield ("inner", self._M())

    m = MockModel()
    base = {"max_new_tokens": 16, "do_sample": False, "_input_len": 8000}
    out = _px_gen_kwargs(m, base)
    assert "_input_len" not in out, \
        f"_input_len should be popped after _px_gen_kwargs: {out}"
    print("[OK] _input_len cleanup")


if __name__ == "__main__":
    tests = [
        ("short input no chunked",      test_short_input_no_chunked),
        ("long input triggers chunked", test_long_input_triggers_chunked),
        ("small model no chunked",      test_small_model_no_chunked),
        ("user override respected",     test_user_override_respected),
        ("input_len cleanup",           test_input_len_cleanup),
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