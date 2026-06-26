"""tests/test_auto_tune_defaults.py — TDD tests for auto_tune_defaults module.

Run: python tests/test_auto_tune_defaults.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradio_tabs.auto_tune_defaults import (
    AUTO_TUNABLE_PARAMS,
    calibrated_gamma,
    bridge_top_p,
    bridge_repetition_penalty,
    scale_adaptive_temperature,
    calibrated_values,
    resolve_for_backend,
)


def test_calibrated_gamma_1b():
    assert calibrated_gamma("gemma3-1b-it") == 0.12


def test_calibrated_gamma_4b():
    assert calibrated_gamma("gemma3-4b-it") == 0.05


def test_calibrated_gamma_270m():
    assert calibrated_gamma("gemma3-270m-it") == 0.08


def test_calibrated_gamma_unknown():
    assert calibrated_gamma("nonexistent-model") is None


def test_bridge_top_p():
    assert bridge_top_p() == 0.9


def test_bridge_repetition_penalty():
    assert bridge_repetition_penalty() == 1.15


def test_scale_adaptive_temperature_1b():
    assert scale_adaptive_temperature("gemma3-1b-it") == 0.6


def test_scale_adaptive_temperature_4b():
    assert scale_adaptive_temperature("gemma3-4b-it") == 1.0


def test_scale_adaptive_temperature_270m():
    assert scale_adaptive_temperature("gemma3-270m-it") == 0.3


def test_calibrated_values_1b():
    cv = calibrated_values("gemma3-1b-it")
    assert cv["px_gamma"] == 0.12
    assert cv["top_p"] == 0.9
    assert cv["repetition_penalty"] == 1.15


def test_resolve_auto_tune_on():
    r = resolve_for_backend("gemma3-1b-it", True,
                            {"temperature": 0.7, "max_tokens": 1024})
    assert r == {"px_gamma": None, "top_p": 0.9,
                 "repetition_penalty": 1.15,
                 "temperature": 0.7, "max_tokens": 1024}


def test_resolve_auto_tune_off_passthrough():
    r = resolve_for_backend("gemma3-1b-it", False,
                            {"px_gamma": 0.08, "top_p": 0.95,
                             "repetition_penalty": 1.15,
                             "temperature": 0.7, "max_tokens": 1024})
    assert r["px_gamma"] == 0.08


def test_resolve_auto_tune_off_defaults_empty():
    r = resolve_for_backend("gemma3-1b-it", False, {})
    assert r == {"px_gamma": None, "top_p": 0.9,
                 "repetition_penalty": 1.15,
                 "temperature": 0.7, "max_tokens": 1024}


def test_auto_tunable_params():
    assert "px_gamma" in AUTO_TUNABLE_PARAMS
    assert "top_p" in AUTO_TUNABLE_PARAMS
    assert "repetition_penalty" in AUTO_TUNABLE_PARAMS


_TESTS = [
    test_calibrated_gamma_1b,
    test_calibrated_gamma_4b,
    test_calibrated_gamma_270m,
    test_calibrated_gamma_unknown,
    test_bridge_top_p,
    test_bridge_repetition_penalty,
    test_scale_adaptive_temperature_1b,
    test_scale_adaptive_temperature_4b,
    test_scale_adaptive_temperature_270m,
    test_calibrated_values_1b,
    test_resolve_auto_tune_on,
    test_resolve_auto_tune_off_passthrough,
    test_resolve_auto_tune_off_defaults_empty,
    test_auto_tunable_params,
]


def main():
    passed = 0
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())