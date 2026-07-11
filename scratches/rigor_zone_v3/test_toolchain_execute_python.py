"""
test_toolchain_execute_python.py — TDD-Tests für execute_python
================================================================
Coverage:
    1. execute_python: print(2+2) → exit=0, stdout="4"
    2. execute_python: timeout (sleep > timeout) → timeout=True
    3. execute_python: Exception → exit=1, stderr nicht leer
    4. execute_python: RuntimeError → sauber behandelt
"""
import os
import sys

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")

from toolchain import execute_python  # noqa: E402


def test_execute_python_success():
    """print(2+2) → exit=0, stdout='4'."""
    result = execute_python("print(2+2)", timeout=5)
    assert result["exit_code"] == 0, f"Exit: {result.get('exit_code')}"
    assert "4" in result["stdout"], f"stdout: {result.get('stdout')}"
    assert result["timeout"] is False
    print(f"  ✓ print(2+2) → exit=0, stdout='{result['stdout'].strip()}'")


def test_execute_python_timeout():
    """time.sleep(10) mit timeout=1 → timeout=True."""
    result = execute_python("import time; time.sleep(10)", timeout=1)
    assert result["timeout"] is True
    assert result["exit_code"] != 0
    assert "Timeout" in result["stderr"] or "timeout" in result["stderr"].lower()
    print(f"  ✓ timeout nach {result['runtime_sec']:.1f}s: timeout={result['timeout']}")


def test_execute_python_exception():
    """raise Exception → exit=1, stderr nicht leer."""
    result = execute_python("raise ValueError('test error')", timeout=5)
    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"] or "test error" in result["stderr"]
    print(f"  ✓ Exception: exit={result['exit_code']}, stderr contains 'ValueError'")


def test_execute_python_stderr_only():
    """print to stderr only → stdout leer, stderr hat Inhalt."""
    result = execute_python("import sys; sys.stderr.write('err_msg')", timeout=5)
    assert result["exit_code"] == 0
    assert "err_msg" in result["stderr"]
    assert result["stdout"] == ""
    print(f"  ✓ stderr-only: stderr='{result['stderr'].strip()}'")


def test_execute_python_imports():
    """import math; math.factorial(5) → '120'."""
    result = execute_python("import math; print(math.factorial(5))", timeout=5)
    assert result["exit_code"] == 0
    assert "120" in result["stdout"]
    print(f"  ✓ math.factorial(5) → '{result['stdout'].strip()}'")


def main():
    print("=" * 70)
    print("execute_python TDD-Tests (v3.5 RIGOR)")
    print("=" * 70)
    tests = [
        ("test_execute_python_success", test_execute_python_success),
        ("test_execute_python_timeout", test_execute_python_timeout),
        ("test_execute_python_exception", test_execute_python_exception),
        ("test_execute_python_stderr_only", test_execute_python_stderr_only),
        ("test_execute_python_imports", test_execute_python_imports),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"Ergebnis: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
