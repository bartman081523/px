"""Tests for crash_handler.py — verify the install() hooks actually kill
the process on unhandled exceptions, and the PX_HARD_CRASH=0 kill-switch
disables them.

Run: /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_crash_handler.py

Each test launches a fresh subprocess (since the hooks affect the current
process and we can't easily undo them in-process). We assert on the
subprocess return code + stderr content.
"""
import os
import subprocess
import sys
import textwrap


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PYTHON = sys.executable


def _run_subprocess(script, env_extra=None, timeout=10):
    """Run ``script`` (Python code) in a fresh subprocess and return
    (returncode, stderr). Inject sys.path so ``import crash_handler``
    resolves to the project copy."""
    full_env = os.environ.copy()
    if env_extra:
        full_env.update(env_extra)
    full_env["PYTHONPATH"] = _ROOT + os.pathsep + full_env.get("PYTHONPATH", "")
    r = subprocess.run(
        [_PYTHON, "-c", script],
        capture_output=True, text=True, timeout=timeout, env=full_env,
    )
    return r.returncode, r.stderr


# ── Tests ────────────────────────────────────────────────────────────────

def test_sys_hook_kills_process():
    """sys.excepthook: raise RuntimeError → exit code 1 + CRASH_HANDLER banner."""
    script = textwrap.dedent("""
        import crash_handler
        crash_handler.install()
        raise RuntimeError("deliberate sys-hook test crash")
    """)
    rc, stderr = _run_subprocess(script)
    assert rc == 1, f"expected exit 1, got {rc}; stderr={stderr!r}"
    assert "CRASH_HANDLER" in stderr, f"missing banner in stderr={stderr!r}"
    assert "deliberate sys-hook test crash" in stderr, stderr
    assert "Traceback" in stderr, stderr
    assert "sys.excepthook" in stderr, stderr


def test_threading_hook_kills_process():
    """threading.excepthook: Thread raises → process dies with banner."""
    script = textwrap.dedent("""
        import crash_handler
        crash_handler.install()
        import threading, time
        def boom():
            raise RuntimeError("deliberate threading-hook test crash")
        t = threading.Thread(target=boom, name="test-boomer")
        t.start()
        # Main thread sleeps; the worker crash should fire threading.excepthook
        # which calls os._exit(1) and tears down the whole process.
        time.sleep(10)
    """)
    rc, stderr = _run_subprocess(script, timeout=15)
    assert rc == 1, f"expected exit 1, got {rc}; stderr={stderr!r}"
    assert "CRASH_HANDLER" in stderr, f"missing banner in stderr={stderr!r}"
    assert "deliberate threading-hook test crash" in stderr, stderr
    assert "threading.excepthook" in stderr, stderr
    assert "test-boomer" in stderr, stderr


def test_env_flag_disables_hook():
    """PX_HARD_CRASH=0 → install() is no-op; unhandled exception propagates
    normally (Python prints traceback, exit code != 0 but NOT 1 from us)."""
    script = textwrap.dedent("""
        import crash_handler
        crash_handler.install()
        raise RuntimeError("kill-switch test crash")
    """)
    rc, stderr = _run_subprocess(script, env_extra={"PX_HARD_CRASH": "0"})
    # When hook is disabled, Python's default behavior prints the traceback
    # and exits with code 1 (uncaught exception). The CRASH_HANDLER banner
    # must NOT appear (that's the kill-switch assertion).
    assert "CRASH_HANDLER" not in stderr, (
        f"banner should be suppressed when PX_HARD_CRASH=0, got stderr={stderr!r}"
    )
    # The exception itself still propagates — that's the point of disabling
    # the hook is NOT to hide bugs from the user, but to opt out of the
    # forced process termination.
    assert "kill-switch test crash" in stderr, stderr
    # Default uncaught-exception exit code is 1 too, so rc check is the same.
    assert rc == 1, f"expected exit 1 (Python default), got {rc}"


def test_keyboard_interrupt_not_caught():
    """KeyboardInterrupt must NOT trigger hard-crash — graceful Ctrl-C."""
    script = textwrap.dedent("""
        import crash_handler
        crash_handler.install()
        raise KeyboardInterrupt()
    """)
    rc, stderr = _run_subprocess(script)
    # KeyboardInterrupt exits with 130 by default (128 + SIGINT=2).
    # Our hook must NOT fire the CRASH_HANDLER banner.
    assert "CRASH_HANDLER" not in stderr, (
        f"banner should NOT appear for KeyboardInterrupt, got stderr={stderr!r}"
    )
    assert rc != 1, f"KeyboardInterrupt should not exit with 1, got {rc}"


# ── Runner ────────────────────────────────────────────────────────────────

def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)