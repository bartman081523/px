"""
crash_handler.py — Hard-Crash auf unbehandelte Exceptions
=========================================================
sys/threading/asyncio-Hooks + faulthandler. ENV-Flag PX_HARD_CRASH=0
deaktiviert (default: 1, hard crash).

Run-local.sh teet stderr nach local_debug.log — Tracebacks landen persistent
und sind tail-bar. Ohne diese Hooks schlucken server.py + generators.py +
chat_tab.py die Exceptions an mehreren Stellen und verwandeln sie in stille
500/HTML/Partial-Responses (siehe Plan 7.1 für die exakte Architektur).

Hard Rule: KEIN motor-touch. Diese Hooks sitzen ausschließlich am
Process-Boundary — keine px_patches, keine generators.py-Logik.

Usage (in app.py):
    import crash_handler
    crash_handler.install()                      # sys + threading + faulthandler
    # ... uvicorn.Server statt uvicorn.run() ...
    crash_handler.install_asyncio(server.loop)   # asyncio
"""
import asyncio
import faulthandler
import os
import sys
import threading
import traceback


_BANNER = (
    "\n" + "=" * 72 + "\n"
    "[CRASH_HANDLER] Unhandled exception — terminating process\n"
    + "=" * 72 + "\n"
)


def _log_and_die(prefix, exc_type, exc_value, exc_tb):
    """Print full traceback to stderr, then os._exit(1).

    os._exit (nicht sys.exit) umgeht alle atexit/shutdown-Handler, damit
    ein Bug nicht durch partielle Cleanup-Logik verschleiert wird.
    """
    sys.stderr.write(_BANNER)
    sys.stderr.write(f"[CRASH_HANDLER] hook={prefix}\n")
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()
    os._exit(1)


def _sys_hook(exc_type, exc_value, exc_tb):
    """sys.excepthook — Main-Thread unbehandelte Exceptions."""
    # KeyboardInterrupt soll weiterhin graceful Ctrl-C bleiben (nicht crashen).
    if issubclass(exc_type, KeyboardInterrupt):
        return sys.__excepthook__(exc_type, exc_value, exc_tb)
    _log_and_die("sys.excepthook", exc_type, exc_value, exc_tb)


def _threading_hook(args):
    """threading.excepthook — Unhandled exceptions in Thread(target=...).

    Fängt z.B. generators.py:554, 604, 723 + chat_tab.py:496 (alle unguarded,
    Python-Thread-Semantik würde sie ohne unseren Hook einfach verschlucken).
    """
    _log_and_die(
        f"threading.excepthook thread={args.thread.name!r}",
        args.exc_type, args.exc_value, args.exc_traceback,
    )


def _asyncio_hook(loop, context):
    """asyncio loop.set_exception_handler — uvicorn-Task-Fehler."""
    exc = context.get("exception")
    if exc is None:
        # Non-exception async errors (z.B. "Task was destroyed but it is
        # pending"). Trotzdem fatal — meist Logic-Bug.
        sys.stderr.write(_BANNER)
        sys.stderr.write(f"[CRASH_HANDLER] asyncio non-exc: {context}\n")
        sys.stderr.flush()
        os._exit(1)
    _log_and_die("asyncio", type(exc), exc, exc.__traceback__)


def install():
    """Install sys + threading + faulthandler hooks. Idempotent.

    Kill-Switch via ENV: PX_HARD_CRASH=0 macht install() zum no-op.
    """
    if os.environ.get("PX_HARD_CRASH", "1") == "0":
        return
    sys.excepthook = _sys_hook
    threading.excepthook = _threading_hook
    faulthandler.enable()  # SIGSEGV / native C++ crashes → stderr


def install_asyncio(loop):
    """Attach asyncio exception handler to uvicorn's event loop.

    MUSS nach Loop-Creation aufgerufen werden — uvicorn erstellt den Loop
    intern in Server.run(), daher der uvicorn.Server.startup()-Patch in
    app.py statt direkter Aufruf vor uvicorn.run().
    """
    if os.environ.get("PX_HARD_CRASH", "1") == "0":
        return
    loop.set_exception_handler(_asyncio_hook)