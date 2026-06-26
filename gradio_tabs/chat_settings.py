"""Chat settings — round-trip between UI widgets and persisted dict.

Module split
------------
* `settings_from_widgets(...)` — pure helper, kwargs in -> dict out.
  Called once per user turn to capture the current widget state and
  feed it into the persistence layer.
* `widget_updates_from_settings(settings)` — pure helper, dict in ->
  dict of `gr.update(...)` kwargs out. Used by `handle_load_saved` to
  restore widget values (and lock-state) from a persisted session.
* `SettingsDebouncer` — small class that batches widget-change events
  through a single `threading.Timer`. Avoids disk-write storms when
  the user drags a Slider (e.g. temperature 0.0..1.0 yields ~50 events/s).

No Gradio event-loop coupling: the class is plain `threading` so it
works under both Gradio's asyncio loop (the timer thread is daemon)
and synchronous CLI usage (tests, `streaming_bridge.py`).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

import gradio as gr

from sessions import SETTINGS_DEFAULTS, SETTINGS_FIELDS


def settings_from_widgets(*, model_id: str, px_preset: str,
                          auto_tune: bool,
                          temperature: float, top_p: float,
                          max_tokens: int, rep_p: float,
                          px_gamma: float,
                          relay_sign: int, relay_alpha: float,
                          relay_layer: int,
                          system_profile: str,
                          system_prompt_text: Optional[str],
                          tts_engine: str, tts_sample_rate: int,
                          tts_auto: bool) -> dict:
    """Build a settings dict from current widget values.

    Type coercion (`float`, `int`, `bool`) is done here so downstream
    code never has to defend against string-typed widget values.
    """
    return {
        "model_id": model_id,
        "px_preset": px_preset,
        "auto_tune": bool(auto_tune),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "rep_p": float(rep_p),
        "px_gamma": float(px_gamma),
        "relay_sign": int(relay_sign),
        "relay_alpha": float(relay_alpha),
        "relay_layer": int(relay_layer),
        "system_profile": system_profile,
        "system_prompt_text": system_prompt_text or "",
        "tts_engine": tts_engine,
        "tts_sample_rate": int(tts_sample_rate),
        "tts_auto": bool(tts_auto),
    }


def widget_updates_from_settings(settings: dict) -> Dict[str, gr.update]:
    """Return ``gr.update(...)`` kwargs for every settings widget.

    Auto-Tune ON locks the four model-calibrated sliders (temperature,
    top_p, repetition_penalty, px_gamma) read-only via
    `gr.update(interactive=False)`. The other widgets (model_select,
    relay_*, max_tokens, system_*, tts_*) stay interactive regardless.

    Missing keys in ``settings`` fall back to SETTINGS_DEFAULTS so
    freshly-loaded older sessions still render correct widget state.
    """
    s = {**SETTINGS_DEFAULTS, **(settings or {})}
    auto = bool(s["auto_tune"])
    return {
        "model_id": gr.update(value=s["model_id"]),
        "px_preset": gr.update(value=s["px_preset"]),
        "auto_tune": gr.update(value=auto),
        "temperature": gr.update(
            value=s["temperature"], interactive=not auto),
        "top_p": gr.update(value=s["top_p"], interactive=not auto),
        "max_tokens": gr.update(value=s["max_tokens"]),
        "rep_p": gr.update(value=s["rep_p"], interactive=not auto),
        "px_gamma": gr.update(
            value=s["px_gamma"], interactive=not auto),
        "relay_sign": gr.update(value=s["relay_sign"]),
        "relay_alpha": gr.update(value=s["relay_alpha"]),
        "relay_layer": gr.update(value=s["relay_layer"]),
        "system_profile": gr.update(value=s["system_profile"]),
        "system_prompt_text": gr.update(value=s["system_prompt_text"]),
        "tts_engine": gr.update(value=s["tts_engine"]),
        "tts_sample_rate": gr.update(value=s["tts_sample_rate"]),
        "tts_auto": gr.update(value=s["tts_auto"]),
    }


class SettingsDebouncer:
    """Batch widget-change events through one ``threading.Timer``.

    Usage
    -----
    >>> debouncer = SettingsDebouncer(
    ...     session_id_getter=lambda: current_state.value,
    ...     on_save=lambda sid, patch: sessions.update_settings(sid, **patch),
    ...     delay_ms=400,
    ... )
    >>> debouncer.schedule(temperature=0.9)
    >>> # ... 400ms later, if no further events, on_save("abc", {"temperature": 0.9})

    Thread-safety
    -------------
    All public methods are safe to call from any thread. The timer
    callback runs on a daemon thread (so the process can exit even if
    a flush is pending).
    """

    def __init__(self,
                 session_id_getter: Callable[[], Optional[str]],
                 on_save: Callable[[str, Dict[str, Any]], None],
                 delay_ms: int = 400) -> None:
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._pending: Dict[str, Any] = {}
        self._get_sid = session_id_getter
        self._on_save = on_save
        self._delay_s = max(delay_ms, 0) / 1000.0

    def schedule(self, **kwargs: Any) -> None:
        """Queue ``kwargs`` to be written after ``delay_ms`` of silence.

        Repeated calls before the timer fires are merged into a single
        ``_pending`` dict (last-write-wins per key).
        """
        with self._lock:
            self._pending.update(kwargs)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(
                self._delay_s, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def flush_now(self) -> None:
        """Cancel any pending timer and flush immediately. No-op if
        ``_pending`` is empty. Used by ``handle_load_saved`` (cancel
        writes for the old session before switching) and by tests."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            pending = self._pending.copy()
            self._pending.clear()
        if pending:
            sid = self._get_sid()
            if sid:
                self._on_save(sid, pending)

    def _flush(self) -> None:
        with self._lock:
            pending = self._pending.copy()
            self._pending.clear()
            self._timer = None
        if not pending:
            return
        sid = self._get_sid()
        if sid:
            self._on_save(sid, pending)
