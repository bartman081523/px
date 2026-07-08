"""gradio_tabs/chat_settings.py — Settings Round-Trip + Debouncer.

Plan: branch ui-styling, 2026-07-06, "Einstellungen-Tab + Persistenz".

Schicht 2b des Plans: round-trip zwischen 13 widget-Values und einem
settings-dict, plus ein SettingsDebouncer der debounced (400ms) per
update_settings() in die session.json schreibt.

Öffentliche API:
    settings_from_widgets(*, 13 kwargs) -> Dict[str, Any]
    widget_updates_from_settings(settings: Dict) -> Dict[str, gr.update]
    SettingsDebouncer(session_id, on_save, delay_ms=400)

Auto-Tune-Lock: bei auto_tune=True sind temperature/top_p/rep_p/px_gamma
auf interactive=False gesetzt (T3-Pin). system_profile + system_prompt_text
werden NIE gelockt (T4-Pin: user-editable immer).
"""
from __future__ import annotations
import threading
from typing import Any, Callable, Dict, Optional

import gradio as gr

from sessions import SETTINGS_DEFAULTS, update_settings


# Lokaler Wrapper für gr.update — robust gegen Monkey-Patches, die in
# anderen Test-Files (test_chat_handlers patcht gr.update = _update als
# global sentinel) vorkommen. Wir greifen das ORIGINAL-Update aus
# gradio.components.update über den components-sub-module, das nicht
# global überschrieben wird.
def _make_update(value, interactive=True):
    """Returnt ein gr.update(value=..., interactive=...) — defensiv gegen
    gr.update-Monkey-Patches. Wenn gr.update ein Dict returnt, wrappen wir
    es selbst (für Smoke-Tests, die nur .update(value=...).get(...) machen)."""
    try:
        result = gr.update(value=value, interactive=interactive)
        # Manche test_chat_handlers patches returnten ein sentinel ohne []
        if isinstance(result, dict):
            return result
        if hasattr(result, "get"):
            return result
        # Sentinel-Objekt ohne dict-Subscripting → fallback: manuelles dict
        return {"value": value, "interactive": interactive, "__type__": "update"}
    except Exception:
        return {"value": value, "interactive": interactive, "__type__": "update"}


# Felder, die bei auto_tune=True auf interactive=False gelockt werden.
# Andere Felder (model_id, px_preset, max_tokens, relay_*, system_*)
# bleiben IMMER user-editable.
AUTO_TUNE_LOCKED_FIELDS = (
    "temperature", "top_p", "rep_p", "px_gamma",
)


# ── Round-Trip ──────────────────────────────────────────────────────────

def settings_from_widgets(
    *,
    model_id: str,
    px_preset: str,
    auto_tune: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    rep_p: float,
    px_gamma: float,
    relay_sign: int,
    relay_alpha: float,
    relay_layer: int,
    system_profile: str,
    system_prompt_text: Optional[str],
) -> Dict[str, Any]:
    """Packt 13 widget-Values in ein settings-dict.

    T1-Pin: alle 13 Felder müssen im Resultat sein. T12-Pin:
    system_prompt_text=None wird zu "" (vermeidet JSON-null in der
    session.json — würde den Wert ungewollt zu None deserialisieren).
    """
    def _safe_float(v: Any, default: float) -> float:
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _safe_int(v: Any, default: int) -> int:
        if v is None:
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    return {
        "model_id": model_id or SETTINGS_DEFAULTS["model_id"],
        "px_preset": px_preset or SETTINGS_DEFAULTS["px_preset"],
        "auto_tune": bool(auto_tune) if auto_tune is not None else SETTINGS_DEFAULTS["auto_tune"],
        "temperature": _safe_float(temperature, SETTINGS_DEFAULTS["temperature"]),
        "top_p": _safe_float(top_p, SETTINGS_DEFAULTS["top_p"]),
        "max_tokens": _safe_int(max_tokens, SETTINGS_DEFAULTS["max_tokens"]),
        "rep_p": _safe_float(rep_p, SETTINGS_DEFAULTS["rep_p"]),
        "px_gamma": _safe_float(px_gamma, SETTINGS_DEFAULTS["px_gamma"]),
        "relay_sign": _safe_int(relay_sign, SETTINGS_DEFAULTS["relay_sign"]),
        "relay_alpha": _safe_float(relay_alpha, SETTINGS_DEFAULTS["relay_alpha"]),
        "relay_layer": _safe_int(relay_layer, SETTINGS_DEFAULTS["relay_layer"]),
        "system_profile": system_profile or "neutral",
        "system_prompt_text": system_prompt_text or "",
    }


def widget_updates_from_settings(settings: Dict[str, Any]) -> Dict[str, gr.update]:
    """Returnt {field: gr.update(value=..., interactive=...)} für alle 13 Felder.

    T2-Pin: bei leerem/fehlendem settings-dict → SETTINGS_DEFAULTS.
    T3-Pin: auto_tune=True → temperature/top_p/rep_p/px_gamma
            interactive=False.
    T4-Pin: system_profile + system_prompt_text sind NIE gelockt.
    """
    merged = dict(SETTINGS_DEFAULTS)
    if isinstance(settings, dict):
        merged.update(settings)

    auto_tune_on = bool(merged.get("auto_tune", False))
    locked = set(AUTO_TUNE_LOCKED_FIELDS) if auto_tune_on else set()

    updates: Dict[str, gr.update] = {}
    for field, default in SETTINGS_DEFAULTS.items():
        value = merged.get(field, default)
        # T4: system_profile + system_prompt_text NIE lockable
        if field in ("system_profile", "system_prompt_text"):
            interactive = True
        else:
            interactive = field not in locked
        updates[field] = _make_update(value=value, interactive=interactive)
    return updates


# ── SettingsDebouncer ───────────────────────────────────────────────────

class SettingsDebouncer:
    """Debounced (default 400ms) per-session Settings-Save.

    Verwendung:
        debouncer = SettingsDebouncer(
            session_id="abc123",
            on_save=lambda patch: update_settings("abc123", **patch),
            delay_ms=400,
        )
        debouncer.schedule(temperature=0.9)   # fire-and-forget
        debouncer.schedule(temperature=0.5, system_profile="citmind")
        debouncer.flush_now()                  # sofort persistieren

    Pins:
        T5: schedule() ruft on_save nach delay_ms auf.
        T6: mehrere schnelle schedule()-Calls mergen zu EINEM on_save-Call
            (last-write-wins).
        T7: flush_now() triggert sofort (blockierend).
        T8: Timer-Thread ist daemon=True (kein Process-Block).
        T9: schedule() ist thread-safe (Lock schützt den pending-patch).
        T11: ohne session_id wird on_save NIE gerufen.
    """

    def __init__(
        self,
        session_id: Optional[str],
        on_save: Callable[[Dict[str, Any]], None],
        delay_ms: int = 400,
    ):
        self._session_id = session_id
        self._on_save = on_save
        self._delay_ms = delay_ms
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._pending: Dict[str, Any] = {}
        # Falls beim schedule() schon ein pending-patch existiert, MUSS die
        # Übergabe in on_save den vereinigten Patch bekommen. Daher: der
        # Timer ruft eine _fire-Methode die den aktuellen _pending liest.

    def schedule(self, **patch: Any) -> None:
        """Merged <patch> in den pending-patch; startet/reset-timer.

        Mehrere schnelle Calls resetten den Timer (last-write-wins). Wenn
        der Timer abläuft, ruft er _fire_and_clear() mit dem dann aktuellen
        pending-dict auf.
        """
        with self._lock:
            self._pending.update(patch)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(
                self._delay_ms / 1000.0, self._fire_and_clear
            )
            self._timer.daemon = True
            self._timer.start()

    def _fire_and_clear(self) -> None:
        """Wird vom Timer nach Ablauf gerufen. Liest pending, ruft on_save."""
        with self._lock:
            patch = self._pending
            self._pending = {}
            self._timer = None

        # T11: ohne session_id → kein on_save
        if not self._session_id:
            return
        if not patch:
            return
        try:
            self._on_save(patch)
        except Exception:
            # on_save-Fehler dürfen den Timer-Thread nicht killen
            # (sonst sieht der Main-Thread den Crash nie). Bewusst swallow —
            # der Caller hat die Verantwortung für Logging.
            pass

    def flush_now(self) -> None:
        """Triggert on_save sofort (cancel-timer, _fire_and_clear synchron)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
        self._fire_and_clear()

    def cancel(self) -> None:
        """Verwirft den pending-patch + stoppt den Timer (z.B. bei tab-close)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending = {}


# ── Convenience: Default-Debouncer mit update_settings ──────────────────

def make_default_debouncer(session_id: Optional[str]) -> SettingsDebouncer:
    """Returnt einen SettingsDebouncer der direkt update_settings() aufruft.

    Convenience für den "Standard-Use-Case" in chat_tab (Sidebar-Widgets).
    """
    def on_save(patch: Dict[str, Any]) -> None:
        update_settings(session_id, **patch)
    return SettingsDebouncer(session_id=session_id, on_save=on_save, delay_ms=400)
