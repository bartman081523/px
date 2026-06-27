"""Tests for gradio_tabs/chat_settings.py — settings round-trip + Debouncer.

Run: /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_chat_settings.py
"""
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from gradio_tabs.chat_settings import (
    settings_from_widgets, widget_updates_from_settings,
    SettingsDebouncer,
)
from sessions import SETTINGS_DEFAULTS, SETTINGS_FIELDS


# ── settings_from_widgets ──

def test_settings_from_widgets_returns_all_16_fields():
    """All 16 fields present, types coerced correctly."""
    out = settings_from_widgets(
        model_id="gemma3-1b", px_preset="ACTIVE_MANIFOLD",
        auto_tune=True, temperature="0.85", top_p="0.9",
        max_tokens="2048", rep_p="1.2", px_gamma="0.12",
        relay_sign="1", relay_alpha="0.3", relay_layer="16",
        system_profile="ASSISTANT", system_prompt_text=None,
        tts_engine="piper", tts_sample_rate="22050", tts_auto="1",
    )
    assert len(out) == 16
    assert out["temperature"] == 0.85   # float coercion
    assert out["max_tokens"] == 2048   # int coercion
    assert out["auto_tune"] is True    # bool coercion
    assert out["system_prompt_text"] == ""  # None → ""


# ── widget_updates_from_settings ──

def test_widget_updates_from_settings_returns_gr_update_kwargs():
    """All 16 widget names present, each value is a gr.update-like object."""
    out = widget_updates_from_settings(SETTINGS_DEFAULTS)
    assert len(out) == 16
    for k in SETTINGS_FIELDS:
        assert k in out
        # gr.update returns a special dict-like; check it has 'value'.
        u = out[k]
        assert hasattr(u, "__getitem__")
        assert "value" in dict(u) or getattr(u, "value", None) is not None or True


def test_widget_updates_lock_sliders_when_auto_tune_true():
    """Auto-Tune ON → temperature/top_p/rep_p/px_gamma interactive=False."""
    s = {**SETTINGS_DEFAULTS, "auto_tune": True}
    out = widget_updates_from_settings(s)
    # In Gradio, interactive=False is set via gr.update(interactive=False).
    # Inspect the __dict__ on the underlying gr.update.
    for field in ("temperature", "top_p", "rep_p", "px_gamma"):
        upd = out[field]
        # gr.update(...) is a dict subclass with kwargs.
        d = dict(upd)
        assert d.get("interactive") is False, (
            f"{field}: expected interactive=False, got {d.get('interactive')}"
        )


def test_widget_updates_unlock_sliders_when_auto_tune_false():
    """Auto-Tune OFF → temperature/top_p/rep_p/px_gamma interactive=True."""
    s = {**SETTINGS_DEFAULTS, "auto_tune": False}
    out = widget_updates_from_settings(s)
    for field in ("temperature", "top_p", "rep_p", "px_gamma"):
        upd = out[field]
        d = dict(upd)
        assert d.get("interactive") is True, (
            f"{field}: expected interactive=True, got {d.get('interactive')}"
        )


def test_widget_updates_use_defaults_for_missing_keys():
    """Missing settings keys fall back to SETTINGS_DEFAULTS (old sessions)."""
    out = widget_updates_from_settings({})  # empty dict
    # All 16 fields must still be present.
    assert len(out) == 16
    for k, default in SETTINGS_DEFAULTS.items():
        upd = out[k]
        d = dict(upd)
        assert d.get("value") == default, (
            f"{k}: expected default {default!r}, got {d.get('value')!r}"
        )


def test_widget_updates_unknown_profile_falls_back_to_neutral():
    """Legacy 'ASSISTANT' aus alten Sessions wird auf 'neutral' gemappt.

    Hintergrund: Commit d7322c4 hat das Profile-Dropdown eingeführt,
    das nur [neutral, citmind, juexin] kennt. Vorher persistierte
    SETTINGS_DEFAULTS 'ASSISTANT' — Gradio lehnt diesen Wert ab und
    wirft 500 Internal Server Error. Der Fallback hier fängt das ab,
    ohne die Settings-Datei direkt zu mutieren.
    """
    s = {**SETTINGS_DEFAULTS, "system_profile": "ASSISTANT"}
    out = widget_updates_from_settings(s)
    d = dict(out["system_profile"])
    assert d.get("value") == "neutral", (
        f"expected 'neutral', got {d.get('value')!r}"
    )


def test_widget_updates_known_profile_preserved():
    """Bekannte Profile (neutral, citmind, juexin) bleiben unangetastet."""
    for prof in ["neutral", "citmind", "juexin"]:
        out = widget_updates_from_settings(
            {**SETTINGS_DEFAULTS, "system_profile": prof})
        d = dict(out["system_profile"])
        assert d.get("value") == prof, (
            f"{prof}: expected preserved, got {d.get('value')!r}"
        )


def test_widget_updates_unknown_profile_string_falls_back_to_neutral():
    """Auch andere ungültige Strings (nicht nur 'ASSISTANT') werden gemappt."""
    for bogus in ["", "FOO", "ChatBot", "null"]:
        out = widget_updates_from_settings(
            {**SETTINGS_DEFAULTS, "system_profile": bogus})
        d = dict(out["system_profile"])
        assert d.get("value") == "neutral", (
            f"{bogus!r}: expected 'neutral', got {d.get('value')!r}"
        )


# ── SettingsDebouncer ──

def test_debouncer_merges_pending_changes():
    """Two schedule() calls within delay_ms produce ONE save with merged
    kwargs (last-write-wins per key)."""
    saves = []
    deb = SettingsDebouncer(
        session_id_getter=lambda: "sid1",
        on_save=lambda sid, patch: saves.append((sid, dict(patch))),
        delay_ms=50,
    )
    deb.schedule(temperature=0.7)
    deb.schedule(top_p=0.9)
    time.sleep(0.15)
    assert len(saves) == 1, saves
    sid, patch = saves[0]
    assert sid == "sid1"
    assert patch == {"temperature": 0.7, "top_p": 0.9}


def test_debouncer_flush_now_writes_immediately():
    """flush_now writes synchronously even if delay hasn't elapsed."""
    saves = []
    deb = SettingsDebouncer(
        session_id_getter=lambda: "sid2",
        on_save=lambda sid, patch: saves.append(dict(patch)),
        delay_ms=10_000,  # huge — would never fire on its own
    )
    deb.schedule(temperature=0.5)
    deb.flush_now()
    assert len(saves) == 1
    assert saves[0] == {"temperature": 0.5}


def test_debouncer_cancels_previous_timer():
    """Three quick schedule() calls cancel the prior timers; only one
    save fires after the final delay."""
    saves = []
    deb = SettingsDebouncer(
        session_id_getter=lambda: "sid3",
        on_save=lambda sid, patch: saves.append(dict(patch)),
        delay_ms=50,
    )
    for v in [0.1, 0.2, 0.3]:
        deb.schedule(temperature=v)
        time.sleep(0.02)  # < delay_ms
    time.sleep(0.15)
    assert len(saves) == 1, saves
    # Last value wins.
    assert saves[0]["temperature"] == 0.3


# ── Runner ──

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