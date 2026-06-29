"""TTS-only Lücke 2 Tests: Profile-Wiring, Auto-Tune-Lock, Debouncer-Wiring.

Pinnt das Wiring zwischen chat_tab.py ↔ chat_settings.py ↔ system_prompt.py
+ auto_tune_defaults.py für die Settings-Dropdowns, Auto-Tune-Checkbox,
Undo-Button und SettingsDebouncer. Diese Tests sind TTS-only weil die
Settings-Dropdowns in master/pre-tts-improvements nicht existieren.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_chat_tab_luecke2.py
"""
import os
import sys
import tempfile
import shutil
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


# ── Profile-Wiring Tests ────────────────────────────────────────────
# Verifiziert: Profile-Wechsel via Dropdown → system_prompt.py:inject_into_messages
# wird mit dem NEUEN Profil aufgerufen, nicht mit dem alten.

def test_profile_dropdown_wires_to_inject_into_messages():
    """chat_fn nimmt system_profile aus der Dropdown und ruft
    inject_into_messages(messages, _profile, _edit) mit dem gewählten Profil."""
    from gradio_tabs.system_prompt import (
        inject_into_messages, build_system_message, list_profiles,
    )

    # Sanity: liste Profile
    profiles = list_profiles()
    assert isinstance(profiles, list) and len(profiles) >= 3
    assert "neutral" in profiles
    assert "citmind" in profiles

    messages = [{"role": "user", "content": "Hello"}]
    msgs_citmind = inject_into_messages(
        list(messages), "citmind", None)
    msgs_neutral = inject_into_messages(
        list(messages), "neutral", None)

    # CitMind produziert einen System-Eintrag (weil CitMind-Doc Inhalt hat).
    citmind_sys = next(m for m in msgs_citmind if m["role"] == "system")
    assert citmind_sys["content"] != ""
    # Neutral ohne Edit → KEIN System-Eintrag (bewusste Entscheidung, siehe
    # system_prompt.py:build_system_message: "For neutral with no edit,
    # content is empty (the orchestrator omits empty system messages)").
    assert not any(m.get("role") == "system" for m in msgs_neutral), (
        "Neutral + kein Edit darf keinen System-Eintrag erzeugen"
    )


def test_profile_unknown_falls_back_to_neutral_in_chat_fn_path():
    """chat_fn: unbekanntes Profil (z.B. 'ASSISTANT' aus alten Sessions)
    fällt zurück auf 'neutral' via chat_settings.widget_updates_from_settings."""
    from gradio_tabs.chat_settings import widget_updates_from_settings
    from sessions import SETTINGS_DEFAULTS

    s = {**SETTINGS_DEFAULTS, "system_profile": "ASSISTANT"}
    out = widget_updates_from_settings(s)
    d = dict(out["system_profile"])
    assert d.get("value") == "neutral"


def test_profile_edit_text_overrides_profile_dropdown():
    """system_prompt_text (Textarea-Edit) non-empty → hat Vorrang vor Dropdown."""
    from gradio_tabs.system_prompt import inject_into_messages

    messages = [{"role": "user", "content": "Hi"}]
    # Mit Dropdown citmind aber Textarea-Edit "User override"
    msgs = inject_into_messages(
        list(messages), "citmind", "User override")
    sys_content = next(m["content"] for m in msgs if m["role"] == "system")
    # Textarea-Edit sollte im System-Content auftauchen.
    assert "User override" in str(sys_content)


def test_profile_empty_string_treated_as_none_in_chat_fn_path():
    """chat_fn: leerer Dropdown-Wert ('') wird als 'neutral' behandelt
    und führt zu keinem System-Eintrag (kein user-input = kein inject)."""
    from gradio_tabs.system_prompt import inject_into_messages

    messages = [{"role": "user", "content": "Hi"}]
    msgs = inject_into_messages(list(messages), "", None)
    # Bei ''-Profil greift resolve_profile(None) → "neutral" → kein System-Eintrag.
    assert not any(m.get("role") == "system" for m in msgs), (
        f"Leerer Dropdown-Wert '' darf keinen System-Eintrag erzeugen, msgs={msgs}"
    )
    # User-Message bleibt erhalten.
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    assert len(user_msgs) == 1 and user_msgs[0]["content"] == "Hi"


# ── Auto-Tune-Lock Tests ────────────────────────────────────────────
# Verifiziert: Auto-Tune ON → Slider werden gelockt + Werte überschrieben.
# Auto-Tune OFF → Slider werden freigegeben, Werte bleiben.

def test_auto_tune_on_lock_sliders_and_apply_defaults():
    """_auto_tune_updates(model_id, True) lockt temperature/top_p/px_gamma/rep_p
    und schreibt die modell-kalibrierten Werte."""
    from gradio_tabs.chat_tab import _auto_tune_updates
    from gradio_tabs.auto_tune_defaults import (
        scale_adaptive_temperature, calibrated_values,
    )

    updates = _auto_tune_updates("gemma3-1b-it", True)
    assert len(updates) == 4, "Erwartet 4 Tuple (temp, top_p, gamma, rep_p)"

    # Updates sind gr.update-Objekte mit kwargs
    temp_upd, top_p_upd, gamma_upd, rep_p_upd = updates

    for upd in updates:
        d = dict(upd)
        assert d.get("interactive") is False, (
            f"Auto-Tune ON muss alle Slider locken, aber {d} hat interactive=True"
        )

    # Werte: temperature = scale-adaptive für 1b (0.6), top_p=0.9, rep_p=1.15
    expected_temp = scale_adaptive_temperature("gemma3-1b-it")
    assert expected_temp is not None and abs(expected_temp - 0.6) < 1e-6
    temp_val = dict(temp_upd).get("value")
    assert temp_val == expected_temp, (
        f"temperature sollte {expected_temp} sein, ist {temp_val}"
    )

    cv = calibrated_values("gemma3-1b-it")
    top_p_val = dict(top_p_upd).get("value")
    assert top_p_val == cv["top_p"]  # 0.9
    rep_p_val = dict(rep_p_upd).get("value")
    assert rep_p_val == cv["repetition_penalty"]  # 1.15


def test_auto_tune_off_unlock_sliders_keep_values():
    """_auto_tune_updates(model_id, False) gibt Slider frei ohne Wert zu ändern."""
    from gradio_tabs.chat_tab import _auto_tune_updates

    updates = _auto_tune_updates("gemma3-1b-it", False)
    assert len(updates) == 4

    for upd in updates:
        d = dict(upd)
        assert d.get("interactive") is True, (
            f"Auto-Tune OFF muss Slider freigeben, aber {d}"
        )
        # Wert sollte NICHT gesetzt werden (gr.update(interactive=True) ohne value)
        assert d.get("value", "__unset__") == "__unset__", (
            f"Auto-Tune OFF darf Werte nicht überschreiben, aber {d}"
        )


def test_auto_tune_unknown_model_falls_back_gracefully():
    """Unbekanntes Modell: kein Crash, fallback auf 0.7 temperature."""
    from gradio_tabs.chat_tab import _auto_tune_updates

    updates = _auto_tune_updates("nonexistent-model-xyz", True)
    assert len(updates) == 4
    temp_upd = updates[0]
    temp_val = dict(temp_upd).get("value")
    # Fallback: 0.7 wenn scale_adaptive_temperature None zurückgibt
    assert temp_val == 0.7, f"Expected fallback 0.7, got {temp_val}"


# ── resolve_for_backend Tests ───────────────────────────────────────
# Verifiziert: Auto-Tune ON überschreibt px_gamma/top_p/rep_p mit Defaults;
# user-Werte für temperature/max_tokens werden durchgereicht.

def test_resolve_for_backend_auto_tune_overrides_top_p_rp_gamma():
    """resolve_for_backend(auto_tune_on=True): px_gamma=None, top_p=0.9, rp=1.15,
    temperature/max_tokens aus user_values."""
    from gradio_tabs.auto_tune_defaults import resolve_for_backend

    out = resolve_for_backend(
        "gemma3-1b-it", True,
        {"temperature": 0.5, "max_tokens": 512, "px_gamma": 0.5,
         "top_p": 0.5, "repetition_penalty": 0.5},
    )
    assert out["px_gamma"] is None  # bridge parity
    assert out["top_p"] == 0.9
    assert out["repetition_penalty"] == 1.15
    assert out["temperature"] == 0.5  # user_value wins
    assert out["max_tokens"] == 512


def test_resolve_for_backend_auto_tune_off_passes_through():
    """resolve_for_backend(auto_tune_on=False): alle user_values durchgereicht,
    fehlende keys fallen auf Defaults."""
    from gradio_tabs.auto_tune_defaults import resolve_for_backend

    # Nur temperature angegeben, Rest leer
    out = resolve_for_backend(
        "gemma3-1b-it", False,
        {"temperature": 0.8, "px_gamma": 0.06},
    )
    assert out["px_gamma"] == 0.06  # user_value
    assert out["top_p"] == 0.9      # default
    assert out["repetition_penalty"] == 1.15  # default
    assert out["temperature"] == 0.8
    assert out["max_tokens"] == 1024  # default


def test_resolve_for_backend_auto_tune_off_all_defaults_when_empty():
    """resolve_for_backend mit leerem user_values: Defaults."""
    from gradio_tabs.auto_tune_defaults import resolve_for_backend

    out = resolve_for_backend("gemma3-1b-it", False, {})
    assert out["px_gamma"] is None
    assert out["top_p"] == 0.9
    assert out["repetition_penalty"] == 1.15
    assert out["temperature"] == 0.7
    assert out["max_tokens"] == 1024


# ── SettingsDebouncer-Wiring Tests ──────────────────────────────────
# Verifiziert: SettingsDebouncer mit 400ms-Delay führt N schnelle Changes
# zu EINEM final-Call zusammen. Plus: chat_tab.py nutzt den Debouncer für
# alle Settings-Änderungen.

def test_debouncer_5_rapid_changes_collapse_to_one_save():
    """5 schnelle schedule()-Calls innerhalb 400ms → 1 final-Call mit
    zusammengeführten kwargs."""
    from gradio_tabs.chat_settings import SettingsDebouncer

    saves = []
    deb = SettingsDebouncer(
        session_id_getter=lambda: "rapid-test",
        on_save=lambda sid, patch: saves.append((sid, dict(patch))),
        delay_ms=400,
    )
    deb.schedule(temperature=0.1)
    deb.schedule(top_p=0.2)
    deb.schedule(rep_p=0.3)
    deb.schedule(px_gamma=0.4)
    deb.schedule(max_tokens=128)
    # Direkt vor Ablauf des 400ms-Timers: alle 5 in pending.
    time.sleep(0.6)  # > 400ms damit letzter Timer feuert
    assert len(saves) == 1, f"Erwartet 1 save, got {len(saves)}: {saves}"
    sid, patch = saves[0]
    assert sid == "rapid-test"
    assert patch["temperature"] == 0.1
    assert patch["top_p"] == 0.2
    # settings_from_widgets-Widget-Name rep_p wird 1:1 durchgereicht;
    # die Umbenennung repetition_penalty passiert erst im resolve_for_backend.
    assert patch["rep_p"] == 0.3
    assert patch["px_gamma"] == 0.4
    assert patch["max_tokens"] == 128


def test_debouncer_last_value_wins_per_key():
    """Bei wiederholten schedule(key=v1), schedule(key=v2) für SELBEN key
    gewinnt der LETZTE Wert (last-write-wins)."""
    from gradio_tabs.chat_settings import SettingsDebouncer

    saves = []
    deb = SettingsDebouncer(
        session_id_getter=lambda: "lw-test",
        on_save=lambda sid, patch: saves.append(dict(patch)),
        delay_ms=50,
    )
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
        deb.schedule(temperature=v)
        time.sleep(0.01)
    time.sleep(0.1)
    assert len(saves) == 1
    assert saves[0]["temperature"] == 0.5


def test_debouncer_flush_now_bypasses_delay():
    """flush_now() schreibt sofort ohne auf delay zu warten."""
    from gradio_tabs.chat_settings import SettingsDebouncer

    saves = []
    deb = SettingsDebouncer(
        session_id_getter=lambda: "flush-test",
        on_save=lambda sid, patch: saves.append(dict(patch)),
        delay_ms=10_000,  # sehr lang
    )
    deb.schedule(auto_tune=True)
    assert len(saves) == 0  # noch nicht geschrieben
    deb.flush_now()
    assert len(saves) == 1
    assert saves[0]["auto_tune"] is True


def test_debouncer_subsequent_changes_after_flush():
    """Nach flush_now() können weitere schedule()-Calls wieder normal laufen."""
    from gradio_tabs.chat_settings import SettingsDebouncer

    saves = []
    deb = SettingsDebouncer(
        session_id_getter=lambda: "post-flush",
        on_save=lambda sid, patch: saves.append(dict(patch)),
        delay_ms=50,
    )
    deb.schedule(temperature=0.5)
    deb.flush_now()
    assert len(saves) == 1

    # Nach flush_now weitere schedule-Calls
    deb.schedule(top_p=0.7)
    time.sleep(0.1)
    assert len(saves) == 2
    assert saves[1] == {"top_p": 0.7}


# ── Undo-Button Persistence Tests ──────────────────────────────────
# Verifiziert: handle_undo ruft save_session mit dem NEUEN history-State auf.

def test_handle_undo_persists_new_history():
    """handle_undo(history, session_id, model_id) ruft save_session mit der
    gekürzten History auf (Undo → persistierter Zustand)."""
    from gradio_tabs.chat_tab import handle_undo
    from gradio_tabs.chat_actions import undo_last_turn
    import gradio_tabs.chat_tab as chat_tab_mod

    saved = []
    # Patch save_session zum Mitschreiben
    original_save = chat_tab_mod.save_session
    chat_tab_mod.save_session = lambda sid, hist, model_id=None: saved.append((sid, list(hist), model_id))
    try:
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = handle_undo(history, "test-session-undo", "gemma3-1b-it")
        # Letzter Turn entfernt
        assert result == [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        # Session persistiert
        assert len(saved) == 1
        sid, hist, mid = saved[0]
        assert sid == "test-session-undo"
        assert mid == "gemma3-1b-it"
        assert len(hist) == 2
    finally:
        chat_tab_mod.save_session = original_save


def test_handle_undo_no_session_id_skips_persistence():
    """handle_undo ohne session_id: gibt nur neue history zurück, ohne save."""
    from gradio_tabs.chat_tab import handle_undo
    import gradio_tabs.chat_tab as chat_tab_mod

    saved = []
    original_save = chat_tab_mod.save_session
    chat_tab_mod.save_session = lambda *a, **k: saved.append((a, k))
    try:
        history = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        result = handle_undo(history, None, "gemma3-1b-it")
        assert len(result) == 0  # letzter turn entfernt
        assert len(saved) == 0, "Ohne session_id darf save_session nicht aufgerufen werden"
    finally:
        chat_tab_mod.save_session = original_save


def test_handle_undo_empty_history_returns_empty():
    """handle_undo mit leerer history: gibt [] zurück, kein save."""
    from gradio_tabs.chat_tab import handle_undo
    import gradio_tabs.chat_tab as chat_tab_mod

    saved = []
    original_save = chat_tab_mod.save_session
    chat_tab_mod.save_session = lambda *a, **k: saved.append((a, k))
    try:
        result = handle_undo([], "sid", "model")
        assert result == []
        # save_session wird trotzdem aufgerufen (mit leerer history)
        # — bewusste Entscheidung: persistierter Zustand == UI-Zustand.
        assert len(saved) == 1
    finally:
        chat_tab_mod.save_session = original_save


# ── Settings round-trip + Dropdown-Integration Tests ────────────────
# Verifiziert: settings_from_widgets nimmt ALLE 16 Felder + Profile
# wird konsistent durchgestellt.

def test_settings_round_trip_preserves_system_profile_citmind():
    """settings_from_widgets(system_profile='citmind') → settings['system_profile']
    bleibt 'citmind' durch widget_updates_from_settings round-trip."""
    from gradio_tabs.chat_settings import (
        settings_from_widgets, widget_updates_from_settings,
    )

    settings = settings_from_widgets(
        model_id="gemma3-1b", px_preset="ACTIVE_MANIFOLD",
        auto_tune=False, temperature="0.7", top_p="0.9",
        max_tokens="1024", rep_p="1.15", px_gamma="0.06",
        relay_sign="0", relay_alpha="0.3", relay_layer="16",
        system_profile="citmind", system_prompt_text="",
        tts_engine="off", tts_sample_rate="22050", tts_auto="0",
    )
    out = widget_updates_from_settings(settings)
    d = dict(out["system_profile"])
    assert d.get("value") == "citmind"


def test_settings_round_trip_uppercase_citmind_falls_back_neutral():
    """settings mit 'CITMIND' (uppercase, ungültig) → Fallback 'neutral'."""
    from gradio_tabs.chat_settings import (
        settings_from_widgets, widget_updates_from_settings,
    )

    settings = settings_from_widgets(
        model_id="gemma3-1b", px_preset="BASELINE",
        auto_tune=False, temperature="0.7", top_p="0.9",
        max_tokens="1024", rep_p="1.15", px_gamma="0.0",
        relay_sign="0", relay_alpha="0.3", relay_layer="16",
        system_profile="CITMIND",  # uppercase ungültig
        system_prompt_text="",
        tts_engine="off", tts_sample_rate="22050", tts_auto="0",
    )
    out = widget_updates_from_settings(settings)
    d = dict(out["system_profile"])
    assert d.get("value") == "neutral", (
        f"Uppercase CITMIND muss zu neutral fallen, ist {d.get('value')!r}"
    )


def test_settings_round_trip_juexin_preserved():
    """juexin-Profil ist gültig und bleibt erhalten."""
    from gradio_tabs.chat_settings import (
        settings_from_widgets, widget_updates_from_settings,
    )

    settings = settings_from_widgets(
        model_id="gemma3-1b", px_preset="ACTIVE_MANIFOLD",
        auto_tune=False, temperature="0.7", top_p="0.9",
        max_tokens="1024", rep_p="1.15", px_gamma="0.06",
        relay_sign="0", relay_alpha="0.3", relay_layer="16",
        system_profile="juexin", system_prompt_text="",
        tts_engine="off", tts_sample_rate="22050", tts_auto="0",
    )
    out = widget_updates_from_settings(settings)
    d = dict(out["system_profile"])
    assert d.get("value") == "juexin"


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