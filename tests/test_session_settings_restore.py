"""tests/test_session_settings_restore.py — Restore + Defaults-Wiring.

Plan 2026-10-05 (User-Request): Session-Einstellungen (Modell, px_preset,
Parameter, Relay, System-Prompt) werden in der session.json gespeichert
und beim Session-Load in die Widgets gerendert; Modell-/Preset-Wechsel
wenden automatisch die per-Model-Defaults an (Injektions-Layer inkl.
Slider-Bounds, sign, alpha, px_gamma).

Pinnt:
  R1:   restore ohne Session/settings → 14 no-op updates
  R2:   restore voller Settings (Roundtrip via settings_from_widgets) —
        Reihenfolge = SETTINGS_WIDGET_FIELDS, auto_tune=False → nicht
        gelockt
  R3:   auto_tune FEHLT → NICHT gelockt (Deviation gegen den chat_settings-
        T2-Pin: die UI hat kein auto_tune-Widget — fehlender Key darf den
        Parametern nie die Interaktivität nehmen)
  R4:   relay_layer-Bounds mitrestored (ternary L34 → maximum 64)
  R5:   Profil-Behavior-Load-Suppress: abweichendes Profil → 1 Token;
        on_profile_change_load_body konsumiert → gr.skip(); danach normal
  R5b:  gleiches Profil → KEIN Token (kein .change-Feuering → kein Leak)
  R6:   apply_px_defaults Suppress-Regel (settings matchen model+preset →
        6 no-ops, KEIN persist von Relay/Thinking-Feldern)
  R7:   apply_px_defaults: gemma3-1b-it → Artifact-Layer 21 / max 26 /
        sign 1 / alpha 0.30 / gamma 0.12 + Thinking-Widgets versteckt
        (gemma3 nicht thinking-capable, Phase 3)
  R8:   apply_px_defaults: minicpm5-1b → no-ops; persist NICHTS Relay-
        Feldes (relay nicht verfügbar)
  R9:   apply_px_defaults: unbekannte model_id → no-ops + Thinking hidden
        + persist model/preset
  R10:  chat_fn-Source-Pins: settings_from_widgets + 2× settings=chat_settings
        + strip-before-generate-Reihenfolge unangetastet
  R11:  build_chat_tab-Source-Pins: apply_px_defaults auf .input (beide
        Owner), restore_session_settings in beiden .then-Chains
  R12:  app.py-Source-Pins: 18er demo.load (inputs incl. system_profile)
  R13:  on_load → 18-Tupel; frische Session → 14 no-op updates
  R14:  restore bonsai LEGACY-partial (keine Thinking-Keys) → Modell-
        Template-Defaults (True + xhigh) + choices
  R15:  restore bonsai stored thinking=False → False (stored schlägt
        Template-Default)
  R16:  restore bonsai stored thinking_effort="medium" → medium
  R17:  restore bonsai stored thinking_effort="bogus" → Template-Default
        xhigh (kein invalid-value-Rendering)
  R18:  restore gemma4 → Checkbox False/visible, effort-Radio versteckt
        (kein Budget-Parameter im installierten Stack)
  R19:  restore nicht-capable (gemma3) mit stale thinking-Junk aus einem
        bonsai-Chat → beides visible=False OHNE value (Junk rutscht
        nicht ins UI)

Gr.update-shape: je nach Test-Reihenfolge kann das Modul unter dem
_MockSentinel-Patch aus test_chat_handlers laufen (gleicher pytest-Prozess)
— upd_kwargs() ist shape-agnostisch (dict ODER sentinel.kwargs).

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_session_settings_restore.py
"""
import os
import sys
import re
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sessions import save_session, load_session, SESSION_DIR
from gradio_tabs.chat_tab import (
    SETTINGS_WIDGET_FIELDS,
    restore_session_settings,
    apply_px_defaults,
    on_load,
    on_profile_change_load_body,
    _reset_profile_suppress_for_tests,
    _persist_setting_field,
)
from gradio_tabs.chat_settings import settings_from_widgets
from gradio_tabs.settings_persist import (
    flush_settings_save,
    _reset_persister_for_tests,
)


# --- Shape-agnostische update-Helfer -------------------------------------

def upd_kwargs(u):
    """gr.update-Dict ODER _MockSentinel(**kwargs) → kwargs-dict."""
    if isinstance(getattr(u, "kwargs", None), dict) and not isinstance(u, dict):
        return dict(u.kwargs)
    return {k: v for k, v in dict(u).items() if k != "__type__"}


def is_noop(u):
    """No-op = kein 'value'-Key."""
    return "value" not in upd_kwargs(u)


def _session_path(sid):
    return os.path.join(SESSION_DIR, f"{sid}.json")


def _wipe(sids):
    for sid in sids:
        p = _session_path(sid)
        if os.path.exists(p):
            os.unlink(p)
    _reset_persister_for_tests()
    _reset_profile_suppress_for_tests()


@pytest.fixture(autouse=True)
def _clean_state():
    _reset_persister_for_tests()
    _reset_profile_suppress_for_tests()
    yield
    _reset_persister_for_tests()
    _reset_profile_suppress_for_tests()


# --- R1: No-op-Restore ---------------------------------------------------

def test_r1_unknown_session_gives_14_noops():
    """R1: Session-Datei fehlt → 14 no-op updates (kein Reset auf Defaults)."""
    updates = restore_session_settings("r1_nosuch_session", "neutral")
    assert len(updates) == 14
    for u in updates:
        assert is_noop(u), upd_kwargs(u)


def test_r1b_empty_settings_gives_14_noops():
    """R1b: settings={} (legacy-Kompatibilitäts-Pfad) → ebenso no-op."""
    sid = "r1b_empty"
    _wipe([sid])
    try:
        save_session(sid, [], settings={})
        updates = restore_session_settings(sid, "neutral")
        assert len(updates) == 14
        for u in updates:
            assert is_noop(u), upd_kwargs(u)
    finally:
        _wipe([sid])


# --- R2/R3: Werte-Restore + auto_tune-Deyiation --------------------------

def _full_settings():
    # Phase 3: bonsai-27b — thinking-capable, damit R2 auch die beiden
    # Thinking-Widgets (value+visible+interactive) mitprüfen kann.
    return settings_from_widgets(
        model_id="ternary-bonsai-27b",
        px_preset="ACTIVE_MANIFOLD_RELAY",
        auto_tune=False,
        temperature=0.55,
        top_p=0.9,
        max_tokens=768,
        rep_p=1.2,
        px_gamma=0.1,
        thinking=True,
        thinking_effort="medium",
        relay_sign=-1,
        relay_alpha=0.45,
        relay_layer=34,
        system_profile="neutral",
        system_prompt_text="Eigener Frame.",
    )


def test_r2_full_settings_restore_values_in_order():
    """R2: 14 Outputs in SETTINGS_WIDGET_FIELDS-Reihenfolge mit den
    gespeicherten Werten; auto_tune=False → Slider interaktiv; bonsai →
    Thinking-Checkbox/Radio mit den gespeicherten Werten sichtbar."""
    sid = "r2_full"
    _wipe([sid])
    try:
        save_session(sid, [], model_id="ternary-bonsai-27b",
                     settings=_full_settings())
        updates = restore_session_settings(sid, "neutral")
        assert len(updates) == 14
        for field, u in zip(SETTINGS_WIDGET_FIELDS, updates):
            kw = upd_kwargs(u)
            assert "value" in kw, (field, kw)
            assert kw.get("interactive", False) is True, (field, kw)
        vals = [upd_kwargs(u)["value"] for u in updates]
        expect = ["ternary-bonsai-27b", "ACTIVE_MANIFOLD_RELAY", 0.55, 0.9,
                  768, 1.2, 0.1, True, "medium", -1, 0.45, 34, "neutral",
                  "Eigener Frame."]
        assert vals == expect
    finally:
        _wipe([sid])


def test_r3_missing_auto_tune_never_locks():
    """R3: settings OHNE auto_tune-Key → Slider NICHT interaktiv=False."""
    sid = "r3_noautotune"
    _wipe([sid])
    try:
        partial = {"model_id": "gemma3-1b-it", "px_preset": "ACTIVE_MANIFOLD",
                   "temperature": 0.6, "relay_layer": 21}
        save_session(sid, [], settings=partial)
        updates = restore_session_settings(sid, "neutral")
        kw_temper = upd_kwargs(updates[SETTINGS_WIDGET_FIELDS.index("temperature")])
        assert kw_temper.get("interactive") is True
        assert kw_temper["value"] == 0.6
    finally:
        _wipe([sid])


# --- R4: Slider-Bounds ----------------------------------------------------

def test_r4_relay_layer_bounds_restored():
    """R4: ternary L34 → value 34 UND maximum 64 (Slider-Init war 1..18/25)."""
    sid = "r4_bounds"
    _wipe([sid])
    try:
        save_session(sid, [], model_id="ternary-bonsai-27b", settings={
            "model_id": "ternary-bonsai-27b",
            "px_preset": "ACTIVE_MANIFOLD_RELAY",
            "relay_layer": 34,
        })
        updates = restore_session_settings(sid, "neutral")
        kw = upd_kwargs(updates[SETTINGS_WIDGET_FIELDS.index("relay_layer")])
        assert kw["value"] == 34
        assert kw["maximum"] == 64
        assert kw.get("interactive") is True
    finally:
        _wipe([sid])


# --- R5: Profil-Load-Suppress ---------------------------------------------

def test_r5_differing_profile_consumes_one_suppress_token():
    """R5: Restore abweichendes Profil → change-Feuering wird gefressen."""
    sid = "r5_suppress"
    source = "neutral"
    try:
        saved = settings_from_widgets(
            model_id="gemma3-1b-it", px_preset="ACTIVE_MANIFOLD",
            auto_tune=False, temperature=0.7, top_p=0.95, max_tokens=1024,
            rep_p=1.15, px_gamma=0.08, relay_sign=1, relay_alpha=0.3,
            relay_layer=21, system_profile="citmind",
            system_prompt_text="",
        )  # citmind ≠ neutral → 1 Token
        save_session(sid, [], settings=saved)
        restore_session_settings(sid, source)
        # der NÄCHSTE change (programmatischer Restore) no-op't
        res = on_profile_change_load_body("citmind")
        assert not isinstance(res, str), f"Body wurde geladen: {type(res)}"
        # und danach ist die Suppression leer → normale User-Klicks laden Body
        res2 = on_profile_change_load_body("citmind")
        assert isinstance(res2, str) and res2, "Token-Leak: zweiter Aufruf gedrosselt"
    finally:
        _wipe([sid])


def test_r5b_same_profile_no_suppress_token():
    """R5b: restored_profile == current → kein Token (gradio feuert kein
    .change bei Gleichheit) → User-Klick lädt Body sofort."""
    sid = "r5b_same"
    try:
        saved = settings_from_widgets(
            model_id="gemma3-1b-it", px_preset="ACTIVE_MANIFOLD",
            auto_tune=False, temperature=0.7, top_p=0.95, max_tokens=1024,
            rep_p=1.15, px_gamma=0.08, relay_sign=1, relay_alpha=0.3,
            relay_layer=21, system_profile="neutral", system_prompt_text="x",
        )
        save_session(sid, [], settings=saved)
        restore_session_settings(sid, "neutral")
        res = on_profile_change_load_body("juexin")
        assert isinstance(res, str), "Body-Load sollte aktiv sein"
    finally:
        _wipe([sid])


# --- R6-R9: apply_px_defaults ---------------------------------------------

def test_r6_defaults_suppressed_when_settings_match():
    """R6: settings matchen model_id+px_preset → Restore-Fall → 6 no-ops
    (Klassiker + Thinking) und KEIN Relay-Persist (sonst würden die
    wiederhergestellten Werte von den Defaults überschrieben)."""
    sid = "r6_suppress"
    _wipe([sid])
    try:
        save_session(sid, [], model_id="gemma3-1b-it", settings={
            "model_id": "gemma3-1b-it", "px_preset": "ACTIVE_MANIFOLD_RELAY",
        })
        outs = apply_px_defaults("gemma3-1b-it", "ACTIVE_MANIFOLD_RELAY", sid)
        assert len(outs) == 6
        for u in outs:
            assert is_noop(u), upd_kwargs(u)
        # kein Relay-Persist nachgerollt
        flush_settings_save(sid)
        settings = load_session(sid).get("settings", {})
        assert set(settings.keys()) == {"model_id", "px_preset"}, settings
    finally:
        _wipe([sid])


def test_r7_defaults_applied_for_gemma3_1b_it():
    """R7: User pickt gemma3-1b-it → Artifact-Layer 21, max 26, sign +1,
    alpha 0.30, gamma 0.12; Thinking-Widgets VERSTECKT (gemma3 kennt die
    Template-Variablen nicht); persistiert (debounced) ohne thinking-Junk."""
    sid = "r7_apply"
    _wipe([sid])
    try:
        sign_u, alpha_u, layer_u, gamma_u, t_u, e_u = apply_px_defaults(
            "gemma3-1b-it", "ACTIVE_MANIFOLD_RELAY", sid)
        assert upd_kwargs(sign_u) == {"value": 1}
        assert upd_kwargs(alpha_u) == {"value": 0.30}
        assert upd_kwargs(layer_u) == {"value": 21, "minimum": 1,
                                       "maximum": 26, "interactive": True}
        assert upd_kwargs(gamma_u) == {"value": 0.12}
        # Phase 3: nicht-capable → beides versteckt, ohne value
        assert upd_kwargs(t_u) == {"visible": False}
        assert upd_kwargs(e_u) == {"visible": False}
        flush_settings_save(sid)
        settings = load_session(sid).get("settings", {})
        assert settings == {
            "model_id": "gemma3-1b-it", "px_preset": "ACTIVE_MANIFOLD_RELAY",
            "relay_layer": 21, "relay_sign": 1, "relay_alpha": 0.3,
            "px_gamma": 0.12,
        }, settings
    finally:
        _wipe([sid])


def test_r8_defaults_noop_for_minicpm_and_no_relay_persist():
    """R8: minicpm5-1b → 6 no-ops (Klassiker + Thinking hidden, is_noop-
    kompatibel); persist NUR model/preset (kein Relay-Feld, kein gamma,
    kein thinking — MiniCPM ist nicht thinking-capable)."""
    sid = "r8_minicpm"
    _wipe([sid])
    try:
        outs = apply_px_defaults("minicpm5-1b", "ACTIVE_MANIFOLD_RELAY", sid)
        assert len(outs) == 6
        for u in outs:
            assert is_noop(u), upd_kwargs(u)
        flush_settings_save(sid)
        settings = load_session(sid).get("settings", {})
        assert settings == {"model_id": "minicpm5-1b",
                            "px_preset": "ACTIVE_MANIFOLD_RELAY"}, settings
    finally:
        _wipe([sid])


def test_r9_defaults_noop_for_unknown_model():
    """R9: unbekannte model_id → 4 no-ops + Thinking hidden + persist
    model/preset."""
    sid = "r9_unknown"
    _wipe([sid])
    try:
        outs = apply_px_defaults("gibts-nicht", "BASELINE", sid)
        assert len(outs) == 6
        for u in outs:
            assert is_noop(u), upd_kwargs(u)
        flush_settings_save(sid)
        settings = load_session(sid).get("settings", {})
        assert settings == {"model_id": "gibts-nicht", "px_preset": "BASELINE"}
    finally:
        _wipe([sid])


# --- R10-R12: Source-Pins (Wiring-Reihenfolgen) ---------------------------

def _source(path):
    with open(os.path.join(REPO_ROOT, path), "r", encoding="utf-8") as f:
        return f.read()


def _chat_fn_source(src):
    match = re.search(r"^def chat_fn\(.*?(?=^def |^class )", src,
                      re.DOTALL | re.MULTILINE)
    assert match, "chat_fn-Block nicht auffindbar"
    return match.group(0)


def test_r10_chat_fn_persists_settings_at_both_save_points():
    """R10: chat_fn baut die 13-Feld-Einstellung und schreibt sie an BEIDE
    Save-Points (Phase-63 pre-save + post-save)."""
    src = _source("gradio_tabs/chat_tab.py")
    block = _chat_fn_source(src)
    assert "settings_from_widgets(" in block
    assert block.count("settings=chat_settings") == 2, "beide save_points!"
    # Strip-Reihenfolge-Pin (test_chat_tab_token_type_ids_strip) unangetastet:
    assert block.index("strip_unsupported_model_kwargs(") < block.index(
        "model.generate(**gen_kwargs)")


def test_r11_build_chat_tab_wires_defaults_and_restore():
    """R11: Defaults auf .input beider Owner; Restore in load/import-.then."""
    src = _source("gradio_tabs/chat_tab.py")
    block = src[src.index("def build_chat_tab("):]
    assert block.count("apply_px_defaults") >= 2  # def + wiring
    assert "inputs=[model_select, px_preset, session_id_state]" in block
    # Phase 3: 6er-Outputs (Klassiker + Thinking-Widgets)
    assert re.search(r"outputs=\[relay_sign, relay_alpha, relay_layer, "
                     r"px_gamma,\s*thinking, thinking_effort\]", block)
    assert block.count("restore_session_settings") >= 2  # def+... (2× .then)
    assert block.count("inputs=[session_id_state, system_profile]") == 2
    assert "_persist_setting_field" in block
    # Thinking-Widgets existieren + persistieren (Phase 3)
    assert '("thinking", thinking), ("thinking_effort", thinking_effort)' in block
    # Return: 18er-Tupel (4 klassische + 14 Settings-Widgets)
    assert "system_prompt_text,\n    )" in block


def test_r12_app_py_demo_load_18_outputs():
    """R12: app.py demo.load — 18 Outputs, inputs um system_profile erweitert."""
    src = _source("app.py")
    assert "def init_app(session_id, current_profile)" in src
    assert "inputs=[session_id_state, system_profile]" in src
    assert "system_prompt_text]" in src
    for name in ("relay_layer", "system_profile", "px_preset_widgets",
                 "thinking", "thinking_effort"):
        assert name in src, name


def test_r13_on_load_returns_18_tuple_with_noops_for_fresh_session():
    """R13: frische Session → (id, [], choices, id, 14×no-op)."""
    result = on_load(None, "neutral")
    assert len(result) == 18
    session_id, history, choices, session_id2 = result[:4]
    assert isinstance(session_id, str) and len(session_id) == 8
    assert history == []
    assert isinstance(session_id2, str)
    for u in result[4:]:
        assert is_noop(u), upd_kwargs(u)
    # frische ID wieder wegwerfen (wurde nur für den Aufruf gebaut)
    p = _session_path(session_id)
    if os.path.exists(p):
        os.unlink(p)


# --- R14-R19: Thinking-Restore (Phase 3, 2026-10-05) -----------------------

def t_kw(updates, field):
    return upd_kwargs(updates[SETTINGS_WIDGET_FIELDS.index(field)])


def test_r14_bonsai_legacy_partial_gets_template_defaults():
    """R14: bonsai-Session OHNE thinking-Keys → Template-Defaults:
    Checkbox True + effort-Radio xhigh mit den 3 Stufen als choices."""
    sid = "r14_bonsai_legacy"
    _wipe([sid])
    try:
        save_session(sid, [], model_id="ternary-bonsai-27b", settings={
            "model_id": "ternary-bonsai-27b",
            "px_preset": "ACTIVE_MANIFOLD_RELAY",
        })
        updates = restore_session_settings(sid, "neutral")
        kw_t, kw_e = t_kw(updates, "thinking"), t_kw(updates, "thinking_effort")
        assert kw_t == {"value": True, "visible": True, "interactive": True}, kw_t
        assert kw_e["value"] == "xhigh"
        assert kw_e["choices"] == ["xhigh", "medium", "low"]
        assert kw_e["visible"] is True
    finally:
        _wipe([sid])


def test_r15_bonsai_stored_thinking_false_beats_template_default():
    """R15: stored thinking=False (bonsai OFF) → Checkbox False — der
    gespeicherte User-Wert schlägt den Template-Default True."""
    sid = "r15_bonsai_off"
    _wipe([sid])
    try:
        save_session(sid, [], model_id="ternary-bonsai-27b", settings={
            "model_id": "ternary-bonsai-27b",
            "px_preset": "ACTIVE_MANIFOLD_RELAY", "thinking": False,
        })
        updates = restore_session_settings(sid, "neutral")
        kw_t = t_kw(updates, "thinking")
        assert kw_t["value"] is False
        assert kw_t["visible"] is True
        # effort: kein Key → Template-Default xhigh
        assert t_kw(updates, "thinking_effort")["value"] == "xhigh"
    finally:
        _wipe([sid])


def test_r16_bonsai_stored_effort_medium_roundtrips():
    """R16: stored thinking_effort="medium" → Radio rendert medium."""
    sid = "r16_bonsai_medium"
    _wipe([sid])
    try:
        save_session(sid, [], model_id="ternary-bonsai-27b", settings={
            "model_id": "ternary-bonsai-27b",
            "px_preset": "ACTIVE_MANIFOLD_RELAY", "thinking": True,
            "thinking_effort": "medium",
        })
        updates = restore_session_settings(sid, "neutral")
        kw_e = t_kw(updates, "thinking_effort")
        assert kw_e["value"] == "medium"
        assert kw_e["choices"] == ["xhigh", "medium", "low"]
    finally:
        _wipe([sid])


def test_r17_bonsai_stored_invalid_effort_falls_back_to_default():
    """R17: stored thinking_effort="bogus" (z.B. korrupter Import) → nicht
    gerendert, Template-Default xhigh — kein invalid-Value im Radio."""
    sid = "r17_bonsai_bogus"
    _wipe([sid])
    try:
        save_session(sid, [], model_id="ternary-bonsai-27b", settings={
            "model_id": "ternary-bonsai-27b",
            "px_preset": "ACTIVE_MANIFOLD_RELAY",
            "thinking": True, "thinking_effort": "bogus",
        })
        updates = restore_session_settings(sid, "neutral")
        assert t_kw(updates, "thinking_effort")["value"] == "xhigh"
    finally:
        _wipe([sid])


def test_r18_gemma4_restore_checkbox_visibility_effort_hidden():
    """R18: gemma4-e2b-it → Checkbox visible (stored/default False —
    Template default(false)) UND effort-Radio versteckt (kein Budget-
    Parameter im installierten Stack)."""
    sid = "r18_gemma4"
    _wipe([sid])
    try:
        save_session(sid, [], model_id="gemma4-e2b-it", settings={
            "model_id": "gemma4-e2b-it", "px_preset": "ACTIVE_MANIFOLD",
        })
        updates = restore_session_settings(sid, "neutral")
        kw_t, kw_e = t_kw(updates, "thinking"), t_kw(updates, "thinking_effort")
        assert kw_t["value"] is False
        assert kw_t["visible"] is True
        assert kw_e == {"visible": False}, kw_e
    finally:
        _wipe([sid])


def test_r19_incapable_restore_hides_stale_thinking_junk():
    """R19: gemma3-Session mit stale thinking=True/"medium" (z.B. nach
    bonsai-Chat in derselben Session) → beide visible=False OHNE value —
    der Junk darf nicht in die (unsichtbaren) Widgets rutschen."""
    sid = "r19_gemma3_junk"
    _wipe([sid])
    try:
        save_session(sid, [], model_id="gemma3-1b-it", settings={
            "model_id": "gemma3-1b-it", "px_preset": "ACTIVE_MANIFOLD",
            "thinking": True, "thinking_effort": "medium",
        })
        updates = restore_session_settings(sid, "neutral")
        assert t_kw(updates, "thinking") == {"visible": False}
        assert t_kw(updates, "thinking_effort") == {"visible": False}
    finally:
        _wipe([sid])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))