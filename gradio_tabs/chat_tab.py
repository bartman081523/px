"""
chat_tab.py — Gradio Chat Tab with ChatInterface & Subjective Mode
================================================================
Integrated chat interface with session management and PX steering.
"""

import gradio as gr
import torch
import asyncio
import os
import json
import statistics
import threading
from typing import Optional, List, Dict, Any
from threading import Thread
from transformers import TextIteratorStreamer

from config import MODEL_REGISTRY
from model_manager import ModelManager
from sessions import save_session, load_session, get_new_session_id, list_sessions
from telemetry import telemetry
from gradio_tabs.px_defaults import get_px_defaults, get_thinking_defaults
from gradio_tabs.settings_persist import schedule_settings_save
from gradio_tabs.multimodal_input import (
    normalize_multimodal_message,
    is_empty_message,
    extract_text_blocks,
    _normalize_history_for_chatbot,
)


# ── Session-Settings: Feld-Reihenfolge + Restore ────────────────────────
# Plan 2026-10-05 (User-Request): Session-Settings (Modell, px_preset,
# Parameter, Relay, System-Prompt) werden in der session.json gespeichert
# (settings_persist.schedule_settings_save via Widget-.input, plus beide
# chat_fn-save_points) und beim Laden der Session zurück in die Widgets
# gerendert. Die Output-Reihenfolge der Restore-Handler ist DIESE Liste —
# app.py (demo.load) und die .then-Chains müssen exakt dazu passen.
SETTINGS_WIDGET_FIELDS = (
    "model_id", "px_preset", "temperature", "top_p", "max_tokens",
    "rep_p", "px_gamma", "thinking", "thinking_budget", "thinking_effort",
    "relay_sign", "relay_alpha", "relay_layer",
    "system_profile", "system_prompt_text",
)

_SETTINGS_NOOP_UPDATES: tuple = None  # lazy erzeugt (gr.update())


def noop_settings_updates():
    """15 no-op gr.update() in SETTINGS_WIDGET_FIELDS-Reihenfolge."""
    global _SETTINGS_NOOP_UPDATES
    if _SETTINGS_NOOP_UPDATES is None:
        _SETTINGS_NOOP_UPDATES = tuple(
            gr.update() for _ in SETTINGS_WIDGET_FIELDS
        )
    return list(_SETTINGS_NOOP_UPDATES)


def restore_session_settings(session_id, current_profile=None):
    """Session-Settings → Widget-Updates (15 Outputs, fixe Reihenfolge).

    Restore-Semantik (Plan 2026-10-05):
    - Session-Datei OHNE 'settings'-key (alte/leere Sessions) → ALLE
      no-op updates: die Widgets bleiben wie sie stehen, kein Reset auf
      Defaults beim Laden alter Sessions.
    - settings={} ebenso (nichts zu wiederherstellen).
    - Sonst: widget_updates_from_settings (chat_settings.py) — fehlende
      Felder fallen auf SETTINGS_DEFAULTS, auto_tune lockt temp/top_p/
      rep_p/px_gamma interaktiv.

    Thinking (Phase 3, 2026-10-05): model-aware — für thinking-kapable
    Modelle (gemma4-e2b-it, ternary-bonsai-27b) werden die Widgets
    mit den gespeicherten Werten (Fallback: per-Modell-Template-Default
    aus px_defaults.get_thinking_defaults, legacy-Sessions haben die Keys
    meist nicht) + korrekter effort-choices-Liste gerendert; für nicht-
    capable Modelle (gemma3-*, minicpm5-1b, unbekannt) werden sie per
    visible=False versteckt OHNE Wert-Update (persistiertes thinking-
    Junk aus anderen Modellen soll nicht in die Widgets rutschen).
    thinking_budget (Plan 2026-10-05): nur gemma4-capable sichtbar;
    gespeicherter int 0..max gilt (0 = unbegrenzt), sonst Modell-Default.

    auto_tune-Abweichung vom chat_settings-Pin: die UI hat KEIN
    auto_tune-Widget (dead field, Plan 2026-10-05) — ein fehlender Key
    darf therefore nie locken (sonst sind temperature/top_p/rep_p/
    px_gamma nach dem Load unveränderbar, nur weil das Feld nicht
    persistiert wurde). Fehlt der Key → False (nicht locken).

    current_profile (optional): aktuell gerendertes Profil-Dropdown.
    Weicht der wiederherzustellende Profil-Wert davon ab, feuert der
    programmatische Dropdown-Update das system_profile.change-Event —
    dessen Body-Load würde die freie Textarea überschreiben → dafür
    wird hier EIN Suppress-Token gezählt (siehe
    on_profile_change_load_body). Gleicher Wert feuert kein .change →
    kein Zähler-Leak.
    """
    data = {} if not session_id else load_session(session_id)
    settings = (data or {}).get("settings")
    if not isinstance(settings, dict) or not settings:
        return noop_settings_updates()
    from gradio_tabs.chat_settings import widget_updates_from_settings
    settings = dict(settings)
    settings.setdefault("auto_tune", False)
    updates = widget_updates_from_settings(settings)
    ordered = [updates[f] for f in SETTINGS_WIDGET_FIELDS]

    # Slider-Bounds für den Injektions-Layer mitrestoren: der gespeicherte
    # Layer (z.B. ternary L34) muss in den Slider-Bounds liegen, sonst
    # clamp't Gradio den Restore-Wert. maximum = max(gespeicherter Wert,
    # n_layers des restoreten Modells).
    model_id = settings.get("model_id")
    per_model = get_px_defaults(model_id) if isinstance(model_id, str) else None
    if per_model is not None:
        stored_layer = settings.get("relay_layer")
        value = (int(stored_layer) if isinstance(stored_layer, int)
                 and not isinstance(stored_layer, bool)
                 else per_model["inject_layer"])
        if value is not None:
            idx = SETTINGS_WIDGET_FIELDS.index("relay_layer")
            ordered[idx] = gr.update(
                value=value, minimum=1,
                maximum=max(value, per_model["n_layers"]),
                interactive=True,
            )

    # Thinking-Widgets (Phase 3): model-aware rendern. Kapabilität kommt aus
    # px_defaults.get_thinking_defaults; die gespeicherten Werte schlagen
    # die Template-Defaults (legacy-Sessions haben meist gar keinen Key —
    # dann gilt der Modell-Default). Nicht-capable → ohne Wert verstecken
    # (stale thinking-Keys anderer Modelle rutschen nicht ins UI).
    thinking_idx = SETTINGS_WIDGET_FIELDS.index("thinking")
    effort_idx = SETTINGS_WIDGET_FIELDS.index("thinking_effort")
    budget_idx = SETTINGS_WIDGET_FIELDS.index("thinking_budget")
    tcap = get_thinking_defaults(model_id) if isinstance(model_id, str) else None
    if tcap is None:
        ordered[thinking_idx] = gr.update(visible=False)
        ordered[effort_idx] = gr.update(visible=False)
        ordered[budget_idx] = gr.update(visible=False)
    else:
        stored_thinking = settings.get("thinking")
        ordered[thinking_idx] = gr.update(
            value=(bool(stored_thinking) if isinstance(stored_thinking, bool)
                   else tcap["default"]),
            visible=True, interactive=True,
        )
        if tcap["efforts"]:
            stored_effort = settings.get("thinking_effort")
            ordered[effort_idx] = gr.update(
                value=(stored_effort if stored_effort in tcap["efforts"]
                       else tcap["effort_default"]),
                choices=list(tcap["efforts"]),
                visible=True, interactive=True,
            )
        else:
            ordered[effort_idx] = gr.update(visible=False)
        # Budget (Plan 2026-10-05): nur gemma4 (budget_range gesetzt).
        # Gespeicherter int 0..max gilt (0 = unbegrenzt); alles andere
        # (None/legacy/außerhalb) → Modell-Default.
        if tcap["budget_range"]:
            stored_budget = settings.get("thinking_budget")
            b_lo, b_hi = tcap["budget_range"]
            b_value = (int(stored_budget)
                       if isinstance(stored_budget, int)
                       and not isinstance(stored_budget, bool)
                       and 0 <= stored_budget <= b_hi
                       else tcap["budget_default"])
            ordered[budget_idx] = gr.update(
                value=b_value, minimum=b_lo, maximum=b_hi,
                visible=True, interactive=True,
            )
        else:
            ordered[budget_idx] = gr.update(visible=False)

    restored_profile = settings.get("system_profile")
    if restored_profile is not None and restored_profile != current_profile:
        _suppress_profile_body_load_once()
    return ordered


def apply_px_defaults(model_id, px_preset, session_id):
    """model-select/.px_preset-.input-Handler (User-Aktion).

    Wendet per-Model-Defaults an (gradio_tabs/px_defaults.py): Relay-
    Richtung/Alpha/Injektions-Layer (aus dem d_width-Artefakt der hf_id,
    Fallback statische Tabelle) + px_gamma (SCALE_DEFAULTS pro hidden_
    size). Slider-Bounds (n_layers) werden im update mitgesetzt — der
    ternary-Modell-34 war mit dem alten Slider (max 25) unerreichbar.

    Suppress-Regel: existiert die Session-Datei mit settings, und
    settings.model_id == model_id UND settings.px_preset == px_preset →
    programmatischer Restore (Session-Load rendert die Widgets), keine
    User-Änderung → NICHTS überschreiben (sonst würfen wir die gerade
    wiederhergestellten User-Werte weg). Nur persisten wäre auch falsch
    (no-op-Merge) → direkt no-op-Updates returnen.

    Persistiert sonst (debounce) model_id + px_preset + die angewendeten
    Defaults in die Session.

    Returns: 7-Tupel (relay_sign_u, relay_alpha_u, relay_layer_u,
    px_gamma_u, thinking_u, thinking_budget_u, thinking_effort_u) —
    WIRKLICH angewendete Felder bekommen value(+visible), nicht
    angewendete (MiniCPM: kein Relay/gamma-Default) no-op. Thinking
    (Phase 3): capable Modelle kriegen Modell-Default + Sichtbarkeit/
    choices, nicht-capable alle visible=False; der Session-Patch trägt
    die Values nur capable. Budget (Plan 2026-10-05): nur gemma4
    (budget_range) → Modell-Default + Slider-Bounds, sonst hidden.
    """
    data = {} if not session_id else load_session(session_id)
    settings = (data or {}).get("settings")
    restored = (
        isinstance(settings, dict)
        and settings.get("model_id") == model_id
        and settings.get("px_preset") == px_preset
    )
    if restored:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update())

    defaults = get_px_defaults(model_id)
    tcap = get_thinking_defaults(model_id)
    if defaults is None:
        # Unbekannte model_id: nichts anwenden, nur Auswahl persistieren.
        # Thinking-Widgets verstecken (kein Modell → kein Thinking-UI).
        schedule_settings_save(session_id, model_id=model_id, px_preset=px_preset)
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False))

    relay_available = defaults["relay_available"]
    layer_update = gr.update(
        value=defaults["inject_layer"], maximum=defaults["n_layers"],
        minimum=1, interactive=True,
    ) if relay_available and defaults["inject_layer"] is not None else gr.update()
    sign_update = gr.update(value=defaults["relay_sign"]) if relay_available else gr.update()
    alpha_update = gr.update(value=defaults["relay_alpha"]) if relay_available else gr.update()
    gamma_update = (
        gr.update(value=defaults["px_gamma"])
        if defaults["px_gamma"] is not None else gr.update()
    )

    # Thinking (Phase 3): Modell-Default render + persist. Bonsai hat
    # reasoning_effort-Stufen (choices-Reset im Update), gemma4 nur den
    # Checkbox-Toggle (effort-Widget bleibt/ wird versteckt). Nicht-
    # capable → beide verstecken; Session-Patch ohne thinking-Junk.
    if tcap is None:
        thinking_update = gr.update(visible=False)
        effort_update = gr.update(visible=False)
        budget_update = gr.update(visible=False)
    else:
        thinking_update = gr.update(
            value=tcap["default"], visible=True, interactive=True,
        )
        if tcap["efforts"]:
            effort_update = gr.update(
                value=tcap["effort_default"],
                choices=list(tcap["efforts"]),
                visible=True, interactive=True,
            )
        else:
            effort_update = gr.update(visible=False)
        # Budget (Plan 2026-10-05): nur gemma4 (budget_range) — Modell-
        # Default + Slider-Bounds; bonsai (effort IST der Budget-Parameter)
        # bekommt bewusst kein zweites Budget-Widget → hidden.
        if tcap["budget_range"]:
            budget_update = gr.update(
                value=tcap["budget_default"],
                minimum=tcap["budget_range"][0],
                maximum=tcap["budget_range"][1],
                visible=True, interactive=True,
            )
        else:
            budget_update = gr.update(visible=False)
    patch = {
        "model_id": model_id,
        "px_preset": px_preset,
        "relay_layer": defaults["inject_layer"],
        "relay_sign": defaults["relay_sign"],
        "relay_alpha": defaults["relay_alpha"],
        "px_gamma": defaults["px_gamma"],
    }
    # None-Felder NICHT persistieren (= "UI-Wert ist okay")
    patch = {k: v for k, v in patch.items() if v is not None}
    if tcap is not None:
        # Capable: Modell-Defaults persistieren. Gemma4 (efforts=None)
        # schreibt thinking_effort=None EXPLIZIT — sonst lebt ein stale
        # bonsai-"medium" in der Session weiter, obwohl das Widget
        # versteckt ist.
        patch["thinking"] = bool(tcap["default"])
        # thinking_effort/thinking_budget None EXPLIZIT — sonst lebt stale
        # bonsai-"medium" bzw. eine gemma4-Budget-Zahl in der Session weiter,
        # obwohl das Widget versteckt ist (gleiche Logik wie oben).
        patch["thinking_effort"] = tcap["effort_default"] if tcap["efforts"] else None
        patch["thinking_budget"] = (int(tcap["budget_default"])
                                    if tcap["budget_range"] else None)
    schedule_settings_save(session_id, **patch)
    return (sign_update, alpha_update, layer_update, gamma_update,
            thinking_update, budget_update, effort_update)


def _persist_setting_field(field: str):
    """Widget-.input-Persist-Handler-Factory (User-Aktion pro Feld).

    Closed-over field-name → schedule_settings_save(session_id,
    <field>=value). outputs-frei; Fehler in der Persistenz dürfen das
    UI-Event nicht crashen (fire-and-forget im Debouncer sowieso).
    """
    def handler_with_session(session_id, value):
        schedule_settings_save(session_id, **{field: value})

    return handler_with_session


# ── Session Handlers ──

def _stringify_content(content):
    """Ensure content is a string for text-only templates."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item and "files" not in item: # Handle some Gradio formats
                    parts.append(item["text"])
        return "\n".join(parts)
    if isinstance(content, dict):
        return content.get("text", str(content))
    return str(content)

def _clean_history(history):
    """Filter empty messages and merge consecutive same-role messages."""
    result = []
    for msg in (history or []):
        if not isinstance(msg, dict):
            continue
            
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if isinstance(content, str):
            if not content.strip():
                continue
        elif not content:
            continue
            
        if result and result[-1]["role"] == role:
            prev_content = result[-1]["content"]
            if isinstance(prev_content, str) and isinstance(content, str):
                result[-1]["content"] += "\n" + content
            elif isinstance(prev_content, list) and isinstance(content, list):
                result[-1]["content"].extend(content)
            elif isinstance(prev_content, list) and isinstance(content, str):
                result[-1]["content"].append({"type": "text", "text": content})
            elif isinstance(prev_content, str) and isinstance(content, list):
                result[-1]["content"] = [{"type": "text", "text": prev_content}] + content
        else:
            result.append({"role": role, "content": content})
    return result

def on_load(session_id, current_profile=None):
    """Called when the page loads.

    Plan 2026-10-05 (Session-Settings-Restore): liefert NACH den 4 klassischen
    Werten die 15 Settings-Widget-Updates (SETTINGS_WIDGET_FIELDS-Reihen-
    folge) — Settings-lose/legacy-Sessions → no-op updates, Widgets bleiben
    wie gebaut. current_profile = aktueller Wert des Profil-Dropdowns
    (Suppress-Ermittlung, siehe restore_session_settings).
    """
    if session_id is None or session_id == "":
        session_id = get_new_session_id()

    data = load_session(session_id)
    # Normalize history for Gradio 6.15.2 Chatbot (OpenAI blocks + untyped dicts
    # would otherwise crash _postprocess_content; see multimodal_input.py).
    history = _normalize_history_for_chatbot(data.get("history", []))
    return (session_id, history, gr.update(choices=list_sessions()), session_id,
            *restore_session_settings(session_id, current_profile))

def handle_new_session():
    new_id = get_new_session_id()
    return new_id, [], gr.update(choices=list_sessions()), new_id

def handle_load_saved(session_id):
    if not session_id:
        return gr.skip(), [], gr.skip(), gr.skip()
    data = load_session(session_id)
    return session_id, _normalize_history_for_chatbot(data.get("history", [])), gr.skip(), session_id

def handle_export(session_id, history):
    if not history:
        return gr.update(visible=False)
    path = f"exported_session_{session_id}.json"
    # Plan 2026-10-05 (Session-Settings-Restore): das Export-JSON trägt
    # zusätzlich model_id + settings (best effort — legacy Sessions ohne
    # passende Datei exportieren wie bisher nur session_id+history).
    export_data = {"session_id": session_id, "history": history}
    try:
        stored = load_session(session_id) if session_id else {}
        if isinstance(stored.get("model_id"), str):
            export_data["model_id"] = stored["model_id"]
        if isinstance(stored.get("settings"), dict) and stored["settings"]:
            export_data["settings"] = stored["settings"]
    except (OSError, ValueError):
        pass
    with open(path, "w") as f:
        json.dump(export_data, f, indent=2)
    return gr.update(value=path, visible=True)

def handle_import(file_obj):
    if file_obj is None:
        return gr.skip(), [], gr.skip(), gr.skip()
    try:
        with open(file_obj.name, "r") as f:
            data = json.load(f)
        new_id = data.get("session_id", get_new_session_id())
        history = data.get("history", [])
        # Plan 2026-10-05: Import nimmt model_id + settings mit — eine
        # exportierte Session läuft nach dem Import exakt weiter (gleiche
        # Einstellungen, gleiche Widgets beim Load).
        _import_settings = data.get("settings")
        _import_model = data.get("model_id")
        save_session(
            new_id, history,
            model_id=_import_model if isinstance(_import_model, str) else None,
            settings=_import_settings if isinstance(_import_settings, dict) else None,
        )
        return new_id, history, gr.update(choices=list_sessions(), value=new_id), new_id
    except Exception as e:
        print(f"Import error: {e}")
        return gr.skip(), [], gr.skip(), gr.skip()

def handle_refresh():
    return gr.update(choices=list_sessions())


# Plan ui-styling 2026-07-08: Undo-Button für "letzte Nachricht rückgängig"
# (User-Feedback: vorher poppte er das ganze (user, assistant)-Paar — User
# wollte aber nur 1 Element rückgängig machen, entweder die letzte User- oder
# die letzte Agent-Antwort). Nutzt chat_actions.undo_last_entry statt
# undo_last_turn. Persistiert die gekürzte History sofort via save_session.
def handle_undo(session_id, history):
    """Click-handler für den Undo-Button.

    Returns: (updated_history, status_text)
        - updated_history: gekürzte History (oder skip wenn nichts zu undo)
        - status_text: "✓ Undone" oder "⚠ Nothing to undo"
    """
    from gradio_tabs.chat_actions import undo_last_entry, can_undo_entry
    if not can_undo_entry(history):
        return gr.skip(), "⚠ Nothing to undo"
    new_history = undo_last_entry(history)
    if session_id:
        save_session(session_id, new_history)
    return new_history, f"✓ Undone (history: {len(new_history)} msgs)"


def _spaces_gpu(fn):
    """Plan hf-space-v4-publish: ZeroGPU-Lease für die Chat-Generation.
    60s-Default-Lease (spaces/zero/client.py DEFAULT_SCHEDULE_DURATION)
    reicht nicht für bonsai: snapshot_download 6 GB + GF3-Build + triton-JIT
    + Stream im Erstaufruf. Auf lokaler/regulärer GPU-Hardware no-op: spaces'
    _GPU gibt fn unverändert zurück, wenn SPACES_ZERO_GPU nicht gesetzt ist
    (wheel-Inspection 0.50.4, Config.zero_gpu).

    duration=PX_SPACES_LEASE (env/Space-Secret, Bisekt 2026-10-08): der Client
    skaliert die Laufzeit mit dem duration_factor des GPU-Profils (configs.json:
    Blackwell RTX PRO 6000 = 1.5). GEMESSEN (A-Grad, serververweigert): 900 →
    Request 1350 s abgelehnt; 180 → Request 270 s ebenfalls abgelehnt — der
    Free-Cap liegt UNTER 270 s ("... subscribe to PRO ... up to 40 min").
    Default 60 → Request 90 s; per Space-Secret PX_SPACES_LEASE hochbisektieren.
    Caches überleben das Lease-Ende containerweit und hf_hub_download resümiert
    .incomplete-Blobs per Range-Request — also: scheitert die Lease am Zeitlimit,
    macht der nächste Lauf Fortschritt statt von vorn zu beginnen."""
    if not os.environ.get("SPACES_ZERO_GPU"):
        return fn
    import spaces
    _lease = int(os.environ.get("PX_SPACES_LEASE", "60"))
    return spaces.GPU(duration=_lease)(fn)


@_spaces_gpu
def chat_fn(message, history, model_id, px_preset, temp, tp, mt, rp, gamma,
            relay_sign, relay_alpha, relay_layer,
            system_profile, system_prompt_text,
            thinking, thinking_budget, thinking_effort,
            session_id, manager: ModelManager):
    """Core chat logic with history management and model generation.

    Plan ui-styling 2026-07-06: zwei neue Parameter (system_profile,
    system_prompt_text) — kommen aus dem Einstellungen-Tab. Vor dem
    chat_template-apply wird inject_into_messages() aufgerufen, das die
    System-Message an Index 0 setzt (und alle existing system-Einträge
    strippt). Bei neutral+leerer Edit: no-op (Original-Liste).

    Phase 3 (2026-10-05): thinking/thinking_effort — die Widget-Values
    gehen KAPABILITÄTS-GEGATED in apply_chat_template (_thinking_template_
    kwargs): bonsai-27b → enable_thinking + reasoning_effort (qwen3.5-
    Template, budget als Stufe), gemma4-e2b-it → enable_thinking.
    Gemma3/MiniCPM/unbekannt: {} — die Template-Extras werden nie
    hingeschickt.

    Plan 2026-10-05 (Gemma4 Thinking-Budget): thinking_budget — Widget-
    Value als Token-Budget im thought-Kanal (max_thinking_tokens-
    Semantik, App-Level LogitsProcessor). NUR im plain-Pfad (gemma4 ist
    nie long/chunked-capable) und NUR bei Thinking an, sonst None
    (kein Budget am generate). 0 = unbegrenzt.
    """
    print(f"DEBUG: history received from Gradio (UI state): {len(history) if history else 0} messages")
    # verstärkbar Relay-Parameter nur beim RELAY-Preset durchreichen (sonst None
    # → kein Surprise-Relay auf BASELINE/LEAN/ACTIVE_MANIFOLD; diese verhalten
    # sich exakt wie vorher). Bei RELAY steuert die UI (Radio/Slider).
    if px_preset == "ACTIVE_MANIFOLD_RELAY":
        _rsign, _ralpha, _rlayer = relay_sign, relay_alpha, relay_layer
    else:
        _rsign = _ralpha = _rlayer = None
    # 1. Update config
    loop = asyncio.new_event_loop()
    try:
        model_entry = loop.run_until_complete(
            manager.get_model(
                model_id,
                px_subjective=(px_preset != "BASELINE"),
                px_gamma=gamma,
                px_config_preset=px_preset,
                px_relay_sign=_rsign,
                px_relay_alpha=_ralpha,
                px_relay_layer=_rlayer,
            )
        )
    finally:
        loop.close()

    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]

    # 2. Build history (cleaned)
    # If history is empty (e.g. after loading a session or if save_history=False),
    # load it from the session storage to ensure continuity.
    if (not history or len(history) == 0) and session_id:
        data = load_session(session_id)
        history = data.get("history", [])
    
    cleaned_history = _clean_history(history)
    print(f"DEBUG: Initial cleaned_history length: {len(cleaned_history)}")

    # Plan ui-styling 2026-07-06: System-Prompt injizieren (Frame-Orientierer).
    # inject_into_messages setzt System-Message an Index 0 und strippt alle
    # existing system-Einträge. Bei neutral+leerem Edit: no-op (Pin T3).
    from gradio_tabs.system_prompt import inject_into_messages
    cleaned_history = inject_into_messages(
        cleaned_history, system_profile, system_prompt_text,
    )

    # Normalize the Gradio MultimodalTextbox value into chat_fn's expected
    # message content (plain str, or content-list with text + image blocks).
    # All shape-handling lives in gradio_tabs.multimodal_input — kept pure
    # for testability (tests/test_multimodal_input.py).
    actual_message = normalize_multimodal_message(message)

    messages = cleaned_history + [{"role": "user", "content": actual_message}]
    print(f"DEBUG: Combined messages length: {len(messages)}")

    # SR-61b: Explicitly clear cache to prevent OOM on 12GB cards
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 63: Proactive Auto-save (save user message before generation)
    # Plan 2026-10-05: BEIDE chat_fn-Save-Points persistieren zusätzlich die
    # komplette Einstellung (Model, px_preset, Parameter, Thinking + Budget,
    # Relay, System-prompt) — Session-Load rendert die UI exakt so wieder
    # (T1-Pin der chat_settings-Roundtrips bleibt erhalten: alle 16 Felder
    # im dict).
    from gradio_tabs.chat_settings import settings_from_widgets
    chat_settings = settings_from_widgets(
        model_id=model_id,
        px_preset=px_preset,
        auto_tune=False,  # UI hat kein auto_tune-Widget (dead field)
        temperature=temp,
        top_p=tp,
        max_tokens=mt,
        rep_p=rp,
        px_gamma=gamma,
        thinking=thinking,
        thinking_budget=thinking_budget,
        thinking_effort=thinking_effort,
        relay_sign=relay_sign,
        relay_alpha=relay_alpha,
        relay_layer=relay_layer,
        system_profile=system_profile,
        system_prompt_text=system_prompt_text,
    )
    save_session(session_id, messages, model_id=model_id, settings=chat_settings)

    # 3. Generate with streaming
    # Robustness: Flatten to strings if no images are present to satisfy text-only templates.
    # Plan 2026-07-09: Image-Detection erweitert. Vorher: ``type == "image"``
    # (altes _file_block-Format). Jetzt: ``type == "file"`` mit mime_type
    # image/* (Gradio file-Block) ODER legacy ``type == "image"`` (zur
    # Sicherheit — sollte nicht mehr auftreten, aber defensive Programmierung).
    has_images = any(
        isinstance(m.get("content"), list) and any(
            isinstance(c, dict) and (
                c.get("type") == "image"  # legacy pre-2026-07-09
                or (
                    c.get("type") == "file"
                    and str((c.get("file") or {}).get("mime_type", "")).startswith("image/")
                )
            )
            for c in m["content"]
        )
        for m in messages
    )
    
    if not has_images:
        processed_messages = [{"role": m["role"], "content": _stringify_content(m["content"])} for m in messages]
    else:
        processed_messages = messages

    # Phase 3 (2026-10-05): Thinking-Kontrolle über die etablierte
    # apply_chat_template-Methode — template-Variablen je nach Modell-
    # Kapabilität (gemma4: enable_thinking; bonsai: + reasoning_effort).
    # Nicht-capable → {} (Extras landen nie im Jinja-Kontext).
    thinking_kwargs = _thinking_template_kwargs(model_id, thinking, thinking_effort)
    input_text = tokenizer.apply_chat_template(
        processed_messages, tokenize=False, add_generation_prompt=True,
        **thinking_kwargs,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # Plan 2026-10-05 (RTPF-A4): Sampling am User-Slider — auch für
    # PX-Presets. Begründung (TT0, scratches/rtpf/tt0_result.json, seed 42,
    # T=5326): Weder Greedy 1e-10 noch temp 0.7 reproduzieren den
    # f31eff3e-Loop im Replay; der Live-Loop war trajektorienabhängig —
    # plausible Kaskade aus dem Streamer-Skip-Defekt (verschluckter
    # Eröffnungstoken persistierte im Verlauf und degradierte Folge-Turne;
    # Fix in long_context.generate_long, Tests test_long_streamer_skip.py).
    # User-Sanction (Plan-Beschluss 2026-10-05): do_sample=True explizit
    # erlaubt. SSE-Parität: streaming_bridge streamt mit Request-temp.
    # temp=0 ist Greedy (1e-10/do_sample=False) — die deterministische Mode
    # bleibt über den Slider erreichbar.
    _temperature = temp if temp > 0 else 1e-10
    _do_sample = temp > 0
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(mt),
        temperature=_temperature,
        top_p=tp,
        repetition_penalty=rp,
        do_sample=_do_sample,
    )

    # Inject EOS/EOT and PX-specific kwargs (SR-61b: StopOnEOT criteria)
    # Plan 2026-10-05 (Live-Crash nach TXT-Anhang): `_input_len` an
    # `_px_gen_kwargs` übergeben — wie in allen generators-Pfaden
    # (generate/generate_stream/_generate_long_completion). Vorher lief der
    # UI-Chat IMMER den HF-Short-Pfad (model.generate) und starb im
    # SDPA-math-Backend (O(n²)-Score-Matrix) bei langen Prompts. KEIN
    # Token-Cap (globale User-Regel): KV-4Bit-Cache + Chunked-Prefill
    # fangen lange Kontexte, OOM-Guard hält den Server am Leben.
    use_long_stream = False
    use_chunked_stream = False
    oom_error = None
    try:
        from generators import (_px_gen_kwargs, _inject_eot_eos,
                                strip_unsupported_model_kwargs,
                                _thinking_budget_kwargs)
        gen_kwargs["_input_len"] = int(inputs["input_ids"].shape[1])
        gen_kwargs = _inject_eot_eos(gen_kwargs, tokenizer)
        gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
        # Marker-Pops wie im Server-Stream-Pfad: die Marker sind
        # generate()-fremd (ValueError in _validate_model_kwargs) und
        # schalten auf die Lang-Kontext-Worker um.
        use_long_stream = gen_kwargs.pop("_px_use_long_ctx", False)
        use_chunked_stream = gen_kwargs.pop("_px_use_chunked_prefill", False)
        # Plan 7.2 / Live-Crash 2026-06-30: Llama-Pfade (z.B. MiniCPM5-1B)
        # lehnen `token_type_ids` ab, das der Tokenizer fälschlich setzt.
        # model.generate() validiert VOR dem ersten forward → muss hier
        # gestrippt werden, nicht erst im forward.
        gen_kwargs = strip_unsupported_model_kwargs(model, gen_kwargs)
        # Plan 2026-10-05 (Gemma4 Thinking-Budget): App-Level LogitsProcessor
        # (generators._thinking_budget_kwargs) — nur im plain-Pfad: generate_
        # long/chunked_generate haben keine LogitsProcessor-Route, und dort
        # fehlt die Kanal-Infrastruktur ohnehin (bonsai/gemma3 → Helper
        # no-op). Thinking aus → None → kein Budget am generate.
        if not (use_long_stream or use_chunked_stream):
            gen_kwargs = _thinking_budget_kwargs(
                gen_kwargs, inputs["input_ids"],
                thinking_budget if thinking else None, tokenizer)
    except ImportError:
        pass

    def _run_long_stream():
        nonlocal oom_error
        from generators import _import_long_context
        g = _long_ctx_generate_kwargs(gen_kwargs)
        eos_field = g.get("eos_token_id", [])
        if isinstance(eos_field, int):
            eos_field = [eos_field]
        registry_max = model_entry.get("registry", {}).get("max_length")
        try:
            generate_long = _import_long_context()
            generate_long(model, inputs["input_ids"],
                          max_new_tokens=int(mt),
                          streamer=streamer,
                          eos_token_ids=tuple(eos_field),
                          max_total_seq=registry_max,
                          verbose=True, **g)
        except Exception as _exc:
            # Server-Parität (generators._long_worker): Stream sauber
            # beenden statt crash_handler zu triggern; OOM zusätzlich
            # anzeigen (Chat-Note, s.u.).
            import traceback as _tb
            _tb.print_exc()
            if _is_cuda_oom(_exc):
                oom_error = _exc
            try:
                streamer.end()
            except Exception:
                pass

    def _run_chunked_stream():
        nonlocal oom_error
        # Server-Parität (generators: scratches/4b-image chunked_generate,
        # gemma3-4b — 10x schneller als use_cache=False).
        try:
            from chunked_prefill import chunked_generate as _chunked_generate
        except ImportError:
            import sys as _sys
            _SCRATCHES = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "scratches", "4b-image",
            )
            if _SCRATCHES not in _sys.path:
                _sys.path.insert(0, _SCRATCHES)
            from chunked_prefill import chunked_generate as _chunked_generate
        eos_field = gen_kwargs.pop("eos_token_id", None)
        if isinstance(eos_field, list):
            eos_id = eos_field[0] if eos_field else None
        elif isinstance(eos_field, int):
            eos_id = eos_field
        else:
            eos_id = tokenizer.eos_token_id
        do_sample = gen_kwargs.get("do_sample", False)
        try:
            _chunked_generate(
                model,
                inputs["input_ids"],
                max_new_tokens=int(mt),
                do_sample=do_sample,
                eos_token_id=eos_id,
                streamer=streamer,
                pixel_values=inputs.get("pixel_values"),
                **({"token_type_ids": inputs["token_type_ids"]} if inputs.get("token_type_ids") is not None else {}),
            )
        except Exception as _exc:
            import traceback as _tb
            _tb.print_exc()
            if _is_cuda_oom(_exc):
                oom_error = _exc
            try:
                streamer.end()
            except Exception:
                pass

    def generate_with_lock():
        nonlocal oom_error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        manager.lock_model(model_id)
        try:
            # Server-Parität: multimodal = pixel_values (inputs), NICHT
            # has_images (UI-Content-Shapes) — im UI-Chat sind pixel_values
            # aktuell immer None (tokenizer-Pfad), Bilder gehen als Text.
            is_multimodal_inputs = inputs.get("pixel_values") is not None
            if use_long_stream and not is_multimodal_inputs:
                _run_long_stream()
            elif use_chunked_stream and is_multimodal_inputs:
                gen_kwargs["use_cache"] = False
                print(f"[chat_tab] multimodal + long context "
                      f"(T={inputs['input_ids'].shape[1]}) → use_cache=False "
                      f"Fallback (kein chunked+vision)", flush=True)
                model.generate(**gen_kwargs)
            elif use_chunked_stream:
                _run_chunked_stream()
            else:
                try:
                    model.generate(**gen_kwargs)
                except Exception as _exc:
                    # OOM-Guard (Live-Crash 2026-10-05): ein CUDA-OOM darf
                    # den Server NICHT killen (crash_handler/threading-
                    # excepthook hatte den Prozess terminiert). Andere
                    # Fehler propagieren unverändert (Crash-Policy).
                    if not _is_cuda_oom(_exc):
                        raise
                    oom_error = _exc
                    try:
                        streamer.end()
                    except Exception:
                        pass
        finally:
            # KV-4Bit-Prefill hinterlässt große Allocator-Blöcke
            # (generators._px_pre_generation_cache_flush-Analogon).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            manager.unlock_model(model_id)

    thread = Thread(target=generate_with_lock)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        if len(partial_text) % 20 == 0:
             print(f"DEBUG: Yielding partial_text length: {len(partial_text)}")
        yield partial_text

    # 4. Record Telemetry
    px_metrics = manager.get_px_metrics(model_id)
    telemetry.record(
        model_id=model_id,
        prompt_tokens=inputs["input_ids"].shape[1],
        completion_tokens=len(tokenizer.encode(partial_text)),
        px_metrics=px_metrics
    )

    # Plan 2026-10-05 (OOM nach TXT-Anhang): sichtbare Diagnose im Chat.
    # Die Note wird NICHT in die Session persistiert (kein Modell-Output —
    # sie würde sonst im nächsten Kontext wieder auftauchen): nur die
    # Chatbot-Anzeige (display_text) bekommt sie, save_session schreibt
    # den reinen Modell-Output.
    display_text = partial_text
    if oom_error is not None:
        print(f"[chat_tab] OOM-Guard: Generierung abgebrochen, Server läuft "
              f"weiter (prompt={inputs['input_ids'].shape[1]} Token)",
              flush=True)
        display_text = partial_text + _oom_note_text(
            int(inputs["input_ids"].shape[1]))
        yield display_text

    # 5. Save session on completion (Plan 2026-10-05: + komplette Einstellung)
    full_history = messages + [{"role": "assistant", "content": partial_text}]
    save_session(session_id, full_history, model_id=model_id, settings=chat_settings)


# ── Lang-Kontext-Dispatch-Helper (Plan 2026-10-05, OOM nach TXT-Anhang) ──
# Bewusst NACH chat_fn und als echte Funktionen (keine Closures) — für
# tests/test_chat_long_ctx_dispatch.py runtime-testbar ohne GPU. Die
# Übersetzung folgt exakt generators.stream_chat_completion (_long_worker,
# Stufe-3e): model.generate-Welt → long_context.generate_long-Welt.

def _long_ctx_generate_kwargs(gen_kwargs):
    """generate_long-kwargs aus model.generate-kwargs ableiten.

    Doppel-Args vermeiden (input_ids/streamer/max_new_tokens werden von
    _run_long_stream explizit übergeben), Tensor-Keys + HF-Generation-Junk
    wegwerfen (decode_loop kennt sie nicht), top_k default 0 (HF-Parität:
    UI setzt nur top_p; Temperatur kommt per Slider durch — RTPF-A4;
    multinomial läuft bei 1e-10 greedy-exakt).
    generate_long filtert sample_cfg selbst — do_sample & Co. dürfen
    durchgelassen werden (werden mit verbose-Hinweis gedroppt).
    """
    g = dict(gen_kwargs)
    g.pop("max_new_tokens", None)
    g.pop("streamer", None)
    g.pop("input_ids", None)
    for tensor_key in ("attention_mask", "token_type_ids", "inputs_embeds"):
        g.pop(tensor_key, None)
    for junk in ("stop_strings", "tokenizer", "stopping_criteria",
                 "_px_use_chunked_prefill", "_px_use_long_ctx",
                 "use_cache", "pad_token_id", "_input_len",
                 # Plan 2026-10-05 (Gemma4 Thinking-Budget): defensiv —
                 # LogitsProcessor hat keine generate_long-Route.
                 "logits_processor"):
        g.pop(junk, None)
    g.setdefault("top_k", 0)
    return g


def _is_cuda_oom(exc):
    """True für CUDA-OutOfMemory (torch ≥ 2.5: OutOfMemoryError-Klasse;
    älter: RuntimeError mit 'out of memory' im Text)."""
    _oom_cls = getattr(torch, "OutOfMemoryError", None)
    if _oom_cls is not None and isinstance(exc, _oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _oom_note_text(prompt_tokens):
    """Sichtbare OOM-Diagnose im Chat (NICHT in die Session persistiert —
    kein Modell-Output, sie würde sonst im nächsten Kontext auftauchen)."""
    return (
        "\n\n⚠️ CUDA OOM: Generierung abgebrochen "
        f"(Prompt: {prompt_tokens} Token) — der Server läuft weiter. "
        "Hinweis: nicht benötigte Modelle im Modell-Tab entladen, "
        "um VRAM freizugeben."
    )


# ── Thinking-Template-Kontext (Phase 3, 2026-10-05) ─────────────────────
# chat_fn baut den Jinja-Extra-Kontext für apply_chat_template. Bewusst
# NACH chat_fn platziert (echte Funktion, kein Closure — für
# tests/test_thinking_toggle.py runtime-testbar) — Kapabilitäts-Gate
# liegt in px_defaults.get_thinking_defaults, die extras-Weiterreichung
# an generators._chat_template_kwargs (gleiche Semantik wie Bridge/Server
# aus Task #15: enable_thinking / reasoning_effort).

def _thinking_template_kwargs(model_id, thinking, thinking_effort):
    """Chat-Template-Extras für thinking/thinking_effort (kapabilitäts-gegated).

    Rückgabe (kwargs für tokenizer.apply_chat_template):
        {}                                            — nicht capable ODER
                                                        thinking=None (UI-Null)
        {"enable_thinking": True|False}               — gemma4 (Budget läuft
                        separat: App-Level LogitsProcessor, s.o.)
        {"enable_thinking": ..., "reasoning_effort": <str>} — bonsai (Stufe)

    Bonsai-Template-Semantik: `enable_thinking is undefined or is true` →
    denken; reasoning_effort|default('xhigh'). Das Toggle-Widget liefert
    IMMER einen bool → kein undefined-Pfad im Chat. Effort-Validierung:
    nur Stufen aus px_defaults.get_thinking_defaults(model_id)["efforts"]
    werden durchgereicht, alles andere → None (Template-Default 'xhigh'
    greift) — kein raise_exception-Crash im Template.

    Plan 2026-10-05 (Gemma4 Thinking-Budget): das Budget ist KEIN Template-
    Extra — max_thinking_tokens wird App-Level über
    generators.ThinkingBudgetLogitsProcessor realisiert (echte Kanal-Tokens
    <|channel>/thought/<channel|>), nicht über den Jinja-Kontext. Deshalb
    nimmt diese Funktion bewusst KEINEN Budget-Parameter.
    """
    cap = get_thinking_defaults(model_id) if isinstance(model_id, str) else None
    if cap is None or thinking is None:
        return {}
    effort = thinking_effort if (thinking_effort in (cap["efforts"] or ())) else None
    from generators import _chat_template_kwargs
    return _chat_template_kwargs(thinking=bool(thinking), thinking_effort=effort)


# ── Sidebar System-Prompt Helper (Plan 2026-07-08) ───────────────────
# Diese Helper werden als click/change-Handler in der Sidebar gemountet.
# Sie sind pure-Funktionen (kein Side-Effect außer den Rückgabewerten)
# und werden von tests direkt getestet.

def on_preset_change_load_profile(preset: str) -> tuple:
    """px_preset.change-Handler (LEGACY, nicht mehr gemountet seit 2026-07-09).

    Lädt Profil-Name + Profil-Body passend zum PX-Preset. Wurde am
    2026-07-09 entkoppelt: User will, dass px_preset NICHT mehr
    automatisch citmind/juexin lädt. Bleibt als pure-Funktion
    erhalten, damit ggf. Tests und Backward-Compat gewahrt sind.
    """
    from gradio_tabs.system_prompt import (
        preset_to_profile,
        load_profile_for_preset,
    )
    profile_name = preset_to_profile(preset)
    profile_body = load_profile_for_preset(preset)
    return profile_name, profile_body


# ── Profil-Body-Load-Suppress (Plan 2026-10-05) ──────────────────────────
# Gradio feuert .change auch bei PROGRAMMATISCHEN Widget-Updates — der
# Session-Restore (restore_session_settings) rendert also das Profil-
# Dropdown und würde damit den Profil-Body in die (evtl. frei editierte,
# aus der Session wiederhergestellte) Textarea schieben. Gegen-Wehrturm:
# Fire-Counter — der Restore zählt (nur bei abweichendem Profilwert; gleiche
# Werte feuern kein .change, also kein Zähler-Leak), der change-Handler
# konsumiert einen Token und no-op't via gr.skip().
# Test-Reset: _reset_profile_suppress_for_tests().
_PROFILE_BODY_SUPPRESS_LOCK = threading.Lock()
_PROFILE_BODY_SUPPRESS_COUNT = 0


def _suppress_profile_body_load_once() -> None:
    global _PROFILE_BODY_SUPPRESS_COUNT
    with _PROFILE_BODY_SUPPRESS_LOCK:
        _PROFILE_BODY_SUPPRESS_COUNT += 1


def _profile_body_load_suppressed() -> bool:
    global _PROFILE_BODY_SUPPRESS_COUNT
    with _PROFILE_BODY_SUPPRESS_LOCK:
        if _PROFILE_BODY_SUPPRESS_COUNT > 0:
            _PROFILE_BODY_SUPPRESS_COUNT -= 1
            return True
        return False


def _reset_profile_suppress_for_tests() -> None:
    global _PROFILE_BODY_SUPPRESS_COUNT
    with _PROFILE_BODY_SUPPRESS_LOCK:
        _PROFILE_BODY_SUPPRESS_COUNT = 0


def on_profile_change_load_body(profile_name: str) -> str:
    """system_profile.change-Handler: lädt Profil-Body in die Textarea.

    Plan 2026-07-09: User klickt in der Sidebar auf einen Profil-Eintrag
    (z.B. "juexin", "citmind", "neutral") → der gerenderte Body wird
    in das system_prompt_text-Feld geladen. Der User kann danach den
    Text frei editieren — was auch immer in der Textarea steht, geht
    in den nächsten Chat (build_system_message nutzt edit_text, wenn
    vorhanden, sonst den Profil-Body).

    Plan 2026-10-05 (Session-Settings-Restore): Programmatische Profil-
    Updates (Session-Load) feuern dieses .change ebenfalls — hier wird
    EIN Suppress-Token konsumiert (siehe restore_session_settings) und
    die Textarea via gr.skip() UNBERÜHRT gelassen; die wiederhergestellte
    Textarea bleibt Source-of-Truth.

    Returns:
        profile_body — String für gr.update(value=...) auf der Textarea
        (oder gr.skip() bei programmatischem Restore-Fireing).
    """
    if _profile_body_load_suppressed():
        return gr.skip()
    from gradio_tabs.system_prompt import load_profile_body
    return load_profile_body(profile_name)


def on_reset_prompt_click() -> str:
    """Reset-Button click-handler: leert die Edit-Textbox.

    User klickt "↺ Reset auf Profil-Default" → das Edit-Text-Feld wird
    geleert, das Profil selbst bleibt unverändert. Nach dem Reset
    wird beim nächsten chat_fn der Profil-Body (statt Edit-Override)
    verwendet (siehe build_system_message in system_prompt.py).
    """
    return ""


def build_chat_tab(manager: ModelManager):
    """Build and return the Chat tab components."""

    # ── Client-side state ──
    session_id_state = gr.BrowserState(default_value=None, storage_key="px_session_id")


    model_choices = list(MODEL_REGISTRY.keys())

    # Plan ui-styling Task #183: position="right" verschiebt die Sidebar
    # auf die rechte Seite (Standard in Gradio 6.x ist "left"; rechts ist
    # moderner für Chat-UIs). width=320 ist Default und explizit gesetzt.
    # Plan ui-styling Task #184: Status-Pill oben (live-Anwendung von
    # _styles.pill_style — accent=Hintergrund, pill=Radius, weißer Text).
    with gr.Sidebar(label="PX Controls", position="right", width=320):
        from gradio_tabs._styles import pill_style
        gr.HTML(
            f"<div style='{pill_style('Engine: PX', 'accent')}'>Engine: PX ✓</div>"
        )
        gr.Markdown("### Model Selection")
        model_select = gr.Dropdown(
            choices=model_choices,
            value=model_choices[0],
            label="Current Model",
        )
        px_preset = gr.Dropdown(
            choices=["BASELINE", "ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN", "ACTIVE_MANIFOLD_RELAY"],
            value="ACTIVE_MANIFOLD_RELAY",
            label="PX Mode Preset",
        )

        with gr.Accordion("Parameters", open=False):
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top P")
            max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Max Tokens")
            rep_p = gr.Slider(1.0, 2.0, value=1.15, step=0.05, label="Repetition Penalty")
            px_gamma = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="PX Gamma")
            # Phase 3 (2026-10-05): Thinking-Toggle + Budget. Initial-
            # Sichtbarkeit/Werte aus dem Start-Modell (model_choices[0]);
            # jeder Modellwechsel rendert alles neu (apply_px_defaults).
            # bonsai-27b: enable_thinking + reasoning_effort (qwen3.5-Template,
            # Budget ALS STUFE — der Modell-Doku gemäß). gemma4-e2b:
            # enable_thinking + thinking_budget (max_thinking_tokens-Semantik,
            # App-Level LogitsProcessor — im installierten transformers-Stack
            # existiert der generate()-Parameter nicht, PR #42112 ungemerged).
            # gemma3*/minicpm5/Unbekannt: Templates/Kanal-Infrastruktur fehlt
            # → versteckt.
            _tinit = get_thinking_defaults(model_choices[0])
            thinking = gr.Checkbox(
                value=bool(_tinit["default"]) if _tinit else False,
                visible=_tinit is not None,
                label="Thinking (enable_thinking)",
                info=(
                    "Template-Variable enable_thinking: gemma4 ON= <|think|> im "
                    "System-Turn; bonsai ON=denken, OFF=forced-closed think-block."
                ),
            )
            thinking_budget = gr.Slider(
                minimum=(_tinit["budget_range"][0]
                         if _tinit and _tinit["budget_range"] else 0),
                maximum=(_tinit["budget_range"][1]
                         if _tinit and _tinit["budget_range"] else 8192),
                step=64,
                value=(_tinit["budget_default"]
                       if _tinit and _tinit["budget_default"] else 0),
                visible=bool(_tinit and _tinit["budget_range"]),
                label="Thinking Budget (max_thinking_tokens)",
                info=(
                    "gemma4: Token-Budget im thought-Kanal (<|channel>…"
                    "<channel|>) — bei Überschreitung wird <channel|> "
                    "erzwungen (App-Level LogitsProcessor). 0 = unbegrenzt; "
                    "wirkt bei Thinking an."
                ),
            )
            thinking_effort = gr.Radio(
                choices=(list(_tinit["efforts"]) if _tinit and _tinit["efforts"]
                         else ["xhigh", "medium", "low"]),
                value=(str(_tinit["effort_default"])
                       if _tinit and _tinit["effort_default"] else "xhigh"),
                visible=bool(_tinit and _tinit["efforts"]),
                label="Thinking Budget (reasoning_effort)",
                info=(
                    "bonsai-27b: Denktiefe als STUFE (xhigh/medium/low; "
                    "Template-Default xhigh) — das ist der etablierte "
                    "Budget-Parameter dieses Modells."
                ),
            )

        with gr.Accordion("verstärkbar Relay (seite15)", open=False):
            gr.Markdown(
                "Re-Injektion der modell-eigenen Zustands-Richtung `d_width` am "
                "post-recur Layer (Motor unangetastet, forward_hook). Wirksam mit "
                "**ACTIVE_MANIFOLD_RELAY** (default sign=+1) oder sign≠0 auf jedem "
                "Preset. Beim Modellwechsel werden Richtung/Alpha/Layer (incl. "
                "Slider-Bounds) automatisch per d_width-Artefakt gesetzt — "
                "Artefakte für gemma3-270m-it / 1b-it / 1b-pt / 4b-it / 4b-pt / "
                "gemma4-E2B / ternary-27b existieren; MiniCPM5→ relay no-op, "
                "LEAN-Engine läuft. siehe scratches/psychomotrik/LESUNG15.md"
            )
            relay_sign = gr.Radio(
                choices=[("+1  (WIDE / expansiv / aktiv)", 1),
                         (" 0  (relay off)", 0),
                         ("-1  (NARROW / eng / still)", -1)],
                value=1, label="Relay Richtung (sign)"
            )
            relay_alpha = gr.Slider(0.0, 1.5, value=0.30, step=0.05,
                                    label="Relay Alpha (Bruchteil L21-Norm; kohärenter Chat ~0.30, seite15-stark=0.5)")
            # Init-Bounds aus dem Start-Modell (model_choices[0]) — je
            # Modell setzen die apply_px_defaults-Handler die Bounds neu
            # (maximum=n_layers). Der alte starre Slider (1..25) machte
            # ternary L34 / E2B L26 unerreichbar.
            _init_defaults = get_px_defaults(model_choices[0]) or {}
            _init_layers = int(_init_defaults.get("n_layers") or 25)
            _init_inject = int(_init_defaults.get("inject_layer") or 21)
            relay_layer = gr.Slider(
                1, _init_layers, value=_init_inject, step=1,
                label="Relay Injektions-Layer",
            )

        gr.Markdown("---")
        # Plan 2026-07-08: System-Prompt prominent in der Sidebar.
        # User-Wahl: Einstellungs-Tab weg, System-Prompt NUR in Sidebar.
        # Beim Wechsel des px_preset wird automatisch der passende
        # System-Prompt in die Textarea geladen (preset_to_profile +
        # load_profile_for_preset in system_prompt.py).
        from gradio_tabs.system_prompt import (
            list_profiles as _list_profiles,
            load_profile_for_preset as _load_profile_for_preset,
        )
        _profile_choices = _list_profiles()
        with gr.Accordion("System-Prompt (Frame-Orientierer)", open=True):
            gr.Markdown(
                "Frame = Orientierer, nicht 观-Produzent. Klicke auf "
                "'Profil' um einen Default in die Textarea zu laden, oder "
                "schreibe direkt deinen eigenen Text. Der Textarea-Inhalt "
                "geht direkt in den nächsten Chat."
            )
            system_profile = gr.Dropdown(
                choices=_profile_choices,
                value="neutral" if "neutral" in _profile_choices else _profile_choices[0],
                label="Profil",
                info="citmind=PX-Frame, juexin=RELAY/Kontemplation, neutral=kein Frame",
            )
            system_prompt_text = gr.Textbox(
                label="System-Prompt (editierbar)",
                placeholder=(
                    "Klicke unten auf 'Profil' um einen Default zu laden, oder "
                    "schreibe direkt deinen eigenen Text. Leer = kein System-Prompt."
                ),
                lines=6,
                info=(
                    "Source-of-Truth: was hier steht, geht in den Chat. "
                    "Profil-Auswahl überschreibt das Edit-Feld."
                ),
            )
            reset_prompt_btn = gr.Button(
                "↺ Reset auf Profil-Default",
                size="sm",
                variant="secondary",
            )

        gr.Markdown("---")
        gr.Markdown("### Sessions")
        new_session_btn = gr.Button("New Session", variant="secondary")
        with gr.Row():
            session_dropdown = gr.Dropdown(choices=list_sessions(), label="Saved Sessions", scale=4)
            refresh_sessions_btn = gr.Button("🔄", scale=1)
        load_session_btn = gr.Button("Load Selected", size="sm")
        session_id_display = gr.Textbox(label="Current ID", interactive=False)
        
        gr.Markdown("---")
        export_btn = gr.Button("Download JSON", size="sm")
        export_file = gr.File(label="Export", visible=False)
        import_file = gr.File(label="Import JSON", file_types=[".json"])
        import_btn = gr.Button("Import & Load", size="sm")

    # ── Chat Components ──
    # Plan ui-styling Task #182: Markdown-Renderer + Bubble-Alignment.
    # render_markdown=True: Assistant-Antworten mit ``` Code-Blöcken werden
    #   gerendert (default).
    # sanitize_html=True: HTML in Messages wird escaped (default).
    # line_breaks=True: Newlines in Plain-Text bleiben sichtbar.
    # group_consecutive_messages=True: aufeinanderfolgende gleiche Rollen
    #   werden zu einer Bubble gruppiert (default in 6.15.2).
    # avatar_images=(None, None): kein Bild-Avatar (Plan-Default).
    # Hinweis: gr.Blocks(css=...) wird in app.py gesetzt (siehe _styles.py).
    chatbot = gr.Chatbot(
        autoscroll=False,
        scale=1,
        render_markdown=True,
        sanitize_html=True,
        line_breaks=True,
        group_consecutive_messages=True,
        avatar_images=(None, None),
        height=520,
        show_label=False,
    )
    
    with gr.Row():
        # Plan 2026-07-08: Bild-/Anhang-Upload im Chat. gr.MultimodalTextbox
        # gibt {"text": str, "files": [path_or_dict, ...]} zurück — wird
        # in chat_fn via normalize_multimodal_message() in einen
        # content-list mit text+image-Blöcken konvertiert. Server-Side
        # (streaming_bridge._build_image_data_url, generators._extract_images,
        # schemas) ist bereits da; nur die UI fehlte.
        # Plan 2026-10-05: "+ TXT anhängen" (User-Request) — der Adapter
        # (multimodal_input._file_block) inlined Text-Dateien (.txt, plus
        # md/py/json/csv/log/… via TEXT_EXTS) längst als ```txt-Textblock
        # (64-KiB-Cap); es fehlte nur der File-Picker-Filter. Gradio 6.15.2
        # dokumentiert gemischte Kategorien+Extensions ('image', '.json').
        msg_input = gr.MultimodalTextbox(
            placeholder="Type a message, or attach an image or .txt file…",
            show_label=False,
            scale=9,
            container=False,
            file_count="multiple",
            file_types=["image", ".txt"],
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    # Plan ui-styling 2026-07-08: Undo-Button umbenannt von "Undo Last Turn"
    # zu "Undo Last Message" — semantisch korrekt: poppt nur 1 Element
    # (User oder Agent), nicht das ganze (user, assistant)-Paar. Eigene Zeile
    # unter dem Input, damit er optisch von der Haupt-Action (Send) getrennt
    # ist. Status-Markdown zeigt "✓ Undone" / "⚠ Nothing to undo".
    with gr.Row():
        undo_btn = gr.Button("↶ Undo Last Message", size="sm", variant="secondary")
        undo_status = gr.Markdown("")

    # ── Logic ──

    def user_message(message, history):
        # Plan 2026-07-08: Bild-Upload. MultmodalTextbox liefert dict
        # {"text": ..., "files": [...]} (oder str bei reiner Text-Eingabe).
        # Normalisierung in multimodal_input.normalize_multimodal_message
        # damit der Chat-Eintrag exakt das Format hat das chat_fn erwartet.
        if is_empty_message(message):
            return gr.update(), history
        content = normalize_multimodal_message(message)
        return gr.update(value=None), history + [{"role": "user", "content": content}]

    def bot_response(history, model_id, px_preset, temp, tp, mt, rp, gamma,
                     thinking, thinking_budget, thinking_effort,
                     relay_sign, relay_alpha, relay_layer,
                     system_profile, system_prompt_text,
                     session_id):
        # 2. Call our core chat_fn and yield full updated history
        # We pass history (which now has the user message) to chat_fn
        # But chat_fn also does its own history recovery if needed.
        # To avoid duplication, we ensure chat_fn sees the 'true' state.

        # Generator for streaming updates
        generator = chat_fn(
            message=history[-1]["content"], # Last message is the user message
            history=history[:-1],           # Everything before is the history
            model_id=model_id,
            px_preset=px_preset,
            temp=temp,
            tp=tp,
            mt=mt,
            rp=rp,
            gamma=gamma,
            thinking=thinking,
            thinking_budget=thinking_budget,
            thinking_effort=thinking_effort,
            relay_sign=relay_sign,
            relay_alpha=relay_alpha,
            relay_layer=relay_layer,
            system_profile=system_profile,
            system_prompt_text=system_prompt_text,
            session_id=session_id,
            manager=manager
        )
        
        # Since chat_fn now yields only partial_text (string), 
        # we need to append it to history for the UI
        current_history = list(history)
        current_history.append({"role": "assistant", "content": ""})
        
        for partial_text in generator:
            current_history[-1]["content"] = partial_text
            yield current_history

    # Events
    msg_input.submit(
        fn=user_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False
    ).then(
        fn=bot_response,
        inputs=[chatbot, model_select, px_preset, temperature, top_p, max_tokens, rep_p, px_gamma, thinking, thinking_budget, thinking_effort, relay_sign, relay_alpha, relay_layer, system_profile, system_prompt_text, session_id_state],
        outputs=[chatbot]
    )

    submit_btn.click(
        fn=user_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False
    ).then(
        fn=bot_response,
        inputs=[chatbot, model_select, px_preset, temperature, top_p, max_tokens, rep_p, px_gamma, thinking, thinking_budget, thinking_effort, relay_sign, relay_alpha, relay_layer, system_profile, system_prompt_text, session_id_state],
        outputs=[chatbot]
    )

    # ── Internal connections ──
    
    new_session_btn.click(
        fn=handle_new_session,
        outputs=[session_id_state, chatbot, session_dropdown, session_id_display]
    )
    
    # Plan 2026-10-05 (Session-Settings-Restore): Load & Import rendern die
    # in der session.json gespeicherte Einstellung zurück in die 15 Widgets —
    # Outputs-Reihenfolge = SETTINGS_WIDGET_FIELDS (chat_tab.py-Header).
    # handle_new_session bewusst OHNE Restore: frische Session, die Widgets
    # bleiben wie sie gerade stehen.
    _settings_restore_out = [
        model_select, px_preset, temperature, top_p, max_tokens, rep_p,
        px_gamma, thinking, thinking_budget, thinking_effort, relay_sign,
        relay_alpha, relay_layer, system_profile, system_prompt_text,
    ]
    load_session_btn.click(
        fn=handle_load_saved,
        inputs=[session_dropdown],
        outputs=[session_id_state, chatbot, session_dropdown, session_id_display]
    ).then(
        fn=restore_session_settings,
        inputs=[session_id_state, system_profile],
        outputs=_settings_restore_out,
    )

    export_btn.click(fn=handle_export, inputs=[session_id_state, chatbot], outputs=[export_file])
    import_btn.click(
        fn=handle_import,
        inputs=[import_file],
        outputs=[session_id_state, chatbot, session_dropdown, session_id_display]
    ).then(
        fn=restore_session_settings,
        inputs=[session_id_state, system_profile],
        outputs=_settings_restore_out,
    )
    refresh_sessions_btn.click(fn=handle_refresh, outputs=[session_dropdown])

    # Plan ui-styling 2026-07-06: Undo-Button click-handler.
    # Outputs: chatbot (gekürzte History) + undo_status (Feedback).
    undo_btn.click(
        fn=handle_undo,
        inputs=[session_id_state, chatbot],
        outputs=[chatbot, undo_status],
    )

    # ── Session-Settings-Roundtrip (Plan 2026-10-05) ─────────────────────
    # (a) Auto-Defaults: User wählt Modell oder px_preset (.input = nur echte
    #     User-Aktionen; programmatische Restore-Updates feuern kein .input) →
    #     per-Model-Defaults in die Relay-/Gamma-/Thinking-Controls (incl.
    #     Slider-Bounds maximum=n_layers aus gradio_tabs/px_defaults.py;
    #     Thinking Sichtbarkeit+Modell-Default) + persistiert in die
    #     session.json. Suppress-Regel gegen Restore-Freing in apply_px_defaults.
    for _owner in (model_select, px_preset):
        _owner.input(
            fn=apply_px_defaults,
            inputs=[model_select, px_preset, session_id_state],
            outputs=[relay_sign, relay_alpha, relay_layer, px_gamma,
                     thinking, thinking_budget, thinking_effort],
        )

    # (b) Persistenz aller freien User-Felder (debounced 400ms via
    #     settings_persist.schedule_settings_save). model_id/px_preset/relay_*/
    #     px_gamma/thinking persisten zusätzlich durch (a) und beide
    #     chat_fn-Save-Points; dieser Loop deckt auch die Fälle ab, in denen
    #     User Werte hand-anpasst, ohne das Modell zu wechseln. outputs-frei
    #     (fire-and-forget).
    for _field, _wid in (
        ("temperature", temperature), ("top_p", top_p),
        ("max_tokens", max_tokens), ("rep_p", rep_p),
        ("px_gamma", px_gamma),
        ("thinking", thinking), ("thinking_budget", thinking_budget),
        ("thinking_effort", thinking_effort),
        ("relay_sign", relay_sign), ("relay_alpha", relay_alpha),
        ("relay_layer", relay_layer),
        ("system_profile", system_profile),
        ("system_prompt_text", system_prompt_text),
    ):
        _wid.input(
            fn=_persist_setting_field(_field),
            inputs=[session_id_state, _wid],
            outputs=None,
        )

    # Plan 2026-07-09: Preset und System-Prompt sind komplett entkoppelt.
    # User-Entscheidung 2026-07-09: "ich will nicht automatisch citmind
    # oder juexin paden, default soll der systemprompt leer sein. und
    # wenn ich auf die systemprompt liste einen eintrag klicke, dann
    # soll der geladen werden, ins systemprompt feld. und von dort soll
    # er immer für den chat benutzt werden, also wenn ich da was ändere,
    # dass es gleich auswirkungen auf den chat danach hat".
    #
    # → px_preset wählt NUR den PX-Engine-Modus (BASELINE/ACTIVE_MANIFOLD/
    #   ACTIVE_MANIFOLD_RELAY), ändert NICHT den System-Prompt.
    # → system_profile wählt NUR den Profil-Default, lädt dessen Body in
    #   die Textarea (überschreibt aktuelles Edit).
    # → system_prompt_text ist die Source-of-Truth: was dort steht, geht
    #   in den Chat (build_system_message nutzt edit_text, wenn vorhanden).
    # → reset_prompt_btn leert die Textarea (Profil-Default wird beim
    #   nächsten system_profile.change() neu geladen).
    # Plan 2026-10-05: nach dem Profil-Body-Load persistiert der .then die
    # neue Textarea (die .input-Persist oben feuert NICHT — der Textarea-
    # Update ist programmatisch). Gleiche Kette für den Reset-Button (""
    # → persistiert, sonst lebt der alte Text in der session.json weiter).
    system_profile.change(
        fn=on_profile_change_load_body,
        inputs=[system_profile],
        outputs=[system_prompt_text],
    ).then(
        fn=_persist_setting_field("system_prompt_text"),
        inputs=[session_id_state, system_prompt_text],
        outputs=None,
    )
    reset_prompt_btn.click(
        fn=on_reset_prompt_click,
        inputs=[],
        outputs=[system_prompt_text],
    ).then(
        fn=_persist_setting_field("system_prompt_text"),
        inputs=[session_id_state, system_prompt_text],
        outputs=None,
    )

    # Plan 2026-10-05: 19er-Tupel — die ersten 4 wie bisher (app.py-Unpack),
    # danach die 15 Settings-Widgets in EXAKT der SETTINGS_WIDGET_FIELDS-
    # Reihenfolge (app.py demo.load-Outputs + .then-Chains spiegeln das).
    return (
        session_id_state, chatbot, session_dropdown, session_id_display,
        model_select, px_preset, temperature, top_p, max_tokens, rep_p,
        px_gamma, thinking, thinking_budget, thinking_effort, relay_sign,
        relay_alpha, relay_layer, system_profile, system_prompt_text,
    )
