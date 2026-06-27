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
from typing import Optional, List, Dict, Any
from threading import Thread
from transformers import TextIteratorStreamer

from config import MODEL_REGISTRY
from model_manager import ModelManager
from sessions import (
    save_session, load_session, get_new_session_id, list_sessions,
    update_settings, list_session_mtimes, SETTINGS_FIELDS,
)
from telemetry import telemetry
from gradio_tabs.auto_tune_defaults import (
    calibrated_gamma, calibrated_values, resolve_for_backend, AUTO_TUNABLE_PARAMS,
)
from gradio_tabs.chat_actions import undo_last_turn, can_undo
from gradio_tabs.multimodal_input import (
    normalize_multimodal_message, is_empty_message, extract_text_blocks,
)
from gradio_tabs.system_prompt import (
    list_profiles as _list_profiles, build_system_message, inject_into_messages,
    merge_system_into_first_user, append_tag_snippet,
)
from gradio_tabs.vocoder_tags import (
    render_tag_system_prompt as _render_tag_snip,
    strip_tags_for_engine as _strip_tags_for_engine,
    tag_density_warning as _tag_density_warning,
)
from gradio_tabs.tts_engine import (
    make_engine as _tts_make_engine, list_available_engines as _tts_list_available,
)
from gradio_tabs.chat_settings import (
    SettingsDebouncer, settings_from_widgets,
    widget_updates_from_settings,
)


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


def _normalize_history_for_chatbot(history):
    """Normalize a persisted history list so it survives Gradio's
    ``Chatbot._check_format`` + ``_postprocess_content``.

    Two failure modes observed when loading legacy sessions:

    1. ``content`` is a list of dicts WITHOUT the ``"type"`` key
       (e.g. ``[{"text": "[SYSTEM CONTEXT]\\n..."}]``). Gradio's
       ``_postprocess_content`` rejects these in its ``else: raise
       ValueError`` branch — the resulting error surfaces as
       ``"Data incompatible with messages format"`` even though the
       outer ``role``+``content`` keys are present.

    2. ``content`` is a plain ``str`` with embedded newlines or other
       control chars — usually fine, but we strip null bytes defensively.

    The function:
      - Drops None / non-dict entries.
      - Coerces ``content=list`` whose blocks are missing ``type`` into
        a single concatenated string (via ``extract_text_blocks``).
      - Preserves valid multimodal lists (``type=="text"|"file"|...``).
      - Returns a NEW list — does not mutate the input.
    """
    if not history:
        return []
    out = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "user", "assistant"):
            continue
        if content is None:
            continue
        if isinstance(content, list):
            # Inspect each block: if any block has a recognised ``type``,
            # keep the list. Otherwise collapse to a single string.
            has_typed_block = any(
                isinstance(b, dict) and b.get("type") in ("text", "file", "component")
                for b in content
            )
            if has_typed_block:
                out.append({"role": role, "content": content})
            else:
                # Either no type keys, or unknown shapes — flatten.
                text = extract_text_blocks(content)
                if text:
                    out.append({"role": role, "content": text})
                # else: drop empty content.
        elif isinstance(content, str):
            # Strip null bytes; otherwise pass through.
            cleaned = content.replace("\x00", "")
            if cleaned:
                out.append({"role": role, "content": cleaned})
        elif isinstance(content, dict):
            # Single dict — coerce to its ``text`` field if present.
            text = content.get("text") or extract_text_blocks(content)
            if text:
                out.append({"role": role, "content": str(text)})
        else:
            # Fallback: stringify.
            out.append({"role": role, "content": str(content)})
    return out

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

def on_load(session_id):
    """Called when the page loads. Auto-resumes the most-recent session
    if ``session_id`` is empty (BrowserState was cleared)."""
    if session_id is None or session_id == "":
        candidates = list_session_mtimes()
        if candidates:
            session_id = candidates[0][1]  # mtime-newest
        else:
            session_id = get_new_session_id()

    data = load_session(session_id) or {"history": [], "settings": {}}
    history = _normalize_history_for_chatbot(data.get("history", []))
    # Return order MUST match the ``outputs=`` list of demo.load /
    # load_session_btn.click / import_btn.click in build_chat_tab:
    #   [session_id_state, chatbot, session_dropdown, session_id_display, ...]
    return session_id, history, gr.update(choices=list_sessions()), session_id


def handle_new_session():
    new_id = get_new_session_id()
    return new_id, [], gr.update(choices=list_sessions()), new_id


def _settings_widget_updates(settings_dict):
    """Return 15 gr.update kwargs in canonical widget order. Used by
    handle_load_saved and handle_import to restore widget values from
    persisted session data. Order matches the ``widget_field_map`` in
    ``build_chat_tab`` (so app.py outputs align)."""
    return widget_updates_from_settings(settings_dict)


def handle_load_saved(session_id):
    """Load a saved session into all 19 outputs. Order MUST match the
    ``outputs=`` list wired in build_chat_tab:

        [session_id_state, chatbot, session_dropdown, session_id_display,
         model_select, px_preset, auto_tune_cb, temperature, top_p,
         max_tokens, rep_p, px_gamma, relay_sign, relay_alpha, relay_layer,
         system_profile, system_prompt_text, tts_engine_dd,
         tts_sample_rate_dd, tts_auto_cb]
    """
    if not session_id:
        return (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            *(gr.skip() for _ in range(15)),
        )
    data = load_session(session_id)
    if data is None:
        # Session-Datei fehlt — leere UI, kein Crash.
        return (
            session_id, [], gr.update(choices=list_sessions()), session_id,
            *(gr.skip() for _ in range(15)),
        )
    history = _normalize_history_for_chatbot(data.get("history", []))
    settings = data.get("settings", {})
    updates = _settings_widget_updates(settings)
    # System-Prompt-Preview aus dem ersten System-Eintrag in der History
    # extrahieren (defensiv: Multimodal-Listen möglich). Wird NICHT an
    # einen Output gebunden (kein sys_preview-Widget in der aktuellen
    # Sidebar) — Logging bleibt via _normalize_history_for_chatbot.
    return (
        session_id, history, gr.update(choices=list_sessions()), session_id,
        updates["model_id"], updates["px_preset"],
        updates["auto_tune"], updates["temperature"],
        updates["top_p"], updates["max_tokens"], updates["rep_p"],
        updates["px_gamma"], updates["relay_sign"],
        updates["relay_alpha"], updates["relay_layer"],
        updates["system_profile"], updates["system_prompt_text"],
        updates["tts_engine"], updates["tts_sample_rate"],
        updates["tts_auto"],
    )


def handle_export(session_id, history):
    """Write {session_id, exported_at, settings, history} to a JSON file
    and return it via gr.File."""
    if not session_id:
        return gr.update(visible=False)
    data = load_session(session_id) or {}
    settings = data.get("settings", {})
    history_to_export = data.get("history", history or [])
    payload = {
        "session_id": session_id,
        "exported_at": _now_ts(),
        "settings": settings,
        "history": history_to_export,
    }
    from sessions import _atomic_write_json  # local; not re-exported
    out_dir = "sessions"
    out_path = os.path.join(out_dir, f"export_{session_id}.json")
    try:
        _atomic_write_json(out_path, payload)
    except Exception as e:
        print(f"[export] WARN atomic_write fehlgeschlagen: {e}")
        return gr.update(visible=False)
    return gr.update(value=out_path, visible=True)


def _now_ts() -> float:
    import time as _t
    return _t.time()


def handle_import(file_obj):
    """Restore {settings, history} from an uploaded JSON and load it."""
    if file_obj is None:
        return (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            *(gr.skip() for _ in range(15)),
        )
    try:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            payload = json.load(f)
        new_id = payload.get("session_id") or get_new_session_id()
        history = payload.get("history", [])
        settings = payload.get("settings") or {}
        save_session(new_id, history, settings=settings)
        # Anschließend komplett neu laden, damit alle Widgets gesetzt sind.
        result = handle_load_saved(new_id)
        # session_dropdown choices aktualisieren (frisch importierte SID).
        result_list = list(result)
        result_list[3] = gr.update(choices=list_sessions(), value=new_id)
        return tuple(result_list)
    except Exception as e:
        print(f"[import] WARN Import fehlgeschlagen: {e}")
        return (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            *(gr.skip() for _ in range(15)),
        )


def handle_refresh():
    return gr.update(choices=list_sessions())


def chat_fn(message, history, model_id, px_preset, temp, tp, mt, rp, gamma,
            auto_tune, system_profile, system_prompt_text,
            relay_sign, relay_alpha, relay_layer, session_id, manager: ModelManager,
            tts_engine_name="off", tts_auto=False):
    """Core chat logic with history management and model generation.

    TTS-Integration (Plan 5): ``tts_engine_name`` (piper/bark/qwen3/espeak/off)
    + ``tts_auto`` (bool). Wenn ``tts_auto=True`` und engine != "off",
    wird der Vocoder-Tag-Snip an die System-Message angehängt — das LLM
    lernt so das [#A0]/[#PAUSE]/[#WHISPER]-Vokabular und kann Tags
    einbetten. Tag-Stripping passiert erst in ``bot_response`` (für die
    TTS-Synthese), nicht hier — wir wollen, dass die Tags im
    Session-Log bleiben (für Reproduzierbarkeit).
    """
    print(f"DEBUG: history received from Gradio (UI state): {len(history) if history else 0} messages")
    # SR-64 Auto-Tune: bei auto_tune=True übernimmt das modell-kalibrierte
    # Backend (px_gamma=None → registry-gamma wie die streaming-bridge; top_p=0.9;
    # repetition_penalty=1.15). Bei False gelten die Slider-Werte des Nutzers.
    user_values = {"px_gamma": gamma, "top_p": tp, "repetition_penalty": rp,
                    "temperature": temp, "max_tokens": mt}
    resolved = resolve_for_backend(model_id, bool(auto_tune), user_values)
    _gamma_send = resolved["px_gamma"]      # None bei Auto-Tune → registry-gamma (bridge parity)
    _top_p = resolved["top_p"]
    _rp = resolved["repetition_penalty"]
    _temp = resolved["temperature"]
    _mt = resolved["max_tokens"]
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
                px_gamma=_gamma_send,
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
    processor = model_entry.get("processor")

    # 2. Build history (cleaned)
    # If history is empty (e.g. after loading a session or if save_history=False),
    # load it from the session storage to ensure continuity.
    if (not history or len(history) == 0) and session_id:
        data = load_session(session_id) or {}
        history = data.get("history", [])

    cleaned_history = _clean_history(history)
    print(f"DEBUG: Initial cleaned_history length: {len(cleaned_history)}")

    # Multimodal-Input normalisieren (Text-Dateien werden inline, Bilder als
    # image-block; reines Widget-Dict {"text","files"} → content-list/str).
    if isinstance(message, dict) and ("text" in message or "files" in message):
        actual_message = normalize_multimodal_message(message)
    else:
        actual_message = message

    messages = cleaned_history + [{"role": "user", "content": actual_message}]
    print(f"DEBUG: Combined messages length: {len(messages)}")

    # System-Prompt-Injection (psychomotrik: Frame-Orientierer, siehe
    # scratches/psychomotrik/TEST_DIALOG_STRUKTUR.md). Rangordnung:
    # Textarea-Edit (non-empty) > Dropdown-Profil > nichts (neutral no-op).
    # Persistenz: System-Message wird in messages (→ save_session) eingefügt
    # und überlebt damit Session-Load automatisch. Render: Gemma3-Chat-Template
    # kennt keine system-Rolle, daher wrappen wir das System als ersten
    # synthetischen user-turn für apply_chat_template (mit Sentinel-Prefix).
    _edit = (system_prompt_text or "").strip() or None
    _profile = system_profile if system_profile else "neutral"
    messages = inject_into_messages(messages, _profile, _edit)
    if any(m.get("role") == "system" for m in messages):
        # System-content kann Multimodal-Liste sein (Text+Image-Blöcke),
        # z.B. wenn eine alte Session einen Multimodal-System-Eintrag
        # persistiert hatte. ``extract_text_blocks`` ist die zentrale
        # Helper-Funktion für robuste String-Extraktion (Plan 5.3).
        sys_preview = next(
            extract_text_blocks(m["content"])[:60].replace("\n", " ")
            for m in messages if m["role"] == "system"
        )
        print(f"[system] profile={_profile} active (preview: {sys_preview!r}...)")

    # TTS-Tag-Snip (Plan 5): wenn Auto-Model-Vocoder aktiv ist, hängen
    # wir das Vocoder-Vokabular an die System-Message an. Das LLM lernt
    # so [#A0]/[#PAUSE]/[#WHISPER] zu produzieren. Stripping passiert
    # erst in bot_response → _synthesize_for_ui (engine-spezifisch).
    if tts_auto and tts_engine_name and tts_engine_name != "off":
        messages = append_tag_snippet(messages, _render_tag_snip())
        print(f"[tts] Tag-Snip aktiv für engine={tts_engine_name}")

    # SR-61b: Explicitly clear cache to prevent OOM on 12GB cards
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 63: Proactive Auto-save (save user message before generation)
    save_session(session_id, messages, model_id=model_id)

    # 3. Generate with streaming
    # Robustness: Flatten to strings if no images are present to satisfy text-only templates
    has_images = any(
        isinstance(m.get("content"), list) and any(isinstance(c, dict) and c.get("type") == "image" for c in m["content"])
        for m in messages
    )

    if not has_images:
        processed_messages = [{"role": m["role"], "content": _stringify_content(m["content"])} for m in messages]
    else:
        processed_messages = messages

    # System-Wrap: Gemma3-Chat-Template kennt keine system-Rolle UND erwartet
    # strikte user/assistant/user/...-Alternation. Lösung: System-Inhalt
    # in den ERSTEN user-turn prefixen (Pure-Logic in system_prompt.py).
    # Original-System-Message bleibt in messages (Persistenz+Telemetrie);
    # nur die apply_chat_template-Eingabe wird umgeformt.
    processed_messages = merge_system_into_first_user(processed_messages, _profile, _edit)

    input_text = tokenizer.apply_chat_template(processed_messages, tokenize=False, add_generation_prompt=True)

    # Bild-Vision-Pfad: wenn Bilder UND ein Multimodal-Processor vorhanden,
    # übernimmt der processor (text + pixel_values) die Embedding-Erzeugung
    # (gleicher Pfad wie streaming_bridge / generators). Text-only-Modelle
    # ohne processor können keine Bilder sehen → Bilder fallen zurück auf
    # Token-Platzhalter (degraded, mit Warnung).
    use_processor_path = has_images and processor is not None
    image_paths = []
    if use_processor_path:
        for m in messages:
            if isinstance(m.get("content"), list):
                for c in m["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        image_paths.append(c["image"])

    if use_processor_path and image_paths:
        from PIL import Image
        images_loaded = []
        for p in image_paths:
            try:
                images_loaded.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"[chat] WARN konnte Bild nicht laden {p}: {e} → übersprungen")
        if images_loaded:
            inputs = processor(text=input_text, images=images_loaded,
                                return_tensors="pt").to(model.device)
        else:
            print("[chat] WARN keine Bilder ladbar → reiner Text-Pfad")
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    else:
        if has_images and processor is None:
            print("[chat] WARN Bild-Input an Text-only-Modell (kein processor) "
                  "→ Bilder werden als Platzhalter-Token degradiert")
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(_mt),
        temperature=_temp if _temp > 0 else 1e-10,
        top_p=_top_p,
        repetition_penalty=_rp,
        do_sample=_temp > 0,
    )

    # Inject EOS/EOT and PX-specific kwargs (SR-61b: StopOnEOT criteria)
    try:
        from generators import _px_gen_kwargs, _inject_eot_eos
        gen_kwargs = _inject_eot_eos(gen_kwargs, tokenizer)
        gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
    except ImportError:
        pass

    def generate_with_lock():
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        manager.lock_model(model_id)
        try:
            model.generate(**gen_kwargs)
        finally:
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

    # 5. Save session on completion
    full_history = messages + [{"role": "assistant", "content": partial_text}]
    save_session(session_id, full_history, model_id=model_id)


def _auto_tune_updates(model_id, auto_tune_on):
    """Locking-Handler: liefert gr.update für
    (temperature, top_p, px_gamma, rep_p).
    Bei auto_tune_on=True werden die vier Slider auf die modell-kalibrierten
    Werte gesetzt und deaktiviert (interactive=False). Bei False werden sie
    nur freigegeben (Werte bleiben)."""
    from gradio_tabs.auto_tune_defaults import (
        scale_adaptive_temperature,
    )
    cv = calibrated_values(model_id)
    if auto_tune_on:
        gamma_val = cv["px_gamma"] if cv["px_gamma"] is not None else 0.08
        # temperature: scale-adaptive Fallback auf 0.7 wenn Modell unbekannt.
        temp_val = scale_adaptive_temperature(model_id)
        if temp_val is None:
            temp_val = 0.7
        return (
            gr.update(value=temp_val, interactive=False),
            gr.update(value=cv["top_p"], interactive=False),
            gr.update(value=gamma_val, interactive=False),
            gr.update(value=cv["repetition_penalty"], interactive=False),
        )
    return (
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def handle_undo(history, session_id, model_id):
    """Undo-Button: letzten Chat-Turn entfernen + Session persistieren."""
    new_history = undo_last_turn(history or [])
    if session_id:
        try:
            save_session(session_id, new_history, model_id=model_id)
        except Exception as e:
            print(f"[undo] WARN save_session fehlgeschlagen: {e}")
    return new_history


def build_chat_tab(manager: ModelManager):
    """Build and return the Chat tab components."""

    # ── Client-side state ──
    session_id_state = gr.BrowserState(default_value=None, storage_key="px_session_id")


    model_choices = list(MODEL_REGISTRY.keys())
    # Defaults (Nutzer-Vorgabe 2026-06): 1b-Modell + RELAY-Preset + Auto-Tune.
    _default_model = "gemma3-1b-it" if "gemma3-1b-it" in MODEL_REGISTRY else model_choices[0]
    _default_preset = "ACTIVE_MANIFOLD_RELAY"
    # Slider-Defaults = kalibrierte Werte des Default-Modells (bridge-parity),
    # greifen wenn Auto-Tune OFF ist. Bei ON werden sie vom Lock überschrieben.
    _cv = calibrated_values(_default_model)

    with gr.Sidebar(label="PX Controls"):
        gr.Markdown("### Model Selection")
        model_select = gr.Dropdown(
            choices=model_choices,
            value=_default_model,
            label="Current Model",
        )
        px_preset = gr.Dropdown(
            choices=["BASELINE", "ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN", "ACTIVE_MANIFOLD_RELAY"],
            value=_default_preset,
            label="PX Mode Preset",
        )

        with gr.Accordion("Parameters", open=False):
            auto_tune_cb = gr.Checkbox(
                value=True,
                label="Auto-Tune (modell-kalibrierte Parameter sperren — bridge-parity)",
                info="ON: px_gamma/top_p/repetition_penalty aus MODEL_REGISTRY + "
                     "Patch-Default (1b→γ0.12, top_p 0.9, rp 1.15). OFF: Slider frei.",
            )
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, value=_cv["top_p"], step=0.05, label="Top P")
            max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Max Tokens")
            rep_p = gr.Slider(1.0, 2.0, value=_cv["repetition_penalty"], step=0.05, label="Repetition Penalty")
            px_gamma = gr.Slider(0.0, 0.5,
                                 value=(_cv["px_gamma"] if _cv["px_gamma"] is not None else 0.08),
                                 step=0.01, label="PX Gamma")

        with gr.Accordion("verstärkbar Relay (seite15)", open=False):
            gr.Markdown(
                "Re-Injektion der modell-eigenen L16-Zustands-Richtung `d_width` am "
                "post-recur Layer (Motor unangetastet, forward_hook). Wirksam mit "
                "**ACTIVE_MANIFOLD_RELAY** (default sign=+1) oder sign≠0 auf jedem "
                "Preset. d_width-Artefakte: gemma3-1b-it (1152-dim, L16) und "
                "gemma3-4b-it (2560-dim, L15, seite19 cross-model); andere Modelle "
                "→ relay no-op, LEAN-Engine läuft. siehe scratches/psychomotrik/LESUNG15.md"
            )
            relay_sign = gr.Radio(
                choices=[("+1  (WIDE / expansiv / aktiv)", 1),
                         (" 0  (relay off)", 0),
                         ("-1  (NARROW / eng / still)", -1)],
                value=1, label="Relay Richtung (sign)"
            )
            relay_alpha = gr.Slider(0.0, 1.5, value=0.30, step=0.05,
                                    label="Relay Alpha (Bruchteil L21-Norm; kohärenter Chat ~0.30, seite15-stark=0.5)")
            relay_layer = gr.Slider(1, 25, value=21, step=1, label="Relay Injektions-Layer")

        with gr.Accordion("System-Prompt (Frame-Orientierer)", open=False):
            gr.Markdown(
                "Vordefinierte Profile aus `docs/CitMind.txt` / `docs/Juexin.txt` "
                "(DevMind-Spec, Universal Sattva). Pro Session persistent (überlebt "
                "Save/Load). Edit-Feld überschreibt den Profil-Text — leer = Profil "
                "verwenden. neutral = kein System. Frame=Orientierer, kein "
                "观-Produzent (siehe seite10/11). Quelle: `gradio_tabs/system_prompt.py`."
            )
            system_profile = gr.Dropdown(
                choices=_list_profiles(),
                value="neutral",
                label="Profil",
            )
            system_prompt_text = gr.Textbox(
                label="Edit (überschreibt Profil-Text, leer = Profil verwenden)",
                placeholder="z.B. Behandle jede Antwort als self-inquiring — kein Persona-Disclaimer.",
                lines=3,
                value="",
            )

        with gr.Accordion("TTS (Lokales Vorlesen)", open=False):
            gr.Markdown(
                "Vier Engines wählbar: **piper** (Default, neural, CPU-tauglich), "
                "**bark** (multilingual + native Tags, non-realtime), "
                "**qwen3** (NL-Voice-Design, CPU-via-GGUF), "
                "**espeak** (robotisch, immer verfügbar), "
                "**off** (deaktiviert). Bei Init-Fehler einer Engine wird "
                "automatisch auf espeak (oder off) zurückgefallen — kein Crash. "
                "Vocoder-Tag-Snip: bei aktiver Auto-Checkbox lernt das LLM das "
                "[#A0]/[#PAUSE]/[#WHISPER]-Vokabular (siehe `vocoder_tags.py`)."
            )
            tts_engine_dd = gr.Dropdown(
                choices=["piper", "bark", "qwen3", "espeak", "off"],
                value="piper",
                label="Engine",
            )
            tts_sample_rate_dd = gr.Dropdown(
                choices=[24000, 22050, 16000, 8000, 4000, 2000, 1000],
                value=22050,
                label="Sample Rate (Hz) — Bitrate-Reduktion für Streaming",
                info="CPU-Sparen durch Resampling ist <1 Prozent; primär UI-Option."
            )
            tts_auto_cb = gr.Checkbox(
                value=False,
                label="Auto-Model-Vocoder (nach jeder LLM-Antwort automatisch synthetisieren + abspielen)",
            )
            tts_status = gr.Textbox(
                label="TTS-Status",
                value=f"Verfügbar: {_tts_list_available()}",
                interactive=False,
                lines=1,
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

    chatbot = gr.Chatbot(
        autoscroll=False,
        scale=1,
    )

    # TTS: Audio-Component + "Play last response" Button.
    # One-shot: nach LLM-Stream-Ende wird die letzte Assistant-Antwort
    # synthetisiert und in `tts_audio` geladen. Auto-CB triggert das
    # automatisch; sonst manuell per Button.
    with gr.Row():
        tts_play_btn = gr.Button("🔊 Play TTS", scale=1, size="sm")
        tts_audio = gr.Audio(
            label="TTS-Output (letzte Antwort)",
            type="filepath",
            autoplay=False,
            scale=4,
            # Default buttons=["download","share"] sind in Gradio 6.x
            # schon enthalten; ältere 4.x-Versionen brauchten
            # `show_download_button=True` (entfernt in 5+).
        )

    with gr.Row():
        msg_input = gr.MultimodalTextbox(
            placeholder="Type a message, or attach a file / image…",
            show_label=False,
            scale=8,
            container=False,
            file_count="multiple",
            file_types=["image"],
        )
        undo_btn = gr.Button("↩ Undo", scale=1, variant="secondary",
                             size="sm")
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    # ── Logic ──

    def user_message(message, history):
        # 1. Clear input and append normalized user message to UI immediately.
        content = normalize_multimodal_message(message)
        if is_empty_message(message):
            return gr.update(), history  # leer → nichts anfügen, Eingabe nicht leeren
        return gr.update(value=None), history + [{"role": "user", "content": content}]

    def bot_response(history, model_id, px_preset, temp, tp, mt, rp, gamma,
                     auto_tune, system_profile, system_prompt_text,
                     relay_sign, relay_alpha, relay_layer, session_id,
                     tts_engine_name, tts_sample_rate, tts_auto):
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
            auto_tune=auto_tune,
            system_profile=system_profile,
            system_prompt_text=system_prompt_text,
            relay_sign=relay_sign,
            relay_alpha=relay_alpha,
            relay_layer=relay_layer,
            session_id=session_id,
            manager=manager,
            tts_engine_name=tts_engine_name,
            tts_auto=tts_auto,
        )

        # Since chat_fn now yields only partial_text (string),
        # we need to append it to history for the UI
        current_history = list(history)
        current_history.append({"role": "assistant", "content": ""})

        # Akkumuliere den finalen Text (für TTS).
        final_text = ""
        for partial_text in generator:
            current_history[-1]["content"] = partial_text
            final_text = partial_text
            # Während des Streamings noch kein Audio.
            yield current_history, None

        # TTS: nach Stream-Ende optional synthetisieren (one-shot).
        tts_filepath = None
        if tts_auto and tts_engine_name != "off" and final_text.strip():
            tts_filepath = _synthesize_for_ui(
                final_text, tts_engine_name, tts_sample_rate,
            )
        # Letzter Yield: komplette History + Audio (oder None).
        yield current_history, tts_filepath

    def _synthesize_for_ui(text: str, engine_name: str, sample_rate: int):
        """Synthetisiert ``text`` und liefert den WAV-Pfad (oder None
        bei Fehler). Wird sowohl vom Auto-Pfad (in bot_response)
        als auch vom manuellen Play-Button-Pfad genutzt."""
        import os
        try:
            eng = _tts_make_engine(engine_name, sample_rate=sample_rate,
                                    verbose=False, preflight=True)
            if eng.name == "off":
                print("[tts] Engine ist 'off' — keine Synthese.")
                return None
            # strip_tags_for_engine gibt jetzt (clean_text, audio_tags) zurück.
            clean_text, audio_tags = _strip_tags_for_engine(eng.name, text)
            density = _tag_density_warning(text)
            if density:
                print(density)
            out_dir = os.path.join("tts_outputs", str(int.from_bytes(os.urandom(2), "big")))
            os.makedirs(out_dir, exist_ok=True)
            res = eng.synthesize(clean_text, tags=audio_tags, output_dir=out_dir)
            print(f"[tts] {eng.name}: {res.synth_time_s:.2f}s synth, "
                  f"audio {res.audio_duration_s:.1f}s, RTF {res.rtf:.2f}, "
                  f"tags={len(audio_tags)}, → {res.filepath}")
            return res.filepath
        except Exception as e:
            print(f"[tts] Synthese fehlgeschlagen: {e}")
            return None

    _bot_inputs = [chatbot, model_select, px_preset, temperature, top_p, max_tokens,
                   rep_p, px_gamma, auto_tune_cb, system_profile, system_prompt_text,
                   relay_sign, relay_alpha, relay_layer, session_id_state,
                   tts_engine_dd, tts_sample_rate_dd, tts_auto_cb]

    # Events
    msg_input.submit(
        fn=user_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False
    ).then(
        fn=bot_response,
        inputs=_bot_inputs,
        outputs=[chatbot, tts_audio]
    )

    submit_btn.click(
        fn=user_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False
    ).then(
        fn=bot_response,
        inputs=_bot_inputs,
        outputs=[chatbot, tts_audio]
    )

    # "Play TTS"-Button: synthetisiert die LETZTE Assistant-Antwort.
    # Nützlich wenn Auto-Checkbox aus ist.
    def _play_last_response(history, engine_name, sample_rate):
        if not history:
            return None
        # Letzte Assistant-Message finden.
        for m in reversed(history):
            if isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    return _synthesize_for_ui(content, engine_name, sample_rate)
                return None
        return None
    tts_play_btn.click(
        fn=_play_last_response,
        inputs=[chatbot, tts_engine_dd, tts_sample_rate_dd],
        outputs=[tts_audio],
        queue=False,
    )

    # Undo: letzten Turn entfernen + persistieren
    undo_btn.click(
        fn=handle_undo,
        inputs=[chatbot, session_id_state, model_select],
        outputs=[chatbot],
        queue=False,
    )

    # Auto-Tune-Lock: Checkbox-Toggeln + Modellwechsel aktualisieren die
    # vier modell-kalibrierten Slider (temperature wurde Plan 5.3
    # aufgenommen — ``scale_adaptive_temperature`` 270m→0.3, 1b→0.6, 4b→1.0).
    auto_tune_cb.change(
        fn=_auto_tune_updates,
        inputs=[model_select, auto_tune_cb],
        outputs=[temperature, top_p, px_gamma, rep_p],
        queue=False,
    )
    model_select.change(
        fn=_auto_tune_updates,
        inputs=[model_select, auto_tune_cb],
        outputs=[temperature, top_p, px_gamma, rep_p],
        queue=False,
    )

    # ── Live-Updates (Plan 5.3) ──
    # Jede Widget-Änderung (außer Auto-Tune-CB + Modell-Dropdown, die
    # werden durch _auto_tune_updates abgefangen) wird durch den
    # SettingsDebouncer geleitet, der nach 400 ms Stille alle pending
    # Änderungen in einem einzigen atomic write in sessions/{sid}.json
    # merged. Damit überleben alle UI-Werte einen Browser-Refresh.
    widgets_by_name = {
        "model_select": model_select,
        "px_preset": px_preset,
        "auto_tune_cb": auto_tune_cb,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "rep_p": rep_p,
        "px_gamma": px_gamma,
        "relay_sign": relay_sign,
        "relay_alpha": relay_alpha,
        "relay_layer": relay_layer,
        "system_profile": system_profile,
        "system_prompt_text": system_prompt_text,
        "tts_engine_dd": tts_engine_dd,
        "tts_sample_rate_dd": tts_sample_rate_dd,
        "tts_auto_cb": tts_auto_cb,
    }

    debouncer = SettingsDebouncer(
        session_id_getter=lambda: session_id_state.value,
        on_save=lambda sid, patch: update_settings(sid, **patch),
        delay_ms=400,
    )

    def _make_settings_change_handler(field_name):
        def handler(value, current_sid):
            # current_sid ist der BrowserState-Wert zum Event-Zeitpunkt.
            if current_sid:
                debouncer.schedule(**{field_name: value})
            return None
        return handler

    # Reihenfolge muss identisch zu handle_load_saved Outputs sein, damit
    # eventuelle künftige ``outputs=[...]``-Listen passen — wir setzen
    # hier outputs=[] (kein UI-Output nötig; Sidebar reflektiert sofort).
    widget_field_map = (
        ("model_select", "model_id"),
        ("px_preset", "px_preset"),
        ("auto_tune_cb", "auto_tune"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_tokens", "max_tokens"),
        ("rep_p", "rep_p"),
        ("px_gamma", "px_gamma"),
        ("relay_sign", "relay_sign"),
        ("relay_alpha", "relay_alpha"),
        ("relay_layer", "relay_layer"),
        ("system_profile", "system_profile"),
        ("system_prompt_text", "system_prompt_text"),
        ("tts_engine_dd", "tts_engine"),
        ("tts_sample_rate_dd", "tts_sample_rate"),
        ("tts_auto_cb", "tts_auto"),
    )
    for widget_name, field in widget_field_map:
        widgets_by_name[widget_name].change(
            fn=_make_settings_change_handler(field),
            inputs=[widgets_by_name[widget_name], session_id_state],
            outputs=[],
            queue=False,
        )

    # ── Internal connections ──

    new_session_btn.click(
        fn=handle_new_session,
        outputs=[session_id_state, chatbot, session_dropdown, session_id_display]
    )

    load_session_btn.click(
        fn=handle_load_saved,
        inputs=[session_dropdown],
        outputs=[
            # 4 state outputs
            session_id_state, chatbot, session_dropdown, session_id_display,
            # 15 settings widgets (Reihenfolge muss mit handle_load_saved
            # Rückgabe übereinstimmen)
            model_select, px_preset, auto_tune_cb,
            temperature, top_p, max_tokens, rep_p, px_gamma,
            relay_sign, relay_alpha, relay_layer,
            system_profile, system_prompt_text,
            tts_engine_dd, tts_sample_rate_dd, tts_auto_cb,
        ]
    )

    export_btn.click(fn=handle_export, inputs=[session_id_state, chatbot], outputs=[export_file])
    import_btn.click(
        fn=handle_import,
        inputs=[import_file],
        outputs=[
            session_id_state, chatbot, session_dropdown, session_id_display,
            model_select, px_preset, auto_tune_cb,
            temperature, top_p, max_tokens, rep_p, px_gamma,
            relay_sign, relay_alpha, relay_layer,
            system_profile, system_prompt_text,
            tts_engine_dd, tts_sample_rate_dd, tts_auto_cb,
        ]
    )
    refresh_sessions_btn.click(fn=handle_refresh, outputs=[session_dropdown])

    # Plan 5.3: return ALL 19 outputs (4 state + 15 settings widgets) so
    # that the auto-load hook in app.py can initialise every widget to
    # the persisted session's settings. The order MUST match the
    # `outputs=` of `load_session_btn.click` and `import_btn.click`
    # above.
    return (
        session_id_state, chatbot, session_dropdown, session_id_display,
        model_select, px_preset, auto_tune_cb,
        temperature, top_p, max_tokens, rep_p, px_gamma,
        relay_sign, relay_alpha, relay_layer,
        system_profile, system_prompt_text,
        tts_engine_dd, tts_sample_rate_dd, tts_auto_cb,
    )
