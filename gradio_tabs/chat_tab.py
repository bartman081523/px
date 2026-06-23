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
from sessions import save_session, load_session, get_new_session_id, list_sessions
from telemetry import telemetry


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

def on_load(session_id):
    """Called when the page loads."""
    if session_id is None or session_id == "":
        session_id = get_new_session_id()
    
    data = load_session(session_id)
    history = data.get("history", [])
    return session_id, history, gr.update(choices=list_sessions()), session_id

def handle_new_session():
    new_id = get_new_session_id()
    return new_id, [], gr.update(choices=list_sessions()), new_id

def handle_load_saved(session_id):
    if not session_id:
        return gr.skip(), [], gr.skip(), gr.skip()
    data = load_session(session_id)
    return session_id, data.get("history", []), gr.skip(), session_id

def handle_export(session_id, history):
    if not history:
        return gr.update(visible=False)
    path = f"exported_session_{session_id}.json"
    with open(path, "w") as f:
        json.dump({"session_id": session_id, "history": history}, f, indent=2)
    return gr.update(value=path, visible=True)

def handle_import(file_obj):
    if file_obj is None:
        return gr.skip(), [], gr.skip(), gr.skip()
    try:
        with open(file_obj.name, "r") as f:
            data = json.load(f)
        new_id = data.get("session_id", get_new_session_id())
        history = data.get("history", [])
        save_session(new_id, history)
        return new_id, history, gr.update(choices=list_sessions(), value=new_id), new_id
    except Exception as e:
        print(f"Import error: {e}")
        return gr.skip(), [], gr.skip(), gr.skip()

def handle_refresh():
    return gr.update(choices=list_sessions())


def chat_fn(message, history, model_id, px_preset, temp, tp, mt, rp, gamma,
            relay_sign, relay_alpha, relay_layer, session_id, manager: ModelManager):
    """Core chat logic with history management and model generation."""
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
    
    actual_message = message
    if isinstance(message, dict):
        text = message.get("text", "")
        files = message.get("files", [])
        if files:
            actual_message = [{"type": "text", "text": text}]
            for f in files:
                fpath = f["path"] if isinstance(f, dict) and "path" in f else f
                actual_message.append({"type": "image", "image": fpath})
        else:
            actual_message = text

    messages = cleaned_history + [{"role": "user", "content": actual_message}]
    print(f"DEBUG: Combined messages length: {len(messages)}")

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

    input_text = tokenizer.apply_chat_template(processed_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(mt),
        temperature=temp if temp > 0 else 1e-10,
        top_p=tp,
        repetition_penalty=rp,
        do_sample=temp > 0,
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


def build_chat_tab(manager: ModelManager):
    """Build and return the Chat tab components."""

    # ── Client-side state ──
    session_id_state = gr.BrowserState(default_value=None, storage_key="px_session_id")


    model_choices = list(MODEL_REGISTRY.keys())

    with gr.Sidebar(label="PX Controls"):
        gr.Markdown("### Model Selection")
        model_select = gr.Dropdown(
            choices=model_choices,
            value=model_choices[0],
            label="Current Model",
        )
        px_preset = gr.Dropdown(
            choices=["BASELINE", "ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN", "ACTIVE_MANIFOLD_RELAY"],
            value="ACTIVE_MANIFOLD",
            label="PX Mode Preset",
        )

        with gr.Accordion("Parameters", open=False):
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top P")
            max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Max Tokens")
            rep_p = gr.Slider(1.0, 2.0, value=1.15, step=0.05, label="Repetition Penalty")
            px_gamma = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="PX Gamma")

        with gr.Accordion("verstärkbar Relay (seite15)", open=False):
            gr.Markdown(
                "Re-Injektion der modell-eigenen L16-Zustands-Richtung `d_width` am "
                "post-recur Layer (Motor unangetastet, forward_hook). Wirksam mit "
                "**ACTIVE_MANIFOLD_RELAY** (default sign=+1) oder sign≠0 auf jedem "
                "Preset. Nur gemma3-1b-it hat ein d_width-Artefakt; andere Modelle "
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
    
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Type a message...",
            show_label=False,
            scale=9,
            container=False
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    # ── Logic ──

    def user_message(message, history):
        # 1. Clear input and append user message to UI immediately
        return "", history + [{"role": "user", "content": message}]

    def bot_response(history, model_id, px_preset, temp, tp, mt, rp, gamma,
                     relay_sign, relay_alpha, relay_layer, session_id):
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
            relay_sign=relay_sign,
            relay_alpha=relay_alpha,
            relay_layer=relay_layer,
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
        inputs=[chatbot, model_select, px_preset, temperature, top_p, max_tokens, rep_p, px_gamma, relay_sign, relay_alpha, relay_layer, session_id_state],
        outputs=[chatbot]
    )

    submit_btn.click(
        fn=user_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False
    ).then(
        fn=bot_response,
        inputs=[chatbot, model_select, px_preset, temperature, top_p, max_tokens, rep_p, px_gamma, relay_sign, relay_alpha, relay_layer, session_id_state],
        outputs=[chatbot]
    )

    # ── Internal connections ──
    
    new_session_btn.click(
        fn=handle_new_session,
        outputs=[session_id_state, chatbot, session_dropdown, session_id_display]
    )
    
    load_session_btn.click(
        fn=handle_load_saved,
        inputs=[session_dropdown],
        outputs=[session_id_state, chatbot, session_dropdown, session_id_display]
    )
    
    export_btn.click(fn=handle_export, inputs=[session_id_state, chatbot], outputs=[export_file])
    import_btn.click(fn=handle_import, inputs=[import_file], outputs=[session_id_state, chatbot, session_dropdown, session_id_display])
    refresh_sessions_btn.click(fn=handle_refresh, outputs=[session_dropdown])

    return session_id_state, chatbot, session_dropdown, session_id_display
