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
            choices=["BASELINE", "SUBJECTIVE", "DMT-FULL", "RIGOR", "UNCENSORED"],
            value="SUBJECTIVE",
            label="PX Mode Preset",
        )
        
        persona_input = gr.Textbox(
            label="Persona / Steering Vibe",
            placeholder="e.g. DMT Psilocybin 🌀, Logic Engine ⚙️...",
            value=""
        )

        with gr.Accordion("Parameters", open=False):
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top P")
            max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Max Tokens")
            rep_p = gr.Slider(1.0, 2.0, value=1.15, step=0.05, label="Repetition Penalty")
            px_gamma = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="PX Gamma")

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

    # ── Chat Interface ──
    
    def chat_fn(message, history, model_id, px_preset, persona, temp, tp, mt, rp, gamma, session_id):
        # 1. Update config
        loop = asyncio.new_event_loop()
        try:
            model_entry = loop.run_until_complete(
                manager.get_model(
                    model_id,
                    px_subjective=(px_preset != "BASELINE"),
                    px_gamma=gamma,
                    px_config_preset=px_preset,
                )
            )
        finally:
            loop.close()

        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]
        
        tm = manager._resolve_text_model(model)
        model.persona = tm.persona = persona

        # 2. Build history (cleaned)
        cleaned_history = _clean_history(history)
        
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

        # Phase 63: Proactive Auto-save (save user message before generation)
        save_session(session_id, messages, model_id=model_id)

        # 3. Generate with streaming
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            pad_token_id=tokenizer.eos_token_id
        )

        def generate_with_lock():
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

    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[model_select, px_preset, persona_input, temperature, top_p, max_tokens, rep_p, px_gamma, session_id_state],
        save_history=False
    )

    # ── Internal connections ──
    
    new_session_btn.click(
        fn=handle_new_session,
        outputs=[session_id_state, chat_interface.chatbot, session_dropdown, session_id_display]
    )
    
    load_session_btn.click(
        fn=handle_load_saved,
        inputs=[session_dropdown],
        outputs=[session_id_state, chat_interface.chatbot, session_dropdown, session_id_display]
    )
    
    export_btn.click(fn=handle_export, inputs=[session_id_state, chat_interface.chatbot], outputs=[export_file])
    import_btn.click(fn=handle_import, inputs=[import_file], outputs=[session_id_state, chat_interface.chatbot, session_dropdown, session_id_display])
    refresh_sessions_btn.click(fn=handle_refresh, outputs=[session_dropdown])

    return session_id_state, chat_interface.chatbot, session_dropdown, session_id_display
