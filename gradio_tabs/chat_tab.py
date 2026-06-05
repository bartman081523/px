"""
chat_tab.py — Gradio Chat Tab with ChatInterface & DMT Protocol
================================================================
Integrated chat interface with session management and DMT protocol steering.
"""

import gradio as gr
import torch
import asyncio
import os
import json
from typing import Optional, List, Dict, Any
from threading import Thread
from transformers import TextIteratorStreamer

from config import MODEL_REGISTRY
from model_manager import ModelManager
from sessions import save_session, load_session, get_new_session_id, list_sessions


# ── Session Handlers ──

def on_load(session_id):
    """Called when the page loads."""
    if session_id is None or session_id == "":
        session_id = get_new_session_id()
    
    data = load_session(session_id)
    history = data.get("history", [])
    # Return 4 values to match app.py init_app outputs
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
        return gr.skip(), []
    try:
        with open(file_obj.name, "r") as f:
            data = json.load(f)
        new_id = data.get("session_id", get_new_session_id())
        history = data.get("history", [])
        save_session(new_id, history)
        return new_id, history
    except Exception as e:
        print(f"Import error: {e}")
        return gr.skip(), []


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
        px_subjective = gr.Checkbox(
            label="PX Subjective Mode",
            value=False,
        )
        dmt_protocol = gr.Checkbox(
            label="DMT Protocol (High-Fi)",
            value=False,
            info="Enables Memory, ERPU, Agency"
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
        session_dropdown = gr.Dropdown(choices=list_sessions(), label="Saved Sessions")
        load_session_btn = gr.Button("Load Selected", size="sm")
        session_id_display = gr.Textbox(label="Current ID", interactive=False)
        
        gr.Markdown("---")
        export_btn = gr.Button("Download JSON", size="sm")
        export_file = gr.File(label="Export", visible=False)
        import_file = gr.File(label="Import JSON", file_types=[".json"])
        import_btn = gr.Button("Import & Load", size="sm")

    # ── Chat Interface ──
    
    def chat_fn(message, history, model_id, px_subj, dmt_on, temp, tp, mt, rp, gamma, session_id):
        # 1. Update config based on DMT Protocol toggle
        loop = asyncio.new_event_loop()
        try:
            # Subjective must be on for DMT
            is_subjective = px_subj or dmt_on
            preset = "DMT" if dmt_on else "SUBJECTIVE"
            
            model_entry = loop.run_until_complete(
                manager.get_model(
                    model_id,
                    px_subjective=is_subjective,
                    px_gamma=gamma,
                    px_config_preset=preset
                )
            )
            # Re-patch if preset changed
            if model_entry.get("px_config_preset") != preset:
                 manager._reapply_patch(model_id, px_subjective=is_subjective, px_gamma=gamma, px_config_preset=preset)
                 model_entry["px_config_preset"] = preset
        finally:
            loop.close()

        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]

        # 2. Build history
        messages = history + [{"role": "user", "content": message}]

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

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            yield partial_text

        # 4. Save session on completion
        full_history = messages + [{"role": "assistant", "content": partial_text}]
        save_session(session_id, full_history, model_id=model_id)

    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[model_select, px_subjective, dmt_protocol, temperature, top_p, max_tokens, rep_p, px_gamma, session_id_state],
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
    import_btn.click(fn=handle_import, inputs=[import_file], outputs=[session_id_state, chat_interface.chatbot])

    return session_id_state, chat_interface.chatbot, session_dropdown, session_id_display
