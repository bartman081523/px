"""
chat_tab.py — Gradio Chat Tab with Full Parameter Control & Session Management
=============================================================================
Chat interface with model selector, parameter sliders, and PX-specific controls.
Includes session management (BrowserState, JSON Import/Export).
"""

import gradio as gr
import torch
import asyncio
import os
import json
from typing import Optional, List, Dict, Any

from config import MODEL_REGISTRY
from model_manager import ModelManager
from generators import generate_chat_completion
from sessions import save_session, load_session, get_new_session_id, list_sessions


# ── Session Handlers ──

def on_load(session_id):
    """Called when the page loads."""
    if session_id is None:
        session_id = get_new_session_id()
    
    data = load_session(session_id)
    history = data.get("history", [])
    return session_id, history, gr.update(choices=list_sessions())

def handle_new_session():
    new_id = get_new_session_id()
    return new_id, [], gr.update(choices=list_sessions())

def handle_load_saved(session_id):
    if not session_id:
        return gr.update(), [], gr.update(choices=list_sessions())
    data = load_session(session_id)
    return session_id, data.get("history", []), gr.update(choices=list_sessions())

def handle_export(session_id, history):
    if not history:
        return gr.update(visible=False)
    path = f"exported_session_{session_id}.json"
    with open(path, "w") as f:
        json.dump({"session_id": session_id, "history": history}, f, indent=2)
    return gr.update(value=path, visible=True)

def handle_import(file_obj):
    if file_obj is None:
        return gr.update(), []
    try:
        with open(file_obj.name, "r") as f:
            data = json.load(f)
        new_id = data.get("session_id", get_new_session_id())
        history = data.get("history", [])
        save_session(new_id, history)
        return new_id, history
    except Exception as e:
        print(f"Import error: {e}")
        return gr.update(), []


def build_chat_tab(manager: ModelManager):
    """Build and return the Chat tab components."""

    # ── Client-side state ──
    # BrowserState persists in the user's browser (localStorage)
    session_id_state = gr.BrowserState(default_value=None, storage_key="px_session_id")

    model_choices = list(MODEL_REGISTRY.keys())

    with gr.Sidebar(label="Sessions"):
        gr.Markdown("### Session Management")
        new_session_btn = gr.Button("New Session", variant="secondary")
        session_list_refresh = gr.Button("Refresh List", size="sm")
        session_dropdown = gr.Dropdown(choices=list_sessions(), label="Saved Sessions")
        load_session_btn = gr.Button("Load Selected", size="sm")
        
        gr.Markdown("---")
        gr.Markdown("### Import/Export")
        export_btn = gr.Button("Download Session (JSON)")
        export_file = gr.File(label="Exported JSON", visible=False)
        
        import_file = gr.File(label="Import Session JSON", file_types=[".json"])
        import_btn = gr.Button("Import & Load")

    with gr.Row():
        model_select = gr.Dropdown(
            choices=model_choices,
            value=model_choices[0],
            label="Model",
            scale=3,
        )
        px_subjective = gr.Checkbox(
            label="PX Subjective Mode",
            value=False,
            visible=True,
            scale=1,
        )

    with gr.Row():
        # Standard parameters
        with gr.Accordion("Model Parameters", open=True):
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
            max_tokens = gr.Slider(1, 4096, value=512, step=1, label="Max Tokens")

        # PX-specific parameters
        with gr.Accordion("PX Parameters", open=False) as px_accordion:
            px_gamma = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="Gamma (LTI/ADC strength)")
            px_routing_mode = gr.Dropdown(
                choices=["adaptive", "fixed"],
                value="adaptive",
                label="Routing Mode",
            )

    # Chat interface
    chatbot = gr.Chatbot(height=500)
    msg_input = gr.Textbox(
        placeholder="Type your message... (Enter to send, Shift+Enter for newline)",
        show_label=False,
        lines=2,
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary", scale=2)
        clear_btn = gr.Button("Clear Chat", scale=1)

    # PX metrics display
    with gr.Accordion("PX Cognitive Metrics (last response)", open=False):
        px_metrics_display = gr.JSON(label="PX Metrics")

    # ── Model change: update PX visibility ──
    def on_model_change(model_id):
        is_px = MODEL_REGISTRY.get(model_id, {}).get("patch_dir") is not None
        return (
            gr.update(visible=is_px),   # px_subjective
            gr.update(visible=is_px),   # px_accordion (containing gamma/routing)
        )

    model_select.change(
        fn=on_model_change,
        inputs=[model_select],
        outputs=[px_subjective, px_accordion],
    )

    # ── Chat function ──
    def chat_respond(message, history, model_id, px_subj, temp, tp, mt, gamma, routing, session_id):
        """Send message and get response."""
        if not message.strip():
            return history, "", None, session_id

        # Build messages list from history (Gradio messages format: list of dictionaries)
        messages = history + [{"role": "user", "content": message}]

        # Get model entry (async call from sync context)
        loop = asyncio.new_event_loop()
        try:
            model_entry = loop.run_until_complete(
                manager.get_model(
                    model_id,
                    px_subjective=px_subj,
                    px_gamma=gamma if gamma != 0.08 else None,
                    px_routing_mode=routing if routing != "adaptive" else None,
                )
            )
        finally:
            loop.close()

        # Generate response
        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]

        # Note: apply_chat_template expects a list of dictionaries
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(mt),
                temperature=temp if temp > 0 else 1e-10,
                top_p=tp,
                do_sample=temp > 0,
            )

        new_tokens = outputs[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Update history
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": text},
        ]

        # Save to disk
        save_session(session_id, history, model_id=model_id)

        # Get PX metrics
        px_metrics = manager.get_px_metrics(model_id)

        return history, "", px_metrics, session_id

    # ── Wire up events ──
    
    send_btn.click(
        fn=chat_respond,
        inputs=[msg_input, chatbot, model_select, px_subjective,
                temperature, top_p, max_tokens, px_gamma, px_routing_mode, session_id_state],
        outputs=[chatbot, msg_input, px_metrics_display, session_id_state],
    )

    msg_input.submit(
        fn=chat_respond,
        inputs=[msg_input, chatbot, model_select, px_subjective,
                temperature, top_p, max_tokens, px_gamma, px_routing_mode, session_id_state],
        outputs=[chatbot, msg_input, px_metrics_display, session_id_state],
    )

    clear_btn.click(
        fn=lambda sid: ([], "", None, sid),
        inputs=[session_id_state],
        outputs=[chatbot, msg_input, px_metrics_display, session_id_state],
    )

    new_session_btn.click(
        fn=handle_new_session,
        outputs=[session_id_state, chatbot, session_dropdown]
    )
    
    load_session_btn.click(
        fn=handle_load_saved,
        inputs=[session_dropdown],
        outputs=[session_id_state, chatbot, session_dropdown]
    )
    
    export_btn.click(
        fn=handle_export,
        inputs=[session_id_state, chatbot],
        outputs=[export_file]
    )
    
    import_btn.click(
        fn=handle_import,
        inputs=[import_file],
        outputs=[session_id_state, chatbot]
    )

    session_list_refresh.click(
        fn=lambda: gr.update(choices=list_sessions()),
        outputs=[session_dropdown]
    )

    return session_id_state, chatbot, session_dropdown
