"""
chat_tab.py — Gradio Chat Tab with Full Parameter Control
=========================================================
Chat interface with model selector, parameter sliders, and PX-specific controls.
"""

import gradio as gr
import torch
import asyncio
from typing import Optional

from config import MODEL_REGISTRY
from model_manager import ModelManager
from generators import generate_chat_completion


def build_chat_tab(manager: ModelManager):
    """Build and return the Chat tab components."""

    model_choices = list(MODEL_REGISTRY.keys())

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
    def chat_respond(message, history, model_id, px_subj, temp, tp, mt, gamma, routing):
        """Send message and get response."""
        if not message.strip():
            return history, ""

        # Build messages list from history (Gradio Chatbot: list of [user, assistant] pairs)
        messages = []
        for pair in history:
            if pair[0]:
                messages.append({"role": "user", "content": pair[0]})
            if pair[1]:
                messages.append({"role": "assistant", "content": pair[1]})
        messages.append({"role": "user", "content": message})

        # Get model entry (async call from sync context)
        loop = asyncio.new_event_loop()
        try:
            model_entry = loop.run_until_complete(
                manager.get_model(
                    model_id,
                    px_subjective=px_subj,
                    px_gamma=gamma if gamma != 0.08 else None,  # Skip if default
                    px_routing_mode=routing if routing != "adaptive" else None,
                )
            )
        finally:
            loop.close()

        # Generate response
        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]

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

        # Update history (Gradio format: list of [user, assistant] pairs)
        history = history + [[message, text]]

        # Get PX metrics
        px_metrics = manager.get_px_metrics(model_id)

        return history, "", px_metrics

    # ── Wire up events ──
    send_btn.click(
        fn=chat_respond,
        inputs=[msg_input, chatbot, model_select, px_subjective,
                temperature, top_p, max_tokens, px_gamma, px_routing_mode],
        outputs=[chatbot, msg_input, px_metrics_display],
    )

    msg_input.submit(
        fn=chat_respond,
        inputs=[msg_input, chatbot, model_select, px_subjective,
                temperature, top_p, max_tokens, px_gamma, px_routing_mode],
        outputs=[chatbot, msg_input, px_metrics_display],
    )

    clear_btn.click(
        fn=lambda: ([], "", None),
        outputs=[chatbot, msg_input, px_metrics_display],
    )

    return model_select