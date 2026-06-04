"""
app.py — Single Entry Point: FastAPI API + Gradio UI
=====================================================
Mounts Gradio onto the FastAPI app at /gradio.
One uvicorn process serves both the OpenAI-compatible API (/v1/...)
and the full Gradio UI (/gradio).

Usage:
  python app.py
  # Or via run.sh
"""

import os
import sys

# ── Disable PX debug/telemetry output for clean serving ──
os.environ.setdefault("DEBUG_ROUTING", "0")
os.environ.setdefault("DEBUG_PX", "0")
os.environ.setdefault("SUBJECTIVE_TELEMETRY", "0")

import gradio as gr
from server import app as fastapi_app, manager
from config import MODEL_REGISTRY, SERVER_CONFIG
from benchmark_engine import BenchmarkEngine
from gradio_tabs.chat_tab import build_chat_tab
from gradio_tabs.cognitive_tests_tab import build_cognitive_tests_tab
from gradio_tabs.pzombie_eval_tab import build_pzombie_eval_tab
from gradio_tabs.telemetry_tab import build_telemetry_tab


# ── Create shared benchmark engine ──
engine = BenchmarkEngine(manager)


# ── Build Gradio Blocks ──
with gr.Blocks(
    title="PX Cognitive Architecture Explorer",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 1200px; margin: auto; }
    """
) as demo:
    gr.Markdown("""
    # 🧠 PX Cognitive Architecture Explorer
    **Phenomenological eXtension** — Model-agnostic PX patch system with cognitive evaluation.

    Models: **PX-patched** (gemma3-270m-px, minicpm5-1b-px) | **Baseline** (gemma3-270m-base, gemma3-270m-it, minicpm5-1b-base)

    API: `http://localhost:8000/v1/` (OpenAI-compatible) | UI: `/gradio`
    """)

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            build_chat_tab(manager)

        with gr.Tab("🧪 Cognitive Tests"):
            build_cognitive_tests_tab(manager, engine)

        with gr.Tab("🔬 P-Zombie Evaluation"):
            build_pzombie_eval_tab(manager, engine)

        with gr.Tab("📊 Telemetry"):
            build_telemetry_tab(manager)


# ── Mount Gradio onto FastAPI at /gradio ──
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")


if __name__ == "__main__":
    import uvicorn
    print(f"[PX Explorer] Starting on http://0.0.0.0:{SERVER_CONFIG['port']}")
    print(f"[PX Explorer] API:  http://localhost:{SERVER_CONFIG['port']}/v1/")
    print(f"[PX Explorer] UI:   http://localhost:{SERVER_CONFIG['port']}/gradio")
    uvicorn.run(
        app,
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        log_level="info",
    )