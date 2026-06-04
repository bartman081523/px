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
with gr.Blocks(title="PX Cognitive Architecture Explorer") as demo:
    # Resolve protocol for UI display
    protocol = "https" if SERVER_CONFIG.get("ssl_cert") and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), SERVER_CONFIG.get("ssl_cert"))) else "http"
    
    gr.Markdown(f"""
    # 🧠 PX Cognitive Architecture Explorer
    **Phenomenological eXtension** — Model-agnostic PX patch system with cognitive evaluation.

    Models: **PX-patched** (gemma3-270m-px, minicpm5-1b-px) | **Baseline** (gemma3-270m-base, gemma3-270m-it, minicpm5-1b-base)

    API: `{protocol}://localhost:{SERVER_CONFIG['port']}/v1/` (OpenAI-compatible) | UI: `/gradio`
    """)

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            session_id_state, chatbot, session_dropdown = build_chat_tab(manager)

        with gr.Tab("🧪 Cognitive Tests"):
            build_cognitive_tests_tab(manager, engine)

        with gr.Tab("🔬 P-Zombie Evaluation"):
            build_pzombie_eval_tab(manager, engine)

        with gr.Tab("📊 Telemetry"):
            build_telemetry_tab(manager)

    # ── Initialization ──
    def init_app(session_id):
        from sessions import load_session, get_new_session_id, list_sessions
        if session_id is None:
            session_id = get_new_session_id()
        data = load_session(session_id)
        history = data.get("history", [])
        return session_id, history, gr.update(choices=list_sessions())

    demo.load(
        fn=init_app,
        inputs=[session_id_state],
        outputs=[session_id_state, chatbot, session_dropdown]
    )

    gr.Markdown("""
    ---
    **PX Explorer v1.1** | [OpenAI API Documentation](/v1/docs) | [System Status](/)
    *Built with SciMind4 Rigor Protocol. Anti-Zombie Grade: B+*
    """)


# ── Mount Gradio onto FastAPI at /gradio ──
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")


if __name__ == "__main__":
    import uvicorn
    from config import SERVER_CONFIG
    
    # SSL Configuration
    ssl_cert = SERVER_CONFIG.get("ssl_cert")
    ssl_key = SERVER_CONFIG.get("ssl_key")
    
    # Resolve relative paths
    if ssl_cert and not os.path.isabs(ssl_cert):
        ssl_cert = os.path.join(os.path.dirname(os.path.abspath(__file__)), ssl_cert)
    if ssl_key and not os.path.isabs(ssl_key):
        ssl_key = os.path.join(os.path.dirname(os.path.abspath(__file__)), ssl_key)
        
    use_ssl = ssl_cert and os.path.exists(ssl_cert) and ssl_key and os.path.exists(ssl_key)
    protocol = "https" if use_ssl else "http"
    
    print(f"[PX Explorer] Starting on {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']} (SSL: {use_ssl})")
    print(f"[PX Explorer] API:  {protocol}://localhost:{SERVER_CONFIG['port']}/v1/")
    print(f"[PX Explorer] UI:   {protocol}://localhost:{SERVER_CONFIG['port']}/gradio")
    
    run_kwargs = {
        "app": app,
        "host": SERVER_CONFIG["host"],
        "port": SERVER_CONFIG["port"],
        "log_level": "info",
    }
    
    if use_ssl:
        run_kwargs["ssl_certfile"] = ssl_cert
        run_kwargs["ssl_keyfile"] = ssl_key
        
    uvicorn.run(**run_kwargs)