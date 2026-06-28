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

# SR-61b: Mitigate OOM on RTX 2060 (12GB)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"

# ── Disable PX debug/telemetry output for clean serving ──
os.environ.setdefault("DEBUG_ROUTING", "0")
os.environ.setdefault("DEBUG_PX", "0")
os.environ.setdefault("SUBJECTIVE_TELEMETRY", "0")

# Plan 7.1: Hard-Crash auf unbehandelte Exceptions — sys/threading/faulthandler-
# Hooks installieren. ENV PX_HARD_CRASH=0 macht install() zum no-op.
import crash_handler
crash_handler.install()

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
            session_id_state, chatbot, session_dropdown, session_id_display = build_chat_tab(manager)

        with gr.Tab("🧪 Cognitive Tests"):
            build_cognitive_tests_tab(manager, engine)

        with gr.Tab("🔬 P-Zombie Evaluation"):
            build_pzombie_eval_tab(manager, engine)

        with gr.Tab("📊 Telemetry"):
            build_telemetry_tab(manager)

    # ── Initialization ──
    def init_app(session_id):
        from gradio_tabs.chat_tab import on_load
        return on_load(session_id)

    demo.load(
        fn=init_app,
        inputs=[session_id_state],
        outputs=[session_id_state, chatbot, session_dropdown, session_id_display]
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

    # Plan 7.1: uvicorn.Server statt uvicorn.run() damit wir nach
    # Loop-Creation den asyncio-Hook attachen können (uvicorn.run()
    # versteckt den Loop hinter einer Fire-and-forget-Wrapper-Funktion).
    config = uvicorn.Config(**run_kwargs)
    server = uvicorn.Server(config)

    _orig_startup = server.startup
    async def _patched_startup(sockets=None):
        await _orig_startup(sockets=sockets)
        # uvicorn ≥0.30 speichert den Loop erst während serve() als
        # ``server.main_loop``. Hier läuft unser Patched-Startup im
        # selben Loop, also können wir den aktuellen direkt greifen.
        import asyncio as _asyncio
        crash_handler.install_asyncio(_asyncio.get_running_loop())
    server.startup = _patched_startup

    server.run()