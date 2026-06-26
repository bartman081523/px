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
            # Plan 5.3: build_chat_tab returns 19 outputs (4 state +
            # 15 settings widgets) so that the demo.load auto-resume
            # hook can initialise every widget to the persisted session's
            # settings without an extra round-trip through the UI.
            (
                session_id_state, chatbot, session_dropdown,
                session_id_display,
                model_select, px_preset, auto_tune_cb,
                temperature, top_p, max_tokens, rep_p, px_gamma,
                relay_sign, relay_alpha, relay_layer,
                system_profile, system_prompt_text,
                tts_engine_dd, tts_sample_rate_dd, tts_auto_cb,
            ) = build_chat_tab(manager)

        with gr.Tab("🧪 Cognitive Tests"):
            build_cognitive_tests_tab(manager, engine)

        with gr.Tab("🔬 P-Zombie Evaluation"):
            build_pzombie_eval_tab(manager, engine)

        with gr.Tab("📊 Telemetry"):
            build_telemetry_tab(manager)

    # ── Initialization (Plan 5.3: Auto-Resume) ──
    def init_app(session_id):
        """Auto-Resume der zuletzt-modifizierten Session, wenn
        ``session_id`` (BrowserState) leer ist (Cookies gelöscht, neuer
        Browser). Liefert alle 19 Outputs."""
        from gradio_tabs.chat_tab import handle_load_saved
        if session_id is None or session_id == "":
            from sessions import list_session_mtimes
            candidates = list_session_mtimes()
            if candidates:
                session_id = candidates[0][1]  # mtime-newest
            else:
                # Kein Session-File vorhanden → komplette Skip-Reihe,
                # damit die UI leer bleibt.
                return (
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    *(gr.skip() for _ in range(15)),
                )
        return handle_load_saved(session_id)

    demo.load(
        fn=init_app,
        inputs=[session_id_state],
        outputs=[
            session_id_state, chatbot, session_dropdown, session_id_display,
            model_select, px_preset, auto_tune_cb,
            temperature, top_p, max_tokens, rep_p, px_gamma,
            relay_sign, relay_alpha, relay_layer,
            system_profile, system_prompt_text,
            tts_engine_dd, tts_sample_rate_dd, tts_auto_cb,
        ]
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