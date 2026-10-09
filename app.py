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

# Plan t4-wheel (Deploy-#15): stdout line-buffered — Warmup/Chat-Prints
# erscheinen in HF /logs/run SOFORT statt erst beim Puffer-Flush; zusammen
# mit crash_handler (stderr + faulthandler) ist jede Todesursache dann
# sichtbar. Fehlschlag-tolerant (TextIO ohne reconfigure → no-op).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

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

# Plan hf-space-v4-publish (ZeroGPU): spaces MUSS vor jedem torch-Import
# stehen (spaces patcht torch.cuda beim ersten GPU-Call; gradio/blocks.py:99
# tut denselben Import zur Laufzeit — explizit ist die Reihenfolge garantiert).
# Wheel-Inspection spaces 0.50.4: Config.zero_gpu = SPACES_ZERO_GPU env; der
# GPU-Decorator ist außerhalb ZeroGPU ein no-op (_GPU: `if not Config.zero_gpu:
# return task`). Lokal/ reguläre GPU-Hardware: env unset → guarded no-op.
if os.environ.get("SPACES_ZERO_GPU"):
    import spaces  # noqa: F401

import gradio as gr
from server import app as fastapi_app, manager
from config import MODEL_REGISTRY, SERVER_CONFIG
from benchmark_engine import BenchmarkEngine
from gradio_tabs.chat_tab import build_chat_tab
from gradio_tabs.cognitive_tests_tab import build_cognitive_tests_tab
from gradio_tabs.pzombie_eval_tab import build_pzombie_eval_tab
from gradio_tabs.telemetry_tab import build_telemetry_tab
from gradio_tabs._styles import get_css


# ── Create shared benchmark engine ──
engine = BenchmarkEngine(manager)


# ── Build Gradio Blocks ──
# Plan ui-styling Task #181: CSS-Variablen aus _styles.py werden in
# mount_gradio_app(css=...) injiziert (Gradio 6 API: css ist launch-time,
# nicht Blocks-constructor — siehe UserWarning sonst).
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
            # Plan 2026-10-05 / Phase 3 2026-10-05 (Thinking-Widgets + Budget):
            # build_chat_tab returnt 19 Komponenten — die ersten 4 wie
            # bisher, danach die 15 Settings-Widgets in EXAKT der
            # SETTINGS_WIDGET_FIELDS-Reihenfolge (gradio_tabs/chat_tab.py-
            # Header). Diese Reihenfolge spiegeln die demo.load-Outputs unten.
            (session_id_state, chatbot, session_dropdown, session_id_display,
             model_select, px_preset_widgets, temperature, top_p, max_tokens,
             rep_p, px_gamma, thinking, thinking_budget, thinking_effort,
             relay_sign, relay_alpha, relay_layer,
             system_profile, system_prompt_text) = build_chat_tab(manager)

        with gr.Tab("🧪 Cognitive Tests"):
            build_cognitive_tests_tab(manager, engine)

        with gr.Tab("🔬 P-Zombie Evaluation"):
            build_pzombie_eval_tab(manager, engine)

        with gr.Tab("📊 Telemetry"):
            build_telemetry_tab(manager)

    # ── Initialization ──
    # Plan 2026-10-05: Page-Load rendert zusätzlich die 15 Settings-Widgets
    # aus der session.json (Session-Settings-Restore, +thinking/budget/effort)
    # — 19 Outputs, Reihen-folge identisch mit build_chat_tab-Return oben.
    # inputs um system_profile erweitert: restore_session_settings braucht den
    # aktuell gerenderten Profil-Wert für den Suppress (siehe chat_tab.py).
    def init_app(session_id, current_profile):
        from gradio_tabs.chat_tab import on_load
        return on_load(session_id, current_profile)

    demo.load(
        fn=init_app,
        inputs=[session_id_state, system_profile],
        outputs=[session_id_state, chatbot, session_dropdown, session_id_display,
                 model_select, px_preset_widgets, temperature, top_p,
                 max_tokens, rep_p, px_gamma, thinking, thinking_budget,
                 thinking_effort, relay_sign, relay_alpha, relay_layer,
                 system_profile, system_prompt_text]
    )

    gr.Markdown("""
    ---
    **PX Explorer v1.1** | [OpenAI API Documentation](/v1/docs) | [System Status](/)
    *Built with SciMind4 Rigor Protocol. Anti-Zombie Grade: B+*
    """)


# HF-Space-Mode: KEIN mount_gradio_app! HF's gradio-SDK ruft die
# `app`-Variable ggf. via uvicorn auf, was Port 7860 vorab belegt
# → demo.launch(7860) crasht mit "Cannot find empty port".
# Plan: in HF-Mode nur `demo` exportieren, Mount nur in local-mode.
if os.environ.get("SPACES_RUN_MODE") or os.environ.get("SPACE_ID"):
    app = None  # HF-SDK: nichts zu mounten
else:
    # ── Mount Gradio onto FastAPI at /gradio ──
    # Plan ui-styling: CSS aus _styles.py an mount_gradio_app weiterreichen.
    app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio", css=get_css())


if __name__ == "__main__":
    import uvicorn
    from config import SERVER_CONFIG

    # HF-Space-Detection: HF startet das gradio-App via gr.launch().
    # Auf HF-Space haben wir:
    #   - /v1/ API:  NICHT verfügbar (kein uvicorn-Server)
    #   - /gradio UI: verfügbar via demo.launch()
    # Das ist der Trade-off — HF's gradio-SDK kann nur ein einzelnes
    # gr.Blocks mounten, nicht ein FastAPI+Gradio-Composite.
    # Plan: portiere /v1/ API auf Gradio-API-Endpoints in einem
    # separaten Schritt (siehe TODO app.py:port-v1-to-gradio).
    if os.environ.get("SPACES_RUN_MODE") or os.environ.get("SPACE_ID"):
        print(f"[PX Explorer] HF-Space detected — launching gr.Blocks via demo.launch() (no /v1/ API on HF)")

        def _prefetch_hub_snapshots():
            """Plan hf-space-v4-publish: ZeroGPU-Snapshot-Prefetch im Startup
            (ohne GPU-Lease). Der 6-GB-bonsai-Download zählt andernfalls in
            die Lease des ersten Chat-Calls hinein (Request-Cap gemessen
            <270 s); hier ist er leasen-frei, und der Container-Cache überlebt
            Restarts. Targets per Env PX_PREFETCH_MODELS (Kommaliste,
            Registry-Keys)."""
            targets = [t.strip() for t in
                       os.environ.get("PX_PREFETCH_MODELS",
                                      "ternary-bonsai-27b").split(",")
                       if t.strip()]
            if not targets:
                return

            import threading
            from huggingface_hub import snapshot_download

            def _work():
                for mid in targets:
                    cfg = MODEL_REGISTRY.get(mid) or {}
                    hf_id = cfg.get("hf_id") or mid
                    try:
                        path = snapshot_download(hf_id,
                                                 token=os.environ.get("HF_TOKEN"))
                        print(f"[prefetch] {mid}: {path}")
                    except Exception as exc:  # Startup-Prefetch darf nicht töten
                        print(f"[prefetch] {mid} ({hf_id}) FEHLER: {exc}")

            threading.Thread(target=_work, daemon=True,
                             name="hf-snapshot-prefetch").start()

        _prefetch_hub_snapshots()

        # Plan t4-wheel (Deploy-#18): Request-Status-Counter — die 422-Sturm-
        # Rate war im Journal unsichtbar (Python-warnings drucken die
        # StarletteDeprecation nur EINMAL pro Prozess; der Rest verschwindet
        # im Sidecar-Dedup identischer Zeilen). Der Counter zählt ALLE
        # Requests nach (status, path) und wird vom MEM-Heartbeat mit
        # ausgegeben. Gradio-6: routes.App ist Starlette-Subklasse — der
        # Middleware-Stack wird via build_middleware_stack gebaut → sauber
        # monkeypatched, fail-soft (keine App bei Fehler).
        _HTTP_COUNTER = {"n": 0, "st": {}, "paths": {}, "loop": None}
        try:
            import gradio.routes as _groutes

            class _CountingASGI:
                def __init__(self, inner):
                    self.inner = inner

                async def __call__(self, scope, receive, send):
                    if scope.get("type") != "http":
                        return await self.inner(scope, receive, send)
                    path = scope.get("path", "?")[:64]
                    _HTTP_COUNTER["n"] += 1
                    _HTTP_COUNTER["paths"][path] = \
                        _HTTP_COUNTER["paths"].get(path, 0) + 1
                    if _HTTP_COUNTER["loop"] is None:
                        import asyncio as _aio
                        _HTTP_COUNTER["loop"] = _aio.get_running_loop()
                    status_holder = {}

                    async def _send_wrapper(message):
                        if message.get("type") == "http.response.start":
                            status_holder["s"] = message.get("status")
                        return await send(message)

                    try:
                        await self.inner(scope, receive, _send_wrapper)
                        s = status_holder.get("s", "?")
                        _HTTP_COUNTER["st"][s] = _HTTP_COUNTER["st"].get(s, 0) + 1
                    except Exception as exc:
                        key = f"EXC:{type(exc).__name__}"
                        _HTTP_COUNTER["st"][key] = \
                            _HTTP_COUNTER["st"].get(key, 0) + 1
                        raise

            _orig_stack = _groutes.App.build_middleware_stack

            def _counting_stack(self):
                return _CountingASGI(_orig_stack(self))

            _groutes.App.build_middleware_stack = _counting_stack
            print("[Deploy#18] HTTP-Counter-Middleware installiert.", flush=True)
        except Exception as exc:
            print(f"[Deploy#18] HTTP-Counter NICHT installiert "
                  f"(fail-soft): {exc!r}", flush=True)

        def _spawn_jit_warmup():
            """t4-small-Befund 2026-10-08: erste bonsai-Generierung nach
            Boot = ~8.5 min Triton-Compile (GF3-GEMV + fla-GDN auf 2-vCPU)
            am Stück. Warmup-Daemon-Thread kompiliert die Kernels beim
            BOOT (nach Prefetch, im User-Chat-UI schon verfügbar), statt
            im ersten User-Request zu hängen. Env PX_STARTUP_WARMUP:
            "auto" (Default) → nur auf HF-Space mit GPU; "1"/"0" forcing."""
            import threading
            from server import manager
            mode = os.environ.get("PX_STARTUP_WARMUP", "auto").strip().lower()
            if mode in ("0", "off", "false", "no"):
                return
            # "auto" (und alle on-Werte): immer anstellen — _load_model
            # enthält den CUDA-Guard (ValueError <1 s bei cpu-basic)
            # → fail-soft, kein Boot-Risiko.

            # Boot-Race-Gate (Docker-Migration 2026-10-09): ein User-Chat,
            # der DURCH den laufenden Warmup tritt, löste einen parallelen
            # Zweit-Load aus (Live-Beweis debug19_journal 04:57-05:08:
            # zwei Instanzen im VRAM, generate sah den Phase-2-Patch-Swap
            # unter sich → 0 Tokens nach 451-s-Compile) und der Seed
            # ("erster Chat = Cache-Hit") kam 10 min zu spät. Flag VOR dem
            # Thread-Start gesetzt (kein µs-Fenster), im Runner exception-
            # sicher wieder gelöscht; get_model wartet darauf statt selbst
            # zu laden — danach trifft der Chat direkt das geseedete Entry.
            manager._boot_warmup_running = True

            def _boot_warmup_runner():
                try:
                    manager.warmup_bonsai_jit()
                finally:
                    manager._boot_warmup_running = False

            threading.Thread(
                target=_boot_warmup_runner,
                daemon=True, name="px-jit-warmup").start()

        def _spawn_mem_heartbeat():
            """Plan t4-wheel (Deploy-#17+#18): RAM-Heartbeat für die stille
            Container-Tod-Frage (Tode 0.6-19.6 min nach Phase-2, KEIN Banner
            — SIGKILL- oder Freeze-Klasse). Alle 30 s: VmRSS/VmHWM +
            cgroup-memory.current + HTTP-Counter-Δ (Status-Raten — der
            warnings-Dedup versteckt die 422-Sturm-Rate) +
            run_coroutine_threadsafe-Loop-Probe (loop=OK/STUCK unterscheidet
            Prozess-Tod von eingefrorenem Event-Loop). Der letzte Heartbeat
            vor einem Boot-Wechsel = der Zustand beim Tod. Fehlschlag-tolerant
            (cgroup-v1/v2/ohne → no-op)."""
            import threading

            def _work():
                import time
                n = 0
                prev = {}  # Deploy-#18: Counter-Snapshot für Δ-Raten
                while True:
                    n += 1
                    try:
                        rss = hwm = "-"
                        with open("/proc/self/status") as fh:
                            for line in fh:
                                if line.startswith("VmRSS:"):
                                    rss = line.split()[1]
                                elif line.startswith("VmHWM:"):
                                    hwm = line.split()[1]
                        cg = ""
                        try:
                            with open("/sys/fs/cgroup/memory.current") as fh:
                                cur = int(fh.read().strip())
                            with open("/sys/fs/cgroup/memory.max") as fh:
                                mx = fh.read().strip()
                            if mx != "max":
                                mx = f"{int(mx)//(1024*1024)}m"
                            cg = f" cg={cur//(1024*1024)}m/{mx}"
                        except Exception:
                            pass
                        cu = ""
                        try:
                            import torch
                            if torch.cuda.is_available() \
                                    and torch.cuda.is_initialized():
                                a = torch.cuda.memory_allocated() / (1 << 30)
                                r = torch.cuda.memory_reserved() / (1 << 30)
                                cu = f" cuda={a:.2f}/{r:.2f}GiB"
                        except Exception:
                            pass
                        th = len(threading.enumerate())
                        # Deploy-#19 (D1): GPU-Utilization — "es generiert"
                        # (User-Meldung 2026-10-09: GPU-Aktivität sichtbar,
                        # Text kam nicht an — Tod vor erstem Token) wurde
                        # bisher durch Messzahlen unbestätigt; util>0 in der
                        # letzten MEM-Zeile vor einem Tod beweist Compile/
                        # Decode. pynvml, Fallback nvidia-smi-Binary.
                        gu = ""
                        try:
                            import pynvml as _nv
                            _nv.nvmlInit()
                            _h = _nv.nvmlDeviceGetHandleByIndex(0)
                            _u = _nv.nvmlDeviceGetUtilizationRates(_h)
                            gu = f" gpu={_u.gpu}%"
                        except Exception:
                            try:
                                import subprocess as _sp
                                _r = _sp.run(
                                    ["nvidia-smi",
                                     "--query-gpu=utilization.gpu",
                                     "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True,
                                    timeout=10)
                                _v = _r.stdout.strip()
                                gu = f" gpu={_v}%" if _v else " gpu=?"
                            except Exception:
                                gu = " gpu=?"
                        # Deploy-#18: HTTP-Counter-Delta (Rate je 30-s-
                        # Fenster — macht die 422-Sturm-Rate sichtbar, die
                        # der warnings-Dedup versteckt) + Loop-Probe:
                        # run_coroutine_threadsafe(sleep(0)) auf dem in der
                        # Middleware gekaperten Server-Loop; 10-s-Timeout
                        # unterscheidet "Prozess tot" (kein MEM mehr) von
                        # "Event-Loop eingefroren" (MEM läuft, loop=STUCK).
                        http = ""
                        loopinfo = ""
                        try:
                            cur_n = _HTTP_COUNTER["n"]
                            d_n = cur_n - prev.get("n", 0)
                            d_st = {k: v - prev.get("st", {}).get(k, 0)
                                    for k, v in _HTTP_COUNTER["st"].items()}
                            d_paths = {k: v - prev.get("paths", {}).get(k, 0)
                                       for k, v in
                                       _HTTP_COUNTER["paths"].items()}
                            prev = {"n": cur_n,
                                    "st": dict(_HTTP_COUNTER["st"]),
                                    "paths": dict(_HTTP_COUNTER["paths"])}
                            if cur_n or d_n:
                                st_txt = ",".join(
                                    f"{k}:{v}" for k, v in sorted(d_st.items())
                                    if v) or "-"
                                p_txt = ",".join(
                                    f"{k.split('?')[0]}:{v}" for k, v in
                                    sorted(d_paths.items(),
                                           key=lambda kv: -kv[1])[:4]) or "-"
                                http = f" httpΔ={d_n} ({st_txt} | {p_txt})"
                            loop = _HTTP_COUNTER["loop"]
                            if loop is not None and loop.is_running():
                                import asyncio as _aio
                                t0 = time.monotonic()
                                try:
                                    fut = _aio.run_coroutine_threadsafe(
                                        _aio.sleep(0), loop)
                                    fut.result(timeout=10)
                                    loopinfo = (
                                        f" loop=OK("
                                        f"{time.monotonic() - t0:.2f}s)")
                                except Exception as exc:
                                    loopinfo = f" loop=STUCK({exc!r})"
                            elif loop is not None:
                                loopinfo = " loop=NOT_RUNNING"
                        except Exception as exc:
                            http = f" http=ERR({exc!r})"
                        print(f"[MEM#{n} {time.strftime('%H:%M:%S')}] "
                              f"VmRSS={rss}kB VmHWM={hwm}kB "
                              f"(threads={th}){cg}{cu}{gu}{http}{loopinfo}",
                              flush=True)
                    except Exception as exc:  # Heartbeat darf nicht töten
                        print(f"[MEM#{n}] heartbeat-FEHLER: {exc!r}",
                              flush=True)
                    time.sleep(30)

            threading.Thread(target=_work, daemon=True,
                             name="px-mem-heartbeat").start()

        def _spawn_gc_keepalive():
            """Deploy-#19 (D4): Chat-Keepalive gegen den Platform-Auto-Sleep
            (gcTimeout=300 s, ZeroGPU-Ära, NICHT autonom änderbar — User:
            "sleep soll 5 minuten"). Der gc zählt NUR abgeschlossene
            HTTP-Requests — offene /queue/data-SSE-Verbindungen und GPU-Load
            zählen nicht; deshalb starb am 2026-10-09 der Cold-Wake-Chat
            mitten im ~6-min-Triton-Compile, BEVOR der erste Token ankam
            (User: "2x hallo, keine Antwort"). Pinger: alle 20 s ein GET auf
            die ÖFFENTLICHE /config-Route (intern-localhost zählt nicht, der
            gc sitzt an der Plattform-Logstufe) — NUR solange _CHAT_ACTIVE
            ["n"] > 0 (Chat-Tab-Zähler, stamp refreshed je Yield). Leerlauf
            → keine Pings → der 5-min-Sleep bleibt (kein Dauer-Billing).
            Leak-Schutz: Flag >900 s ohne Yield-Refresh → Reset + Print."""
            import threading

            def _kwork():
                import time
                import urllib.request
                space_id = os.environ.get("SPACE_ID", "")
                if not space_id:
                    print("[Keepalive] SPACE_ID nicht gesetzt — pinger aus.",
                          flush=True)
                    return
                url = ("https://" + space_id.replace("/", "-")
                       + ".hf.space/config")
                print(f"[Keepalive] armiert: {url} (nur bei aktivem Chat)",
                      flush=True)
                pings = 0
                while True:
                    time.sleep(20)
                    try:
                        from gradio_tabs.chat_tab import _CHAT_ACTIVE
                        n_act = _CHAT_ACTIVE["n"]
                        if n_act <= 0:
                            continue
                        if time.monotonic() - _CHAT_ACTIVE["stamp"] > 900.0:
                            print("[Keepalive] Flag-Leak (>900s ohne Yield-"
                                  "Refresh) — Reset (5-min-Sleep kehrt zurück).",
                                  flush=True)
                            _CHAT_ACTIVE["n"] = 0
                            continue
                        with urllib.request.urlopen(url, timeout=15) as _resp:
                            _resp.read(64)
                        pings += 1
                        if pings % 10 == 1:
                            print(f"[Keepalive] ping #{pings} (active="
                                  f"{n_act})", flush=True)
                    except Exception:
                        pass

            threading.Thread(target=_kwork, daemon=True,
                             name="gc-keepalive").start()

        # Plan 2026-07-09: ssr_mode=False ist KRITISCH auf HF-Space.
        # Default (True) startet einen Node-SSR-Proxy auf 7860, der im
        # HF-Container scheitert (kein Node installiert). Gradio fällt
        # auf 7861 zurück, aber SvelteKit-SSR-Loader will trotzdem
        # /info JSON von 127.0.0.1:7861 fetchen — was der BROWSER nicht
        # kann (localhost → user-machine). Resultat: data:null im
        # Page-Render → "Could not get API info" → Login-Panel.
        # ssr_mode=False → client-side-render → /info ist relative URL
        # → HF-Proxy leitet korrekt zu Gradio-Server auf 7861 weiter.
        #
        # Deploy-#11-Befund 2026-10-09 (Boot-CRASH, RUNTIME_ERROR): der
        # Warmup-Thread VOR launch() stallt gradios Localhost-Check
        # (blocks.py:3014 `url_ok` — EIN 3-s-Timeout/ConnectError = sofort
        # False; die 5-Loop gilt nur für Nicht-2xx-Antworten), sobald
        # model_manager.load parallel importiert/streamt → ValueError
        # "localhost is not accessible". Fix: launch kehrt mit
        # prevent_thread_lock=True NACH dem Check zurück, der Warmup wird
        # erst danach angestellt, und block_thread() (gradio-eigener
        # Shutdown-Wait, identisch mit dem Skript-Default-Pfad) hält den
        # Prozess am Leben. Ohne Warmup (PX_STARTUP_WARMUP=0) ist das
        # Laufzeitverhalten exakt wie vorher.
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            ssr_mode=False,
            prevent_thread_lock=True,
        )
        _spawn_jit_warmup()
        _spawn_mem_heartbeat()
        _spawn_gc_keepalive()
        demo.block_thread()
    else:
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