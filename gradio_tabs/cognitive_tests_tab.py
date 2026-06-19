"""
cognitive_tests_tab.py — Gradio Cognitive Tests Tab
====================================================
Run capability benchmarks, rigorous evaluations against any model.
"""

import gradio as gr
import json
import asyncio
from typing import Optional

from config import MODEL_REGISTRY
from model_manager import ModelManager
from benchmark_engine import BenchmarkEngine


def build_cognitive_tests_tab(manager: ModelManager, engine: BenchmarkEngine):
    """Build and return the Cognitive Tests tab components."""

    model_choices = list(MODEL_REGISTRY.keys())

    with gr.Row():
        test_model = gr.Dropdown(
            choices=model_choices,
            value=model_choices[0],
            label="Model",
            scale=3,
        )
        test_preset = gr.Dropdown(
            choices=["BASELINE", "ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN"],
            value="ACTIVE_MANIFOLD",
            label="PX Mode",
            scale=2,
        )

    with gr.Row():
        test_type = gr.Radio(
            choices=["Capability Benchmark", "Ultra Hard Benchmark", "P-Zombie Evaluation", "Baseline Comparison"],
            value="Capability Benchmark",
            label="Test Type",
        )

    # ── Preregistered Hypotheses (Anti-Texas-Sharpshooter) ──
    with gr.Accordion("Pre-registered Hypotheses (SciMind4 Protocol)", open=False):
        gr.Markdown("""
**Capability Benchmark:**
- H1: PX-patched models maintain or improve accuracy vs unpatched baseline
- A1: PX recursive loops degrade output quality (regression)

**P-Zombie Evaluation:**
- H2: Subjective extensions produce measurably different cognitive signatures
- A2: Subjective modules change hidden_states cosmetically without functional improvement
- Classification: R²(TD) < 0.3 → ANTI-P-ZOMBIE, > 0.7 → P-ZOMBIE

**Baseline Comparison:**
- H3: PX patch causes no regression on logic/math tasks
- A3: PX patch harms task performance
        """)

    run_btn = gr.Button("Run Test", variant="primary")
    status_text = gr.Textbox(label="Status", value="Ready", interactive=False)

    # Results
    with gr.Row():
        results_df = gr.Dataframe(
            headers=["Category", "Accuracy", "Tasks"],
            label="Summary Results",
            row_count=5,
            column_count=3,
        )
        results_json = gr.JSON(label="Full Results")

    # ── Run test callback ──
    def run_test(model_id, px_preset, test_type_name, progress=gr.Progress()):
        progress(0, desc="Starting...")

        if engine.is_running:
            return "Benchmark already running — please wait", None, None

        def progress_cb(done, total):
            progress(done / max(total, 1), desc=f"Task {done}/{total}")

        # Subjective is enabled for anything non-baseline
        px_subj = (px_preset != "BASELINE")

        if test_type_name == "Capability Benchmark":
            result = engine.run_capability_benchmark(
                model_id, px_subjective=px_subj, px_config_preset=px_preset, progress_cb=progress_cb
            )
            if "error" in result:
                return result["error"], None, None

            # Build summary dataframe
            rows = [
                ("Logic", result.get("logic_accuracy", 0), 30),
                ("Math", result.get("math_accuracy", 0), 10),
                ("Overall", result.get("overall_accuracy", 0), 40),
            ]
            # Add PX metrics if available
            px = result.get("px_metrics", {})
            if px.get("patched", True):
                rows.append(("PX Zone", px.get("zone", "N/A"), "-"))
                rows.append(("PX Phi", f"{px.get('cognitive_signature', {}).get('phi', 'N/A'):.4f}" if isinstance(px.get('cognitive_signature', {}).get('phi'), float) else "N/A", "-"))

            return f"Done — Accuracy: {result['overall_accuracy']:.1%}", rows, result

        elif test_type_name == "Ultra Hard Benchmark":
            result = engine.run_ultra_hard_benchmark(
                model_id, px_subjective=px_subj, px_config_preset=px_preset, progress_cb=progress_cb
            )
            if "error" in result:
                return result["error"], None, None

            rows = [
                ("Overall", result.get("overall_accuracy", 0), result.get("total_tasks", 0)),
            ]
            px = result.get("px_metrics", {})
            if px.get("patched", True):
                rows.append(("PX Zone", px.get("zone", "N/A"), "-"))
                rows.append(("PX Phi", f"{px.get('cognitive_signature', {}).get('phi', 'N/A'):.4f}" if isinstance(px.get('cognitive_signature', {}).get('phi'), float) else "N/A", "-"))

            return f"Done — Accuracy: {result['overall_accuracy']:.1%}", rows, result

        elif test_type_name == "P-Zombie Evaluation":
            result = engine.run_p_zombie_eval(
                model_id, px_subjective=px_subj, px_config_preset=px_preset, progress_cb=progress_cb
            )
            if "error" in result:
                return result["error"], None, result

            rows = [
                (cat, f"H={d['mean']:.4f}±{d['std']:.4f}", d["n"])
                for cat, d in result.get("category_entropies", {}).items()
            ]
            rows.append(("η²", f"{result.get('eta_squared', 0):.4f}", "-"))
            rows.append(("R²(TD)", f"{result.get('r_squared_td', 0):.4f}", "-"))
            rows.append(("Status", result.get("zombie_status", "?"), "-"))

            return f"Done — {result.get('zombie_status', 'Unknown')}", rows, result

        elif test_type_name == "Baseline Comparison":
            result = engine.run_baseline_comparison(
                model_id, px_config_preset=px_preset, progress_cb=progress_cb
            )
            if "error" in result:
                return result["error"], None, result

            base = result.get("base_result", {})
            px = result.get("px_result", {})
            rows = [
                ("Base Accuracy", f"{base.get('overall_accuracy', 0):.4f}", base.get("total_tasks", 0)),
                ("PX Accuracy", f"{px.get('overall_accuracy', 0):.4f}", px.get("total_tasks", 0)),
                ("Delta", f"{result.get('delta_accuracy', 0):+.4f}", "-"),
            ]
            return f"Done — Delta: {result.get('delta_accuracy', 0):+.2%}", rows, result

        return "Unknown test type", None, None

    run_btn.click(
        fn=run_test,
        inputs=[test_model, test_preset, test_type],
        outputs=[status_text, results_df, results_json],
    )
