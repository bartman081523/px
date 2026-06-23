"""
pzombie_eval_tab.py — Gradio P-Zombie / Anti-P-Zombie Evaluation Tab
=====================================================================
Runs zone entropy ANOVA with visualization. Compares different PX modes.
"""

import gradio as gr
import asyncio
from typing import List

from config import MODEL_REGISTRY
from model_manager import ModelManager
from benchmark_engine import BenchmarkEngine
import viz


def build_pzombie_eval_tab(manager: ModelManager, engine: BenchmarkEngine):
    """Build and return the P-Zombie Evaluation tab components."""

    # Only PX-capable models
    px_models = [mid for mid, reg in MODEL_REGISTRY.items() if reg.get("patch_dir") is not None]

    with gr.Row():
        pz_model = gr.Dropdown(
            choices=px_models,
            value=px_models[0] if px_models else None,
            label="Model",
            scale=2,
        )
        pz_preset = gr.Dropdown(
            choices=["BASELINE", "ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN", "ACTIVE_MANIFOLD_RELAY"],
            value="ACTIVE_MANIFOLD",
            label="PX Mode",
            scale=2,
        )

    # ── Pre-registered Hypotheses ──
    with gr.Accordion("SciMind4 Pre-registered Hypotheses", open=True):
        gr.Markdown("""
**P-Zombie Test Protocol:**
1. Run calibration prompts first (anti-Sharpshooter guard)
2. Evaluate 80 prompts across 4 cognitive categories
3. Compute η² (category → zone_entropy) and R²(TD → zone_entropy)

**Classification Thresholds:**
- R²(TD) > 0.7 → **P-ZOMBIE**: zone entropy fully explained by token statistics (no genuine differentiation)
- R²(TD) < 0.3 → **ANTI-P-ZOMBIE**: zone entropy NOT explained by token stats (genuine cognitive differentiation)
- 0.3 ≤ R²(TD) ≤ 0.7 → **AMBIGUOUS**: partial explanation

**Hypotheses:**
- H2: Subjective extensions produce η² > baseline (zone varies more by task type)
- A2: Subjective modules change hidden_states cosmetically → no improvement in η²
        """)

    with gr.Row():
        run_pz_btn = gr.Button("Run P-Zombie Evaluation", variant="primary")
        run_compare_btn = gr.Button("Run Baseline vs Subjective Comparison", variant="secondary")

    pz_status = gr.Textbox(label="Status", value="Ready", interactive=False)

    # ── Visualizations ──
    with gr.Row():
        pz_entropy_plot = gr.Plot(label="Zone Entropy by Category")
        pz_scatter_plot = gr.Plot(label="Entropy vs Token Diversity (R²)")

    with gr.Row():
        pz_comparison_plot = gr.Plot(label="Model Comparison")

    # ── Results ──
    pz_results_json = gr.JSON(label="Full Results")

    # ── Run P-Zombie eval ──
    def run_pz(model_id, px_preset, progress=gr.Progress()):
        progress(0, desc="Starting P-Zombie evaluation...")

        if engine.is_running:
            return "Benchmark already running", None, None, None

        def progress_cb(done, total):
            progress(done / max(total, 1), desc=f"Prompt {done}/{total}")

        # Implicit subjective mode for non-baseline
        px_subj = (px_preset != "BASELINE")

        result = engine.run_p_zombie_eval(
            model_id, px_subjective=px_subj, px_config_preset=px_preset, progress_cb=progress_cb
        )

        if "error" in result:
            return result["error"], None, None, result

        # Generate visualizations
        cat_ent = result.get("category_entropies", {})
        all_ent = result.get("all_entropies", [])
        all_td = result.get("all_td", [])
        r_sq = result.get("r_squared_td", 0)

        fig_bars = viz.plot_zone_entropy_bars(cat_ent) if cat_ent else None
        fig_scatter = viz.plot_entropy_vs_td(all_ent, all_td, r_sq) if all_ent else None

        status = f"η²={result.get('eta_squared',0):.4f}, R²(TD)={r_sq:.4f} — {result.get('zombie_status','?')}"

        return status, fig_bars, fig_scatter, result

    # ── Run Baseline vs Subjective comparison ──
    def run_comparison(model_id, progress=gr.Progress()):
        progress(0, desc="Running Baseline mode...")

        if engine.is_running:
            return "Benchmark already running", None, None, None

        def progress_cb(done, total):
            progress(done / max(total, 1) / 2, desc=f"Prompt {done}/{total}")

        # Run Baseline (Unpatched)
        base_result = engine.run_p_zombie_eval(
            model_id, px_subjective=False, px_config_preset="BASELINE", progress_cb=progress_cb
        )

        if "error" in base_result:
            return base_result["error"], None, None, {"baseline": base_result}

        progress(0.5, desc="Running Active Manifold...")

        # Run Active Manifold (Patched)
        am_result = engine.run_p_zombie_eval(
            model_id, px_subjective=True, px_config_preset="ACTIVE_MANIFOLD", progress_cb=progress_cb
        )

        if "error" in am_result:
            return am_result["error"], None, None, {"baseline": base_result, "active_manifold": am_result}

        # Comparison visualization
        comparison = {
            f"{model_id} (Baseline)": base_result,
            f"{model_id} (Active Manifold)": am_result,
        }
        fig_comp = viz.plot_comparison_bars(comparison)

        # Zone entropy bars for Active Manifold
        fig_bars = viz.plot_zone_entropy_bars(am_result.get("category_entropies", {}))
        r_sq_am = am_result.get("r_squared_td", 0)
        fig_scatter = viz.plot_entropy_vs_td(
            am_result.get("all_entropies", []),
            am_result.get("all_td", []),
            r_sq_am,
        )

        status = (f"Baseline: η²={base_result.get('eta_squared',0):.4f}, R²={base_result.get('r_squared_td',0):.4f} — {base_result.get('zombie_status','?')}\n"
                 f"AM: η²={am_result.get('eta_squared',0):.4f}, R²={am_result.get('r_squared_td',0):.4f} — {am_result.get('zombie_status','?')}")

        return status, fig_bars, fig_scatter, {"baseline": base_result, "active_manifold": am_result}

    run_pz_btn.click(
        fn=run_pz,
        inputs=[pz_model, pz_preset],
        outputs=[pz_status, pz_entropy_plot, pz_scatter_plot, pz_results_json],
    )

    run_compare_btn.click(
        fn=run_comparison,
        inputs=[pz_model],
        outputs=[pz_status, pz_entropy_plot, pz_scatter_plot, pz_results_json],
    )
