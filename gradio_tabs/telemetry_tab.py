"""
telemetry_tab.py — Gradio Telemetry Visualization Tab
=====================================================
Real-time visualization of cognitive metrics: phi traces, zone distribution,
kurtosis histograms, and emancipation trajectories.
"""

import gradio as gr

from config import MODEL_REGISTRY
from model_manager import ModelManager
from telemetry import telemetry
import viz


def build_telemetry_tab(manager: ModelManager):
    """Build and return the Telemetry Visualization tab components."""

    with gr.Row():
        refresh_btn = gr.Button("Refresh", variant="primary")
        auto_refresh = gr.Checkbox(label="Auto-refresh (5s)", value=False)

    # ── Summary Stats ──
    with gr.Row():
        summary_df = gr.Dataframe(
            headers=["Metric", "Value"],
            label="Server Summary",
            row_count=5,
            column_count=2,
        )

    # ── Visualizations ──
    with gr.Row():
        phi_plot = gr.Plot(label="Phi Trace — Recent Requests")
        zone_plot = gr.Plot(label="Zone Weight Distribution")

    with gr.Row():
        kurtosis_plot = gr.Plot(label="Kurtosis Distribution")
        emancipation_plot = gr.Plot(label="Emancipation Trajectory")

    # ── Per-model details ──
    model_choices = list(MODEL_REGISTRY.keys())
    with gr.Row():
        tel_model = gr.Dropdown(
            choices=model_choices,
            value=model_choices[0],
            label="Select Model for Detail View",
        )

    with gr.Row():
        model_metrics_json = gr.JSON(label="Current Model PX Metrics")

    # ── Refresh callback ──
    def refresh_telemetry(model_id=None):
        summary = telemetry.get_summary()

        # Summary dataframe
        rows = [
            ("Total Requests", summary["total_requests"]),
            ("Total Tokens Generated", summary["total_tokens_generated"]),
            ("Total Prompt Tokens", summary["total_prompt_tokens"]),
            ("Recent Entries", len(summary["recent"])),
        ]

        # Extract data from recent entries
        phis = []
        zone_weights_agg = {}
        kurtosis_vals = []
        emancipation_trajs = []

        for entry in summary["recent"]:
            px = entry.get("px_metrics", {})
            cs = px.get("cognitive_signature", {})

            # Phi
            phi = cs.get("phi", px.get("phi"))
            if phi is not None and isinstance(phi, float):
                phis.append(phi)

            # Zone weights
            zw = px.get("zone_weights", {})
            for k, v in zw.items():
                zone_weights_agg[k] = zone_weights_agg.get(k, 0) + v

            # Kurtosis
            k = cs.get("kurtosis")
            if k is not None and isinstance(k, float):
                kurtosis_vals.append(k)

            # Emancipation
            traj = px.get("emancipation_trajectory", [])
            if traj and isinstance(traj, list):
                emancipation_trajs.extend(traj)

        # Normalize zone weights
        total_zw = sum(zone_weights_agg.values()) if zone_weights_agg else 1
        zone_weights_norm = {k: v/total_zw for k, v in zone_weights_agg.items()} if total_zw > 0 else {}

        # Generate plots
        fig_phi = viz.plot_phi_traces(summary["recent"])
        fig_zone = viz.plot_zone_distribution(zone_weights_norm)
        fig_kurtosis = viz.plot_kurtosis_histogram(kurtosis_vals) if kurtosis_vals else viz.plot_kurtosis_histogram([])
        fig_emancipation = viz.plot_emancipation_trajectory(emancipation_trajs) if emancipation_trajs else viz.plot_emancipation_trajectory([])

        # Per-model metrics
        model_metrics = manager.get_px_metrics(model_id) if model_id else {}

        return rows, fig_phi, fig_zone, fig_kurtosis, fig_emancipation, model_metrics

    refresh_btn.click(
        fn=refresh_telemetry,
        inputs=[tel_model],
        outputs=[summary_df, phi_plot, zone_plot, kurtosis_plot, emancipation_plot, model_metrics_json],
    )

    tel_model.change(
        fn=refresh_telemetry,
        inputs=[tel_model],
        outputs=[summary_df, phi_plot, zone_plot, kurtosis_plot, emancipation_plot, model_metrics_json],
    )

    # Auto-refresh timer
    timer = gr.Timer(value=5, active=False)
    timer.tick(
        fn=refresh_telemetry,
        inputs=[tel_model],
        outputs=[summary_df, phi_plot, zone_plot, kurtosis_plot, emancipation_plot, model_metrics_json],
    )

    auto_refresh.change(
        fn=lambda active: gr.Timer(active=active),
        inputs=[auto_refresh],
        outputs=[timer],
    )