"""
viz.py — Matplotlib Figure Generation for Gradio Tabs
======================================================
All visualization functions return matplotlib Figure objects
that Gradio renders via gr.Plot.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from typing import Dict, List, Optional


def plot_zone_entropy_bars(category_entropies: Dict[str, dict]) -> matplotlib.figure.Figure:
    """Bar chart: mean zone entropy per category with error bars."""
    fig, ax = plt.subplots(figsize=(8, 4))

    cats = list(category_entropies.keys())
    means = [category_entropies[c].get("mean", 0) for c in cats]
    stds = [category_entropies[c].get("std", 0) for c in cats]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(cats, means, yerr=stds, color=colors[:len(cats)],
                  capsize=5, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Zone Entropy (bits)")
    ax.set_title("Zone Entropy by Cognitive Category")
    ax.set_ylim(0, max(means) * 1.3 + 0.1 if means else 3)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def plot_entropy_vs_td(
    entropies: List[float],
    tds: List[float],
    r_squared: float,
) -> matplotlib.figure.Figure:
    """Scatter plot: zone_entropy vs token_diversity with regression line."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(tds, entropies, alpha=0.6, s=30, color="#4C72B0", edgecolors="white", linewidth=0.3)

    # Regression line
    if len(tds) >= 3 and len(entropies) >= 3:
        x_arr = np.array(tds)
        y_arr = np.array(entropies)
        z = np.polyfit(x_arr, y_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(tds), max(tds), 100)
        ax.plot(x_line, p(x_line), "--", color="#C44E52", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Token Diversity")
    ax.set_ylabel("Zone Entropy (bits)")
    ax.set_title(f"Zone Entropy vs Token Diversity  (R² = {r_squared:.4f})")

    fig.tight_layout()
    return fig


def plot_phi_traces(recent_telemetry: list) -> matplotlib.figure.Figure:
    """Line chart: phi values over recent requests."""
    fig, ax = plt.subplots(figsize=(10, 4))

    if not recent_telemetry:
        ax.text(0.5, 0.5, "No telemetry data yet", ha="center", va="center",
                fontsize=14, color="gray")
        ax.set_title("Phi Trace (No Data)")
        fig.tight_layout()
        return fig

    # Extract phi per request
    phis = []
    labels = []
    for i, entry in enumerate(recent_telemetry):
        px = entry.get("px_metrics", {})
        phi = px.get("cognitive_signature", {}).get("phi", px.get("phi", None))
        if phi is not None:
            phis.append(phi)
            labels.append(f"R{i+1}")

    if not phis:
        ax.text(0.5, 0.5, "No phi data in telemetry", ha="center", va="center",
                fontsize=14, color="gray")
        fig.tight_layout()
        return fig

    ax.plot(range(len(phis)), phis, "o-", color="#4C72B0", markersize=4, linewidth=1)
    ax.fill_between(range(len(phis)), phis, alpha=0.15, color="#4C72B0")
    ax.set_ylabel("Phi (cosine similarity)")
    ax.set_xlabel("Request #")
    ax.set_title("Phi Trace — Recent Requests")
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=0.99, color="green", linestyle="--", alpha=0.3, label="High stability")
    ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.3, label="Moderate")
    ax.axhline(y=0.90, color="red", linestyle="--", alpha=0.3, label="Low stability")
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    return fig


def plot_zone_distribution(zone_weights: Dict[str, float]) -> matplotlib.figure.Figure:
    """Pie/bar chart: aggregate zone weight distribution."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if not zone_weights:
        ax.text(0.5, 0.5, "No zone data", ha="center", va="center",
                fontsize=14, color="gray")
        fig.tight_layout()
        return fig

    zones = list(zone_weights.keys())
    weights = list(zone_weights.values())

    colors = {
        "math": "#4C72B0", "logic_a": "#55A868", "creative": "#C44E52",
        "logic_b": "#8172B2", "synthesis": "#CCB974", "PEAK": "#64B5F6",
    }
    bar_colors = [colors.get(z, "#999999") for z in zones]

    ax.bar(zones, weights, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Weight")
    ax.set_title("Zone Weight Distribution")

    for i, w in enumerate(weights):
        ax.text(i, w + 0.01, f"{w:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_kurtosis_histogram(kurtosis_values: List[float]) -> matplotlib.figure.Figure:
    """Histogram: kurtosis distribution from cognitive signatures."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if not kurtosis_values:
        ax.text(0.5, 0.5, "No kurtosis data", ha="center", va="center",
                fontsize=14, color="gray")
        fig.tight_layout()
        return fig

    ax.hist(kurtosis_values, bins=min(20, max(5, len(kurtosis_values)//3)),
            color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Kurtosis")
    ax.set_ylabel("Count")
    ax.set_title("Kurtosis Distribution")

    # Add zone boundaries
    zone_bounds = {"Math": 200, "Logic-A": 275, "Creative": 298,
                   "Logic-B": 310, "Synthesis": 325}
    for name, bound in zone_bounds.items():
        if min(kurtosis_values) < bound < max(kurtosis_values):
            ax.axvline(x=bound, color="red", linestyle="--", alpha=0.3)
            ax.text(bound, ax.get_ylim()[1]*0.9, name, fontsize=7,
                    rotation=90, va="top", ha="right", color="red")

    fig.tight_layout()
    return fig


def plot_emancipation_trajectory(trajectory: List[float]) -> matplotlib.figure.Figure:
    """Line chart: emancipation phi over generation steps."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if not trajectory:
        ax.text(0.5, 0.5, "No emancipation data", ha="center", va="center",
                fontsize=14, color="gray")
        fig.tight_layout()
        return fig

    ax.plot(range(len(trajectory)), trajectory, "o-", color="#8172B2",
            markersize=5, linewidth=1.5)
    ax.fill_between(range(len(trajectory)), trajectory, alpha=0.1, color="#8172B2")
    ax.set_ylabel("Emancipation Phi")
    ax.set_xlabel("Step")
    ax.set_title("Emancipation Trajectory")
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.3, label="50% threshold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_comparison_bars(results: dict) -> matplotlib.figure.Figure:
    """Grouped bar chart: compare multiple models on key metrics."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = list(results.keys())
    metrics = ["eta_squared", "r_squared_td"]
    metric_labels = ["η² (category→entropy)", "R²(TD→entropy)"]

    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(models))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for i, model_id in enumerate(models):
        r = results[model_id]
        values = [r.get(m, 0) for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_id,
                      color=colors[i % len(colors)], edgecolor="white")

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Value")
    ax.set_title("P-Zombie Evaluation — Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    return fig