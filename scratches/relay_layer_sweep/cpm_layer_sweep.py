"""scratches/relay_layer_sweep/cpm_layer_sweep.py — Layer-Sweep auf MiniCPM.

Hintergrund (Plan: branch relay-layer-discovery, 2026-07-02):
  Fährt einen Layer-Sweep über mehrere Schichten des MiniCPM-Modells und
  bestimmt für jeden Layer das R² (WIDE vs NARROW linear separabel?).
  Output: scratches/relay_layer_sweep/out/cpm_layers.json mit
    { "model_id", "hf_id", "hidden_size", "n_layers",
      "layers": { L: {r2, n_prompts, sep, n_trainable_params} } }

  Plus: scratches/relay_layer_sweep/out/cpm_relay_dwidth.json (Artefakt
  für die beste Layer) — kann dann vom Resolver gelesen werden.

  Verwendet:
    - scratches/relay_layer_sweep/dwidth_lib (pure-logic helpers)
    - scratches/relay_layer_sweep/battery (7 prompts, 3 kategorien)
    - px_patches/_relay_layer_resolver (für update_cache_entry am Ende)

CLI:
  python scratches/relay_layer_sweep/cpm_layer_sweep.py --model-id openbmb/MiniCPM-1B-sft-bf16
  python scratches/relay_layer_sweep/cpm_layer_sweep.py --smoke  # nur 1 prompt, 2 layers

Output (committed):
  scratches/relay_layer_sweep/out/cpm_layers.json
  scratches/relay_layer_sweep/out/cpm_relay_dwidth.json (beste Layer)
  scratches/relay_layer_sweep/out/cpm_RELAY_LAYER_REPORT.md
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import datetime
from pathlib import Path

import numpy as np

# Repo-Root in sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from scratches.relay_layer_sweep.dwidth_lib import (
    load_captures_for_layer,
    build_state_directions,
    format_dwidth_artefact,
    atomic_write_json,
    linear_decoder_r2,
)
from scratches.relay_layer_sweep.battery import (
    BATTERY, get_layer_range,
)


OUT_DIR = os.path.join(HERE, "out")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Layer-Sweep über PX-Engine (MiniCPM-1B-sft-bf16 default)"
    )
    p.add_argument(
        "--model-id", default="openbmb/MiniCPM-1B-sft-bf16",
        help="HF model_id (default: MiniCPM-1B-sft-bf16)"
    )
    p.add_argument(
        "--n-layers", type=int, default=None,
        help="Anzahl Layer (default: auto-detect via model.config.num_hidden_layers)"
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke-Mode: 1 prompt × 2 layers × 1 seed (schnell)"
    )
    p.add_argument(
        "--captures-dir", default=None,
        help="Wurzel mit Captures (default: out/captures/{model_safe_id}/)"
    )
    p.add_argument(
        "--out-dir", default=OUT_DIR,
        help=f"Output-Verzeichnis (default: {OUT_DIR})"
    )
    return p.parse_args()


# ── Sweep-Funktionen ──────────────────────────────────────────────────

def detect_n_layers(model_id: str) -> int:
    """Versucht n_layers aus dem Modell-Config zu lesen. Fallback = 40."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        n = getattr(cfg, "num_hidden_layers", None) or getattr(
            getattr(cfg, "text_config", None), "num_hidden_layers", None
        )
        if n:
            return int(n)
    except Exception as e:
        print(f"[sweep] WARN: AutoConfig fehlgeschlagen für {model_id}: {e}",
              file=sys.stderr)
    return 40  # MiniCPM-1B default


def safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "_")


def run_sweep(args) -> dict:
    """Fährt den Layer-Sweep und gibt das Result-Dict zurück.

    HINWEIS: Diese Funktion ist die "Wire-up"-Schicht. Sie erwartet:
      {captures_dir}/{arm}/{pid}_L{layer}.pt für alle arms × pids × layers.

    Das Wire-up (PX-Engine: generate + capture hidden_states) wird in
    einer separaten Funktion `run_capture()` ergänzt, die aber GPU + Modell
    braucht. Hier in cpm_layer_sweep.py NUR die Auswertung der bereits
    existierenden Captures.
    """
    model_id = args.model_id
    n_layers = args.n_layers or detect_n_layers(model_id)
    layer_range = get_layer_range(model_id, n_layers)
    if args.smoke:
        # Smoke: 1 prompt, 2 layers
        layer_range = layer_range[:2]

    captures_dir = args.captures_dir or os.path.join(
        args.out_dir, "captures", safe_model_id(model_id)
    )
    if not os.path.isdir(captures_dir):
        print(f"[sweep] FEHLER: captures_dir nicht gefunden: {captures_dir}",
              file=sys.stderr)
        print(f"[sweep] Erzeuge zuerst Captures via run_capture.py --model-id {model_id}",
              file=sys.stderr)
        sys.exit(1)

    # Layer-Sweep
    arms = ["BASELINE", "NARROW", "DEFAULT", "WIDE"]
    layer_results = {}
    print(f"[sweep] n_layers={n_layers} | sweep range: {layer_range}", file=sys.stderr)

    for layer in layer_range:
        try:
            d_width, means, common, sep = build_state_directions(
                captures_dir, hi="WIDE", lo="NARROW", arms=arms,
                layer=layer,
            )
        except Exception as e:
            print(f"[sweep] WARN build_state_directions@{layer}: {e}", file=sys.stderr)
            continue

        # Linear-Decoder: kann WIDE vs NARROW pro Layer trennen?
        try:
            data = load_captures_for_layer(captures_dir, arms=arms, layer=layer)
            wide_h = np.stack([data["WIDE"][p] for p in common])
            narrow_h = np.stack([data["NARROW"][p] for p in common])
            X = np.vstack([wide_h, narrow_h])
            y = np.concatenate([np.ones(len(common)), np.zeros(len(common))])
            r2 = linear_decoder_r2(X, y, l2=1.0)
        except Exception as e:
            print(f"[sweep] WARN linear_decoder@{layer}: {e}", file=sys.stderr)
            r2 = float("nan")

        layer_results[layer] = {
            "r2": r2,
            "n_prompts": len(common),
            "sep_WIDE_NARROW": sep,
        }
        print(f"[sweep] L{layer:02d}  r2={r2:+.3f}  sep={sep:.3f}  "
              f"n_prompts={len(common)}", file=sys.stderr)

    # Beste Layer
    best_layer = None
    if layer_results:
        best_layer = max(
            layer_results.items(),
            key=lambda kv: kv[1]["r2"] if not np.isnan(kv[1]["r2"]) else float("-inf"),
        )[0]

    result = {
        "model_id": model_id,
        "n_layers": n_layers,
        "layer_range_tested": layer_range,
        "arms": arms,
        "layers": {str(k): v for k, v in layer_results.items()},
        "best_layer": best_layer,
        "best_layer_r2": (layer_results[best_layer]["r2"]
                          if best_layer is not None else None),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return result


def write_outputs(result: dict, args) -> dict:
    """Schreibt cpm_layers.json + cpm_relay_dwidth.json + Report.md.

    Returns dict mit Pfaden.
    """
    os.makedirs(args.out_dir, exist_ok=True)
    model_id = result["model_id"]
    safe_id = safe_model_id(model_id)

    # 1) cpm_layers.json
    layers_path = os.path.join(args.out_dir, f"{safe_id}_layers.json")
    atomic_write_json(layers_path, result)
    print(f"[sweep] saved: {layers_path}", file=sys.stderr)

    paths = {"layers": layers_path}

    # 2) cpm_relay_dwidth.json (Artefakt für die beste Layer)
    best = result.get("best_layer")
    if best is not None:
        # Lade Captures für beste Layer neu
        captures_dir = args.captures_dir or os.path.join(
            args.out_dir, "captures", safe_id
        )
        try:
            d_width, means, common, sep = build_state_directions(
                captures_dir, hi="WIDE", lo="NARROW",
                arms=result["arms"],
            )
            # hidden_size aus erster Capture
            sample = load_captures_for_layer(
                captures_dir, arms=result["arms"], layer=best,
            )
            hidden = next(iter(sample[result["arms"][0]].values())).shape[-1]
            artefact = format_dwidth_artefact(
                model_id=model_id.split("/")[-1],
                hf_id=model_id,
                hidden_size=hidden,
                capture_layer=best,
                inject_layer=best + 1,  # default: eine Layer nach capture
                dwidth=d_width,
                sep=sep,
                common=common,
                direction=f"WIDE_minus_NARROW_L{best}",
            )
            dwidth_path = os.path.join(args.out_dir, f"{safe_id}_relay_dwidth.json")
            atomic_write_json(dwidth_path, artefact)
            paths["dwidth"] = dwidth_path
            print(f"[sweep] saved: {dwidth_path}", file=sys.stderr)
        except Exception as e:
            print(f"[sweep] WARN: konnte dwidth-Artefakt nicht schreiben: {e}",
                  file=sys.stderr)

    # 3) Report.md (menschliche Lesung)
    report_path = os.path.join(args.out_dir, f"{safe_id}_RELAY_LAYER_REPORT.md")
    write_report(result, report_path, args)
    paths["report"] = report_path
    print(f"[sweep] saved: {report_path}", file=sys.stderr)

    return paths


def write_report(result: dict, path: str, args) -> None:
    """Schreibt eine Markdown-Report-Datei (für Commit + manuelle Lesung)."""
    lines = []
    lines.append(f"# Layer-Sweep Report — {result['model_id']}")
    lines.append("")
    lines.append(f"**Generated:** {result['timestamp']}  ")
    lines.append(f"**n_layers:** {result['n_layers']}  ")
    lines.append(f"**layer_range_tested:** {result['layer_range_tested']}  ")
    lines.append(f"**arms:** {result['arms']}  ")
    lines.append(f"**CLI:** `python cpm_layer_sweep.py {' '.join(sys.argv[1:])}`  ")
    lines.append("")

    lines.append("## Per-Layer Mechanik")
    lines.append("")
    lines.append("| Layer | R² (LOO-CV) | sep(WIDE-NARROW) | n_prompts |")
    lines.append("|------:|:-----------:|:----------------:|:---------:|")
    for layer in sorted(result["layer_range_tested"],
                        key=lambda x: int(x) if isinstance(x, int) else x):
        if str(layer) in result["layers"]:
            d = result["layers"][str(layer)]
            r2 = d["r2"]
            r2_str = f"{r2:+.3f}" if not np.isnan(r2) else "n/a"
            lines.append(f"| L{layer:02d} | {r2_str} | {d['sep_WIDE_NARROW']:.3f} | "
                         f"{d['n_prompts']} |")
    lines.append("")

    best = result.get("best_layer")
    if best is not None:
        lines.append(f"## Beste Layer: L{best}")
        lines.append("")
        best_r2 = result["best_layer_r2"]
        lines.append(f"- **R²:** {best_r2:+.3f}")
        lines.append("- **Empfehlung:** `find_relay_layer(mode='cached')` "
                     f"returnt jetzt L{best} für `{result['model_id']}`.")
        lines.append("- **Manuelle Lesung der Antworten (siehe out/captures/"
                     f"{safe_model_id(result['model_id'])}/{{arm}}/{{pid}}_L{best}.pt) "
                     "ist PFLICHT — Mechanik allein reicht nicht (DevMind-Regel).")
    else:
        lines.append("## Beste Layer")
        lines.append("")
        lines.append("**Keine Layer gefunden** — bitte captures_dir prüfen.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"[sweep] starting sweep for {args.model_id}", file=sys.stderr)
    result = run_sweep(args)
    paths = write_outputs(result, args)
    print(f"[sweep] done. paths: {paths}", file=sys.stderr)


if __name__ == "__main__":
    main()
