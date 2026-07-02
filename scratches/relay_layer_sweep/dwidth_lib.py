"""scratches/relay_layer_sweep/dwidth_lib.py — pure-logic helpers for RELAY layer discovery.

Hintergrund (Plan: branch relay-layer-discovery, 2026-07-02):
  Generalisierung der d_width-Generierung (vorher hardcoded auf gemma3-1b-it
  in scratches/psychomotrik/save_relay_dwidth.py). Plus Linear-Decoder für
  Layer-Discovery (welche Schicht eines Modells trägt die stärkste
  Selbstwahrnehmungs-Signatur?).

Funktionen (alle pure-logic, keine Modell-Loads, keine GPU):
  - build_state_directions(captures_dir, hi, lo, arms) → (d_width, means, common, sep)
  - load_captures_for_layer(captures_dir, arms, layer) → {arm: {pid: ndarray}}
  - format_dwidth_artefact(...) → dict
  - atomic_write_json(path, data) → None (write-temp + rename)
  - linear_decoder_r2(X, y, l2) → float R²

Captures-Format (vom Layer-Sweep-Script erzeugt):
  {captures_dir}/{arm}/{pid}_L{layer}.pt mit torch.save({
    'arm': str, 'pid': str, 'layer': int, 'h': tensor[1, hidden],
  })

Verwendung:
  from scratches.relay_layer_sweep import dwidth_lib as L
  d, m, common, sep = L.build_state_directions(captures_dir, hi="WIDE", lo="NARROW")
  art = L.format_dwidth_artefact(model_id=..., dwidth=d, sep=sep, common=common, ...)
  L.atomic_write_json("px_manifolds/google_gemma-3-1b-it_relay_dwidth.json", art)
"""
from __future__ import annotations
import os
import json
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch


# ── Captures-Loader ──────────────────────────────────────────────────────

def load_captures_for_layer(
    captures_dir: str,
    arms: List[str],
    layer: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Lade alle Captures für einen Layer L (oder auto-detect wenn None).

    Args:
        captures_dir: Wurzelverzeichnis mit arm-Unterordnern.
        arms: Liste der Arm-Namen, z.B. ["WIDE", "NARROW", "DEFAULT", "BASELINE"].
        layer: Layer-Index, z.B. 16. None = auto-detect (erstes .pt-File in arms[0]).

    Returns:
        {arm: {pid: ndarray[hidden]}}. h ist gemittelt über seq_len wenn
        seq_len > 1, sonst direkt h[0].

    Raises:
        FileNotFoundError: wenn captures_dir/arm/ nicht existiert.
        ValueError: wenn keine .pt-Files gefunden werden oder layer=None und
            auto-detect scheitert.
    """
    if layer is None:
        # auto-detect: erstes .pt-File in arms[0] → layer aus Filename
        sample_armdir = os.path.join(captures_dir, arms[0])
        sample_files = [f for f in os.listdir(sample_armdir) if f.endswith(".pt")]
        if not sample_files:
            raise FileNotFoundError(f"no .pt captures in {sample_armdir}")
        fn = sample_files[0]
        try:
            layer = int(fn.rsplit("_L", 1)[-1].split(".pt")[0])
        except (IndexError, ValueError):
            raise ValueError(
                f"capture filename not in {{pid}}_L{{layer}}.pt format: {fn}"
            )
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for arm in arms:
        armdir = os.path.join(captures_dir, arm)
        if not os.path.isdir(armdir):
            raise FileNotFoundError(f"captures arm dir missing: {armdir}")
        arm_data: Dict[str, np.ndarray] = {}
        for fn in sorted(os.listdir(armdir)):
            if not fn.endswith(".pt"):
                continue
            # Dateinamen-Konvention: {pid}_L{layer}.pt
            expected = f"_L{layer}.pt"
            if not fn.endswith(expected):
                continue  # anderer Layer — skip
            path = os.path.join(armdir, fn)
            c = torch.load(path, weights_only=False)
            h = c["h"]  # tensor[1, hidden] oder [seq, hidden]
            if hasattr(h, "numpy"):
                arr = h.numpy().astype(np.float32)
            else:
                arr = np.asarray(h, dtype=np.float32)
            if arr.ndim == 2:
                # mittel über seq_len, dann first/only position
                arr = arr.mean(axis=0)
            pid = c.get("pid", fn.replace(expected, ""))
            arm_data[pid] = arr
        out[arm] = arm_data
    return out


# ── State-Directions ─────────────────────────────────────────────────────

def build_state_directions(
    captures_dir: str,
    hi: str = "WIDE",
    lo: str = "NARROW",
    arms: Optional[List[str]] = None,
    layer: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str], float]:
    """Berechne d_width = unit(mean(hi) − mean(lo)) per-Prompt-diff.

    Logik analog seite15 build_state_directions():
      - Für jeden Prompt p: diff[p] = h[hi][p] − h[lo][p]
        (inner-Prompt-Diff hebt Content auf, lässt nur State übrig.)
      - d_width = unit(mean(diff)) — die "verstärkbar"-Richtung.
      - means[arm] = mean(h[arm][p] for p in common) — Centroids.
      - common = intersection aller pids über alle arms.
      - sep = ||mean_WIDE − mean_NARROW|| — Magnitude der Trennung.

    Args:
        captures_dir: Wurzelverzeichnis.
        hi: "high state" arm (default WIDE).
        lo: "low state" arm (default NARROW).
        arms: alle arms (default = those actually present in captures_dir).

    Returns:
        (d_width[hidden], means{arm: ndarray[hidden]}, common[pid], sep)
    """
    if arms is None:
        # auto-detect: alle arm-Unterordner
        arms = sorted(d for d in os.listdir(captures_dir)
                      if os.path.isdir(os.path.join(captures_dir, d)))
    if hi not in arms or lo not in arms:
        raise ValueError(f"hi={hi!r} or lo={lo!r} not in arms={arms}")

    # Layer wird durch load_captures_for_layer() automatisch aus dem ersten
    # .pt-File in arms[0] extrahiert — pro build_state_directions()-Aufruf ist
    # das genau ein Layer. Im cpm_layer_sweep wird build_state_directions
    # mehrfach mit verschiedenen layer-Werten aufgerufen, indem der sweeploop
    # load_captures_for_layer(layer=L) zuerst aufruft (das setzt das Layer im
    # Cache), dann build_state_directions() ohne layer-Parameter.
    raw = load_captures_for_layer(captures_dir, arms=arms, layer=layer)
    pids_per_arm = [set(raw[a].keys()) for a in arms]
    common = sorted(set.intersection(*pids_per_arm))
    if not common:
        raise ValueError(f"no common prompts across arms {arms}")

    def per_prompt_diff(h_arm: str, l_arm: str) -> np.ndarray:
        diffs = [raw[h_arm][p] - raw[l_arm][p] for p in common]
        return np.mean(diffs, axis=0).astype(np.float32)

    def unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 1e-12 else v.astype(np.float32)

    raw_width = per_prompt_diff(hi, lo)
    means = {a: np.mean([raw[a][p] for p in common], axis=0).astype(np.float32) for a in arms}
    d_width = unit(raw_width)
    sep = float(np.linalg.norm(means[hi] - means[lo]))
    return d_width, means, common, sep


# ── Artefact-Format ──────────────────────────────────────────────────────

def format_dwidth_artefact(
    *,
    model_id: str,
    hf_id: str,
    hidden_size: int,
    capture_layer: int,
    inject_layer: int,
    dwidth: np.ndarray,
    sep: float,
    common: List[str],
    direction: str,
    source: str = "scratches/relay_layer_sweep build_state_directions (LOO-diff, unit-norm)",
) -> dict:
    """Baut das JSON-kompatible Artefakt-Dict für px_manifolds/{id}_relay_dwidth.json.

    Pflichtfelder (für relay_inject.load_dwidth()):
      model_id, hf_id, hidden_size, capture_layer, inject_layer,
      direction, dwidth, sep_WIDE_NARROW_L{mean_K?}, norm, n_prompts, prompts, source.
    """
    norm = float(np.linalg.norm(dwidth))
    return {
        "model_id": model_id,
        "hf_id": hf_id,
        "hidden_size": int(hidden_size),
        "capture_layer": int(capture_layer),
        "inject_layer": int(inject_layer),
        "direction": direction,
        "source": source,
        "n_prompts": len(common),
        "prompts": list(common),
        "sep": float(sep),
        "norm": norm,
        "dwidth": dwidth.astype(np.float32).tolist(),
    }


# ── Atomic Write ─────────────────────────────────────────────────────────

def atomic_write_json(path: str, data: dict) -> None:
    """Schreibe JSON atomar: write-temp + os.replace. Falls write-temp crasht,
    bleibt die Ziel-Datei entweder unangetastet (alter Inhalt) oder nicht-existent —
    niemals halb-geschrieben."""
    target_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".dwidth_", suffix=".json.tmp", dir=target_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        # cleanup temp file on any failure
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


# ── Linear-Decoder für Layer-Discovery ───────────────────────────────────

def linear_decoder_r2(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> float:
    """L2-regularisierter Linear-Decoder mit Leave-One-Out-CV.

    Args:
        X: ndarray[n_samples, n_features] — hidden_states pro Sample.
        y: ndarray[n_samples] — binäre Labels (0 oder 1).
        l2: L2-Regularisierungsstärke (default 1.0).

    Returns:
        R² ∈ (-∞, 1]. Höher = bessere Trennung.
        LOO-CV ist pessimistisch bei kleinem N (oft negativ bei noise).

    Hintergrund: seite15/seite19 nutzt einen L2-Decoder um zu prüfen, ob
    eine Schicht tatsächlich WIDE-vs-NARROW-Zustand linear kodiert. Layer
    mit höchstem R² = informativster Layer für RELAY.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score

    n = X.shape[0]
    if n < 3:
        return 0.0  # zu wenig Samples für LOO

    loo = LeaveOneOut()
    preds = np.zeros(n, dtype=np.float32)
    for train_idx, test_idx in loo.split(X):
        model = Ridge(alpha=l2)
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])

    return float(r2_score(y, preds))
