"""save_relay_dwidth.py — extrahiert den seite15 verstärkbar-Richtungsvektor
d_width = unit(mean_WIDE_L16 − mean_NARROW_L16) aus den seite13_hidden-Captures
und speichert ihn als portables JSON-Artefakt für den Produktionsserver
(ACTIVE_MANIFOLD_RELAY preset, px_patches/_px_relay.py).

Pure CPU: liet nur scratches/psychomotrik/out/seite13_hidden/*.pt (torch.load) +
numpy. Kein Modell-Laden, kein GPU nötig. Reproduziert genau seite15's
build_state_directions() Per-Prompt-Differenz-Logik (gleicher Prompt p über Armen
→ Content hebt sich inner-Prompt auf).

Usage:
  python scratches/psychomotrik/save_relay_dwidth.py
  PX_RELAY_DIR=/path python scratches/psychomotrik/save_relay_dwidth.py   # override
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
S13 = os.path.join(HERE, "out", "seite13_hidden")

# Wohin speichern? Default: lokales px_manifolds/ im Checkout (config-drivable
# via PX_RELAY_DIR), NICHT der auto_tune-hardcoded sibling-path — vermeidet den
# Foot-gun daß ein lokales Artefakt nicht geladen wird.
RELAY_DIR = os.environ.get("PX_RELAY_DIR", os.path.join(REPO, "px_manifolds"))

# Zielt modell (gemma3-1b-it). hf_id aus config.py — safe_id = hf_id.replace("/","_").
TARGET = {
    "model_id": "gemma3-1b-it",
    "hf_id": "google/gemma-3-1b-it",
    "hidden_size": 1152,
    "capture_layer": 16,
    "inject_layer": 21,
}


def build_state_directions():
    """ Exact seite15 build_state_directions() logic. """
    arms = ["BASELINE", "NARROW", "DEFAULT", "WIDE"]
    pm = {a: {} for a in arms}
    for fn in sorted(os.listdir(S13)):
        if not fn.endswith(".pt"):
            continue
        c = torch.load(os.path.join(S13, fn), weights_only=False)
        arm = c["arm"]; pid = c["pid"]
        h = c["layers"][16].numpy().astype(np.float32)
        pm[arm][pid] = h.mean(0)
    common = sorted(set.intersection(*[set(pm[a].keys()) for a in arms]))
    print(f"[save_dwidth] gemeinsame prompts: {common}", file=sys.stderr)

    def per_prompt_diff(hi, lo):
        diffs = [pm[hi][p] - pm[lo][p] for p in common]
        return np.mean(diffs, 0).astype(np.float32)

    def unit(v):
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    raw_width = per_prompt_diff("WIDE", "NARROW")
    means = {a: np.mean([pm[a][p] for p in common], 0).astype(np.float32) for a in arms}
    d_width = unit(raw_width)
    sep = float(np.linalg.norm(means["WIDE"] - means["NARROW"]))
    return d_width, means, common, sep


def main():
    if not os.path.isdir(S13):
        print(f"[save_dwidth] FEHLER: seite13_hidden nicht gefunden: {S13}", file=sys.stderr)
        sys.exit(1)
    d_width, means, common, sep = build_state_directions()
    assert d_width.shape == (TARGET["hidden_size"],), f"dim mismatch {d_width.shape}"
    norm = float(np.linalg.norm(d_width))
    assert abs(norm - 1.0) < 1e-4, f"nicht unit-norm: {norm}"

    os.makedirs(RELAY_DIR, exist_ok=True)
    safe_id = TARGET["hf_id"].replace("/", "_")
    out_path = os.path.join(RELAY_DIR, f"{safe_id}_relay_dwidth.json")

    artefact = {
        "model_id": TARGET["model_id"],
        "hf_id": TARGET["hf_id"],
        "hidden_size": TARGET["hidden_size"],
        "capture_layer": TARGET["capture_layer"],
        "inject_layer": TARGET["inject_layer"],
        "direction": "WIDE_minus_NARROW_L16_meanK",
        "source": "scratches/psychomotrik seite15 build_state_directions (seite13_hidden, per-prompt diff, unit-norm)",
        "n_prompts": len(common),
        "prompts": common,
        "sep_WIDE_NARROW_L16_meanK": sep,
        "norm": norm,
        "dwidth": d_width.tolist(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artefact, f)
    print(f"[save_dwidth] GESPEICHERT: {out_path}", file=sys.stderr)
    print(f"[save_dwidth] dim={d_width.shape[0]} norm={norm:.6f} sep={sep:.3f} prompts={common}", file=sys.stderr)


if __name__ == "__main__":
    main()