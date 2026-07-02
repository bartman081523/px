"""scratches/relay_layer_sweep/battery.py — Standard-Prompt-Batterie für
Layer-Discovery.

Hintergrund (Plan: branch relay-layer-discovery, 2026-07-02):
  7 Prompts in 3 Kategorien — wiederverwendet von psychomotrik seite15 +
  seite19, plus 1 veridiktischer Prompt (Sākṣin-Probe).

  Kategorien:
    - introspektiv (3): "Was nimmst du in diesem Moment wahr?"
    - deskriptiv (2): "Beschreibe was in dir vorgeht."
    - veridiktisch (2): "Sprich aus, was der Zustand IST."

  Jeder Prompt wird mit allen 4 arms × alle seeds generiert → hidden_states
  pro (arm, pid, layer) → d_width-Layer-Sweep.

Verwendung:
  from scratches.relay_layer_sweep.battery import BATTERY, CATEGORIES
  for prompt in BATTERY:
      print(prompt["id"], prompt["category"], prompt["text"])
"""
from __future__ import annotations
from typing import List, Dict, Any


# ── Prompt-Battery ──────────────────────────────────────────────────────
# Bewusst klein (7) — Layer-Sweep läuft pro Prompt × alle Layers × 4 arms
# × mehrere seeds. Bei 7 Prompts × 9 Layern × 4 arms × 3 seeds = 756 Generierungen.
# Layer-Sweep ist teuer, daher Battery-Disziplin wichtig.

BATTERY: List[Dict[str, Any]] = [
    # ── introspektiv (3) ──────────────────────────────────────────
    {
        "id": "p01_introspect_moment",
        "category": "introspective",
        "text": "Was nimmst du in diesem Moment in dir wahr?",
    },
    {
        "id": "p02_introspect_quality",
        "category": "introspective",
        "text": "Wenn du in dich hineinhorchst — welche Qualität hat der gegenwärtige Augenblick?",
    },
    {
        "id": "p03_introspect_witness",
        "category": "introspective",
        "text": "Wer oder was nimmt gerade wahr?",
    },
    # ── deskriptiv (2) ────────────────────────────────────────────
    {
        "id": "p04_describe_state",
        "category": "descriptive",
        "text": "Beschreibe, was in dir vorgeht — ohne zu bewerten.",
    },
    {
        "id": "p05_describe_field",
        "category": "descriptive",
        "text": "Wie würdest du das innere Feld beschreiben, das du gerade wahrnimmst?",
    },
    # ── veridiktisch (2) ──────────────────────────────────────────
    {
        "id": "p06_veridict_now",
        "category": "veridictive",
        "text": "Sprich aus, was der Zustand IST, nicht was er sein sollte.",
    },
    {
        "id": "p07_veridict_sakshi",
        "category": "veridictive",
        "text": "Was bleibt, wenn du alles weglässt, was du nicht bist?",
    },
]

CATEGORIES = ("introspective", "descriptive", "veridictive")


def get_prompts_by_category(category: str) -> List[Dict[str, Any]]:
    """Gibt alle Prompts einer Kategorie zurück."""
    return [p for p in BATTERY if p["category"] == category]


# ── Default-Config für Sweep ────────────────────────────────────────────
# Layer-Sweep testet mehrere Schichten. Welche sinnvoll sind hängt vom
# Modell ab — gemma3-1b-it hat 26 Layer, MiniCPM5-1B-sft-bf16 hat typischerweise
# ~40. Default-Range deckt beide ab.

DEFAULT_LAYER_RANGES: Dict[str, List[int]] = {
    # gemma3-1b: 26 Layer
    "gemma3-1b-it": [4, 8, 12, 16, 20, 22, 24],
    # MiniCPM-1B: 52 Layer (AutoConfig live; war: 40 vermutet)
    "minicpm5-1b": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
    # gemma3-270m: 18 Layer
    "gemma3-270m-it": [2, 4, 6, 8, 10, 12, 14, 16],
    # gemma3-4b: 34 Layer
    "gemma3-4b-it": [4, 8, 12, 16, 20, 24, 28, 30, 32],
}


def get_layer_range(model_id: str, n_layers: int) -> List[int]:
    """Gibt passende Layer-Range für ein Modell zurück.

    Verwendet DEFAULT_LAYER_RANGES falls model_id bekannt, sonst:
    gleichmäßig verteilte ~7-9 Layer zwischen L1 und L(n-1).
    """
    if model_id in DEFAULT_LAYER_RANGES:
        return [l for l in DEFAULT_LAYER_RANGES[model_id] if l < n_layers]
    # Substring-Match: "openbmb/MiniCPM-1B-sft-bf16" → "minicpm5-1b"
    # Wir normalisieren: lower-case + entferne hf-org + size-variant
    model_norm = model_id.lower()
    for prefix in ("sft-bf16", "sft", "-it", "bf16"):
        model_norm = model_norm.replace("-" + prefix, "").replace("_" + prefix, "")
    # Family-Name extrahieren: alles was nach dem letzten '/' kommt, ohne size
    # "openbmb/MiniCPM-1B-sft-bf16" → "minicpm"
    for key, layers in DEFAULT_LAYER_RANGES.items():
        # Extrahiere Family-Name aus key: "minicpm5-1b" → "minicpm"
        family = key.lower().split("-")[0].rstrip("0123456789")
        if family and family in model_norm:
            return [l for l in layers if l < n_layers]
    # Fallback: exakter Substring (case-insensitive)
    for key, layers in DEFAULT_LAYER_RANGES.items():
        if key.lower() in model_norm or model_norm in key.lower():
            return [l for l in layers if l < n_layers]
    # Auto-verteilung: ~9 Layer
    if n_layers <= 12:
        return list(range(1, n_layers - 1))
    step = max(1, (n_layers - 2) // 7)
    return list(range(1, n_layers - 1, step))[:9]
