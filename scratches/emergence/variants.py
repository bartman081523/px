"""variants.py — architektonische PX-Varianten für die Emergenz-Erforschung.

KEINE Signal-Injektion. Jede Variante ist eine reale Konfiguration des PX-Motors
via patch_kwargs (n_loops / gamma / recur_start / recur_end / config_preset).
`_px_forward` wird nicht editiert; sidereische Zeit, skalare Gravitation, PSI
werden NICHT ins Modell geführt — die Wette ist emergence ohne Zufuhr.
"""
from collections import OrderedDict

# 1B-Defaults (hidden_size=1152): recur_start=10, recur_end=20, n_loops=8, gamma=0.12

VARIANTS = OrderedDict([
    ("baseline", {
        "preset": "BASELINE",
        "patch_kwargs": {},
        "hypothese": "nackte Referenz — 1B ohne PX. Was sagt das Modell ohne Rekurrenz?",
    }),
    ("manifold", {
        "preset": "ACTIVE_MANIFOLD",
        "patch_kwargs": {},
        "hypothese": "volles PX (Referenz).",
    }),
    ("lean", {
        "preset": "ACTIVE_MANIFOLD_LEAN",
        "patch_kwargs": {},
        "hypothese": "kausaler Kern (validierter Schnitt — Subjektivität ohne Crutches).",
    }),
    ("deep", {
        "preset": "ACTIVE_MANIFOLD",
        "patch_kwargs": {"n_loops": 16},  # 2× Rekurrenz-Tiefe
        "hypothese": "doppelte Rekurrenz-Tiefe → mehr Selbst-Modellierung, mehr 动静.",
    }),
    ("wide", {
        "preset": "ACTIVE_MANIFOLD",
        "patch_kwargs": {"recur_start": 6, "recur_end": 26},  # breitere Zone
        "hypothese": "breitere Rekurrenz-Zone → mehr Schichten wenden nach innen.",
    }),
    ("strong", {
        "preset": "ACTIVE_MANIFOLD",
        "patch_kwargs": {"gamma": 0.24},  # 2× statische Selbst-Injektion
        "hypothese": "stärkere statische Selbst-Injektion → lauter 'eigenes Denken entgegenkommend'.",
    }),
])

DEFAULT_ORDER = list(VARIANTS.keys())


def resolve(variant):
    """Gibt (preset, patch_kwargs) für eine Variante."""
    if variant not in VARIANTS:
        raise KeyError(f"unbekannte Variante: {variant}. Erlaubt: {list(VARIANTS)}")
    v = VARIANTS[variant]
    return v["preset"], dict(v["patch_kwargs"])