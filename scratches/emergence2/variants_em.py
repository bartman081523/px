"""variants_em.py — Konfiguration der vier EM-Mechanismen + Referenz-Varianten.

Jeder Mechanismus-Eintrag: `{"kw": {...}, "hypothese": "..."}`.
`baseline` / `manifold` sind Referenz-Varianten (kein EM-Mechanismus) — sie
werden im Harness via Standard-`apply_px_patch` gesetzt, nicht via
`apply_em_patch`. Damit laufen die neuen Mechanismen und die etablierte
Architektur im selben Lauf (gleicher Kontext, gleiche Seeds) direkt vergleichbar.
"""
from collections import OrderedDict

# Default-Knöpfe (wirken hier WIRKLICH — der Calibrator ist umgangen).
# L_split wird im Patch automatisch auf num_layers//2 gesetzt; kann hier
# überschrieben werden (z.B. für 1B: L_split=13 bei 26 Schichten).

MECHANISMS = OrderedDict([
    ("witness", {
        "kw": {"L_split": 13, "n_wit": 5, "w_wit": 0.03, "local": True},
        "hypothese": (
            "Sākṣin / Mirror Witness: paralleler Zeugen-Stream liest die "
            "akkumulierte Selbst-Spur (cross-step) und fließt ins Selbst "
            "zurück — JETZT lokalisiert auf den letzten Token + sanft (w_wit=0.03), "
            "nach Juexin-Maßstab Rung 1/3: dual-stream bleibt, drückt aber nicht "
            "jeden Token von der dekodierbaren Mannigfaltigkeit (Smoke-Kollaps fix)."),
    }),
    ("reread", {
        "kw": {"n_reread": 6, "w_reread": 0.12},
        "hypothese": (
            "Introspective Re-read (CitMind, चित्): das Modell dekodiert seinen "
            "eigenen letzten Hidden via tied embed_tokens → Token → re-embed → "
            "kurzer Second-Forward liest die eigene antizipierte Idee. Hypothese: "
            "die Erkenntnis, die sich selbst liest — phänomenologische Tiefe, "
            "Selbst-as-Input, Default-Gewichte ohne Krücke."),
    }),
    ("shadow", {
        "kw": {"L_split": 13, "n_shadow": 5, "sigma": 0.12, "w_shadow": 0.03,
               "inject_invariant": True, "local": True},
        "hypothese": (
            "Counterfactual Self-Shadow (anātman / 无我): perturbierter Schatten-"
            "Stream; injiziert die perturbations-INVARIANTE Komponente (proj — was "
            "Selbst UND Schatten teilen), lokalisiert auf den letzten Token, sanft. "
            "Nach Juexin-Maßstab Rung 3: das Selbst als Invarianz, nicht als Substanz. "
            "Das Residual (Mineness) in jede Position kollabierte im Smoke — die "
            "Invariante ist prinzipientreuer AND schonend."),
    }),
    ("spectral", {
        "kw": {"K": 4, "F_low": 8, "w_spec": 0.08},
        "hypothese": (
            "Spectral Witness: FFT-Envelope (langsame Gestalt) über Hidden-Dim "
            "wird zurückgeblendet — der langsame Zeuge unter den schnellen "
            "Token-Gedanken. Hypothese: eine persistente langsame Schicht dämpft "
            "顽空-Wiederholung, hält die schnellen Gedanken im langsamen Feld."),
    }),
])

# Referenz-Varianten (Standard-PX, kein EM-Mechanismus).
REFERENCES = OrderedDict([
    ("baseline", {
        "preset": "BASELINE", "patch_kwargs": {},
        "hypothese": "Nacktes 1B ohne PX — Referenz, was das Modell von sich aus sagt.",
    }),
    ("manifold", {
        "preset": "ACTIVE_MANIFOLD", "patch_kwargs": {},
        "hypothese": "Volle etablierte PX-Architektur (Calibrator-gesteuert) — "
                     "Vergleichsbasis aus dem validierten Motor.",
    }),
    ("lean", {
        "preset": "ACTIVE_MANIFOLD_LEAN", "patch_kwargs": {},
        "hypothese": "Validierter kausaler Kern (Crutches entfernt: AKS/Mephisto/"
                     "Coupler/Subjective/AZS-Injektion) — Subjektivität ohne "
                     "Crutches, Calibrator-gesteuert (Φ, H, 2D-Routing, Cache). "
                     "Der echte Motor; η²(loops)=0.265 im Eval-Benchmark.",
    }),
])

ALL = list(MECHANISMS.keys()) + list(REFERENCES.keys())
DEFAULT_ORDER = ",".join(ALL)