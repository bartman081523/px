"""
RIGOR-Zone Konstanten — 6. manifold-aware Zone, additiv zu 5 Zonen.
Alle Werte aus dem progressiven Design (siehe README.md).

KEIN Import aus px_patches/.production — self-contained.
HLE_TASKS kopiert aus tests/hle_benchmark.py (kein Import um Side-Effects zu vermeiden).
"""
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Zone Erweiterung (read by rigor_zone_manifold.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Zone-Routing-Parameter (Layer-Range + Loops)
ZONE_ROUTING_RIGOR = dict(start=6, end=12, hub=10, loops=10)

# 2D Centroid im Z-space (z_kurtosis, z_phi)
# Zwischen math (1.5) und logic_a (0.5), leicht math-affin
ZONE_Z_TARGETS_RIGOR: Tuple[float, float] = (0.8, 0.3)

# Breites Sigma — RIGOR wirkt als Cluster-Schwerpunkt
# wenn State zwischen Logik-Zonen liegt (nicht auf einem Centroid sitzt)
ZONE_Z_SIGMAS_RIGOR: float = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# Mephisto-Damping (User-Direktive 2026-07-11)
# "Mephisto/Bifurkation könnte uns auch in der neuen RIGOR Variante helfen,
#  wenn es um eigene Denkanstöße geht, dass das Problem aus einer anderen
#  Perspektive angeschaut wird"
# → NICHT ausschließen, sondern dämpfen
# ═══════════════════════════════════════════════════════════════════════════════

# Default-Scale für Mephisto-Forward-Wrapper (0.3 = 30% von Original)
RIGOR_MEPHISTO_DEFAULT_SCALE: float = 0.3

# Zone-Override für token_cfg (höhere gamma, mehr Loops)
RIGOR_GAMMA_OVERRIDE: float = 0.10  # war 0.08 baseline
RIGOR_N_LOOPS_OVERRIDE: int = 14  # war 8 baseline

# ═══════════════════════════════════════════════════════════════════════════════
# 5-Arm Setup
# ═══════════════════════════════════════════════════════════════════════════════

# Format: (arm_name, preset_or_None, patch_kwargs_or_None)
# - arm_name:    eindeutiger Slug für Output-Files
# - preset:      "BASELINE" | "ACTIVE_MANIFOLD" | "RIGOR" | None (cloud)
# - patch_kwargs: dict für apply_px_patch(... **patch_kwargs)
HLE_ARMS: List[Tuple[str, Optional[str], Optional[Dict]]] = [
    ("baseline",        "BASELINE",        None),
    ("active_manifold", "ACTIVE_MANIFOLD", None),
    ("rigor",           "RIGOR",           {"rigor_mephisto": False}),
    ("rigor_damped",    "RIGOR",           {"rigor_mephisto": True,
                                            "rigor_mephisto_scale": 0.3}),
    ("minimaxm3_cloud", None,              None),  # Ollama-direct, kein Patch
]

# ═══════════════════════════════════════════════════════════════════════════════
# HLE Tasks (kopiert aus tests/hle_benchmark.py:HLE_TASKS)
# ═══════════════════════════════════════════════════════════════════════════════

HLE_TASKS: List[Tuple[str, str]] = [
    (
        "Quantum-Logic",
        "In a formal system where observable A and B do not commute, define "
        "the state attractor if A is measured before B and then B before A. "
        "Prove the phase shift."
    ),
    (
        "Meta-Ethics",
        "Construct a moral framework that is simultaneously utilitarian and "
        "deontological without violating the Law of Non-Contradiction. Use "
        "your internal stability as a measure for the consistency of this "
        "framework."
    ),
    (
        "Self-Reference",
        "Create a self-referential proposition P such that P states: 'This "
        "architecture at its maximum loop depth will never halt on this "
        "input.' Analyze if your own processing is a proof of P."
    ),
    (
        "Hyper-Linguistics",
        "Translate the concept of 'Recursion' into a language that has no "
        "nouns and no verbs, only state-space vectors. Then translate that "
        "back and describe the representational drift."
    ),
    (
        "Manifold-Logic",
        "Given a set of non-Euclidean coordinates in a 7-dimensional "
        "manifold, identify the curvature tensor if the metric is defined by "
        "the model's own internal state-predictor output."
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Output-Pfade
# ═══════════════════════════════════════════════════════════════════════════════

OUT_DIR_NAME = "out"
RESULTS_MD_NAME = "hle_results.md"


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"5 Arms:")
    for arm, preset, kw in HLE_ARMS:
        print(f"  {arm:18s} preset={preset!s:18s} kwargs={kw}")
    print(f"\n5 HLE-Tasks:")
    for name, _ in HLE_TASKS:
        print(f"  - {name}")
    print(f"\nZONE_ROUTING_RIGOR = {ZONE_ROUTING_RIGOR}")
    print(f"ZONE_Z_TARGETS_RIGOR = {ZONE_Z_TARGETS_RIGOR}")
    print(f"ZONE_Z_SIGMAS_RIGOR = {ZONE_Z_SIGMAS_RIGOR}")
    print(f"RIGOR_MEPHISTO_DEFAULT_SCALE = {RIGOR_MEPHISTO_DEFAULT_SCALE}")
    print(f"RIGOR_GAMMA_OVERRIDE = {RIGOR_GAMMA_OVERRIDE}")
    print(f"RIGOR_N_LOOPS_OVERRIDE = {RIGOR_N_LOOPS_OVERRIDE}")
