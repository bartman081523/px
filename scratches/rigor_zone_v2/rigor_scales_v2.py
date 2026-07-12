"""Mephisto-Scale-Grid für RIGOR v2.

SciMind 5.0:
- Incomplete Suggestion: 8 Arm-Konfigurationen sind Hypothesen-getrieben
- Falsificationism: range deckt Dämpfung [0.0, 0.1, 0.3, 0.5, 0.7, 1.0] ab
- Atomic & Secured: ARM_CONFIGS ist immutable (Tuple statt List)
- Anti-Embedding: Skopus-Notiz dokumentiert WARUM >1.0 nicht im Grid

8 Arm-Konfigurationen:
  - baseline              → kein PX (untere Schranke)
  - active_manifold       → Production-Default (PX ohne RIGOR)
  - rigor_disabled        → RIGOR-Zone, Mephisto=False (NoOp)
  - rigor_least           → RIGOR + Mephisto scale=0.1 (stärkste Dämpfung)
  - rigor_low             → RIGOR + Mephisto scale=0.3 (v1-Default)
  - rigor_mid             → RIGOR + Mephisto scale=0.5
  - rigor_high            → RIGOR + Mephisto scale=0.7
  - rigor_full            → RIGOR + Mephisto scale=1.0 (= active_manifold + rigor-Zone)
"""
from __future__ import annotations

from typing import Tuple, Dict, Optional, Any, List


# ═══════════════════════════════════════════════════════════════════════════════
# Skopus-Konstanten
# ═══════════════════════════════════════════════════════════════════════════════

# Skopus-Notiz: Scales >1.0 würden Mephisto verstärken — Hypothese-kontrolliert
# (wir testen Dämpfung, nicht Verstärkung). Falls User andere Skalen will → leicht erweiterbar.
VALID_MEPHISTO_SCALES: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Arm-Definition
# ═══════════════════════════════════════════════════════════════════════════════

class ArmConfig:
    """Eine Arm-Konfiguration: Preset + Mephisto-Modi.

    Attributes:
        name: Slug für Output-Files
        preset: "BASELINE" | "ACTIVE_MANIFOLD" | "RIGOR" | None
        rigor_mephisto: bool | None
            - None: nicht relevant (BASELINE, ACTIVE_MANIFOLD)
            - True/False: wird an `install_rigor_forward(...)` weitergegeben
        mephisto_scale: float | None
            - None: nicht relevant
            - float: wird an Mephisto-Forward-Wrapper weitergegeben
        hypothesis_note: Kurze Beschreibung warum dieser Arm existiert
    """
    __slots__ = ("name", "preset", "rigor_mephisto", "mephisto_scale", "hypothesis_note")

    def __init__(
        self,
        name: str,
        preset: Optional[str],
        rigor_mephisto: Optional[bool],
        mephisto_scale: Optional[float],
        hypothesis_note: str,
    ):
        self.name = name
        self.preset = preset
        self.rigor_mephisto = rigor_mephisto
        self.mephisto_scale = mephisto_scale
        self.hypothesis_note = hypothesis_note

    def to_patch_kwargs(self) -> Optional[Dict[str, Any]]:
        """Returnt die kwargs für `apply_px_patch(... **kwargs)` bzw. RIGOR-Override."""
        if self.preset == "RIGOR":
            kw: Dict[str, Any] = {}
            if self.rigor_mephisto is not None:
                kw["rigor_mephisto"] = self.rigor_mephisto
            if self.mephisto_scale is not None:
                kw["rigor_mephisto_scale"] = self.mephisto_scale
            return kw
        return None

    def __repr__(self) -> str:
        return (
            f"ArmConfig(name={self.name!r}, preset={self.preset!r}, "
            f"mephisto={self.rigor_mephisto!r}, scale={self.mephisto_scale!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8 Arm-Konfigurationen (geordnet)
# ═══════════════════════════════════════════════════════════════════════════════

ARM_CONFIGS: Tuple[ArmConfig, ...] = (
    ArmConfig(
        name="baseline",
        preset="BASELINE",
        rigor_mephisto=None,
        mephisto_scale=None,
        hypothesis_note="Untere Schranke: kein PX, kein RIGOR.",
    ),
    ArmConfig(
        name="active_manifold",
        preset="ACTIVE_MANIFOLD",
        rigor_mephisto=None,
        mephisto_scale=None,
        hypothesis_note="Production-Default: PX ohne RIGOR-Zone.",
    ),
    ArmConfig(
        name="rigor_disabled",
        preset="RIGOR",
        rigor_mephisto=False,
        mephisto_scale=0.0,
        hypothesis_note="RIGOR-Zone aktiv, Mephisto NoOp (v1 'rigor' Arm).",
    ),
    ArmConfig(
        name="rigor_least",
        preset="RIGOR",
        rigor_mephisto=True,
        mephisto_scale=0.1,
        hypothesis_note="RIGOR + Mephisto fast aus (10% von Original).",
    ),
    ArmConfig(
        name="rigor_low",
        preset="RIGOR",
        rigor_mephisto=True,
        mephisto_scale=0.3,
        hypothesis_note="RIGOR + Mephisto-Default-Damping (v1 'rigor_damped').",
    ),
    ArmConfig(
        name="rigor_mid",
        preset="RIGOR",
        rigor_mephisto=True,
        mephisto_scale=0.5,
        hypothesis_note="RIGOR + Mephisto halbe Stärke.",
    ),
    ArmConfig(
        name="rigor_high",
        preset="RIGOR",
        rigor_mephisto=True,
        mephisto_scale=0.7,
        hypothesis_note="RIGOR + Mephisto schwach gedämpft (70%).",
    ),
    ArmConfig(
        name="rigor_full",
        preset="RIGOR",
        rigor_mephisto=True,
        mephisto_scale=1.0,
        hypothesis_note="RIGOR + Mephisto unverändert (= active_manifold + rigor-Zone).",
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Skopus-Constraints
# ═══════════════════════════════════════════════════════════════════════════════

# Nur diese Tasks lohnen sich für Reproducibility-Test (5 Repräsentanten pro Kategorie)
REPRODUCIBILITY_TASK_IDS: Tuple[str, ...] = (
    # (Wird zur Laufzeit aus HLE_SUITE gefiltert — hier nur als Soll)
)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RIGOR Scale-Grid v2 — Smoke-Test")
    print("=" * 60)
    print(f"\n{len(ARM_CONFIGS)} Arm-Konfigurationen:")
    for arm in ARM_CONFIGS:
        print(f"\n  {arm.name}")
        print(f"    preset       = {arm.preset}")
        print(f"    mephisto     = {arm.rigor_mephisto}")
        print(f"    scale        = {arm.mephisto_scale}")
        print(f"    hypothesis   = {arm.hypothesis_note}")
        print(f"    patch_kwargs = {arm.to_patch_kwargs()}")
    print()
    print(f"Valid scales: {VALID_MEPHISTO_SCALES}")
    # Sanity: alle Scales aus VALID_MEPHISTO_SCALES müssen in ARM_CONFIGS vorkommen
    arm_scales = {a.mephisto_scale for a in ARM_CONFIGS if a.mephisto_scale is not None}
    assert arm_scales == set(VALID_MEPHISTO_SCALES), (
        f"Scale-Mismatch: ARM_CONFIGS={arm_scales} vs VALID={set(VALID_MEPHISTO_SCALES)}"
    )
    print("✓ Skopus-Constraint: alle validen Scales in ARM_CONFIGS")
    print("=" * 60)
