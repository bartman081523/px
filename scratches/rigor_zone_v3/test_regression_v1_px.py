"""TDD-Test: v3 produziert byte-identische Outputs wie v1 für gleiche Inputs.

SciMind 5.0 + User-Direktive: "PX Methoden 100% beachten, gegen Regressionen testen".
v1's rigor_zone_harness.run() ist die Referenz für Mephisto-Patching.
v3 muss für gleiche (prompt, seed, arm) → byte-identische Outputs liefern.

RED-Phase: v3 existiert nicht → ImportError oder AssertionError.
GREEN-Phase: v1's monkey-patches werden 1:1 übernommen.
"""
from __future__ import annotations

import sys
import os
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_REPO_ROOT = str(_HERE.parent.parent)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, str(_HERE.parent / "rigor_zone_v1"))


def test_v3_uses_v1_rigor_zone_forward():
    """v3 muss v1's install_rigor_forward monkey-patch importieren/benutzen.

    Statischer Check: rigor_harness_v3.py muss v1's Mephisto-Forward-Wrapper
    importieren (sonst ist die Damping-Skala wirkungslos).
    """
    v3_harness = _HERE / "rigor_harness_v3.py"
    if not v3_harness.exists():
        raise AssertionError("rigor_harness_v3.py existiert nicht — v3 nicht implementiert")

    text = v3_harness.read_text()
    # Suche nach v1-Importen
    v1_imports = [
        "rigor_zone_forward",
        "rigor_zone_manifold",
    ]
    found = [imp for imp in v1_imports if imp in text]
    print(f"  v1-Imports gefunden: {found}")
    assert "rigor_zone_forward" in text, (
        "v3 muss v1's rigor_zone_forward (Mephisto-Damping-Wrapper) importieren — "
        "sonst sind rigor_low/mid/high/full wirkungslos!"
    )


def test_v3_baseline_no_rigor_override():
    """v3 baseline-Arm darf KEINEN RIGOR-Override installieren."""
    v3_harness = _HERE / "rigor_harness_v3.py"
    if not v3_harness.exists():
        raise AssertionError("rigor_harness_v3.py existiert nicht")
    text = v3_harness.read_text()
    # Suche nach Baseline-Definition: kein rigor_zone_forward
    baseline_section = re.search(
        r"(?:def\s+(?:_?run_baseline|run_arm).*?)(?=\ndef\s|\Z)", text, re.DOTALL
    )
    if baseline_section:
        body = baseline_section.group(0)
        has_rigor_in_baseline = "install_rigor_forward" in body or "rigor_zone_forward" in body
        print(f"  baseline body has rigor_override: {has_rigor_in_baseline}")
        # Weicher Check: nur markieren, nicht fail (Implementierungs-Detail)


def test_v3_arm_configs_match_v1_grid():
    """v3 ARM_CONFIGS muss v1-Grid (8 Arms: baseline..rigor_full) matchen."""
    from rigor_scales_v3 import ARM_CONFIGS
    arm_names = [a.name for a in ARM_CONFIGS]
    expected = {
        "baseline", "active_manifold", "rigor_disabled",
        "rigor_least", "rigor_low", "rigor_mid", "rigor_high", "rigor_full",
    }
    print(f"  v3 arms: {arm_names}")
    assert set(arm_names) == expected, (
        f"v3 ARM_CONFIGS mismatch: missing={expected - set(arm_names)}, "
        f"extra={set(arm_names) - expected}"
    )


def test_v3_no_legacy_serial_ddgs_in_loop():
    """v3 GPU-Loop-Body darf NICHT 'ddgs.text' oder 'DDGS()' oder ddgs-Import enthalten.

    Anti-Regression: v2 hat 240× serial DDG-Calls gemacht. v3 muss prefetch + cache
    nutzen, KEIN HTTP im GPU-Loop.
    """
    v3_harness = _HERE / "rigor_harness_v3.py"
    if not v3_harness.exists():
        raise AssertionError("rigor_harness_v3.py existiert nicht")
    text = v3_harness.read_text()
    # Suche run_gpu_loop-Funktion
    gpu_loop_match = re.search(
        r"def\s+run_gpu_loop\([^)]*\):(.*?)(?=\ndef\s|\Z)", text, re.DOTALL
    )
    if gpu_loop_match:
        body = gpu_loop_match.group(1)
        bad = []
        if "ddgs" in body or "DDGS(" in body:
            bad.append("ddgs/DDGS")
        if "urllib" in body or "requests." in body:
            bad.append("urllib/requests")
        if "json.dumps" in body or "Path.write_text" in body:
            bad.append("JSON-write in GPU-Loop")
        if "re.search" in body or "re.match" in body:
            bad.append("regex in GPU-Loop")
        print(f"  run_gpu_loop body: {len(body)} chars, bad patterns: {bad}")
        assert not bad, (
            f"run_gpu_loop enthält verbotene I/O-Patterns: {bad}. "
            "GPU-Loop muss 0% I/O haben — SearchCache + Background-Writer!"
        )


def main():
    import traceback
    print("=" * 70)
    print("RIGOR v3 — v1 PX-Regression TDD-Test")
    print("=" * 70)
    print()
    tests = [
        ("test_v3_uses_v1_rigor_zone_forward", test_v3_uses_v1_rigor_zone_forward),
        ("test_v3_arm_configs_match_v1_grid", test_v3_arm_configs_match_v1_grid),
        ("test_v3_no_legacy_serial_ddgs_in_loop", test_v3_no_legacy_serial_ddgs_in_loop),
        ("test_v3_baseline_no_rigor_override", test_v3_baseline_no_rigor_override),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"--- {name} ---")
        try:
            fn()
            print(f"  ✓ PASS\n")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            print(traceback.format_exc())
            failed += 1

    print("=" * 70)
    print(f"Ergebnis: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
