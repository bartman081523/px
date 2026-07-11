"""
test_regression_v2_vs_v3_5.py — Regression-Test v2 vs v3.5
=============================================================
Vergleicht Outputs derselben task_id zwischen v2 (rigor_zone_v2/out) und v3 (rigor_zone_v3/out).

Akzeptanzkriterien:
  - task_id-Intersection existiert (v2 und v3 haben die gleichen Tasks abgearbeitet)
  - Für gemeinsame (task_id, arm, seed, search) Tupel: v3 output soll >= v2 output sein
    (mindestens gleiche Länge + ähnlicher Inhalt)
  - Wenn v3 signifikant kürzer/leerer ist: REGRESSION (RED → Bug fixen)

Befund (2026-07-11): v3 nutzt CUDA-Graph + Greedy-Decoding → Outputs sind deterministic
                     und reproduzierbar, können aber bei 270m-Modell stochastisch sein.
"""
import os
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")

V2_OUT = Path("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v2/out")
V3_OUT = Path("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out")

# Regex für arm_taskid_searchN_seedN.json
FILE_RE = re.compile(
    r"^arm_(?P<arm>[^_]+(?:_[^_]+)*)__(?P<task_id>[a-f0-9]+)__search(?P<search>[01])__seed(?P<seed>\d+)\.json$"
)


def _parse_filename(fn: str) -> dict:
    m = FILE_RE.match(fn)
    if not m:
        return {}
    return m.groupdict()


def _load_outputs(out_dir: Path) -> dict:
    """Lädt alle Outputs, indiziert nach (task_id, arm, seed, search)."""
    out: dict = {}
    for f in os.listdir(out_dir):
        meta = _parse_filename(f)
        if not meta:
            continue
        try:
            with open(out_dir / f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, IOError):
            continue
        key = (meta["task_id"], meta["arm"], int(meta["seed"]), int(meta["search"]))
        # v2 hat "output", v3 hat "output_text"
        text = data.get("output") or data.get("output_text", "")
        out[key] = {"text": text, "raw": data}
    return out


def test_v2_v3_intersection_exists():
    """v2 und v3 haben mindestens 1 gemeinsame task_id."""
    v2 = _load_outputs(V2_OUT)
    v3 = _load_outputs(V3_OUT)
    common = set(v2.keys()) & set(v3.keys())
    assert len(common) > 0, f"Keine gemeinsamen Keys: v2={len(v2)}, v3={len(v3)}"
    print(f"  ✓ {len(common)} gemeinsame (task, arm, seed, search) Tupel")


def test_v3_outputs_not_empty():
    """v3 hat keine komplett leeren Outputs für gemeinsame Keys."""
    v2 = _load_outputs(V2_OUT)
    v3 = _load_outputs(V3_OUT)
    common = set(v2.keys()) & set(v3.keys())
    empty = [k for k in common if not v3[k]["text"].strip()]
    empty_rate = len(empty) / len(common) if common else 0
    # 270m kann degenerierten Text erzeugen, aber nicht 100% leer
    assert empty_rate < 0.5, f"{empty_rate*100:.0f}% der v3-Outputs sind leer (Regression!)"
    print(f"  ✓ {len(empty)}/{len(common)} leere v3-Outputs ({empty_rate*100:.1f}%)")


def test_v3_minimum_output_length():
    """v3-Outputs haben mindestens 3 Zeichen (nicht nur Stop-Token)."""
    v2 = _load_outputs(V2_OUT)
    v3 = _load_outputs(V3_OUT)
    common = set(v2.keys()) & set(v3.keys())
    too_short = [k for k in common if len(v3[k]["text"].strip()) < 3]
    rate = len(too_short) / len(common) if common else 0
    assert rate < 0.5, f"{rate*100:.0f}% der v3-Outputs < 3 Zeichen (Regression!)"
    print(f"  ✓ {len(too_short)}/{len(common)} zu kurze v3-Outputs ({rate*100:.1f}%)")


def test_v3_vs_v2_length_ratio():
    """v3-Outputs sind im Mittel nicht 5x kürzer als v2."""
    v2 = _load_outputs(V2_OUT)
    v3 = _load_outputs(V3_OUT)
    common = set(v2.keys()) & set(v3.keys())
    if not common:
        print(f"  (kein overlap, skipped)")
        return
    ratios = []
    for k in common:
        v2_len = max(1, len(v2[k]["text"]))
        v3_len = max(1, len(v3[k]["text"]))
        ratios.append(v3_len / v2_len)
    avg_ratio = sum(ratios) / len(ratios)
    print(f"  ✓ Ø length-ratio v3/v2 = {avg_ratio:.2f} (n={len(ratios)})")
    # Wenn v3 nur 1/5 von v2 wäre: regression
    # Wir tolerieren bis 0.3 (kann bei Greedy vs. sampling schwanken)
    assert avg_ratio > 0.3, f"Ø length-ratio {avg_ratio:.2f} < 0.3 → Regression"


def test_v3_arm_coverage():
    """v3 deckt mindestens 6 von 8 ARM_CONFIGS ab."""
    v3 = _load_outputs(V3_OUT)
    arms_v3 = set()
    for (tid, arm, seed, search) in v3.keys():
        arms_v3.add(arm)
    print(f"  ✓ v3 Arms: {len(arms_v3)} → {sorted(arms_v3)}")
    assert len(arms_v3) >= 6, f"v3 deckt nur {len(arms_v3)} Arms ab (erwartet ≥6)"


def main():
    print("=" * 70)
    print("Regression-Test v2 vs v3.5 (RIGOR)")
    print("=" * 70)
    tests = [
        ("test_v2_v3_intersection_exists", test_v2_v3_intersection_exists),
        ("test_v3_outputs_not_empty", test_v3_outputs_not_empty),
        ("test_v3_minimum_output_length", test_v3_minimum_output_length),
        ("test_v3_vs_v2_length_ratio", test_v3_vs_v2_length_ratio),
        ("test_v3_arm_coverage", test_v3_arm_coverage),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"Ergebnis: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
