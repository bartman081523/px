"""arms.py — Zustands-Induktions-Arme für emergence5.

9 Arme, die den rekurrenten Ablauf (recur-Zonen-BREITE, Zone, Perturbation)
künstlich induzieren, über saubere Monkeypatches auf den AutoCalibrator:
  - cal.get_routing_params -> {dynamic_start, dynamic_end, dynamic_hub, n_loops}
    (dynamic_start/end/hub werden respektiert; n_loops wird bei patch.py:403
     recomputed, deshalb steuern wir recur-WORK über die Zonen-BREITE, nicht
     über n_loops — siehe Plan §Konfirmierte Mechanik / BLOCKER).
  - cal.get_zone_weights -> one-hot-Dict (lowercase keys math/logic_a/logic_b/
    creative/synthesis) forciert Zone + Routing gleichzeitig.

Motor unangetastet (nur Calibrator-Rückgänge + Forward-Hooks, wie Phase X).
Originale werden auf cal._em5_orig_* gesichert und zwischen Armen restored.
"""
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches", "emergence"))
sys.path.insert(0, os.path.join(_REPO, "scratches", "consolidation"))

from model_manager import _migrate_preset  # noqa: E402
from config import MODEL_REGISTRY  # noqa: E402
from px_patches.gemma3_270m_px_baseline.patch import (  # noqa: E402
    apply_px_patch, remove_px_patch,
)
from reduction import apply_reduction  # noqa: E402
from eval.runner import _calibrator_warmup, _SCALE_WARMUP_DEFAULTS  # noqa: E402
from em_patches import _resolve_text_model  # noqa: E402


# Arme: routing = get_routing_params-Override-Dict (oder None = original);
#       zone   = get_zone_weights-Override one-hot-Dict (oder None);
#       perturb= True -> Forward-Hooks auf L[10,13,16,19] (in capture.py installiert);
#       baseline = True -> kein PX (remove_px_patch), kein lean.
ARMS = {
    "BASELINE":      dict(routing=None, zone=None, perturb=False, baseline=True),
    "RECUR_OFF":     dict(routing={"dynamic_start": 10, "dynamic_end": 10,
                                   "dynamic_hub": 10, "n_loops": 1},
                          zone=None, perturb=False, baseline=False),
    "RECUR_STD":     dict(routing=None, zone=None, perturb=False, baseline=False),
    "RECUR_NARROW":  dict(routing={"dynamic_start": 16, "dynamic_end": 18,
                                   "dynamic_hub": 17, "n_loops": 8},
                          zone=None, perturb=False, baseline=False),
    "RECUR_WIDE":    dict(routing={"dynamic_start": 4, "dynamic_end": 22,
                                   "dynamic_hub": 10, "n_loops": 8},
                          zone=None, perturb=False, baseline=False),
    "RECUR_EXTREME": dict(routing={"dynamic_start": 2, "dynamic_end": 24,
                                   "dynamic_hub": 10, "n_loops": 8},
                          zone=None, perturb=False, baseline=False),
    "ZONE_MATH":     dict(routing=None,
                          zone={"math": 1.0, "logic_a": 0.0, "logic_b": 0.0,
                                "creative": 0.0, "synthesis": 0.0},
                          perturb=False, baseline=False),
    "ZONE_CREATIVE": dict(routing=None,
                          zone={"math": 0.0, "logic_a": 0.0, "logic_b": 0.0,
                                "creative": 1.0, "synthesis": 0.0},
                          perturb=False, baseline=False),
    "PERTURB":       dict(routing=None, zone=None, perturb=True, baseline=False),
}

ARM_ORDER = ["BASELINE", "RECUR_OFF", "RECUR_STD", "RECUR_NARROW", "RECUR_WIDE",
             "RECUR_EXTREME", "ZONE_MATH", "ZONE_CREATIVE", "PERTURB"]


def setup_baseline(model):
    """Arm BASELINE: kein PX, kein recur (unpatched Gemma3 single-pass)."""
    remove_px_patch(model)
    print("[em5] BASELINE: unpatched (kein PX, kein recur)", file=sys.stderr)


def setup_lean(model, model_id):
    """Lean-Konfig EINMAL pro Modell-Ladung: LEAN-Preset + 5-Crutche-Reduktion
    + Calibrator-Warmup. Arme wechseln danach nur Override-Monkeypatches."""
    remove_px_patch(model)
    registry = MODEL_REGISTRY[model_id]
    kw = dict(registry.get("patch_kwargs", {}))
    kw["config_preset"] = _migrate_preset("ACTIVE_MANIFOLD_LEAN")
    apply_px_patch(model, **kw)
    tm0 = model.model if hasattr(model, "model") else model
    apply_reduction(tm0, drop="all")
    wcfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
    _calibrator_warmup(model, n_warmup=10,
                       kurtosis_seed=wcfg["seed"], kurtosis_jitter=wcfg["jitter"])
    print("[em5] lean + reduction + warmup (recur ON, kausaler Kern)",
          file=sys.stderr)


def _get_cal(model):
    tm = _resolve_text_model(model)
    cal = getattr(tm, "_px_calibrator", None)
    if cal is None:
        raise RuntimeError("kein _px_calibrator am text_model — lean-setup nötig")
    return tm, cal


def apply_overrides(model, arm_name):
    """Installiert die arm-spezifischen Calibrator-Overrides (saved orig)."""
    arm = ARMS[arm_name]
    if arm["baseline"]:
        return  # BASELINE hat keine Overrides
    tm, cal = _get_cal(model)
    # Originale einmal sichern
    if not hasattr(cal, "_em5_orig_routing"):
        cal._em5_orig_routing = cal.get_routing_params
    if not hasattr(cal, "_em5_orig_zone_weights"):
        cal._em5_orig_zone_weights = cal.get_zone_weights
    # Routing-Override
    if arm["routing"] is not None:
        _forced = arm["routing"]

        def _routing(*a, **k):
            return dict(_forced)
        cal.get_routing_params = _routing
    else:
        cal.get_routing_params = cal._em5_orig_routing
    # Zone-Override (one-hot) — propagiert zu classify_zone UND get_routing_params
    if arm["zone"] is not None:
        _zone = arm["zone"]

        def _zw(*a, **k):
            return dict(_zone)
        cal.get_zone_weights = _zw
    else:
        cal.get_zone_weights = cal._em5_orig_zone_weights
    print(f"[em5] arm={arm_name} overrides installiert "
          f"(routing={'forced' if arm['routing'] else 'orig'}, "
          f"zone={'forced' if arm['zone'] else 'orig'}, "
          f"perturb={arm['perturb']})", file=sys.stderr)


def clear_overrides(model):
    """Restored Calibrator-Originale (zwischen Armen)."""
    try:
        tm, cal = _get_cal(model)
    except RuntimeError:
        return
    if hasattr(cal, "_em5_orig_routing"):
        cal.get_routing_params = cal._em5_orig_routing
    if hasattr(cal, "_em5_orig_zone_weights"):
        cal.get_zone_weights = cal._em5_orig_zone_weights


if __name__ == "__main__":
    for a in ARM_ORDER:
        print(a, ARMS[a])