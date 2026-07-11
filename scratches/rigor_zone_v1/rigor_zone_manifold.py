"""
Subclass / Monkey-Patch von px_patches.gemma3_270m_px_baseline.auto_tune.AutoCalibrator
mit RIGOR als 6. Zone. KEINE Änderung am Original-File — alles passiert zur Laufzeit.

Drei additive Overrides:
  1. ZONE_ROUTING / ZONE_Z_TARGETS / ZONE_Z_SIGMAS Dict-Mutation
     → _get_kurtosis_weights (auto_tune.py:190-209) sieht das 6. Element
     → get_zone_weights (auto_tune.py:226-240) blendet 'rigor' automatisch
     → get_routing_params (auto_tune.py:246-267) gewichtet 'rigor' mit
     → classify_zone (auto_tune.py:242-244) wählt max() — RIGOR gewinnt
       wenn sein Centroid in (k, phi)-Distanz am nächsten ist
  2. _compute_scf_weights Monkey-Patch
     → 'rigor' als Maximum-Uncertainty-Key (sonst KeyError in
       get_zone_weights wenn k_weights 'rigor' enthält)
  3. load_manifold Monkey-Patch
     → Backfill 'rigor' Centroid aus ZONE_Z_TARGETS wenn Manifest nur
       5 Zonen hat (gilt für alle existierenden manifold-Files)

WICHTIG: Dieses Modul importieren VOR allen anderen Usern von auto_tune,
denn die Dicts sind global und werden mutiert. Tests in Production-Tests
NICHT betroffen, weil wir in scratches/ laufen.
"""
import os
import sys
from typing import Optional, Dict

# Sicherstellen dass das Repo-Root im Path ist
_REPO_ROOT = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from px_patches.gemma3_270m_px_baseline import auto_tune as _at
from px_patches.gemma3_270m_px_baseline.auto_tune import (
    AutoCalibrator, ZONE_ROUTING, ZONE_Z_TARGETS, ZONE_Z_SIGMAS,
    _sigmoid,  # für exakte SCF-Berechnung
)

# rigor_zone_config liegt im selben Verzeichnis — direkt importieren
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigor_zone_config import (
    ZONE_ROUTING_RIGOR, ZONE_Z_TARGETS_RIGOR, ZONE_Z_SIGMAS_RIGOR,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ZONE_* Dict-Mutation (additiv)
# ═══════════════════════════════════════════════════════════════════════════════

# Idempotent: wenn schon gemorpht, kein zweites Mal
if "rigor" not in ZONE_ROUTING:
    ZONE_ROUTING["rigor"] = dict(ZONE_ROUTING_RIGOR)
    ZONE_Z_TARGETS["rigor"] = tuple(ZONE_Z_TARGETS_RIGOR)
    ZONE_Z_SIGMAS["rigor"] = float(ZONE_Z_SIGMAS_RIGOR)
    _RIGOR_ZONE_INSTALLED = True
else:
    _RIGOR_ZONE_INSTALLED = False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _compute_scf_weights Monkey-Patch
# ═══════════════════════════════════════════════════════════════════════════════
#
# Original (auto_tune.py:211-224) returnt Dict mit 5 Keys. Wenn k_weights
# (aus _get_kurtosis_weights) 6 Keys hat, knallt es in get_zone_weights:238
# (`scf_weights[z]` für 'rigor' → KeyError).
#
# Fix: Wir wrappen die Original-Methode, fügen 'rigor' als 6. Key hinzu.
# 'rigor' = Maximum-Uncertainty (phi_signal nahe 0.5), semantisch passend
# zur RIGOR-Definition "weder eindeutig Math noch eindeutig Creative".

_orig_scf = _at.AutoCalibrator._compute_scf_weights

def _patched_scf(self, token_diversity):
    res = _orig_scf(self, token_diversity)
    if res is None:
        return None
    # 'rigor' als Maximum-Uncertainty: phi_signal * (1 - phi_signal) ohne
    # den 0.5-Faktor (sonst dominiert 'synthesis'). Eigene Wichtung.
    # Wir recomputen phi_signal aus dem _orig_scf-Aufruf — leider ist die
    # local variable nicht sichtbar. Alternative: direkt aus token_diversity.
    if self.token_diversity_mean is not None and self.token_diversity_std is not None:
        z_td = (token_diversity - self.token_diversity_mean) / (self.token_diversity_std + 1e-9)
        phi_signal = _sigmoid(-z_td)
    else:
        phi_signal = 0.5  # fallback
    res["rigor"] = phi_signal * (1.0 - phi_signal)  # max bei phi_signal=0.5
    W = sum(res.values()) + 1e-9
    return {k: v / W for k, v in res.items()}

if not getattr(_at.AutoCalibrator._compute_scf_weights, "_rigor_patched", False):
    _at.AutoCalibrator._compute_scf_weights = _patched_scf
    _at.AutoCalibrator._compute_scf_weights._rigor_patched = True
    _SCF_PATCHED = True
else:
    _SCF_PATCHED = False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. load_manifold Monkey-Patch (Backfill fehlender 'rigor' Centroid)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Original (auto_tune.py:286-308) lädt learned_centroids aus JSON. Wenn das
# Manifest 5 Zonen hat (alle existierenden 270m/1b/4b Manifieste), fehlt
# 'rigor'. _get_kurtosis_weights (Z. 202) iteriert aber
# `self.learned_centroids.items()` — d.h. ohne 'rigor' Key keine
# 'rigor' Contribution in k_weights → Backfill ist nötig.
#
# Backfill-Formel: analog zu calibrate() Z. 167-169:
#   raw_k = k_mean + zk * k_std
#   raw_p = phi_mean + zp * phi_std
# Wir nehmen die ZONE_Z_TARGETS-Werte direkt (die Werte, die in der
# Production-calibrate() ebenfalls als Anker dienen).

_orig_load_manifold = _at.AutoCalibrator.load_manifold

def _patched_load_manifold(self):
    _orig_load_manifold(self)
    # Backfill NUR wenn calibrate() nicht gelaufen ist (sonst überschreiben
    # wir echte gelernte Centroids mit synthetischen)
    if "rigor" not in self.learned_centroids and self.k_mean is not None:
        zk, zp = ZONE_Z_TARGETS_RIGOR
        # self.k_mean + zk * k_std  (gleiche Formel wie calibrate)
        self.learned_centroids["rigor"] = (
            self.k_mean + zk * self.k_std,
            self.phi_mean + zp * self.phi_std,
        )

if not getattr(_at.AutoCalibrator.load_manifold, "_rigor_patched", False):
    _at.AutoCalibrator.load_manifold = _patched_load_manifold
    _at.AutoCalibrator.load_manifold._rigor_patched = True
    _LOAD_PATCHED = True
else:
    _LOAD_PATCHED = False


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-Test
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    print("="*60)
    print("RIGOR Zone Manifold — Smoke-Test")
    print("="*60)
    print(f"_RIGOR_ZONE_INSTALLED: {_RIGOR_ZONE_INSTALLED}")
    print(f"_SCF_PATCHED: {_SCF_PATCHED}")
    print(f"_LOAD_PATCHED: {_LOAD_PATCHED}")
    print()
    print(f"ZONE_ROUTING hat {len(ZONE_ROUTING)} Zonen: {list(ZONE_ROUTING.keys())}")
    print(f"ZONE_Z_TARGETS hat {len(ZONE_Z_TARGETS)} Einträge: {list(ZONE_Z_TARGETS.keys())}")
    print(f"ZONE_Z_SIGMAS hat {len(ZONE_Z_SIGMAS)} Einträge: {list(ZONE_Z_SIGMAS.keys())}")
    print()
    print(f"ZONE_ROUTING['rigor'] = {ZONE_ROUTING['rigor']}")
    print(f"ZONE_Z_TARGETS['rigor'] = {ZONE_Z_TARGETS['rigor']}")
    print(f"ZONE_Z_SIGMAS['rigor'] = {ZONE_Z_SIGMAS['rigor']}")
    print()
    # Instanziiere AutoCalibrator für 270m — load_manifold wird rigor backfillen
    try:
        a = AutoCalibrator(
            hidden_size=640,
            model_id="google/gemma-3-270m-it",
        )
        print(f"AutoCalibrator(640) erstellt.")
        print(f"learned_centroids: {list(a.learned_centroids.keys())}")
        if "rigor" in a.learned_centroids:
            rk, rp = a.learned_centroids["rigor"]
            print(f"  'rigor' centroid: k={rk:.2f}, phi={rp:.4f}")
        # Test classify_zone
        for k, p in [(280, 0.92), (260, 0.91), (250, 0.88), (240, 0.86)]:
            zone = a.classify_zone(k, p)
            weights = a.get_zone_weights(k, p)
            print(f"  classify_zone(k={k}, phi={p}) = {zone!r:18s}  weights={dict((z, round(w, 3)) for z, w in weights.items())}")
        # Test get_routing_params
        params = a.get_routing_params(260, 0.91, hidden_size=640)
        print(f"  get_routing_params(260, 0.91, 640) = {params}")
    except Exception as e:
        print(f"  [WARN] AutoCalibrator-Test fehlgeschlagen: {e}")
        print(f"  (oft normal ohne venv_openmythos — json.load funktioniert aber)")
    print("="*60)


if __name__ == "__main__":
    _smoke_test()
