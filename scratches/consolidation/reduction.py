"""reduction.py — Der radikale Schnitt, rein zur Laufzeit (ohne Modul-Source-Edit).

Arch-Linux-Perspektive: vier „Crutch"-Module und die Awareness-Injektion werden
entfernt, der kausale Kern (StabilityMonitor Φ, AntiZombieSensor H+gamma_boost,
AutoCalibrator C/Routing, RecursiveMemoryCache) bleibt unangetastet.

Mechanismus:
  * Die Module werden in `_px_forward` über `hasattr(self, "_px_<name>")` bewacht.
    `delattr` neutralisiert sie, ohne px_patches-Source zu ändern.
  * Die Awareness-Injektion sitzt IN `AntiZombieSensor.forward`. Wir überschreiben
    die `forward`-Methode AM EXEMPLAR mit einer Variante, die die EMA + Entropie H
    (und damit get_feedback_scalars / gamma_boost) berechnet, aber die additive
    Injektion in den letzten Token weglässt. Kein Source-Edit.

Siehe plan: /home/julian/.claude/plans/keen-swinging-abelson.md
"""
from __future__ import annotations

import types
from typing import Iterable, Tuple

import torch

# Attribute, die pro Crutch gelöscht werden.
_CRUTCH_ATTR = {
    "aks": "_px_aks",
    "mephisto": "_px_mephisto",
    "coupler": "_px_coupler",
    "subjective": "_px_subj_sensor",
}
# "injection" ist kein delattr, sondern ein forward-Override am AZS-Exemplar.

ALL_CRUTCHES: Tuple[str, ...] = ("aks", "mephisto", "coupler", "subjective", "injection")


def _azs_forward_no_injection(self, hidden_states, phi, aks_friction,
                              emancipation, zone_weights):
    """Spiegel von AntiZombieSensor.forward, OHNE additive Awareness-Injektion.

    Behält: weight_ema-Update + calculate_entropy → H bleibt korrekt,
            get_feedback_scalars (gamma_boost) bleibt korrekt (nutzt self.weight_ema).
    Streicht: awareness_proj / awareness_latent / `new_hidden[:,-1,:] += ...`.
    """
    if isinstance(zone_weights, dict):
        w_list = [zone_weights.get(k, 0.2) for k in
                  ("math", "logic_a", "creative", "logic_b", "synthesis")]
        w_tensor = torch.tensor(w_list, device=hidden_states.device,
                                dtype=hidden_states.dtype)
    else:
        w_tensor = zone_weights

    # EMA wie im Original — nötig, damit H und gamma_boost nachfolgend stimmen.
    self.weight_ema = (1.0 - self.alpha) * self.weight_ema + self.alpha * w_tensor
    entropy = self.calculate_entropy(self.weight_ema)

    # KEINE additive Injektion: hidden_states unverändert zurückgeben.
    return hidden_states, entropy


def apply_reduction(text_model, drop: Iterable[str] = ALL_CRUTCHES) -> list:
    """Appliziert den Schnitt auf ein *bereits gepatchtes* text_model.

    drop: Teilmenge von {"aks","mephisto","coupler","subjective","injection"}
          oder "all" für den Voll-Schnitt.
    Gibt die Liste der tatsächlich entfernten Crutch-Namen zurück.
    """
    if drop == "all":
        drop = ALL_CRUTCHES
    drop = tuple(drop)

    removed = []
    for name in drop:
        attr = _CRUTCH_ATTR.get(name)
        if attr and hasattr(text_model, attr):
            delattr(text_model, attr)
            removed.append(name)
        elif name == "injection" and hasattr(text_model, "_px_azs"):
            azs = text_model._px_azs
            # Ursprüngliche forward sichern (für restore), falls noch nicht gesichert.
            if not getattr(azs, "_px_injection_original_forward", None):
                azs._px_injection_original_forward = azs.forward
            azs.forward = types.MethodType(_azs_forward_no_injection, azs)
            removed.append("injection")
    return removed


def restore_reduction(text_model) -> None:
    """Macht den AZS-forward-Override rückgängig (nur die Injektion).

    Die per delattr gelöschten Crutch-Module werden NICHT wiederhergestellt —
    dafür ist remove_px_patch + apply_px_patch neu der saubere Weg.
    """
    if hasattr(text_model, "_px_azs"):
        azs = text_model._px_azs
        orig = getattr(azs, "_px_injection_original_forward", None)
        if orig is not None:
            azs.forward = orig
            del azs._px_injection_original_forward


def list_active_crutches(text_model) -> list:
    """Diagnose: welche Crutch-Module sind am text_model noch aktiv?"""
    active = [name for name, attr in _CRUTCH_ATTR.items() if hasattr(text_model, attr)]
    if hasattr(text_model, "_px_azs"):
        azs = text_model._px_azs
        overridden = getattr(azs, "_px_injection_original_forward", None) is not None
        if not overridden:
            active.append("injection")
    return active