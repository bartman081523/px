"""tci_probe.py — Live-Forward-Hook: Krümmung des semantischen Raums pro Token.

Das ist der im Gespräch zugesagte Hook für die Gemma-4-Attention-/Residual-
Layer. Er zeichnet bei *jedem* generierten Token auf der GPU auf:

  * H(l)            Hidden-State der letzten Position, alle Layer (L x d)
  * H72             Shannon-Entropie der Projektion auf einen 72-dim
                    Repräsentations-Unterraum, normiert mit ln(72)  (P3)
  * H_logit         Shannon-Entropie der Token-Verteilung, normiert mit ln(V)
  * wick_curvature  mittlerer Winkel zwischen benachbarten Layern (P4)
  * layer_step      ||h_{l+1}-h_l|| / ||h_l||  — die empirische "Duty-Cycle"-
                    Größe, die mit alpha_TCI = 0.00730 verglichen wird (P1)
  * norm_ratio      ||h_t|| / ||h_{t-1}||  (P2, Informationserhaltung)

Der Hook ist zustandslos bezüglich des Modells (nur forward_hooks, keine
Gewichtsänderung) und wird über attach()/detach() strikt ab- und aufgebaut.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


# ── 72-dim Repräsentations-Unterraum ────────────────────────────────────────
class Basis72:
    """PCA-Unterraum (d -> 72) aus einem Kalibrierungssatz von Hidden-States.

    Der Unterraum ist *nicht* pro Prompt gefittet: sonst wäre die 72-dim
    Entropie eine Eigenheit des jeweiligen Textes statt eine feste Karte des
    Repräsentationsraums (Leakage).
    """

    def __init__(self, mean: torch.Tensor, basis: torch.Tensor):
        self.mean = mean.float()
        self.basis = basis.float()          # [d, D]

    @classmethod
    def fit(cls, states: List[torch.Tensor], dim: int = 72) -> "Basis72":
        X = torch.stack([s.float().flatten() for s in states], dim=0)
        mean = X.mean(dim=0)
        Xc = X - mean
        # SVD der Datenmatrix: rechte Singulärvektoren = Hauptachsen
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        d = Vh.shape[1]
        dim = min(dim, d)
        return cls(mean, Vh[:dim].T.contiguous())

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return (h.float().flatten() - self.mean) @ self.basis


def spectral_entropy_norm(z: torch.Tensor) -> float:
    p = z.float() ** 2
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p.clamp_min(1e-30)
    return float(-(p * p.log()).sum() / math.log(z.numel()))


# ── Live-Hook ───────────────────────────────────────────────────────────────
@dataclass
class StepRecord:
    step: int
    layer_states: torch.Tensor              # [L, d] float32 CPU
    hidden_norm: float


@dataclass
class TCIHolographicProbe:
    """Forward-Hook auf allen Decoder-Layern; sammelt pro Forward-Call die
    letzten Positionen. Genau ein Record pro generiertem Token (plus Prefill).
    """

    text_model: object
    records: List[StepRecord] = field(default_factory=list)
    handles: List[object] = field(default_factory=list)
    _cur: Dict[int, torch.Tensor] = field(default_factory=dict)
    _n_layers: int = 0

    def attach(self) -> "TCIHolographicProbe":
        self.detach()
        layers = self.text_model.layers
        self._n_layers = len(layers)
        self._cur = {}

        def mk(i):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                self._cur[i] = h[:, -1, :].detach().float().cpu()[0]
                if i == self._n_layers - 1:
                    mat = torch.stack([self._cur[j] for j in range(self._n_layers)], dim=0)
                    self.records.append(StepRecord(len(self.records), mat,
                                                   float(mat[-1].norm())))
            return hook

        for i, layer in enumerate(layers):
            self.handles.append(layer.register_forward_hook(mk(i)))
        return self

    def detach(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        self._cur = {}

    def clear(self) -> None:
        self.records.clear()

    # ── Auswertung ──────────────────────────────────────────────────────
    def metrics(self, basis: Optional[Basis72] = None,
                logit_entropies: Optional[List[float]] = None,
                vocab_size: int = 262144) -> Dict[str, object]:
        if not self.records:
            return {}
        per_step = []
        for rec in self.records:
            H = rec.layer_states                       # [L, d]
            L = H.shape[0]
            # Wick-Krümmung: Winkel zwischen benachbarten Layern (letzte Position)
            cos = torch.nn.functional.cosine_similarity(H[:-1], H[1:], dim=-1)
            angles = torch.acos(cos.clamp(-1.0, 1.0))
            # empirische Schrittweite pro Layer
            rel = (H[1:] - H[:-1]).norm(dim=-1) / (H[:-1].norm(dim=-1) + 1e-12)
            row = {
                "step": rec.step,
                "wick_curvature": float(angles.mean()),
                "layer_step_median": float(rel.median()),
                "layer_step_mean": float(rel.mean()),
                "hidden_norm": rec.hidden_norm,
            }
            if basis is not None:
                row["H72_norm"] = spectral_entropy_norm(basis.project(H[-1]))
            per_step.append(row)

        out = {
            "n_steps": len(per_step),
            "per_step": per_step,
        }
        if logit_entropies:
            he = np.array([x / math.log(vocab_size) for x in logit_entropies])
            out["H_logit_norm_mean"] = float(he.mean())
            out["H_logit_frac_above_phi_inv"] = float((he > 1 / ((1 + 5 ** 0.5) / 2)).mean())
        if any("H72_norm" in r for r in per_step):
            h72 = np.array([r["H72_norm"] for r in per_step])
            out["H72_norm_mean"] = float(h72.mean())
            out["H72_norm_frac_above_phi_inv"] = float((h72 > 1 / ((1 + 5 ** 0.5) / 2)).mean())
            out["H72_norm_std"] = float(h72.std())
        out["wick_curvature_mean"] = float(np.mean([r["wick_curvature"] for r in per_step]))
        out["layer_step_median"] = float(np.median([r["layer_step_median"] for r in per_step]))
        out["mean_hidden_norm"] = float(np.mean([r["hidden_norm"] for r in per_step]))

        # Profil über alle Layer (Mittel über die Tokens einer Session)
        prof = []
        for l in range(self.records[0].layer_states.shape[0] - 1):
            vals = []
            for rec in self.records:
                H = rec.layer_states
                vals.append(float((H[l + 1] - H[l]).norm() / (H[l].norm() + 1e-12)))
            prof.append(float(np.mean(vals)))
        out["layer_step_profile"] = prof
        return out


# ── Steuerung (äquivalent zum Relay-Hook, hier lokal gehalten) ──────────────
class RelaySteer:
    """h[:, -1, :] += sign * alpha * ||h|| * d_unit auf Layer `layer`.

    Identisch zur Semantik von px_patches/gemma4_2b_px/relay_inject.install_relay
    (nur ohne Modell-Registry), damit die Wide/Narrow-Arme von Eval O dieselbe
    Intervention benutzen wie Eval L.
    """

    def __init__(self, text_model, vector: torch.Tensor, layer: int = 26,
                 sign: int = 0, alpha: float = 0.0):
        self.text_model = text_model
        self.layer = layer
        self.sign = int(sign)
        self.alpha = float(alpha)
        self.d_unit = torch.nn.functional.normalize(vector.float().flatten(), dim=0)
        self.handle = None

    def __enter__(self):
        if self.sign == 0 or self.alpha == 0.0:
            return self

        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            if h.shape[1] != 1:            # Prefill verwerfen (wie relay_inject)
                return out
            d = self.d_unit.to(h.device, h.dtype)
            h[:, -1, :] = h[:, -1, :] + self.sign * self.alpha * h[:, -1, :].norm() * d
            return out

        self.handle = self.text_model.layers[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False
