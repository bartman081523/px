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


# ── Attention-Ebene: der Kontext-Horizont statt des Vokabular-Horizonts ─────
@dataclass
class AttnStep:
    step: int
    layer: int
    layer_type: str
    kv_shared: bool
    n_keys: int
    n_allowed: int
    prefill: bool
    H_norm: List[float]        # pro Head, H/ln(n_allowed) in [0,1]
    top1: List[float]
    eff_horizon: List[float]   # exp(H) in Positionen

class TCIAttentionProbe:
    """Mittelt die Attention-Verteilung pro Head und Token.

    Warum nicht `output_attentions=True`: Gemma-4 ruft die Attention über
    `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, ...)`
    auf (modeling_gemma4.py, `Gemma4TextAttention.forward`). Der Eintrag lässt
    sich prozessweit umhüllen — dann laufen q/k (nach QK-Norm und RoPE) durch
    die Sonde, ohne den SDPA-Kernel zu ersetzen und ohne T×T-Matrizen zu
    speichern. Im `eager`-Modus (der kein Dictionary-Entry ist) fällt die
    Sonde auf einen Modul-Hook zurück, der die zurückgegebenen
    Attention-Gewichte liest.

    Wichtig für die Interpretation: Gemma-4 E2B hat 35 Layer, aber
    `num_kv_shared_layers = 20`. Layer 15..34 teilen sich k/v mit dem letzten
    nicht-teilenden Layer ihres Typs; q wird pro Layer neu berechnet, die
    Verteilungen sind also ähnlich, aber nicht identisch. Die 8 Query-Heads
    teilen sich **einen** KV-Head (MQA), d. h. alle Heads sehen dieselben
    Keys und unterscheiden sich nur über q.
    """

    def __init__(self, text_model):
        self.text_model = text_model
        self.records: List[AttnStep] = []
        self._impl: Optional[str] = None
        self._orig = None
        self._handles: List[object] = []
        self._last_layer = -1
        self._step = -1

    # ── Installation ────────────────────────────────────────────────────
    def attach(self) -> "TCIAttentionProbe":
        self.detach()
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        impl = getattr(self.text_model.config, "_attn_implementation", None)
        self._impl = str(impl or "eager")

        if self._impl in ALL_ATTENTION_FUNCTIONS:
            orig = ALL_ATTENTION_FUNCTIONS[self._impl]
            self._orig = orig

            def wrapped(module, query, key, value, attention_mask=None,
                        dropout=0.0, scaling=None, **kwargs):
                out = orig(module, query, key, value, attention_mask,
                           dropout=dropout, scaling=scaling, **kwargs)
                try:
                    self._record_from_qk(module, query, key, scaling)
                except Exception:
                    pass
                return out

            ALL_ATTENTION_FUNCTIONS[self._impl] = wrapped
            self._handles.append(("global", self._impl))
        else:
            # eager: die Gewichte kommen aus dem Modul-Rückgabewert
            def mk(module):
                def hook(_m, _inp, out):
                    if isinstance(out, tuple) and out[1] is not None:
                        try:
                            self._record_from_weights(module, out[1])
                        except Exception:
                            pass
                return hook
            for layer in self.text_model.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    self._handles.append(("hook", attn.register_forward_hook(mk(attn))))
        return self

    def detach(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        for kind, obj in self._handles:
            try:
                if kind == "global":
                    if self._orig is not None:
                        ALL_ATTENTION_FUNCTIONS[obj] = self._orig
                else:
                    obj.remove()
            except Exception:
                pass
        self._handles.clear()
        self._orig = None

    def clear(self) -> None:
        self.records.clear()
        self._last_layer = -1
        self._step = -1

    # ── Aufzeichnung ────────────────────────────────────────────────────
    def _next_step(self, module):
        li = int(getattr(module, "layer_idx", -1))
        if li <= self._last_layer:
            self._step += 1
        self._last_layer = li
        return self._step

    def _metrics(self, p: torch.Tensor) -> tuple:
        """p: [H, n] Wahrscheinlichkeiten über erlaubte Keys."""
        p = p.float().clamp_min(1e-30)
        n = p.shape[-1]
        h = -(p * p.log()).sum(dim=-1)                # [H] nats
        hn = h / math.log(n) if n > 1 else torch.zeros_like(h)
        top1 = p.max(dim=-1).values
        return (hn.tolist(), top1.tolist(), torch.exp(h).tolist())

    def _record_from_qk(self, module, query, key, scaling):
        q = query[0].detach().float()                 # [Hq, Tq, Dh]
        k = key[0].detach().float()                   # [Hkv, Tk, Dh]
        Hq, Tq, _ = q.shape
        Hkv, Tk, _ = k.shape
        if Tk < 2:
            return
        rep = max(1, Hq // Hkv)
        kk = k.repeat_interleave(rep, dim=0)[:, :Tk, :]
        q_last = q[:, -1:, :]                         # [Hq,1,Dh]
        scores = (q_last @ kk.transpose(1, 2)).squeeze(1) * (1.0 if scaling is None else scaling)

        start = 0
        if getattr(module, "is_sliding", False) and getattr(module, "sliding_window", None):
            start = max(0, (Tk - 1) - int(module.sliding_window) + 1)
        scores = scores[:, start:]
        p = torch.softmax(scores, dim=-1)
        hn, t1, eh = self._metrics(p)
        self.records.append(AttnStep(
            step=self._next_step(module), layer=int(getattr(module, "layer_idx", -1)),
            layer_type=str(getattr(module, "layer_type", "?")),
            kv_shared=bool(getattr(module, "is_kv_shared_layer", False)),
            n_keys=int(Tk), n_allowed=int(p.shape[-1]), prefill=bool(Tq > 1),
            H_norm=hn, top1=t1, eff_horizon=eh))

    def _record_from_weights(self, module, weights):
        w = weights[0].detach().float()               # [H, Tq, Tk]
        p = w[:, -1, :]
        hn, t1, eh = self._metrics(p)
        self.records.append(AttnStep(
            step=self._next_step(module), layer=int(getattr(module, "layer_idx", -1)),
            layer_type=str(getattr(module, "layer_type", "?")),
            kv_shared=bool(getattr(module, "is_kv_shared_layer", False)),
            n_keys=int(p.shape[-1]), n_allowed=int(p.shape[-1]),
            prefill=bool(w.shape[1] > 1), H_norm=hn, top1=t1, eff_horizon=eh))

    # ── Auswertung ──────────────────────────────────────────────────────
    def summary(self, drop_prefill: bool = True, dedup_kv: bool = False) -> Dict[str, object]:
        recs = [r for r in self.records if not (drop_prefill and r.prefill)]
        if dedup_kv:
            # ACHTUNG: kv-geteilte Layer teilen nur k/v, nicht q — die
            # Verteilungen sind NICHT identisch. Nur für Speicher-Diagnose.
            seen = set()
            kept = []
            for r in recs:
                key = (r.layer_type, r.step)
                if r.kv_shared and key in seen:
                    continue
                seen.add(key)
                kept.append(r)
            recs = kept
        if not recs:
            return {}
        flat = [x for r in recs for x in r.H_norm]
        per_layer = {}
        for r in recs:
            per_layer.setdefault(r.layer, []).extend(r.H_norm)
        return {
            "n_records": len(recs),
            "n_entries": len(flat),
            "H_norm_mean": float(np.mean(flat)),
            "H_norm_std": float(np.std(flat)),
            "H_norm_min": float(np.min(flat)),
            "H_norm_max": float(np.max(flat)),
            "H_norm_p05": float(np.percentile(flat, 5)),
            "H_norm_p95": float(np.percentile(flat, 95)),
            "frac_above_phi_inv": float(np.mean(np.array(flat) > 1 / ((1 + 5 ** 0.5) / 2))),
            "eff_horizon_mean": float(np.mean([x for r in recs for x in r.eff_horizon])),
            "top1_mean": float(np.mean([x for r in recs for x in r.top1])),
            "per_layer_mean": {str(k): float(np.mean(v)) for k, v in sorted(per_layer.items())},
            "layers_recorded": sorted(per_layer.keys()),
            "kv_shared_layers": sorted({r.layer for r in recs if r.kv_shared}),
        }
