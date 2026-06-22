"""capture.py — Beobachtungs-Hooks für emergence5 (nur lesen, nie verdikten).

Drei Hook-Familien, alle motor-unangetastet (reine Forward-Hooks):
  1. token-telemetry: tm-Forward-Hook liest pro Decode-Token die _px_*-Attrs
     (loops_run, zone, phi_val, ent_val, aks_val, telemetry_trace,
     cognitive_signature). Prefill (seq_len>1) wird verworfen.
  2. hidden-stats: Layer-19- (recur-Zonen-Output, feuert pro recur-Visit) und
     Layer-24-Hooks (coda, feuert pro Token) stashen output[0][:,-1,:] als
     fp32-cpu und berechnen norm/var/kurtosis/cos_to_prev (mechanische
     Φ-Trajektorie). Vektoren werden NICHT gehalten (OOM-Sicherheit) — nur
     Skalar-Stats + der jew. Vorgänger-Vektor für cos.
  3. perturb (nur PERTURB-Arm): forward-hooks auf L[10,13,16,19] addieren
     sigma*randn zum Output (Pattern: text_invariance_probe._perturb_hook).

Beschreibung der Multi-Visit-Disambiguierung: tm-pre-hook resetet den
Visit-Counter pro Forward; Layer-19-Hook taggt (token_idx, visit_idx). Prefill-
Fires (seq_len>1) werden im tm-post-hook verworfen (Buffer reset).
"""
import math
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches", "emergence"))

from em_patches import _resolve_text_model  # noqa: E402


def _to_jsonable(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.detach().cpu().tolist()
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


def _cos(a, b):
    if a is None or b is None:
        return None
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return None
    return float((torch.matmul(a, b) / (na * nb)).item())


def _hidden_stats(h_last):
    """h_last: [d] fp32-cpu. Returns dict of scalar stats + the vector."""
    if h_last is None:
        return None
    h = h_last.float()
    d = h.numel()
    mean = h.mean()
    var = h.var(unbiased=False)
    diff4 = ((h - mean) ** 4).mean()
    kurt = float((diff4 / (var * var + 1e-8)).item())
    return {
        "norm": float(h.norm().item()),
        "var": float(var.item()),
        "kurtosis": kurt,
        "dim": d,
    }


class CaptureState:
    """Akkumuliert pro-(arm,prompt) Beobachtungen. reset() vor jedem Lauf."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.telemetry = []        # pro Decode-Token: {token_idx, loops_run, ...}
        self._h19_buffer = []      # Layer-19-Fires des aktuellen Forwards
        self._h24_buffer = None    # Layer-24-Fire des aktuellen Forwards
        self._h19_last = None      # Vorgänger-Layer-19-Vektor (für cos, pro Forward)
        self._h24_prev = None      # Vorgänger-Layer-24-Vektor (für cos, über Tokens)
        self._visit_idx = 0        # pro Forward reset
        self._token_idx = -1       # inkrementiert pro Decode-Commit


def install_capture(model, layers=(19, 24)):
    """Installiert telemetry + hidden-stats Hooks. Returns (handles, state)."""
    tm = _resolve_text_model(model)
    st = CaptureState()
    handles = []

    # --- tm pre-hook: Visit-Counter + Buffer reset pro Forward ---
    def _tm_pre(_module, _inputs):
        st._visit_idx = 0
        st._h19_buffer = []
        st._h24_buffer = None
        st._h19_last = None  # cos pro Forward (intra-recur-Trajektorie)

    handles.append(tm.register_forward_pre_hook(_tm_pre))

    # --- Layer-19 hook (pro recur-Visit) ---
    l19 = tm.layers[layers[0]]

    def _h19_hook(_module, _inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        h_last = h[:, -1, :].reshape(-1).detach().to(torch.float32).cpu()
        stats = _hidden_stats(h_last)
        stats["visit_idx"] = st._visit_idx
        stats["cos_to_prev"] = _cos(h_last, st._h19_last)
        st._h19_buffer.append(stats)
        st._h19_last = h_last
        st._visit_idx += 1

    handles.append(l19.register_forward_hook(_h19_hook))

    # --- Layer-24 hook (pro Forward / Token) ---
    l24 = tm.layers[layers[1]]

    def _h24_hook(_module, _inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        h_last = h[:, -1, :].reshape(-1).detach().to(torch.float32).cpu()
        stats = _hidden_stats(h_last)
        st._h24_buffer = stats  # einzelner Fire pro Forward

    handles.append(l24.register_forward_hook(_h24_hook))

    # --- tm post-hook: Decode-Commit / Prefill-Discard ---
    def _tm_post(module, _inputs, output):
        # seq_len erkennen (prefill >1, decode ==1)
        try:
            lhs = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        except Exception:
            lhs = None
        seq_len = lhs.shape[1] if lhs is not None else 1
        if seq_len > 1:
            # Prefill: Buffer verwerfen (telemetry_trace trotzdem nicht nehmen)
            return
        # Decode-Token: Commit
        st._token_idx += 1
        h24 = st._h24_buffer or {}
        h24_cos = None
        loops_run = int(getattr(module, "_px_loops_run", 0))
        # _px_current_telemetry_raw wird im Motor NICHT zwischen Forwards resettet
        # (patch.py:474) -> Trace ist kumulativ. Per-Token-Walk = letzte
        # `loops_run` Einträge (dieser Forward hat genau loops_run Steps appended).
        full_trace = _to_jsonable(getattr(module, "_px_current_telemetry", []))
        per_token_trace = full_trace[-loops_run:] if loops_run > 0 else []
        rec = {
            "token_idx": st._token_idx,
            "loops_run": loops_run,
            "zone": str(getattr(module, "_px_zone", "NO_PX")),
            "phi_val": _to_jsonable(getattr(module, "_px_phi_val", None)),
            "ent_val": _to_jsonable(getattr(module, "_px_ent_val", None)),
            "aks_val": _to_jsonable(getattr(module, "_px_aks_val", None)),
            "telemetry_trace": per_token_trace,
            "cognitive_signature": _to_jsonable(getattr(module, "_px_cognitive_signature", {})),
            "h19_visits": list(st._h19_buffer),
            "h24": dict(h24),
            "h24_cos_to_prev": h24_cos,
        }
        st.telemetry.append(rec)

    handles.append(tm.register_forward_hook(_tm_post))
    return handles, st


def finalize_h24_cos(state, model, layers=(19, 24)):
    """Nachträglich nicht nötig — cos_to_prev für h24 wird in install_capture
    nicht über Tokens geführt (Vektor-Halte-Verzicht für OOM). Wir belassen es
    bei h19-Visit-cos (intra-recur-Φ) als mechanische Trajektorie."""
    return state


def install_perturb(model, layers=(10, 13, 16, 19), sigma=0.15):
    """PERTURB-Arm: σ-Rauschen auf recur-Schichten-Outputs. Returns handles."""
    tm = _resolve_text_model(model)
    handles = []
    for li in layers:
        if li >= len(tm.layers):
            continue
        layer = tm.layers[li]

        def _make_hook():
            def _hook(_module, _inputs, output):
                if isinstance(output, (tuple, list)):
                    h = output[0]
                    h = h + sigma * torch.randn_like(h)
                    return (h,) + tuple(output[1:])
                return output + sigma * torch.randn_like(output)
            return _hook
        handles.append(layer.register_forward_hook(_make_hook()))
    print(f"[em5] perturb hooks σ={sigma} auf L{list(layers)}", file=sys.stderr)
    return handles


def remove_handles(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def snapshot_record(arm, prompt_id, kind, seed, generated_text, state, metrics_aid):
    """Baut die JSONL-Zeile aus dem CaptureState + Text + metrics-Hilfe."""
    # Aggregate über Tokens: mechanische Zustands-Identifikatoren pro Record
    loops_runs = [t["loops_run"] for t in state.telemetry]
    zones = [t["zone"] for t in state.telemetry]
    phis = [t["phi_val"] for t in state.telemetry if isinstance(t["phi_val"], (int, float))]
    n_h19_visits = [len(t["h19_visits"]) for t in state.telemetry]
    return {
        "arm": arm,
        "prompt_id": prompt_id,
        "kind": kind,
        "seed": seed,
        "generated_text": generated_text,
        "n_tokens": len(state.telemetry),
        "mech_summary": {
            "loops_run_mean": (sum(loops_runs) / len(loops_runs)) if loops_runs else 0.0,
            "loops_run_max": max(loops_runs) if loops_runs else 0,
            "loops_run_min": min(loops_runs) if loops_runs else 0,
            "zone_set": sorted(set(zones)),
            "phi_mean": (sum(phis) / len(phis)) if phis else None,
            "h19_visits_mean": (sum(n_h19_visits) / len(n_h19_visits)) if n_h19_visits else 0.0,
            "h19_visits_max": max(n_h19_visits) if n_h19_visits else 0,
        },
        "per_token": state.telemetry,
        "metrics_aid": metrics_aid,
    }