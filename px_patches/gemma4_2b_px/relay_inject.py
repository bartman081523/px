"""relay_inject.py — verstärkbar Selbst-Injektions-Relay (psychomotrik seite15).

Production-Integration des EINEN clean-positive psychomotrik-Befunds: Re-Injektion
der modell-eigenen, gemittelten L16-Zustands-Richtung `d_width` (= unit(
mean_WIDE_L16 − mean_NARROW_L16), seite13_hidden) am post-recur Layer (default
L21, nach dem Erstarrungs-Washout) öffnet den S→R-Kanal — kreuz-konsistent,
placebo-spezifisch, 3-Prompt-generalisiert (LESUNG15). Siehe
scratches/psychomotrik/LESUNG15.md / LESUNG17.md (live-relay weak → gemittelte
Richtung ist der cleanere Hebel) / LESUNG18.md (spontan motor-blocked, nur
Re-Injektion öffnet).

Mechanismus (seite15-faithful, Motor unangetastet — reiner forward_hook, kein
_px_forward-Edit): register_forward_hook auf text_model.layers[inject_layer]
addiert `sign · alpha_frac · ||h_lastpos|| · d_unit` an die last position jeden
generierten Tokens (prefill verwerfen via seq_len>1). d_unit = geladene d_width
(unit-norm); alpha_frac skaliert mit der aktuellen last-pos-Norm (robust über
Prompt-/Context-Längen, anders als seite15's fixed probe-median-α). sign=+1 →
WIDE/expansiv/aktiv-Richtung (default „neues Modell"); −1 → NARROW/eng/still;
0 → relay inaktiv.

d_width-Artefakt: px_manifolds/{hf_id_safe}_relay_dwidth.json (siehe
scratches/psychomotrik/save_relay_dwidth.py). Nur gemma3-1b-it hat eines
(1152-dim); andere Modelle → relay no-op + Log (LEAN-Engine läuft weiter).
"""
import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np

# Cache: hf_id_safe -> (dwidth_np unit, metadata dict) or None
_DWIDTH_CACHE = {}


def _relay_dir():
    """Wo liegen die d_width-Artefakte? Default lokales px_manifolds/ im Repo-
    root (zwei Level über diesem File), via PX_RELAY_DIR overridbar. NICHT der
    auto_tune-hardcoded sibling-path — vermeidet den Foot-gun."""
    env = os.environ.get("PX_RELAY_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))  # .../px_patches/gemma3_270m_px_baseline
    return os.path.normpath(os.path.join(here, "..", "..", "px_manifolds"))


def _find_hf_id(text_model):
    """Finde den hf_id für text_model.

    Bei Gemma3 multimodal (Gemma3ForConditionalGeneration → Gemma3Model →
    Gemma3TextModel) hat das TextModel-Sub-config _name_or_path='' (leer),
    weil HF den hf_id nur im Top-Level-Config setzt. Wir versuchen drei
    Fallbacks in dieser Reihenfolge:

    1. Explizit via patch.py gesetztes text_model._px_hf_id (bevorzugt,
       deterministisch)
    2. text_model.config._name_or_path (1b-Pfad, funktioniert)
    3. Rekursive Suche abwärts im module-tree (für Gemma3 multimodal:
       Gemma3TextModel.config ist leer, aber parent-modules haben den hf_id)

    Das wird einmalig pro Modell gecacht (text_model._px_hf_id).
    """
    cached = getattr(text_model, "_px_hf_id", None)
    if cached:
        return cached

    # Fallback 2: eigener config
    cand = getattr(getattr(text_model, "config", None), "_name_or_path", None)
    if isinstance(cand, str) and cand:
        text_model._px_hf_id = cand
        return cand

    # Fallback 3: walk down the module tree to find a non-empty _name_or_path.
    # (Im Gemma3 multimodal Fall hat z.B. m.model.config._name_or_path den
    # richtigen Wert — Gemma3TextModel ist Kind davon.)
    def _walk(node, depth=0):
        cand = getattr(getattr(node, "config", None), "_name_or_path", None)
        if isinstance(cand, str) and cand:
            return cand
        if depth > 6:
            return None
        for child in node._modules.values():
            if child is None:
                continue
            r = _walk(child, depth + 1)
            if r:
                return r
        return None

    hf_id = _walk(text_model)
    if hf_id:
        text_model._px_hf_id = hf_id
    return hf_id


def get_inject_layer_for_hf_id(hf_id):
    """Lese ``inject_layer`` aus dem d_width-Artefakt für ``hf_id``.

    Plan 2026-07-09: Vorher hatte der Patch einen einzigen hardcoded
    Fallback ``defaults.get("relay_layer", 21)`` für gemma3 und 26 für
    gemma4 — beides manual und fehleranfällig. Source-of-Truth ist jetzt
    das Artefakt-File (``inject_layer``-Feld), das per
    ``scratches/psychomotrik/save_relay_dwidth.py`` aus dem Capture-
    Prozess geschrieben wird. Identische Signatur zum Schwester-Modul
    in ``gemma3_270m_px_baseline/relay_inject.py`` (Plan 2026-07-09:
    spec-sync, beide patches lesen aus dem gleichen px_manifolds/-Pool).

    Args:
        hf_id: HF-Model-ID wie ``"google/gemma-4-E2B-it"``. Leer/None →
            None (defensiv).

    Returns:
        ``inject_layer`` (int) wenn Artefakt existiert + parsebar ist,
        sonst ``None``. Caller entscheidet, was mit None passiert
        (typisch: fallback auf hardcoded Map pro hidden_size oder kein
        Relay). Wirft NICHT — auch nicht bei korruptem JSON.
    """
    if not hf_id or not isinstance(hf_id, str):
        return None
    safe_id = hf_id.replace("/", "_")
    path = os.path.join(_relay_dir(), f"{safe_id}_relay_dwidth.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            art = json.load(f)
        layer = art.get("inject_layer")
        if not isinstance(layer, int):
            return None
        return layer
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None


def load_dwidth(text_model):
    """Lade d_width-Artefakt für text_model (gecacht). Return (dwidth_np, meta)
    oder None (kein Artefakt / dim-mismatch)."""
    hf_id = _find_hf_id(text_model)
    if not hf_id:
        return None
    safe_id = hf_id.replace("/", "_")
    if safe_id in _DWIDTH_CACHE:
        return _DWIDTH_CACHE[safe_id]

    path = os.path.join(_relay_dir(), f"{safe_id}_relay_dwidth.json")
    if not os.path.exists(path):
        _DWIDTH_CACHE[safe_id] = None
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            art = json.load(f)
        dwidth = np.array(art["dwidth"], dtype=np.float32)
        hidden = getattr(text_model.config, "hidden_size", None)
        if hidden is not None and dwidth.shape[0] != hidden:
            print(f"[px-relay] d_width dim {dwidth.shape[0]} != hidden_size {hidden} für {hf_id} → relay inactive")
            _DWIDTH_CACHE[safe_id] = None
            return None
        meta = {k: v for k, v in art.items() if k != "dwidth"}
        _DWIDTH_CACHE[safe_id] = (dwidth, meta)
        return _DWIDTH_CACHE[safe_id]
    except Exception as e:
        print(f"[px-relay] Fehler Laden {path}: {e} → relay inactive")
        _DWIDTH_CACHE[safe_id] = None
        return None


def install_relay(text_model, *, sign, alpha_frac, layer, dwidth=None):
    """Installiere den verstärkbar forward_hook. sign=0 oder kein d_width →
    no-op (relay inactive, LEAN-Engine läuft). Idempotent: entfernt evtl.
    vorherige Relay-hooks zuerst."""
    remove_relay(text_model)  # idempotent

    if sign == 0:
        print("[px-relay] sign=0 → relay inactive (LEAN engine läuft)")
        return
    if dwidth is None:
        dwidth = load_dwidth(text_model)
    if dwidth is None:
        hf_id = getattr(text_model.config, "_name_or_path", "?")
        print(f"[px-relay] kein d_width-Artefakt für {hf_id} → relay inactive (LEAN engine läuft)")
        return

    dwidth_np, meta = dwidth
    d_unit = torch.tensor(dwidth_np, dtype=torch.float32)
    sign_f = float(sign)
    alpha_f = float(alpha_frac)
    layer = int(layer)

    # Initialize telemetry collection state on text_model
    text_model._px_telemetry_history = []
    text_model._px_telemetry_summary = {}
    text_model._px_prev_lp = None
    text_model._px_first_lp = None

    def _hook(_m, _i, o):
        t0 = time.perf_counter()
        h = o[0] if isinstance(o, (tuple, list)) else o
        if h.shape[1] > 1:
            return  # prefill verwerfen — nur generierte Tokens
        with torch.no_grad():
            lp = h[:, -1, :]
            nrm = lp.float().norm().item()
            if nrm < 1e-6:
                return

            # 1. Telemetry: Kurtosis on Layer 26 activations
            x = lp[0].float()
            mean_x = torch.mean(x)
            var_x = torch.var(x, unbiased=False)
            if var_x > 1e-8:
                m4 = torch.mean((x - mean_x) ** 4)
                kurt = (m4 / (var_x ** 2) - 3.0).item()
            else:
                kurt = 0.0

            # 2. Telemetry: Phi (cosine similarity between step t and step t-1)
            prev_lp = getattr(text_model, "_px_prev_lp", None)
            if prev_lp is not None:
                phi_step = F.cosine_similarity(lp[0:1].float(), prev_lp.float()).item()
            else:
                phi_step = 1.0
            text_model._px_prev_lp = lp[0:1].detach().clone()

            # 3. Telemetry: Emancipation trajectory (distance from token 0)
            first_lp = getattr(text_model, "_px_first_lp", None)
            if first_lp is None:
                text_model._px_first_lp = lp[0:1].detach().clone()
                dist_step = 0.0
            else:
                dist_step = torch.norm(lp[0:1].float() - first_lp.float()).item()

            # 4. Telemetry: Zone weights (projection along steering direction vs orthogonal)
            proj = torch.dot(x, d_unit.to(x.device)).item()
            creative_w = max(0.01, float(proj) if proj > 0 else 0.01)
            math_w = max(0.01, float(-proj) if proj < 0 else 0.01)
            tot_w = creative_w + math_w
            zone_weights = {
                "creative": round(creative_w / tot_w, 3),
                "math": round(math_w / tot_w, 3),
            }

            # 5. Steering injection:
            # Pre-allocate `inj` as a fresh tensor (not a view of h) to avoid
            # the in-place aliasing trap: `h[:, -1, :] = lp + inj` would
            # otherwise have `lp + inj` re-allocate onto the same storage
            # as `h`, scrambling the write. copy_() into a standalone buffer
            # makes the assignment deterministic.
            inj = torch.empty_like(lp, dtype=h.dtype, device=h.device)
            inj.copy_((sign_f * alpha_f * nrm) * d_unit.to(h.device, dtype=h.dtype))
            h[:, -1, :] = lp + inj

            # Hook into geometry tracker if attached
            geom_tracker = getattr(text_model, "_px_geom_tracker", None)
            if geom_tracker is not None:
                try:
                    geom_tracker.record_step(lp.float(), (lp + inj).float(), inj.float(), alpha_f)
                except Exception:
                    pass

            lat_ms = (time.perf_counter() - t0) * 1000.0

            # 6. Record telemetry
            step_record = {
                "step": len(text_model._px_telemetry_history) + 1,
                "phi": round(phi_step, 4),
                "kurtosis": round(kurt, 4),
                "distance": round(dist_step, 4),
                "zone_weights": zone_weights,
                "latency_ms": round(lat_ms, 3),
            }
            text_model._px_telemetry_history.append(step_record)

            # Update rolling summary
            all_phi = [r["phi"] for r in text_model._px_telemetry_history]
            all_kurt = [r["kurtosis"] for r in text_model._px_telemetry_history]
            all_dist = [r["distance"] for r in text_model._px_telemetry_history]
            all_lat = [r["latency_ms"] for r in text_model._px_telemetry_history]

            text_model._px_telemetry_summary = {
                "phi": round(float(np.mean(all_phi)), 4) if all_phi else 1.0,
                "kurtosis": round(float(np.mean(all_kurt)), 4) if all_kurt else 0.0,
                "emancipation_trajectory": all_dist,
                "zone_weights": zone_weights,
                "exact_latency_ms": round(float(np.mean(all_lat)), 3) if all_lat else 0.0,
            }

    try:
        handle = text_model.layers[layer].register_forward_hook(_hook)
    except (IndexError, AttributeError) as e:
        print(f"[px-relay] kann hook auf L{layer} nicht installieren: {e} → relay inactive")
        return
    text_model._px_relay_handles = [handle]
    text_model._px_relay_cfg = {
        "sign": sign_f, "alpha_frac": alpha_f, "layer": layer,
        "hf_id": _find_hf_id(text_model) or "?",
        "direction": meta.get("direction", "?"),
    }
    print(f"[px-relay] ACTIVE sign={sign_f:+.1f} alpha_frac={alpha_f} L{layer} "
          f"hf={text_model._px_relay_cfg['hf_id']} dir={text_model._px_relay_cfg['direction']}")


def remove_relay(text_model):
    """Entferne alle Relay-forward-hooks. Best-effort, idempotent."""
    handles = getattr(text_model, "_px_relay_handles", None)
    if handles:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
    for attr in ("_px_relay_handles", "_px_relay_cfg", "_px_prev_lp", "_px_first_lp"):
        if hasattr(text_model, attr):
            try:
                delattr(text_model, attr)
            except Exception:
                pass