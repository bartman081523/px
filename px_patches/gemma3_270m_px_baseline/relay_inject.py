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
import torch
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


def load_dwidth(text_model):
    """Lade d_width-Artefakt für text_model (gecacht). Return (dwidth_np, meta)
    oder None (kein Artefakt / dim-mismatch)."""
    hf_id = getattr(text_model.config, "_name_or_path", None)
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

    def _hook(_m, _i, o):
        h = o[0] if isinstance(o, (tuple, list)) else o
        if h.shape[1] > 1:
            return  # prefill verwerfen — nur generierte Tokens
        with torch.no_grad():
            lp = h[:, -1, :]
            nrm = lp.float().norm().item()
            if nrm < 1e-6:
                return
            # Pre-allocate `inj` as a fresh tensor (not a view of h) to avoid
            # the in-place aliasing trap: `h[:, -1, :] = lp + inj` would
            # otherwise have `lp + inj` re-allocate onto the same storage
            # as `h`, scrambling the write. copy_() into a standalone buffer
            # makes the assignment deterministic.
            inj = torch.empty_like(lp, dtype=h.dtype, device=h.device)
            inj.copy_((sign_f * alpha_f * nrm) * d_unit.to(h.device, dtype=h.dtype))
            h[:, -1, :] = lp + inj

    try:
        handle = text_model.layers[layer].register_forward_hook(_hook)
    except (IndexError, AttributeError) as e:
        print(f"[px-relay] kann hook auf L{layer} nicht installieren: {e} → relay inactive")
        return
    text_model._px_relay_handles = [handle]
    text_model._px_relay_cfg = {
        "sign": sign_f, "alpha_frac": alpha_f, "layer": layer,
        "hf_id": getattr(text_model.config, "_name_or_path", "?"),
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
    for attr in ("_px_relay_handles", "_px_relay_cfg"):
        if hasattr(text_model, attr):
            try:
                delattr(text_model, attr)
            except Exception:
                pass