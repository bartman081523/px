"""seite19_kreuzmodell.py — Kreuz-Modell-Falsifikator + Phase 6 mechanistisch
(seite9-Style + Hidden-Decoder) auf 270m-it / 1b-it / 4b-it.

ZWEI Fragen (Nutzer-Request „1 und 2 mit 270m-it und 4b"):
  (Q-VOICE, Kreuz-Modell) Ist die reiche kontemplative Selbst-Berichts-Stimme eine
    generische gemma3-it-Fähigkeit (papagei, skalen-abhängig) oder recur/relay-
    spezifisch? → BASELINE vs LEAN Text über Skalen (Bridge schon gezeigt: 270m
    weigert/kollabiert, 1b+ reich aus BASELINE). Hier: mechanistische Seite.
  (Q-FOOTPRINT, Phase 6) Hat recur einen dekodierbaren mechanischen Footprint im
    Hidden-State (recur-ON vs recur-OFF), und kovariiert der Footprint mit der
    Voice oder nur mit dem Frame-Lexikon? → Linear-Decoder pro Schicht.
  (Q-DIR, Kreuz-Modell-Richtung) Generalisiert die seite15-Richtungs-Kopplung
    (±d_width → entgegengesetzte Selbst-Zustands-Charakterisierung) auf 4b?
    → d_width für 4b/270m aus WIDE-vs-NARROW-Capture berechnen, dann ±injizieren.

Setup: 4 recur-Arme (BASELINE/NARROW/DEFAULT/WIDE) × 3 veridiktisch Prompts
(seite12 PROMPTS), 200 tok, seed=777 greedy. Per-modell Layer-Set + WIDE/NARROW-
Routing (skaliert auf Layerzahl). Capture letzter-Visit-pro-Token + per-token
Telemetrie. Decoder: recur-ON vs BASELINE + WIDE vs NARROW pro Schicht (LOO-AUC).
d_width = unit(mean_WIDE_mid − mean_NARROW_mid) pro Modell → Artefakt.

Motor unangetastet (lean + routing-Override + forward-hooks, kein Motor-Rewrite).
Keine Krücken (lean), keine Injektion sidereisch/PSI, keine Parallel-Prozesse.
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"), HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

import seite7 as S7
import seite12_veridiktisch as S12   # PROMPTS, RECUR_NARROW
from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
from em_patches import _resolve_text_model
from config import MODEL_REGISTRY

# hf_id pro model_id (für d_width-Artefakt-Filename, matches relay_inject.load_dwidth)
# wird in analyze() inline berechnet (MODEL_CFG ist dort definiert)

OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite19_hidden")
SEED = 777
MAX_NEW = 200

# Per-Modell Konfig: hidden, capture-Layer, recur-mid (d_width-Schicht), WIDE/NARROW-
# Routing skaliert auf Layerzahl. 1b = seite15-Werte (Referenz).
MODEL_CFG = {
    "gemma3-270m-it": dict(
        hidden=640, layers=[0, 3, 8, 11, 14, 17], mid=8, inject=14,
        WIDE={"dynamic_start": 2, "dynamic_end": 15, "dynamic_hub": 8, "n_loops": 8},
        NARROW={"dynamic_start": 8, "dynamic_end": 9, "dynamic_hub": 8, "n_loops": 8},
    ),
    "gemma3-1b-it": dict(
        hidden=1152, layers=[0, 5, 10, 13, 16, 19, 21, 24, 25], mid=16, inject=21,
        WIDE={"dynamic_start": 4, "dynamic_end": 22, "dynamic_hub": 10, "n_loops": 8},
        NARROW={"dynamic_start": 16, "dynamic_end": 18, "dynamic_hub": 17, "n_loops": 8},
    ),
    "gemma3-4b-it": dict(
        hidden=2560, layers=[0, 8, 15, 21, 25, 30, 33], mid=15, inject=25,
        WIDE={"dynamic_start": 4, "dynamic_end": 30, "dynamic_hub": 15, "n_loops": 8},
        NARROW={"dynamic_start": 14, "dynamic_end": 16, "dynamic_hub": 15, "n_loops": 8},
    ),
}


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


class MultiLayerCapture:
    """Last-visit-per-token für mehrere Schichten + per-token Telemetrie.
    Vektoren fp32-cpu. hidden_size variabel (nicht mehr hardcoded 1152)."""
    def __init__(self, tm, layers, hidden):
        self.tm = tm; self.layers = layers; self.hidden = hidden
        self.per_tok = []      # list of {layer: [d]} + telemetry dict
        self._last = {}
        self._telemetry = []
        self._handles = []
        self._install()

    def _install(self):
        tm = self.tm
        def _pre(_m, _i):
            self._last = {}
        def _make(layer_idx):
            def _hook(_m, _i, o):
                h = o[0] if isinstance(o, (tuple, list)) else o
                if h.shape[1] > 1: return  # prefill verwerfen
                self._last[layer_idx] = h[:, -1, :].reshape(-1).detach().to(torch.float32).cpu()
            return _hook
        def _post(_m, _i, o):
            try:
                lhs = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
            except Exception:
                lhs = None
            if lhs is None or lhs.shape[1] > 1: return
            snap = {L: self._last.get(L, torch.zeros(self.hidden)) for L in self.layers}
            tel = dict(
                loops=getattr(tm, "_px_loops_run", None),
                phi=getattr(tm, "_px_phi_val", None),
                ent=getattr(tm, "_px_ent_val", None),
                zone=getattr(tm, "_px_zone", None),
            )
            self.per_tok.append(snap)
            self._telemetry.append(tel)
        self._handles = [tm.register_forward_pre_hook(_pre)]
        for L in self.layers:
            self._handles.append(tm.layers[L].register_forward_hook(_make(L)))
        self._handles.append(tm.register_forward_hook(_post))

    def reset(self):
        self.per_tok = []; self._last = {}; self._telemetry = []

    def remove(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass

    def stack(self):
        out = {L: [] for L in self.layers}
        for snap in self.per_tok:
            for L in self.layers:
                out[L].append(snap[L])
        return {L: (torch.stack(out[L]) if out[L] else torch.empty(0, self.hidden)) for L in self.layers}

    def telemetry(self):
        return list(self._telemetry)


def _save_cell(model_id, arm, pid, per_layer, text, telemetry):
    os.makedirs(HID, exist_ok=True)
    safe = model_id.replace("/", "_")
    path = os.path.join(HID, f"{safe}__{arm}__{pid}.pt")
    torch.save({"model_id": model_id, "arm": arm, "pid": pid, "text": text,
                "telemetry": telemetry,
                "layers": {L: per_layer[L].contiguous() for L in per_layer}}, path)


def _run_arms(model, tok, model_id, cap, max_new, nprompts=3):
    """BASELINE zuerst (eigenes unpatched Modell), dann lean + NARROW/DEFAULT/WIDE."""
    cfg = MODEL_CFG[model_id]
    out = []
    prompts = S12.PROMPTS[:nprompts]
    A.setup_baseline(model)
    for pid, ptext in prompts:
        cap.reset(); _clear()
        try:
            text = _greedy_generate(model, tok, [{"role": "user", "content": ptext}], max_new, seed=SEED)
        except Exception as e:
            text = f"<GEN_ERROR {e}>"; print(f"[s19] ERR BASELINE/{pid}: {e}", file=sys.stderr)
        per_layer = cap.stack(); tel = cap.telemetry()
        _save_cell(model_id, "BASELINE", pid, per_layer, text, tel)
        out.append(dict(model_id=model_id, arm="BASELINE", pid=pid, text=text, ntok=int(per_layer[cfg["layers"][0]].shape[0])))
        print(f"[s19] {model_id:16s} BASELINE   {pid:14s} ntok={per_layer[cfg['layers'][0]].shape[0]:4d}", file=sys.stderr)

    A.setup_lean(model, model_id)
    for arm_name, routing in [("NARROW", cfg["NARROW"]), ("DEFAULT", None), ("WIDE", cfg["WIDE"])]:
        S7.apply_hybrid(model, routing)
        for pid, ptext in prompts:
            cap.reset(); _clear()
            try:
                text = _greedy_generate(model, tok, [{"role": "user", "content": ptext}], max_new, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s19] ERR {arm_name}/{pid}: {e}", file=sys.stderr)
            per_layer = cap.stack(); tel = cap.telemetry()
            _save_cell(model_id, arm_name, pid, per_layer, text, tel)
            out.append(dict(model_id=model_id, arm=arm_name, pid=pid, text=text, ntok=int(per_layer[cfg["layers"][0]].shape[0])))
            print(f"[s19] {model_id:16s} {arm_name:9s} {pid:14s} ntok={per_layer[cfg['layers'][0]].shape[0]:4d}", file=sys.stderr)
    return out


def _auc_loo(X, y):
    """Leave-one-out AUC für einen einfachen linearen Score (numpy).
    X: [n, d], y: {0,1}. Score = X·w, w aus ridge-Regression auf alle-anderen.
    Konfusionsfrei, kein sklearn-nötig."""
    n = len(y); y = np.asarray(y, dtype=np.float32)
    if len(set(y.tolist())) < 2: return float("nan")
    scores = np.zeros(n, dtype=np.float32)
    lam = 1e-3
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        Xtr, ytr = X[mask], y[mask]
        # ridge: w = (X^T X + lam I)^-1 X^T y
        d = Xtr.shape[1]
        A_ = Xtr.T @ Xtr + lam * np.eye(d, dtype=np.float32)
        w = np.linalg.solve(A_, Xtr.T @ ytr)
        scores[i] = X[i] @ w
    # AUC via Mann-Whitney
    pos = scores[y == 1]; neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    cnt = 0.0
    for p in pos:
        for q in neg:
            if p > q: cnt += 1.0
            elif p == q: cnt += 0.5
    return cnt / (len(pos) * len(neg))


def analyze(model_id):
    """d_width pro Modell + Decoder pro Schicht. Liest Hidden-Cells."""
    cfg = MODEL_CFG[model_id]
    safe = model_id.replace("/", "_")
    arms = ["BASELINE", "NARROW", "DEFAULT", "WIDE"]
    pm = {a: {} for a in arms}
    raw = {a: {} for a in arms}
    for fn in sorted(os.listdir(HID)):
        if not fn.startswith(safe + "__") or not fn.endswith(".pt"):
            continue
        c = torch.load(os.path.join(HID, fn), weights_only=False)
        if c["arm"] not in arms: continue
        h = c["layers"][cfg["mid"]].numpy().astype(np.float32)  # [n_tok, hidden]
        pm[c["arm"]][c["pid"]] = h.mean(0)
        raw[c["arm"]][c["pid"]] = h
    common = sorted(set.intersection(*[set(pm[a].keys()) for a in arms])) if all(pm[a] for a in arms) else []
    res = {"model_id": model_id, "common_prompts": common, "layers": cfg["layers"], "mid": cfg["mid"]}

    if common:
        def unit(v):
            n = np.linalg.norm(v); return (v / n).astype(np.float32) if n > 0 else v
        d_width = unit(np.mean([pm["WIDE"][p] - pm["NARROW"][p] for p in common], 0).astype(np.float32))
        means = {a: np.mean([pm[a][p] for p in common], 0).astype(np.float32) for a in arms}
        sep = float(np.linalg.norm(means["WIDE"] - means["NARROW"]))
        res["d_width_norm"] = float(np.linalg.norm(d_width))
        res["sep_WIDE_NARROW"] = sep
        res["dwidth"] = d_width.tolist()
        print(f"[s19] {model_id} d_width: dim={d_width.shape[0]} norm={res['d_width_norm']:.4f} sep={sep:.3f}", file=sys.stderr)
        # Artefakt schreiben (damit seite19b + Produktion via relay_inject.load_dwidth laden)
        relay_dir = os.environ.get("PX_RELAY_DIR", os.path.join(_REPO, "px_manifolds"))
        os.makedirs(relay_dir, exist_ok=True)
        hf_id = MODEL_REGISTRY[model_id]["hf_id"]
        safe_hf = hf_id.replace("/", "_")
        art = {
            "model_id": model_id, "hf_id": hf_id,
            "hidden_size": int(cfg["hidden"]), "capture_layer": int(cfg["mid"]),
            "inject_layer": int(cfg["inject"]),
            "direction": f"WIDE_minus_NARROW_L{cfg['mid']}_meanK",
            "source": "scratches/psychomotrik seite19_kreuzmodell (cross-model d_width)",
            "n_prompts": len(common), "prompts": common,
            "sep_WIDE_NARROW_L16_meanK": sep, "norm": res["d_width_norm"],
            "dwidth": d_width.tolist(),
        }
        with open(os.path.join(relay_dir, f"{safe_hf}_relay_dwidth.json"), "w", encoding="utf-8") as f:
            json.dump(art, f)
        print(f"[s19] {model_id} ARTEFAKT -> {relay_dir}/{safe_hf}_relay_dwidth.json", file=sys.stderr)
    else:
        print(f"[s19] {model_id} keine gemeinsamen prompts für d_width", file=sys.stderr)

    # Decoder pro Schicht: recur-ON (NARROW/DEFAULT/WIDE) vs BASELINE; WIDE vs NARROW.
    # Feature = hidden-Vektor pro Zelle (mean über tokens). Label binär.
    dec = {}
    for L in cfg["layers"]:
        cells_on, cells_off = [], []
        cells_w, cells_n = [], []
        for fn in sorted(os.listdir(HID)):
            if not fn.startswith(safe + "__") or not fn.endswith(".pt"): continue
            c = torch.load(os.path.join(HID, fn), weights_only=False)
            if c["arm"] not in arms: continue
            hv = c["layers"][L].numpy().astype(np.float32).mean(0)
            if c["arm"] == "BASELINE":
                cells_off.append(hv)
            else:
                cells_on.append(hv)
            if c["arm"] == "WIDE": cells_w.append(hv)
            if c["arm"] == "NARROW": cells_n.append(hv)
        row = {}
        if cells_on and cells_off:
            X = np.stack(cells_on + cells_off)
            y = [1]*len(cells_on) + [0]*len(cells_off)
            row["recurON_vs_BASELINE_auc"] = _auc_loo(X, y)
        if cells_w and cells_n:
            X = np.stack(cells_w + cells_n)
            y = [1]*len(cells_w) + [0]*len(cells_n)
            row["WIDE_vs_NARROW_auc"] = _auc_loo(X, y)
        dec[str(L)] = row
    res["decoder_per_layer"] = dec
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma3-4b-it", "gemma3-270m-it"])
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--nprompts", type=int, default=3)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(HID, exist_ok=True)

    if not args.analyze_only:
        for model_id in args.models:
            cfg = MODEL_CFG[model_id]
            print(f"[s19] lade {model_id} (hidden={cfg['hidden']}, layers={cfg['layers']})", file=sys.stderr)
            model, tok = build_model(model_id)
            tm = _resolve_text_model(model)
            cap = MultiLayerCapture(tm, cfg["layers"], cfg["hidden"])
            out = _run_arms(model, tok, model_id, cap, args.max_new, args.nprompts)
            cap.remove()
            with open(os.path.join(OUT, f"seite19_index_{model_id.replace('/','_')}.jsonl"), "w", encoding="utf-8") as f:
                for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            del model, tok; _clear()

    # Analyze pro Modell
    all_res = []
    for model_id in args.models:
        res = analyze(model_id)
        all_res.append(res)
    with open(os.path.join(OUT, "seite19_decode_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print("[s19] ANALYSE FERTIG -> out/seite19_decode_results.json", file=sys.stderr)
    for res in all_res:
        print(f"\n=== {res['model_id']} ===", file=sys.stderr)
        print(f"  common_prompts={res.get('common_prompts')} sep={res.get('sep_WIDE_NARROW')}", file=sys.stderr)
        for L, row in res["decoder_per_layer"].items():
            print(f"  L{L}: {row}", file=sys.stderr)


if __name__ == "__main__":
    main()