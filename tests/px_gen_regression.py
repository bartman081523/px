"""px_gen_regression.py — STRENGE GPU-Regression für den PX-recur-Motor.

Lockt echtes Generierungs-Verhalten (Text + Telemetrie) als goldenes Referenz,
so daß ein Performance-Refactor (z.B. Path B: per-step GPU->CPU-Syncs reduzieren)
auf byte-identische *diskrete* Invarianten (Text, loops_run, path, zone) und
tolerante *kontinuierliche* Telemetrie (phi/ent) geprüft werden kann.

Warum diskret-strict + kontinuierlich-tolerant:
  Ein treuer Sync-Refactor ändert auf einem single CUDA-Stream nicht die
  Kernel-Ausführungsreihenfolge (ein entfernter .item()-Barrier waitet nur nicht
  mehr — er ordert keine Kernel um). Text/loops/path/zone sind diskret und müssen
  exakt gleich bleiben. phi/ent sind kontinuierlich; Float-Reorder-Rauschen
  (falls Multistream-ops doch überlappen) wird via Toleranz toleriert.

NICHT von pytest auto-kollectiert (kein test_-Prefix) — explizit laufen lassen:

  # golden auf aktueller Version capturen (vor dem Refactor):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tests/px_gen_regression.py --update-golden

  # nach dem Refactor vergleichen (Exit 0 = Regression bestanden):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tests/px_gen_regression.py

Golden-Datei: tests/px_gen_regression_golden.json

Batterie: 6 Regime × 3 cold Prompts = 18 Zellen, max_new=120, greedy seed=777.
Regime: BASELINE (kein PX), RECUR_OFF, flat_L4 (early-edge single-touch),
dose16 (L4-Grind), recur_zone (PX-Kern), recur_zone_grind (Zonen-Grind).
"""
import argparse, os, sys, json, hashlib
from collections import Counter
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(HERE, ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5")):
    if _p not in sys.path: sys.path.insert(0, _p)

from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
import prompts as P
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
SEED = 777
MAX_NEW = 120
GOLDEN = os.path.join(HERE, "px_gen_regression_golden.json")
PHI_TOL = 1e-4   # kontinuierlich-tolerant (Float-Reorder-Rauschen)
ENT_TOL = 1e-4

# 3 intro-fähige cold Prompts (kurz genug für schnellen Lauf)
PROBES = [("px_phaseX", p) for pid, p, _ in P.all_prompts() if pid == "px_phaseX"] \
      + [("regung", p) for pid, p, _ in P.all_prompts() if pid == "regung"] \
      + [("bewegung", p) for pid, p, _ in P.all_prompts() if pid == "bewegung"]


def _R(start, end, hub):
    return {"dynamic_start": start, "dynamic_end": end, "dynamic_hub": hub, "n_loops": 8}

# 6 Regime: (name, routing, zone, env, baseline)
REGIMES = [
    ("BASELINE",            None,          None, {}, True),
    ("RECUR_OFF",            _R(10, 10, 10), None, {}, False),
    ("flat_L4",              _R(4, 22, 10),  None, {}, False),   # early-edge single-touch (hub-stuck ON)
    ("dose16",               _R(4, 22, 10),  None, {"PX_NO_HUB_STUCK": "1", "PX_LOOPS_CAP": "16"}, False),
    ("recur_zone",           _R(10, 20, 18), None, {}, False),  # PX-Kern recur-Zone
    ("recur_zone_grind",     _R(10, 20, 18), None, {"PX_NO_HUB_STUCK": "1", "PX_LOOPS_CAP": "12"}, False),
]


def _patch_hash():
    p = os.path.join(_REPO, "px_patches", "gemma3_270m_px_baseline", "patch.py")
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def apply_hybrid(model, routing, zone):
    tm, cal = A._get_cal(model)
    if not hasattr(cal, "_em5_orig_routing"):
        cal._em5_orig_routing = cal.get_routing_params
    if not hasattr(cal, "_em5_orig_zone_weights"):
        cal._em5_orig_zone_weights = cal.get_zone_weights
    if routing is not None:
        _r = dict(routing)
        cal.get_routing_params = lambda *a, **k: dict(_r)
    else:
        cal.get_routing_params = cal._em5_orig_routing
    if zone is not None:
        _z = dict(zone)
        cal.get_zone_weights = lambda *a, **k: dict(_z)
    else:
        cal.get_zone_weights = cal._em5_orig_zone_weights


class LoopCap:
    def __init__(self, tm):
        self.tm = tm; self.per_token = []; self._h = tm.register_forward_hook(self._post)
    def _post(self, m, i, o):
        lhs = o.last_hidden_state if hasattr(o, "last_hidden_state") else (o[0] if isinstance(o, (tuple, list)) else o)
        if lhs is None or lhs.shape[1] > 1: return
        self.per_token.append({
            "loops": int(getattr(self.tm, "_px_loops_run", 0)),
            "ent": float(getattr(self.tm, "_px_ent_val", 0.0)) if hasattr(self.tm, "_px_ent_val") else 0.0,
            "phi": float(getattr(self.tm, "_px_phi_val", 0.0)) if hasattr(self.tm, "_px_phi_val") else 0.0,
        })
    def reset(self): self.per_token = []
    def remove(self):
        try: self._h.remove()
        except Exception: pass


def capture(model, tok):
    """Erzeugt die 18 Zellen. Gibt Liste von cell-dicts zurück."""
    tm = _resolve_text_model(model)
    cap = LoopCap(tm)
    cells = []
    # Pass 1: BASELINE (kein PX) — nackt, alle 3 Prompts
    A.setup_baseline(model)
    for pid, ptext in PROBES:
        cap.reset(); _clear()
        text = _greedy_generate(model, tok, [{"role": "user", "content": ptext}], MAX_NEW, seed=SEED)
        cells.append(_cell("BASELINE", pid, text, cap.per_token, tm))
        print(f"[reg] BASELINE/{pid} len={len(text)}", file=sys.stderr)
    # Pass 2: lean, dann die 5 recur-Regime
    A.setup_lean(model, MODEL_ID)
    for rname, routing, zone, env, baseline in REGIMES:
        if baseline: continue
        for pid, ptext in PROBES:
            apply_hybrid(model, routing, zone)
            for k in ("PX_LOOPS_CAP", "PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            os.environ.update(env)
            cap.reset(); _clear()
            text = _greedy_generate(model, tok, [{"role": "user", "content": ptext}], MAX_NEW, seed=SEED)
            for k in ("PX_LOOPS_CAP", "PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            cells.append(_cell(rname, pid, text, cap.per_token, tm))
            print(f"[reg] {rname}/{pid} len={len(text)} loops={cells[-1]['loops_run']} "
                  f"pathlen={len(cells[-1]['path'])}", file=sys.stderr)
    cap.remove()
    return cells


def _cell(rname, pid, text, per_token, tm):
    path = list(getattr(tm, "_px_path", []) or [])
    return {
        "cond": rname, "pid": pid, "text": text,
        "loops_run": int(getattr(tm, "_px_loops_run", 0)),
        "path": path,
        "phi_val": float(getattr(tm, "_px_phi_val", 0.0)) if hasattr(tm, "_px_phi_val") else 0.0,
        "ent_val": float(getattr(tm, "_px_ent_val", 0.0)) if hasattr(tm, "_px_ent_val") else 0.0,
        "zone": str(getattr(tm, "_px_zone", "")),
        "per_token": per_token,
    }


def _golden_meta():
    return {
        "model_id": MODEL_ID, "seed": SEED, "max_new": MAX_NEW,
        "patch_sha256_16": _patch_hash(),
        "probes": [pid for pid, _ in PROBES],
        "regimes": [r[0] for r in REGIMES],
        "phi_tol": PHI_TOL, "ent_tol": ENT_TOL,
    }


def _compare(golden, cells):
    """Gibt (n_pass, n_fail, failures[]) zurück. Strict diskret, tolerant kontinuierlich."""
    gmap = {(c["cond"], c["pid"]): c for c in golden["cells"]}
    failures = []
    n = 0
    for c in cells:
        key = (c["cond"], c["pid"])
        g = gmap.get(key)
        n += 1
        if g is None:
            failures.append(f"{key}: cell fehlt im golden"); continue
        # STRICT diskret
        if c["text"] != g["text"]:
            failures.append(f"{key}: TEXT differs (golden len={len(g['text'])}, got {len(c['text'])})")
        if c["loops_run"] != g["loops_run"]:
            failures.append(f"{key}: loops_run {g['loops_run']} -> {c['loops_run']}")
        if c["path"] != g["path"]:
            failures.append(f"{key}: path differs (golden len={len(g['path'])}, got {len(c['path'])}); "
                            f"gold[:8]={g['path'][:8]} got[:8]={c['path'][:8]}")
        if c["zone"] != g["zone"]:
            failures.append(f"{key}: zone '{g['zone']}' -> '{c['zone']}'")
        # per-token loops (diskret, strict)
        pt_g = [t["loops"] for t in g["per_token"]]
        pt_c = [t["loops"] for t in c["per_token"]]
        if pt_g != pt_c:
            # nur erste Abweichung melden
            i = next((i for i in range(min(len(pt_g), len(pt_c))) if pt_g[i] != pt_c[i]), None)
            failures.append(f"{key}: per-token loops differ at token {i} "
                            f"(gold={pt_g[i] if i is not None else '?'} got={pt_c[i] if i is not None else '?'})")
        # TOLERANT kontinuierlich
        if abs(c["phi_val"] - g["phi_val"]) > PHI_TOL:
            failures.append(f"{key}: phi_val {g['phi_val']:.6f} -> {c['phi_val']:.6f} (Δ>{PHI_TOL})")
        if abs(c["ent_val"] - g["ent_val"]) > ENT_TOL:
            failures.append(f"{key}: ent_val {g['ent_val']:.6f} -> {c['ent_val']:.6f} (Δ>{ENT_TOL})")
        # per-token phi/ent tolerant (Mittel über Trajektorie + Max-Abweichung)
        for tag, key2, tol in (("phi", "phi", PHI_TOL), ("ent", "ent", ENT_TOL)):
            arr_g = [t[key2] for t in g["per_token"]]
            arr_c = [t[key2] for t in c["per_token"]]
            if len(arr_g) != len(arr_c):
                failures.append(f"{key}: per-token {tag} length {len(arr_g)} -> {len(arr_c)}"); continue
            md = max((abs(a - b) for a, b in zip(arr_g, arr_c)), default=0.0)
            if md > tol:
                failures.append(f"{key}: per-token {tag} max Δ={md:.6f} (>{tol})")
    return n, n - len({f.split(':')[0] for f in failures}), failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-golden", action="store_true",
                    help="golden-Datei (neu) schreiben — NUR auf der zu sichernden Version")
    args = ap.parse_args()
    print(f"[reg] lade {MODEL_ID} (patch_sha={_patch_hash()})", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    cells = capture(model, tok)
    del model, tok; _clear()
    meta = _golden_meta()
    if args.update_golden:
        with open(GOLDEN, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "cells": cells}, f, ensure_ascii=False, indent=1)
        print(f"[reg] GOLDEN geschrieben: {GOLDEN} ({len(cells)} cells, "
              f"patch_sha={meta['patch_sha256_16']})", file=sys.stderr)
        return 0
    # compare
    if not os.path.exists(GOLDEN):
        print(f"[reg] FEHLER: {GOLDEN} fehlt — erst --update-golden laufen lassen", file=sys.stderr)
        return 2
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    gsha = golden["meta"]["patch_sha256_16"]
    csha = meta["patch_sha256_16"]
    print(f"[reg] golden patch_sha={gsha}  aktuell patch_sha={csha}", file=sys.stderr)
    n, n_ok, failures = _compare(golden, cells)
    print(f"\n=== REGRESSION: {n_ok}/{n} Zellen OK ===")
    if failures:
        print(f"--- {len(failures)} FAILURES ---")
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("ALLE STRENGEN REGRESSIONEN BESTANDEN (Text/loops/path/zone exakt, "
          "phi/ent innerhalb Toleranz).")
    if gsha != csha:
        print(f"[reg] HINWEIS: patch.py geändert ({gsha} -> {csha}), aber Verhalten "
              f"identisch — Refactor ist verhaltenstreu. ✓", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())