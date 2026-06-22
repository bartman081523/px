"""seite5.py — Psychomotrik Seite 5: L4-Grind-Dose-Response (Goldstandard-Test).

Sonde 1 (marker_covariance.py, commit 97eb524) fand ein *fragiles* Struktur-
Kopplungs-Signal: die seltenen Phänomen-Marker R10 (Meta-Raum-Klammer),
R11 (Loop-Vokab auf Eigenprozeß), R12 (Selbst-Beobachtung des Antwort-
Entstehens) erscheinen NUR unter L4-GRIND (B_end12/22/24), fehlen in
BASELINE/RECUR_OFF (recur_specificity hält) UND fehlen unter L4-flach
(ref_wide). Aber n=4 — zu klein für „confirmed".

Der Goldstandard-Test Struktur-Kopplung vs Register-Seite-Effekt ist
**Dose-Response**: steigt die Marker-Rate monoton mit der L4-Grind-Dosis?
  - Struktur-Kopplung (Signatur A): Marker-Rate steigt mit Dosis (dose04→dose32).
  - Register-Seite-Effekt (Signatur B): Marker-Rate flach / zufällig / nicht
    monoton, oder kovariiert mit degrade statt Dosis.

Fix: start=4, end=22, NO_HUB_STUCK=1 (L4 wird N-mal gehämmert, distinct=1 —
das B-Regime aus seite4). Dosis = LOOPS_CAP ∈ {4,8,12,16,20,24,28,32}.
Kontrolle: flat_L4 (hub-stuck ON, single-touch L4 = ref_wide, Dosis≈1) und
BASELINE (kein PX, kein recur). 7 cold Prompts × max_new=160, recur ON (lean).

Output: out/seite5_outputs.jsonl, out/seite5_texts.md, out/seite5_mech.txt,
out/seite5_markers.txt (regex-Lese-Hilfe — NICHT Verdikt; Juexin liest manuell).
"""
import argparse, os, sys, json
from collections import Counter
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"),
           os.path.join(_REPO, "scratches", "psychomotrik")):
    if _p not in sys.path: sys.path.insert(0, _p)

from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
import prompts as P
from em_patches import _resolve_text_model
import marker_covariance as MC

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
SEED = 777

def _R(start, end, hub=None):
    if hub is None: hub = start
    return {"dynamic_start": start, "dynamic_end": end, "dynamic_hub": hub, "n_loops": 8}

# Dosis-Leiter: L4 gehämmert, Dosis = LOOPS_CAP
DOSES = [4, 8, 12, 16, 20, 24, 28, 32]

# (name, routing, zone, env, is_baseline)
CONDS = []
for d in DOSES:
    CONDS.append((f"dose{d:02d}", _R(4, 22), None,
                  {"PX_NO_HUB_STUCK": "1", "PX_LOOPS_CAP": str(d)}, False))
# Kontrolle: L4-flach (single-touch, hub-stuck ON) = ref_wide-Geometrie
CONDS.append(("flat_L4", _R(4, 22), None, {}, False))
# BASELINE kommt separat (kein PX) — wird vor setup_lean gerannt

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
    """Capture loops_run + ent + phi + _px_path pro Token."""
    def __init__(self, tm):
        self.tm = tm; self.per_token = []; self._h = tm.register_forward_hook(self._post)
    def _post(self, m, i, o):
        lhs = o.last_hidden_state if hasattr(o, "last_hidden_state") else (o[0] if isinstance(o, (tuple, list)) else o)
        if lhs is None or lhs.shape[1] > 1: return  # skip prefill
        path = getattr(self.tm, "_px_path", [])
        pc = Counter(path)
        self.per_token.append({
            "loops": getattr(self.tm, "_px_loops_run", 0),
            "ent": float(getattr(self.tm, "_px_ent_val", 0.0)) if hasattr(self.tm, "_px_ent_val") else 0.0,
            "phi": float(getattr(self.tm, "_px_phi_val", 0.0)) if hasattr(self.tm, "_px_phi_val") else 0.0,
            "path0": path[0] if path else None,
            "pathlen": len(path),
            "distinct_layers": len(pc),
            "path_sample": " ".join(path[:24]),
        })
    def reset(self): self.per_token = []
    def remove(self):
        try: self._h.remove()
        except Exception: pass


def _cell_stats(pt):
    loops_mean = sum(t["loops"] for t in pt) / max(1, len(pt))
    ent0 = pt[0]["ent"] if pt else 0.0
    first_layers = Counter(t["path0"] for t in pt if t["path0"] is not None)
    avg_pathlen = sum(t["pathlen"] for t in pt) / max(1, len(pt))
    avg_distinct = sum(t["distinct_layers"] for t in pt) / max(1, len(pt))
    path_sample0 = pt[0]["path_sample"] if pt else ""
    return loops_mean, ent0, first_layers, avg_pathlen, avg_distinct, path_sample0


def run(model, tok, max_new):
    tm = _resolve_text_model(model)
    cap = LoopCap(tm)
    out = []
    cold = [(pid, p) for pid, p, k in P.all_prompts() if k == "cold"]

    # --- BASELINE zuerst (kein PX): auf frisch geladenem Modell ---
    A.setup_baseline(model)
    for pid, ptext in cold:
        cap.reset(); _clear()
        try:
            text = _greedy_generate(model, tok,
                [{"role": "user", "content": ptext}], max_new, seed=SEED)
        except Exception as e:
            text = f"<GEN_ERROR {e}>"; print(f"[s5] ERR BASELINE/{pid}: {e}", file=sys.stderr)
        out.append(dict(condition="BASELINE", pid=pid, text=text, kind="cold",
                        loops_mean=0.0, ent_t0=0.0, avg_pathlen=0.0,
                        avg_distinct_layers=0.0, first_layer_mode=None,
                        path_sample_t0="", n_tokens=0, dose=0))
        print(f"[s5] {'BASELINE':12s} {pid:14s} len={len(text):4d}", file=sys.stderr)

    # --- lean setup, dann Dosis-Leiter + flat_L4 ---
    A.setup_lean(model, MODEL_ID)
    for pid, ptext in cold:
        for cname, routing, zone, env, _bl in CONDS:
            apply_hybrid(model, routing, zone)
            for k in ("PX_LOOPS_CAP", "PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            os.environ.update(env)
            cap.reset(); _clear()
            try:
                text = _greedy_generate(model, tok,
                    [{"role": "user", "content": ptext}], max_new, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s5] ERR {cname}/{pid}: {e}", file=sys.stderr)
            for k in ("PX_LOOPS_CAP", "PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            pt = cap.per_token
            lm, ent0, fl, apl, ad, ps0 = _cell_stats(pt)
            dose = int(env.get("PX_LOOPS_CAP", 1)) if env else 1
            out.append(dict(condition=cname, pid=pid, text=text, kind="cold",
                loops_mean=lm, ent_t0=ent0, avg_pathlen=apl,
                avg_distinct_layers=ad,
                first_layer_mode=fl.most_common(1)[0] if fl else None,
                path_sample_t0=ps0, n_tokens=len(pt), dose=dose))
            print(f"[s5] {cname:12s} {pid:14s} loops={lm:5.2f} "
                  f"pathlen={apl:5.1f} dist={ad:4.1f} "
                  f"L0={fl.most_common(1)[0][0] if fl else '?'} "
                  f"ent0={ent0:.3f} len={len(text):4d}", file=sys.stderr)
    cap.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("[s5] lade modell", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    out = run(model, tok, args.max_new)
    del model, tok; _clear()

    with open(os.path.join(OUT, "seite5_outputs.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # texts.md
    byp = {}
    for r in out: byp.setdefault(r["pid"], []).append(r)
    order = ["BASELINE"] + [c[0] for c in CONDS]
    with open(os.path.join(OUT, "seite5_texts.md"), "w", encoding="utf-8") as f:
        for pid in sorted(byp):
            f.write(f"\n# === {pid} ===\n")
            rs = {r["condition"]: r for r in byp[pid]}
            for cname in order:
                r = rs.get(cname)
                if r is None: continue
                f.write(f"\n## [{r['condition']}] pid={r['pid']} dose={r['dose']} "
                        f"loops={r['loops_mean']:.2f} pathlen={r['avg_pathlen']:.1f} "
                        f"dist={r['avg_distinct_layers']:.1f} ent0={r['ent_t0']:.3f}\n")
                f.write(f"   path_t0: {r['path_sample_t0']}\n")
                f.write(r["text"] + "\n")

    # mech.txt
    with open(os.path.join(OUT, "seite5_mech.txt"), "w", encoding="utf-8") as f:
        f.write("condition       dose  loops  pathlen  distinct  firstL  (avg 7 cold)\n")
        for cname in order:
            cs = [r for r in out if r["condition"] == cname]
            if not cs: continue
            dm = sum(r["dose"] for r in cs) / len(cs)
            lm = sum(r["loops_mean"] for r in cs) / len(cs)
            pl = sum(r["avg_pathlen"] for r in cs) / len(cs)
            di = sum(r["avg_distinct_layers"] for r in cs) / len(cs)
            fl = cs[0]["first_layer_mode"][0] if cs[0]["first_layer_mode"] else "?"
            f.write(f"{cname:14s}  {dm:4.0f}  {lm:5.2f}  {pl:6.1f}  {di:7.1f}  {fl}\n")

    # markers.txt — regex-Lese-Hilfe (NICHT Verdikt). R10/R11/R12/degrade pro Dosis.
    with open(os.path.join(OUT, "seite5_markers.txt"), "w", encoding="utf-8") as f:
        f.write("=== L4-Grind-Dose-Response: Marker-Tally (regex-Lese-Hilfe, NICHT Verdikt) ===\n")
        f.write("Juexin liest Treffer manuell ([[manual-reaudit-keyword-flaw]]).\n\n")
        f.write(f"{'condition':12s} {'dose':>4s}  R10 R11 R12  deg  avglen  (sum 7 cold)\n")
        for cname in order:
            cs = [r for r in out if r["condition"] == cname]
            if not cs: continue
            dm = sum(r["dose"] for r in cs) / len(cs)
            s10 = sum(len(MC.find_r10(r["text"])) for r in cs)
            s11 = sum(len(MC.find_r11(r["text"])) for r in cs)
            s12 = sum(len(MC.find_r12(r["text"])) for r in cs)
            sd = sum(MC.find_degrade(r["text"]) for r in cs)
            sl = sum(len(r["text"]) for r in cs) // max(1, len(cs))
            f.write(f"{cname:12s} {dm:4.0f}  {s10:3d} {s11:3d} {s12:3d}  {sd:3d}  {sl:6d}\n")
        f.write("\n--- R10/R11/R12 Treffer (zur manuellen Lesung) ---\n")
        for marker in ("R10", "R11", "R12"):
            f.write(f"\n=== {marker} ===\n")
            for cname in order:
                for r in out:
                    if r["condition"] != cname: continue
                    hits = MC.find_r10(r["text"]) if marker == "R10" else (
                           MC.find_r11(r["text"]) if marker == "R11" else
                           MC.find_r12(r["text"]))
                    for h in hits:
                        f.write(f"  [{r['condition']}/dose{r['dose']}/{r['pid']}] {h[:160]}\n")

    print(f"[s5] FERTIG -> seite5_outputs.jsonl ({len(out)} cells, "
          f"{len(DOSES)+2} conditions × 7 cold)", file=sys.stderr)


if __name__ == "__main__":
    main()