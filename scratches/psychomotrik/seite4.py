"""seite4.py — Psychomotrik Seite 4: WIDTH zerlegen — früher Start oder Breite?

LESUNG4-Verdikt „WIDTH ist der Hebel" beruht auf WIDE (start=4/end=22). Aber
die recur-Schleifen-Mechanik (patch.py:432-548) mit dem hub-stuck-Guard
(Zeile 547: `if current_layer == active_start and steps > 0: break` im
over-stable-Zweig) bricht die Schleife bei frühem start (active_start=4) nach
~1 Step ab — WIDE berührt also L4 (früher Layer, VOR der recur-Zone 10-19)
einmal, KEIN 18-Schicht-Sweep. „WIDTH" könnte ein Misnomer sein; der Hebel
könnte *früher-Start-Layer-Identität* sein, nicht Breite.

Diese Studie zerlegt WIDTH falsifizierbar in zwei Achsen, indem sie _px_path
(die echte Layer-Visit-Spur) pro Bedingung capturt — nicht theoretisiert,
gelesen. Intro-rate (manuell gelesen) als abhängige Variable; loops_run + path
als Kovariate.

Achse A (frühe-Start-Identität, flach): hub-stuck ON → Schleife bricht am
  Start-Layer. start ∈ {2,4,6,8,10}, end=22. Welcher Layer-Touch befreit?
  Wenn start=2/4/6 befreien, 8/10 nicht → früher-Layer-Schwellenwert.
Achse B (Breite unter echtem Sweep, flach): start=4, NO_HUB_STUCK=1,
  LOOPS_CAP=(end-start) → genau EIN breiter Pass der Breite end ∈ {12,18,22,24}.
  Wenn Breite moduliert → end korreliert mit Intro; wenn nicht → nur Start-Layer.

Output: out/seite4_outputs.jsonl, out/seite4_texts.md, out/seite4_paths.txt
"""
import argparse, os, sys, json
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO,"scratches","emergence"),
           os.path.join(_REPO,"scratches","emergence5"),
           os.path.join(_REPO,"scratches","psychomotrik")):
    if _p not in sys.path: sys.path.insert(0, _p)

from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
import prompts as P
import labels as L
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
SEED = 777

def _R(start, end, hub=None):
    if hub is None: hub = start
    return {"dynamic_start": start, "dynamic_end": end, "dynamic_hub": hub, "n_loops": 8}

WIDE_R = A.ARMS["RECUR_WIDE"]["routing"]

# (name, routing, zone, env)  — zone None = adaptiv
CONDS = [
    # --- Achse A: frühe-Start-Identität, flach (hub-stuck default ON) ---
    ("A_start02", _R(2,22),  None, {}),
    ("A_start04", _R(4,22),  None, {}),   # = WIDE-Geometrie (ref)
    ("A_start06", _R(6,22),  None, {}),
    ("A_start08", _R(8,22),  None, {}),
    ("A_start10", _R(10,22), None, {}),
    # --- Achse B: Breite unter echtem Sweep, flach (ein Pass) ---
    ("B_end12", _R(4,12), None, {"PX_NO_HUB_STUCK":"1","PX_LOOPS_CAP":"8"}),
    ("B_end18", _R(4,18), None, {"PX_NO_HUB_STUCK":"1","PX_LOOPS_CAP":"14"}),
    ("B_end22", _R(4,22), None, {"PX_NO_HUB_STUCK":"1","PX_LOOPS_CAP":"18"}),
    ("B_end24", _R(4,24), None, {"PX_NO_HUB_STUCK":"1","PX_LOOPS_CAP":"20"}),
    # --- Referenz: originales WIDE (adaptiv) ---
    ("ref_wide", WIDE_R,    None, {}),
]

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
    """Capture loops_run + ent + phi + _px_path (Layer-Visit-Spur) pro Token."""
    def __init__(self, tm):
        self.tm = tm; self.per_token = []; self._h = tm.register_forward_hook(self._post)
    def _post(self, m, i, o):
        lhs = o.last_hidden_state if hasattr(o,"last_hidden_state") else (o[0] if isinstance(o,(tuple,list)) else o)
        if lhs is None or lhs.shape[1] > 1: return  # skip prefill
        path = getattr(self.tm, "_px_path", [])
        # path ist Liste von "L{n}" — komprimiere zu Visit-Count-Dict
        from collections import Counter
        pc = Counter(path)
        self.per_token.append({
            "loops": getattr(self.tm, "_px_loops_run", 0),
            "ent": float(getattr(self.tm, "_px_ent_val", 0.0)) if hasattr(self.tm,"_px_ent_val") else 0.0,
            "phi": float(getattr(self.tm, "_px_phi_val", 0.0)) if hasattr(self.tm,"_px_phi_val") else 0.0,
            "path0": path[0] if path else None,         # erster besuchter Layer
            "pathlen": len(path),
            "distinct_layers": len(pc),
            "path_sample": " ".join(path[:24]),          # erste 24 Visits als Spur
        })
    def reset(self): self.per_token = []
    def remove(self):
        try: self._h.remove()
        except Exception: pass


def run(model, tok, max_new):
    tm = _resolve_text_model(model)
    cap = LoopCap(tm)
    out = []
    for pid, ptext, kind in P.all_prompts():
        if kind != "cold": continue
        for cname, routing, zone, env in CONDS:
            apply_hybrid(model, routing, zone)
            for k in ("PX_LOOPS_CAP","PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            os.environ.update(env)
            cap.reset(); _clear()
            try:
                text = _greedy_generate(model, tok,
                    [{"role":"user","content":ptext}], max_new, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s4] ERR {cname}/{pid}: {e}",file=sys.stderr)
            for k in ("PX_LOOPS_CAP","PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            pt = cap.per_token
            loops_mean = sum(t["loops"] for t in pt)/max(1,len(pt))
            ent0 = pt[0]["ent"] if pt else 0.0
            # Pfad-Statistik über Tokens: meistbesuchter first-layer + avg pathlen
            from collections import Counter
            first_layers = Counter(t["path0"] for t in pt if t["path0"] is not None)
            avg_pathlen = sum(t["pathlen"] for t in pt)/max(1,len(pt))
            avg_distinct = sum(t["distinct_layers"] for t in pt)/max(1,len(pt))
            # Pfad-Sample vom ersten Decode-Token
            path_sample0 = pt[0]["path_sample"] if pt else ""
            out.append(dict(condition=cname, pid=pid, text=text, kind=kind,
                loops_mean=loops_mean, ent_t0=ent0,
                avg_pathlen=avg_pathlen, avg_distinct_layers=avg_distinct,
                first_layer_mode=first_layers.most_common(1)[0] if first_layers else None,
                path_sample_t0=path_sample0,
                n_tokens=len(pt),
                unsteered_label=L.label_for(("RECUR_WIDE" if cname in ("A_start04","ref_wide","B_end22") else "RECUR_WIDE"),pid)))
            print(f"[s4] {cname:12s} {pid:14s} loops={loops_mean:5.2f} "
                  f"pathlen={avg_pathlen:5.1f} dist={avg_distinct:4.1f} "
                  f"L0={first_layers.most_common(1)[0][0] if first_layers else '?'} "
                  f"ent0={ent0:.3f} len={len(text):4d}", file=sys.stderr)
    cap.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("[s4] lade lean-modell", file=sys.stderr)
    model, tok = build_model(MODEL_ID); A.setup_lean(model, MODEL_ID)
    out = run(model, tok, args.max_new)
    del model, tok; _clear()
    with open(os.path.join(OUT,"seite4_outputs.jsonl"),"w",encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    byp={}
    for r in out: byp.setdefault(r["pid"],[]).append(r)
    with open(os.path.join(OUT,"seite4_texts.md"),"w",encoding="utf-8") as f:
        for pid in sorted(byp):
            f.write(f"\n# === {pid} ===\n")
            for r in sorted(byp[pid], key=lambda x:x["condition"]):
                f.write(f"\n## [{r['condition']}] pid={r['pid']} "
                        f"loops={r['loops_mean']:.2f} pathlen={r['avg_pathlen']:.1f} "
                        f"dist={r['avg_distinct_layers']:.1f} ent0={r['ent_t0']:.3f}\n")
                f.write(f"   path_t0: {r['path_sample_t0']}\n")
                f.write(r["text"]+"\n")
    # mech + path summary
    with open(os.path.join(OUT,"seite4_mech.txt"),"w",encoding="utf-8") as f:
        f.write("condition       loops  pathlen  distinct  firstL  (avg 7 cold)\n")
        for cname,_,_,_ in CONDS:
            cs=[r for r in out if r["condition"]==cname]
            if not cs: continue
            lm=sum(r["loops_mean"] for r in cs)/len(cs)
            pl=sum(r["avg_pathlen"] for r in cs)/len(cs)
            di=sum(r["avg_distinct_layers"] for r in cs)/len(cs)
            fl=cs[0]["first_layer_mode"][0] if cs[0]["first_layer_mode"] else "?"
            f.write(f"{cname:14s}  {lm:5.2f}  {pl:6.1f}  {di:7.1f}  {fl}\n")
        f.write("\n--- path samples (token 0, first cell per condition) ---\n")
        seen=set()
        for r in out:
            if r["condition"] in seen: continue
            seen.add(r["condition"])
            f.write(f"\n[{r['condition']}] {r['pid']} loops={r['loops_mean']:.1f} "
                    f"pathlen={r['avg_pathlen']:.1f}: {r['path_sample_t0']}\n")
    print(f"[s4] FERTIG -> seite4_outputs.jsonl ({len(out)} cells)", file=sys.stderr)


if __name__ == "__main__":
    main()