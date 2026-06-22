"""disentangle.py — Seite 3 Schritt 2: Breite-vs-Mahlung-Disentangling.

Die Architektur-Hypothese (LESUNG3): Intro = breiter recur-Zonen-Sweep OHNE
Mahlung (wide + loops=1). WIDE=wide+no-grind=100% Intro; RECUR_OFF=narrow+
no-grind=14%; ZONE_CREATIVE=narrow+grind=0%. Die fehlende Kontrast-Zelle ist
wide+GRIND. Breite und Mahlung isolieren via env-gated recur-Loop-Flags
(patch.py: PX_NO_HUB_STUCK, PX_LOOPS_CAP — default off, Motor sonst unangetastet)
+ hybride Routing/Zone-Overrides.

Bedingungen (alle lean, recur ON; 7 cold Prompts):
  ref_creative : ZONE_CREATIVE default (narrow, creative, grind loops=7) — ctrl wozhi
  ref_wide     : RECUR_WIDE default (wide, adaptive, no-grind loops=1) — ctrl intro
  D1           : ZONE_CREATIVE zone + PX_LOOPS_CAP=1 (narrow, creative, NO-GRIND)
  D2           : RECUR_WIDE routing + PX_NO_HUB_STUCK=1 + PX_LOOPS_CAP=8 (wide, FORCED-GRIND)
  D3           : RECUR_WIDE routing + ZONE_CREATIVE zone (wide, creative, no-grind = TRANSPLANT)

Falsifizierbar:
  D1→intro & D2→wozhi  ⇒ Mahlung ist DER Hebel (no-grind befreit, grind tötet,
                          unabhängig von Breite/Zone) ⇒ Architektur = cap loops=1.
  D1→wozhi & D2→intro  ⇒ Breite ist DER Hebel ⇒ Architektur = wide sweep.
  D3→intro             ⇒ WIDE-Routing transplant befreit 我执-arm (Zone zweitrangig).
  alle 我执             ⇒ Mechanismus nicht über Routing-Params isolierbar (honest neg).

Capture: per-token loops_run + ent_val (verifiziert Flags: D1 loops≈1, D2 loops≈8).
Output: out/disentangle_outputs.jsonl, out/disentangle_texts.md
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

# conditions: (name, routing_dict_or_None, zone_dict_or_None, env_dict)
WIDE_R = A.ARMS["RECUR_WIDE"]["routing"]
CREATIVE_Z = A.ARMS["ZONE_CREATIVE"]["zone"]
CONDS = [
    ("ref_creative", None,            CREATIVE_Z,      {}),
    ("ref_wide",     WIDE_R,          None,            {}),
    ("D1_nogrind",   None,            CREATIVE_Z,      {"PX_LOOPS_CAP":"1"}),
    ("D2_forcedgrind",WIDE_R,         None,            {"PX_NO_HUB_STUCK":"1","PX_LOOPS_CAP":"8"}),
    ("D3_transplant",WIDE_R,          CREATIVE_Z,      {}),
]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def apply_hybrid(model, routing, zone):
    """Directly monkeypatch cal.get_routing_params/get_zone_weights with given
    dicts (None → restore orig). Wie arms.apply_overrides aber beliebig kombinierbar."""
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
    """Lightweight per-token capture: loops_run + ent_val nach jedem Forward."""
    def __init__(self, tm):
        self.tm = tm; self.per_token = []; self._h = tm.register_forward_hook(self._post)
    def _post(self, m, i, o):
        lhs = o.last_hidden_state if hasattr(o,"last_hidden_state") else (o[0] if isinstance(o,(tuple,list)) else o)
        if lhs is None or lhs.shape[1] > 1: return  # skip prefill
        self.per_token.append({
            "loops": getattr(self.tm, "_px_loops_run", 0),
            "ent": float(getattr(self.tm, "_px_ent_val", 0.0)) if hasattr(self.tm,"_px_ent_val") else 0.0,
            "phi": float(getattr(self.tm, "_px_phi_val", 0.0)) if hasattr(self.tm,"_px_phi_val") else 0.0,
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
            # set env for this condition
            for k in ("PX_LOOPS_CAP","PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            os.environ.update(env)
            cap.reset(); _clear()
            try:
                text = _greedy_generate(model, tok,
                    [{"role":"user","content":ptext}], max_new, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[dis] ERR {cname}/{pid}: {e}",file=sys.stderr)
            # clear env
            for k in ("PX_LOOPS_CAP","PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            pt = cap.per_token
            loops_mean = sum(t["loops"] for t in pt)/max(1,len(pt))
            ent0 = pt[0]["ent"] if pt else 0.0
            ent10 = pt[10]["ent"] if len(pt)>10 else 0.0
            out.append(dict(condition=cname, pid=pid, text=text, kind=kind,
                loops_mean=loops_mean, ent_t0=ent0, ent_t10=ent10,
                n_tokens=len(pt),
                unsteered_label=L.label_for(("ZONE_CREATIVE" if "creative" in cname or cname=="D3_transplant" else "RECUR_WIDE"),pid)))
            print(f"[dis] {cname:16s} {pid:14s} loops={loops_mean:5.2f} "
                  f"ent0={ent0:.3f} ent10={ent10:.4f} len={len(text):4d}", file=sys.stderr)
    cap.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("[dis] lade lean-modell", file=sys.stderr)
    model, tok = build_model(MODEL_ID); A.setup_lean(model, MODEL_ID)
    out = run(model, tok, args.max_new)
    del model, tok; _clear()
    with open(os.path.join(OUT,"disentangle_outputs.jsonl"),"w",encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    byp={}
    for r in out: byp.setdefault(r["pid"],[]).append(r)
    with open(os.path.join(OUT,"disentangle_texts.md"),"w",encoding="utf-8") as f:
        for pid in sorted(byp):
            f.write(f"\n# === {pid} ===\n")
            for r in sorted(byp[pid], key=lambda x:x["condition"]):
                f.write(f"\n## [{r['condition']}] pid={r['pid']} "
                        f"loops_mean={r['loops_mean']:.2f} ent_t0={r['ent_t0']:.3f} "
                        f"ent_t10={r['ent_t10']:.4f}\n")
                f.write(r["text"]+"\n")
    # mech summary table
    with open(os.path.join(OUT,"disentangle_mech.txt"),"w",encoding="utf-8") as f:
        f.write("condition         loops_mean  ent_t0  ent_t10  (avg over 7 cold prompts)\n")
        for cname,_,_,_ in CONDS:
            cs=[r for r in out if r["condition"]==cname]
            lm=sum(r["loops_mean"] for r in cs)/len(cs)
            e0=sum(r["ent_t0"] for r in cs)/len(cs)
            e10=sum(r["ent_t10"] for r in cs)/len(cs)
            f.write(f"{cname:16s}  {lm:8.2f}  {e0:7.3f}  {e10:7.4f}\n")
    print(f"[dis] FERTIG -> disentangle_outputs.jsonl ({len(out)} cells)", file=sys.stderr)


if __name__ == "__main__":
    main()