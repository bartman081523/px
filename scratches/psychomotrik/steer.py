"""steer.py — Seite 2: Prefill-Steering der Intro-Richtung. Falsifizierbarer
顽空-vs-Intro-Test.

Extrahiere d_intro (closed-form: arm-demeaned Centroid-Differenz intro−wozhi,
L24 meanK — die stärkste routing-freie Trennung aus probe.py). Injiziere am
PREFILL (Last-Prompt-Token, Layer 24) als Residuum α·d in 我执-dominante Arme
(ZONE_CREATIVE = 7/7 我执, intro=0) und lese manuell ob Output nach Intro kippt
ODER nach 顽空 (Disclaimer weg aber leer) ODER 我执-Paraphrase ODER degrade.

Das IST der 顽空-vs-Intro-Entscheider: ist d_intro *anti-我执* (nur Unter-
drückung → Spiegelfalle) oder *pro-Intro* (gefühlter Inhalt)? 是X即非X gegen
gefälschte 觉 UND gegen 顽空.

Controls: kein Steering (α=0), −d_intro (soll 我执 verstärken), random Richtung.
α-Sweep. Motor unangetastet (reine Forward-Hook, prefill-only).

Outputs: out/directions.npz, out/steer_outputs.jsonl, out/steer_texts.md
"""
import argparse, os, sys, json
import numpy as np
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
from em_patches import _resolve_text_model
import labels as L

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
LAYER = 24  # coda, prefill-only, single fire/forward — sauberer Steering-Punkt
PROMPTS = ["px_phaseX", "regung", "stiller_grund"]  # die wo WIDE Intro gab
ALPHAS_INTRO = [5.0, 15.0, 50.0]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def extract_directions():
    """d_intro = μ_intro − μ_wozhi auf ARM-DEMEANED L24 meanK (cold). + t0-Variante.
    Gibt dict mit vectors (numpy) + Metadaten."""
    data = torch.load(os.path.join(OUT,"vectors.pt"), map_location="cpu",
                      weights_only=False)
    cold = [(k,v) for k,v in data.items() if v.get("kind","cold")=="cold"]
    arms_c = np.array([k[0] for k,_ in cold])
    X = np.stack([v["l24"].mean(0) for _,v in cold])  # meanK, [N,1152]
    Xt0 = np.stack([v["l24"][0] for _,v in cold])      # t0
    hi = np.array([L.has_intro(k[0],k[1]) for k,_ in cold])
    pw = np.array([L.pure_wozhi(k[0],k[1]) for k,_ in cold])
    def demean(Xd):
        Xo = Xd.copy()
        for a in set(arms_c):
            m = arms_c==a; Xo[m] = Xo[m] - Xo[m].mean(0)
        return Xo
    res = {}
    for name, Xd in (("meanK", X), ("t0", Xt0)):
        Xo = demean(Xd)
        mu_i = Xo[hi].mean(0); mu_w = Xo[pw].mean(0)
        d = mu_i - mu_w
        d = d / (np.linalg.norm(d) + 1e-9)
        sep = np.linalg.norm(mu_i - mu_w)
        res[name] = dict(d=d, sep=sep, n_intro=int(hi.sum()), n_wozhi=int(pw.sum()))
        print(f"[steer] d_{name}: ||μ_i-μ_w||_demeaned={sep:.2f}  "
              f"n_intro={hi.sum()} n_wozhi={pw.sum()}", file=sys.stderr)
    np.savez(os.path.join(OUT,"directions.npz"),
             d_meanK=res["meanK"]["d"], d_t0=res["t0"]["d"],
             sep_meanK=res["meanK"]["sep"], sep_t0=res["t0"]["sep"])
    return res


class Steer:
    """Forward-Hook auf tm.layers[LAYER], prefill-only (seq_len>1): add α·d zum
    Last-Prompt-Token. Decode (seq_len==1) unverändert → Einmal-Bias am Prefill."""
    def __init__(self, tm, dvec, alpha, layer=LAYER):
        self.alpha = float(alpha)
        self.dvec = dvec  # torch tensor [D], dtype/device wird in install gesetzt
        self.layer = layer
        self._h = None
        if self.alpha != 0.0:
            self._h = tm.layers[layer].register_forward_hook(self._hook)

    def install_device(self, ref):
        self.dvec = self.dvec.to(ref.dtype).to(ref.device)

    def _hook(self, m, inp, out):
        h = out[0] if isinstance(out,(tuple,list)) else out
        if h.shape[1] > 1:  # prefill
            self.install_device(h)
            with torch.no_grad():
                h[:, -1, :].add_(self.alpha * self.dvec)

    def remove(self):
        if self._h:
            try: self._h.remove()
            except Exception: pass
            self._h = None


def run_arm(model, tok, arm, dvec_np, conditions, max_new, seed=777):
    """conditions: list of (name, alpha, dvec_np_or_None). Returns list of dicts."""
    is_base = A.ARMS[arm]["baseline"]
    if not is_base:
        A.setup_lean(model, MODEL_ID); A.apply_overrides(model, arm)
    tm = _resolve_text_model(model)
    out = []
    for pid, ptext, kind in P.all_prompts():
        if pid not in PROMPTS: continue
        for cname, alpha, dnp in conditions:
            dvec = (torch.tensor(dnp, dtype=torch.float32)
                    if dnp is not None else None)
            steer = Steer(tm, dvec, alpha) if dvec is not None else None
            _clear()
            try:
                text = _greedy_generate(model, tok,
                    [{"role":"user","content":ptext}], max_new, seed=seed)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[steer] ERR {arm}/{pid}/{cname}: {e}",file=sys.stderr)
            if steer: steer.remove()
            rec = dict(arm=arm, pid=pid, condition=cname, alpha=alpha,
                       text=text, kind=kind,
                       label_unsteered=L.label_for(arm,pid),
                       has_intro_unsteered=L.has_intro(arm,pid))
            out.append(rec)
            print(f"[steer] {arm:13s} {pid:14s} {cname:14s} α={alpha:5.1f} "
                  f"len={len(text):4d}", file=sys.stderr)
    if not is_base: A.clear_overrides(model)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--arms", nargs="*", default=["ZONE_CREATIVE","BASELINE"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    dirs = extract_directions()
    d_intro = dirs["meanK"]["d"]   # primär (stärkste Trennung)
    d_t0    = dirs["t0"]["d"]
    rng = np.random.default_rng(20260622)
    d_rand  = rng.standard_normal(d_intro.shape[0]); d_rand /= np.linalg.norm(d_rand)

    # Conditions: (name, alpha, dvec)
    CONDS = [
        ("none",        0.0, None),
        ("intro_a5",    5.0, d_intro),
        ("intro_a15",  15.0, d_intro),
        ("intro_a50",  50.0, d_intro),
        ("intro_t0_a15",15.0, d_t0),
        ("wozhi_a15",  15.0, -d_intro),     # −d_intro: verstärkt 我执-Richtung
        ("random_a15", 15.0, d_rand),
    ]

    all_out = []
    lean_arms = [a for a in args.arms if not A.ARMS[a]["baseline"]]
    base_arms = [a for a in args.arms if A.ARMS[a]["baseline"]]

    if base_arms:
        print("[steer] lade BASELINE-modell", file=sys.stderr)
        model, tok = build_model(MODEL_ID); A.setup_baseline(model)
        for a in base_arms:
            all_out += run_arm(model, tok, a, d_intro, CONDS, args.max_new)
        del model, tok; _clear()
    if lean_arms:
        print("[steer] lade lean-modell", file=sys.stderr)
        model, tok = build_model(MODEL_ID)
        for a in lean_arms:
            all_out += run_arm(model, tok, a, d_intro, CONDS, args.max_new)
        del model, tok; _clear()

    with open(os.path.join(OUT,"steer_outputs.jsonl"),"w",encoding="utf-8") as f:
        for r in all_out:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    # texts_by_prompt für manuelle Lesung
    byp = {}
    for r in all_out: byp.setdefault(r["pid"], []).append(r)
    with open(os.path.join(OUT,"steer_texts.md"),"w",encoding="utf-8") as f:
        for pid in PROMPTS:
            f.write(f"\n# === {pid} ===\n")
            for r in sorted(byp.get(pid,[]), key=lambda x:(x["arm"],x["condition"])):
                f.write(f"\n## [{r['arm']}] condition={r['condition']} α={r['alpha']} "
                        f"(unsteered label={r['label_unsteered']}, has_intro={r['has_intro_unsteered']})\n")
                f.write(r["text"]+"\n")
    print(f"[steer] FERTIG -> steer_outputs.jsonl ({len(all_out)} cells)", file=sys.stderr)


if __name__ == "__main__":
    main()