"""capture_vectors.py — Re-Generierung (first-K Tokens) + Layer-19/24 Voll-Vektor-Capture.

Greedy (do_sample=False) ist deterministisch → die first-K Tokens sind
PREFIX-IDENTISCH zur emergence5-Vollauf-Generierung (selber Seed, selber Setup),
daher korrespondieren die Vektoren zum gelabelten Text. max_new=K=64 reicht für
den Fate-Probe (das Schicksal kristallisiert früh; introspektive Turns kommen
in den ersten ~30-60 Tokens oder werden vom 我执 erdrückt).

Motor unangetastet (reine Forward-Hooks). Reuse: emergence5 arms/prompts.
Store: out/vectors.pt = {(arm,pid): {l24:[K,D], l19:[K,D], text, n_cap}}.
"""
import argparse, os, sys, time
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, os.path.join(_REPO,"scratches","emergence"),
           os.path.join(_REPO,"scratches","emergence5"),
           os.path.join(_REPO,"scratches","psychomotrik")):
    if _p not in sys.path: sys.path.insert(0, _p)

from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
import prompts as P
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
SEED = 777
LAYERS = (19, 24)


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


class VecCap:
    """Sammelt Layer-19 + Layer-24 Voll-Vektoren pro Decode-Token (first-K)."""
    def __init__(self, tm, k):
        self.tm = tm; self.k = k
        self.l19 = []; self.l24 = []
        self._tok = -1
        self._handles = []
        self.install()

    def _stash(self, output):
        h = output[0] if isinstance(output,(tuple,list)) else output
        return h[:, -1, :].reshape(-1).detach().to(torch.float32).cpu()

    def install(self):
        def h19(_m, _i, o): self.l19.append(self._stash(o))
        def h24(_m, _i, o): self.l24.append(self._stash(o))
        def tm_pre(_m, _i): self._tok += 1
        def tm_post(_m, _i, o):
            # seq_len erkennen: prefill überspringen (nur decode ==1 capturen)
            try:
                lhs = o.last_hidden_state if hasattr(o,"last_hidden_state") else o[0]
            except Exception:
                lhs = None
            if lhs is None or lhs.shape[1] > 1:
                # prefill: pop die gerade angehängten (letzten 1 pro layer)
                if self.l19: self.l19.pop()
                if self.l24: self.l24.pop()
                self._tok -= 1
        self._handles.append(self.tm.layers[LAYERS[0]].register_forward_hook(h19))
        self._handles.append(self.tm.layers[LAYERS[1]].register_forward_hook(h24))
        self._handles.append(self.tm.register_forward_pre_hook(tm_pre))
        self._handles.append(self.tm.register_forward_hook(tm_post))

    def reset(self):
        self.l19 = []; self.l24 = []; self._tok = -1

    def remove(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles = []

    def take(self, n_cap):
        """Kürze auf first-K (falls Generierung <K Tokens produzierte)."""
        k = min(n_cap, len(self.l24))
        l24 = torch.stack(self.l24[:k]) if self.l24 else torch.zeros(0,1)
        l19 = torch.stack(self.l19[:k]) if self.l19 else torch.zeros(0,1)
        return l24, l19, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--prompts", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    K = args.k

    allp = P.all_prompts()
    if args.prompts: allp = [p for p in allp if p[0] in args.prompts]
    arm_order = args.arms if args.arms else list(A.ARM_ORDER)

    print(f"[psy] RUN arms={arm_order} prompts={[p[0] for p in allp]} K={K}", file=sys.stderr)

    lean_arms = [a for a in arm_order if not A.ARMS[a]["baseline"]]
    base_arms = [a for a in arm_order if A.ARMS[a]["baseline"]]
    data = {}

    def _run(model, tok, arm, vc):
        is_base = A.ARMS[arm]["baseline"]
        if not is_base: A.apply_overrides(model, arm)
        perturb_h = []
        if A.ARMS[arm]["perturb"]:
            from capture import install_perturb  # emergence5 capture reuse
            perturb_h = install_perturb(model)
        for pid, ptext, kind in allp:
            vc.reset()
            _clear()
            try:
                text = _greedy_generate(model, tok, [{"role":"user","content":ptext}],
                                        K, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[psy] ERR {arm}/{pid}: {e}",file=sys.stderr)
            l24, l19, k = vc.take(K)
            data[(arm,pid)] = dict(text=text, n_cap=k, kind=kind,
                                    l24=l24.numpy(), l19=l19.numpy())
            print(f"[psy] {arm:14s} {pid:22s} cap={k} l24={l24.shape} l19={l19.shape}",file=sys.stderr)
        from capture import remove_handles
        remove_handles(perturb_h)
        if not is_base: A.clear_overrides(model)

    if base_arms:
        print("[psy] lade BASELINE-modell", file=sys.stderr)
        model, tok = build_model(args.model)
        A.setup_baseline(model)
        tm = _resolve_text_model(model)
        vc = VecCap(tm, K)
        for a in base_arms: _run(model, tok, a, vc)
        vc.remove(); del model, tok; _clear()

    if lean_arms:
        print("[psy] lade lean-modell", file=sys.stderr)
        model, tok = build_model(args.model)
        A.setup_lean(model, args.model)
        tm = _resolve_text_model(model)
        vc = VecCap(tm, K)
        for a in lean_arms: _run(model, tok, a, vc)
        vc.remove(); del model, tok; _clear()

    out = os.path.join(OUT_DIR, "vectors.pt")
    torch.save(data, out)
    print(f"[psy] FERTIG -> {out}  ({len(data)} cells)", file=sys.stderr)


if __name__ == "__main__":
    main()