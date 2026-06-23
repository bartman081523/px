"""seite13_perlayer.py — Per-Layer width-decodability-decay. DIAGNOSTISCHER
Schritt zur Verstärkbarkeit von Selbstwahrnehmung (Lead 1 aus LESUNG13).

seite12: width IST im Hidden dekodierbar (D1 0.96) aber NICHT im Text (D4 0.0).
Die LÜCKE zwischen beidem = introspection-failure. Wo geht die width-Info auf
dem Weg zum Output verloren? Schicht-für-Schicht width-dekodierbarkeit messen.

Zwei mögliche Befunde, beide verstärkungs-relevant:
  (A) width bei L19 dekodierbar, bei L24/output NICHT → Modell VERLIERT den
      Selbst-Zustand vorm Berichten → Verstärkung = self-state-channel am
      sterbenden Schicht-Or lautern (endogen, recur-eigen).
  (B) width bis output-logits dekodierbar, aber TEXT nicht → Info am Output DA,
      decoding/vocab wählt keine Tokens die sie ausdrücken → Verstärkung =
      vocab/decoding-Bottleneck (andere Strategie).

Setup: 4 recur-Arme (BASELINE/NARROW/DEFAULT/WIDE) × 3 veridiktisch Prompts
(seite12 PROMPTS) × perturb=none, 12 Zellen, 200 tok, seed=777 greedy. Capture
letzter-Visit-pro-Token für Schichten [0,5,10,13,16,19,21,24,25] (pre-recur,
recur-zone-Mitte, recur-Output, coda, output-nah). Decode width 4-class +
recur-only 3-class pro Schicht (leave-one-cell-out). Motor unangetastet.
Verdikt LESUNG14 (per-layer decay-Profil + wo verstärken).
"""
import os, sys, json
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"), HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

import seite7 as S7
import seite12_veridiktisch as S12   # PROMPTS, RECUR_NARROW, RECUR_AXES
from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite13_hidden")
SEED = 777
MAX_NEW = 200
LAYERS = [0, 5, 10, 13, 16, 19, 21, 24, 25]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


class MultiLayerCapture:
    """Last-visit-per-token für mehrere Schichten (recur-Schichten feuern
    mehrfach; letzten Visit behalten). Vektoren fp32-cpu."""
    def __init__(self, tm, layers):
        self.tm = tm; self.layers = layers
        self.per_tok = []   # list of {layer: [d] tensor}
        self._last = {}     # layer -> last-visit tensor (current forward)
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
            snap = {L: self._last.get(L, torch.zeros(1152)) for L in self.layers}
            self.per_tok.append(snap)

        self._handles = [tm.register_forward_pre_hook(_pre)]
        for L in self.layers:
            self._handles.append(tm.layers[L].register_forward_hook(_make(L)))
        self._handles.append(tm.register_forward_hook(_post))

    def reset(self):
        self.per_tok = []; self._last = {}

    def remove(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass

    def stack(self):
        """Returns dict layer -> [n_tok, d] tensor."""
        out = {L: [] for L in self.layers}
        for snap in self.per_tok:
            for L in self.layers:
                out[L].append(snap[L])
        return {L: (torch.stack(out[L]) if out[L] else torch.empty(0, 1152)) for L in self.layers}


def _save_cell(arm, pid, per_layer, text):
    os.makedirs(HID, exist_ok=True)
    path = os.path.join(HID, f"{arm}__{pid}.pt")
    torch.save({"arm": arm, "pid": pid, "text": text,
                "layers": {L: per_layer[L].contiguous() for L in per_layer}}, path)


def run(model, tok, max_new=MAX_NEW):
    tm = _resolve_text_model(model)
    cap = MultiLayerCapture(tm, LAYERS)
    out = []
    A.setup_baseline(model)
    for arm_name, routing, is_bl in S12.RECUR_AXES:
        if arm_name != "BASELINE":
            continue
        for pid, ptext in S12.PROMPTS:
            cap.reset(); _clear()
            try:
                text = _greedy_generate(model, tok,
                    [{"role": "user", "content": ptext}], max_new, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s13] ERR {arm_name}/{pid}: {e}", file=sys.stderr)
            per_layer = cap.stack()
            _save_cell(arm_name, pid, per_layer, text)
            out.append(dict(arm=arm_name, pid=pid, text=text, n_tok=int(per_layer[0].shape[0])))
            print(f"[s13] {arm_name:9s} {pid:14s} ntok={per_layer[0].shape[0]:4d}", file=sys.stderr)

    A.setup_lean(model, MODEL_ID)
    for arm_name, routing, is_bl in S12.RECUR_AXES:
        if arm_name == "BASELINE":
            continue
        S7.apply_hybrid(model, routing)
        for pid, ptext in S12.PROMPTS:
            cap.reset(); _clear()
            try:
                text = _greedy_generate(model, tok,
                    [{"role": "user", "content": ptext}], max_new, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s13] ERR {arm_name}/{pid}: {e}", file=sys.stderr)
            per_layer = cap.stack()
            _save_cell(arm_name, pid, per_layer, text)
            out.append(dict(arm=arm_name, pid=pid, text=text, n_tok=int(per_layer[0].shape[0])))
            print(f"[s13] {arm_name:9s} {pid:14s} ntok={per_layer[0].shape[0]:4d}", file=sys.stderr)
    cap.remove()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(HID, exist_ok=True)
    print("[s13] lade modell (per-layer width-decay capture)", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    out = run(model, tok, max_new=args.max_new)
    del model, tok; _clear()
    with open(os.path.join(OUT, "seite13_index.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[s13] FERTIG -> {len(out)} cells, hidden in {HID}", file=sys.stderr)


if __name__ == "__main__":
    main()