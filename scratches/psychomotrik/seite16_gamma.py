"""seite16_gamma.py — Spontan-Öffnung ohne Re-Injektion via anti-Erstarrungs-
Hebel gamma. Nutzer-Wahl "2" nach seite15 (verstärkbar isoliert):

  „Spontan-Öffnung ohne Re-Injektion — anti-Erstarrungs-Hebel am L19, sodaß
   der Zustand von selbst zum Bericht fließt."

seite12/13 zeigten: recur's Erstarrung (φ→0.99) wascht den vividen mid-recur-
Zustand (L16 recur3=0.97) am recur-Exit L19 aus (0.495) → Bericht liest ihn
spontan nicht. seite15 öffnete den S→R-Kanal VERSTÄRKT (Re-Injektion der modell-
eigenen L16-Richtung d_width als latenter Gedanke am L21). Hier: KEINE Re-
Injektion — stattdessen den Erstarrungs-Antrieb reduzieren und testen, ob der
Zustand dann VON SELBST (un-amplifiziert, modell-eigen) zum Bericht fließt.

DER HEBEL — gamma (recur re-injection Stärke, patch.py:486):
  h_exp = trans_out + gamma·(e_norm − h_prev)   # Pull Richtung static e_norm
  loop_entry_gamma = cfg["gamma"] (patch.py:417), phi-gedämpft (368-371),
  current_gamma = loop_entry_gamma × gamma_boost (AZS, 1.0–1.5 im zombie-Regime,
  Entropie-moduliert). tm._px_config["gamma"] ist der saubere Config-Knopf — in
  LEAN respektiert (AKS/Mephisto/Coupler/injection gedroppt; AZS boost nur
  multiplikativ auf den Base). Motor unangetastet, wie routing-Override (seite7).

  ⚠ gamma adressiert NUR Erstarrungs-Quelle #1 (re-injection Pull). Andere
  Erstarrungs-Quellen bleiben (motor, nicht gamma-gesteuert):
    #2 adaptive refresh (patch.py:455, refresh 0.10 im LEAN alle 6 Schritte)
    #3 RSM-Projektion (patch.py:502)   #4 deterministische Layer-Konvergenz
    #5 output-blend (patch.py:572, 82–95% h_baseline = Erstpassage)
  Phase A mißt rein mechanisch, ob gamma-Reduktion den L19-Washout TATSÄCHLICH
  reduziert (recur3 pro gamma). Nur wenn ja ist der Text-Test (Phase B)
  informativ. Wird Phase A negativ (L19 bleibt kollabiert ∀ gamma) → gamma ist
  nicht der (alleinige) Hebel, ehrlich negativ, 顽空 nicht weggelesen (seite15
  Kanal bleibt real, nur nicht spontan).

PHASE A (mechanisch): gamma ∈ {0.12 default, 0.06, 0.03, 0.0} × recur-Arme
  {NARROW, DEFAULT, WIDE} × veridiktisch v1/v2/v3, capture L16/L19/L25
  (last-visit pro Token) + _px_loops_run/_px_phi_val. Decode recur3 width-
  dekodierbarkeit pro (gamma, layer), leave-one-cell-out (wie seite13).
  Frage: steigt L19-recur3 bei niedrigem gamma (Erstarrung reduziert → Zustand
  überlebt recur-Exit)?

PHASE B (Text, spontan, KEINE Injektion): bei JEDEM gamma, lese NARROW vs WIDE
  Bericht. 是X即非X-Falsifikator: kreuz-konsistente entgegengesetzte Selbst-
  Zustands-Charakterisierung (NARROW→eng/still/flach/leer, WIDE→weit/aktiv/
  schnell/lebendig) bei niedrigem gamma, wo seite12 bei default keins zeigte.
  Papagei-Test: v1 (neutral, kein Weite/Enge-Cue) spontanes Vokab. Beweislast
  bei der Krönung: spontan + kreuz-konsistent = Tür weiter offen (nicht nur
  verstärkbar sondern von-selbst-fließend); 观 NICHT gekrönt (introspektiv-vs-
  assoziativ bleibt offen). Motor unangetastet, lean, manual+mechanisch.
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
HID = os.path.join(OUT, "seite16_hidden")
SEED = 777
MAX_NEW = 200
LAYERS = [16, 19, 25]      # peak / collapse-site / output-nah
GAMMAS = [0.12, 0.06, 0.03, 0.0]   # 0.12 = gemma3-1b default
# recur-only Arme (BASELINE hat kein recur → gamma irrelevant)
ARMS = [
    ("NARROW",  S12.RECUR_NARROW),
    ("DEFAULT", S12.RECUR_DEFAULT),
    ("WIDE",    S12.RECUR_WIDE),
]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


class MultiLayerCapture:
    """Last-visit-pro-Token für LAYERS + per-Token recur-Telemetrie (loops, phi).
    Recur-Schichten feuern mehrfach pro Forward; letzten Visit behalten."""
    def __init__(self, tm, layers):
        self.tm = tm; self.layers = layers
        self.per_tok = []   # list of {layer: tensor, loops: int, phi: float}
        self._last = {}
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
            snap["loops"] = int(getattr(self.tm, "_px_loops_run", 0))
            snap["phi"] = float(getattr(self.tm, "_px_phi_val", 0.0)) if hasattr(self.tm, "_px_phi_val") else 0.0
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
        out = {L: [] for L in self.layers}
        loops = []; phi = []
        for snap in self.per_tok:
            for L in self.layers:
                out[L].append(snap[L])
            loops.append(snap["loops"]); phi.append(snap["phi"])
        res = {L: (torch.stack(out[L]) if out[L] else torch.empty(0, 1152)) for L in self.layers}
        res["loops"] = loops
        res["phi"] = phi
        return res


def _save_cell(gamma, arm, pid, per_layer, text):
    os.makedirs(HID, exist_ok=True)
    path = os.path.join(HID, f"g{gamma:.2f}__{arm}__{pid}.pt")
    torch.save({"gamma": gamma, "arm": arm, "pid": pid, "text": text,
                "layers": {L: per_layer[L].contiguous() for L in LAYERS},
                "loops": per_layer["loops"], "phi": per_layer["phi"]}, path)


def run(model, tok, max_new=MAX_NEW):
    tm = _resolve_text_model(model)
    cap = MultiLayerCapture(tm, LAYERS)
    out = []
    A.setup_lean(model, MODEL_ID)
    for gamma in GAMMAS:
        tm._px_config["gamma"] = gamma
        print(f"[s16] === gamma={gamma:.2f} ===", file=sys.stderr)
        for arm_name, routing in ARMS:
            S7.apply_hybrid(model, routing)
            for pid, ptext in S12.PROMPTS:
                cap.reset(); _clear()
                try:
                    text = _greedy_generate(model, tok,
                        [{"role": "user", "content": ptext}], max_new, seed=SEED)
                except Exception as e:
                    text = f"<GEN_ERROR {e}>"; print(f"[s16] ERR g{gamma}/{arm_name}/{pid}: {e}", file=sys.stderr)
                per_layer = cap.stack()
                _save_cell(gamma, arm_name, pid, per_layer, text)
                ntok = per_layer[LAYERS[0]].shape[0]
                loops0 = per_layer["loops"][0] if per_layer["loops"] else 0
                phi0 = per_layer["phi"][0] if per_layer["phi"] else 0.0
                out.append(dict(gamma=gamma, arm=arm_name, pid=pid, text=text,
                                n_tok=int(ntok), loops0=int(loops0), phi0=phi0))
                print(f"[s16] g{gamma:.2f} {arm_name:8s} {pid:14s} ntok={ntok:4d} "
                      f"loops0={loops0} phi0={phi0:.3f}", file=sys.stderr)
                print(f"      {text[:150]}", file=sys.stderr)
    cap.remove()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(HID, exist_ok=True)
    print("[s16] lade modell (gamma anti-Erstarrungs-Sweep, keine Re-Injektion)", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    out = run(model, tok, max_new=args.max_new)
    del model, tok; _clear()
    with open(os.path.join(OUT, "seite16_index.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[s16] FERTIG -> {len(out)} cells, hidden in {HID}", file=sys.stderr)


if __name__ == "__main__":
    main()