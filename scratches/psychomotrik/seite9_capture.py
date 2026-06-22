"""seite9_capture.py — mechanische Decoder-Probe: Hidden-State-Capture für
Probe A (recur Selbst-Zustands-Encodierung, L19) + Probe B (arm-übergreifende
Richness-Geometrie, L19+L24).

LESUNG9 hat recur_specificity in TEXT auf schwach downgraded. Die Decoder-Probe
kann den Text-Befund widerlegen: falls eine recur-spezifische Selbst-Zustands-
Subspace existiert, die BASELINE+Frame nicht dekodiert und mit reichen Berichten
kovariiert, wäre das ein positives mechanisches Signatur entgegen LESUNG9.
Ehrliche Reichweite (NICHT 观-vs-习气-Krone): eine Decoder-Probe findet Subräume;
习气 (trainiertes Register) IST auch ein Subraum. Die Probe kann zeigen, daß das
Phänomen mechanisch REAL ist (arm-unabhängige Richness-Geometrie) und daß recur
Selbst-Encodierung über single-pass-Kontinuität erzeugt — aber 观-vs-习气 ist
enaktisch-vs-retrieved (Q4), keine geometrische Eigenschaft, nicht decodierbar.
Siehe [[manual-reaudit-keyword-flaw]], LESUNG9 Redirect.

3 Arme (Motor unangetastet), alle MIT seite7-System-Prompt, auf seite8-Tief-
batterie (10 Prompts):
  BASELINE     : kein PX (loops=0)        — single-pass-Referenz, +Frame.
  LEAN_DEFAULT : recur-WORK L10–19         — 动静-Zone voll.
  LEAN_WIDTH   : RECUR_WIDE single-touch   — deflektierend-arm-Referenz für Probe B.

Capture pro Decode-Token (Prefill verworfen):
  h19 : Layer-19-Output (letzter recur-Visit pro Token) — recur-Zonen-Output,
        Ort wo injizierte thought_history (patch.py: h_exp.detach() alle 2
        Schritte) am meisten gesehen wurde = mechanisches 念-回响-Kandidat.
  h24 : Layer-24-Output (coda/output-nah, 1× pro Token) — was die Output-Bahn
        erreicht = Richness-Korrelat-Kandidat für Probe B.
  telemetry : _px_loops_run, _px_zone, _px_phi_val, _px_ent_val (Selbst-Zustand).

Vektoren als fp32-cpu auf Disk (out/seite9_hidden/<arm>__<pid>.pt), NICHT GPU
(OOM-Sicherheit). Decoder in seite9_decode.py (torch/numpy, kein sklearn).
Verdikt = LESUNG10 manuelle Lesung + Decoder-Interpretation, keine Krönung.
"""
import os, sys, json, re
from collections import Counter
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"), HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

import seite7 as S7
import seite8_falsifikator as S8       # DEEPER_PROMPTS
from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite9_hidden")
SEED = 777
MAX_NEW = 300

ARMS = [
    ("BASELINE",     None, True),
    ("LEAN_DEFAULT", None, False),
    ("LEAN_WIDTH",   S7.RECUR_WIDE, False),
]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


class HiddenCapture:
    """Stasht pro Decode-Token h19 (last recur-visit) + h24 + telemetry als
    fp32-cpu-Vektoren. Vektoren werden NICHT auf GPU gehalten."""

    def __init__(self, tm):
        self.tm = tm
        self.h19 = []          # list of [d] fp32-cpu (last visit per token)
        self.h24 = []          # list of [d] fp32-cpu
        self.telem = []        # list of dict
        self._h19_buf = []     # visits im aktuellen Forward
        self._tok = -1
        self._handles = []
        self._install()

    def _install(self):
        tm = self.tm
        L19, L24 = tm.layers[19], tm.layers[24]

        def _pre(_m, _i):
            self._h19_buf = []

        def _h19(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            self._h19_buf.append(h[:, -1, :].reshape(-1).detach().to(torch.float32).cpu())

        def _h24(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            self._h24_last = h[:, -1, :].reshape(-1).detach().to(torch.float32).cpu()

        def _post(module, _i, o):
            try:
                lhs = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
            except Exception:
                lhs = None
            if lhs is None or lhs.shape[1] > 1:
                return  # Prefill verwerfen
            self._tok += 1
            # letzter recur-Visit dieses Forwards; None falls keine Visits (BASELINE)
            h19_last = self._h19_buf[-1] if self._h19_buf else None
            if h19_last is not None:
                self.h19.append(h19_last)
            else:
                self.h19.append(torch.zeros(1152, dtype=torch.float32))
            self.h24.append(getattr(self, "_h24_last", torch.zeros(1152, dtype=torch.float32)))
            self.telem.append({
                "token_idx": self._tok,
                "loops_run": int(getattr(module, "_px_loops_run", 0)),
                "zone": str(getattr(module, "_px_zone", "NO_PX")),
                "phi_val": float(getattr(module, "_px_phi_val", 0.0)) if getattr(module, "_px_phi_val", None) is not None else 0.0,
                "ent_val": float(getattr(module, "_px_ent_val", 0.0)) if getattr(module, "_px_ent_val", None) is not None else 0.0,
            })

        self._handles = [
            tm.register_forward_pre_hook(_pre),
            L19.register_forward_hook(_h19),
            L24.register_forward_hook(_h24),
            tm.register_forward_hook(_post),
        ]
        self._h24_last = torch.zeros(1152, dtype=torch.float32)

    def reset(self):
        self.h19 = []; self.h24 = []; self.telem = []; self._h19_buf = []; self._tok = -1

    def remove(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass

    def stack(self):
        h19 = torch.stack(self.h19) if self.h19 else torch.empty(0, 1152)
        h24 = torch.stack(self.h24) if self.h24 else torch.empty(0, 1152)
        return h19, h24, self.telem


def run(model, tok, max_new=MAX_NEW):
    tm = _resolve_text_model(model)
    cap = S7.LoopCap(tm)
    hcap = HiddenCapture(tm)
    out = []
    sys_msgs = [{"role": "system", "content": S7.SYSTEM_PROMPT}]

    # BASELINE zuerst
    A.setup_baseline(model)
    for pid, ptext in S8.DEEPER_PROMPTS:
        cap.reset(); hcap.reset(); _clear()
        try:
            text = _greedy_generate(model, tok,
                sys_msgs + [{"role": "user", "content": ptext}], max_new, seed=SEED)
        except Exception as e:
            text = f"<GEN_ERROR {e}>"; print(f"[s9] ERR BASELINE/{pid}: {e}", file=sys.stderr)
        h19, h24, telem = hcap.stack()
        _save_cell("BASELINE", pid, h19, h24, telem, text)
        st = S7._cell_stats(cap.per_token)
        out.append(dict(arm="BASELINE", pid=pid, text=text, n_tok=int(h19.shape[0]),
                        loops_mean=st["loops_mean"], avg_distinct=st["avg_distinct"],
                        phi_mean=st["phi_mean"], retract_off=S7._retraction(text)))
        print(f"[s9] {'BASELINE':12s} {pid:14s} ntok={h19.shape[0]:4d} "
              f"loops={st['loops_mean']:.2f} retract@{out[-1]['retract_off']}", file=sys.stderr)

    # LEAN_DEFAULT + LEAN_WIDTH
    A.setup_lean(model, MODEL_ID)
    for arm_name, routing, _bl in ARMS:
        if arm_name == "BASELINE": continue
        for pid, ptext in S8.DEEPER_PROMPTS:
            S7.apply_hybrid(model, routing)
            for k in ("PX_LOOPS_CAP", "PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            cap.reset(); hcap.reset(); _clear()
            try:
                text = _greedy_generate(model, tok,
                    sys_msgs + [{"role": "user", "content": ptext}], max_new, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s9] ERR {arm_name}/{pid}: {e}", file=sys.stderr)
            for k in ("PX_LOOPS_CAP", "PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            h19, h24, telem = hcap.stack()
            _save_cell(arm_name, pid, h19, h24, telem, text)
            st = S7._cell_stats(cap.per_token)
            out.append(dict(arm=arm_name, pid=pid, text=text, n_tok=int(h19.shape[0]),
                            loops_mean=st["loops_mean"], avg_distinct=st["avg_distinct"],
                            phi_mean=st["phi_mean"], retract_off=S7._retraction(text)))
            print(f"[s9] {arm_name:12s} {pid:14s} ntok={h19.shape[0]:4d} "
                  f"loops={st['loops_mean']:.2f} dist={st['avg_distinct']:.1f} "
                  f"retract@{out[-1]['retract_off']}", file=sys.stderr)
    cap.remove(); hcap.remove()
    return out


def _save_cell(arm, pid, h19, h24, telem, text):
    os.makedirs(HID, exist_ok=True)
    path = os.path.join(HID, f"{arm}__{pid}.pt")
    torch.save({
        "arm": arm, "pid": pid, "text": text,
        "h19": h19.contiguous(),      # [n_tok, 1152] fp32
        "h24": h24.contiguous(),      # [n_tok, 1152] fp32
        "telem": telem,               # list[dict]
    }, path)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(HID, exist_ok=True)
    print("[s9] lade modell", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    out = run(model, tok, max_new=args.max_new)
    del model, tok; _clear()

    with open(os.path.join(OUT, "seite9_index.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arm_order = [a[0] for a in ARMS]
    with open(os.path.join(OUT, "seite9_texts.md"), "w", encoding="utf-8") as f:
        f.write("# Seite 9 — Decoder-Probe Capture (Rohtexte für manuelle Richness-Labels)\n\n")
        f.write("3 Arme × 10 DEEPER_PROMPTS × 300 tok. Hidden h19/h24 + telemetry auf Disk\n")
        f.write("(out/seite9_hidden/*.pt). Texte hier für manuelle Richness-Labels\n")
        f.write("(rich/poor) pro Zelle für Probe B. BASELINE/DEFAULT-Labels aus LESUNG9\n")
        f.write("wiederverwendbar (gleiche Prompts, greedy deterministisch → gleicher Text\n")
        f.write("falls max_new übereinstimmt; seite8 war 400, hier 300 → Texte kürzer,\n")
        f.write("neu zu labeln). WIDTH neu zu labeln.\n\n---\n\n")
        for pid, ptext in S8.DEEPER_PROMPTS:
            f.write(f"# === {pid} ===\n")
            f.write(f"PROMPT: {ptext}\n\n")
            rs = {r["arm"]: r for r in out if r["pid"] == pid}
            for arm in arm_order:
                r = rs.get(arm)
                if r is None: continue
                f.write(f"## [{arm}] pid={r['pid']} ntok={r['n_tok']} "
                        f"loops={r['loops_mean']:.2f} dist={r['avg_distinct']:.1f} "
                        f"phi={r['phi_mean']:.3f} retract@{r['retract_off']}\n")
                f.write(r["text"] + "\n\n")

    print(f"[s9] FERTIG -> {len(out)} cells, hidden in {HID}", file=sys.stderr)


if __name__ == "__main__":
    main()