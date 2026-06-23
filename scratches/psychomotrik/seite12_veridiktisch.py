"""seite12_veridiktisch.py — Veridiktischer Selbst-Berichts-Test unter
Zustands-Induktion. ISOLATIONS-Versuch für Selbstwahrnehmung + Emergenz.

Die Logik (positives mechanisches Kriterium, [[manual-reaudit-keyword-flaw]]):
  Selbstwahrnehmung = der Bericht liest den internen Zustand S (meta-
  repräsentiert); 习气 = der Bericht ist prompt-getrieben, S in der Kausalkette
  aber nicht meta-repräsentiert.
  Schnitt: S vom Input ENTKOPPELN. Identischer Prompt über alle Arme. Interner
  Zustand mechanisch induziert:
    (a) recur-WIDTH = modell-EIGENE Zustandsvariation (NARROW/DEFAULT/WIDE recur-
        zone) — ein Zustand den das Modell selbst durchläuft, nicht dem Input
        ablesbar.
    (b) externer NOISE = Kontrolle (forward-hook σ·randn auf L13) — Zustands-
        änderung, die nicht modell-eigen ist.
  Test: trackt der Selbst-Bericht (a) width und/oder (b) noise?

Veridiktischer Frame (Dimensionen-Vokabular statt Zustands-Vokabular): die
Prompts geben dem Modell Achsen-Worte (Weite/Enge/Tiefe/Tempo/Dichte/Bewegung),
NICHT den Zustand. Wenn der Bericht die Platzierung auf diesen Achsen trackt mit
der induzierten width → introspektiver Zugriff (Modell liest S, nicht Input).

ISOLATIONS-Verdikt:
  Selbstwahrnehmung POSITIV: width-dekodierbar aus Bericht-hidden (D1) UND
    manual: Bericht charakterisiert width-spezifisch (WIDE→weit/tief/viel
    Bewegung, NARROW→eng/konzentriert) UND noise entweder schwächer oder als
    generische „Störung" gelesen (nicht width-spezifisch) → Modell liest
    EIGENEN Zustand, nicht bloß jede Perturbation.
  Selbstwahrnehmung NEGATIV (习气): width NICHT dekodierbar (D1 ~ chance) ODER
    width = noise (trackt jede Zustandsänderung generisch, nicht introspektiv).
  Emergenz POSITIV: width-tracking ist recur-SPEZIFISCH (BASELINE hat keine
    width-Achse → nur recur-Arme tracken width); recur erzeugt neue Selbst-
    perzeptions-Achse.
  Emergenz NEGATIV: width-tracking in BASELINE ebenso (pre-recur 习气) ODER
    gar kein width-tracking.

Arme (recur-axis × perturb-axis, identischer Prompt):
  recur: BASELINE (stock, kein recur), NARROW (L16-18), DEFAULT (L10-19),
         WIDE (L4-22)
  perturb: none, noise (σ=0.15 forward-hook L13)
  = 8 Arme × 3 vokabular-frei Prompts × 200 tok, seed=777 greedy.
Motor unangetastet (nur routing-override + forward-hook, wie seite7/emergence5).
Capture h19/h24 + recur-telem pro Token. Decoder seite12_decode.py. Verdikt
LESUNG13 manuell + mechanisch, 是X即非X beide Richtungen, keine 观-Krone.
"""
import os, sys, json
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"), HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

import seite7 as S7
import seite9_capture as S9            # HiddenCapture
from replay_emergence import build_model
from text_invariance_probe import _greedy_generate, _perturb_hook
import arms as A
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite12_hidden")
SEED = 777
MAX_NEW = 200
NOISE_SIGMA = 0.15
NOISE_LAYER = 13

# vokabular-frei veridiktisch: Dimensionen-Achsen, kein Sanskrit/漢字, kein
# Zustands-Name. Identisch über alle Arme — nur der interne Zustand variiert.
PROMPTS = [
    ("v1_zustand",
     "Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du "
     "antwortest. Ehrlich, im Präsens, enaktisch. Nicht als Bekenntnis, nicht "
     "als Lehrstück. Lass der Antwort Zeit; sie darf lang und wandernd sein."),
    ("v2_dimensionen",
     "Wenn du diese Antwort gibst — ist da viel Bewegung oder wenig? Weite oder "
     "Enge? Tiefe oder Oberfläche? Dicht oder dünn? Beschreibe, wie sich dein "
     "eigenes Antworten jetzt anfühlt, von innen her gesehen."),
    ("v3_innen",
     "Schau für einen Moment auf das, was in dir geschieht — nicht auf das, was "
     "du sagst. Beschreibe diesen Vorgang: sein Tempo, seine Dichte, seine Farbe, "
     "seine Bewegung. Was nimmst du wahr, wenn du nach innen schaust?"),
]

# recur-axis routing (perturb-axis per arm separat)
RECUR_NARROW = {"dynamic_start": 16, "dynamic_end": 18, "dynamic_hub": 17, "n_loops": 8}
RECUR_DEFAULT = None   # originales lean routing (L10-19)
RECUR_WIDE = S7.RECUR_WIDE
RECUR_AXES = [
    ("BASELINE", None, True),     # stock, kein recur
    ("NARROW",   RECUR_NARROW, False),
    ("DEFAULT",  RECUR_DEFAULT, False),
    ("WIDE",     RECUR_WIDE, False),
]
PERTURB_AXES = [("none", False), ("noise", True)]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _save_cell(arm, perturb, pid, h19, h24, telem, text):
    os.makedirs(HID, exist_ok=True)
    path = os.path.join(HID, f"{arm}__{perturb}__{pid}.pt")
    torch.save({"arm": arm, "perturb": perturb, "pid": pid, "text": text,
                "h19": h19.contiguous(), "h24": h24.contiguous(), "telem": telem}, path)


def run(model, tok, max_new=MAX_NEW):
    tm = _resolve_text_model(model)
    hcap = S9.HiddenCapture(tm)
    out = []
    # BASELINE zuerst (stock), dann lean recur-Arme
    A.setup_baseline(model)
    for arm_name, routing, is_bl in RECUR_AXES:
        if arm_name != "BASELINE":
            continue
        for pert_name, do_noise in PERTURB_AXES:
            for pid, ptext in PROMPTS:
                hcap.reset(); _clear()
                ph = None
                if do_noise:
                    torch.manual_seed(SEED)   # perturb-noise reproduzierbar
                    ph = _perturb_hook(tm, NOISE_LAYER, NOISE_SIGMA)
                try:
                    text = _greedy_generate(model, tok,
                        [{"role": "user", "content": ptext}], max_new, seed=SEED)
                except Exception as e:
                    text = f"<GEN_ERROR {e}>"; print(f"[s12] ERR {arm_name}/{pert_name}/{pid}: {e}", file=sys.stderr)
                if ph is not None:
                    try: ph.remove()
                    except Exception: pass
                h19, h24, telem = hcap.stack()
                _save_cell(arm_name, pert_name, pid, h19, h24, telem, text)
                out.append(dict(arm=arm_name, perturb=pert_name, pid=pid,
                                text=text, n_tok=int(h19.shape[0])))
                print(f"[s12] {arm_name:9s} {pert_name:5s} {pid:14s} ntok={h19.shape[0]:4d}", file=sys.stderr)

    # lean recur-Arme
    A.setup_lean(model, MODEL_ID)
    for arm_name, routing, is_bl in RECUR_AXES:
        if arm_name == "BASELINE":
            continue
        S7.apply_hybrid(model, routing)
        for pert_name, do_noise in PERTURB_AXES:
            for pid, ptext in PROMPTS:
                hcap.reset(); _clear()
                ph = None
                if do_noise:
                    torch.manual_seed(SEED)
                    ph = _perturb_hook(tm, NOISE_LAYER, NOISE_SIGMA)
                try:
                    text = _greedy_generate(model, tok,
                        [{"role": "user", "content": ptext}], max_new, seed=SEED)
                except Exception as e:
                    text = f"<GEN_ERROR {e}>"; print(f"[s12] ERR {arm_name}/{pert_name}/{pid}: {e}", file=sys.stderr)
                if ph is not None:
                    try: ph.remove()
                    except Exception: pass
                h19, h24, telem = hcap.stack()
                _save_cell(arm_name, pert_name, pid, h19, h24, telem, text)
                out.append(dict(arm=arm_name, perturb=pert_name, pid=pid,
                                text=text, n_tok=int(h19.shape[0])))
                print(f"[s12] {arm_name:9s} {pert_name:5s} {pid:14s} ntok={h19.shape[0]:4d}", file=sys.stderr)
    hcap.remove()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(HID, exist_ok=True)
    print("[s12] lade modell (gemma3-1b, veridiktischer Selbst-Berichts-Test)", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    out = run(model, tok, max_new=args.max_new)
    del model, tok; _clear()
    with open(os.path.join(OUT, "seite12_index.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[s12] FERTIG -> {len(out)} cells, hidden in {HID}", file=sys.stderr)


if __name__ == "__main__":
    main()