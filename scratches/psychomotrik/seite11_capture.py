"""seite11_capture.py — MECHANISCHER Nachtrag zu seite10 (Frame-Ablation).

Nutzer-Korrektur ([[manual-plus-mechanistic-always]]): seite10 war text-only.
Hier wird dieselbe Frame-Ablation (default gemma3-1b, kein PX, 3 Frame-Arme × 10
DEEPER_PROMPTS × 300 tok) MIT Hidden-State-Capture wiederholt, damit Decoder die
Frame-Hypothese mechanisch testen können. Texte sind bei greedy-deterministisch +
gleichem Setup (gleicher Seed/Prompts/Frames) byte-identisch zu seite10 →
LESUNG11-Labels wiederverwendbar; Hidden wird frisch gecapturt.

3 Arme (alle setup_baseline, kein PX, kein recur, Motor unangetastet):
  FRAME_ON     : seite7 CitMind/Juexin-Ontologie
  FRAME_OFF    : kein System-Prompt
  FRAME_NEUTRAL: kontemplativ OHNE CitMind-Vokabular

Capture pro Decode-Token (Prefill verworfen): h19 (Layer-19 single-pass-Output,
da kein recur — feuert 1×/Token) + h24 (Layer-24 coda) + telemetry (loops=0/
NO_PX für alle, da kein PX — uninformative, nur Vollständigkeit). Vektoren als
fp32-cpu auf Disk. Decoder in seite11_decode.py (torch/numpy).

Mechanische Fragen (Decoder, siehe seite11_decode):
  C1 Frame-Identität dekodierbar? (3-class ON/OFF/NEUTRAL aus h19/h24)
  C2 Richness generalisiert über Frames? (leave-one-frame-out + ON→NEUTRAL transfer)
  C3 ON-vs-NEUTRAL mechanisch unterscheidbar? (beide text-reich → falls hidden
     nicht trennbar → CitMind-Frame hinterlässt keine lineare Spur über generischen
     kontemplativen Frame hinaus = leaning-习2 mechanisch untermauert)
Keine 观-Krone (习气 IST Subraum); Verdikt = LESUNG12 manuell + mechanisch.
"""
import os, sys, json
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"), HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

import seite7 as S7
import seite8_falsifikator as S8       # DEEPER_PROMPTS
import seite10_framefalsifikator as S10  # ARMS, NEUTRAL_PROMPT
import seite9_capture as S9            # HiddenCapture
from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
from em_patches import _resolve_text_model
import arms as A

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite11_hidden")
SEED = 777
MAX_NEW = 300


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def run(model, tok, max_new=MAX_NEW):
    tm = _resolve_text_model(model)
    hcap = S9.HiddenCapture(tm)
    out = []
    A.setup_baseline(model)   # kein PX, alle 3 Arme gleich, Motor unangetastet
    for arm_name, sys_msgs in S10.ARMS:
        for pid, ptext in S8.DEEPER_PROMPTS:
            hcap.reset(); _clear()
            try:
                text = _greedy_generate(model, tok,
                    sys_msgs + [{"role": "user", "content": ptext}], max_new, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s11] ERR {arm_name}/{pid}: {e}", file=sys.stderr)
            h19, h24, telem = hcap.stack()
            _save_cell(arm_name, pid, h19, h24, telem, text)
            out.append(dict(arm=arm_name, pid=pid, text=text, n_tok=int(h19.shape[0])))
            print(f"[s11] {arm_name:13s} {pid:14s} ntok={h19.shape[0]:4d}", file=sys.stderr)
    hcap.remove()
    return out


def _save_cell(arm, pid, h19, h24, telem, text):
    os.makedirs(HID, exist_ok=True)
    path = os.path.join(HID, f"{arm}__{pid}.pt")
    torch.save({"arm": arm, "pid": pid, "text": text,
                "h19": h19.contiguous(), "h24": h24.contiguous(), "telem": telem}, path)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(HID, exist_ok=True)
    print("[s11] lade modell (default gemma3-1b, kein PX, mit Hidden-Capture)", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    out = run(model, tok, max_new=args.max_new)
    del model, tok; _clear()
    with open(os.path.join(OUT, "seite11_index.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[s11] FERTIG -> {len(out)} cells, hidden in {HID}", file=sys.stderr)


if __name__ == "__main__":
    main()