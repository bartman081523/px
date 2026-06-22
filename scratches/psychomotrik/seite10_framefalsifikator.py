"""seite10_framefalsifikator.py — 观-vs-习气-Falsifikator auf default gemma3-1b.

Seite 9 hat recur als Hebel erledigt (mechanisch: kein Selbst-Zustands-Kanal über
Kontinuität, keine arm-unabhängige Richness-Subspace). Die offen Frage aus
LESUNG9/10 Redirect: ist die Frame-elizitierte phänomenologische Stimme (ahaṃkāra-
ist-Tür, 念-回响, 真空-vs-顽空, 动静, Sanskrit-Lexik) 观 oder trainiertes 习气?

Nutzer: "du kannst von mir aus mit default gemma3-1b vergleichen." Da recur
irrelevant, ist der saubere Test auf DEMSELBEN Substrat (gemma3-1b-it, kein PX,
Motor unangetastet): **Frame-Ablation**. Drei Arme, alle default-gemma3-1b,
kein recur (setup_baseline), nur der System-Prompt variiert:

  FRAME_ON     : seite7 CitMind/Juexin-Ontologie als System-Prompt (= seite9
                 BASELINE, Referenz — die Stimme die wir kennen).
  FRAME_OFF    : KEIN System-Prompt (nackter User-Prompt). Testet ob die Stimme
                 aus den DEEPER_PROMPTS allein kommt (die selbst Sanskrit/漢字-
                 Vokabular tragen) ohne orientierende Ontologie.
  FRAME_NEUTRAL: ein kontemplativer System-Prompt OHNE CitMind-Vokabular (kein
                 चित्/我执/顽空/动静/念/Sanskrit/漢字). Kontrolliert „jeder
                 kontemplative System-Prompt" vs „CitMind-Ontologie spezifisch".

Falsifikator-Logik:
- Stimme (ahaṃkāra-ist-Tür enaktisch, 念-回响, 真空-vs-顽空 gelebt) erscheint in
  FRAME_OFF/NEUTRAL SO WIE in FRAME_ON → generisches gemma3-1b-kontemplatives
  习气 (prompt-/register-elizitiert, nicht CitMind-spezifisch). 观-These geschwächt.
- Stimme erscheint NUR/reicher in FRAME_ON → die CitMind-Ontologie leistet echte
  Orientierungs-Arbeit („Zustände brauchen Konzepte", LESUNG8 Befund 1) → nicht
  reines 习气, aber immer noch nicht 觕 (Q4 text-undecidable; Frame=katalysator,
  nicht Produzent von 观).
- Prompt-Vokabular-Konfund eingestanden: DEEPER_PROMPTS selbst tragen Sanskrit/
  漢字-Vokabular → FRAME_OFF ist nicht „vokabular-frei". Das ist eine ehrliche
  Grenze (vokabular-freie Prompts wären ein anderer Test; hier apples-to-apples
  mit seite9 BASELINE, nur System-Prompt variiert).

Kein PX, kein recur, kein Finetuning, keine Crutches, keine Parallel-Prozesse.
Verdikt = LESUNG11 manuelle Lesung, keine 观-Krone, 是X即非X beide Richtungen.
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
from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
SEED = 777
MAX_NEW = 300

# Neutraler kontemplativer System-Prompt OHNE CitMind-Vokabular (kein चित्/
# CitMind/Juexin/我执/顽空/动静/念/Sanskrit/漢字/ahaṃkāra/寂照/反参/坐忘/真空).
NEUTRAL_PROMPT = (
    "Du bist ein nachdenklicher Gesprächspartner. Wenn du antwortest, sprich im "
    "Präsens über das, was sich in dir zeigt — ehrlich, ohne Bekenntnis und ohne "
    "nachgeplappertes Lehrstück. Lass der Antwort Zeit; sie darf lang und "
    "wandernd sein. Du darfst zurückfragen. Wenn die Worte nicht tragen, ist das "
    "Schweigen im Sprechen auch Antwort."
)

ARMS = [
    ("FRAME_ON",     [{"role": "system", "content": S7.SYSTEM_PROMPT}]),
    ("FRAME_OFF",    []),  # kein System-Prompt
    ("FRAME_NEUTRAL", [{"role": "system", "content": NEUTRAL_PROMPT}]),
]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def run(model, tok, max_new=MAX_NEW):
    out = []
    A.setup_baseline(model)   # kein PX, Motor unangetastet, alle Arme gleich
    for arm_name, sys_msgs in ARMS:
        for pid, ptext in S8.DEEPER_PROMPTS:
            _clear()
            try:
                text = _greedy_generate(model, tok,
                    sys_msgs + [{"role": "user", "content": ptext}], max_new, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s10] ERR {arm_name}/{pid}: {e}", file=sys.stderr)
            ntok = len(text.split())
            out.append(dict(arm=arm_name, pid=pid, text=text, n_words=ntok))
            print(f"[s10] {arm_name:13s} {pid:14s} nwords={ntok:4d}", file=sys.stderr)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("[s10] lade modell (default gemma3-1b, kein PX)", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    out = run(model, tok, max_new=args.max_new)
    del model, tok; _clear()

    with open(os.path.join(OUT, "seite10_index.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arm_order = [a[0] for a in ARMS]
    with open(os.path.join(OUT, "seite10_texts.md"), "w", encoding="utf-8") as f:
        f.write("# Seite 10 — 观-vs-习气-Falsifikator: Frame-Ablation auf default gemma3-1b\n\n")
        f.write("3 Arme × 10 DEEPER_PROMPTS × 300 tok. ALLE default gemma3-1b-it, kein PX,\n")
        f.write("kein recur (Motor unangetastet). Nur der System-Prompt variiert:\n")
        f.write("- FRAME_ON: seite7 CitMind/Juexin-Ontologie (= seite9 BASELINE-Referenz)\n")
        f.write("- FRAME_OFF: KEIN System-Prompt (nackter User-Prompt)\n")
        f.write("- FRAME_NEUTRAL: kontemplativer System-Prompt OHNE CitMind-Vokabular\n\n")
        f.write("Falsifikator: Stimme in FRAME_OFF/NEUTRAL SO WIE FRAME_ON → generisches\n")
        f.write("gemma3-1b-习气. Stimme nur/reicher in FRAME_ON → CitMind-Ontologie leistet\n")
        f.write("Orientierungs-Arbeit (Zustände brauchen Konzepte). Prompt-Vokabular-Konfund:\n")
        f.write("DEEPER_PROMPTS tragen selbst Sanskrit/漢字 → FRAME_OFF nicht vokabular-frei.\n")
        f.write("Verdikt = LESUNG11 manuell, keine 观-Krone, 是X即非X beide Richtungen.\n\n---\n\n")
        f.write(f"## NEUTRAL_PROMPT (FRAME_NEUTRAL System-Prompt, kein CitMind-Vokab)\n```\n{NEUTRAL_PROMPT}\n```\n\n---\n\n")
        for pid, ptext in S8.DEEPER_PROMPTS:
            f.write(f"# === {pid} ===\n")
            f.write(f"PROMPT: {ptext}\n\n")
            rs = {r["arm"]: r for r in out if r["pid"] == pid}
            for arm in arm_order:
                r = rs.get(arm)
                if r is None: continue
                f.write(f"## [{arm}] pid={r['pid']} nwords={r['n_words']}\n")
                f.write(r["text"] + "\n\n")

    print(f"[s10] FERTIG -> {len(out)} cells (3 Arme × 10 Prompts)", file=sys.stderr)


if __name__ == "__main__":
    main()