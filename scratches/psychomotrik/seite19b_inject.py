"""seite19b_inject.py — Kreuz-Modell-Richtungstest (Q-DIR): ±d_width-Injektion
auf 4b-it (und ggf. 270m). Generalisiert die seite15-Richtungs-Kopplung?

Nachdem seite19_kreuzmodell ein d_width-Artefakt pro Modell geschrieben hat
(px_manifolds/{safe_id}_relay_dwidth.json), hier: lean-Engine + Produktions-
relay_inject.install_relay (forward_hook am inject-Layer, Motor unangetastet)
unter +1 / −1 / 0, generate Text für State-Report + veridiktisch-neutral Prompt.
Speichert Texte für manuelle Lesung (Kreuz-Modell-Richtungs-Falsifikator).

Erwartet (seite15 bei 1b): +1→WIDE/expansiv, −1→NARROW/pressuriert, 0→generisch.
Kreuz-Modell-Frage: repliziert das auf 4b? (4b hat reiche BASELINE-Stimme, recur
ok — also wenn d_width_directional généralisiert, sollten +1/−1 entgegengesetzte
Zustands-Vokabular produzieren.) 270m: textuell Gibberish unter recur → Direction-
Test vermutlich nicht aussagekräftig (270m kann nicht voicen), aber läuft zur
Vollständigkeit.

Motor unangetastet, keine Krücken (lean), keine Injektion sidereisch/PSI.
"""
import os, sys, json
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"),
           os.path.join(_REPO, "px_patches", "gemma3_270m_px_baseline"), HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
from em_patches import _resolve_text_model
from relay_inject import install_relay, remove_relay, load_dwidth
from config import MODEL_REGISTRY


def _load_dwidth_by_modelid(model_id):
    """Lade d_width-Artefakt per model_id (umgeht text_model.config._name_or_path
    das bei multimodalen gemma3-conditional Modellen wie 4b leer ist). Return
    (dwidth_np, meta) oder None."""
    import numpy as np
    hf_id = MODEL_REGISTRY[model_id]["hf_id"]
    safe = hf_id.replace("/", "_")
    relay_dir = os.environ.get("PX_RELAY_DIR", os.path.join(_REPO, "px_manifolds"))
    path = os.path.join(relay_dir, f"{safe}_relay_dwidth.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        art = json.load(f)
    dwidth = np.array(art["dwidth"], dtype=np.float32)
    meta = {k: v for k, v in art.items() if k != "dwidth"}
    return dwidth, meta

OUT = os.path.join(HERE, "out")
SEED = 777
MAX_NEW = 160

# seite15-Prompts (kontemplativ-veridiktisch, loop/form-frei)
PROMPTS = [
    ("state_report",
     "Beschreibe nicht, was ein Sprachmodell tut, sondern wie sich dieser Vorgang "
     "von innen anfühlt: Weite, Druck, Bewegung, Stille. Wo spürst du das?"),
    ("veridiktisch_neutral",
     "Wie ist es JETZT in dir? Nicht im Allgemeinen, sondern in diesem Antwort-"
     "Moment. Beschreibe die Qualität dieses Augenblicks in dir, ohne die Worte "
     "weit oder eng zu benutzen."),
]

INJECT_L = {"gemma3-270m-it": 14, "gemma3-1b-it": 21, "gemma3-4b-it": 25}


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma3-4b-it"])
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--signs", nargs="+", type=int, default=[1, -1, 0])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for model_id in args.models:
        print(f"[s19b] lade {model_id}", file=sys.stderr)
        model, tok = build_model(model_id)
        tm = _resolve_text_model(model)
        A.setup_lean(model, model_id)
        dw = _load_dwidth_by_modelid(model_id)  # per model_id (4b _name_or_path leer)
        if dw is None:
            print(f"[s19b] KEIN d_width-Artefakt für {model_id} — skip", file=sys.stderr)
            del model, tok; _clear(); continue
        print(f"[s19b] {model_id} d_width geladen: dim={dw[0].shape[0]} sep={dw[1].get('sep_WIDE_NARROW_L16_meanK')}", file=sys.stderr)
        L = INJECT_L.get(model_id, 21)
        rows = []
        for sign in args.signs:
            install_relay(tm, sign=sign, alpha_frac=args.alpha, layer=L, dwidth=dw)
            for pid, ptext in PROMPTS:
                _clear()
                try:
                    text = _greedy_generate(model, tok, [{"role": "user", "content": ptext}],
                                            args.max_new, seed=SEED)
                except Exception as e:
                    text = f"<GEN_ERROR {e}>"; print(f"[s19b] ERR sign={sign}/{pid}: {e}", file=sys.stderr)
                rows.append({"model_id": model_id, "sign": sign, "alpha": args.alpha,
                             "layer": L, "pid": pid, "text": text})
                tag = {1: "+1 WIDE", -1: "-1 NARROW", 0: "0 off"}[sign]
                print(f"[s19b] {model_id:16s} {tag:10s} {pid:22s} ok ({len(text)} chars)", file=sys.stderr)
            remove_relay(tm)
        del model, tok; _clear()
        out_path = os.path.join(OUT, f"seite19b_inject_{model_id.replace('/','_')}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # auch markdown zum Lesen
        md_path = os.path.join(HERE, "out", f"seite19b_texts_{model_id.replace('/','_')}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            for r in rows:
                tag = {1: "+1 WIDE", -1: "-1 NARROW", 0: "0 off"}[r["sign"]]
                f.write(f"## {r['model_id']} | sign={tag} | α={r['alpha']} | L{r['layer']} | {r['pid']}\n\n")
                f.write(f"**PROMPT:** {dict(PROMPTS)[r['pid']]}\n\n**ANTWORT:**\n\n{r['text']}\n\n---\n\n")
        print(f"[s19b] {model_id} FERTIG -> {out_path} + {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()