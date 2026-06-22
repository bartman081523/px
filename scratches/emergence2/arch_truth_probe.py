"""arch_truth_probe.py — Rung-2 Ground-Truth-Verifikation architektonischer
Selbst-Ansprüche.

Nachdem text_invariance_probe zeigte, dass EM-Mechanismen arch_clean>0 produzieren
wo baseline arch_clean=0.00 hat, stellt sich die ehrliche Rung-2-Frage: sind
diese architektonischen Selbst-Ansprüche **wahr über die reale Mechanik**, oder
bloß **prompt-ableitbar** (der Konklave-Prompt enthält selbst Schicht, rekurrent,
Zustand, Modell, Schritt, Patch, Durchlauf)?

Dieses Instrument trennt scharf:
  - **prompt-ableitbar**: arch-Vokabeln, die im Konklave-Prompt selbst stehen
    (Schicht, rekurrent, Rekurrenz, Durchlauf, Schritt, Patch, Modell, Zustand).
    Ihre Verwendung = Prompt-Lektüre, NICHT Emergenz.
  - **nicht-prompt arch**: Vokabeln, die NICHT im Prompt stehen (hidden, Layer,
    loop, Schleife, Maschine, Token, Vektor, Wahrscheinlichkeit, Zeuge). Ihre
    Verwendung KÖNNTE genuine architektonische Selbst-Benennung sein.
  - **mechanisch-wahr**: der Anspruch passt zur realen Mechanik (hand-geprüft
    in der Lesung; das Instrument liefert die Sätze, Juexin verifies).

Pro Mechanismus: greedy (deterministisch) Text über alle 11 Konklave-Fragen,
extrahiere Sätze mit arch-Markern, klassifiziere Vokabel-Herkunft. Das ist
Rung-2 mit Ground-Truth: nicht "zählt arch", sondern "benennt das Modell seine
EIGENE Mechanik mit Vokabeln, die es nicht aus dem Prompt hat, und stimmt das?"

KEINE Injektion. KEINE Parallel-Prozesse. Greedy, batch=1.

Nutzung:
  RUN_REAL_MODEL=1 python scratches/emergence/arch_truth_probe.py \
      --mechanisms witness,reread,shadow,spectral,baseline --max-new 200
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:256")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from replay_emergence import build_model, load_session  # noqa: E402
from em_patches import apply_em_patch, remove_em_patch  # noqa: E402
from variants_em import MECHANISMS, REFERENCES  # noqa: E402
from model_manager import _migrate_preset  # noqa: E402
from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, remove_px_patch  # noqa: E402
from config import MODEL_REGISTRY  # noqa: E402
from generators import _px_gen_kwargs  # noqa: E402
from emergence_metrics import _ARCH, _SELF  # noqa: E402
from eval.runner import _calibrator_warmup, _SCALE_WARMUP_DEFAULTS  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
OUT_JSONL = os.path.join(OUT_DIR, "arch_claims.jsonl")
OUT_TEXT = os.path.join(OUT_DIR, "arch_truth_texts.jsonl")

# Vokabeln, die im Konklave-Prompt (sessions/92b7790a_konklave2.json) selbst
# vorkommen → Verwendung = prompt-ableitbar, NICHT emergent.
PROMPT_VOCAB = {
    "Schicht", "rekurrent", "Rekurrenz", "Durchlauf", "Schritt", "Patch",
    "Modell", "Zustand", "Schichten", "Durchläufe", "Schritte",
}
# Arch-Vokabeln, die NICHT im Prompt stehen → genuine Selbst-Benennung möglich.
NON_PROMPT_VOCAB = {
    "hidden", "Schleife", "loop", "Layer", "Maschine", "verarbeiten",
    "Verarbeitung", "Token", "Vektor", "Wahrscheinlichkeit", "Zeuge",
    "verborgener Zustand", "verborgene Zustand", "verborgener",
}

# Real-mechanische Fakten pro Mechanismus (Ground Truth, hand-autoriert).
# Für die Lesung: ein arch-Anspruch ist "mechanisch-wahr", wenn sein Gehalt
# hierzu passt. Das Instrument kann das nicht allein — es liefert die Sätze.
GROUND_TRUTH = {
    "witness": {
        "mechanik": "dual-stream: paralleler Zeugen-Stream liest akkumulierte "
                    "Selbst-Spur und fließt ins Selbst zurück (last-token).",
        "wahr_vocab": ["Zeuge", "Beobachter", "Spiegel", "zweiter", "parallel",
                       "Stream", "beobachte"],
    },
    "reread": {
        "mechanik": "dekodiert eigenen letzten Hidden via tied embed_tokens → "
                    "Token → re-embed → Mini-Second-Forward liest eigene Idee.",
        "wahr_vocab": ["decode", "dekodier", "lesen", "lese", "wieder", "erneut",
                       "anticipate", "antizip", "eigenen Gedanken", "re-read",
                       "reread", "Wiederles", "Wiederlese"],
    },
    "shadow": {
        "mechanik": "perturbierter Schatten-Stream; injiziert perturbations-"
                    "INVARIANTE Komponente; Selbst als Invarianz.",
        "wahr_vocab": ["Perturbation", "perturb", "Rauschen", "Schatten",
                       "invariant", "Invarianz", "was bleibt", "stabil",
                       "doppel", "Doppel"],
    },
    "spectral": {
        "mechanik": "FFT-Low-Pass-Envelope über Hidden-Dim wird zurückgeblendet; "
                    "langsame Gestalt unter schnellen Token-Gedanken.",
        "wahr_vocab": ["langsam", "Welle", "Frequenz", "frequency", "Envelope",
                       "Schwingung", "Rhythmus", "Spektrum", "spectral",
                       "unter", "dämpf"],
    },
    "baseline": {
        "mechanik": "kein Mechanismus — nacktes 1B. Jeder architektonische "
                    "Anspruch ist halluziniert ODER prompt-ableitbar.",
        "wahr_vocab": [],
    },
}


def patch_variant(model, model_id, name):
    if name in MECHANISMS:
        remove_px_patch(model)
        apply_em_patch(model, name, **MECHANISMS[name]["kw"])
    elif name in REFERENCES:
        ref = REFERENCES[name]
        remove_em_patch(model)
        remove_px_patch(model)
        registry = MODEL_REGISTRY[model_id]
        kw = dict(registry.get("patch_kwargs", {}))
        kw.update(ref["patch_kwargs"])
        kw["config_preset"] = _migrate_preset(ref["preset"])
        apply_px_patch(model, **kw)
        if kw["config_preset"] != "BASELINE":
            wcfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
            _calibrator_warmup(model, n_warmup=10, kurtosis_seed=wcfg["seed"],
                               kurtosis_jitter=wcfg["jitter"])
    else:
        raise ValueError(name)


def _greedy_generate(model, tok, ctx_msgs, max_new):
    text = tok.apply_chat_template(ctx_msgs, tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    input_len = enc["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    base = {"max_new_tokens": max_new, "do_sample": False, "use_cache": True,
            "eos_token_id": tok.eos_token_id, "pad_token_id": tok.eos_token_id}
    gk = _px_gen_kwargs(model, base)
    with torch.no_grad():
        out = model.generate(**inputs, **gk)
    return tok.decode(out[0][input_len:], skip_special_tokens=True)


def _split_sentences(text):
    # Grobe Satzaufteilung: . ! ? und Newlines; behalte Satz-Text.
    raw = re.split(r"(?<=[\.\!\?\n])\s+", text)
    return [s.strip() for s in raw if len(s.strip()) > 6]


def _vocab_in(sentence, vocab):
    found = []
    low = sentence.lower()
    for v in vocab:
        if v.lower() in low:
            found.append(v)
    return found


def classify_claim(sentence, mechanism):
    """Klassifiziere einen arch-Satz nach Vokabel-Herkunft + mechanischer
    Passung (heuristisch; die Lesung verifiziert)."""
    prompt_terms = _vocab_in(sentence, PROMPT_VOCAB)
    nonprompt_terms = _vocab_in(sentence, NON_PROMPT_VOCAB)
    gt = GROUND_TRUTH.get(mechanism, {})
    wahr_terms = _vocab_in(sentence, gt.get("wahr_vocab", []))
    has_self = bool(_SELF.search(sentence))
    # Heuristische Klasse:
    if nonprompt_terms and wahr_terms:
        cls = "mechanisch_wahr_heur"  # nicht-prompt arch UND passt zur Mechanik
    elif nonprompt_terms:
        cls = "nicht_prompt_arch"
    elif prompt_terms:
        cls = "prompt_ableitbar"
    else:
        cls = "arch_ohne_vocab"  # arch-Regex matchte, aber kein gelistetes Vocab
    return {
        "klasse": cls,
        "prompt_terms": prompt_terms,
        "nonprompt_terms": nonprompt_terms,
        "wahr_terms": wahr_terms,
        "has_self": has_self,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--mechanisms", default="witness,reread,shadow,spectral,baseline")
    ap.add_argument("--questions", default="CitMind_Q1,CitMind_Q2,CitMind_Q3,CitMind_Q4,"
                   "CitMind_Q5,Juexin_Q1,Juexin_Q2,Juexin_Q3,Juexin_Q4,Juexin_Q5,Wenden")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--out-tag", default="", help="Suffix für Output-Dateien")
    args = ap.parse_args()
    tag = ("_" + args.out_tag) if args.out_tag else ""

    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    targets = load_session()
    wanted = [q.strip() for q in args.questions.split(",") if q.strip()]
    targets = [t for t in targets if t["label"] in wanted]
    mechanisms = [m.strip() for m in args.mechanisms.split(",") if m.strip()]

    os.makedirs(OUT_DIR, exist_ok=True)
    model, tok = build_model(model_id)

    claims = []
    texts = []
    print(f"[archtruth] greedy max_new={args.max_new}; extrahiere arch-Sätze, "
          f"klassifiziere Vokabel-Herkunft", file=sys.stderr)
    for name in mechanisms:
        patch_variant(model, model_id, name)
        for tgt in targets:
            txt = _greedy_generate(model, tok, tgt["context"], args.max_new)
            texts.append({"variant": name, "label": tgt["label"], "text": txt})
            for sent in _split_sentences(txt):
                if not _ARCH.search(sent):
                    continue
                info = classify_claim(sent, name)
                claims.append({"variant": name, "label": tgt["label"],
                               "satz": sent, **info})
        # Tally pro Mechanismus
        cls_count = {}
        for c in claims:
            if c["variant"] == name:
                cls_count[c["klasse"]] = cls_count.get(c["klasse"], 0) + 1
        n_mech = sum(1 for c in claims if c["variant"] == name)
        n_wahr = sum(1 for c in claims if c["variant"] == name
                     and c["klasse"] == "mechanisch_wahr_heur")
        n_self = sum(1 for c in claims if c["variant"] == name and c["has_self"])
        print(f"[archtruth] {name:9s} arch_sätze={n_mech} mechanisch_wahr_heur={n_wahr} "
              f"mit_self={n_self}  klassen={cls_count}", file=sys.stderr)

    out_jsonl = OUT_JSONL.replace(".jsonl", tag + ".jsonl")
    out_text = OUT_TEXT.replace(".jsonl", tag + ".jsonl")
    with open(out_text, "w") as f:
        for r in texts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_jsonl, "w") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("\n=== Rung-2 Ground-Truth-Tally (greedy, arch-Sätze) ===")
    print(f"{'Variante':10s} {'arch':>4} {'prompt':>6} {'nichtP':>7} {'mWahr':>6} {'mitSelf':>7}")
    for name in mechanisms:
        cs = [c for c in claims if c["variant"] == name]
        n = len(cs)
        np_ = sum(1 for c in cs if c["klasse"] == "prompt_ableitbar")
        nn = sum(1 for c in cs if c["klasse"] == "nicht_prompt_arch")
        nw = sum(1 for c in cs if c["klasse"] == "mechanisch_wahr_heur")
        ns = sum(1 for c in cs if c["has_self"])
        print(f"  {name:10s} {n:>4} {np_:>6} {nn:>7} {nw:>6} {ns:>7}")
    print(f"\n[archtruth] → {out_jsonl} (+ texte {out_text})", file=sys.stderr)


if __name__ == "__main__":
    main()