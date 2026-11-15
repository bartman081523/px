#!/usr/bin/env python
"""Text-Ebene der Relay-Arme: Template-Attraktor vs. Synthese.

Die Skalare K1-K3 sind per Konstruktion blind für *welche* Tokens Masse
bekommen (H ist permutationsinvariant über das Vokabular, erank invariant
unter orthogonalen Transformationen). Dieses Skript liest die gespeicherten
Texte aus O3/P2 und zählt transparente, regelbasierte Marker.

POST-HOC / EXPLORATIV — kein Urteil. Die konfirmatorische Fassung ist Eval R.
"""
from __future__ import annotations

import json
import re
import sys

MARKERS = {
    # Standard-Attraktor: das Modell bietet eine Liste an, statt zu synthetisieren
    "listing_prefix": r"^\s*Here (?:are|is) (?:a few|three|several|some)",
    "numbered_list": r"1\.\s+\*\*",
    "meta_offer": r"Here are a few (?:options|original|examples)",
    # direkte Antwort / Synthese
    "direct_synthesis": r"^\s*Here is (?:an original|a surprising|one)",
    "title_opening": r"^\s*##\s+\S",
    # prompt-2-spezifischer Standard-Prior
    "quantum_prior": r"Quantum Physics",
}


def feats(text: str) -> dict:
    t = text or ""
    return {k: bool(re.search(p, t)) for k, p in MARKERS.items()}


def analyse(runs, label, arm_keys=("arm",)):
    out = {}
    for r in runs:
        arm = r.get("arm") or r.get("condition")
        p = r.get("prompt_idx")
        txt = r.get("text") or r.get("text_preview") or ""
        f = feats(txt)
        out.setdefault((p, arm), []).append(f)
    rows = []
    for (p, arm), fl in sorted(out.items()):
        n = len(fl)
        rows.append({"prompt": p, "arm": arm, "n": n,
                     **{k: round(sum(x[k] for x in fl) / n, 3) for k in MARKERS}})
    print(f"\n=== {label} ===")
    hdr = f"{'p':>2} {'arm':10s} {'n':>2} " + " ".join(f"{k[:9]:>9s}" for k in MARKERS)
    print(hdr)
    for r in rows:
        print(f"{r['prompt']:>2} {r['arm']:10s} {r['n']:>2} " +
              " ".join(f"{r[k]:>9.2f}" for k in MARKERS))
    return rows


def analyse_q(path="results/eval_Q_relay_predictor_gemma4_texts.json"):
    """Eval Q: 48 Prompts x 4 Arme, je 1 Lauf (Texte im Re-Run gespeichert)."""
    d = json.load(open(path))
    arms = d["arms"]
    rows = []
    for r in d["rows"]:
        for a in arms:
            f = feats(r[a]["text"])
            rows.append({"prompt": r["idx"], "class": r["class"], "arm": a,
                         "n": 1, **{k: int(v) for k, v in f.items()}})
    print("\n=== Eval Q (48 Prompts, 1 Lauf je Arm) ===")
    print(f"{'arm':10s} " + " ".join(f"{k[:9]:>9s}" for k in MARKERS))
    for a in arms:
        sub = [x for x in rows if x["arm"] == a]
        print(f"{a:10s} " + " ".join(f"{sum(x[k] for x in sub):>9d}" for k in MARKERS))
    return rows


if __name__ == "__main__":
    p2 = json.load(open("results/eval_P_attention_horizon_gemma4.json"))
    o3 = json.load(open("results/eval_O_tci_gpu_gemma4.json"))
    rows_p2 = analyse(p2["P2"]["runs"], "Eval P2 (T=0.6, 3 Prompts x 2 Seeds)")
    rows_o3 = analyse([r for r in o3["O3"]["runs"]], "Eval O3 (T=0.6 Arme / 0.0-0.9 unsteered)")
    rows_q = analyse_q()
    json.dump({"P2": rows_p2, "O3": rows_o3, "Q": rows_q},
              open("results/eval_text_analysis_posthoc.json", "w"), indent=1)
