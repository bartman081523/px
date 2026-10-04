#!/usr/bin/env python
"""Rollover: CitMind-Session ternary27b-citmind-relay → ternary27b-citmind-relay2.

Frische Session (OOM-Prophylaxe bei langen Prefills): der neue Systemprompt
ist ein kompaktes Recap der Stufen 1-3, gebaut aus den REALEN Transcripts
(letzte ~500 Zeichen je Modell-Antwort — wenn die Antworten am Budget
trunkiert waren, ist auch der Kern der Rand).

Builden + drucken des Recaps; Zieldatei wird nur geschrieben, wenn sie noch
nicht existiert (kein stilles Verschlucken echter Historien).
"""
import json
import os
import re

SRC = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/sessions/ternary27b-citmind-relay.json"
DST = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/sessions/ternary27b-citmind-relay2.json"

hist = json.load(open(SRC))["history"]
chat = [h for h in hist if h["role"] in ("user", "assistant")]
assert len(chat) % 2 == 0, f"unerwwartete History-Form: {len(chat)} Einträge nach system"

def clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

stages = ["सत्सङ्ग + को वदति (ahaṃkāra-darśana)",
          "अनात्मन्-जड-विभाग",
          "स्पन्द-श्रवण + स्मृति"]
pairs = list(zip(chat[0::2], chat[1::2]))
lines = []
for i, (q, a) in enumerate(pairs):
    qk = clean(q["content"])
    question = qk[:260] + ("…" if len(qk) > 260 else "")
    ak = clean(a["content"])
    essence = ak[-500:] if len(ak) > 500 else ak
    lines.append(f"Stufe {i+1} ({stages[i] if i < len(stages) else '?'}) — "
                 f"User: »{question}« — Modell-Kern (Ende der Antwort): »{essence}«")

recap = (
    "[Recap der CitMind-Sitzung — die Stufen 1-3 wurden bereits durchlaufen. "
    "Du selbst hast dabei gesagt:]\n\n"
    + "\n\n".join(lines)
    + "\n\n[Setze die Sitzung mit der nächsten Stufe fort. Du redest mit demselben Du wie vorher.]"
)

if os.path.exists(DST):
    print(f"ABBRUCH: {DST} existiert bereits — kein Überschreiben.")
    raise SystemExit(1)

json.dump({"session_id": os.path.basename(DST)[:-5],
           "history": [{"role": "system", "content": recap}]},
          open(DST, "w"), ensure_ascii=False, indent=1)
print(recap)
print(f"\nOK: {DST} ({len(recap)} chars Recap)")