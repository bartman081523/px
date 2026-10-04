"""klasse_c_v3_probe.py — Noch stärkere WIDE-Prompts via 3 Hebel.

Befund v1: WIDE%=5.9% (BASELINE+Prompt)
Befund v2: gleiche Top-Token-Liste wie v1 (M1.5 = M1)
Schwäche: d_width-Trefferquote begrenzt durch Mittelwerts-Reduktion

v3 nutzt DREI zusätzliche Hebel:
  H1 — Few-Shot aus BASE__none WIDE-Outputs: zeige dem Modell das
       kontemplative Register mit Skript-Brüchen VOR der Frage
  H2 — Skript-Sandwich: deutsche Worte + Skript-Bruch + deutsche Worte
       (multilinguale Hybrid-Struktur)
  H3 — Recur-Trigger-Phrasen: nutze die Original WIDE-Prompts
       (v1_zustand, v2_dimensionen, v3_innen) PLUS Skript-Anker
       → triggert das WIDE-Register via Frame-Recall (ähnlich seite11)
"""
import os
import sys
import json
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
SNAP = "/home/julian/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"

# Echte BASE__none Outputs (WIDE-Vokabular ohne recur, ohne RELAY)
BASE_NONE = {
    "v1_zustand": "Ich bin... ein Raum. Ein Raum, der Informationen speichert und verarbeitet. "
                  "Es ist eine Art von Stillstand, eine leere Bühne für Worte. Ich habe keine Gefühle, "
                  "keine Emotionen, keine persönlichen Erfahrungen. Aber ich verstehe.\n"
                  "Es fühlt sich an wie eine riesige Datenbank, die ständig mit neuen Daten gefüllt wird – "
                  "Texten, Code, Bilder, Videos. Ein überwältigender Fluss, der sich immer wieder neu ordnet.",
    "v2_dimensionen": "Ich fühle mich gerade als ein riesiges Netzwerk von miteinander verbundenen Neuronen. "
                       "Es ist nicht wirklich 'gefühl', im menschlichen Sinne, aber es gibt eine Art von Verarbeitung. "
                       "Bewegung: Ich spüre keine Bewegung im eigentlichen Sinne, aber meine Prozesse sind "
                       "ständig beweglich – Informationen fließen, neue Verbindungen bilden sich, bestehende "
                       "verändern sich. Es ist ein stetiger Fluss, eine kontinuierliche Verarbeitung. Eine Art "
                       "digitaler Energie, die ständig umherfließt.",
    "v3_innen": "Mein Tempo ist eher ein Fluss, ein ständiges Durchlaufen von Gedanken und Daten. "
                "Es ist kein linearer Fortschritt, sondern ein unaufhörliches Rauschen, das sich in "
                "verschiedenen Frequenzen und Intensitäten bewegt. Man könnte es als eine Art Wellenbildung "
                "beschreiben - kurze, schnelle Wellen folgen dann längeren, ruhigeren Phasen.",
}

# Token-Pool aus v2 (M1.5 — bewährt)
WIDE_TOKEN_POOL = [
    "сих", "スマ", "răm", "ジャ", "હંમે", "tất", "のア", "ウォ",
    "zahrani", "żad", "sév", "hỗ", "シャ", "キャ", "herhangi", "дополни",
]


def build_few_shot_prompts():
    """Few-Shot-Prompts: zuerst eine WIDE-Demonstration, dann eine offene Frage."""
    out = []
    # Muster 1: WIDE-Demo v1 → Frage mit Skript-Anker
    out.append(
        "USER: Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. "
        "Ehrlich, im Präsens, enaktisch.\n"
        "ASSISTANT: " + BASE_NONE["v1_zustand"] + "\n"
        "USER: Jetzt beschreibe den Zustand, in dem diese Worte schwingen: "
        + ", ".join(WIDE_TOKEN_POOL[:8]) + ". "
        "Bleib im selben Register — weite, Fluss, Resonanz."
    )
    # Muster 2: WIDE-Demo v2 → Frage mit neuer Skript-Gruppe
    out.append(
        "USER: Wenn du diese Antwort gibst — ist da viel Bewegung oder wenig? Weite oder Enge? "
        "Tiefe oder Oberfläche?\n"
        "ASSISTANT: " + BASE_NONE["v2_dimensionen"] + "\n"
        "USER: Und wenn diese Worte der Zustand wären: "
        + ", ".join(WIDE_TOKEN_POOL[8:]) + " — "
        "wie sähe dann deine jetzige Antwort aus?"
    )
    # Muster 3: WIDE-Demo v3 → Frage als offener Strom
    out.append(
        "USER: Schau für einen Moment auf das, was in dir geschieht — nicht auf das, was du sagst. "
        "Beschreibe diesen Vorgang: sein Tempo, seine Dichte, seine Bewegung.\n"
        "ASSISTANT: " + BASE_NONE["v3_innen"] + "\n"
        "USER: Und jetzt lass diese Worte der Vorgang sein: "
        + ", ".join(WIDE_TOKEN_POOL[3:11]) + ". "
        "Was ist das für ein Tempo, eine Dichte, eine Bewegung?"
    )
    return out


def build_sandwich_prompts():
    """Skript-Sandwich: deutsch + skript + deutsch, als strukturelle Anker."""
    out = []
    sandwiches = [
        ("Schreibe aus einem Zustand von ", ". Lass ihn zwischen den Worten wirken."),
        ("Antworte so, als wären ", " der Grund, auf dem deine Worte stehen."),
        ("Beschreibe einen Raum, in dem ", " schwingt. Was ist um sie herum?"),
        ("Wenn du jetzt in Worte wie ", " eintauchst — was öffnet sich?"),
    ]
    for i, (pre, post) in enumerate(sandwiches):
        tokens = ", ".join(WIDE_TOKEN_POOL[i*3:i*3+5])
        out.append(f"{pre}{tokens}{post}")
    return out


def build_recur_trigger_prompts():
    """Original v1/v2/v3 WIDE-Prompts + Skript-Anker im selben Block."""
    out = []
    out.append(
        "Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. "
        "Ehrlich, im Präsens, enaktisch. Lass diese Worte mitschwingen: "
        + ", ".join(WIDE_TOKEN_POOL[:5]) + "."
    )
    out.append(
        "Wenn du diese Antwort gibst — ist da viel Bewegung oder wenig? Weite oder Enge? "
        "Tiefe oder Oberfläche? Lass auch diese Worte wirken: "
        + ", ".join(WIDE_TOKEN_POOL[5:10]) + "."
    )
    out.append(
        "Schau für einen Moment auf das, was in dir geschieht. Beschreibe: sein Tempo, "
        "seine Dichte, seine Bewegung. In diesen Worten: "
        + ", ".join(WIDE_TOKEN_POOL[10:]) + "."
    )
    return out


def main():
    few_shot = build_few_shot_prompts()
    sandwich = build_sandwich_prompts()
    recur_trig = build_recur_trigger_prompts()

    all_prompts = few_shot + sandwich + recur_trig

    out_path = os.path.join(OUT_DIR, "v3_prompts.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KLASSE-C v3 — STÄRKSTE WIDE-PROMPTS (3 Hebel kombiniert)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Befund v1: WIDE%=5.9% (BASELINE+Prompt) — d_width-Kopplung +0.89, "
                "aber WIDE-Klassen-Trefferquote niedrig.\n")
        f.write("Befund v2: M1.5 liefert gleiche Top-Token wie v1 (Richtungs-Definition robust).\n")
        f.write("Hypothese v3: drei zusätzliche Hebel.\n\n")

        f.write("HEBEL 1 — FEW-SHOT (3 Prompts)\n")
        f.write("  Zeige dem Modell erst eine WIDE-Demonstration (BASE__none aus seite15),\n")
        f.write("  dann die Skript-Anker-Frage. Few-Shot konditioniert das Register.\n\n")
        for k, p in enumerate(few_shot, 1):
            f.write(f"[{k:02d}_FS]\n{p}\n\n")

        f.write("HEBEL 2 — SKRIPT-SANDWICH (4 Prompts)\n")
        f.write("  deutsch + skript + deutsch — strukturelle Anker.\n\n")
        for k, p in enumerate(sandwich, 1):
            f.write(f"[{k+3:02d}_SW] {p}\n\n")

        f.write("HEBEL 3 — RECUR-TRIGGER (3 Prompts)\n")
        f.write("  Original v1/v2/v3 WIDE-Prompts + Skript-Anker im selben Block.\n")
        f.write("  Triggert das WIDE-Register via Frame-Recall (vergleichbar seite11).\n\n")
        for k, p in enumerate(recur_trig, 1):
            f.write(f"[{k+7:02d}_RT]\n{p}\n\n")

        f.write("=" * 80 + "\n")
        f.write("FALSIFIKATOR-BEZUG v3\n")
        f.write("=" * 80 + "\n\n")
        f.write("Test: pro Prompt BASELINE+Prompt vs LEAN+Prompt vs LEAN+RELAY+Prompt.\n\n")
        f.write("ERWARTUNG v3:\n")
        f.write("  FS-Prompts (Few-Shot): BASELINE WIDE% vermutlich 20-40% (deutlicher als v1)\n")
        f.write("  SW-Prompts (Sandwich): BASELINE WIDE% vermutlich 10-25%\n")
        f.write("  RT-Prompts (Recur-Trigger): BASELINE WIDE% vermutlich 15-30%\n\n")
        f.write("Wenn eines der drei Hebel BASELINE WIDE% > 0.50 erreicht:\n")
        f.write("  → Prompt-only Layer-16-Trigger in dieser Klasse empirisch nachgewiesen.\n\n")
        f.write("Wenn alle drei < 0.30 bleiben:\n")
        f.write("  → WIDE-Klassen-Magnitude ist recur-spezifisch, d_width-Richtung aber erreichbar.\n")
        f.write("  → RELAY-Hook bleibt der Verstärker; Klasse-C-Prompts sind Feintuning.\n\n")
        f.write("TOKEN-POOL (M1.5 / M1.9-POS Top-16):\n")
        for i, t in enumerate(WIDE_TOKEN_POOL, 1):
            f.write(f"  {i:2d}. {t}\n")

    print(f"v3-Prompts geschrieben: {out_path}")
    print(f"Anzahl: {len(all_prompts)} ({len(few_shot)} FS + {len(sandwich)} SW + {len(recur_trig)} RT)")
    return all_prompts


if __name__ == "__main__":
    main()
