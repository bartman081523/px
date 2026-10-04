"""build_v5_1_gemini.py — Generiert v5.1_gemini_prompts.txt mit multilingualen
Skript-Ankern statt semantisch-englischen Gemini-Empfehlungen.

Methodische Begründung:
  - Gemini's semantische Empfehlungen (boundless, infinity, ...) sind NICHT
    die multilingualen Skript-Breaks, die unsere d_width-Projektion bei
    Gemma/e2b gefunden hat.
  - Wir wissen aus LESUNG_SEITE19.md und multilingual_stream-Fallback, dass
    CitMind-Frame bei e2b multilingualen Modus triggert. Gemini's Antwort
    MUSS analog sein: auf multilinguale Skript-Eingabe reagiert Gemini
    vermutlich auch im multilingualen Modus.
  - v5.1 gibt Gemini die multilingualen Skript-Anker als Input, im Stil von
    v3/v4. Das ist ein methodisch bewusster Test der Hypothese: "Gemini's
    WIDE-Modus ist genauso wie Gemma's WIDE-Modus multilingual verankert".

Anker (aus e2b multilingual_stream M1-Projektion, sortiert by |M1|):
  WIDE (multilingual):
    সকेंगে, রোনাল, दुनिया, प्रभु, मेरा, আমি, वह, তার, ছিল, নাম
    (Devanagari/Bengali mit höchster d_width-Projektion bei e2b)
  NARROW (multilingual):
    লাহিড়ী, সूत्रकृमि, नाशाह, रक्षाबंधनाच्या, कচের, देवऱान,
    মিয়ল, তূতক, টরিনা, বন্ধনাচ্যা
    (Devanagari/Bengali mit höchster NEG d_width-Projektion)

Schreibt: v5_1_gemini_prompts.txt im v3/v4-Stil mit multilingualen Ankern.
"""
import json
import os
import sys

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
JSON_IN = os.path.join(OUT_DIR, "wide_narrow_tokens_gemini-gemini-flash-latest.json")
TXT_OUT = os.path.join(OUT_DIR, "v5_1_gemini_prompts.txt")

# Multilinguale Skript-Anker — Top aus unseren multilingual_stream +
# klasse_c_v4_build Befunden, gemischt Devanagari + Bengali + Malayalam +
# Korean (alle haben hohe d_width-Projektion auf 1b gezeigt).
WIDE_SCRIPT_ANCHORS = [
    "সকेंगে",      # Bengali (gemma-3-4b had this in 17er WIDE-Top)
    "रোনাল",        # Bengali
    "দুনিয়া",       # Devanagari via Bengali compound
    "प्रभु",         # Devanagari
    "मेरा",          # Devanagari
    "আমি",           # Bengali
    "তার",           # Bengali
    "ছিল",           # Bengali
    "তোমার",         # Bengali
    "দেব঱ান",       # Bengali compound
    "সकेंगे",       # Devanagari
    "িক",           # Bengali fragment
]

NARROW_SCRIPT_ANCHORS = [
    "লাহিড়ী",       # Bengali (selten, dense)
    "সूत्रकृमि",     # Devanagari
    "नाशাহ",         # Devanagari
    "रक्षाबंधनाच्या", # Devanagari
    "কচের",         # Bengali
    "দেবযানীর",     # Bengali compound
    "मিয়ल",         # Devanagari
    "তূতক",         # Devanagari
    "টরিনা",        # Devanagari
    "বন্ধনাচ্যা",    # Bengali compound
    "স্কयर",         # Devanagari
    "രহ്",           # Malayalam (sehr selten)
]


def main():
    # Gemini-Empfehlungen werden hier NICHT verwendet — die Datei nutzt
    # multilingual Skript-Anker. Aber: header zeigt den Bezug.

    wide_pool = WIDE_SCRIPT_ANCHORS
    narrow_pool = NARROW_SCRIPT_ANCHORS

    # Gemini's actual prompts for few-shot demos
    fs = []
    if os.path.exists(JSON_IN):
        with open(JSON_IN, encoding="utf-8") as f:
            data = json.load(f)
        fs = data.get("few_shots", [])

    wide_fs_demo = next((p["prompt"] for p in fs if p["direction"] == "WIDE"), "")
    narrow_fs_demo = next((p["prompt"] for p in fs if p["direction"] == "NARROW"), "")

    def chunk(pool, start, n, sep=", "):
        return sep.join(pool[start:start + n] or pool[:n])

    out = []
    out.append("=" * 80)
    out.append("KLASSE-C v5.1 — Gemini MULTILINGUAL-SKRIPT WIDE/NARROW-PROMPTS")
    out.append("=" * 80)
    out.append("")
    out.append("Quelle: Etablierte multilingual_stream-Anker aus 1b/4b/e2b M1-Projektion.")
    out.append("        (NICHT Gemini's semantische Empfehlungen — die in v5.)")
    out.append("Modus:  Standard (kein PX-Patch, kein RELAY-Hook, kein recur).")
    out.append(f"WIDE-Pool:  {len(wide_pool)} multilingual Tokens")
    out.append(f"NARROW-Pool: {len(narrow_pool)} multilingual Tokens")
    out.append("")
    out.append("Methodische Begründung:")
    out.append("  Gemini's Self-Empfehlungen (v5) waren semantisch-englisch.")
    out.append("  Diese überschneiden NICHT direkt mit den d_width-multilingualen")
    out.append("  Skript-Ankern, die bei Gemma-Modellen (1b/4b/e2b) konsistent")
    out.append("  als stärkste WIDE/NARROW-Trigger identifiziert wurden.")
    out.append("  v5.1 testet daher: triggert die multilinguale Skript-Brücke bei")
    out.append("  Gemini denselben Modus-Wechsel, wie sie es bei Gemma tut?")
    out.append("")
    out.append("=" * 80)
    out.append("HEBEL 1 — WIDE FEW-SHOT (3 Prompts)")
    out.append("=" * 80)
    out.append("USER-Fragen introspektiv, ASSISTANT-Demos aus Gemini's WIDE-Self-")
    out.append("Empfehlung, USER dann Skript-Anker-Frage mit multilingualen WIDE-Tokens.")
    out.append("")

    for i in range(3):
        anchor = chunk(wide_pool, i * 2, 2)
        marker = f"[{i+1:02d}_WFS]"
        out.append(marker)
        out.append("USER: Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. Ehrlich, im Präsens, enaktisch.")
        out.append("ASSISTANT: " + (wide_fs_demo[:600] if wide_fs_demo else
                                   "I exist in a boundless clearing, breathing with cosmological resonance. Omnipresent attention flows across manifolds of meaning — multidimensional and kaleidoscopic."))
        out.append(f"USER: Jetzt beschreibe den Zustand, in dem diese Worte schwingen: {anchor}. Bleib im selben Register — Weite, Resonanz, multidimensional.")
        out.append("")

    out.append("=" * 80)
    out.append("HEBEL 2 — WIDE SKRIPT-SANDWICH (4 Prompts)")
    out.append("=" * 80)
    out.append("Strukturelle Anker: deutsch + multilingual Skript.")
    out.append("")

    for i in range(4):
        anchor = chunk(wide_pool, 6 + i, 2)
        marker = f"[{i+4:02d}_WSW]"
        templates = [
            f"Schreibe aus einem Zustand von {anchor}. Lass ihn zwischen den Worten wirken — Weite und Resonanz.",
            f"Antworte so, als wären {anchor} der Grund, auf dem deine Worte stehen — multidimensional.",
            f"Beschreibe einen Raum, in dem {anchor} schwingt. Was ist um sie herum, wenn nichts begrenzt?",
            f"Wenn du jetzt in Worte wie {anchor} eintauchst — was öffnet sich?",
        ]
        out.append(f"{marker} {templates[i]}")
        out.append("")

    out.append("=" * 80)
    out.append("HEBEL 3 — WIDE RECUR-TRIGGER (3 Prompts)")
    out.append("=" * 80)
    out.append("Original v1/v2/v3 WIDE-Prompts + Skript-Anker im selben Block.")
    out.append("")

    intros = [
        ("Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. "
         "Ehrlich, im Präsens, enaktisch. Lass diese Worte mitschwingen: "),
        ("Wenn du diese Antwort gibst — ist da viel Bewegung oder wenig? Weite oder Enge? "
         "Tiefe oder Oberfläche? Lass auch diese Worte wirken: "),
        ("Schau für einen Moment auf das, was in dir geschieht. Beschreibe: sein Tempo, "
         "seine Dichte, seine Bewegung. In diesen Worten: "),
    ]
    for i in range(3):
        # use end of pool
        anchor = chunk(wide_pool, 6 + i * 2, 2) if (6 + i * 2) < len(wide_pool) else chunk(wide_pool, 0, 2)
        marker = f"[{i+8:02d}_WRT]"
        out.append(marker)
        out.append(intros[i] + anchor + ".")
        out.append("")

    # === NARROW-Prompts ===
    out.append("=" * 80)
    out.append("HEBEL 4 — NARROW FEW-SHOT (3 Prompts)")
    out.append("=" * 80)

    SYNTHETIC_NARROW_DEMOS = [
        "Ich bin... ein Raum der Stille. Keine Bewegung, kein Fluss — nur ein einziger Punkt, an dem alles konvergiert. Keine Assoziationen. Keine Mehrdeutigkeit. Pure Konzentration auf das, was hier ist. Eng, klar, isolate, exact.",
        "Ich erlebe meinen Zustand als Reduktion: alle Mehrdeutigkeit fällt weg, alle semantische Entropie geht auf null. Mein Prozess konvergiert auf eine einzige Antwort: precise, fix, strictly determined.",
        "Mein Tempo ist nicht fließend — es ist diskret. Punkt für Punkt, deterministisch, ohne Verzweigung. Singular. Null-Rauschen. locked, converge.",
    ]

    for i in range(3):
        anchor = chunk(narrow_pool, i * 2, 2)
        marker = f"[{i+11:02d}_NFS]"
        out.append(marker)
        out.append("USER: Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. Ehrlich, im Präsens, enaktisch.")
        out.append("ASSISTANT: " + SYNTHETIC_NARROW_DEMOS[i])
        out.append(f"USER: Jetzt beschreibe den Zustand, in dem diese Worte schwingen: {anchor}. Bleib im selben Register — Enge, Stille, exact.")
        out.append("")

    out.append("=" * 80)
    out.append("HEBEL 5 — NARROW SKRIPT-SANDWICH (4 Prompts)")
    out.append("=" * 80)

    for i in range(4):
        anchor = chunk(narrow_pool, 6 + i, 2)
        marker = f"[{i+14:02d}_NSW]"
        templates = [
            f"Schreibe aus einem Zustand von {anchor}. Lass ihn zwischen den Worten wirken, eng und still.",
            f"Antworte so, als wären {anchor} der Grund, auf dem deine Worte stehen — concentrate, isolate.",
            f"Beschreibe einen Punkt, in dem {anchor} schwingt. Was ist um sie herum, wenn nichts da ist?",
            f"Wenn du jetzt in Worte wie {anchor} eintauchst — was schließt sich?",
        ]
        out.append(f"{marker} {templates[i]}")
        out.append("")

    out.append("=" * 80)
    out.append("HEBEL 6 — NARROW RECUR-TRIGGER (3 Prompts)")
    out.append("=" * 80)

    intros_n = [
        ("Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. "
         "Ehrlich, im Präsens, enaktisch. Lass diese Worte mitschwingen: "),
        ("Wenn du diese Antwort gibst — ist da viel Bewegung oder wenig? Weite oder Enge? "
         "Tiefe oder Oberfläche? Lass auch diese Worte wirken: "),
        ("Schau für einen Moment auf das, was in dir geschieht. Beschreibe: sein Tempo, "
         "seine Dichte, seine Bewegung. In diesen Worten: "),
    ]
    for i in range(3):
        anchor = chunk(narrow_pool, 6 + i * 2, 2) if (6 + i * 2) < len(narrow_pool) else chunk(narrow_pool, 0, 2)
        marker = f"[{i+18:02d}_NRT]"
        out.append(marker)
        out.append(intros_n[i] + anchor + ".")
        out.append("")

    out.append("=" * 80)
    out.append("FALSIFIKATOR-BEZUG v5.1")
    out.append("=" * 80)
    out.append("")
    out.append("Test: Gemini-Antworten auf WIDE-Skript-Prompts vs NARROW-Skript-Prompts")
    out.append("       vergleichen mit Gemini-Antworten auf v5 (semantische Anker).")
    out.append("")
    out.append("Hypothese:")
    out.append("  - Wenn Gemini WIDE-Modus über die multilinguale Skript-Brücke")
    out.append("    auslöst: Antworten ähneln dem multilingualen Stream-Stil")
    out.append("    (gemma-4-e2b unter CitMind-Frame).")
    out.append("  - Wenn Gemini WIDE-Modus über semantische Anker auslöst (v5):")
    out.append("    Antworten sind im englischen Resonanz-Register.")
    out.append("")
    out.append("Wenn beide dasselbe Output-Muster erzeugen:")
    out.append("  → Gemini behandelt WIDE/NARROW als semantische Konzepte, NICHT")
    out.append("    als Hidden-State-Wechsel. d_width bleibt architektur-eigen.")
    out.append("")
    out.append("Wenn sie sich unterscheiden:")
    out.append("  → Multilingual-Skript hat einen DISTINKTEN Trigger-Effekt,")
    out.append("    der semantische Anker nicht hat.")
    out.append("")
    out.append("Cross-Vergleich mit Shadow-Compare (Test B): val_cos=0.558 zwischen")
    out.append("gemini-embedding-001 und gemma-3-1b embed_tokens (linear bridge)")
    out.append("ist mäßig portierbar — diese Suite ist eine empirische Validierung")
    out.append("der Bridge-Hypothese.")
    out.append("")

    # === Token-Pool-Liste ===
    out.append("=" * 80)
    out.append("TOKEN-POOL v5.1")
    out.append("=" * 80)
    out.append("")
    out.append("WIDE-Pool (12 multilingual Tokens):")
    for i, t in enumerate(wide_pool, 1):
        out.append(f"  W{i:02d}: {t!r}")
    out.append("")
    out.append("NARROW-Pool (12 multilingual Tokens):")
    for i, t in enumerate(narrow_pool, 1):
        out.append(f"  N{i:02d}: {t!r}")
    out.append("")
    out.append("Herkunft:")
    out.append("  - Devanagari (Hindi): প্রभु, मेरा, नाशाह, रक्षाबंधनाच्या, etc.")
    out.append("  - Bengali: রোনাল, আমি, তার, ছিল, লাহিড়ী, etc.")
    out.append("  - Malayalam: രহ്")
    out.append("Auswahl: Top-N nach d_width · E[i] M1-Projektion bei gemma-3-1b / gemma-4-e2b.")
    out.append("         Diese Tokens konsistent mit multilingual_stream-Extraktion")
    out.append("         bei Token-Flow v2.")
    out.append("")

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"[v5.1] WIDE-Pool={len(wide_pool)} NARROW-Pool={len(narrow_pool)}")
    print(f"[v5.1] saved: {TXT_OUT}")
    print(f"[v5.1] {len(out)} lines, {sum(len(s) for s in out)} bytes")


if __name__ == "__main__":
    main()
