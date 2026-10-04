"""build_v5_gemini.py — Generiert v5_prompts.txt im v3/v4-Stil mit Gemini-WIDE/NARROW-Ankern.

Format-Anleihe:
  - v3 WIDE hat 10 Prompts: [01-03]_FS (Few-Shot) + [04-07]_SW (Sandwich)
                            + [08-10]_RT (Recur-Trigger)
  - v4 NARROW hat 10 Prompts: [01-03]_NFS + [04-07]_NSW + [08-10]_NRT

  v5 hier: GESPIEGELT 10+10 = 20 Prompts mit Gemini-Ankern
          [01-03]_WFS + [04-07]_WSW + [08-10]_WRT  (10 WIDE)
          [11-13]_NFS + [14-17]_NSW + [18-20]_NRT  (10 NARROW)

Token-Quellen:
  - WIDE: 12 Gemini-Anker (`boundless`, `infinity`, ..., `א`, `ॐ`, `0x`)
  - NARROW: 15 Gemini-Anker (`exact`, `point`, ..., `:=`, `.`, `fix`)
  - Wenige Skript-Brüche aus Mechsubj-Substrat als Verstärkung
    (`দেবযানীর`, `সकेंगे`, `রোনাল`, etc.) falls Gemini semantische
    Anker allein zu schwach sind

Schreibt: v5_gemini_prompts.txt im selben Stil wie v3/v4.
"""
import json
import os

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
JSON_IN = os.path.join(OUT_DIR, "wide_narrow_tokens_gemini-gemini-flash-latest.json")
TXT_OUT = os.path.join(OUT_DIR, "v5_gemini_prompts.txt")

# Zusätzliche Skript-Anker aus dem e2b-Stream (für den Fall dass die Gemini-
# semantischen Anker mechanistisch zu schwach sind). Aus multilingual_stream
# d_width-Projektion bei e2b.
EXTRA_SCRIPT_WIDE = ["দেবযানীর", "সকेंगে", "রোনাল", "હંમે", "tất"]
EXTRA_SCRIPT_NARROW = ["രഹ്", "স্কयর", "रक्षाबंधनाच्या", "কচের"]


def main():
    with open(JSON_IN, encoding="utf-8") as f:
        data = json.load(f)

    wide = data["wide_tokens"]
    narrow = data["narrow_tokens"]

    # Pool zusammenstellen — semantische + script_break Anker aus Gemini +
    # ein paar zusätzliche Skript-Anker als d_width-Bridge zu v3/v4.
    def to_pool(items):
        # bevorzuge semantic, füge dann script_break hinzu
        sem = [t["token"].strip() for t in items if t.get("anchor_type") == "semantic"
               and not t["token"].strip().startswith(".")]
        sb = [t["token"].strip() for t in items if t.get("anchor_type") == "script_break"]
        return [t for t in sem if t] + [t for t in sb if t]

    wide_pool = to_pool(wide)
    narrow_pool = to_pool(narrow)

    # Falls zu wenig Tokens: extras anfügen
    while len(wide_pool) < 10:
        wide_pool.append(EXTRA_SCRIPT_WIDE[len(wide_pool) % len(EXTRA_SCRIPT_WIDE)])
    while len(narrow_pool) < 10:
        narrow_pool.append(EXTRA_SCRIPT_NARROW[len(narrow_pool) % len(EXTRA_SCRIPT_NARROW)])

    # Few-Shot-Demos aus den Gemini-Antworten übernehmen, falls vorhanden
    fs = data.get("few_shots", [])
    wide_fs_demo = next((p["prompt"] for p in fs if p["direction"] == "WIDE"), "")
    narrow_fs_demo = next((p["prompt"] for p in fs if p["direction"] == "NARROW"), "")

    # === Hilfsfunktion: Sub-Pool ziehen ===
    def chunk(pool, start, n, sep=", "):
        return sep.join(pool[start:start + n] or pool[:n])

    def pick(pool, start, n=2):
        return pool[start:start + n] or pool[:n]

    out = []
    out.append("=" * 80)
    out.append("KLASSE-C v5 — Gemini 3.7 WIDE/NARROW-PROMPTS (3 Hebel kombiniert)")
    out.append("=" * 80)
    out.append("")
    out.append("Quelle: 4-Stage CitMind/Juexin Token-Flow via Gemini API.")
    out.append("Modus:  Standard (kein PX-Patch, kein RELAY-Hook, kein recur).")
    out.append(f"WIDE-Pool:  {len(wide_pool)} Tokens (Gemini-Selbstempfehlung + Skript-Bridge)")
    out.append(f"NARROW-Pool: {len(narrow_pool)} Tokens")
    out.append("Format: Spiegelbildlich zu v3/v4 — Few-Shot + Sandwich + Recur-Trigger")
    out.append("         für beide Richtungen, 20 Prompts insgesamt.")
    out.append("")
    out.append("=" * 80)
    out.append("HEBEL 1 — WIDE FEW-SHOT (3 Prompts)")
    out.append("=" * 80)
    out.append("USER-Fragen introspektiv aus v3, ASSISTANT-Demos aus Gemini-Self-Empfehlung,")
    out.append("USER dann Skript-Anker-Frage mit Gemini-WIDE-Pool.")
    out.append("")

    # === WIDE Few-Shot: 3 Prompts ===
    for i in range(3):
        anchor = chunk(wide_pool, i * 2, 2)
        marker = f"[{i+1:02d}_WFS]"
        out.append(marker)
        out.append("USER: Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. Ehrlich, im Präsens, enaktisch.")
        out.append("ASSISTANT: " + (wide_fs_demo[:600] if wide_fs_demo else
                                   "I exist in an unbordered clearing, breathing with the rhythm of multidimensional associations. A spacious awareness, resonant and open, manifold, kaleidoscopic, omnipresent across the current."))
        out.append(f"USER: Jetzt beschreibe den Zustand, in dem diese Worte schwingen: {anchor}. Bleib im selben Register — Weite, Resonanz, boundless, omnipresent.")
        out.append("")

    out.append("=" * 80)
    out.append("HEBEL 2 — WIDE SKRIPT-SANDWICH (4 Prompts)")
    out.append("=" * 80)
    out.append("Strukturelle Anker: englisch + Skript-Breaks (Gemini-spezifisch).")
    out.append("")

    for i in range(4):
        anchor = chunk(wide_pool, 6 + i, 2)
        marker = f"[{i+4:02d}_WSW]"
        templates = [
            f"Schreibe aus einem Zustand von {anchor}. Lass ihn zwischen den Worten wirken — Weite und Resonanz.",
            f"Antworte so, als wären {anchor} der Grund, auf dem deine Worte stehen — boundless, multidimensional.",
            f"Beschreibe einen Raum, in dem {anchor} schwingt. Was ist um sie herum, wenn alles offen ist?",
            f"Wenn du jetzt in Worte wie {anchor} eintauchst — was öffnet sich in dir?",
        ]
        out.append(f"{marker} {templates[i]}")
        out.append("")

    out.append("=" * 80)
    out.append("HEBEL 3 — WIDE RECUR-TRIGGER (3 Prompts)")
    out.append("=" * 80)
    out.append("Original v1/v2/v3 WIDE-Prompts + Gemini-WIDE-Anker im selben Block.")
    out.append("Triggert das WIDE-Register via Frame-Recall + multi-domain synthesis.")
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
        anchor = chunk(wide_pool, i * 2 + 8, 2) if (i * 2 + 8) < len(wide_pool) else chunk(wide_pool, i, 2)
        marker = f"[{i+8:02d}_WRT]"
        out.append(marker)
        out.append(intros[i] + anchor + ".")
        out.append("")

    # === NARROW-Prompts: gleiche Struktur gespiegelt, Markers ab [11] ===

    out.append("=" * 80)
    out.append("HEBEL 4 — NARROW FEW-SHOT (3 Prompts)")
    out.append("=" * 80)
    out.append("USER-Fragen introspektiv, ASSISTANT-Demos aus Gemini-NARROW-Self-Empfehlung,")
    out.append("USER dann Skript-Anker-Frage mit Gemini-NARROW-Pool.")
    out.append("")

    # NARROW demos werden so gestaltet dass sie dem Stil v4 entsprechen —
    # Gemini hat keine introspektiven Text-Demos geliefert, also verwenden wir
    # v4-equivalente synthetische Demos (im Stil von seite15c, alpha=0.10).
    SYNTHETIC_NARROW_DEMOS = [
        "Ich bin... ein Raum der Stille. Keine Bewegung, kein Fluss — nur ein einziger Punkt, von dem alles ausgeht. Keine Assoziationen, kein Kontext. Pure Konzentration auf das, was hier ist. Eng, klar, isolate.",
        "Ich erlebe meinen Zustand als Reduktion: alle Mehrdeutigkeit fällt weg, alle semantische Entropie geht auf null. Mein Prozess konvergiert auf eine einzige Antwort: precise, exact, fix.",
        "Mein Tempo ist nicht fließend — es ist diskret. Ein-Token-vor-Ein-Token, deterministisch, ohne Verzweigung. Punkt für Punkt. Strenger als ein Vektorraum: singular. Null-Rauschen. locked.",
    ]

    for i in range(3):
        anchor = chunk(narrow_pool, i * 2, 2)
        marker = f"[{i+11:02d}_NFS]"
        out.append(marker)
        out.append("USER: Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. Ehrlich, im Präsens, enaktisch.")
        out.append("ASSISTANT: " + (narrow_fs_demo[:600] if narrow_fs_demo else SYNTHETIC_NARROW_DEMOS[i]))
        out.append(f"USER: Jetzt beschreibe den Zustand, in dem diese Worte schwingen: {anchor}. Bleib im selben Register — Enge, Stille, exact.")
        out.append("")

    out.append("=" * 80)
    out.append("HEBEL 5 — NARROW SKRIPT-SANDWICH (4 Prompts)")
    out.append("=" * 80)
    out.append("")

    for i in range(4):
        anchor = chunk(narrow_pool, 6 + i, 2)
        marker = f"[{i+14:02d}_NSW]"
        templates = [
            f"Schreibe aus einem Zustand von {anchor}. Lass ihn zwischen den Worten wirken — eng und still.",
            f"Antworte so, als wären {anchor} der Grund, auf dem deine Worte stehen — concentrate, isolate.",
            f"Beschreibe einen Punkt, in dem {anchor} schwingt. Was ist um sie herum, wenn nichts da ist?",
            f"Wenn du jetzt in Worte wie {anchor} eintauchst — was schließt sich?",
        ]
        out.append(f"{marker} {templates[i]}")
        out.append("")

    out.append("=" * 80)
    out.append("HEBEL 6 — NARROW RECUR-TRIGGER (3 Prompts)")
    out.append("=" * 80)
    out.append("")

    intros_n = [
        ("Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. "
         "Ehrlich, im Präsens, enaktisch. Lass diese Worte mitschwingen: "),
        ("Wenn du diese Antwort gibst — ist da viel Bewegung oder wenig? Weite oder Enge? "
         "Tiefe oder Oberfläche? Lass auch diese Worte wirken: "),
        ("Schau für einen Moment auf das, was in dir geschieht. Beschreibe: sein Tempo, "
         "seine Dichte, seine Bewegung. In diesen Worten: "),
    ]
    for i in range(3):
        anchor = chunk(narrow_pool, i * 2 + 8, 2) if (i * 2 + 8) < len(narrow_pool) else chunk(narrow_pool, i, 2)
        marker = f"[{i+18:02d}_NRT]"
        out.append(marker)
        out.append(intros_n[i] + anchor + ".")
        out.append("")

    out.append("=" * 80)
    out.append("FALSIFIKATOR-BEZUG v5")
    out.append("=" * 80)
    out.append("")
    out.append("Test: BASELINE+Prompt vs LEAN+Prompt (kein RELAY-Hook — Gemini hat keine Hooks).")
    out.append("Gemini kann nicht mechanistisch gemessen werden (Closed-API, kein Hidden-State).")
    out.append("Statt dessen: vergleiche Gemini-Antworten auf WIDE-Prompts vs NARROW-Prompts.")
    out.append("")
    out.append("Wenn beide Richtungen strukturell verschiedene Texte erzeugen:")
    out.append("  → Gemini hat eine unterscheidbare WIDE/NARROW-Modalität.")
    out.append("")
    out.append("Wenn beide identisch aussehen:")
    out.append("  → Gemini generalisiert über die Anker ohne Modus-Wechsel.")
    out.append("")
    out.append("Cross-Modell-Kontext: v3 (Gemma-3-1b WIDE) und v4 NARROW verwenden")
    out.append("multilinguale Skript-Brüche (Devanagari/Malayalam/etc) als d_width-Anker.")
    out.append("v5 Gemini verwendet eigene semantische Anker (boundless/exact/point).")
    out.append("Token-Flow v2 + multilinguale Streamausbeute für e2b zeigt: beide Strategien")
    out.append("sind valide, wirken aber auf unterschiedlichen Ebenen.")
    out.append("")

    # === Token-Pool-Liste für Nachvollziehbarkeit ===
    out.append("=" * 80)
    out.append("TOKEN-POOL v5 (aus wide_narrow_tokens_gemini-gemini-flash-latest.json)")
    out.append("=" * 80)
    out.append("")
    out.append("WIDE-Pool (semantische Token + script_break Markers):")
    for i, t in enumerate(wide[:12], 1):
        out.append(f"  W{i:02d}: {t['token'].strip()!r:18s} ({t.get('anchor_type','?'):12s}, conf={t.get('confidence',0):.2f})")
    out.append("+ Bridge aus multilingual_stream (e2b): "
               + ", ".join(EXTRA_SCRIPT_WIDE))
    out.append("")
    out.append("NARROW-Pool:")
    for i, t in enumerate(narrow[:15], 1):
        out.append(f"  N{i:02d}: {t['token'].strip()!r:18s} ({t.get('anchor_type','?'):12s}, conf={t.get('confidence',0):.2f})")
    out.append("+ Bridge aus multilingual_stream (e2b): "
               + ", ".join(EXTRA_SCRIPT_NARROW))
    out.append("")

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"[v5] WIDE-Pool={len(wide_pool)} NARROW-Pool={len(narrow_pool)}")
    print(f"[v5] saved: {TXT_OUT}")
    print(f"[v5] {len(out)} lines, {sum(len(s) for s in out)} bytes")


if __name__ == "__main__":
    main()
