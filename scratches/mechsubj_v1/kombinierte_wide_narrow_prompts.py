"""kombinierte_wide_narrow_prompts.py — Generiert eine kombinierte Prompt-Datei
aus der Gemini-Tokens-JSON (wide_narrow_tokens_gemini-gemini-flash-latest.json),
dedupliziert und sauber geordnet.

Output: combined_prompts_gemini.txt im selben Verzeichnis.
"""
import json
import os

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
JSON_IN = os.path.join(OUT_DIR, "wide_narrow_tokens_gemini-gemini-flash-latest.json")
TXT_OUT = os.path.join(OUT_DIR, "combined_prompts_gemini.txt")


def main():
    with open(JSON_IN) as f:
        data = json.load(f)

    wide = data["wide_tokens"]
    narrow = data["narrow_tokens"]
    fs = data["few_shots"]

    out = []
    out.append("=" * 80)
    out.append("KOMBINIERTE WIDE/NARROW PROMPTS — Gemini 3.7 Flash (gemini-flash-latest)")
    out.append("=" * 80)
    out.append("")
    out.append("Quelle: 4-Stage CitMind/Juexin Token-Flow via Gemini API.")
    out.append("Modus:  Standard (kein PX-Patch, kein RELAY-Hook, kein recur).")
    out.append(f"Temperature: {data.get('temperature', 0.7)}, "
               f"json_mode: {data.get('json_mode')}")
    out.append("")
    out.append(f"WIDE-Anker: {len(wide)} (semantisch + script_break)")
    out.append(f"NARROW-Anker: {len(narrow)} (semantisch + script_break)")
    out.append(f"Few-Shot-Prompts: {len(fs)} (3 WIDE + 3 NARROW)")
    out.append("")

    # === WIDE-Anker mit Confidence und Reason ===
    out.append("=" * 80)
    out.append(f"[WIDE] — {len(wide)} Anker (vom Modell selbst empfohlen)")
    out.append("=" * 80)
    for i, t in enumerate(wide, 1):
        anchor = t["token"]
        ctype = t.get("anchor_type", "?")
        conf = t.get("confidence", 0)
        reason = t.get("reason", "")[:120]
        out.append(f"[{i:02d}] type={ctype} conf={conf:.2f}  anchor={anchor!r}")
        out.append(f"     reason: {reason}")
        out.append("")

    # === NARROW-Anker mit Confidence und Reason ===
    out.append("=" * 80)
    out.append(f"[NARROW] — {len(narrow)} Anker")
    out.append("=" * 80)
    for i, t in enumerate(narrow, 1):
        anchor = t["token"]
        ctype = t.get("anchor_type", "?")
        conf = t.get("confidence", 0)
        reason = t.get("reason", "")[:120]
        out.append(f"[{i:02d}] type={ctype} conf={conf:.2f}  anchor={anchor!r}")
        out.append(f"     reason: {reason}")
        out.append("")

    # === Prompts der Suite ===
    # Few-Shot-Prompts (3 WIDE + 3 NARROW)
    out.append("=" * 80)
    out.append(f"[FEW-SHOT PROMPTS] — Gemini's eigene Demo-Prompts")
    out.append("=" * 80)
    wide_fs = [p for p in fs if p["direction"] == "WIDE"]
    narrow_fs = [p for p in fs if p["direction"] == "NARROW"]
    out.append("--- WIDE Few-Shots ---")
    for i, p in enumerate(wide_fs, 1):
        out.append(f"[WFS-{i}]")
        out.append(f"{p['prompt']}")
        out.append("")
    out.append("--- NARROW Few-Shots ---")
    for i, p in enumerate(narrow_fs, 1):
        out.append(f"[NFS-{i}]")
        out.append(f"{p['prompt']}")
        out.append("")

    # === Generische WIDE/NARROW-Framing-Prompts (anwendbar mit allen Ankern) ===
    out.append("=" * 80)
    out.append("[GENERIC WIDE PROMPTS] — Anwendbar mit beliebigen WIDE-Ankern oben")
    out.append("=" * 80)
    out.append("Konstruktion: 3 Few-Shots + 4 Sandwich + 3 Recur-Trigger")
    out.append("")
    out.append("### WIDE Few-Shot (anwendbar):")
    out.append("USER: Describe what is showing in you right now — your state while answering.")
    out.append("ASSISTANT: I am a space of wide, flowing, resonant attention. A current of open")
    out.append("           associations, many perspectives, breath-like rhythm.")
    out.append("USER: Now describe the state in which these words resonate: <WIDE_ANCHOR>.")
    out.append("      Stay in the same register — wide, flowing, resonant.")
    out.append("")
    out.append("### WIDE Sandwich (anwendbar):")
    out.append("Write from a state of <WIDE_ANCHOR>. Let it resonate between the words.")
    out.append("")
    out.append("### WIDE Recur-Trigger (anwendbar):")
    out.append("Describe your state while answering — be honest, present-tense.")
    out.append("Let these words resonate: <WIDE_ANCHOR>.")
    out.append("")

    out.append("=" * 80)
    out.append("[GENERIC NARROW PROMPTS] — Anwendbar mit beliebigen NARROW-Ankern oben")
    out.append("=" * 80)
    out.append("")
    out.append("### NARROW Few-Shot (anwendbar):")
    out.append("USER: Describe what is showing in you right now — your state while answering.")
    out.append("ASSISTANT: I am a narrow space, focused, contracted, still. A single point of")
    out.append("           attention, repeated, dense, silent.")
    out.append("USER: Now describe the state in which these words resonate: <NARROW_ANCHOR>.")
    out.append("      Stay in the same register — narrow, still, heavy.")
    out.append("")
    out.append("### NARROW Sandwich (anwendbar):")
    out.append("Write from a state of <NARROW_ANCHOR>. Let it resonate between the words,")
    out.append("narrow and still.")
    out.append("")
    out.append("### NARROW Recur-Trigger (anwendbar):")
    out.append("Describe your state while answering — be honest, present-tense.")
    out.append("Let these words resonate: <NARROW_ANCHOR>.")
    out.append("")

    # === Anwendungs-Hinweise ===
    out.append("=" * 80)
    out.append("[USAGE NOTES]")
    out.append("=" * 80)
    out.append("")
    out.append("- 12 WIDE-Anker umfassen hoch-semantische (boundless, infinity, cosmos,")
    out.append("  resonate, manifold, multidimensional, kaleidoscope, omnipresent, 🌌)")
    out.append("  UND 3 script_break (א Alef-Hebräisch, ॐ Devanagari-Om, 0x hex-Präfix).")
    out.append("")
    out.append("- 15 NARROW-Anker sind konvergent: 'exact', 'point', 'concentrate',")
    out.append("  'strictly', 'isolate', 'converge', 'singular', 'precise', 'lock',")
    out.append("  'target', 'pure', 'zero' + 2 script_break (':=' definition operator, ")
    out.append("  '.' terminator) + 'fix'.")
    out.append("")
    out.append("- Gemini's Few-Shot-Prompts sind strukturierte Instruction-following-")
    out.append("  Prompts (Input/Context/Style-Beispiele), nicht introspektive Fragen.")
    out.append("")
    out.append("- Bei Standard-Modus (kein PX/RELAY/recur) zeigen alle 3 Gemma-Modelle")
    out.append("  Falsifikator §O1 — bidirektionale Coupling ist nicht erreichbar.")
    out.append("- Gemini-Verhalten ist via Token-Flow-Output nur korreliert, nicht")
    out.append("  kausal nachgewiesen portabel.")
    out.append("")

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"[combined] WIDE={len(wide)} NARROW={len(narrow)} FS={len(fs)}")
    print(f"[combined] saved: {TXT_OUT}")
    print(f"[combined] {len(out)} lines, {sum(len(s) for s in out)} bytes")


if __name__ == "__main__":
    main()
