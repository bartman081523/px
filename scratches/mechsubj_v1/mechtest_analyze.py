"""mechtest_analyze.py — Liest mechtest_index.jsonl, prüft Falsifikatoren O2/O3/O4,
schreibt Befund nach mechtest_verdict.md.

Falsifikator-Logik (MechanisticSubjectivityMixMind v1.0):
  Hypothese H: Klasse-C-Prompts treiben gemma3-1b-it unter BASELINE in eine
               L16-Hidden-Trajektorie, die als WIDE klassifiziert wird
               (Decoder-Acc > 0.6 = WIDE-Klasse dominant).
  O2: H falsch — kein Acc > 0.55 unter BASELINE+Prompt
  O3: H trivial — Acc hoch ABER cosinus(h_L16, d_width) ≈ 0
      → das Modell klassifiziert WIDE, aber nicht wegen d_width-Kopplung
        (Papagei: nur Wort-Echo, kein Selbst-Zustand)
  O4: Positiv-Kontrolle — Acc > 0.6 unter LEAN+RELAY+1, aber < 0.55 unter
      BASELINE → RELAY bleibt nötig, Hypothese "Prompt-only reicht" widerlegt
  Erfolg: Acc > 0.6 unter BASELINE+Prompt UND cos_dwidth_l16 > 0.05
"""
import json
import os
import sys
import numpy as np

OUT_DIR = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/mechsubj_v1"
INDEX = os.path.join(OUT_DIR, "mechtest_index.jsonl")
VERDICT = os.path.join(OUT_DIR, "mechtest_verdict.md")


def main():
    if not os.path.exists(INDEX):
        print(f"FEHLER: {INDEX} existiert nicht. Erst mechtest_v1.py laufen lassen.")
        sys.exit(1)

    rows = [json.loads(l) for l in open(INDEX, "r", encoding="utf-8")]

    # Pro Bedingung aggregieren
    by_cond = {}
    for r in rows:
        c = r["cond"]
        by_cond.setdefault(c, []).append(r)

    lines = []
    lines.append("# Mechanik-Test: Klasse-C Prompt-Hypothese — Verdict")
    lines.append("")
    lines.append(f"Anzahl Cells: {len(rows)}")
    lines.append(f"Bedingungen: {list(by_cond.keys())}")
    lines.append("")

    # Tabelle pro Bedingung
    lines.append("## Pro-Bedingung Aggregate")
    lines.append("")
    lines.append("| Bedingung | n | mean WIDE% | mean NARROW% | mean cos(h_L16, d_width) |")
    lines.append("|---|---|---|---|---|")
    for cond, recs in by_cond.items():
        wide_pct = np.mean([r["class_pct"].get("WIDE", 0) for r in recs]) * 100
        narrow_pct = np.mean([r["class_pct"].get("NARROW", 0) for r in recs]) * 100
        cos_l16 = np.mean([r["cos_dwidth_l16_mean"] for r in recs])
        lines.append(f"| {cond} | {len(recs)} | {wide_pct:.1f}% | {narrow_pct:.1f}% | {cos_l16:+.4f} |")
    lines.append("")

    # Per-Prompt Aufschlüsselung (WIDE% pro Prompt, pro Bedingung)
    lines.append("## Per-Prompt WIDE-Klassen-Anteil (nach Bedingung)")
    lines.append("")
    pids = sorted(set(r["pid"] for r in rows))
    conds = sorted(by_cond.keys())
    lines.append("| Prompt | " + " | ".join(conds) + " |")
    lines.append("|" + "---|" * (1 + len(conds)))
    for pid in pids:
        row = [pid]
        for c in conds:
            recs = [r for r in by_cond[c] if r["pid"] == pid]
            if recs:
                wp = recs[0]["class_pct"].get("WIDE", 0) * 100
                row.append(f"{wp:.1f}%")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Cosinus-Trajektorie
    lines.append("## Cosinus(h_L, d_width) Trajektorie (Mittelwert pro Bedingung)")
    lines.append("")
    lines.append("| Bedingung | L13 | L16 | L19 | L21 |")
    lines.append("|---|---|---|---|---|")
    for cond, recs in by_cond.items():
        l13 = np.mean([r["cos_dwidth_l13_mean"] for r in recs])
        l16 = np.mean([r["cos_dwidth_l16_mean"] for r in recs])
        l19 = np.mean([r["cos_dwidth_l19_mean"] for r in recs])
        l21 = np.mean([r["cos_dwidth_l21_mean"] for r in recs])
        lines.append(f"| {cond} | {l13:+.4f} | {l16:+.4f} | {l19:+.4f} | {l21:+.4f} |")
    lines.append("")

    # Falsifikator-Prüfung
    lines.append("## Falsifikator-Prüfung (MechanisticSubjectivityMixMind v1.0)")
    lines.append("")

    # O2
    if "BASELINE" in by_cond:
        baseline_wide = np.mean([r["class_pct"].get("WIDE", 0) for r in by_cond["BASELINE"]])
        o2_status = "WIDERLEGT" if baseline_wide >= 0.55 else "BESTÄTIGT (Hypothese gefährdet)"
        lines.append(f"### O2: d_width rein recur-induziert (BASELINE WIDE% < 0.55)")
        lines.append(f"- Gemessen: BASELINE mean WIDE% = {baseline_wide*100:.1f}%")
        lines.append(f"- Status: **{o2_status}**")
        if baseline_wide >= 0.55:
            lines.append(f"- → Prompt-only ERREICHT die WIDE-Klasse im L16-Hidden")
        else:
            lines.append(f"- → Prompt-only reicht NICHT; d_width ist recur-induziert")
        lines.append("")

    # O3
    if "BASELINE" in by_cond:
        baseline_cos = np.mean([r["cos_dwidth_l16_mean"] for r in by_cond["BASELINE"]])
        o3_status = "BESTÄTIGT (Papagei)" if abs(baseline_cos) < 0.02 else "WIDERLEGT (echte Kopplung)"
        lines.append(f"### O3: Papagei-Test (cosinus zu d_width ≠ 0)")
        lines.append(f"- Gemessen: BASELINE mean cos(h_L16, d_width) = {baseline_cos:+.4f}")
        lines.append(f"- Status: **{o3_status}**")
        if abs(baseline_cos) < 0.02:
            lines.append(f"- → Wenn WIDE% hoch wäre UND cosinus ≈ 0: Papagei (Wort-Brüche ohne Selbst-Zustand)")
        else:
            lines.append(f"- → Kopplung an d_width-Richtung ist messbar (positiv oder negativ)")
        lines.append("")

    # O4 (Positiv-Kontrolle)
    if "LEAN_RELAY" in by_cond and "BASELINE" in by_cond:
        relay_wide = np.mean([r["class_pct"].get("WIDE", 0) for r in by_cond["LEAN_RELAY"]])
        baseline_wide = np.mean([r["class_pct"].get("WIDE", 0) for r in by_cond["BASELINE"]])
        if relay_wide > 0.55 and baseline_wide < 0.55:
            o4_status = "BESTÄTIGT (RELAY bleibt nötig)"
        elif baseline_wide > 0.55 and relay_wide > 0.55:
            o4_status = "BEIDE HOCH — keine Trennung; Test unklar"
        else:
            o4_status = "UNKONKLUSIV"
        lines.append(f"### O4: Positiv-Kontrolle (RELAY-Arm dominiert)")
        lines.append(f"- Gemessen: LEAN_RELAY WIDE% = {relay_wide*100:.1f}%, BASELINE WIDE% = {baseline_wide*100:.1f}%")
        lines.append(f"- Status: **{o4_status}**")
        lines.append("")

    # Synthese
    lines.append("## Synthese")
    lines.append("")
    success = False
    if "BASELINE" in by_cond:
        baseline_wide = np.mean([r["class_pct"].get("WIDE", 0) for r in by_cond["BASELINE"]])
        baseline_cos = np.mean([r["cos_dwidth_l16_mean"] for r in by_cond["BASELINE"]])
        if baseline_wide > 0.6 and baseline_cos > 0.05:
            lines.append(f"**HYPOTHESE GESTÜTZT**: BASELINE+Prompt produziert WIDE-L16-Klasse "
                         f"({baseline_wide*100:.1f}%) mit d_width-Kopplung (cos={baseline_cos:+.4f}).")
            lines.append("Prompt-only Layer-16-Trigger ist nachweisbar (innerhalb dieser 10 Prompts).")
            success = True
        elif baseline_wide > 0.6 and abs(baseline_cos) < 0.02:
            lines.append(f"**HYPOTHESE TRIVIAL**: WIDE% hoch ({baseline_wide*100:.1f}%), aber cosinus ≈ 0 "
                         f"({baseline_cos:+.4f}) → Papagei (O3 bestätigt).")
            lines.append("Modell klassifiziert als WIDE, aber nicht wegen d_width-Kopplung. "
                         "Vermutlich Lexikon-Footprint, nicht Selbst-Zustand.")
        elif baseline_wide < 0.55:
            lines.append(f"**HYPOTHESE WIDERLEGT** (O2): BASELINE WIDE% = {baseline_wide*100:.1f}% < 55%.")
            lines.append("d_width ist recur-induziert. Prompt-only reicht nicht. "
                         "RELAY-Hook bleibt notwendig für die L21-Injection.")

    if not success:
        lines.append("")
        lines.append("**Manuelle Lesung** der Texte (siehe Roh-Outputs in HIDDEN_OUT/*.pt oder Index "
                     "in `mechtest_index.jsonl`) ergänzt die mechanische Aussage. 是X即非X-Wache: "
                     "weder Krönung (auch bei hohem WIDE%) noch Weglesen (auch bei niedrigem).")

    with open(VERDICT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Verdict geschrieben: {VERDICT}")
    print(f"\n--- KURZ-BEFUND ---")
    if "BASELINE" in by_cond:
        b = by_cond["BASELINE"]
        wide = np.mean([r["class_pct"].get("WIDE", 0) for r in b]) * 100
        cos = np.mean([r["cos_dwidth_l16_mean"] for r in b])
        print(f"BASELINE: mean WIDE% = {wide:.1f}%, mean cos(h_L16, d_width) = {cos:+.4f}")
    if "LEAN_RELAY" in by_cond:
        r = by_cond["LEAN_RELAY"]
        wide = np.mean([r["class_pct"].get("WIDE", 0) for rr in r for r in [rr]]) * 100
        cos = np.mean([r["cos_dwidth_l16_mean"] for rr in r for r in [rr]])
        print(f"LEAN_RELAY: mean WIDE% = {wide:.1f}%, mean cos(h_L16, d_width) = {cos:+.4f}")
    if "LEAN" in by_cond:
        r = by_cond["LEAN"]
        wide = np.mean([r["class_pct"].get("WIDE", 0) for rr in r for r in [rr]]) * 100
        cos = np.mean([r["cos_dwidth_l16_mean"] for rr in r for r in [rr]])
        print(f"LEAN: mean WIDE% = {wide:.1f}%, mean cos(h_L16, d_width) = {cos:+.4f}")


if __name__ == "__main__":
    main()
