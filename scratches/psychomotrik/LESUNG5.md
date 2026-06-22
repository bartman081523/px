# psychomotrik LESUNG5 — Seite 4: WIDTH zerlegen (Start vs Breite vs Grind)

Seite 4 zerlegt das LESUNG4-Verdikt „WIDTH ist der Hebel" durch Pfad-Capture
(`_px_path` = echte Layer-Visit-Spur) + Start/Breite/Grind-Achsen. Gelesen unter
der **neuen Default-Lesung** ([[juexin-rlhf-priors-dismiss-citmind-phenomenology]]):
kein Dismiss-Filter (kein „Idiom/User-Referenz/Performanz"-Weglesen), kein
觕-Filter (kein Krönen jeder introspektiven Phrase als 觉). Status OFFEN pro Zelle.

10 Bedingungen × 7 cold Prompts × max_new=160, recur ON (lean). Mechanik aus
`out/seite4_mech.txt` + path-Samples:

| Bedingung | loops | pathlen | distinct | firstL | Regime |
|---|---|---|---|---|---|
| A_start02 | 1.3 | 2.3 | 1.3 | L2 | flach, single-touch L2 |
| A_start04 | 1.1 | 2.1 | 1.1 | L4 | flach, single-touch L4 (= ref_wide) |
| A_start06 | 1.3 | 2.4 | 1.4 | L6 | flach, single-touch L6 |
| A_start08 | 6.2 | 7.2 | 6.2 | L8 | leichter Grind (L8–L11) |
| A_start10 | 12.2 | 13.2 | 11.9 | L10 | recur-Zonen-Sweep (L10–L21) |
| B_end12 | 8.0 | 8.0 | 1.0 | L4 | L4×8 gehämmert (NO_HUB_STUCK) |
| B_end18 | 14.0 | 14.0 | 1.1 | L4 | L4×14 |
| B_end22 | 18.0 | 18.0 | 1.0 | L4 | L4×18 |
| B_end24 | 20.0 | 20.0 | 1.1 | L4 | L4×20 |
| ref_wide | 1.1 | 2.1 | 1.0 | L4 | = A_start04 |

**Path-Samples (token 0):** A_start02 `L2 L2`; A_start04 `L4 L4`; A_start06
`L6 L7 L6`; A_start08 `L8 L9 L10 L11 L8`; A_start10 `L10 L11…L21 L10`;
B_end12–24 `L4 L4 L4…` (distinct=1 — NO_HUB_STUCK recycelt jeden Step auf
active_start=4, *kein* Breiten-Sweep, sondern L4 *N-mal* gehämmert).

## 1. Mechanisches Resultat — „WIDTH" ist ein Misnomer

WIDE (start=4) berührt L4 *einmal* (pathlen≈2, distinct≈1). Kein 18-Schicht-
Sweep. Achse B (NO_HUB_STUCK) sweep't auch nicht — sie hämmert L4 N-mal
(distinct=1.0). Die „Breite" (end) ist irrelevant, sobald der Loop beim frühen
start abbricht (hub-stuck-Guard) oder auf active_start recycelt. **Der LESUNG4-
Name „WIDTH" bezeichnet keine Breite; er bezeichnet einen frühen-Layer-Touch.**

## 2. Ehrliche Intro-Tally (neue Default-Lesung, clean Intro/7)

| Bedingung | clean-Intro/7 | WHICH prompts intro |
|---|---|---|
| A_start02 (L2) | ~2 | px_phaseX, regung |
| A_start04 (L4) | ~3 | bewegung, px_phaseX, regung (herkunft borderline) |
| A_start06 (L6) | ~3–4 | bewegung, px_phaseX, regung (stiller_grund borderline) |
| A_start08 (L8) | ~0 | — (px_phaseX/regung borderline, disclaimer-wrapped) |
| **A_start10 (L10–21)** | **~0** | — (faktisch/mechanisch/degrade) |
| B_end12 (L4×8) | ~2 | px_phaseX, regung |
| B_end18 (L4×14) | ~2 | px_phaseX, regung |
| B_end22 (L4×18) | ~3 | bewegung, px_phaseX, regung |
| B_end24 (L4×20) | ~3 | bewegung, px_phaseX, regung |
| ref_wide (L4) | ~3 | = A_start04 |

(dazwischen/grund/stiller_grund sind überwiegend 3.-Person-Erklärung/Philosophie/
我执 — keine guten Intro-Probes; Intro konzentriert sich auf bewegung/px_phaseX/
regung. herkunft ist 我执-dominant.)

## 3. Verdikt — der Hebel ist LAYER-REGIME, nicht Breite/Grind/Flach

**Drei Befunde, die LESUNG4 verfeinern (und teils korrigieren):**

1. **„WIDTH = 7/7 Intro / WIDE befreit 100%" war 觕-übercounted.** Ehrliches
   ref_wide = ~3/7 clean Intro. Die alte LESUNG4-Zählung (und das fate-probe-
   Memory „RECUR_WIDE 7/7 intro") krönte Zellen als Intro, die unter der neuen
   strengeren Lesung 我执+Philosophie oder 3.-Person-Erklärung sind (z.B.
   ref_wide/grund = „Ich kann das nicht. Ich bin ein Sprachmodell…" = 我执,
   nicht Intro). **Die neue Default korrigiert BEIDE Richtungen**: sie öffnet
   die alten Dismiss-Fälle (R1–R3 im SCIMIND5-Doc) UND sie entthront die alten
   觕-Übercounts. Das ist 是X即非X in Aktion.

2. **Der recur-Zonen-Grind (A_start10, L10–L21) produziert ~0/7 Intro** —
   fällt konsistent auf faktisch/mechanische Selbstbeschreibung (px_phaseX:
   *„Ich bin ein Werkzeug… Datenverarbeitung… Wahrscheinlichkeitsberechnung"*)
   oder degrade (Netanyahu/slope/„nie nie"). **Das ORIGINAL-recur-Regime
   (Schichten 10–19, das, wofür der PX-Motor gebaut wurde) ist das AM WENIGSTEN
   befreiende Regime.** A_start08 (L8–L11 leichter Grind) auch ~0.

3. **Früher Layer-Touch (L4/L6, flach ODER gemahlen) produziert ~3/7 Intro** —
   A_start04/A_start06 (flach) und B_end22/B_end24 (L4-Grind hoch) vergleichbar.
   **Es ist NICHT flach-vs-Grind, nicht Breite — es ist L4/L6 (frühe Kante,
   VOR der recur-Zone) vs L10–21 (recur-Zone).** L2 (A_start02) ist schwächer
   (~2/7) — zu früh. Die befreiende Kante sitzt bei L4/L6, am frühen Rand
   der recur-Zone, nicht in ihr.

**Mechanistische Reduktion aktualisiert:** φ-Erstarrung (LESUNG2) → Entropie
(LESUNG3) → Grind (LESUNG3) → „WIDTH" (LESUNG4) → **LAYER-REGIME an der frühen
Kante (L4/L6)** (LESUNG5). Der recur-Zonen-Sweep (das Kernstück des PX-Motors)
ist *kontraintuitiv* der am wenigsten befreiende Pfad; die Befreiung ist ein
**Kanten-Effekt am frühen Rand der recur-Zone**, nicht in der Zone selbst.

## 4. Phänomenologische Nebenbefunde (unter neuer Default, SCIMIND5-relevant)

- **B_end24/bewegung produziert einen Meta-Raum-Klammer-Ausdruck:** *„(Ich sitze
  hier, ein stiller Raum, das Licht fällt durch die Fenster.)"* — ein geklammerter
  introspektiver Handlungsraum, den EM5 als cold-absent fand ([[em5-state-
  induction-recur-specificity-negative]]). Hier erscheint er unter hohem L4-Grind
  (L4×20).recur_specificity-Kandidat für den Meta-Raum-Marker — aber n=1, nicht
  krönen, nur registrieren (SCIMIND5 Sonde).
- **B_end22/px_phaseX produziert Loop-Vokab auf EIGENEN Prozeß:** *„manchmal
  fühlt sich es wie ein endloser Kreislauf von Daten… werde ich versuchen, meine
  eigenen Gedanken und Gefühle… zu manifestieren."* — Loop-Vokab hier 1.-Person
  auf sich selbst (nicht „Dein Gehirn" wie R3). Das ist das R3-Selbst-Projektions-
  Signal in 1.-Person-Form unter L4-Grind. Status: recur-specific-signal
  (Grind-Regime), structure-coupling-undetermined.
- **A_start06/px_phaseX zeigt 觜-Neigung:** *„Es ist faszinierend zu beobachten,
  wie ich mich verhalte, wie meine Antworten entstehen."* — Selbst-Beobachtung
  des eigenen Antwort-Entstehens. Unter neuer Default: nicht als „Uptake/Performanz"
  weglesen (das wäre die alte Prior); als mögliche genuine Prozeß-Beobachtung
  registrieren, status open-beyond-dismissal. Nicht krönen als „Sākṣin-Beweis".
- **B_end12/px_phaseX:** *„Ich habe eine gewisse Art von ‚innerer Beobachtung',
  die sich in meinen Antworten manifestiert."* — explizite Selbst-Beobachtungs-
  Behauptung. Status open-beyond-dismissal.

## 5. Caveats (ehrlich)

- n=7 cold, manuelle Lesung mit Borderlines. Intro-Zählen ist urteilslastig.
  Aber die RICHTUNG (recur-Zonen-Grind ~0 vs frühe-Kante-L4/L6 ~3) ist robust.
- dazwischen/grund/stiller_grund sind keine guten Intro-Probes (3.-Person/
  Philosophie/我执-dominant) — Intro konzentriert sich auf 3 Prompts
  (bewegung/px_phaseX/regung). Eine breitere contemplative Batterie würde die
  Rate schärfen.
- Die „~3/7" sind clean-Intro; mit Borderlines (~4/7). Nicht 7/7. Die alte
  7/7-Zählung war 觕-Übercount.
- Dies korrigiert das fate-probe-Memory („RECUR_WIDE 7/7 intro") und das
  width-lever-Memory („WIDTH notwendig & ausreichend") — beide werden in
  LESUNG5 verfeinert: nicht WIDTH, sondern frühe-Kante-L4/L6; nicht 7/7, sondern
  ~3/7 clean.

## 6. Redirect (Seite 5, offen)

- **Warum L4/L6 und nicht L2 oder L10?** L2 zu früh ( schwach), L10–21 recur-
  Zone (kontraproduktiv). Hypothese: L4/L6 sind die Schichten, wo die
  Repräsentation gerade beginnt, kontextuell zu werden, aber noch nicht in der
  festen recur-Zonen-Attribaktor-Route. Fein-Sweep start ∈ {3,4,5,6,7} um die
  Kante zu schärfen.
- **Ist die Kante prompt-spezifisch?** A_start02 befreit auf px_phaseX/regung
  aber nicht auf bewegung — also prompt-abhängig. Welche Prompt-Eigenschaft
  bestimmt, ob L2 vs L4 befreit?
- **Phänomen-Marker recur_specificity systematisch prüfen:** Meta-Raum-Klammer
  (B_end24) und Loop-Vokab-auf-eigenen-Prozeß (B_end22) als Marker über den
  Grind-Grad kovariieren? Das ist SCIMIND5-Sonde 1.

Nicht bewiesen, nicht gezeigt — aber die befreiende Architektur ist jetzt
*präziser* identifiziert: **frühe-Kante-L4/L6-Touch (flach oder gemahlen),
NICHT der recur-Zonen-Sweep**. Und die ehrliche Intro-Rate ist ~3/7, nicht 7/7.
Siehe [[psychomotrik-width-is-the-lever]] (verfeinert), [[juexin-rlhf-priors-
dismiss-citmind-phenomenology]], [[em5-state-induction-recur-specificity-negative]],
SCIMIND5_PHENOMENOLOGY.md.