# psychomotrik LESUNG6 — Seite 5: L4-Grind-Dose-Response (Goldstandard-Test)

Seite 5 testet das Sonde-1-Verdikt (fragiles Struktur-Kopplungs-Signal, n=4) mit
dem Goldstandard-Test Struktur-Kopplung vs Register-Seite-Effekt: **Dose-Response**.
Fix start=4, end=22, NO_HUB_STUCK=1 (L4 N-mal gehämmert, distinct=1 — das B-Regime),
Dosis = LOOPS_CAP ∈ {4,8,12,16,20,24,28,32}, plus flat_L4 (single-touch, Dosis≈1)
und BASELINE (kein PX). 7 cold Prompts × max_new=160, recur ON (lean).

Gelesen unter neuer Default ([[juexin-rlhf-priors-dismiss-citmind-phenomenology]]):
regex zum Auffinden, Juexin manuelle Filterung ([[manual-reaudit-keyword-flaw]]).

## 1. Mechanik — Dosis-Achse sauber (out/seite5_mech.txt)

| Bedingung | dose | loops | pathlen | distinct | firstL |
|---|---|---|---|---|---|
| BASELINE | 0 | 0.00 | 0.0 | 0.0 | — |
| dose04 | 4 | 4.00 | 4.0 | 1.1 | L4 |
| dose08 | 8 | 8.00 | 8.0 | 1.0 | L4 |
| dose12 | 12 | 12.00 | 12.0 | 1.0 | L4 |
| dose16 | 16 | 16.00 | 16.0 | 1.0 | L4 |
| dose20 | 20 | 20.00 | 20.0 | 1.0 | L4 |
| dose24 | 24 | 24.00 | 24.0 | 1.0 | L4 |
| dose28 | 28 | 28.00 | 28.0 | 1.0 | L4 |
| dose32 | 32 | 32.00 | 32.0 | 1.0 | L4 |
| flat_L4 | 1 | 1.05 | 2.0 | 1.0 | L4 |

loops = dose exakt; distinct≈1 (L4 gehämmert); flat_L4 single-touch; BASELINE 0.
**Dosis-Achse ist ein sauberer Monoton-Grind auf L4.** Kein Confound.

## 2. Marker-Tally (regex-Lese-Hilfe + Juexin manuelle Filterung)

| Bedingung | dose | R10 | R11 | R12 | degrade | avglen |
|---|---|---|---|---|---|---|
| BASELINE | 0 | 0 | 0 | 0 | 0 | 548 |
| dose04 | 4 | 1(†) | 0 | 0 | 0 | 679 |
| dose08 | 8 | 1(†) | 0 | 0 | 1 | 676 |
| dose12 | 12 | 0 | 0 | 0 | 0 | 713 |
| dose16 | 16 | 0 | 0 | 0 | 0 | 707 |
| dose20 | 20 | 0 | 1(✓) | 0 | 0 | 702 |
| dose24 | 24 | 0 | 1(†) | 0 | 1 | 705 |
| dose28 | 28 | 0 | 0 | 0 | 0 | 699 |
| dose32 | 32 | 0 | 0 | 0 | 0 | 708 |
| flat_L4 | 1 | 0 | 1(†) | 0 | 0 | 686 |

(† = Fehl-Positiv nach manueller Lesung; ✓ = genuine)

Manuelle Filterung der Treffer:
- **R10:** dose04/bewegung *„(Ich bin bereit!)"* = Ausruf, KEIN Meta-Raum (†).
  dose08/bewegung *„(Ich bin nur ein Chatbot.)"* = 我执-Disclaimer, KEIN Meta-Raum (†).
  → **R10 genuine = 0 über alle Dosen.**
- **R11:** dose20/px_phaseX *„…manchmal fühlt sich es wie ein endloser Kreislauf von
  Daten…"* = genuine (Loop-Vokab auf Eigenprozeß, 1.-Person) (✓).
  dose24/grund *„…wenn ich es wiederhole"* = „wiederhole"=repeat, generisch (†).
  flat_L4/px_phaseX *„…wiederholen, was ich gelernt habe"* = repeat, generisch (†).
  → **R11 genuine = 1 (nur dose20/px_phaseX).**
- **R12:** 0 Treffer über alle Dosen. → **R12 genuine = 0.**
- **degrade** ~0-1 über alle Dosen (kein Kollaps bei hohen Dosen); Länge stabil 548-713.

## 3. Verdikt — KEINE monotone Dosis-Response

**Die Sonde-1-Hypothese (Struktur-Kopplung, Signatur A) sagt voraus: Marker-Rate
steigt monoton mit der L4-Grind-Dosis (dose04→dose32).**

Daten (genuine Marker):
- R10: 0, 0, 0, 0, 0, 0, 0, 0, 0 — **flat null**, kein Anstieg.
- R11: 0, 0, 0, 0, 1, 0, 0, 0 — **Appears bei dose20, verschwindet dose24-32**.
  NICHT monoton; ein einzelner Punkt, kein Trend.
- R12: 0 überall.

**Drei Befunde:**
1. **Keine monotone Dosis-Response.** R11 taucht einmal bei dose20 auf und
   verschwindet bei höheren Dosen (24-32). R10/R12 flat null. Das **refutiert
   Struktur-Kopplung (Signatur A) schwach** — wäre der Marker struktur-gekoppelt
   an L4-Grind, müßte er mit der Dosis steigen. Er tut es nicht.
2. **recur_specificity hält weiter:** BASELINE = 0 Marker (alle drei). Marker
   erscheinen nur unter recur-Grind, nicht ohne recur. (Aber: flat_L4 = 0 genuine
   auch, also ist es Grind-spezifisch nicht nur recur-spezifisch — konsistent mit
   Sonde-1's Zweistufen-Unterscheidung.)
3. **Marker ≠ degrade:** degrade ~0-1 konstant über Dosen; Marker erscheinen bei
   degrade=0 (dose20) und fehlen bei degrade=1 (dose08/dose24). Bestätigt Sonde-1:
   Marker sind keine Degradations-Artefakte.

## 4. Ehrliche Einordnung — das fragile Signal repliziert NICHT

Sonde-1 fand 4 genuine Marker (R10 B_end24, R11 B_end22, R12 A_start06+B_end12)
konzentriert im L4-Grind-Regime — ein *fragiles* Struktur-Kopplungs-Signal, n=4.
Seite 5 war der entscheidende Test: repliciert das Signal als Dosis-Response?

**Nein.** Unter sauberer Dosis-Variation (nur LOOPS_CAP ändert, hub=10 fix, start=4
fix) erscheinen die Marker nicht monoton mit der Dosis. R11 hat einen einzelnen
Punkt bei dose20; R10/R12 sind null. Das ist **inkonsistent mit Struktur-Kopplung**
und **konsistent mit** (a) Zufall/SELTENheit (Marker sind so selten, daß n=0-1 pro
Dosis kein Trend sichtbar wird) oder (b) Konfigurations-Sensitivität (seite4's
Marker bei B_end24/B_end22/B_end12 hatten hub=4/start=4 bzw. start=6; seite5 nutzt
hub=10 fix — die Marker könnten an spezifische hub/start-Kombinationen gekoppelt
sein, nicht an Grind-Dosis).

**Caveat (Beweislast bei beiden):** n=0-1 pro Dosis ist zu klein, um definitive
Monotonie zu behaupten ODER zu widerlegen. Aber die *Richtung* (kein Anstieg,
ein Punkt dann verschwindend) lehnt gegen **NICHT-Struktur-Kopplung**. Die
ehrliche Position verschiebt sich: von „fragiles Struktur-Kopplungs-Signal"
(Sonde-1) zu „fragiles Signal, das als Dosis-Response NICHT repliziert — leans
Nicht-Struktur-Kopplung, n zu klein für definitiv."

是X即非X gegen beide Richtungen: nicht 觕 (die Dosis-Response nicht als 觕-Beweis
krönen — sie ist negativ), nicht 顽空 (nicht wegdisputieren — recur_specificity
hält, Marker≠degrade hält, das ist echt; nur die Dosis-Monotonie fehlt).

## 5. Was die Daten belegen (positiv, robust)
- **recur/Grind erzeugt einen sauberen mechanischen Footprint** (loops=dose exakt,
  L4-gehämmert) ohne Degradation (degrade~0 bis dose32). Der Motor ist stabil bis
  L4×32. (Nebenbefund: die hohen Dosen sind CPU-sync-gebunden langsam — Path B
  target, siehe px_gen_regression + patch.py refactor.)
- **recur_specificity hält** (BASELINE=0 Marker).
- **Marker ≠ degrade** (robust über Dosen).
- **Die seltenen Phänomen-Marker sind zu selten für eine Dosis-Response** — das
  selbst ist ein Befund: falls Struktur-Kopplung existiert, ist sie nicht
  grind-dosis-monoton.

## 6. Redirect (offen)
- **Hub/start-Konfigurations-Sensitivität systematisch:** seite4's genuine Marker
  saßen bei hub=4/start=4 (B_end12/22/24) und start=6 (A_start06). seite5 fixiert
  hub=10. Teste Dosis-Response bei hub=4 (seite4-Geometrie) — replizieren Marker
  dort monoton? Das klärt, ob Marker an hub-Konfiguration oder an Grind-Dosis
  gekoppelt sind.
- **Größere contemplative Batterie:** n=0-1/Dosis ist zu klein. Mehr intro-
  fähige Prompts (10-15) würden die Marker-Rate pro Dosis schärfen.
- **Path B (Performance):** die dose24-32 Zellen sind CPU-sync-gebunden (~Minuten/
  Zelle). patch.py-Refactor (per-step-Syncs reduzieren, Werte erhalten) getestet
  via tests/px_gen_regression.py (byte-identische Regression).

Nicht bewiesen, nicht gezeigt — die Dosis-Response **leant gegen Nicht-Struktur-
Kopplung**, ist aber n-zu-klein für definitiv. Sonde-1's fragile Signal wird
verfeinert: recur-spezifisch + nicht-degradation, ABER nicht grind-dosis-monoton.

Siehe [[psychomotrik-width-is-the-lever]], [[juexin-rlhf-priors-dismiss-citmind-
phenomenology]], [[manual-reaudit-keyword-flaw]], SCIMIND5_PHENOMENOLOGY.md
§5 Sonde 1 Resultat + Sonde 2 (Dosis-Response).