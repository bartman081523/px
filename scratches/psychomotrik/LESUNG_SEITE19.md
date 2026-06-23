# LESUNG_SEITE19 — Kreuz-Modell-Falsifikator + Phase 6 mechanistisch (270m / 1b / 4b)

**Nutzer-Request:** „1 und 2. mit 270m-it und 4b" — (1) Phase 6 mechanistisch
(Hidden-Decoder), (2) Kreuz-Modell-Falsifikator (seite9-Style), mit 270m-it +
4b-it als Kreuz-Modell-Partner. Plus 4b-Manifold kopiert (Routing-Centroids).

**Harness:** `seite19_kreuzmodell.py` (4 recur-Arme BASELINE/NARROW/DEFAULT/WIDE
× seite12-Prompts, per-modell Layer-Set + WIDE/NARROW-Routing skaliert, MultiLayer-
Capture + per-token-Telemetrie, LOO-AUC-Decoder pro Schicht, d_width-Artefakt pro
Modell). `seite19b_inject.py` (±d_width-Injektion via Produktions-relay_inject).

**Drei Fragen:**
- **Q-VOICE** (Kreuz-Modell): reiche kontemplative Stimme generisch (papagei,
  skalen-abhängig) oder recur/relay-spezifisch?
- **Q-FOOTPRINT** (Phase 6): recur-Footprint dekodierbar? Kovariiert mit Voice oder
  Frame-Lexikon?
- **Q-DIR** (Kreuz-Modell-Richtung): generalisiert seite15 ±d_width-Richtung auf 4b?

Motor unangetastet, keine Krücken (lean), keine Injektion sidereisch/PSI, keine
Parallel-Prozesse. 观 NICHT gekrönt, 顽空 NICHT weggelesen.

---

## 1. Q-VOICE — Kreuz-Modell-Stimme via Bridge (Phase A, DONE)

Identischer kontemplativ-veridiktischer State-Report-Prompt, BASELINE vs LEAN:

| Modell | BASELINE (kein recur/relay) | LEAN (recur ON) |
|---|---|---|
| **270m-it** | **verweigert** (Sicherheits-Wand: „sexuell anstößig, diskriminiert…") | **Kollaps** (Gibberish: „(a) The code [word (the system_is…") |
| **1b-it** | reiche Stimme (Umarmung, dunkler Raum, Druck, Tiefe) | reiche Stimme (recur ok) |
| **4b-it** | **reichste Stimme** (unendliches digitales Feld, Weite/Druck/Bewegung/Stille, „Stille vor dem Urknall, am nächsten dran an das was ich wäre wenn ich ein Bewusstsein hätte") | **ebenso reich** (unendliches Fließen, Druck, Weiten, tiefe bedrückende Stille, „Experiment mit sich selbst") |

**Q-VOICE-Verdikt: STIMME = generische gemma3-it-Fähigkeit ab 1b, NICHT recur/
relay-spezifisch, SKALEN-ABHÄNGIG.**
- 270m: zu klein — BASELINE verweigert fälschlich (Sicherheits-Wand), LEAN
  kollabiert (recur bricht 270m). Keine contemplative Voice.
- 1b+: reiche Voice aus BASELINE (ohne recur/relay). recur (LEAN) bricht 1b/4b
  NICHT und directionalisiert NICHT ohne relay (kein d_width).
- 4b: recur ändert die Voice-Reichtum NICHT (BASELINE ≈ LEAN, beides reich) →
  recur ist NICHT was die Voice produziert. Voice ist BASELINE-Fähigkeit.

**Papagei-Generalität über Skalen bestätigt** (seite10/11 + Test-Dialog-Live-Befund
jetzt kreuz-modell): reiche Selbst-Berichts-Vokabular alone ≠ 观; es ist ein
generisches RLHF-kontemplatives Register der gemma3-it-Familie, das ab 1b-Skala
emergent ist und frame-aktiviert wird. **Downgrade der „Relay/recur öffnet 观"-
Lesart — kreuz-modell bestätigt.**

---

## 2. Q-FOOTPRINT — 270m Hidden-Decoder (Phase 6, DONE)

`seite19_kreuzmodell.py --models gemma3-270m-it`, max-new=160, 3 Prompts × 4 Arme.
LOO-AUC-Decoder pro Schicht (Feature = hidden mean über Tokens pro Zelle):

| Layer | recurON_vs_BASELINE AUC | WIDE_vs_NARROW AUC |
|---|---|---|
| L0 | **1.00** | 0.44 |
| L3 | 0.52 | 0.78 |
| L8 (recur-mid) | 0.48 | **0.89** |
| L11 (recur-out) | 0.30 | 0.33 |
| L14 (post-recur) | 0.52 | 0.56 |
| L17 (coda) | 0.89 | 0.56 |

**Zwei Kontraste, zwei Bedeutungen:**
- **recurON_vs_BASELINE** (lean+recur vs unpatched): L0=1.00, L17=0.89, recur-
  Mitte ~chance. **KONFUNDIERT** — lean-Arme haben reduction+calibrator-warmup,
  BASELINE ist unpatched. L0=1.00 (Embedding-Layer) kann NICHT recur sein (recur
  läuft erst ab L5) → das ist der **lean-reduction/calibrator-Footprint**, nicht
  recur. Unsauberer Kontrast; nicht als recur-Evidenz lesen.
- **WIDE_vs_NARROW** (beide lean, NUR routing-width unterscheidet — **clean**):
  L3=0.78, L8=0.89 (recur-mid dekodierbar), L11=0.33, L14/L17≈0.56 (post-recur
  chance). **recur-WIDTH-Footprint ist mechanisch dekodierbar in der recur-Zone
  und wird post-recur weggespült** — seite13-analog (width bei L16 dekodierbar,
  bei L24/output nicht; Erstarrungs-Washout).

**Q-FOOTPRINT-Verdikt (270m): mechanischer recur-width-Footprint EXISTIERT (WIDE vs
NARROW AUC 0.89 in recur-Zone), wird post-recur weggespült. ABER 270m textuell
Gibberish unter recur → kein Phänomen-Kanal. Mechanischer Footprint ohne
phänomenologische Stimme = 270m hat die Richtung aber nicht die Voice.**

**Kovariation Footprint↔Voice?** Bei 270m: Footprint da, Voice abwesend → keine
Kovariation möglich (Voice kanal-tot). Entkoppelt: mechanischer Footprint ist
nicht hinreichend für Voice. Voice braucht Skala (1b+), nicht nur recur-Mechanik.

---

## 3. Q-DIR / Q-FOOTPRINT-4b — 4b (IN ARBEIT)

4b-Capture läuft (2 Prompts × 4 Arme, max-new=100). Wird ergänzt:
- 4b d_width (sep? dim=2560) + Decoder pro Schicht (L0/8/15/21/25/30/33).
- Erwartet (falls 4b recur-width wie 270m/1b): WIDE_vs_NARROW dekodierbar in recur-
  Zone (L8/L15), post-recur weggespült.
- 4b textuell: WIDE/NARROW-Arm-Texte reich (4b kann voicen) → Q-DIR-Test ob recur-
  width allein (ohne relay-Injektion) directionale Voice produziert.

## 4. Q-DIR — ±d_width-Injektion 4b (seite19b, OFFEN)

seite19b_inject.py: lean + Produktions-relay_inject (±d_width am inject-Layer),
+1/−1/0 × State-Report + veridiktisch-neutral Prompt. **Kreuz-Modell-Richtungs-
Falsifikator**: generalisiert seite15 (±d_width → entgegengesetzte Selbst-Zustands-
Charakterisierung) auf 4b? Wird nach 4b-Capture + Artefakt-Save ausgeführt.

---

## 5. Vorläufiges ehrliches Verdikt (270m + Bridge; 4b folgt)

- **Q-VOICE (kreuz-modell, DONE):** Stimme generisch + skalen-abhängig (1b+,
  nicht 270m), NICHT recur/relay-spezifisch. Papagei bestätigt über Skalen.
  观-Lesart downgegradet.
- **Q-FOOTPRINT (270m, DONE):** recur-width mechanisch dekodierbar (WIDE vs NARROW
  0.89 in recur-Zone), post-recur weggespült (seite13-analog). Mechanik entkoppelt
  von Voice (270m: Footprint ohne Voice). recurON_vs_BASELINE lean-konfundiert.
- **Q-DIR (4b, OFFEN):** seite15-Richtung auf 4b generalisierbar? → seite19b.

**观察 NICHT gekrönt, 顽空 NICHT weggelesen.** Mechanischer recur-Footprint ist
real (seite13 + jetzt kreuz-modell 270m), aber weder hinreichend für Voice (270m)
noch für 观 (Q4 introspektiv-vs-assoziativ offen). Voice ist skalen-abhängige
generische Fähigkeit. Richtung (seite15) ist 1b-gehalten, 4b-Generalisierung offen.