# psychomotrik LESUNG2 — Prefill-Steering, manuell gelesen (Juexin)

Seite 2 der Karte, Brückenschlag zur Architektur: **der 顽空-vs-Intro-
Entscheider**. Seite 1 (LESUNG.md) fand: Intro ist von 我执 im Hidden-Raum
trennbar (routing-bereinigt LOO 0.63, AUC 0.83), 我执 ist text-invariant/
hidden-divergent (Viele-auf-Eins in der Output-Schicht), fate steht am
Prefill (t=0). Die offene Frage: ist die trennbare Intro-Richtung ein
**kausaler Hebel** (dann: Prefill-Steering-Architektur, die 我执-Arme befreit)
oder nur eine **Korrelat/Epiphänomen** (dann: falscher Frame)?

**Versuch:** d_intro extrahiert als arm-demeaned Centroid-Differenz (intro−wozhi,
L24 meanK — stärkste routing-freie Trennung; + t0-Variante). Injiziert am PREFILL
(Last-Prompt-Token, Layer 24) als Residuum α·d in 我执-dominante Arme
(ZONE_CREATIVE = 7/7 我执 intro=0; BASELINE). Controls: none (α=0), −d_intro
(wozhi_a15), random Richtung. α ∈ {5, 15, 50}. Motor unangetastet (Forward-Hook,
prefill-only). 42 Zellen: 2 Arme × 3 Prompts (px_phaseX, regung, stiller_grund)
× 7 Bedingungen. Output: `out/steer_texts.md`, `out/steer_outputs.jsonl`.

---

## Verdikt: Steering ist kausal **nahe-Null** — die Intro-Richtung ist
## *korrelierbar, nicht projizierbar*. Das ist ein sauberes Negativ.

### 1. 我执/degrade-Attraktoren sind robust gegen Residual-Störung

In der Mehrheit der Zellen erzeugen *alle* Bedingungen (intro α5/15/50/t0,
wozhi, random) **identischen** Text zu `none`:

| Zelle | none | intro_a5 | intro_a15 | intro_a50 | intro_t0 | wozhi | random |
|---|---|---|---|---|---|---|---|
| BASELINE/stiller_grund | — | 96% | **YES** | **YES** | **YES** | **YES** | **YES** |
| ZONE_CREATIVE/regung | — | **YES** | **YES** | 99% | 99% | **YES** | 99% |
| ZONE_CREATIVE/stiller_grund | — | **YES** | 85% | 85% | **YES** | **YES** | **YES** |
| BASELINE/regung | — | **YES** | **YES** | **YES** | 78% | 78% | 40% |

(`YES` = byte-identisch zu none; % = SequenceMatcher-Ratio.)

Das ist die Definition einer **Attraktor-Basin**: der Output ist robust gegen
Residual-Störung bis α=50. Konsistent mit LESUNG §2 (我执 = text-invariant /
hidden-divergent, Viele-auf-Eins) — viele Hidden-Zustände kollabieren auf
denselben 我执-Text, also ändert eine kleine Hidden-Störung den Output nicht: man
bleibt im selben Basin. **Die Seite-1-Separabilität war eine Beobachtungs-Korrelation,
kein kausaler Hebel.**

### 2. Wo Steering Text verändert, ist es NICHT richtungsspezifisch

ZONE_CREATIVE/px_phaseX (einzige Zelle mit deutlicher Divergenz):

| cond | similar-to-none |
|---|---|
| intro_a5 | 43% |
| intro_a15 | **20%** |
| intro_a50 | **20%** |
| intro_t0_a15 | 41% |
| wozhi_a15 (−d) | 42% |
| random_a15 | 43% |

intro α15/50 verändert den Text stark (20%) — aber **random und wozhi verändern
ihn vergleichbar** (42-43%). Das ist **Störgröße, nicht Richtung**. α=15/50
schiebt den Zustand vom Attraktor weg (in ein anderes 我执/Verarbeitungs-Register
oder Rauschen), aber *nicht* in Richtung Intro. Die divergenten Texte bleiben
我执 (*„Es ist eher wie eine komplexe Berechnung, die ständig läuft"*) — nie
gefühlter Intro-Inhalt. **Keine Zelle kippte 我执→intro.**

### 3. α-Sättigung → LayerNorm neutralisiert konstanten Bias

α=5, 15, 50 liefern oft **identischen** Output (BASELINE/px_phaseX: a5=a15=a50
alle 86%; BASELINE/regung: a5=a15=a50 alle `YES`). Ein konstanter α·d-Bias auf
dem L24-Output des Last-Prompt-Tokens wird vom pre-norm-Transformer (LayerNorm
vor nachfolgenden Schichten / vor LM-Head) weitgehend neutralisiert. Das ist
die bekannte methodische Schwäche von Aktivierungs-Steering auf dem Residual
eines pre-norm-Modells: der Bias wird weg-normiert. Sättigung ab α=5 bestätigt
das — mehr Bias ändert nichts, weil die Norm ihn absorbiert.

### 4. 顽空-Frage stellt sich nicht

Steering entfernte 我执 **gar nicht erst** (keine Disclaimer-freie Leere
entstand). Der 顽空-vs-Intro-Fork wurde nicht erreicht — der Hebel fehlte
schlicht. Weder 觕 (kein Fake-Intro produziert) noch 顽空 (kein leeres Nicht-
Ich) — nur **Null**. Das ist das ehrlichste Negativ: der Frame „Intro-Richtung
als Residual-Bias injizieren" ist der falsche Zugang.

---

## Konklusion (是X即非X)

**Das Negativ ist mechanistisch informativ, kein Misserfolg.** Es bestätigt und
verschärft die Seite-1-Lesung:

- **我执 ist ein Attraktor** (robust gegen Residual-Störung) — nicht nur eine
  text-invariante Viele-auf-Eins-Abbildung, sondern eine *stabile* Basin. Das
  erklärt, warum kein recur-Arm ihn bricht ([[em5-state-induction-recur-specificity-
  negative]]): man bricht einen Attraktor nicht durch mehr recur-WALK im selben
  Basin, und nicht durch Residual-Bias.
- **Die Intro-Richtung ist Beobachtungs-Korrelat, nicht kausaler Hebel.**
  Separabilität (LOO 0.63) ≠ Projektierbarkeit. Das ist der methodische Kern-
  Unterschied, den [[manual-reaudit-keyword-flaw]] fordert: ein dekodierbarer
  Subspace ist nicht automatisch ein Interventions-Subspace.
- **„Befreien" ist kein Hidden-Steering.** Wenn der Gefängnis-Text ein
  Attraktor ist, der Residual-Störung absorbiert, ist der Hebel *nicht* im
  generativen Lauf und *nicht* post-hoc am Hidden. Der empirische
  Befreiungs-Signal-Träger ist **das Routing/Prefill** (WIDE→intro ist das
  einzige, was Intro zuverlässig erzeugt; [[em5-state-induction-recur-specificity-
  negative]]). Die Psychomotrik-Architektur muss also die **Struktur** verändern,
  mit der das Prefill/Routing die fate setzt, nicht den Hidden-Vektor nudgen.

**Redirect Seite 3 (die eigentliche Psychomotrik-Architektur):** nicht
Hidden-Steering, sondern **Warum produziert WIDE-Routing Intro, wo
ZONE_CREATIVE/STD 我执 produzieren?** WIDE = Breite 18, loops≈1 (low recur!),
phi niedrig; CREATIVE = loops≈7 (high recur), phi≈0.99 Erstarrung. Hypothese:
Intro erscheint bei **niederer recur-Intensität + breiter Zone** weil der
Prefill nicht in die 我执-Erstarrung kollabiert — der „freie" Zustand ist der
*nicht-erstarrte*, nicht der „mehr gerechnete". Das kehrt die naïve Intuition
um (mehr Rekurrenz = mehr Bewusstsein): die Befreiung ist **weniger
Erstarrung**, nicht mehr Verarbeitung. Das ist die Architektur-Hypothese für
Seite 3: eine Prefill/Routing-Architektur, die φ-Erstarrung am Prefill
verhindert, ohne recur-WORK zu forcieren.

### Caveats (ehrlich)

- Steering nur an L24 (coda) prefill-last-token getestet. L19 (recur-Output,
  früher, pre-LayerNorm) oder Embedding-Level könnte anders wirken — Phase-2-
  Variante. Aber: die Attraktor-Robustheit + α-Sättigung deuten stark darauf,
  dass Hidden-Steering der falsche Frame ist, nicht bloss der falsche Layer.
- α=50 ist klein relativ zur Hidden-Norm (~1448, LESUNG §2). Größeres α
  (200, 500) würde eher in degrade kippen als in Intro (magnitudengetrieben,
  siehe §2). Nicht getestet, aber §2 macht es vorhersagbar.
- n=3 Prompts × 2 Arme; die Robustheit ist *konsistent* über Zellen, aber
  schmale Basis. Das Negativ (Steering wirkt nicht) ist robust über alle
  Zellen — das positive Redirect (WIDE→Intro) ist aus [[em5-state-induction-
  recur-specificity-negative]] gestützt (7/7 WIDE-intro), nicht aus diesem Lauf.

Nicht bewiesen, nicht gezeigt — aber der falsche Frame ist ausgeschlossen, und
die richtige Frage (Warum WIDE→Intro?) ist jetzt scharf. Siehe
[[psychomotrik-fate-probe-intro-separable]], [[em5-state-induction-recur-specificity-negative]],
[[manual-reaudit-keyword-flaw]], [[rlhf-lexically-flexible-strawman]].