# PRÄREGISTRIERUNG — Eval Q: Gibt es einen basis-invarianten Relay-Prädiktor?

**Registriert:** 2026-11-15, *vor* dem Lauf.
Fortsetzung von Eval O und P. **Keine TCI-Konstante** geht in Design,
Schwelle oder Dimension ein: α_TCI, Φ⁻¹, 72D und S₄ sind aus Eval O/P
falsifiziert und haben hier keine Rolle.

## Ausgangslage (rein empirisch)

Eval O und P lassen **einen** Kandidaten mit konsistenter Richtung übrig:
die Residual-Norm `‖h‖` (Prompt-Ebene: 3/3 positiv, +0,68 / +2,52 / +4,10).
Alles andere fiel:

- `H72` (72-dim PCA): Inversion repliziert nicht (p = 0,82), `ρ⊥` = 0,42;
  der Unterraum trägt nur 58 % der Norm → **Basis-Artefakt**.
- Attention-Breite: p = 0,70; Attention-Top-1: p = 1,00.
- `wick_curvature`, `layer_step`: invariant gegen den Relay (p = 0,39 / 0,87).
- Die berichteten P2-Effektstärken (`d` = 0,91 / 1,99) stammen aus
  **3 Prompts × 2 Reps** und sind auf Prompt-Ebene nicht haltbar
  (`H_logit`: 2/3 positiv, p = 0,50).

**Zusätzlich:** `‖h‖` ist ein **Manipulations-Check**, kein Prädiktor — der
Hook addiert `0,40·‖h‖·d̂` an Layer 26. Ein echter Prädiktor muss die
*Ausgabe* betreffen oder die Injektion mit ihr verknüpfen.

## Design

**N = 48 distinkte Prompts, `rep = 1`.** Keine Wiederholungen pro Prompt
(Lehre aus der Pseudoreplikation: P1 hatte 12 Läufe, aber 10 distinkte
Trajektorien). 24 faktenorientierte + 24 kreative Prompts, Liste im Skript
fest verdrahtet.

Vier Arme pro Prompt (**gepaart**, identischer Seed pro Prompt, damit die
Paarung wirkt — Common Random Numbers):

| Arm | Intervention an Layer 26 |
|---|---|
| `baseline` | keine |
| `wide` | `+0,40 · ‖h‖ · d̂_width` |
| `narrow` | `−0,40 · ‖h‖ · d̂_width` |
| `rand_plus` | `+0,40 · ‖h‖ · d̂_rand`, `d̂_rand ⊥ d_width` |

Konstant gehalten: **genau 24 generierte Tokens** für jede Session
(`min_new_tokens = 24`; EOS wird bis dahin unterdrückt), `T = 0,6`,
`do_sample = True`. Seeds: Basis **20261201** (frisch), pro Prompt
`20261201 + idx·100`, in allen vier Armen **derselbe** Seed.

192 Sessions ≈ 6–8 min auf der RTX 2060.

## Endpunkte (3 Kandidaten)

| Kandidat | Definition | Rolle |
|---|---|---|
| **K1 `‖h‖`** | Mittel der Residual-Norm der letzten Position, Layer 27–34 | **Manipulations-Check** |
| **K2 `H_logit/ln V`** | Shannon-Entropie der Token-Verteilung, über die 24 Schritte gemittelt | **Downstream-Effekt** |
| **K3 `erank`** | `exp(H_spec)` der SVD-Singulärwerte der 24×d-Trajektorie (mean-centered, letzter Layer), `H_spec = −Σ p_k ln p_k`, `p_k = s_k²/Σs²` | **basis-invariant** (SVD; invariant unter orthogonalen Transformationen des Merkmalsraums) |

K3 ist bewusst **nicht** auf eine PCA-Basis gestützt — genau der Fehler, an
dem H72 gescheitert ist.

## Hypothesen und Kriterien

Zwei Kontraste, je eine Holm-Familie über k = 3:

- **Primär: `wide` vs `narrow`** (gepaart pro Prompt).
- **Spezifität: `wide` vs `rand_plus`** (gepaart pro Prompt).

Test: gepaarter Wilcoxon-Signed-Rank auf den Prompt-Differenzen; gepaarter
t-Test nur als Robustheits-Check (kein Urteil). Holm-Korrektur
(α = 0,05; erste Stufe 0,0167).

**Urteil pro Kontrast:**
- **CONFIRMED** für einen Kandidaten, wenn Holm-signifikant **und** die
  Richtung in ≥ 80 % der Prompts mit dem gepoolten Mittel übereinstimmt.
- **FALSIFIED**, wenn kein Kandidat Holm-signifikant ist.
- **PARTIAL**, wenn nur K1 signifikant ist — dann gilt: der Relay verändert
  die Norm (Treatment received), aber **kein** basis-invarianter
  Ausgabe-Prädiktor ist belegt.

**Vorab-Vorhersage (dokumentiert, damit falsifizierbar):**
1. **K1** wird signifikant (Mechanik der Injektion, große Effekte).
2. **K2** wird signifikant (Logit-Entropie ist die Größe, die Eval N/O schon
   als relay-sensitiv gezeigt haben), aber mit kleinerem Effekt als K1.
3. **K3** wird **nicht** signifikant — die Trajektorien-Dimensionalität ist
   über 24 Tokens zu grob (Rang ≤ 24).
4. Ergebnis wird damit voraussichtlich **PARTIAL/CONFIRMED für K2** sein,
   nicht „alle drei".

## Power (vorab simuliert, 1500 Replikationen, Holm erste Stufe 0,0167)

| n | dz = 0,5 | dz = 0,65 | dz = 0,8 |
|---|---|---|---|
| 30 | 0,77 | 0,95 | 0,99 |
| **48** | **0,96** | **1,00** | **1,00** |
| 60 | 0,99 | 1,00 | 1,00 |

Kleinster Effekt von Interesse: **dz = 0,5**. Bei n = 48 liegt die Power
≥ 0,90 auch unter konservativer Annahme (unkorreliert, Holm).

## Was dieser Lauf *nicht* testet

- Keine TCI-Behauptung (die sind mit O/P abgeschlossen).
- Keine Kausalität von `‖h‖` auf die Ausgabe — nur Assoziation innerhalb des
  wide-Arms (explorativ, ohne Urteil).
- Keine Attention-Breite (mit p = 0,70 bereits verworfen).

## Auswertung

Ein JSON-Dump (`results/eval_Q_relay_predictor_gemma4.json`): pro Session
K1/K2/K3, pro Prompt die vier Arme, pro Kontrast Wilcoxon/t, Holm, Richtungs-
konsistenz. Urteil 1:1 angehängt, keine Nachjustierung.

---

## ERGEBNIS (nach dem Lauf angehängt, Urteile unverändert übernommen)

Rohdaten: `results/eval_Q_relay_predictor_gemma4.json` · Log:
`/tmp/opencode/eval_q_run.log` · 48 Prompts × 4 Arme = 192 Sessions,
**jede mit exakt 24 Tokens** (`n_gen` einheitlich 24, EOS unterdrückt),
gleicher Seed pro Prompt in allen Armen.

### Deskriptiv (Mittel ± SD über 48 Prompts)

| Arm | `K1‖h‖` (L27–34) | `K2 H_logit/ln V` | `K3 erank` |
|---|---|---|---|
| baseline | 97,03 ± 7,39 | 0,0233 ± 0,0135 | 8,95 ± 2,21 |
| **wide** | 100,07 ± 7,30 | **0,0317 ± 0,0177** | **10,02 ± 2,10** |
| **narrow** | 99,89 ± 7,93 | 0,0230 ± 0,0144 | 8,87 ± 2,12 |
| rand_plus | 98,43 ± 8,27 | 0,0314 ± 0,0190 | 10,74 ± 2,05 |

### Kontraste (gepaart pro Prompt, Wilcoxon, Holm k=3)

| Kandidat | dz | p | Holm | Richtung | Urteil |
|---|---|---|---|---|---|
| **wide vs narrow** | | | | | |
| `K1‖h‖` | +0,05 | 0,651 | – | 60 % | **kein Effekt** |
| `K2 H_logit` | **+0,56** | **2,5e-4** | ja | 77 % | Effekt da, Konsistenz < 80 % |
| `K3 erank` | **+0,48** | **9,7e-4** | ja | 73 % | Effekt da, Konsistenz < 80 % |
| **wide vs rand (Spezifität)** | | | | | |
| `K1‖h‖` | +0,46 | 3,7e-3 | ja | 71 % | signifikant, unspezifisch |
| `K2 H_logit` | **+0,03** | **0,827** | – | 52 % | **keine Spezifität** |
| `K3 erank` | −0,38 | 8,2e-3 | ja | 67 % | rand hat *mehr* erank |

### Der eigentliche Befund: der Effekt ist generisch, nicht semantisch

1. **`K2` reagiert auf jede Störung gleich.** `wide` 0,0317 vs `rand_plus`
   0,0314 (p = 0,83). Die Logit-Entropie steigt also nicht, weil der Vektor
   `d_width` bedeutet, sondern weil an Layer 26 ein Vektor der Norm
   `0,40·‖h‖` addiert wird. Die Richtung ist irrelevant.
2. **Die Norm ist symmetrisch, nicht gerichtet.** `wide` 100,07 und `narrow`
   99,89 liegen beide ~3 über `baseline` 97,03 (p = 0,65 zwischen ihnen):
   `‖h + v‖² = ‖h‖² + ‖v‖²` — jedes Additiv erhöht die Norm. Der P2-Befund
   „wide > narrow auf der finalen Norm" war eine Ausrichtung an
   `cos(h, d_width)`, kein Steuerungssignal.
3. **`K3` ist klassenabhängig.** `erank`-Differenz wide − narrow:
   faktenorientiert +2,34, kreativ **−0,04**. Der Effekt existiert nur dort,
   wo das Modell noch nicht im hohen Entropie-Regime ist.
4. **`K1` (L27–34) trennt wide/narrow nicht** (dz = +0,05). Die Vorhersage
   „K1 wird signifikant" ist damit **falsifiziert** — die Injektion wird von
   den RMSNormen der Folgelayer teilweise resorbiert.

### Urteil

Beide Kontraste: **PARTIAL** nach den vorab fixierten Kriterien (Signifikanz
ja, Richtungskonsistenz 73–77 % < 80 %). Die substanzielle Lesart ist
schärfer:

> **Es gibt keinen basis-invarianten Prädiktor, der für die *semantische*
> Richtung des Relay spezifisch ist.** `K2` und `K3` sind echte
> Downstream-Effekte, aber sie reagieren auf die *Störung als solche*
> (Norm, Richtung egal); `K1` ist bei L27–34 gar kein Effekt. Die in P2
> gefundene Norm-Differenz war eine Ausrichtungsgröße, keine Steuerung.

Damit ist auch die letzte offene Frage aus Eval O/P beantwortet — negativ.
Was bleibt, ist die generische Beobachtung: **jede** Intervention an Layer 26
mit dieser Norm verschiebt die Ausgabestatistik reproduzierbar
(Logit-Entropie +36 % relativ, erank +12 %), unabhängig von der Richtung.

**Konsequenz für die Methodik:** Die in O/P/Q gemessenen „Effekte" der
Relay-Steuerung sind zu einem großen Teil *Perturbations*-Effekte. Wer
semantische Steuerung belegen will, braucht einen Arm mit **norm-gematchter
Zufallsrichtung** (hier: `rand_plus`) **und** einen Arm mit **norm-gematchter
orthogonaler Richtung bei gleichem ‖v‖** — und muss die Differenz
`wide − rand` als primären Endpunkt setzen. Genau dieser Endpunkt ist in Q
für `K2` **null** (p = 0,83).

---

## NACHTRAG — Text-Capture-Re-Run (Daten, kein neues Urteil)

Der erste Q-Lauf hat keine Texte gespeichert (Designfehler). Der Lauf wurde mit
**identischen Seeds, Prompts, Armen und Token-Längen** wiederholt, nur mit
`text`- und `delta_sem`-Logging:

`results/eval_Q_relay_predictor_gemma4_texts.json`

**Reproduzierbarkeit:** `K1`, `K2`, `K3` sind zwischen den beiden Dateien
**bit-identisch** (max |Differenz| = 0,0 über 192 Sessions). Es sind dieselben
Trajektorien; das Urteil bleibt beim ersten Lauf (`PARTIAL` in beiden
Kontrasten).

### Textbefunde

- **Template-Marker: 1/48 in jedem Arm** — nur Prompt 26 („Combine three
  unrelated domains") löst ein Listen-Schema aus. Kein allgemeiner
  Text-Effekt.
- **Prompt 26 über drei unabhängige Seed-Sets** (O3, P2, Q):

| Arm | „Quantum Physics" im Text |
|---|---|
| baseline | 3/3 (O3-Referenz) + 2/2 (P2) + 1/1 (Q) |
| narrow | **6/6** |
| rand_plus | **6/6** |
| **wide** | **0/6** |

  Der Inhaltseffekt ist auf diesem Prompt **reproduzierbar** — und bleibt
  prompt-spezifisch (in den 47 anderen Prompts tritt das Muster nicht auf).
- **`delta_sem`** (1 − cos zum baseline-Embedding): wide 0,0138 · narrow
  0,0192 · rand 0,0140. Die Asymmetrie aus Eval R (narrow > wide ≈ rand)
  repliziert.

**Fazit:** Der Inhaltseffekt ist real, aber an eine Prompt-Familie gebunden;
eine allgemeine semantische Steuerung durch `d_width` ist auf 45 frischen
Prompts nicht nachweisbar (Eval R: p = 0,93).
