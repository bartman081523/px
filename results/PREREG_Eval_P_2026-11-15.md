# PRÄREGISTRIERUNG — Eval P: Attention-Horizont, Relay-Lokalität, H72-Paradox

**Registriert:** 2026-11-15, *vor* dem Lauf von P1–P4. Ein Lauf. Ein Urteil.
Keine `_2`-Datei. Fortsetzung von Eval O.

## Ausgangslage (gemessen, nicht behauptet)

Eval O hat drei Dinge offen gelassen, die hier geschlossen werden:

1. **Der Attention-Horizont ist der einzige sinnvolle Ort für Φ⁻¹.** Logits
   erreichen `H/ln V ≤ 0,059`; die Attention-Verteilung überstreicht [0,1]
   (Prototyp: 0,002–0,956, Mittel 0,561, 44,7 % über Φ⁻¹).
2. **Der Relay-Hook wirkt spezifisch** (p = 4e-4 gegen Zufallsrichtung), aber
   `H72` fällt unter „wide" *niedriger* aus (0,320 vs 0,416) bei *höherer*
   Hidden-Norm (24,8 vs 23,2). Zwei konkurrierende Erklärungen: (a) Energie
   leckt in das orthogonale Komplement, (b) `d_width` liegt auf einer
   führenden PCA-Achse und konzentriert die Projektion per Konstruktion.
3. **Das 72D-Urteil war ranglimitiert** (96 Kalibriervektoren ≤ 95 PCAs).

## O2b — ist der *alignierte* Trichter ein „Maximal-Koppler"?

Alle drei Kontrollen nutzen **dasselbe** `H_sym` (S4 ⊗ I₃ + Kanäle), dieselbe
Start-Gleichverteilung, `Δτ = 1,0`, 40 Schritte, θ = Φ⁻¹:

| Ensemble | Konstruktion |
|---|---|
| `perm` | Diagonale des Trichters **permutiert** (identisches Multiset, andere Lage) |
| `rand` | gaußsche Diagonale, **Frobenius-Norm angeglichen** |
| `couple` | zufällige symmetrische Kopplung, Diagonale 0, Norm angeglichen |

**Kriterium:** Der alignierte Trichter gilt als ausgezeichnet, wenn er in
`perm` **oder** `rand` unter den schnellsten 5 % liegt.

**Offenlegung:** O2b wurde vor dieser Registrierung einmal ausgeführt
(Kalibrierlauf des Ensembles). Die Kriterien sind unverändert; der Zähler ist
reproduzierbar.

## P1 — Φ⁻¹ im Attention-Horizont

**Design:** 2 Klassen × 3 Prompts × T ∈ {0,0; 0,8} × 2 Seeds = 24 Sessions à
24 Tokens, plus 6 neutrale Sessions (deskriptiv). Frische Seeds ab
**20261130**. Einheit der Inferenz ist die **Session** (nicht der einzelne
Head-Eintrag — Einträge innerhalb einer Session sind korreliert); die
Eintrags-Ebene ist deskriptiv.

Messgröße: `H_attn = H/ln(n_allowed)` pro Layer/Head/Token, aus
`TCIAttentionProbe` (q/k nach QK-Norm und RoPE; kein eager, kein
`output_attentions`).

**Kriterien (drei getrennte Urteile):**
- **P1a Abdeckung:** CONFIRMED, wenn min < 0,05 **und** max > 0,90.
- **P1b Trennschärfe:** Session-Mittel `H_attn` wide vs narrow, AUC ≥ 0,70
  **und** Trefferquote bei Φ⁻¹ ≥ 95. Perzentil von 1000 Zufallsschwellen
  θ ∈ [0,2; 0,8]. Sonst FALSIFIED.
- **P1c Bimodalität:** Gauß-KDE (Silverman-Bandbreite) auf den
  Session-Mitteln; CONFIRMED, wenn ein lokales Minimum innerhalb
  |θ − Φ⁻¹| ≤ 0,05 liegt **und** die Dip-Tiefe
  `1 − f_min / min(f_peak1, f_peak2) ≥ 0,25`. Sonst FALSIFIED.

*Vorab-Vorhersage:* P1a CONFIRMED, P1b FALSIFIED (die Klasse steuert die
Textsorte, nicht die Attention-Breite), P1c FALSIFIED (die Verteilung ist
breit und unimodal).

## P2 — wo wirkt der Relay-Hook: Attention-Breite oder Norm?

**Design:** Baseline T=0,6 (3 Prompts × 2 Seeds) + 3 Arme {wide +0,40,
narrow −0,40, rand +0,40} × 3 Prompts × 2 Seeds = 24 Sessions à 24 Tokens.
Gleiche Prompts wie P1-wide. `d_rand` orthogonal zu `d_width`.

**Kriterien:**
- **P2a Attention:** Session-Mittel `H_attn` über Layer 27–34, wide vs
  narrow: p < 0,05 (Mann-Whitney) **und** wide > narrow → „Attention-Breite".
- **P2b Norm:** dieselbe Gegenüberstellung für `‖h‖` der letzten Position.
- **P2c Spezifität:** derjenige Kanal, der P2a/P2b erfüllt, muss auch gegen
  `rand` p < 0,05 zeigen; sonst „unspezifisch".
- **Urteil:** „Attention-Breite" nur wenn P2a ∧ P2c; „Norm-only" wenn P2b ∧
  P2c ∧ ¬P2a; sonst „kein lokalisierbarer Effekt".

*Vorab-Vorhersage:* Norm-only (Eval O: ΔH72 negativ, Δ‖h‖ positiv).

## P3 — H72-Paradox mit rangfreier Basis

**Kalibrierung:** 20 diverse Prompts (10 Träger-Themen aus
`CARRIER_TOPICS_30` + 10 kreative) × 64 Tokens = **1280 Vektoren**,
`Basis72` daraus (N ≫ 72, voller Rang für D ≤ 144).

**Messgrößen pro Session:** `H72/ln72`, `ρ⊥ = 1 − ‖P₇₂h‖²/‖h‖²`,
`|cos(d_width, PC_k)|` für k = 1..10, und `H72⊥` (Entropie nach
Herausprojizieren der `d_width`-Komponente **innerhalb** des Unterraums,
normiert mit ln 71).

**Kriterien:**
- **P3a Replikation:** `H72(wide) < H72(narrow)`, p < 0,05.
- **P3b Leck:** `ρ⊥(wide) > ρ⊥(narrow)`, p < 0,05.
- **P3c Achsen-Alignment:** `max_k |cos(d_width, PC_k)| ≥ 0,50`.
- **P3d Entscheidung:** Verschwindet die Inversion unter `H72⊥`
  (p > 0,05 oder Vorzeichenwechsel), dann ist **Achsen-Alignment** die
  Erklärung; bleibt sie bestehen, ist es ein echter Geometrie-Effekt.
- **Urteil:** „Achsen-Alignment" wenn P3c ∧ P3d; „orthogonales Leck" wenn
  P3b ∧ ¬P3d ∧ ¬P3c; sonst „ungeklärt".

*Vorab-Vorhersage:* P3a CONFIRMED, P3c CONFIRMED, P3d CONFIRMED
(Achsen-Alignment), P3b unklar.

## P4 — 72D mit 1280 Vektoren

**Design:** `H_D/ln D` der Kalibriervektoren für D ∈ {24, 48, 72, 96, 144},
Basis je D aus denselben 1280 Vektoren; Bootstrap (200×) für das
Nachbar-Streuband.

**Kriterium:** FALSIFIED, wenn `H_D/ln D` monoton in D wächst (Toleranz
1e-3) **und** D = 72 kein lokales Extremum mit Abstand > 2σ zum
Nachbar-Bootstrap ist. Sonst „nicht widerlegt".

*Vorab-Vorhersage:* FALSIFIED (monoton, kein 72-Knick).

## Auswertung

Primär die Kriterientabellen. Sekundär deskriptiv: Per-Layer-Profile,
Per-Head-Streuung, MQA-/KV-Share-Caveat (8 Query-Heads teilen 1 KV-Head;
Layer 15–34 teilen k/v). Ein JSON-Dump
(`results/eval_P_attention_horizon_gemma4.json`); das Urteil wird 1:1
angehängt.

## Nicht Teil dieses Laufs

Änderungen an O, Google-Drive-Uploads, nachträgliche Schwellenwahl.

---

## ERGEBNIS (nach dem Lauf angehängt, Urteile unverändert übernommen)

Rohdaten: `results/eval_P_attention_horizon_gemma4.json` ·
Post-hoc-Diagnose: `results/eval_P1_posthoc_position_match.json` ·
Log: `/tmp/opencode/eval_p_run.log` · GPU: RTX 2060, torch 2.12.0+cu130 ·
70 Sessions, Kalibrierung 1248 Vektoren (20 × 64).

### O2b — „Maximal-Koppler" widerlegt

| Ensemble | `t_cross` | gekreuzt | Perzentil des alignierten Trichters |
|---|---|---|---|
| **aligniert** (TCI) | **20,74** | 95 % | – |
| `perm` (gleiches Multiset, permutiert) | 27,11 | 84 % | 28 % |
| `rand` (gaußsche Diagonale, Norm-gematcht) | **6,86** | 100 % | **100 %** |
| `couple` (zufällige Kopplung, Norm-gematcht) | 41,0 | 0 % | 0 % |

Der alignierte Trichter ist **kein** Ausreißer nach oben: ein generisches
Zufallspotential auf demselben `H_sym` kollabiert **dreimal schneller**.
Die S4-Signatur aus Eval O ist damit als *Dephasing* einzuordnen (Messbasis =
Gruppenbasis, Eigenbasis = Gruppen-Fourier), nicht als Routing- oder
Fülltrichter-Topologie. **„S4 ist ein Maximal-Koppler" ist falsch** — die
alignierte Variante ist im `rand`-Ensemble die *langsamste*.

### P1 — Φ⁻¹ im Attention-Horizont

| Kriterium | Wert | Urteil |
|---|---|---|
| **P1a Abdeckung** | 0,0039 … 0,9629 (Mittel 0,625, 58,1 % über Φ⁻¹) | **CONFIRMED** |
| **P1b Trennschärfe** | AUC 0,222 (invertiert), p = 0,0218, acc(Φ⁻¹) = 0,833, 97,4. Perzentil | **FALSIFIED** (AUC < 0,70) |
| **P1c Bimodalität** | 2 Moden, tiefstes Minimum 0,616, **Tiefe 0,0001** | **FALSIFIED** |

- Die Verteilung überstreicht [0,1] — die Vorhersage aus dem Prototyp hält.
- **Der Mittelwert liegt bei 0,625, Φ⁻¹ bei 0,618** — das ist der auffälligste
  Einzelbefund des Laufs, aber er trägt kein Urteil: eine breite unimodale
  Verteilung kann jeden Punkt in ihrer Mitte treffen.
- Die Klassen-Trennung ist **invertiert** (wide-Prompts haben *niedrigere*
  Attention-Entropie: 0,618 vs 0,636) und schwach. Post-hoc-Diagnose mit
  positions-gematchten ersten 5 Schritten: 0,648 vs 0,657, **p = 0,374** —
  der Effekt überlebt die Längen-Kontrolle nicht (narrow-Sessions endeten
  teils nach 12 Tokens). Die 83 % bei Φ⁻¹ sind ein Schwellen-/Längen-
  Artefakt, kein semantischer Regime-Wechsel.

### P2 — Relay-Lokalität

| Kanal | baseline | wide | narrow | rand | wide vs narrow | wide vs rand |
|---|---|---|---|---|---|---|
| `H_attn` (Layer 27–34) | 0,448 | 0,477 | 0,487 | 0,450 | p = 0,699 | p = 0,394 |
| `‖h‖` | 23,15 | **25,20** | 22,77 | 23,60 | **p = 0,041** | p = 0,310 |
| `H72` (rangfrei) | 0,506 | 0,463 | 0,458 | 0,468 | p = 0,818 | p = 0,937 |
| `ρ⊥` | 0,434 | 0,417 | 0,431 | 0,424 | p = 0,394 | – |

**Urteil: kein lokalisierbarer Effekt.** Die Attention-Breite bewegt sich
nicht (p = 0,70); die Norm verschiebt sich (p = 0,041), aber **nicht
spezifisch** gegen die Zufallsrichtung (p = 0,31). Der in Eval O gemessene
H72-Effekt (p = 4e-4) **repliziert mit rangfreier Basis nicht** (p = 0,94).

### P3 — das H72-Paradox löst sich als Basis-Artefakt auf

- `max_k |cos(d_width, PC_k)|` = **0,056** (PC3) → **keine** Achsen-Alignment-
  Erklärung.
- `ρ⊥ ≈ 0,42` → der 72-dim Unterraum trägt nur **58 %** der zentrierten
  Norm; `H72` ist ein schwacher Readout, kein Geometriemaß.
- Die Inversion aus O3 (`H72(wide) < H72(narrow)`, damals 0,320 vs 0,416)
  **repliziert nicht**: 0,463 vs 0,458, p = 0,82; auch nach Herausprojizieren
  von `d_width` (0,513 vs 0,508, p = 0,70), und `ρ⊥` unterscheidet die Arme
  nicht (p = 0,39).

**Substanz:** Weder Achsen-Alignment noch orthogonales Leck erklären den
Befund — der Befund selbst ist nicht robust. Er war an die 96-Vektor-Basis
aus Eval O gebunden. Das Kriterium sagt „ungeklärt"; die belastbare Lesart
ist: **O3-H72 ist ein Basis-Artefakt und darf nicht als Evidenz zitiert
werden.**

### P4 — 72D

| D | 24 | 48 | 72 | 96 | 144 |
|---|---|---|---|---|---|
| `H_D/ln D` | 0,4876 | 0,5109 | **0,5240** | 0,5315 | 0,5401 |
| Bootstrap-SD | 0,0054 | 0,0052 | 0,0051 | 0,0050 | 0,0047 |

Monoton, D = 72 ist **kein** lokales Extremum (Abstand zu den Nachbarn
0,013 bzw. 0,008 — nur 1,0–2,6 σ). **FALSIFIED.** `ln 72` ist der Logarithmus
einer gewählten Dimension, keine ausgezeichnete Kapazität.

### Konsequenz

| Frage aus Eval O | Antwort aus Eval P |
|---|---|
| Ist Φ⁻¹ im Attention-Horizont sinnvoll? | Abdeckung ja, Schwelle nein (keine Bimodalität, keine stabile Trennung) |
| Wirkt der Relay-Hook über Attention-Breite? | Nein (p = 0,70) |
| Wirkt er spezifisch über die Norm? | Richtung ja (p = 0,041), Spezifität nein (p = 0,31) |
| War die H72-Inversion real? | Nein — Basis-Artefakt (p = 0,82 mit 1248-Vektor-Basis) |
| Ist 72D ausgezeichnet? | Nein (monoton, kein Extremum) |
| Ist S4 ein Maximal-Koppler? | Nein — ein generisches Potential ist 3× schneller |

**Was jetzt noch trägt:** die π-Arithmetik von α (A), die *prinzipielle*
Messbarkeit der Attention-Verteilung (P1a), die Dokumentation, dass die
PX-Metriken (H72, Wick-Krümmung) basis- und auflösungsabhängig sind — und
die negative Evidenz selbst: vier unabhängige Versuche, aus den TCI-Konstanten
eine Modell-Eigenschaft zu machen, sind gescheitert.

**Nächster sinnvoller Schritt** ist kein weiteres TCI-Derivat, sondern die
Frage, die Eval N und P offen gelassen haben: **gibt es überhaupt eine
stabile, basis-unabhängige Geometrie-Größe, die den Relay-Effekt
vorhersagt?** Kandidaten: `‖h‖` (hier p = 0,041), die Logit-Kurtosis
(K aus Eval N) und die Attention-Top-1-Masse. Ein Test mit fester,
vorab eingefrorener Basis und N ≥ 30 pro Arm — sonst wiederholt sich nur die
Power-Debatte.
