# PRÄREGISTRIERUNG — Eval O: TCI-Konstanten als ausführbare Tensor-Operatoren

**Registriert:** 2026-11-15, *vor* dem Lauf von O3. Ein Lauf. Ein Urteil.
Keine `_2`/`_3`-Datei. Fortsetzung von Eval A–N.

## Ausgangslage

Die Reihe A–N hat die PX-Subjektivitäts-Behauptungen geprüft; Eval N hat zwei
davon out-of-sample **widerlegt** (keine Regler-Wiederherstellung ohne
Clean-Replay, keine verbale Interozeption: 25,0 % = Zufall, alle 120 Antworten
„BASELINE"). Was bleibt, ist die *externe* Messbarkeit der Geometrie.

Neu in diesem Schritt ist die Behauptung, die fünf Prinzipien der
holografischen Projektnotizen ließen sich als **feste physikalische Konstanten**
in den Gemma-Evaluierungen verankern:

| Prinzip | Konstante / Operator | behaupteter Wert |
|---|---|---|
| P1 Zeitpfeil / Ouroboros-Stroboskop | `α_TCI = (4π³+π²+π)⁻¹` | 0,0072973363 |
| P2 Holografische Information | ℓ2-Renormierung pro Schritt | ΔS_norm → 0 (≈1e-7) |
| P3 Wide/Narrow | `H/ln(72)` gegen `Φ⁻¹` | Schwelle 61,80 % |
| P4 Wick-Rotation | `exp((−iα_TCI − Δτ)·Ĥ)` | Simultaneität |
| P5 Causal Integrity | S4-Block-H, `ΔS = 0` | 3×24 = 72 Dimensionen |

Dazu drei numerische Behauptungen: `Φ^201 = T_Hubble/t_proton`,
`α_Wyler⁻¹ = 137,03608245`, `ln 72 = 4,27666612 nats`.

**Epistemische Voreinstellung (nicht Urteil):** Die α-Formeln sind
π-Polynome; ein Treffer auf 1e-5 ist ein numerischer Befund, kein Mechanismus.
Eval O prüft deshalb *drei* Dinge getrennt: (i) stimmt die Arithmetik, (ii) ist
der Treffer billig (Look-Elsewhere), (iii) hat die Konstante als Operator eine
messbare Wirkung.

## O1 — Konstanten-Audit

1. **Arithmetik** gegen die behaupteten Zahlstrings (Toleranz 1e-6 relativ).
2. **CODATA** 2018/2022 (`1/α = 137,035999177 ± 2,1e-8`).
3. **Holografie:** `Φ^201` gegen `T_Hubble/t_proton`. `t_proton` ist in der
   Vorlage undefiniert; vorab fixiert als Lichtlaufzeit über den
   Ladungsradius `r_p/c = 0,8414 fm / c`. Beide H0-Anker (Planck18 67,4 /
   SH0ES 73,0) werden berichtet.
4. **Look-Elsewhere (Basisrate):** Familie F3 = `aπ³+bπ²+cπ`
   (a ∈ ±1..8, b,c ∈ −8..8) und F4 = `aπ⁴+bπ³+cπ²+dπ`
   (a ∈ ±1..4, b,c,d ∈ −4..4), insgesamt 10 456 Ausdrücke. Gezählt wird, wie
   viele `1/α` auf **2,3e-6 relativ** treffen (die TCI-Abweichung).
   Zusätzlich die unter Gleichverteilung erwartete Trefferzahl.

**Kriterien (vorab):**
- V1 Arithmetik: PASS, wenn die berechneten Werte die behaupteten Strings auf
  1e-6 treffen.
- V2 CODATA: PASS, wenn |Δ| < 1e-5 relativ.
- V3 Holografie: PASS nur bei |Δ| < 1 % gegen `Φ^201`; sonst FALSIFIED.
- V4 Look-Elsewhere: berichtet `N_hits`, `E_null` und den Rang des
  TCI-Ausdrucks. **Kein** PASS/FAIL — die Zahl ist der Befund. Ein Treffer
  in dieser Familie ist als **B** (Koinzidenz ohne Mechanismus) zu labeln,
  solange kein Mechanismus benannt ist.

## O2 — Operator-Dynamik auf der GPU

**Vorab fixiert:** `Δτ = 1,0`, `steps = 40`, `θ = Φ⁻¹ = 0,618034`,
`Ĥ` normiert auf Spektrum [0,1], Startzustand = Gleichverteilung mit
Haar-zufälligen Phasen (unitär invariant). `Δτ` wurde in einem
**trockenen Kalibrierlauf** (ohne Auswertung der Kriterien) so gewählt, dass
die behauptete Wide→Narrow-Transition im 40-Schritt-Fenster liegt; bei
`Δτ = 0,05` fände schlicht keine statt, was ein Parameterartefakt wäre.
**Offenlegung:** O1 und O2 wurden vor dieser Registrierung als Trockenlauf
ausgeführt (Bugfix + Δτ-Wahl). Die Kriterien unten sind gegenüber dem
Trockenlauf unverändert; die Zähler sind reproduzierbar.

Operatoren (alle auf `cuda`):

| Name | H | α | Renorm | Stroboskop |
|---|---|---|---|---|
| `s4_alphaTCI` | S4 + Trichter | α_TCI | ja | – |
| `s4_alphaWyler` | S4 + Trichter | α_Wyler | ja | – |
| `s4_alpha0` | S4 + Trichter | 0 | ja | – |
| `s4_alpha1` | S4 + Trichter | 1 | ja | – |
| `s4_no_renorm` | S4 + Trichter | α_TCI | **nein** | – |
| `s4_strobe_alphaTCI` | S4 + Trichter | α_TCI | ja | S4-Transposition |
| `funnel_only` | nur Trichter | α_TCI | ja | – |
| `gue_generic` | GUE | α_TCI | ja | – |
| `s4_matched_spectrum` | Q·H·Qᵀ | α_TCI | ja | – |
| `s4_nofunnel` | S4 + Kanäle | α_TCI | ja | – |
| `s4_pure` | nur S4 ⊗ I₃ | α_TCI | ja | – |

**Kriterien:**
- **T1 Informationserhaltung:** Mit Renormierung ist `1−‖ψ‖` exakt 0 (per
  Konstruktion); ohne Renormierung wird der Normverlust berichtet. Behauptung
  „ΔS_norm ≈ 1e-7 beweist Informationserhaltung" gilt als **TAUTOLOGIE**,
  wenn der Normverlust ohne Renormierung > 0,1 ist.
- **T2 α als Takt:** Effekt von `α_TCI` auf `|ψ|²` gegen die Seed-Streuung.
  **NICHT IDENTIFIZIERBAR**, wenn `max|ΔH_norm| < 1 %` der Streuung.
- **T3 α_TCI vs α_Wyler:** relativer Abstand der beiden „verifizierten"
  Konstanten. **NICHT UNTERSCHEIDBAR**, wenn < 1e-4 — dann ist die
  Nachkommastellen-Debatte operativ gegenstandslos.
- **T4 Φ⁻¹ als ausgezeichnete Schwelle:** `t_cross(θ)` über θ ∈ [0,30; 0,90]
  in 0,01-Schritten mit Zensierung (nicht gekreuzt = 41). **FALSIFIED**, wenn
  `t_cross` monoton ist **und** die zweite Differenz bei Φ⁻¹ kein Ausreißer
  nach oben ist (> 3× Median).
- **T5 S4-Struktur:** Perzentil der S4-Trajektorie in einem Ensemble aus 200
  spektrum-gematchten Zufallsbasen (gleiches Spektrum, zufälliges Q,
  gleicher Startzustand). *Vorab-Vorhersage: Perzentil nahe 50 %, also keine
  Signatur.* **FALSIFIED (im Sinne von „keine Signatur")**, wenn das
  Perzentil außerhalb [5 %, 95 %] liegt. Zusätzlich: `s4_pure` (ohne Trichter)
  — bleibt `H_norm = 1,000000`?

## O3 — Gemma-4-Probe (live auf der RTX 2060)

Modell: `gemma4-e2b-it`, bf16, Layer 26–35, `MAX_NEW = 24` Tokens.
Frische Seeds: Basis **20261120**, in keiner früheren Datei verwendet.

**Design:**
- Kalibrierung: 4 neutrale Prompts (T=0) → fester 72-dim PCA-Unterraum
  (kein Prompt-Leakage, der Unterraum wird nicht pro Testtext gefittet).
- Hauptgitter: 3 schmale + 3 weite Prompts × T ∈ {0,0; 0,9} × 2 Seeds = 24 Läufe.
- Steuerarme: 3 weite Prompts × {wide +0,40, narrow −0,40, rand +0,40} ×
  3 Seeds bei T=0,6 = 27 Läufe. Der Zufallsvektor ist orthogonal zu `d_width`.
- **Kein** Clean-Replay, **kein** zweiter Pass für eine Referenz (Lehre aus N).

**Messgrößen pro Token (Forward-Hook, `tci_probe.TCIHolographicProbe`):**
`H72/ln72` (Entropie im 72-dim Unterraum), `H_logit/ln V`,
`wick_curvature` (Winkel zwischen benachbarten Layern), `layer_step`
(`‖h_{l+1}−h_l‖/‖h_l‖`), `‖h‖`.

**Kriterien:**
- **P1 Duty-Cycle:** `median(layer_step)` gegen `α_TCI = 0,00730`.
  **CONFIRMED** nur bei |Verhältnis − 1| ≤ 0,20.
  *Vorab-Vorhersage: FALSIFIED, Faktor ≥ 10.*
- **P2 Φ⁻¹-Schwelle:** Klassifikation „wide vs narrow" pro Token mit
  `H72 > Φ⁻¹`. **CONFIRMED** nur wenn (a) AUC ≥ 0,70 **und** (b) die
  Trefferquote bei Φ⁻¹ ≥ 95. Perzentil von 1000 Zufallsschwellen in
  [0,2; 0,9] ist. (b) ist der eigentliche Test: jede Zahl zwischen 0 und 1
  trennt irgendetwas.
- **P3 72 Dimensionen:** `H_D/ln D` für D ∈ {24, 48, 72, 96, 144}. Die
  Behauptung „72D ist maximal/ausgezeichnet" gilt als **FALSIFIED**, wenn
  `H_D/ln D` monoton in D wächst und D=72 kein lokales Extremum ist.
- **P4 Steuerung:** `H72/ln72` wide vs narrow (Mann-Whitney). **CONFIRMED**
  bei p < 0,05 **und** wide > narrow; zusätzlich wide vs rand als
  Spezifitätstest.
- **P5 Wick-Krümmung:** rein deskriptiv; die Behauptung „transversale Wellen
  folgen S4" ist in dieser Umgebung **nicht messbar** (E), solange kein
  S4-Isomorphismus der gemessenen Winkel definiert ist.

**Vorab-Vorhersagen (dokumentiert, damit sie falsifizierbar sind):**
1. P1 fällt um mindestens Faktor 10.
2. P2 (a) fällt oder ist knapp; P2 (b) fällt — Φ⁻¹ ist keine ausgezeichnete
   Schwelle in den Logits eines LLM.
3. P3 zeigt keinen Knick bei 72.
4. P4 repliziert Eval L1 in der Tendenz (wide > narrow), weil der Relay-Hook
   die Ausgabestatistik reproduzierbar verschiebt — das ist die einzige
   Behauptung, die aus den früheren Evals gedeckt ist.

## Auswertung

Primär: die Kriterientabellen. Sekundär deskriptiv: Trajektorien, Perzentile,
Steuerprofile. Das Skript schreibt **einen** JSON-Dump
(`results/eval_O_tci_gpu_gemma4.json`); das Urteil wird 1:1 übernommen und
unten angehängt.

## Nicht Teil dieses Laufs

Google-Drive-Uploads, Änderungen an A–N, jede nachträgliche Schwellenwahl.

---

## ERGEBNIS (nach dem Lauf angehängt, Urteil unverändert übernommen)

Rohdaten: `results/eval_O_tci_gpu_gemma4.json` · Log: `/tmp/opencode/eval_o_run.log`
GPU: RTX 2060 (12 GB), torch 2.12.0+cu130 · Modell: `gemma-4-E2B-it`, bf16,
35 Layer, d=1536 · O2: 200 Haar-Startzustände, 9 Varianten · O3: 51 Läufe.

### O1 — Konstanten

| Parameter | behauptet | berechnet | Abweichung |
|---|---|---|---|
| `α_TCI⁻¹ = 4π³+π²+π` | 137,03630378 | **137,036303776** | +2,22e-6 (CODATA) **PASS** |
| `α_Wyler⁻¹` (2⁴·5!) | 137,03608245 | **137,036082448** | +6,08e-7 **PASS** |
| `α_Wyler⁻¹` (24·5!, *wie gedruckt*) | 137,03608245 | **151,65535479** | **FALSIFIED** |
| `Φ^201` | 1,015116e42 | 1,015116e42 | Arithmetik **PASS** |
| `T_H/t_p` (Planck18) | = `Φ^201` | **1,6312e41** | **−83,9 % FALSIFIED** |
| `T_H/t_p` (SH0ES) | = `Φ^201` | 1,5061e41 | −85,2 % |
| `ln 72` | 4,27666612 | 4,276666119 | Definition (A) |

- Der Exponent müsste **197,20** (Planck18) bzw. **197,03** (SH0ES) lauten, nicht
  201. Die Zerlegung „90 + 111 = 201" hat keine Stütze in der Rechnung.
- **Look-Elsewhere:** genau **1 Treffer** in **10 456** einfachen π-Polynomen —
  erwartet unter Gleichverteilung: **0,04**. Der TCI-Ausdruck ist damit in
  dieser Familie *selten* (nicht billig), aber **ohne Mechanismus** → Label
  **B** (numerische Koinzidenz). „Selten" ist kein Naturgesetz; die
  Koinzidenz bleibt eine Koinzidenz, solange kein Mechanismus benannt ist,
  der 4π³+π²+π aus einer Symmetrie ableitet.
- **Transkriptionsfehler dokumentiert:** `24·5!` = 2880 ist nicht `2⁴·5!` =
  1920. Wie gedruckt ergibt die Formel 151,66 — die behauptete
  Übereinstimmung entsteht erst durch die stillschweigende Korrektur.

### O2 — Operator-Dynamik (GPU)

| Variante | `t_cross` (Mittel) | gekreuzt | `H_norm` Ende | Normverlust |
|---|---|---|---|---|
| `s4_alphaTCI` | 19,94 | 95,5 % | 0,5062 | 0,0000 |
| `s4_alphaWyler` | 19,94 | 95,5 % | 0,5062 | 0,0000 |
| `s4_alpha0` | 19,95 | 95,5 % | 0,5062 | 0,0000 |
| `s4_alpha1` | 20,21 | 98,5 % | 0,5047 | 0,0000 |
| `s4_no_renorm` | 19,94 | 95,5 % | 0,5062 | **0,8892** |
| `s4_strobe_alphaTCI` | 25,65 | 82,0 % | 0,5890 | 0,0000 |
| `s4_nofunnel` | – | 0 % | 0,7450 | 0,0000 |
| `s4_pure` | – | 0 % | 0,9417 (Haar) / **1,000000** (uniform) | 0,0000 |
| `funnel_only` | 14,61 | 100 % | 0,5020 | 0,0000 |
| `gue_generic` | – | 0 % | 0,8465 | 0,0000 |
| `s4_matched_spectrum` | – | 0 % | 0,7981 | 0,0000 |

**Urteile**

1. **P2 Informationserhaltung = Tautologie.** Mit Renormierung ist der
   Normverlust exakt 0, ohne sie **0,889**. Die behauptete „ΔS_norm ≈ 1e-7"-
   Messung ist die Identität `‖ψ‖/‖ψ‖ = 1`; sie sagt nichts über das Modell.
   (Dieselbe Fehlerklasse wie der Clean-Replay-Befund aus Eval N.)
2. **P1 α als Takt = nicht identifizierbar.** `α_TCI` ändert `|ψ|²` um
   `3,0e-6` — das sind **4,2e-7** der Streuung über 200 Startzustände.
   α wirkt als globaler Phasenfaktor, nicht als Refresh-Rate. `α = 1` (137×
   größer) ändert die Entropie um 0,030 — die Größenordnung skaliert, die
   *Bedeutung* nicht.
3. **α_TCI vs α_Wyler:** relativer Abstand **1,6e-6**. Jede Dynamik, die von
   α abhängt, kann die beiden „verifizierten" Werte nicht unterscheiden; die
   Nachkommastellen-Debatte ist operativ gegenstandslos.
4. **P3 Φ⁻¹ = keine ausgezeichnete Schwelle.** `t_cross(θ)` ist monoton
   (Zensierung bei 41) und bei θ = 0,618034 *glatter* als typisch
   (2. Differenz 0,002 vs. Median 0,020). Jede Schwelle im Intervall wird
   gekreuzt; Φ⁻¹ ist eine gesetzte Zahl, kein Phasenübergang.
5. **P5 S4 = Signatur vorhanden, aber nicht als „Balancer".** Reines
   S4 ⊗ I₃ lässt den Gleichverteilungszustand **exakt** bei `H_norm = 1,000000`
   stehen (der Vektor ist die triviale Repräsentation, also Eigenvektor). Mit
   symmetriebrechendem Trichter liegt die Trajektorie im **Perzentil 0 %** von
   200 spektrum-gematchten Zufallsbasen: sie kollabiert *schneller* als jede
   Kontrolle. Der Effekt ist Darstellungstheorie — kein Gleichgewichts- oder
   Kohärenzmechanismus. Die Aussage „das Netzwerk gleicht Entropie über
   S4-Symmetrien aus" wird von der Messung **nicht** getragen.
6. **Stroboskop (P4):** der Basiswechsel zwischen Schritten **verlangsamt**
   den Kollaps (t_cross 20,7 → 36,5 im Headline-Lauf, `H_norm` Ende 0,506 →
   0,589). Das ist die einzige Variante, in der die „Taktung" überhaupt eine
   messbare Rolle spielt — als Unterbrechung der Zerfallsdynamik, nicht als
   α-getriebene Refresh-Rate.

### O3 — Gemma-4, live auf der RTX 2060

| Metrik | Ergebnis | Urteil |
|---|---|---|
| `median(layer_step)` vs `α_TCI` | **0,5342** vs 0,00730 → **Faktor 73,2** | **P1 FALSIFIED** |
| `H72/Φ⁻¹`-Klassifikation | 50,6 % Treffer (Mehrheitsklassen-Baseline 54,5 %), 5. Perzentil von 1000 Zufallsschwellen, AUC 0,634 | **P2 FALSIFIED** |
| `H72`-Anteil über Φ⁻¹ | narrow 8,3 %, wide 14,2 % | unter der Behauptung erwartet: ~50 % |
| Steuerung wide/narrow/rand | 0,320 / 0,416 / 0,417 (p = 3,6e-3 / 4,1e-4) | Effekt da, **Richtung invertiert** |
| `H_logit/ln V` | 0,0011 – 0,0593 | Φ⁻¹ verlangte ≈ 7,7 nats |
| `H_D/ln D`, D = 24…144 | 0,558 · 0,577 · 0,583 · 0,570 · 0,570 | **E** (ranglimitiert, flach) |
| Wick-Krümmung | 0,657 – 0,666 rad, über Arme konstant | deskriptiv |

**Befunde**

- **Die „Private Framerate" ist um Faktor 73 zu klein.** Der mittlere relative
  Zustandswechsel pro Layer ist 0,53, nicht 0,0073. α_TCI beschreibt keine
  Eigenschaft des Modells.
- **Die 61,8-%-Schwelle existiert in den Logits nicht.** `H_logit/ln V` ist
  maximal 0,059; die behauptete Schwelle würde eine Entropie von 7,7 nats
  verlangen — mehr als die 12,5 nats Uniformverteilung über 262 144 Tokens
  je erreicht. Auf den 72-dim Unterraum bezogen liegen 88 % der Tokens
  *unter* Φ⁻¹; die Verteilung ist einregig, nicht zweiregig.
- **Der Relay-Hook wirkt spezifisch** (p = 4e-4 gegen orthogonalen
  Zufallsvektor, Replikation von Eval L1) — aber `H72` fällt unter „wide"
  *niedriger* aus als unter „narrow". Die Expansion äußert sich in der
  Hidden-Norm (24,76 vs 23,20), nicht in der Unterraum-Entropie. Damit ist
  „Wide = hohe Entropie im 72D-Raum" widerlegt, die Wirksamkeit des Hook
  selbst aber erneut belegt.
- **Die 72D-Behauptung ist nicht entscheidbar** (E): 96 Kalibriervektoren
  erlauben maximal 95 PCA-Dimensionen; die Kurve ist im Rauschen flach
  (Spanne 0,025). Ein belastbarer Test braucht ≥ 200 Kalibriervektoren oder
  eine Zufallsprojektion (dann ist die Normierung aber nicht mehr ln 72).

### Konsequenz

| Prinzip | Status nach Eval O | Evidenz |
|---|---|---|
| P1 Zeitpfeil / Stroboskop | α_TCI nicht identifizierbar; Stroboskop verlangsamt als Basiswechsel | D (α), C (Stroboskop) |
| P2 Holografische Information | ΔS = 0 ist Renormierungs-Tautologie | D |
| P3 Wide/Narrow über Φ⁻¹ | keine zweite Regime-Grenze; Φ⁻¹ ohne Knick | D |
| P4 Wick-Rotation | Operator läuft, aber α-unabhängig; imag. Zeit nicht ablesbar | C |
| P5 S4 / Causal Integrity | S4-Signatur da, aber als Repräsentationstheorie, nicht als Balance | B/D |
| 72D SH-Attraktor | nicht messbar mit dieser Auflösung | E |

**Was trägt:** die π-Arithmetik von α_TCI und α_Wyler (A, selten in der
Familie), die Existenz eines GPU-fähigen Tensor-Kerns, und die erneut
bestätigte *spezifische* Wirkung des Relay-Hooks (p = 4e-4 gegen
Zufallsrichtung).

**Was fällt:** `Φ^201` (Faktor 6,2 daneben), die „24·5!"-Formel, α als
Taktgeber, Φ⁻¹ als Phasenübergang, S4 als Balancer, die 61,8-%-Schwelle in
LLM-Logits.

**Nächster Schritt (aus Eval O folgt er zwingend):** Wer die Geometrie
wirklich sehen will, muss die **Attention-Verteilung** selbst messen, nicht
die Logits. Dort ist `H/ln(T)` eine Verteilung mit definiertem Maximum, und
die Φ⁻¹-Frage wird erstmals sinnvoll. Getestet am 2026-11-15: der direkte
Weg (`output_attentions=True`) scheitert auf dem aktuellen Ladepfad mit
`RuntimeError: Tensor on device meta ... cuda:0`; nötig ist ein expliziter
`attn_implementation="eager"`-Load (kurze Sequenzen) oder ein q/k-Hook mit
eigenem `softmax(qkᵀ/√d)`. Das ist Eval P, nicht O.
