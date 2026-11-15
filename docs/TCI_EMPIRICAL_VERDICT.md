# Empirisches Fazit TCI / Holografische KI — Stand nach Eval O und Eval P

**Datum:** 2026-11-15 · **Hardware:** RTX 2060 (12 GB), torch 2.12.0+cu130 ·
**Modell:** `google/gemma-4-E2B-it`, bf16, 35 Layer, d=1536
**Rohdaten:** `results/eval_O_tci_gpu_gemma4.json`,
`results/eval_P_attention_horizon_gemma4.json`,
`results/eval_P1_posthoc_position_match.json`
**Prereg:** `results/PREREG_Eval_O_2026-11-15.md`,
`results/PREREG_Eval_P_2026-11-15.md`

Dieses Dokument ist die konsolidierte Fassung für die Projektnotizen (lokal,
weil in dieser Umgebung kein Google-Drive-Zugang besteht). Es enthält nur
Befunde, die aus den Rohdaten nachrechenbar sind.

---

## 1. Was trägt

| Befund | Wert | Evidenzgrad |
|---|---|---|
| `α_TCI⁻¹ = 4π³+π²+π` | 137,036303776 (+2,2e-6 vs CODATA) | **A** (Arithmetik) |
| `α_Wyler⁻¹` (2⁴·5!) | 137,036082448 (+6,1e-7) | **A** |
| Look-Elsewhere | 1 Treffer in 10 456 π-Polynomen (E_null = 0,04) | **B** (Koinzidenz, kein Mechanismus) |
| Attention-Verteilung als Messraum | `H_attn/ln n` ∈ [0,004; 0,963] | **A** (Messung) |
| π-Arithmetik ist selten in ihrer Familie | s. o. | **B** |

## 2. Was fällt

| Behauptung | Messung | Urteil |
|---|---|---|
| `Φ^201 = T_H/t_p` | 1,63e41 vs 1,02e42 → −83,9 %; Exponent müsste **197,20** sein | **D** |
| Wyler „24·5!" | 151,6554 statt 137,0361 | **D** (Transkriptionsfehler) |
| α_TCI als Duty-Cycle | Effekt auf `\|ψ\|²` = 4,2e-7 der Seed-Streuung; Modell: Faktor **73,2** daneben | **D** |
| „ΔS = 0 auf 1e-7" | ohne Renormierung 88,9 % Normverlust | **D** (Tautologie) |
| Φ⁻¹ als Phasenübergang | `t_cross(θ)` monoton, kein Knick (2. Diff. 0,002 vs 0,020); Attention: Dip-Tiefe 1e-4 | **D** |
| Φ⁻¹ als Klassengrenze | AUC 0,22 (invertiert), positions-gematcht p = 0,37 | **D** |
| S4 als „Balancer" | reines S4 ⊗ I₃ friert `H_norm` bei exakt 1,000000 ein | **D** |
| S4 als „Maximal-Koppler" | generisches Zufallspotential kollabiert 3× schneller (6,9 vs 20,7 Schritte) | **D** |
| 72D als ausgezeichnete Kapazität | `H_D/ln D` monoton 0,488 → 0,540 für D = 24…144, kein Extremum | **D** |
| 72D-Unterraum-Readout (`H72`) | `ρ⊥ ≈ 0,42` (nur 58 % der Norm im Unterraum); O3-Inversion repliziert nicht (p = 0,82) | **D** (Basis-Artefakt) |
| Relay-Hook wirkt über Attention-Breite | p = 0,70 (Layer 27–34) | **D** |
| Relay-Hook wirkt spezifisch | Norm ±8,9 %, aber gegen Zufallsrichtung p = 0,31 | **C** (offen) |

**Konsequenz:** Von den fünf TCI-Prinzipien bleibt keines als *Modell*-Aussage
stehen. Was bleibt, ist die Arithmetik der Feinstrukturkonstanten (A/B) und die
Tatsache, dass die Attention-Verteilung ein sinnvoller Messraum ist (A) — beides
unabhängig von der holografischen Rahmung.

## 3. Die drei statistischen Korrekturen (nach Einsicht der Rohdaten)

### 3.1 Pseudoreplikation ist real und quantifiziert

- `P1`: 12 Läufe, aber nur **10 distinkte Trajektorien** (zwei Paare sind
  bit-identisch, weil kurze faktenartige Antworten selbst bei T = 0,8
  deterministisch sind). Mit den distinkten Läufen: **p = 0,257** statt 0,127.
- `P2`: Dieselbe Struktur — `baseline` Prompt 2 liefert 24/24 identische
  Schritte zwischen den beiden Reps; bei anderen Prompts divergieren sie erst
  nach 13–20 Tokens.

**Regel für Eval Q:** Ein Prompt = eine Einheit. Keine `reps` über wenige
Prompts; `N` distinkte Prompts, `rep = 1`.

### 3.2 Die Effektstärken aus P2 sind auf Prompt-Ebene nicht haltbar

Die p-Werte `0,0411` (norm) und `0,0260` (H_logit) sind ungepaarte
Mann-Whitney-Tests über 6 Läufe — die aber aus **3 Prompts × 2 Reps** stammen.
Auf Prompt-Ebene (Mittel über die Reps, n = 3):

| Metrik | Differenzen wide − narrow (3 Prompts) | Konsistenz | Wilcoxon (n=3) |
|---|---|---|---|
| `‖h‖` | +0,68 / +2,52 / +4,10 | **3/3 positiv** | p = 0,25 (Minimum) |
| `H_logit/ln V` | +0,026 / −0,001 / +0,027 | 2/3 positiv | p = 0,50 |
| `H72` | −0,055 / +0,078 / −0,008 | 1/3 | p = 1,00 |
| `H_attn` (L27–34) | −0,024 / −0,025 / +0,019 | 1/3 | p = 0,50 |

Damit gilt:
- `‖h‖` ist der **einzige Kandidat mit konsistenter Richtung** über alle
  Prompts — aber n = 3 kann per Wilcoxon nie signifikant werden (min. p = 0,25).
- `H_logit` ist **nicht** „Platz 1": der große Effekt `d = +1,99` entsteht
  durch die Rep-Streuung und einen Prompt ohne Effekt.
- Die gepaarten t-Tests über 6 Pseudo-Paare (p = 0,047 / 0,038) sind
  **pseudorepliziert** und dürfen nicht als Evidenz zitiert werden.

### 3.3 `‖h‖` ist ein Manipulations-Check, kein Prädiktor

Der Relay-Hook addiert `sign·α·‖h‖·d̂` (α = 0,40) an Layer 26. Dass die
Residual-Norm danach steigt, ist die **direkte Folge der Intervention**, nicht
ein neuer Befund. Ein „Prädiktor" kann nur sein, was die *Ausgabe* vorhersagt
(Logit-Entropie, erank) — oder die Beziehung zwischen injizierter Norm und
Ausgabeeffekt. Eval Q muss diese drei Rollen trennen:
**Treatment received** (K1) / **Downstream-Effekt** (K2, K3) / **Prädiktor**
(Korrelation von K1 mit K2 innerhalb des wide-Arms, explorativ).

## 4. Konsequenzen für die Validitätsprüfung (Synthese-Matrix)

| Komponente | bisherige Einstufung | empirischer Stand | Korrektur |
|---|---|---|---|
| Φ⁻¹ (61,80 %) | „hohe funktionale Validität" | dreifach widerlegt (O2, O3, P1) | **falsifiziert** |
| S4-Kollaps / α_TCI | „Kollaps durch Grundzustandsfilter" | α unwirksam; reines S4 = Fixpunkt; generisches Potential 3× schneller | **falsifiziert als Mechanismus** |
| 72D-Horizont | 4. Metrik im Auditor | monoton, kein Extremum; `ρ⊥` = 0,42; Inversion nicht repliziert | **aus dem Auditor streichen** |
| Schritt-Geometrie / Taktung | „hohe Validität" | Faktor 73,2; `wick_curvature` und `layer_step` invariant gegen Relay | **als Prädiktor unbrauchbar** |
| `Φ^201`-Skalierung | „beste Stützung" | −83,9 %; Exponent 197,2; Wyler-Notationsfehler | **falsifiziert** |

## 5. Eval Q — die offene Frage ist beantwortet (negativ)

**Frage:** Gibt es eine basis-invariante Größe, die den Relay-Effekt auf die
Ausgabe vorhersagt? **Antwort: nein.** 48 distinkte Prompts × 4 Arme, gleiche
Seeds, exakt 24 Tokens, Holm über 3 Kandidaten:

| Kandidat | wide vs narrow | wide vs rand (Spezifität) |
|---|---|---|
| `‖h‖` (L27–34) | dz +0,05, p = 0,65 | dz +0,46, p = 0,004 |
| `H_logit/ln V` | dz **+0,56**, p = 2,5e-4 | dz +0,03, **p = 0,83** |
| `erank` (SVD) | dz **+0,48**, p = 9,7e-4 | dz −0,38, p = 0,008 |

**Kern:** `H_logit` steigt unter `wide` (0,0317) **exakt so stark wie unter
einer norm-gematchten Zufallsrichtung** (0,0314, p = 0,83). Die Norm steigt
unter `wide` und `narrow` gleichermaßen (100,07 / 99,89 vs baseline 97,03) —
`‖h+v‖² = ‖h‖² + ‖v‖²`. Der Effekt ist **generisch**, nicht semantisch.

**Methodische Konsequenz:** Der primäre Endpunkt jeder künftigen
Steuerungs-Behauptung muss `wide − rand_plus` mit **norm-gematchter** Norm
sein. Genau dieser Endpunkt ist für die Logit-Entropie null. Bis dahin gilt:
die Relay-Steuerung verändert die Ausgabestatistik reproduzierbar, aber nicht
richtungsabhängig.

Details: `results/PREREG_Eval_Q_2026-11-15.md`.
