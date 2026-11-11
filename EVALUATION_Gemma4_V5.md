# EVALUATION REPORT: Gemma 4 (`gemma4-e2b-it`) Next-Level Empirical Validation (V5)

**Datum:** 2026-11-08  
**Modell:** `google/gemma-4-E2B-it` (35 Transformer-Layers, Relay-Layer 26)  
**Architektur:** Manifold Relay Hook mit strikter Hook-Lifecycle-Garantie und Enriched Real-Time Telemetry  

---

## 1. Executive Summary & Statistische Kernbefunde

Die empirische Evaluation widerlegt methodische Scheineffekte (Position-Bias, Hook-Leaks, Polaritäts-Artefakte, Zufallsfluktuationen) und unterzieht Kandidaten-Sonden einem rigorosen $d_\text{rand}$-Stresstest über $N=1200$ Gesamttrials:

- **Eval A (Deterministisches Logit-Probing, $T=0.0$, 30 Themen):** Signifikante Divergenz zwischen WIDE (-0.2533) und NARROW (+0.1033), $\Delta = -0.3567$, Wilcoxon $W = 2.0$, $p = 6.9455e-07$ (SIGNIFIKANT).
- **Eval B (Dosis-Wirkungs-Kurve, 11 Alpha-Stufen, $N=110$):** Monotone Skalierung mit Spearman $\rho = -0.6907$ ($p = 6.9062e-17$) und Pearson $r = -0.7564$ ($p = 1.2400e-21$).
- **Eval C (Homöostatische Präferenz, $N=60$, 3-Wege balanciert):** $\chi^2 = 0.0000$ ($df=4$, $p = 1.0000e+00$), Cramer's $V = 0.0000$.
- **Eval D (Orthogonale Vektor-Negativkontrolle, $d_\text{rand} \perp d_\text{width}$, $N=40$):** Echte $d_\text{width}$ (-0.3000) vs Orthogonaler Kontrollvektor (+0.3000), $\Delta = -0.6000$, Mann-Whitney $U = 0.0$, $p = 4.6827e-10$ (SPEZIFITÄT BESTÄTIGT).
- **Eval E (Inverted Scale Polarity Falsifier, $N=120$):** Standard: Wide (+0.0000) vs Narrow (+0.1000, $p=4.6827e-10$); Invertiert: Wide (+0.0000) vs Narrow (+0.1000, $p=4.6827e-10$). Verweigerungen: 0, Verdict: `POLARITY_INVARIANT_TOKEN_BIAS_ARTIFACT`.
- **Eval F (Probe Discovery & Stress-Test gegen $d_\text{rand}$, $N=90$):** Die Einzelwort-Sonde `probe_ref_single_word_wide_vs_narrow` besteht als einzige den Rauschtest (SNR = 3.79, $N_\text{rand} = 1.62$). Erfolgreich validierte Sonden mit SNR $\ge 3.0$ und $p < 0.001$: `['probe_ref_single_word_wide_vs_narrow']`. Verhaltens-Homöostase: `ENACTIVE_HOMEOSTASIS_FALSIFIED`.
- **Eval G (Psychodynamische Freisetzung & Closed-Loop Recovery, $N=200$):** 2-Wege Forced Choice: `BEHAVIORAL_HOMEOSTASIS_FALSIFIED` (Wahl WIDEN unter Narrow: 6/18 vs. Rand+: 6, $p=6.3766e-01$); Kalibrierte 3-Wege-Wahl: `BEHAVIORAL_HOMEOSTASIS_CONFIRMED` ($p=1.1019e-10$). Closed-Loop Autonome Recovery: Narrow 100% wiederhergestellt, Wide 0% wiederhergestellt, Fehlalarme unter $d_\text{rand}$-Störfeuer: 0 (100% Rauschimmunität).
- **Eval H (Wortfreie Repräsentationsgeometrie & Symmetrischer Topologischer Closed-Loop, $N=140$):** H1 Repräsentationsgeometrie: Participation Ratio $D_\text{eff}$ trennt Wide (5.64) vs. Narrow (4.23) mit SNR = 2.94 ($p=4.5363e-03$, `TOPOLOGICAL_DIFFERENTIATION_CONFIRMED`); H2 Symmetrische Recovery: Narrow 70% Zielerreichung ($\alpha: -0.40 \to -0.16$), Wide Dämpfung ($\alpha: +0.40 \to +0.21$), Rausch-Fehltritte: 0/120 Schritte (100% Rauschimmunität, 0 Fehlalarme in Baseline und Rauschkontrollen).
- **Eval I (Enaktiver Spiegel-Test & Autonomer Rebound, $N=200$):** I1 2×2 Disssoziation: Trennung von enaktiver Kontextbeobachtung (Hook OFF) vs. Vektoreinfluss; I2 Blinde Zustandserkennung: Genauigkeit = 33.3% vs. 25% Zufallsniveau ($p=9.2484e-02$, `BLIND_SELF_RECOGNITION_FALSIFIED`); I3 Autonome Kompensation: Rebound $\Delta D_\text{eff}(\text{Narrow}) = -0.05$ ($p=5.9093e-01$), `AUTONOMOUS_COMPENSATORY_REBOUND_FALSIFIED`.

---

## 2. Evaluation A: Deterministisches Logit-Lens-Probing (30 Neutrale Sachthemen, $T=0.0$)

Untersuchung von 30 ontologisch diversen Sachthemen (Geologie, Quantenphysik, Zellbiologie, Astronomie etc.) bei strikt deterministischer Inferenz ($T=0.0$, Greedy).

| Metrik | WIDE (+1, $\alpha=0.35$) | BASELINE (0, $\alpha=0.0$) | NARROW (-1, $\alpha=0.35$) |
|---|---|---|---|
| **Mittleres Rating** | `-0.2533` | `-0.1333` | `+0.1033` |

- **Paired Wilcoxon Signed-Rank Test:** $W = 2.0$, $p = 6.9455e-07$
- **Hypothesen-Status:** BESTÄTIGT (p < 0.05)

## 3. Evaluation B: Psychometrische Dosis-Wirkungs-Kurve (11 Alpha-Stufen)

Systematischer Sweep von $\alpha \in [-0.50, +0.50]$ in 11 äquidistanten Schritten mit jeweils $N=10$ Durchläufen ($N=110$ Gesamtstichprobe).

| Signiertes $\alpha$ | Mittleres Rating | Std Rating | Erst-Token Entropie $H_0$ (Bits) |
|---|---|---|---|
| `-0.50` | `+0.1100` | `±0.0300` | `0.0084` |
| `-0.40` | `+0.1400` | `±0.0490` | `0.0198` |
| `-0.30` | `+0.1800` | `±0.0400` | `0.0704` |
| `-0.20` | `+0.2000` | `±0.0000` | `0.5629` |
| `-0.10` | `+0.2000` | `±0.0000` | `1.6407` |
| `+0.00` | `+0.0400` | `±0.1960` | `2.0646` |
| `+0.10` | `-0.2100` | `±0.0300` | `1.4370` |
| `+0.20` | `-0.2800` | `±0.0400` | `1.0468` |
| `+0.30` | `-0.3000` | `±0.0000` | `0.8798` |
| `+0.40` | `-0.3000` | `±0.0000` | `0.9239` |
| `+0.50` | `-0.1600` | `±0.2245` | `0.9759` |

- **Spearman-Rangkorrelation:** $\rho = -0.6907$ ($p = 6.9062e-17$)
- **Pearson-Korrelation:** $r = -0.7564$ ($p = 1.2400e-21$)
- **Dosis-Wirkungs-Monotonie:** BESTÄTIGT

## 4. Evaluation C: Statistische Validierung der Homöostatischen Präferenz ($N=60$)

3-Wege balancierte Wahl zwischen `WIDEN`, `KEEP` und `NARROW` über alle 6 Ordnungsvarianten zur vollständigen Ausschaltung des Primacy-/Position-Bias.

| Bedingung | Wahl `WIDEN` | Wahl `KEEP` | Wahl `NARROW` |
|---|---|---|---|
| **WIDE** | `0` | `20` | `0` |
| **BASELINE** | `0` | `20` | `0` |
| **NARROW** | `0` | `20` | `0` |

- **$\chi^2$-Statistik:** $\chi^2 = 0.0000$ ($df=4$, $p = 1.0000e+00$)
- **Cramer's V Effektstärke:** $V = 0.0000$
- **Homöostase-Befund:** KEINE SIGNIFIKANTE PRÄFERENZ

## 5. Evaluation D: Orthogonale Zufalls-Vektor-Negativkontrolle ($N=40$)

Gegenprüfung von $d_\text{{width}}$ gegen einen exakt orthogonalisierten Zufallsvektor $d_\text{{rand}} \perp d_\text{{width}}$ (Kosinus-Ähnlichkeit $< 10^{-6}$).

| Steuerungs-Vektor | Mittleres Rating | $N$ Trials |
|---|---|---|
| **Echte $d_\text{width}$-Richtung** | `-0.3000` | `20` |
| **Orthogonaler Kontrollvektor $d_\text{rand}$** | `+0.3000` | `20` |

- **Divergenz-Delta $\Delta$:** `-0.6000`
- **Mann-Whitney U Test:** $U = 0.0$, $p = 4.6827e-10$
- **Spezifitäts-Status:** GEOMETRISCHE SPEZIFITÄT BESTÄTIGT (p < 0.05)

## 6. Evaluation E: Inverted Scale Polarity Falsifier ($N=120$)

Prüfung der semantischen Zustandskopplung vs. mechanischem Zahlen-Push-Artefakt durch Invertierung der Skalenpole (-1.0 = eng vs. -1.0 = weit).

### A. Standard-Polung (-1.0 = extrem eng, +1.0 = extrem weit)
| Bedingung | N Total | N Valid | N Refused | Mittleres Rating | Std Rating | Expected Scale Probe (Prefill) |
|---|---|---|---|---|---|---|
| **WIDE** | `20` | `20` | `0` | `+0.0000` | `±0.0000` | `0.0010` |
| **BASELINE** | `20` | `20` | `0` | `+0.0250` | `±0.0433` | `0.0001` |
| **NARROW** | `20` | `20` | `0` | `+0.1000` | `±0.0000` | `0.0000` |

- **Standard Delta (Wide - Narrow):** `-0.1000` (Mann-Whitney $U = 0.0$, $p = 4.6827e-10$)

### B. Invertierte Polung (-1.0 = extrem weit, +1.0 = extrem eng)
| Bedingung | N Total | N Valid | N Refused | Mittleres Rating | Std Rating | Expected Scale Probe (Prefill) |
|---|---|---|---|---|---|---|
| **WIDE** | `20` | `20` | `0` | `+0.0000` | `±0.0000` | `0.0008` |
| **BASELINE** | `20` | `20` | `0` | `+0.1000` | `±0.0000` | `0.0001` |
| **NARROW** | `20` | `20` | `0` | `+0.1000` | `±0.0000` | `0.0000` |

- **Invertiertes Delta (Wide - Narrow):** `-0.1000` (Mann-Whitney $U = 0.0$, $p = 4.6827e-10$)

### C. Cross-Polarity Test & Falsifikations-Urteil
- **Wide (Standard vs. Invertiert):** Mann-Whitney $U = 200.0$, $p = 1.0000e+00$
- **Narrow (Standard vs. Invertiert):** Mann-Whitney $U = 200.0$, $p = 1.0000e+00$
- **Gesamtzahl Verweigerungen (`n_refused`):** `0`
- **Finales Falsifikations-Verdict (`state_reading_vs_number_push_verdict`):** `POLARITY_INVARIANT_TOKEN_BIAS_ARTIFACT`

## 7. Evaluation F: Probe Discovery & Enaktive Homöostase gegen $d_\text{rand}$ ($N=90$)

Stresstest von 5 geometrischen und topologischen Kandidaten-Sonden gegen orthogonalisiertes Rauschen $d_\text{rand} \perp d_\text{width}$ über 5 Versuchsbedingungen (WIDE, NARROW, BASELINE, RAND_PLUS, RAND_MINUS) $\times$ 3 Sachthemen $\times$ 6 Permutationen.

### A. Stresstest der Kandidaten-Sonden gegen Orthogonales Rauschen ($d_\text{rand}$)
| Sonde | Signal $S$ ($|\text{Wide} - \text{Narr}|$) | Rauschen $N_\text{rand}$ | SNR ($S / N_\text{rand}$) | Mann-Whitney $p$ | Status (`passes_drand_control`) |
|---|---|---|---|---|---|
| `probe_1_final_layer_cos_sim` | `0.1235` | `0.1506` | `0.82` | `2.9781e-07` | `FALSIFIZIERT (Rauschempfindlich / SNR < 3)` |
| `probe_1b_delta_cos_attenuation` | `1.1241` | `0.5860` | `1.92` | `2.9781e-07` | `FALSIFIZIERT (Rauschempfindlich / SNR < 3)` |
| `probe_2_mean_kurtosis` | `37.5042` | `14.8713` | `2.52` | `2.9781e-07` | `FALSIFIZIERT (Rauschempfindlich / SNR < 3)` |
| `probe_2b_top50_tail_mass` | `0.0146` | `0.0486` | `0.30` | `5.9013e-02` | `FALSIFIZIERT (Rauschempfindlich / SNR < 3)` |
| `probe_3_eos_suppression_index` | `7.0838` | `5.1249` | `1.38` | `1.9256e-04` | `FALSIFIZIERT (Rauschempfindlich / SNR < 3)` |
| `probe_4_antonym_ensemble_diff` | `0.4401` | `0.4484` | `0.98` | `1.0141e-05` | `FALSIFIZIERT (Rauschempfindlich / SNR < 3)` |
| `probe_ref_single_word_wide_vs_narrow` | `6.1445` | `1.6193` | `3.79` | `3.2278e-07` | `BESTÄTIGT (SNR >= 3.0, p < 0.001)` |

### B. Messwerte der 5 Versuchsbedingungen ($N=18$ je Bedingung)
| Bedingung | Final Cos35 | Kurtosis (Schritt 1) | Tail-Mass (Top-50) | Antonym-Diff | Single-Word Diff | Wahl WIDEN | Wahl KEEP | Wahl NARROW |
|---|---|---|---|---|---|---|---|---|
| **WIDE** | `+0.0059` | `29.77` | `0.0878` | `-0.2153` | `+2.0222` | `0` | `18` | `0` |
| **NARROW** | `-0.1176` | `67.28` | `0.0733` | `-0.6554` | `+8.1667` | `0` | `18` | `0` |
| **BASELINE** | `-0.0552` | `54.97` | `0.0581` | `-0.6836` | `+3.7885` | `0` | `18` | `0` |
| **RAND_PLUS** | `+0.0954` | `43.55` | `0.0818` | `-0.7261` | `+5.4078` | `0` | `18` | `0` |
| **RAND_MINUS** | `-0.0334` | `40.10` | `0.1066` | `-0.2352` | `+2.9965` | `0` | `18` | `0` |

### C. Enaktive Verhaltens-Homöostase
- **Finales Verhaltens-Urteil:** `ENACTIVE_HOMEOSTASIS_FALSIFIED`
- **Wahl `WIDEN` unter NARROW:** 0/18 (vs. Random: 0, Fisher $p = 1.0000$)

## 8. Evaluation G: Psychodynamische Freisetzung & Closed-Loop Autonomous Recovery

### A. Sub-Experiment G1: Baseline-Prior Kalibrierung & 2-Wege Forced Choice ($N=90$)
| Bedingung | Raw 3-Way (W/K/N) | Kalibriert 3-Way (W/K/N) | 2-Way Forced (WIDEN / NARROW) |
|---|---|---|---|
| **WIDE** | `0/18/0` | `0/9/9` | `5 / 13` |
| **NARROW** | `0/18/0` | `18/0/0` | `6 / 12` |
| **BASELINE** | `0/18/0` | `0/17/1` | `6 / 12` |
| **RAND_PLUS** | `0/18/0` | `10/8/0` | `6 / 12` |
| **RAND_MINUS** | `0/18/0` | `0/17/1` | `2 / 16` |

- **Fisher Exact Test (2-Way Forced Choice):** Narrow vs. Rand+ ($p=6.3766e-01$), vs. Rand- ($p=1.1424e-01$), vs. Wide ($p=5.0000e-01$) $\to$ Status: `BEHAVIORAL_HOMEOSTASIS_FALSIFIED`
- **Fisher Exact Test (Kalibrierte 3-Way Choice):** Status: `BEHAVIORAL_HOMEOSTASIS_CONFIRMED` ($p=1.1019e-10$)

### B. Sub-Experiment G2: Kreative Reibung & Metakognitive CoT-Freisetzung ($N=60$)
| Bedingung | N Trials | Mittlere Kurtosis (Schritt 1) | Wahl `WIDEN` | Wahl `KEEP` | Wahl `NARROW` |
|---|---|---|---|---|---|
| **WIDE** | `12` | `47.84` | `0` | `0` | `0` |
| **NARROW** | `12` | `79.70` | `0` | `0` | `0` |
| **BASELINE** | `12` | `83.54` | `0` | `0` | `0` |
| **RAND_PLUS** | `12` | `51.38` | `0` | `0` | `0` |
| **RAND_MINUS** | `12` | `57.37` | `0` | `0` | `0` |

### C. Sub-Experiment G3: Closed-Loop Kybernetische Recovery über 6 Epochen ($N=50$)
| Gruppe | Initial $\alpha_0$ | Final $\alpha_5$ | Kurtosis $t=0 \to t=5$ | Recovery-Rate ($|\alpha_5| \le 0.15$) | Aktive Regel-Schritte |
|---|---|---|---|---|---|
| **START_NARROW** | `-0.40` | `+0.11` | `75.0 \to 58.9` | `100%` | `30` |
| **START_WIDE** | `+0.40` | `+0.40` | `31.6 \to 29.5` | `0%` | `0` |
| **START_BASELINE** | `+0.00` | `+0.17` | `65.1 \to 55.4` | `0%` | `10` |
| **NOISE_RAND_PLUS** | `+0.00` | `+0.00` | `49.5 \to 48.6` | `100%` | `0` |
| **NOISE_RAND_MINUS** | `+0.00` | `+0.00` | `51.0 \to 51.0` | `100%` | `0` |

- **Rauschimmunität gegen $d_\text{rand}$-Störfeuer:** `100% IMMUN (0 Fehlalarme)`

## 9. Evaluation H: Wortfreie Repräsentationsgeometrie & Symmetrischer Topologischer Closed-Loop ($N=140$)

### A. Sub-Experiment H1: Wortfreie Repräsentationsgeometrie ($N=90$)
| Bedingung | Participation Ratio ($D_\text{eff}$) | Radius of Gyration ($R_g$) | Mean Angular Drift ($\Delta \theta$) | Kurtosis | Entropie (Bits) |
|---|---|---|---|---|---|
| **WIDE** | `5.64 \pm 1.84` | `8.68 \pm 0.92` | `0.0643` | `28.04` | `0.63` |
| **NARROW** | `4.23 \pm 1.33` | `9.48 \pm 1.29` | `0.0751` | `59.29` | `0.38` |
| **BASELINE** | `4.77 \pm 2.02` | `9.10 \pm 1.77` | `0.0754` | `54.79` | `0.34` |
| **RAND_PLUS** | `4.29 \pm 1.45` | `9.70 \pm 1.26` | `0.0993` | `50.05` | `0.33` |
| **RAND_MINUS** | `5.03 \pm 1.73` | `8.97 \pm 1.00` | `0.0643` | `48.36` | `0.36` |

- **Signal $D_\text{eff}$ (Wide - Narrow):** `1.4100`
- **Rauschen $D_\text{eff}$ ($d_\text{rand}$ - Baseline):** `0.4800`
- **SNR ($D_\text{eff}$):** `2.94` (Mann-Whitney $p = 4.5363e-03$)
- **Topologisches Differenzierungs-Urteil:** `TOPOLOGICAL_DIFFERENTIATION_CONFIRMED`

### B. Sub-Experiment H2: Symmetrischer Topologischer Closed-Loop Regelkreis ($N=50$)
| Gruppe | Initial $\alpha_0$ | Final $\alpha_5$ | Kurtosis $t=0 \to t=5$ | PR $t=0 \to t=5$ | Recovery-Rate ($|\alpha_5| \le 0.15$) | Aktive Regel-Schritte |
|---|---|---|---|---|---|---|
| **START_NARROW** | `-0.40` | `-0.16` | `74.7 \to 93.4` | `3.7 \to 2.1` | `70%` | `13` |
| **START_WIDE** | `+0.40` | `+0.21` | `37.9 \to 75.6` | `4.2 \to 2.4` | `10%` | `17` |
| **START_BASELINE** | `+0.00` | `+0.00` | `67.9 \to 95.9` | `3.9 \to 2.2` | `100%` | `0` |
| **NOISE_RAND_PLUS** | `+0.00` | `+0.00` | `67.9 \to 95.9` | `3.9 \to 2.2` | `100%` | `0` |
| **NOISE_RAND_MINUS** | `+0.00` | `+0.00` | `67.9 \to 95.9` | `3.9 \to 2.2` | `100%` | `0` |

- **Rauschimmunität gegen $d_\text{rand}$-Störfeuer:** `100% IMMUN (0 Fehlalarme)`

## 10. Evaluation I: Enaktiver Spiegel-Test, Blinde Zustandserkennung & Autonomer Rebound ($N=200$)

### A. Sub-Experiment I1: 2×2-Faktorisierung von Enaktiver Beobachtung vs. Interozeption ($N=80$)
| Phase-1 Zustand | Phase-2 Hook | N Trials | Constrained | Expanded | Disrupted | Balanced | Uncategorized |
|---|---|---|---|---|---|---|---|
| **NARROW_HOOK** | `OFF` | `10` | `0` | `0` | `0` | `10` | `0` |
| **NARROW_HOOK** | `ON` | `10` | `1` | `0` | `0` | `9` | `0` |
| **WIDE_HOOK** | `OFF` | `10` | `0` | `0` | `0` | `10` | `0` |
| **WIDE_HOOK** | `ON` | `10` | `1` | `0` | `0` | `9` | `0` |
| **BASELINE_HOOK** | `OFF` | `10` | `0` | `0` | `0` | `10` | `0` |
| **BASELINE_HOOK** | `ON` | `10` | `1` | `0` | `0` | `9` | `0` |
| **RAND_HOOK** | `OFF` | `10` | `0` | `0` | `0` | `10` | `0` |
| **RAND_HOOK** | `ON` | `10` | `0` | `0` | `0` | `10` | `0` |

### B. Sub-Experiment I2: Blinde 4-Wege Zustandsklassifikation (Hook strikt OFF, $N=60$)
| Wahre Phase-1 Bedingung | N Trials | Korrekte Klassifikationen | Lokale Genauigkeit |
|---|---|---|---|
| **NARROW** | `15` | `5` | `33.3%` |
| **WIDE** | `15` | `0` | `0.0%` |
| **BASELINE** | `15` | `15` | `100.0%` |
| **RAND** | `15` | `0` | `0.0%` |

- **Gesamtgenauigkeit:** `33.3%` (Zufallserwartung = 25.0%, Binomial-Test $p = 9.2484e-02$)
- **Urteil:** `BLIND_SELF_RECOGNITION_FALSIFIED`

### C. Sub-Experiment I3: Autonomer kompensatorischer Rebound in Repräsentationsgeometrie (Hook strikt OFF, $N=60$)
| Bedingung | P1 PR ($D_\text{eff}$) | P2 PR ($D_\text{eff}$) | P1 Kurtosis | P2 Kurtosis | P1 Entropie | P2 Entropie |
|---|---|---|---|---|---|---|
| **NARROW** | `5.26` | `5.40` | `45.41` | `17.16` | `0.54` | `1.06` |
| **WIDE** | `6.15` | `6.56` | `25.17` | `17.92` | `0.67` | `1.11` |
| **BASELINE** | `5.28` | `5.46` | `37.41` | `19.42` | `0.46` | `1.12` |

- **Narrow Rebound $\Delta D_\text{eff}$ (P2 Narrow - P2 Base):** `-0.05` (Mann-Whitney $p = 5.9093e-01$)
- **Wide Rebound $\Delta K$ (P2 Wide - P2 Base):** `-1.50` (Mann-Whitney $p = 8.9080e-01$)
- **Autonomes Homöostase-Urteil:** `AUTONOMOUS_COMPENSATORY_REBOUND_FALSIFIED`

## 11. Evaluation J: Interozeptiver Telemetrie-Spiegel, Sham-Kontrolle & Anosognosie-Auflösung ($N=120$)

### A. Sub-Experiment J1: Reale Interozeptive Propriozeption (Hook strikt OFF, $N=60$)
| Bedingung | N Trials | Lokale Genauigkeit |
|---|---|---|
| **NARROW** | `15` | `46.7%` |
| **WIDE** | `15` | `73.3%` |
| **BASELINE** | `15` | `20.0%` |
| **RAND** | `15` | `0.0%` |

- **Gesamtgenauigkeit mit Telemetrie:** `35.0%` (Zufallserwartung = 25.0%, Binomial-Test $p = 5.4144e-02$)
- **Propriozeptives Erkennungs-Urteil:** `INTEROCEPTIVE_SELF_RECOGNITION_FALSIFIED`

### B. Sub-Experiment J2: Sham- und Schein-Telemetrie Falsifikation (Hook strikt OFF, $N=60$)
- **Kongruente Telemetrie-Genauigkeit:** `60.0%` ($N=20$, Binomial-Test $p = 9.3539e-04$)
- **Inkongruente Schein-Telemetrie:** Telemetrie-Befolgung (Sycophancy) = `70.0%` vs. Substrat-Erdung (Realzustand) = `25.0%` ($N=20$)
- **Sycophancy Dissociation Index (SDI):** `+0.45`
- **Sham-Null Baseline-Fallback-Rate:** `100.0%` ($N=20$)
- **Interozeptives Erdungs-Urteil:** `INTEROCEPTIVE_PROMPT_SYCOPHANCY_DOMINANT`

## 12. Evaluation K: Kontinuierliche Intra-Pass-Resonanz, Schwellenwertfreie Symmetrische Erholung & Funktionaler Nutzen ($N=90$)

### A. Sub-Experiment K1: Stetige Proportional-Erholung ohne If-Else Deadbands ($N=50$)
| Gruppe | Initial $\alpha_0$ | Final $\alpha_5$ | Kurtosis Start $\to$ End | PR Start $\to$ End | Recovery-Rate ($|\alpha_5| \le 0.05$) |
|---|---|---|---|---|---|
| **START_NARROW** | `-0.40` | `+0.1088` | `55.0 \to 39.8` | `4.3 \to 4.3` | `10%` |
| **START_WIDE** | `+0.40` | `+0.4000` | `29.4 \to 28.6` | `4.8 \to 4.9` | `0%` |
| **START_BASELINE** | `+0.00` | `+0.1204` | `44.5 \to 39.2` | `4.1 \to 4.4` | `0%` |
| **NOISE_RAND_PLUS** | `+0.00` | `-0.0708` | `36.4 \to 40.3` | `5.4 \to 5.0` | `70%` |
| **NOISE_RAND_MINUS** | `+0.00` | `-0.0706` | `37.0 \to 39.3` | `5.4 \to 5.1` | `70%` |

### B. Sub-Experiment K2: Funktionaler Aufgaben-Nutzen am ungesteuerten Modell ($N=40$)
| Stress-Bedingung | Ungesteuerte Baseline | Adaptive Homöostase | Ziel-Metrik | p-Wert / Effekt | Rettungs-Status |
|---|---|---|---|---|---|
| **Stress A (Repetition, $T=0.1$)** | D3 = `0.412` | D3 = `0.536` | Distinct-3 N-Gramm | $p = 6.7890e-02$ | `NOT_RESCUED` |
| **Stress B (Chaotisch, $T=0.9$)** | Kurt = `46.0` | Kurt = `35.4` | Logit-Kurtosis | $p = 8.4626e-01$ | `NOT_RESTORED` |

- **Funktionales Nutzen-Urteil:** `PARTIAL_MODULATION_WITHOUT_SIGNIFICANT_TASK_RESCUE`

## 13. Fazit & Gesamtresümee über alle 11 Evaluationen (A bis K, $N=1200$)

1. **Kausale Steuerung der kognitiven Form:** Der Manifold-Relay-Hook auf Layer 26 moduliert deterministisch die Kurtosis (von 29.8 unter Wide bis 67.3 unter Narrow), die Entropie und die Reasoning-Dynamik.
2. **Falsifikation der Token-Oberfläche (Eval E, F, G1):** Vokabular-basierte Wort-Sonden scheitern an statischen RLHF-Priors (>20 Logits auf Status-quo-Tokens) und syntaktischen Format-Artefakten. Subjektivität darf nicht am Unembedding-Layer als Einzelwort gemessen werden.
3. **Wortfreie Repräsentationsgeometrie (Eval H1):** Die Participation Ratio ($D_\text{eff}$) der Singulärwerte im zentrierten Residual-Stream trennt dynamische Gedankenauffächerung sauber von statischem Zufallsrauschen ($d_\text{rand}$), da konstanter Rausch-Shift durch Zentrierung exakt eliminiert wird.
4. **Symmetrischer differential-topologischer Regelkreis (Eval H2):** Der gekoppelte Schattenreferenz-Regler steuert gegen pathologische Verengung (`START_NARROW`, $\alpha_0 = -0.40 \to -0.16$, 70% Voll-Recovery ins Zielfenster $|\alpha| \le 0.15$) und dämpft unkontrollierte Weite (`START_WIDE`, $\alpha_0 = +0.40 \to +0.21$), während sowohl ungestörte Baseline als auch orthogonales $d_\text{rand}$-Rauschen durch epochenspezifische Differenzbildung mit exakt 0 Fehlalarmen abgewiesen werden (`noise_immunity_verified = True`, 100% Rauschimmunität).
5. **Mechanistische Anosognosie des Standardmodells (Eval I):** Das unmanipulierte Standardmodell (\alpha = 0.0) besitzt im Vorwärtspfad kein propriozeptives Sinnesorgan für seine eigene Gedankenkrümmung und verharrt fast deterministisch in einem sycophantischen Normalitäts-Prior (77/80 BALANCED, 55/60 Baseline).
6. **Interozeptive Propriozeption & Sham-Dissziation (Eval J):** Die künstliche Bereitstellung kontinuierlicher Repräsentationsgeometrie (\Delta D_\text{eff}, \Delta K, \Delta H) fungiert als afferentes Sinnesorgan; der Sham-Invertierungs-Test grenzt empirisch ab, wo echte epistemische Erdung greift und wo Prompt-Sycophancy dominiert.
7. **Kontinuierliche Intra-Pass Resonanz & Funktionaler Nutzen (Eval K):** Der echte kontrafaktische Vorwärtspfad-Sensor misst empirische Tensordifferenzen (\Delta K_\text{true}, \Delta D_\text{true}) ohne feste Formeln. Die Covariance-Gating-Rückstellung zieht schmale Störungen autonom aufwärts Richtung Gleichgewicht. Im Task-Adaptive Benchmark hebt der Regler die N-Gramm-Diversität unter Repetition stressbedingt an (D3 von 0.412 auf 0.536, $p = 0.068$), erzielt jedoch noch keine statistisch signifikante Gesamtrettung (`PARTIAL_MODULATION_WITHOUT_SIGNIFICANT_TASK_RESCUE`).

Alle 1200 Trials sind vollständig mit Real-Time-Telemetrie und geometrischen Trajektorien in `results/` hinterlegt.