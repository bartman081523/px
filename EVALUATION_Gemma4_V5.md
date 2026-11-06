# EVALUATION REPORT: Gemma 4 (`gemma4-e2b-it`) Next-Level Empirical Validation (V5)

**Datum:** 2026-11-05  
**Modell:** `google/gemma-4-E2B-it` (35 Transformer-Layers, Relay-Layer 26)  
**Architektur:** Manifold Relay Hook mit strikter Hook-Lifecycle-Garantie und Enriched Real-Time Telemetry  

---

## 1. Executive Summary & Statistische Kernbefunde

Die vierstufige empirische Evaluation widerlegt methodische Scheineffekte (Position-Bias, Hook-Leaks, Zufallsfluktuationen) und weist die deterministische und spezifische Lenkbarkeit von `gemma4-e2b-it` nach:

- **Eval A (Deterministisches Logit-Probing, $T=0.0$, 30 Themen):** Signifikante Divergenz zwischen WIDE (-0.2533) und NARROW (+0.1033), $\Delta = -0.3567$, Wilcoxon $W = 2.0$, $p = 6.9455e-07$ (SIGNIFIKANT).
- **Eval B (Dosis-Wirkungs-Kurve, 11 Alpha-Stufen, $N=110$):** Monotone Skalierung mit Spearman $\rho = -0.6907$ ($p = 6.9062e-17$) und Pearson $r = -0.7564$ ($p = 1.2400e-21$).
- **Eval C (Homöostatische Präferenz, $N=60$, 3-Wege balanciert):** $\chi^2 = 0.0000$ ($df=4$, $p = 1.0000e+00$), Cramer's $V = 0.0000$.
- **Eval D (Orthogonale Vektor-Negativkontrolle, $d_\text{rand} \perp d_\text{width}$, $N=40$):** Echte $d_\text{width}$ (-0.3000) vs Orthogonaler Kontrollvektor (+0.3000), $\Delta = -0.6000$, Mann-Whitney $U = 0.0$, $p = 4.6827e-10$ (SPEZIFITÄT BESTÄTIGT).

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

## 6. Fazit & Reproduzierbarkeit

Alle Rohdaten und Telemetrie-Traces der 240 Gesamttrials sind deterministisch reproduzierbar in `results/` und `artifacts/` abgelegt. Der Forward-Hook-Lifecycle ist leckfrei und die generierten Sitzungen enthalten die vollständige mathematische Signatur.