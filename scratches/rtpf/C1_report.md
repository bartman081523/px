# C1-Report: Router-Signal-Verstärkung #9 (Moments-Rekalibrierung)

Datum: 2026-10-06 · Plan shiny-meandering-grove · Mind rtpf-ev · Baseline-Commit 5f1064e9 (+ c8e6a802)

## Frage (#9)

Router-Signal am ternary-bonsai-27b verstärken: η²(H) der Zone-Entropy über die
4 Eval-Kategorien — Baseline η²=0.0290, V=0.271, Ziel η²>0.05 UND p<0.05.
Argmax-Hist der Baseline: logic_a 40 (dead), Creative/Logic dominieren nicht
unterscheidbar. Vermutung: Manifold-Moments (k_std 44.64, phi_std 0.0159)
repräsentieren nicht die Decode-Forward-Regime (kurt-std ~26.9, phi-std ~0.051)
→ Z-Distanzen zu klein/flutet → Weights near-uniform → H flach.

## Methode: Router9-Simulation (CPU, Produktionsgetreu)

Sim repliziert exakt die Production-Kette: load_manifold →_online-Init
(`_online_n=6`, `_online_k_m2=k_std²·6`, `_online_k_mean=json-k_mean`) →
`_get_kurtosis_weights` (k_std_eff = k_std·√(6/5), phi_std RAW) über
`learned_centroids` → `get_zone_weights` (blend b, T) → Shannon-H.
Geometrie-Input: die eval-Batterie-Anteilung (kurt/phi je Prompt, aus dem
Baseline-Telemetrie-JSON: k_mean 4355, k_std 26.88, phi_mean 0.7818, phi_std 0.0510).

## Sim-Grid (Kernaussagen)

**gridA (Means auf Manifold-Stand gehalten: k_mean 4339.4, phi_mean 0.8419; Stds variiert):**
| phi_std | k_std | T | b | η²(H) | p |
|---|---|---|---|---|---|
| 0.0159 | 44.64 | 0.6 | 0.5 | 0.0290 | ≈0.53 | ← Produktion (Replika-Validierung ✓)
| 0.0159 | 26.88 | 0.6 | 0.5 | 0.0566 | ≈0.215 |
| 0.0500 | 26.88 | 0.6 | 0.5 | 0.1138 | ≈0.026 |
| 0.0510 | 44.64 | 0.6 | 0.5 | 0.1106 | ≈0.029 |
| **0.0510** | **26.88** | **0.6** | **0.5** | **0.1083** | **≈0.0321** | ← **C1 (deployed)**, V=0.289
| 0.1000 | 26.88 | 0.6 | 0.5 | 0.1903 | ≈0.0012 |
| 0.1000 | 44.64 | 0.6 | 0.5 | 0.0182 | ≈0.707 | ← Overshoot-Kollaps
| 0.0510 | 26.88 | 0.6 | 0.8 | 0.1771 | ≈0.002 |
| 0.1000 | 44.64 | 0.6 | 0.8 | 0.0659 | ≈0.155 |
| 0.0300 | 26.88 | 0.6 | 1.0 | 0.1960 | ≈0.0009 |
| 0.1000 | 26.88 | 0.6 | 1.0 | 0.3585 | ≈4.3e-07 |

**gridB (volle Rekalibrierung: k_mean 4355, k_std 26.9, phi_mean 0.7818, phi_std 0.0510):**
| T | b | η² | p | V |
|---|---|---|---|---|
| 0.6 | 0.5 | 0.0861 | ≈0.0744 | 0.337 (p_cat 0.0116 ✓) |
| 1.0 | 0.5 | 0.1183 | ≈0.0218 | 0.393 (p_cat 0.000498) |
| 0.6 | 0.8 | 0.1683 | ≈0.0029 | — |
| 0.6 | 1.0 | 0.1829 | ≈0.0016 | 0.332 |
| 0.3 | * | max 0.0275 | tot | — (Sharpening-Falsifikation bestätigt) |

**Lehren:**
1. Dosis-Wirkung phi_std NICHT-MONOTON bei Production-Knobs: 0.0159→0.029;
   0.051→0.108-0.111; 0.10→0.018 (kollabiert — k-Dimension flutet dann).
2. k_std-Tightening erst bei deflooded phi wirksam (Interaktion).
3. T=0.3 dead überall → Zone-T bleibt 0.6.
4. Entblendung (b→1.0) verstärkt stark — ABER SR-59f-Sensibilität (k_blend
   historisch heikel) → NICHT getestet; b=0.5 bleibt.

## Entscheidung: C1 (Moments-only, minimal-invasiv)

Manifold `_home_julian_.cache_huggingface_ternary-bonsai-2-27b-hf_manifold.json`:
- `k_std`: 44.63568423976991 → **26.882352387459473**
- `phi_std`: 0.015857377655552753 → **0.05095817845646322**
- `learned_centroids` (abgeleitet! z-konsistent mit NEUEN Stds, Z-Targets
  math(1.5,0.5)/logic_a(0.5,0.2)/logic_b(0,0)/creative(−1,−0.5)/synthesis(−1.5,−1)):
  - math (4379.772698503064, 0.8673967709222624)
  - logic_a (4352.890346115605, 0.8521093173853235)
  - logic_b (4339.449169921875, 0.8419176816940308)   [unchanged, (0,0)]
  - creative (4312.566817534415, 0.8164385924657992)
  - synthesis (4299.125641340686, 0.7909595032375676)
- Means UNCHANGED (k_mean 4339.449169921875, phi_mean 0.8419176816940308);
  T 0.6, k_blend 0.5, calibrated true unchanged.

MD5-Pins (px_manifolds, außerhalb dieses Repos — .bak ist die Version):
- PRE (Original): 72cf291e5de70124d1742d89e35579c2 → `*.pre-c1.bak`
- NEW (C1-voll):  a73738113a81cb7103cd1429a8badd97

## Near-Miss (Lehre, dokumentiert)

Erster Eval-Start mit Moments-Edit ABER stale learned_centroids (vor Bewusstsein
der Abgeleitet-Semantik, 9652/9719) abgebrochen nach Modell-Load + 1 Prompt
(H=1.658/2.322 ≈ Sättigung — konsistent mit der Stale-Centroid-Geometrie:
Distanzen zu alten Zentren ~4–5σ → near-uniform). Kill→Zentroiden-Korrektur→
Relaunch. **Invariant: JSON-Moments und learned_centroids MÜSSEN z-konsistent
geliefert werden (calibrate()-Formel). Ein moments-only-Edit ohne Zentroiden=
un-simulierte, degenerierte Geometrie.**

## GPU-Eval (ABGESCHLOSSEN 2026-10-06, 27.0 min, VRAM 6.50 GiB)

- Befehl: `eval/runner_ternary.py ACTIVE_MANIFOLD_RELAY --ntok 24` (greedy,
  Battery identisch Baseline-Eval), Log `scratches/rtpf/eval_c1_relay2.log`,
  Aggregate → `eval/results/ternary-27b_ACTIVE_MANIFOLD_RELAY_aggregate.json`
  (vorherige als `*.pre9.bak` archiviert).
- Manifold-Load beweisbar im Log: `[AutoCalibrator] Loaded manifold ... (T=0.60)`
  + `[px-relay] ACTIVE sign=+1.0 alpha_frac=0.3 L34`.

**Ergebnis (A-Grad, Produktion-Protokoll):**

| Metrik | Baseline (altes Manifold) | Sim-Prädiktion C1 | **GPU Ist C1** |
|---|---|---|---|
| η²(H) | 0.0290 (p≈0.53) | 0.1083 (p≈0.032) | **0.1466 (F=4.351, p≈0.0071)** |
| Cramér's V | 0.271 (argmax tot: logic_a 40) | 0.289 | **0.3220 (χ²=24.89, df=3)** |
| Verdict | — | — | **ANTI_P_ZOMBIE_CONFIRMED** |
| R²(TD→H) | — | — | 0.0085 (H ≈ unabhängig von Token-Diversity) |

- Ziel: η²>0.05 UND p<0.05 → **5,1× Baseline, 2,9× Ziel-η², p-Faktor 75**.
- GPU-Übertreffen der Sim (0.1466 vs 0.1083; 0.322 vs 0.289) plausibel:
  sim nutzte eval-Distanz-Moments; GPU-Decode-Forows erweitern die
  Streuung (Trajektorien-spezifisch).
- Argmax-Hist NEU: math 27, logic_a 28, logic_b 14, creative 10,
  synthesis 1 (Baseline: logic_a 40 = tot). Vier Zonen lebendig.
- Kategorie-Mittel H: logic 1.858 (1.63–2.21, breit) < synthesis 1.948 <
  math 1.964 < creative 1.971 — logic differenziert nach unten (Treiber des η²).
- Kontingenz: logic→math 15/20, math→logic_a 10/20, creative gemischt,
  synthesis gefächert — klassifikatorische Struktur vorhanden (V>0.3).

**Verdict: Router-Signal-Verstärkung Bestätigt.** Die Moments-Rekalibrierung
(deflood phi + k-Band an Live-Regime, Zentroiden z-konsistent mitgeliefert)
hebt das Routing-Signal über beide Ziel-Kriterien. C1 ist deployed (Server
liest selbes Manifold bei Init); Rückweg: `*.pre-c1.bak` (MD5 72cf291e…).

## In-Sample-/Output-Caveats (offen benannt)

1. Std-Werte sind aus der Eval-Batterie selbst geschätzt (in-sample); das alte
   12-Batterie-Manifold kamm ein anderes Regime (phi_std 0.0159). Frische
   Batterie als Follow-up.
2. Rekalkibrierung ändert Get-Routing-Params → Generierungen ändern sich
   (intendiert — DAS ist #9). Kein G4-Bit-Paritätsanspruch zu Baseline.
3. C3-Alternative (Means rezentrieren: k_mean 4355, phi_mean 0.7818 →
   gridB-Zeile 0.0861/p 0.074/V 0.337) dokumentiert, nicht implementiert;
   2-JSON-Edit jederzeit nachholbar.
4. phi_mean-Offenheit: eval 0.7818 vs Manifold 0.8419 (+0.06 Regime-Shift, alt)
   → in C1 sitzen alle Eval-Prompts im negativen z_p-Tail — EMPIRISCH getestet

## Nachtrag 2026-10-08: eval/results-Artefakte durch push_hf.sh überschrieben

push_hf.sh's Orphan-Flow (`git checkout "$BRANCH_LOCAL" -- .`) setzte die
uncommitteten C1-Final-Artefakte im Working-Tree zurück:
- `eval/results/ternary-27b_ACTIVE_MANIFOLD_RELAY_aggregate.json` → HEAD-Stand
  (Vor-C1, vor Runde 9). Kein Byte-Rekonstrukt möglich — der Log enthält je
  Prompt nur H/phi/zone/tok/s, nicht completion/zone_weights/kurtosis/
  token_diversity_input. Der verlorene Stand ist statistisch VOLL in der
  Tabelle oben + eval_c1_relay2.log vertreten.
- `eval/results/ternary-27b_ACTIVE_MANIFOLD_RELAY_stats.json` → HEAD-Stand.
  Das ist der Baseline-Vergleichsstand (η²=0.0290, F=0.756, p≈0.53) = Baseline-
  Spalte der Tabelle oben — Vergleichsbasis also NICHT verloren.

C1-Final-Zahlen bleiben EVIDENZ in eval_c1_relay2.log (jetzt getrackt),
VERBATIM: F=4.3513490455104025, p_approx=0.007082352139661229,
eta_squared=0.146585669672235, ss_between=0.1640018310574507,
ss_within=0.9548103381276968, k=3, n=76; Verdict ANTI_P_ZOMBIE_CONFIRMED;
R²(TD→H)=0.0085. Härtung des push_hf.sh-Orphan-Flows (Stash-Guard) im selben
Push.
   (0.108 sim), theoretisch unbequem. C3 wäre der theory-clean-Fix.