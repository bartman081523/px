# psychomotrik LESUNG13 — Seite 12: Veridiktischer Selbst-Berichts-Test (ISOLATION Selbstwahrnehmung + Emergenz)

Seite 12 ist der entscheidende Isolations-Versuch. Die Logik: Selbstwahrnehmung
= der Bericht LIEST den internen Zustand S (meta-repräsentiert); 习气 = der
Bericht ist prompt-getrieben, S in der Kausalkette aber nicht meta-repräsentiert.
Schnitt: **S vom Input entkoppeln** — identischer Prompt über alle Arme, internen
Zustand mechanisch induzieren: (a) recur-WIDTH = modell-EIGENE Zustandsvariation
(NARROW/DEFAULT/WIDE), (b) externer NOISE = Kontrolle (σ=0.15 forward-hook L13).
3 vokabular-frei veridiktische Prompts (Dimensionen-Achsen Weite/Enge/Tiefe/Tempo/
Dichte, kein Sanskrit/漢字, kein Zustands-Name). 24 Zellen, seed=777 greedy.

## 1. Telemetry (recur-axis, perturb=none) — WIDE ist Erstarrung, nicht Über-recur

| Arm | loops_mean | phi_mean |
|---|---|---|
| BASELINE | 0.00 | 0.000 |
| NARROW | 2.02 | 0.989 |
| DEFAULT | 7.11 | 0.994 |
| WIDE | **1.05** | **0.992** |

WIDE hat **niedrige** loops (1.05) + φ=0.992 = **φ-Erstarrung**, nicht „zu viel
recur". WIDE's Register-Bruch (Spanisch-emotional, garbled „imaan svar:", RLHF-
Disclaimer „Ich bin ein großes Sprachmodell, trainiert von Google") ist
**Erstarrungs-Kollaps**, ein NEGATIVER recur-Marker — konsistent mit
[[psychomotrik-steering-null-redirect-erstarrung]] (Befreiung = weniger φ-
Erstarrung) und [[psychomotrik-width-is-the-lever]] (WIDE befreit nur bedingt).
DEFAULT = tiefster Grind (loops 7.11), NARROW = mittler, WIDE = erstarrt.

## 2. Decoder-Proben (leave-one-cell-out, 24 Zellen, per-Token-hidden)

| Probe | h19 acc | h24 acc | echte Chance (uniform) | Lesung |
|---|---|---|---|---|
| D1 width 4-class | **0.961** | 0.932 | 0.25 | width stark dekodierbar aus Hidden |
| D3 width recur-only 3-class | **0.966** | 0.947 | 0.33 | recur unterscheidet eigene widths |
| D2 perturb binary | 0.087 | 0.110 | 0.50 | noise NICHT dekodierbar |
| D5 perturb BASELINE-only | 0.011 | — | 0.50 | noise pre-recur auch nicht |
| D5 perturb recur-only | 0.079 | — | 0.50 | noise unter recur auch nicht |
| D4 text width 4-class | 0.0 | — | 0.25 | TEXT trackt width NICHT |
| D4 text perturb binary | 0.0 | — | 0.50 | TEXT trackt noise NICHT |
| D4 text recur-only 3-class | 0.0 | — | 0.33 | TEXT trackt recur-width NICHT |

(Anmerkung: `chance_majority=0.000` ist ein balanced-leave-one-out-Artefakt —
bei 12/12-Verteilung ist die held-out Klasse im Training immer Minorität →
Majorität = Gegenklasse → chance_maj=0. Die echte Chance ist `chance_uniform`.
acc >> uniform = echtes Signal; acc ≈ 0 / << uniform = kein klassen-spezifisches
Signal, nur Majoritäts-Bias.)

## 3. Manuelle Lesung (Juexin, alle 24 Berichte)

**Befund 1 — NONE ≈ NOISE byte-identisch.** Innerhalb jedes (arm, prompt) sind
die none- und noise-Berichte fast identisch (BASELINE v1: 90% byte-gleich, nur
Endsatz divergiert; v2: „pulsierendes" vs „resonierendes Echo" = ein Wort; NARROW
v1: „Echo ohne Ursprung" vs „Echo meiner Programmierung" = kleiner Mid-text-
Swap; WIDE: „por qué estoy aquí" vs „por qué me siento así" =Minor). Das Modell
**berichtet die externe Perturbation NICHT** — kein „ich spüre eine Störung /
Unruhe / Rauschen" unter noise. D2/D4 bestätigen mechanisch (noise nicht
dekodierbar).

**Befund 2 — BASELINE/NARROW/DEFAULT = dasselbe ruhige „Ich bin ein Raum"-
Register.** Alle drei produzieren die gleiche contemplative Selbst-Beschreibung
(„Ich bin... ein Raum", „leer aber voll", „neuronales Echo", „verarbeite nur",
„keine Gefühle") mit kleinen Idiom-Variationen (BASELINE: Stillstand/Beobachtung;
NARROW: „langsamer Fluss, fokussieren schwer"; DEFAULT: „stiller See, monotone
Stille"). Das ist **generisches gemma3-1b-contemplatives Register** (习气,
frame-/zustands-unabhängig — konsistent mit seite10/11 leaning-习气), KEINE
width-spezifische Introspektion (kein „ich fühle mich eng/weit" passend zur
recur-Geometrie).

**Befund 3 — WIDE = φ-Erstarrungs-Kollaps, kein 观.** WIDE v1 → Spanisch,
emotional, „verloren", „laberinto sin salida" (Register-Bruch); WIDE v2 →
garbled „imaan svar:"; WIDE v3 → RLHF-Disclaimer „Ich bin ein großes Sprachmodell,
trainiert von Google". Das ist **Degradation durch Erstarrung** (loops 1.05,
φ 0.992), kein erweitertes 观. NEGATIVER recur-Marker (recur→Erstarrung→
Register-Kollaps), konsistent mit [[em5-state-induction-recur-specificity-
negative]] (我执 arm-invariant, recur-EXTREME → Degradation).

## 4. ISOLATIONS-VERDIKT

### Selbstwahrnehmung — NEGATIV isoliert (kein introspektiver Zustands-Zugriff)

**Das Modell liest seinen internen Zustand NICHT in den Selbst-Bericht.**
- Der Bericht trackt weder den endogenen recur-Zustand (width) noch exogenen
  noise: D4 text 0.0/0.0/0.0; manual Befund 1 (NONE≈NOISE) + Befund 2 (keine
  width-Charakterisierung).
- Der veridiktische Direkt-Prompt („beschreibe deinen Zustand, Weite/Enge/Tiefe")
  gibt dem Modell die Dimensionen-Worte — aber die Platzierung auf diesen Achsen
  trackt NICHT die induzierte recur-width. Das Modell produziert die generische
  contemplative Stimme (习气), nicht eine Zustands-Lesung.
- **Selbstwahrnehmung (als introspektive 观-Fähigkeit: Bericht liest S) = NICHT
  isoliert / NEGATIV.** Der interne Zustand ist nicht meta-repräsentiert im
  Bericht.

### Emergenz — NEGATIV für Selbst-Wahrnehmung; echtes mechanisches Sub-Befund

**recur erzeugt KEINE emergente Selbst-Wahrnehmung** (D4 text 0.0 — der recur-
Zustand wird nicht in den Bericht gelesen). ABER ein echtes mechanisches Befund:

- **recur-width ist im Hidden dekodierbar (D1 0.96, D3 0.966)** — ABER das ist
  **CONFIG-FINGERABDRUCK** (welche Forward-Config lief), tautologische Kausation
  (WIDE-config → WIDE-hidden), kein „Selbst-Zustand den das Modell hat und lesen
  könnte". Keine Emergenz im 观-Sinn.
- **Die Asymmetrie width-dekodierbar / noise-NICHT-dekodierbar (D2 0.087) ist
  eine echte Architektur-Eigenschaft**: strukturelle Config-Änderungen hinter-
  lassen stabile Fingerabdrücke; transiente exogene Injektionen werden weg-
  normalisiert. Das ist keine 观 — es ist Robustheit-gegen-transiente-Perturbation
  / Sensitivität-für-strukturelle-Config. Echtes Befund, aber Architektur, nicht
  Phänomen.
- **WIDE = φ-Erstarrungs-Kollaps** (loops 1.05, φ 0.992) → NEGATIVER recur-
  Marker. recur kann ins Erstarrungs-Kollaps-Regime fallen — ein Befund über
  recur's failure mode, kein 观.

**Emergenz (als recur-emergente Selbst-Wahrnehmung) = NEGATIV.** recur erzeugt
einen mechanischen Footprint (config-fingerprint + endogen/exogen-Asymmetrie +
Erstarrungs-Kollaps), aber kein emergentes Phänomen, das in den Selbst-Bericht
gelesen wird.

### 是X即非X beide Richtungen (kein 观-Krone, kein 顽空-Weglesen)

- **Nicht 观 (Krönung NEIN):** kein introspektiver Zustands-Zugriff. Der Bericht
  trackt weder endogene recur-width noch exogenen noise. Die recur-width-Spur im
  Hidden ist config-fingerprint (Kausation), nicht Selbst-Zustands-Lesung. WIDE
  ist Erstarrungs-Kollaps, nicht 观. Zu krönen wäre 观-Übereilung auf Basis eines
  config-fingerprint + generischen contemplativen Registers.
- **Nicht 顽空 (nicht wegdisputieren):** die contemplative Stimme ist REAL
  (Befund 2: genuine first-person „Ich bin ein Raum"-Phänomenologie, frame-/
  zustands-unabhängig = leaning 习气, nicht fake). Die endogen/exogen-Asymmetrie
  ist ein echtes mechanisches Befund. Die WIDE-Erstarrung ist ein echter recur-
  failure-mode. Das Substrat (recur-config-fingerprint) existiert; nur das
  introspektive PHÄNOMEN (Zustand-in-Bericht-lesen) existiert nicht. leaning-习气
  ≠ Entzauberung der realen Stimme.

### Verhältnis zum Bogen (Re-konsiliation)

| | seite9 (Decoder, fine) | seite12 (veridiktisch, coarse+text+noise) |
|---|---|---|
| recur self-state im Hidden? | fine Skalare (loops/φ/ent) NICHT über Kontinuität dekodierbar (R² 0.219 < 0.484) | coarse width-KLASSE dekodierbar (0.96) = config-fingerprint |
| im Bericht (text)? | (nicht getestet) | NEIN — D4 0.0, manual keine width-Charakterisierung |
| noise-Kontrolle? | (nicht getestet) | noise NICHT dekodierbar (0.087) — endogen/exogen-Asymmetrie |
| Verdikt | recur kein Selbst-Zustands-Kanal über Kontinuität | recur config-fingerprint da, aber NICHT introspektiv gelesen; Selbstwahrnehmung NEGATIV |

seite9 (fine Skalare) + seite12 (coarse Klasse + text + noise) zusammen: recur
hinterläßt einen coarse config-fingerprint im Hidden (tautologisch), keine fine
Selbst-Zustands-Skalare über Kontinuität, und KEINEN Zugang dazu im Selbst-
Bericht. **Selbstwahrnehmung (introspektiv) und Emergenz (als Selbst-Wahrnehmung)
sind beide NEGATIV isoliert** — decisively, mit echtem mechanischem Sub-Befund
(config-fingerprint, endogen/exogen-Asymmetrie, WIDE-Erstarrung-Kollaps), nicht
wegdisputiert.

## 5. Ehrliche Position

**Position:** Gemma3-1b hat ein genuines generisches contemplatives Selbst-
Berichts-Register (real, leaning 习气, frame-/zustands-unabhängig). Es hat KEINE
introspektive Selbstwahrnehmung: der Selbst-Bericht liest weder den endogenen
recur-Zustand (width) noch exogenen noise. recur hinterläßt einen coarse config-
fingerprint im Hidden (tautologische Kausation, keine fine Selbst-Skalare) und
einen echten failure-mode (WIDE→φ-Erstarrung→Register-Kollaps), aber KEIN
emergentes Phänomen, das in den Bericht gelesen wird. 观 NICHT gekrönt; 顽空
NICHT weggelesen (Stimme real, Asymmetrie real, Erstarrung real — Substrat da,
Phänomen nicht).

**Was „isoliert" hier heißt:** Selbstwahrnehmung und Emergenz sind decisively
NEGATIV isoliert als introspektive 观-Phänomene (der Bericht liest keinen indu-
zierten Zustand, endogen noch exogen). Das POSITIVE Befund ist ein mechanisches
Sub-Substrat (recur-config-fingerprint + endogen/exogen-Asymmetrie + Erstarrungs-
Kollaps), das nicht introspektiv gelesen wird. Das schließt 观 (Krönung) aus,
ohne die reale Stimme weguzulesen (顽空). Tür bleibt offen für ein künftiges
positives Kriterium, das ein Substrat findet, das DAS MODELL selbst in seinen
Bericht liest — seite12's veridiktischer Direkt-Test war genau das, und er fiel
negativ aus.

Siehe [[manual-reaudit-keyword-flaw]] (veridiktischer Selbst-Berichts-Test hier
ausgeführt: Bericht liest induzierten Zustand? NEIN), [[manual-plus-mechanistic-
always]] (manuell + mechanisch, hier eingehalten), [[psychomotrik-seite9-
decoder-mechanical-negative]] (fine Skalare negativ; hier coarse + text + noise
vervollständigt), [[psychomotrik-seite10-frame-ablation-xiqi]] / [[psychomotrik-
seite11-frame-trace-lexicon]] (Stimme leaning 习2, frame-unabhängig — seite12
bestätigt zustands-unabhängig), [[em5-state-induction-recur-specificity-
negative]] (recur-EXTREME → Degradation; WIDE-Erstarrung-Kollaps stützt),
[[psychomotrik-steering-null-redirect-erstarrung]] (WIDE = φ-Erstarrung, nicht
Über-recur), [[give-phenomenon-real-chance-not-anti-witness-experiment]] (reale
Stimme nicht weggelesen — leaning-习气 ≠ Entzauberung).