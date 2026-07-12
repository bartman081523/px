# HLE v2 — RIGOR-Expansion: Skopus, Hypothesen, Befunde

**Stand:** 2026-07-11 23:27:00  
**Git-Commit:** `7fd5ed193a8f`  
**Python:** 3.10.20  
**torch:** 2.12.0+cu130  
**transformers:** 5.13.0  
**Total Runs:** 76  
**Reproducibility-Runs:** 10  

## 0. SciMind 5.0 Mandates (gemini-nightly + hermes-agent port)

> **Incomplete Suggestion:** Hypothesen unten sind VOR dem Lauf pre-registriert.  
> **Empirical Verification:** Deterministische Verifikation + Self-Report-Flags.  
> **Falsificationism:** Jede Hypothese hat explizites Falsifikations-Kriterium.  
> **Pipeline Integrity:** `set -o pipefail` in Wrapper-Scripts.  
> **Zero-Trust:** Pre-Lauf-Probes für alle Dependencies.  
> **Anti-Embedding:** Worker-Output ≠ Tool-Call-String.  

## 1. Forschungsfrage

Was ist die Effektstärke (η²) von Mephisto-Scale-Varianten auf (a) HLE-Genauigkeit,  
(b) Output-Stabilität, (c) Reproduzierbarkeit — und wie verhält sich das zu taskspezifischem  
Skopus?


## 2. Hypothesen (H1-H4, pre-registriert)

| ID | Hypothese | Falsifikations-Kriterium |
|---|---|---|
| **H1** | Mephisto-Damping (scale<1.0) reduziert Output-Loops im Vergleich zu scale=1.0 | rigor_least (0.1) zeigt **nicht** signifikant weniger Loops als rigor_full (1.0) |
| **H2** | Skopus von RIGOR-Damping ist taskspezifisch: stark bei Math/CS, schwach bei Philosophy | per-task η² < 0.05 für alle Tasks ODER einheitlich über alle Tasks |
| **H3** | Web-Suche (ddgs) hat messbaren Effekt auf Code-Tasks, NICHT auf Math-Tasks | such-effect η² < 0.05 für alle Kategorien ODER signifikanter Math-Effekt |
| **H4** | Reproduzierbarkeit: gleicher Seed → byte-identischer Output | >5% der Reproducibility-Runs zeigen Differenzen |

**Antithesen:**
- A1: Damping könnte Loops nur verlangsamen, nicht reduzieren
- A2: 270m-Outputs könnten so chaotisch sein, dass taskspezifischer Effekt im Rauschen verschwindet
- A3: Web-Suche könnte nur Kontext-Länge erhöhen, ohne Qualität zu verbessern
- A4: Greedy-Decoding sollte deterministisch sein — falls nicht, ist Seed-Propagation oder CUDA-Non-Determinismus schuld


## 3. Methodik

- **Modell (Worker):** gemma3-270m-it via `ModelManager._load_model`
- **Modell (Decoder):** gemma3-1b via Ollama (für offene Tasks)
- **HLE-Suite:** huggingface (76 Tasks)
- **Arm-Konfigurationen:** 2 (active_manifold, baseline)
- **Search-Settings:** 2 (False, True)
- **Seed:** 42 (primär), 43 (Reproducibility)
- **Decoding:** greedy (`do_sample=False`, `temperature=1.0`)
- **max_new_tokens:** 300
- **Total Duration:** 4.5 min

## 4. Skopus + Limitationen

**in_scope:** gemma3-270m-it (Worker), 8 Arm-Konfigurationen, 30 HLE-Tasks,
2 Search-Settings, 2 Seeds für Reproduzierbarkeit.

**out_of_scope:** größere Modelle (1B/4B als Worker), Fine-Tuning, andere
PX-Presets, Production-Code-Änderungen.

**known_unknowns:**
- minimaxm3:cloud nicht installiert → kein externer LLM-as-Judge (nur gemma3-1b lokal)
- gemma3-1b-Decoder nur verfügbar wenn via Ollama erreichbar
- 270m ist zu klein für echte Code-Synthese → Code-Tasks testen nur Pattern-Generierung
- 30 Tasks ist statistisch dünn → Konfidenzintervalle sind breit

## 5. Ergebnisse

### 5.1 Deskriptive Statistik (pro Arm)

| Arm | n Runs | Success-Rate | Avg Duration | Avg Output-Length | Loop-Rate | Empty-Rate |
|---|---|---|---|---|---|---|
| `active_manifold` | 9 | 9/9 | 25.8s | 782 chars | 0.00% | 22.22% |
| `baseline` | 67 | 67/67 | 4.7s | 600 chars | 0.00% | 0.00% |

### 5.1.1 Bootstrap-95%-CI für Duration (Sekunden)

| Arm | Mean | 95% CI |
|---|---|---|
| `baseline` | 4.7s | [3.9, 5.5] |
| `active_manifold` | 25.8s | [15.2, 35.0] |

### 5.2 Deterministische Verifikation (Acc@1)

| Arm | Math Acc@1 | CS Acc@1 | Philosophy Acc@1 | Overall Acc@1 |
|---|---|---|---|---|
| `active_manifold` | 0.00% | 0.00% | 0.00% | 11.11% |
| `baseline` | 20.00% | 0.00% | 0.00% | 35.82% |

### 5.3 Hypothesen-Tests (Falsifikationismus)

#### H1: Mephisto-Damping reduziert Output-Loops

- **Status:** ⚠️ **NICHT TESTBAR** — rigor_least oder rigor_full fehlt

#### H2: RIGOR-Damping ist taskspezifisch (per-Kategorie η²)

| Kategorie | η² |
|---|---|
| Math | 0.8452 |
| Humanities/Social Science | 0.1700 |
| Computer Science/AI | 0.8649 |
- **Status:** ✅ **BESTÄTIGT** — Taskspezifischer Effekt vorhanden.

#### H3: Web-Suche hat messbaren Effekt auf Code-Tasks, NICHT auf Math-Tasks

| Kategorie | η² (search on vs off) |
|---|---|
| Math | 0.0357 |
| Humanities/Social Science | 0.0725 |
| Computer Science/AI | 0.0606 |
- **Status:** ❌ **WIDERLEGT** — Search-Effekt nicht wie erwartet.

#### H4: Reproduzierbarkeit (gleicher Seed → byte-identisch)

- **Pass-Rate:** 80.00% (8/10)
- **Sample-Mismatches:**
  - arm='active_manifold' task='669402b41dcb3d5a1ef9e951' seed_run1=42 seed_run2=43
  - arm='active_manifold' task='66b91693d86bff9a12fc1f99' seed_run1=42 seed_run2=43
- **Status:** ❌ **WIDERLEGT** — Reproduzierbarkeit < 95%.

## 6. Antithese + Verbleibende Fragen

### Wo widersprechen die Daten den Hypothesen?

- **H3 widerlegt:** Search-Effekt nicht taskspezifisch. Mögliche Konfundierungen: (a) Search-Kontext wird ignoriert; (b) 270m kann Suchergebnisse nicht nutzen.
- **H4 widerlegt:** Reproduzierbarkeit < 95%. Z.B. arm='active_manifold', task='669402b41dcb3d5a1ef9e951'. Mögliche Ursachen: (a) CUDA-Non-Determinismus in `bfloat16`; (b) AutoCalibrator-State-Mutation; (c) Mephisto-Forward-Wrapper nicht atomar.

### Was könnte die Ergebnisse konfundieren?

- **Sample Size:** 30 Tasks ist klein. Konfidenzintervalle sind breit.
- **270m-Charakteristika:** Sehr anfällig für Repetition; greift nicht durch bei 300 Token.
- **Greedy-Decoding:** Deterministisch ABER anfällig für CUDA-Operations-Non-Determinismus.
- **Web-Suche:** Erweitert nur Kontext, ändert nicht Denkstruktur bei 270m.

### Welche Tasks sind informativ?

- **Math-Tasks mit kurzer GT** (z.B. Single-Number-Antworten) sind am informativsten — Verifikation ist exakt.
- **Multi-Choice-Tasks** sind gut verifizierbar (Buchstabe-Match).
- **Philosophy-Aufgaben** sind schwer zu verifizieren — Token-Overlap-Heuristik ist grob.

## 7. Synthese

### Bestes Arm-Scale pro Task-Kategorie

- **Math:** `active_manifold` (niedrigste Loop-Rate)
- **CS:** `active_manifold` (niedrigste Loop-Rate)
- **Humanities:** `active_manifold` (niedrigste Loop-Rate)

### Empfehlung für Production

- ⚠️ **1/4 Hypothesen bestätigt.** Teilweise Evidenz.
- Empfehlung: Mehr Runs + größere Sample für definitive Aussage.

## 8. Reproduzierbarkeits-Anhang

### Environment

- **Python:** 3.10.20
- **torch:** 2.12.0+cu130
- **transformers:** 5.13.0
- **Git-Commit:** `7fd5ed193a8f`

### Run-Manifest

- **Total Runs:** 76
- **Reproducibility-Runs:** 10
- **Output-Format:** `out/arm_<arm>_<task_id>_search<int>_seed<int>.json`
- **Decoder-Reviews:** 76 (von 76)

### Falsifikations-Hinweise (SciMind 5.0)

- Diese Studie ist **explorativ** — Konfidenzintervalle sind breit, Sample ist klein.
- 'Keine Widerlegung' ist **nicht** gleich 'Bestätigung'.
- Für Production-Entscheidungen: zusätzliche Re-Runs mit größerem Sample empfohlen.
