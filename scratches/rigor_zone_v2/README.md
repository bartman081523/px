# RIGOR Zone v2 — Wissenschaftlicher Micro-Harness (HLE-Suite + Scale-Grid)

**Stand:** 2026-07-11 (laufend)
**Vorgänger:** `scratches/rigor_zone_v1/` (aa916bd, gepusht 2026-07-11)

---

## 0. SciMind 5.0 — Core Mandates (gemini-nightly + hermes-agent port)

> **Incomplete Suggestion Protocol:** Dieses README enthält Hypothesen — keine Wahrheiten.
> **Empirical Verification:** Was hier steht wird in `hle_results_v2.md` überprüft oder widerlegt.
> **Falsificationism (Via Negativa):** Wir suchen aktiv Daten, die diese Hypothesen widerlegen.
> **Pipeline Integrity:** Alle Shell-Wrapper benutzen `set -o pipefail`.
> **Zero-Trust Environment:** Pre-Lauf-Probes prüfen Dependencies.
> **Atomic & Secured:** Pro Run wird JSON VOR dem nächsten geschrieben (kein Verlust bei Crash).
> **Anti-Embedding:** Worker-Output (gemma3-270m) wird NIE als Tool-Call gerundet. Worker
> generiert, Decoder (anderes Modell) verifiziert — strikte 2-Stage-Trennung.

---

## 1. Forschungsfrage

**Was ist die Effektstärke (η²) von Mephisto-Scale-Varianten auf (a) HLE-Genauigkeit,
(b) Output-Stabilität, (c) Reproduzierbarkeit — und wie verhält sich das zu taskspezifischem
Skopus?**

Erweiterung von v1 (5 Tasks, 1 Mephisto-Scale) zu v2 (30+ Tasks, 5 Mephisto-Scales,
offizielles HLE-Dataset + 2-Stage-Decoder).

---

## 2. Hypothesen (H1–H4, pre-registriert)

| ID | Hypothese | Falsifikations-Kriterium | Operationalisierung |
|---|---|---|---|
| **H1** | Mephisto-Damping (scale<1.0) reduziert Output-Loops im Vergleich zu scale=1.0 | rigor_least (0.1) zeigt **nicht** signifikant weniger Loops als rigor_full (1.0) | Loop-Rate-Vergleich + Chi²-Test |
| **H2** | Skopus von RIGOR-Damping ist taskspezifisch: stark bei Math/CS, schwach bei Philosophy | per-task η² < 0.05 für alle Tasks ODER einheitlich über alle Tasks | η² pro HLE-Kategorie (math, cs, philosophy) |
| **H3** | Web-Suche (ddgs) hat messbaren Effekt auf Code-Tasks, NICHT auf Math-Tasks | such-effect η² < 0.05 für alle Kategorien ODER signifikanter Math-Effekt | η² für `with_search` pro Kategorie |
| **H4** | Reproduzierbarkeit: gleicher Seed → byte-identischer Output | >5% der Reproducibility-Runs zeigen Differenzen | Paarweise Hash-Vergleich (2-Seed-Runs pro Arm) |

**Antithesen (Falsifikations-Hilfen):**
- **A1:** Damping könnte Loops nur verlangsamen, nicht reduzieren (höhere Duration, gleiche Loop-Rate).
- **A2:** 270m-Outputs könnten so chaotisch sein, dass taskspezifischer Effekt im Rauschen verschwindet.
- **A3:** Web-Suche könnte nur Kontext-Länge erhöhen, ohne Qualität zu verbessern (Loops ↑, Acc →).
- **A4:** Greedy-Decoding sollte deterministisch sein — falls nicht, ist Seed-Propagation oder CUDA-Non-Determinismus schuld.

---

## 3. Methodik

### 3.1 Skopus (in/out of scope)

**in_scope:**
- gemma3-270m-it (Worker) via `ModelManager._load_model`
- 8 Arm-Konfigurationen: baseline, active_manifold, rigor_disabled, rigor_{least,low,mid,high,full}
- 30 HLE-Tasks (Mix: ~10 math, ~10 CS, ~10 philosophy) aus cais/hle oder statischem Fallback
- 2 Search-Settings: ohne / mit ddgs
- 1-2 Seeds für Reproduzierbarkeits-Test

**out_of_scope:**
- Andere Modelle (1B/4B als Worker) — würden v3 brauchen
- Fine-Tuning
- Andere PX-Presets
- Production-Code-Änderungen (User-Direktive: "vorerst keinen Production Code anfassen")

**known_unknowns:**
- minimaxm3:cloud nicht installiert → kein externer LLM-as-Judge (nur lokales gemma3-1b als Decoder)
- HLE-Dataset-Zugang unklar (HuggingFace cais/hle benötigt `datasets` Package + Netzzugang)
- 270m ist zu klein für echte Code-Synthese → Code-Tasks testen nur Pattern-Generierung

### 3.2 Architektur (4 Schichten)

```
scratches/rigor_zone_v2/
├── README.md                  ← dieses File (Hypothesen, Skopus, Methodik)
├── rigor_hle_suite.py         ← Schicht 1: HLE-Fragen-Loader (cais/hle + statischer Fallback)
├── rigor_scales_v2.py         ← Schicht 1: Mephisto-Scale-Grid (8 Arm-Configs)
├── rigor_verify.py            ← Schicht 2: Verifikation (deterministisch + 2-Stage-Decoder)
├── rigor_stats.py             ← Schicht 2: Statistik (η², Bootstrap-CI, Reproducibility)
├── rigor_harness_v2.py        ← Schicht 3: Haupt-Harness (Worker-Loop, 2-Stage)
├── rigor_report.py            ← Schicht 4: Wissenschaftlicher Report
├── out/                       ← Pro-Run JSONs (committed, LFS-track)
│   ├── arm_<name>_<task>_search<0|1>_seed<42>.json
│   └── reproducibility/...
├── hle_results_v2.md          ← generierter Report
└── hle_run_v2.log             ← Pre-Lauf + Run-Loop Log
```

### 3.3 Laufzeit-Schätzung

- 8 Arm × 30 Tasks × 2 Search = 480 Worker-Runs
- + 8 Arm × 5 Tasks × 2 Seed = 80 Reproducibility-Runs
- + Decoder-Review (optional, für offene Tasks): 480 × 50% = 240 Decoder-Calls
- Total: ~600-720 Runs
- Pro Run: ~20-30s (Worker) + ~5s (Decoder) = ~30s/Run
- **Geschätzt: 5-7 Stunden sequentiell auf RTX 2060**

### 3.4 Reproduzierbarkeits-Flags

- `seed=42` (primär), `seed=43` (Reproducibility-Test)
- `do_sample=False` (greedy, deterministisch)
- `temperature=1.0` (explizit von Production-Default abweichend, im Report dokumentiert)
- `max_new_tokens=500` (default; GPU besser ausgelastet)
- Pro Run JSON: `model_id`, `preset`, `arm`, `mephisto_scale`, `seed`, `timestamp`,
  `python_version`, `torch_version`, `transformers_version`, `git_commit`

### 3.5 Performance-Optimierungen (SciMind: CPU-Bottleneck vermeiden)

**Pre-Load aller 8 Arm-Modelle** vor der Run-Loop:
- Spart 7× CPU-Loading (jeder Model-Load dauert 5-10s)
- Speicher-Budget: 8 Modelle × 1.5 GB = 12 GB ≈ RTX 2060 Maximum
- Deaktivieren mit `--no-cache` (z.B. für OOM-Tests)

**Resumable Benchmarks** (default: AN):
- `--resume` (default): Skip Runs deren Output-Files bereits existieren
- `--no-resume`: Alle Runs neu ausführen
- Crash-Recovery: nach Ctrl-C einfach `python rigor_harness_v2.py` (skippt automatisch)

**Batched-Generation (default: bs=8, TDD-validiert):**
- `tokenizer(prompts, padding="longest")` — Rust-beschleunigt statt Python-Loop
- `tokenizer.batch_decode(...)` — ein Call statt 8× decode
- `compute_self_report_flags_batch` + `verify_by_category_batch` — vektorisiert
- **TDD-validiert:** `test_gpu_utilization.py` (4/4 grün)
  - GPU-Util avg 65.7%, max 74% (WARM-Call, 8 Prompts bs=8)
  - 1.04s/prompt (vs. 4.0s/prompt ohne Batching)
- Realistische Laufzeit-Schätzung: ~50-60 min für 520 Runs (RTX 2060)

---

## 4. Best-of-all-Worlds Patterns (User's lokale epistemische Patches)

v2-Micro-Harness portiert folgende Patterns aus 3 Repos:

| Pattern | Quelle | v2-Anwendung |
|---|---|---|
| SciMind 5.0 Mandates | `gemini-nightly/snippets.ts:213-229` | Header in README + hle_results_v2.md |
| Incomplete Suggestion | SciMind 5.0 | H1-H4 pre-registriert vor Code |
| Empirical Verification | SciMind 5.0 | Stdout/Stderr-Checks in `rigor_harness_v2.py` |
| Falsificationism | SciMind 5.0 | Antithese A1-A4 in §2 |
| Zero-Trust | SciMind 5.0 | `which`-Probes vor Lauf |
| Atomic & Secured | SciMind 5.0 | JSON-write vor nächstem Run |
| Anti-Embedding | SciMind 5.0 | Worker-Output ≠ Tool-Call-String |
| Worker+Decoder 2-Stage | `hermes-agent/plugins/worker-decoder/` | gemma3-270m (Worker) → gemma3-1b (Decoder) |
| PerfTestHarness baseline+delta | `gemini-nightly/test-utils/perf-test-harness.ts` | `assertWithinBaseline(scale=0.3, v1_result, tolerance=0.1)` |
| LLMJudge selfConsistency | `gemini-nightly/evals/llm-judge.ts` | 3× Voting für offene Tasks |
| Param-Validation-Tests | `gemini-nightly/validation-regression.test.ts` | Smoketest für jedes v2-Modul |

---

## 5. Pre-Lauf-Probes (SciMind Zero-Trust)

```bash
# Vor jedem Lauf
which ollama || exit 1                  # Ollama-Cloud-Subprocess
python -c "import ddgs" || exit 1        # Web-Suche
python -c "import datasets" || exit 1    # HLE-Dataset
python -c "import torch; print(torch.__version__)"
ollama list | grep -E "gemma3-(270m|1b)" || exit 1
git rev-parse HEAD >> out/git_manifest.txt
```

---

## 6. Verifikation

**Vor Lauf:**
```bash
# Module-Imports OK
cd scratches/rigor_zone_v2
PYTHONPATH=. python -c "from rigor_hle_suite import HLE_SUITE; from rigor_scales_v2 import ARM_CONFIGS; print(f'HLE: {len(HLE_SUITE)}, Arms: {len(ARM_CONFIGS)}')"

# Stats-Modul-Smoketest
PYTHONPATH=. python -c "from rigor_stats import compute_eta2, bootstrap_ci; print(compute_eta2([[1,2,3],[4,5,6]]), bootstrap_ci([1,2,3,4,5]))"

# Verify-Modul-Smoketest
PYTHONPATH=. python -c "from rigor_verify import detect_loops, verify_math; print(detect_loops('a a a a a'), verify_math('4', '4'))"
```

**Nach Lauf:**
```bash
ls out/*.json | wc -l           # erwartet: ~480-560
test -f hle_results_v2.md        # Report existiert
grep -c "^## " hle_results_v2.md # erwartet: ≥8 Sektionen
```

---

## 7. Pending (Stand 2026-07-11)

- Phase A: ✅ README erstellt
- Phase B: HLE-Suite + Scale-Grid Module (TODO)
- Phase C: Verify + Stats Module (TODO)
- Phase D: Haupt-Harness + Report-Generator (TODO)
- Phase E: Hintergrund-Lauf (5-7h, TODO)
- Phase F: User-Review + Commit (kein Push ohne OK)
- η²-Vergleich: rigor vs baseline (Ziel: ≥0.05) folgt nach Lauf
