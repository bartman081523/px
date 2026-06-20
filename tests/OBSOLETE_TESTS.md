# Obsolete-Test-Identifikation (2026-06-20)

*Testlauf: `python -m pytest tests/ -q` → **151 passed, 4 failed, 6 skipped**
(dieselben 4 wie in der Memory vermerkt — vorgehend, unangetastet). Server war
down (Live-Tests nicht lauffähig). pytest 9.0.3, venv
`/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python`.*

## A. Deterministisch scheiternd → veraltete Erwartung / stale Artefakt (OBsoleszent)

### 1. `tests/test_deep_regression.py::test_complex_riddle_regression`
- **Fail:** `AssertionError: 'Entropy' not found in 'MATH'`.
- **Warum obsolete:** Erwartet, dass die Persona „DMT Psilocybin 🌀" die Zone
  auf einen Wert mit „Entropy" lenkt. Aber der AutoCalibrator routet nach
  **Prompt-Kurtosis / Focus-C**, nicht nach der Persona (die ein Surface-Label
  ist). Ein Math-Riddle → MATH-Zone, egal welche Persona. Die Erwartung
  stammt aus einer alten personen-gesteuerten Routing-Annahme, die nicht mehr
  zur Architektur passt.
- **Der Geschwistertest** `test_logic_hallucination_regression` **PASSIERT**
  (prüft nur `steps > 0` — Rekursion aktiv) und ist NICHT obsolete.
- **Empfehlung:** `test_complex_riddle_regression` entfernen oder die
  Erwartung auf Prompt-getriebenes Routing umstellen (Zone = MATH für
  Math-Riddle, Entropy-Modulation an H/gamma_boost prüfen statt am Zonen-Namen).

### 2. `tests/test_eos_token.py::test_post_fix_aggregates_under_70_percent_at_max`
- **Fail:** `1.0 not less than 0.7` — liest stale eval-Results
  `eval/results/1B_ACTIVE_MANIFOLD_v2_eos_fix/1B_ACTIVE_MANIFOLD_aggregate.json`
  (at_max = 100% in jenem alten Artefakt).
- **Warum obsolete:** Der EOT/EOS-Fix (Token 106-Injection) ist im Hauptcode
  und wird bereits geprüft durch
  `test_recursion_regression_suite.py::TestEosEndOfTurnInjection` (6 Tests,
  alle PASS). Dieser Test prüft nicht live, sondern ein **altes Results-File**;
  das File ist stale (aus der v2-eos-fix-Validierung, 17. Jun).
- **Empfehlung:** entfernen ODER neu generieren lassen (eval-Run über alle
  Skalen, dann at_max < 70% prüfen). Da der Fix anderweitig abgedeckt ist,
  ist Entfernen vertretbar.

## B. Environment-Fail (nicht obsolete — fehlendes Plugin / Server down)

### 3. `tests/test_all_models_presets.py::test_all_models_presets`
### 4. `tests/test_gemma4_e2b.py::test_gemma4_e2b`
- **Fail:** `async def functions are not natively supported` — braucht
  `pytest-asyncio` (oder `anyio`/`pytest-trio`). Sind Live-Server-Integration-
  tests (brauchen Server auf 7860 + GPU). **Nicht obsolete**, nur im aktuellen
  Env nicht lauffähig.
- **Empfehlung:** `pytest-asyncio` in das Test-Env aufnehmen ODER mit
  `asyncio.run`-Wrapper ohne Plugin lauffähig machen. Nicht entfernen.

## C. Nicht-pytest-Skripte (in tests/, heißen `test_*`, collecten nichts)

Diese vier sind **Runnable-Skripte** (`async def main()` / `def main()` +
`if __name__ == "__main__"`), keine pytest-Tests. Sie nutzen **Legacy-Preset-
Namen** (RIGOR, SUBJECTIVE), die via `_migrate_preset` → ACTIVE_MANIFOLD
laufen, aber alt-semantisch testen. Brauchen GPU/Server.

| Datei | Muster | Legacy-Ref |
|---|---|---|
| `tests/test_gemma4_e2b_presets.py` | `async def main()` | BASELINE/RIGOR/SUBJECTIVE |
| `tests/test_gemma4_vs_gemma3_presets.py` | `async def main()` | BASELINE/RIGOR/SUBJECTIVE |
| `tests/test_isoliert_subjektiv_parity.py` | `def run()` | `px_config_preset="SUBJECTIVE"` |
| `tests/test_gemma4_e2b_sr59.py` | `def main()` | `px_subjective=True` |

- **Status:** semi-obsolete (Legacy-Preset-Namen, manuelle GPU-Benchmarks,
  nicht in die pytest-Suite integriert). Funktional über Migration evtl. noch
  lauffähig, testen aber alte Benennung.
- **Empfehlung:** entweder nach `benchmarks/`/`scripts/` verschieben (sie sind
  keine Tests) und auf ACTIVE_MANIFOLD[_LEAN] umstellen, ODER als manuelle
  Skripte belassen + im Kopf klar als „manuell, GPU, Legacy-Presets" markieren.

## D. Non-Test-Skripte & stale Fixtures in tests/ (Aufräumen-vorschlag)

Nicht-collectierte Skripte (kein `test_`-Präfix, nicht in Suite):
`benchmark_gpu_utilization.py`, `capability_benchmark.py`,
`debug_coherence.py`, `hle_benchmark.py`, `official_debug.py`,
`repro_gradio_error.py`, `repro_math_drift.py`, `repro_template_error.py`,
`run_cognitive_baseline.py`, `run_ultra_diverse_benchmark.py`,
`verify_termination.py` — Debug-/Benchmark-/Repro-Skripte.

Stale Result-Fixtures: `tests/_*.json` (9 Dateien: _gemma3_vs_gemma4_subjective,
_gemma4_e2b_preset_results, _gemma4_subject_kurtosis, _gemma4_vs_gemma3_results,
_isoliert_subjektiv_parity, preset_test_results, quality_investigation_results,
subjective_diagnostic_results, hle_results_270m) sowie
`tiny_logic_test.json` (124 KB), `_recursion_regression_invariants.json`.
- **Empfehlung:** Skripte → `scripts/` oder `benchmarks/`; stale JSON-Fixtures
  prüfen, ob noch referenziert (die `_recursion_regression_invariants.json`
  wird von der Suite geschrieben/gelesen — behalten).

## Zusammenfassung

| Kategorie | Anzahl | Aktion |
|---|---|---|
| Obsolete (stale Erwartung/Artefakt) | 2 Test-Methoden | entfernen oder umstellen |
| Environment-Fail (Plugin/Server) | 2 Tests | Plugin aufnehmen, nicht entfernen |
| Non-pytest-Skripte (Legacy-Presets) | 4 Dateien | verschieben/umstellen oder markieren |
| Non-Test-Skripte + stale JSON | ~11 Skripte + JSON | aufräumen (optional) |

**Nicht ausgeführt** in diesem Schritt: keine Löschung. Nur Identifikation,
wie gewünscht. Löschung/Umstellung auf Wunsch — die deterministisch-scheiternden
#1/#2 sind die klarsten Kandidaten.

---

## Update (2026-06-20, auf Wunsch „beides / aufräumen") — #1 und #2 erledigt

- **#1 `test_deep_regression::test_complex_riddle_regression` REPURPOSED** —
  Erwartung umgestellt von `assertIn("Entropy", zone)` (Personen-Steuerungs-
  Annahme) auf `assertEqual(zone, "MATH")` (SR-61b Prompt-Kurtosis-Routing) +
  `assertGreater(entropy, 0.0)` (AZS-Kern H aktiv). Testet jetzt die tatsächliche
  Architektur. Geschwister `test_logic_hallucination_regression` unangetastet.
- **#2 `test_eos_token::test_post_fix_aggregates_under_70_percent_at_max`
  ENTFERNT** — redundant mit `TestEosEndOfTurnInjection` (6 Tests PASS), prüfte
  nur stale `v2_eos_fix`-Artefakt. Informations-Test `test_v1_aggregates_show_-
  termination_problem` (immer-pass) bleibt.
- **#3/#4 (async/Plugin):** offen (Environment, nicht obsolete).
- **C (4 Non-pytest-Skripte) + D (Non-Test-Skripte/JSON):** offen (Aufräumen
  optional, nicht dringend).

### Neu gefunden beim Voll-Suite-Re-Run (Server jetzt up) — #5

- **#5 `tests/test_live_space.py::test_live_history`** — API-Drift: nutzt
  `gradio_client` mit `api_name="/chat_wrapper"`, aber das Gradio-Endpoint
  heißt jetzt `chat_fn` (`gradio_tabs/chat_tab.py:119`, `def chat_fn(...)`),
  und die Signatur hat sich geändert (chat_fn nimmt `message, history, model_id,
  px_preset, temp, tp, mt, rp, gamma, session_id, manager` — ≠ die 9 Params des
  Tests). Failt nur, wenn der Server läuft (`/chat_wrapper` nicht gefunden).
  **Obsolete** (alter Gradio-API-Name + alte Signatur). Empfehlung: entfernen
  oder auf `/chat_fn` + aktuelle Signatur re-alignieren. Nicht in diesem
  Schritt gemacht (Live-Server-Integration, größerer Re-Align).