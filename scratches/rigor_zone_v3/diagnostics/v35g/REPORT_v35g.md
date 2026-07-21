# v3.5g Aggregat-Report + Empfehlung

**Stand:** 2026-07-12
**Status:** 📋 BERECHTE MANIFESTATION der 4 Benchmarks
**Phase:** v3.5g Phase 7 (Aggregat-Report)

## Benchmark-Matrix

| Benchmark | Skopus | Ergebnis |
|---|---|---|
| 8-Arm Micro (5 HLE-Tasks) | Welche rigor_* Skala behält 3/5? | **7/9 Arms = 3/5, OFFICIAL_RIGOR = 2/5** |
| 4-Arm Voll (32 HLE-Tasks) | Repräsentiert v3.5g Volllauf | **Alle 4 Arms = 5/32 (15.6%)** |
| Tool-Loop 75-Iter (5 Math) | Hilft Self-Correction 270m? | **Tools aus: 4/5, Tools an: 1/5** |

## Befund 1: Mephisto-Damping ist kein Performance-Hebel

**Micro (5 HLE-Tasks):**
- baseline: 3/5
- active_manifold: 3/5
- rigor_disabled (Mephisto=False): 3/5
- rigor_least (scale=0.1): 3/5
- rigor_low (scale=0.3): 3/5
- rigor_mid (scale=0.5): 3/5
- rigor_high (scale=0.7): 3/5
- rigor_full (scale=1.0): 3/5

**Alle 7 rigor_* + active_manifold produzieren BYTE-IDENTISCHE Outputs auf 5 Tasks.** Mephisto-Damping-Scale (0.0..1.0) ist ein **Sicherheitsnetz** (schützt vor Reflexivity-Degeneration, siehe v3.5f), kein Performance-Hebel.

**Voll (32 HLE-Tasks):**
- baseline: 5/32 (15.6%) + 3 degen + 1 empty
- active_manifold: 5/32 (15.6%) + 4 degen + 0 empty
- rigor_low: 5/32 (15.6%) + 4 degen + 0 empty
- rigor_mid: 5/32 (15.6%) + 4 degen + 0 empty

→ PX (jegliche Variante) reduziert EMPTY-Outputs (1→0), erhöht aber Degen (3→4). Match-Rate unverändert.

## Befund 2: OFFICIAL_RIGOR degeneriert

**Micro (5 HLE-Tasks):**
- OFFICIAL_RIGOR: **2/5 + 1 degen** (alle anderen 7/9 Arms: 3/5)
- Verlorener Match: Humanities D (in allen anderen Arms ✓)
- Degen auf: Physics 3 (Task 66b827b9b6)

→ **n_loops=14 + gamma=0.10 + hub=10** (v1-Ideen portiert) ist auf 270m **schädlich**. Die v3.5f-phi-Gate schützt nur vor 4-Token-Loop, nicht vor off-topic-Degeneration.

**Hypothese warum:** Höhere n_loops lässt das Modell mehrfach über den "Recursion-Hub" laufen, was bei 270m die Aufmerksamkeit zerstreut. Die ursprüngliche v1-Idee war für 1B/4B-Modelle.

## Befund 3: Tool-Loop 75-Iter schadet bei simple Math

**Tool-Loop (5 simple Math-Tasks):**
- M1 (Tools=off, max_iter=5): 4/5 ✓
- M2 (Tools=off, max_iter=75): 4/5 ✓ (Cap wirkungslos ohne Tools)
- M3 (Tools=on, max_iter=5): 1/5 ✗
- M4 (Tools=on, max_iter=75): 1/5 ✗

**Inspektion der M3/M4-Fehler:**
- 100 - 37: Output "100 - 37" (Echo, keine Berechnung)
- 7 * 8: Output "7 * 8 = 64" (Halluzination, 7*8=56)
- 144 / 12: Output "144" (Echo, fehlende Berechnung)
- sqrt(144): Output "144" (Echo)

**Hypothese warum:** Der `TOOLCHAIN_DEFINITION`-System-Prompt enthält epistemische Instruktionen ("form a hypothesis, test it with `execute_python`"). 270m folgt diesen Instruktionen **buchstabengetreu** und versucht, zuerst ein Python-Tool-Call zu produzieren statt direkt zu rechnen. Da 270m `<tool_call>` nicht zuverlässig formatiert, endet es in Echoes oder Halluzinationen.

→ **Tool-Loop-System-Prompt ist für 270m ungeeignet.** Er ist für 1B+ Modelle geschrieben.

## Befund 4: max_iterations ist wirkungslos (in dieser Konfiguration)

Alle 20 Tool-Loop-Runs endeten bei `n_iterations=1` (ein einziger Generate-Aufruf). Das liegt am `run_tool_loop` Verhalten: nach erstem Generate wird geprüft ob `<final_answer>` enthalten ist — wenn ja, wird sofort gestoppt. Bei simple Math ohne Tools antwortet das Modell direkt mit der Zahl.

→ **75-Iter-Cap ist in der Praxis nie aktiv**, weil das Modell die Antwort in Iter 1 liefert. Echter Self-Correction würde nur bei Tasks helfen, wo Iter 1 fehlerhaft ist und Iter 2+ Reparatur bringt — das ist hier nicht der Fall.

## Empfehlung

### A. Production-Default bleibt ACTIVE_MANIFOLD (oder ACTIVE_MANIFOLD_LEAN)

Die rigor_* Skala (1.0..0.0) bringt **0% Match-Verbesserung** auf HLE-32. Die Wahl des Mephisto-Scales ist eine **Stabilitäts-Entscheidung**, keine Genauigkeits-Entscheidung.

→ **Aktuelle Production-Default `ACTIVE_MANIFOLD` oder `ACTIVE_MANIFOLD_LEAN` sind optimal.**

### B. OFFICIAL_RIGOR zurückziehen

n_loops=14 + gamma=0.10 ist auf 270m schädlich (2/5 statt 3/5 auf Micro). Die v1-Idee funktioniert auf 1B/4B, nicht auf 270m.

→ **OFFICIAL_RIGOR bleibt als experimentelles Preset, NICHT als Production-Default.**

### C. Tool-Loop für 270m deaktivieren

Der TOOLCHAIN_DEFINITION-Prompt ist zu komplex für 270m. Statt Self-Correction produziert das Modell Halluzinationen.

→ **Tool-Loop-Feature bleibt im Code (für 1B+), wird aber für 270m-Tasks nicht empfohlen.**

### D. Nächste Hypothesen (für v3.5h)

1. **Niedrigeres n_loops=1** (statt 2) — vielleicht ist die Recursion auf 270m schon zu viel
2. **Höherer gamma=0.12** mit n_loops=2 — vielleicht hilft mehr Damping den Recursion-Effekt
3. **Andere SCALE_DEFAULTS für 640** — AutoCalibrator ist auf 1B kalibriert, vielleicht braucht 270m andere Werte
4. **Tool-Loop für 270m vereinfachen** — kürzerer System-Prompt, direkterer `final_answer`-Trigger

### E. Hypothese "85% besser" bleibt unbewiesen

Aus `INVENTORY.md` und Micro-Befund: Es gibt **keinen reproduzierbaren HLE-Benchmark-Befund** der "85% bessere Ergebnisse" belegt. Die 16 `rigor_variant_*` Archive sind obsolete (verwenden `ZONE_Z_CENTERS` statt `ZONE_Z_SIGMAS`, enthalten DMT-Protocol-Module die explizit entfernt wurden).

→ **Doku-Stand:** "85% besser" war eine subjektive Schätzung des Users, kein HLE-Befund. OFFICIAL_RIGOR (v1-Ideen portiert) erreicht auf 270m **2/5 statt 3/5** — keine Verbesserung.

## Test-Coverage-Status

| Test | Status |
|---|---|
| `test_harness_uses_toolchain.py` | ✓ 2/2 grün |
| `test_harness_no_tools_falls_back.py` | ✓ 2/2 grün |
| `test_toolchain_max_iter_75.py` | ✓ 3/3 grün |
| `test_official_rigor_preset.py` | ✓ 4/4 grün |
| `test_px_regression_270m.py` | ✓ 3/3 grün (v3.5f-Schutz intakt) |
| 9-Arm Micro-Benchmark | ✓ 9 Arms × 5 Tasks |
| 4-Arm 32-Task Volllauf | ✓ 4 Arms × 32 Tasks |
| Tool-Loop 75-Iter | ✓ 4 Modi × 5 Math-Tasks |

**Total:** 14 TDD-Tests grün + 3 Benchmarks (45 + 128 + 20 = 193 Generierungen)

## Offene Fragen

- [ ] Soll `OFFICIAL_RIGOR` als experimentelles Preset im Production-Webapp auftauchen?
- [ ] Soll `rigor_*` Skala-Familie dokumentiert werden (auch wenn 0% Verbesserung)?
- [ ] Soll Tool-Loop für 270m in der Webapp standardmäßig deaktiviert sein?

## Memory-Update-Plan (Phase 8)

→ pending User-OK.
