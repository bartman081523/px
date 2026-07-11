# RIGOR Zone v1 — Isolierter HLE-Benchmark

**Stand:** 2026-07-11
**Status:** Experiment — KEIN Production-Code berührt

## Konzept

RIGOR als **6. manifold-aware Zone** in der PX-Engine, isoliert von der Production getestet.
Die Idee stammt aus den 16 `rigor_variant_*` Patches (Juni 2026, Pre-Manifold-Architektur),
die in einem damaligen HLE-Benchmark "80% mehr Genauigkeit" auf reasoning-starken Agentic-Tasks
brachten. Diese Patches sind heute obsolet (inkompatibel mit `px_manifolds/*_manifold.json`),
und `RIGOR` als Preset ist seit der Konsolidierung nur noch ein Alias für `ACTIVE_MANIFOLD`
(siehe `model_manager.py:_migrate_preset`).

**Ziel:** Empirisch testen, ob eine **additive 6. Zone** im aktuellen AutoCalibrator die
HLE-Genauigkeit verbessert — **ohne** die bestehende 5-Zonen-Architektur zu regressieren.

## Reproduktion

```bash
cd /run/media/julian/ML4/ollama-work/all_space_6_16_stand
python scratches/rigor_zone_v1/rigor_zone_harness.py
```

Erwartete Laufzeit: ~30-60 min auf RTX 2060 (4 PX-Arms × 5 HLE-Tasks × 2 Search-Settings
= 40 lokale Runs + 10 cloud-Runs via Ollama).

## 5 Benchmark-Arme

| # | Arm-Name          | Preset            | Patch-Kwargs                                  | Erwartung                    |
|---|-------------------|-------------------|-----------------------------------------------|------------------------------|
| 1 | `baseline`        | `BASELINE`        | `{}`                                          | untere Schranke               |
| 2 | `active_manifold` | `ACTIVE_MANIFOLD` | `{}`                                          | aktueller Production-Default  |
| 3 | `rigor`           | `RIGOR`           | `{"rigor_mephisto": False}`                   | RIGOR ohne Mephisto (LEAN)    |
| 4 | `rigor_damped`    | `RIGOR`           | `{"rigor_mephisto": True, "rigor_mephisto_scale": 0.3}` | RIGOR mit 30%-Mephisto-Damping |
| 5 | `minimaxm3_cloud` | (kein Patch)      | Ollama-direct                                 | externe Baseline             |

**`RIGOR`-Preset wird via `_migrate_preset` aktuell auf `ACTIVE_MANIFOLD` gemappt.**
Unsere `rigor_zone_manifold.py` und `rigor_zone_forward.py` machen ihre Overrides
**darüber hinaus** (additive Monkey-Patches, KEINE Code-Änderung an Production).

## Progressive RIGOR-Ideen (übernommen)

Aus den 16 `rigor_variant_*` extrahiert, **rein additiv**:

1. **Math-Hub-Lock auf L12** — `ZONE_ROUTING['rigor']['hub']=10` (math-Strenge)
2. **Höhere gamma** — `RIGOR_GAMMA_OVERRIDE=0.10` (war 0.08)
3. **Mehr Loops** — `RIGOR_N_LOOPS_OVERRIDE=14` (war 8)
4. **Mephisto-Damping** — `RIGOR_MEPHISTO_DEFAULT_SCALE=0.3` (Mephisto als Denkanstoß aus anderer Perspektive)
5. **6. Zone-Centroid** — `ZONE_Z_TARGETS['rigor']=(0.8, 0.3)` zwischen math und logic_a
6. **Breites Sigma** — `ZONE_Z_SIGMAS['rigor']=1.0` damit RIGOR als Cluster-Schwerpunkt wirkt

## Bewusst NICHT portiert (Regression-Risiko)

- **DMT-Suite** (CentralMemory, ERPU, AgencyVector, TretaDamper, GroundingAnchor) — explizit in SR-59i entfernt
- **Persona-Engine** (nur in `rigor_variant_0368a749`) — würde AutoCalibrator zerschlagen
- **Hardcoded kurtosis-Thresholds** (`kurtosis < 235.0`) — scale-abhängig
- **`classify_zone` Rewrite** — würde 5-Zonen-Architektur brechen
- **fee777f8 als Template** — 1139 Zeilen mit 6 RIGOR-Effekten, zu viel Code für den aktuellen Stand

## Architektur-Prinzip

```
rigor_zone_*.py  →  monkey-patcht  →  px_patches/gemma3_270m_px_baseline/{auto_tune,patch}.py
                  (zur Laufzeit)     (Production, unverändert)
```

**Keine** Datei wird in Production modifiziert. Alle Overrides passieren über:
- `ZONE_ROUTING`/`ZONE_Z_TARGETS`/`ZONE_Z_SIGMAS` Dict-Mutation (rigor_zone_manifold.py)
- `_compute_scf_weights`/`load_manifold` Method-Patching (rigor_zone_manifold.py)
- Mephisto `forward` Wrapper (rigor_zone_forward.py)

## Output

Nach Ausführung:
- `out/arm_<arm>_<task>_search<0|1>.json` — 50 Per-Run-JSONs (LFS-track)
- `hle_results.md` — Markdown-Report mit Vergleichstabelle, η², Highlight pro Task

## Memory-Verweise

- `scratch-artifacts-stay-in-commits.md` — `scratches/*/out/` wird committed, kein gitignore
- `webapp-bridge-param-parity-autotune.md` — Webapp↔Bridge Preset-Parity (RIGOR-Alias Kontext)
- `openmythos-venv-default.md` — venv_openmythos/bin/python (torch 2.12+cu130)
