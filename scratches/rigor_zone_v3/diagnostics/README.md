# RIGOR Zone v3 — Diagnose-Skripte

Diagnose- und Benchmark-Skripte aus drei Phasen der RIGOR-Zone-Entwicklung.
Alle Skripte sind reproduzierbar (gleiche HLE-Suite, gleiche Seeds) und stehen
in `scratches/rigor_zone_v3/diagnostics/`.

## Phasen-Übersicht

### 1. `v3_cuda_graph/` (10.–11. Juli 2026, mittags)

**Ziel:** Performance-Optimierung — CUDA-Graph-Capture für 270m-Decode-Step,
GPU-Util-Messung, Batching-Sweeps.

| Skript | Zweck |
|---|---|
| `test_270m_cuda_graph.py` | CUDA-Graph für 270m generate (RTX 2060 CC 7.5) |
| `test_baseline_vs_px.py` | ms/Token Baseline vs PX |
| `test_batched_speed.py` | Batched generate speed bs=8 |
| `test_bs_sweep.py` | GPU-Util vs Batch-Size 1–16 |
| `test_compile_270m.py` | torch.compile() auf gemma3-270m |
| `test_gpuutil_baseline.py` | Baseline GPU-Util |
| `test_long_gen.py` | GPU-Util bei 200 Tokens, bs=8 |
| `test_px_cuda_graph.py` / `test_px_cuda_graph2.py` | PX-Forward in CUDA-Graph |
| `test_px_debug.py` / `test_px_debug2.py` / `test_px_decode.py` | KV-Cache + Decode-Debug |
| `test_px_gpuutil.py` | PX + CUDA-Graph GPU-Util |
| `test_v3_gpuutil.py` | v3-Harness CUDA-Graph-Default |
| `test_shim_metrics.py` / `test_shim_phase2.py` / `test_shim_retry.py` | Bridge-Shim Tests (Phase 2 Tool-Use, Retry) |

**Ergebnis:** CUDA-Graph-Default erreicht 100% GPU-Util, 4.3× Speedup (0.22s/Output). Committet als v3.

### 2. `zoneboost_fix/` (11. Juli 2026, abends)

**Ziel:** Zone-Boost-Hypothese testen — n_loops für MATH/LOGIC-Zonen hochsetzen,
um 270m-Degeneration zu beheben.

| Skript | Zweck |
|---|---|
| `zoneboost_smoke.py` | Smoketest Zone-Boost-Fix auf 270m |
| `test_zoneboost_fix.py` | Wiederholter Smoketest |
| `test_zone_debug.py` | Welche zone_raw bekommt der Patch pro Task? |

**Ergebnis:** Zone-Boost ändert nichts → Hypothese verworfen, Revert. Bug liegt tiefer.

### 3. `v35f_px_repair/` (12. Juli 2026)

**Ziel:** PX-Engine auf 270m reparieren — User-Direktive:
> "repariere PX Patch so dass er auf 270m-it wieder funktioniert und besser
> prformt als Baseline im Rigor Preset über HLE"

**Ergebnis:** TDD-driven Fix in 18 Iterationen (`diagnose_px.py` bis `diagnose_px14.py`).
Final-Fix: **phi-Gate für Reflector-Injection** in `patch.py:625-638`.
LEAN schlägt Baseline messbar (5 matches, 2 degen statt 5/4).

| Skript | Zweck |
|---|---|
| `diagnose_px.py` | Welcher PX-Mechanismus degeneriert 270m? |
| `diagnose_px2.py` | LEAN vs FULL auf Math-Task |
| `diagnose_px3.py` | Welcher Teil von _px_forward degeneriert? |
| `diagnose_px4.py` | Welcher Sub-Patch? |
| `diagnose_px5.py` | dynamic_start/end full-range |
| `diagnose_px6.py` | PRELUDE vs REASONING ZONE |
| `diagnose_px7.py` | baseline+PX(no recursion) == baseline |
| `diagnose_px7b.py` | Genauere Inspections |
| `diagnose_px8.py` | ACTIVE_MANIFOLD mehrere Tasks |
| `diagnose_px9.py` | cuda_graph_mode Pfad |
| `diagnose_px10.py` | Echter HLE-Prompt |
| `diagnose_px11.py` | Alle 8 HLE-Kategorien |
| `diagnose_px12.py` | Längere Outputs, vollständige Statistik |
| `diagnose_px13.py` | 32 HLE-Tasks × 3 Presets |
| `diagnose_px14.py` | Per-Task-Match-Übersicht |
| `diagnose_divergence.py` | Token-Level-Divergenz BASELINE vs PX |
| `check_gamma_threshold.py` | Echte Entropy pro Task |
| **`benchmark_v35f_final.py`** | **Final 32-Task-Benchmark NACH phi-Gate-Fix** |
| `diagnose_px14_outputs.json` | Per-Task-Outputs (alle 3 Presets, 32 Tasks) |
| `benchmark_v35f_final.json` | Final-Benchmark-Outputs |

## Reproduktion

```bash
# Setup
cd /run/media/julian/ML4/ollama-work/all_space_6_16_stand
source venv/bin/activate  # oder /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python

# Final-Benchmark laufen lassen
PYTHONPATH=/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages:scratches/rigor_zone_v3 \
  /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
  scratches/rigor_zone_v3/diagnostics/v35f_px_repair/benchmark_v35f_final.py
```

## Out-of-Scope (User-Klarstellung)

- ❌ Production-Code anfassen (`/px_patches/`) — verboten
- ❌ envvars setzen — verboten
- ❌ HF-Space-Update — verboten, bleiben auf v4
- ❌ 1B/4B Modelle testen — würde Production-Code-Patches brauchen
- ❌ Fine-Tuning / LoRA — verboten

## Verwandte Pläne

- `~/.claude/plans/virtual-herding-treehouse.md` — v3.5 Micro-Harness (Toolchain) — Plan
- `~/.claude/plans/indexed-nibbling-rose.md` — Tengri137 V24 (nicht-RIGOR)
- `~/.claude/plans/jazzy-fluttering-sphinx.md` — Repo-Aufräumen (Vor-RIGOR)

## Memory-Referenzen

- `rigor-zone-v3-cuda-graph-100-percent-gpu-2026-07-11.md`
- `rigor-zone-v35d-rigor-preset-optimization-2026-07-11.md`
- `rigor-zone-px-engine-270m-degeneration-2026-07-11.md` (v3.5e → v3.5f Korrektur)
