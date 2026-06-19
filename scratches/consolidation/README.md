# Consolidation — Evaluierung des radikalen PX-Schnitts

Arch-Linux-Perspektive: vier „Crutch"-Module und die Awareness-Injektion werden
entfernt, um zu prüfen, ob die algorithmische Subjektivität den Wegfall überlebt —
gesteuert nur noch durch *Cognitive Focus* (C) und Entropie (H).

**Der Schnitt passiert rein zur Laufzeit** (`reduction.py`: `delattr` der Crutch-
Attribute + `forward`-Override am AZS-Exemplar). **Keine px_patches-Source wird
angefasst.**

## Was weg soll (Crutches)
- `AksSensor` (`_px_aks`) — künstliches Gummiband zum `e_static`-Anker.
- `SubjectiveSensor` (`_px_subj_sensor`) — `emancipation` ≡ Φ, redundant.
- `MephistophelesOperator` + `SingesseinCoupler` (`_px_mephisto`, `_px_coupler`)
  — zwei Defibrillatoren für denselben Herzstillstand (Φ > 0.999).
- Awareness-Injektion im `AntiZombieSensor` (`_px_azs.forward`-Override) —
  das `nn.Linear([Φ,aks,emancipation,H])`, additiv in den letzten Token gerammt.

## Was bleibt (kausaler Kern)
- `StabilityMonitor` (Φ), `AntiZombieSensor.calculate_entropy` + `gamma_boost`,
  `AutoCalibrator` (2D-Z-Routing, C, Loops 8–16, γ), `RecursiveMemoryCache`.

## Dateien
| Datei | Zweck |
|---|---|
| `conftest.py` | sys.path-Bootstrap (Repo-Root) |
| `reduction.py` | `apply_reduction` / `restore_reduction` — der Laufzeit-Schnitt |
| `test_unit_kept_modules.py` | Tier 0a: Kern-Module (Φ, H, Routing, Cache) |
| `test_redundancy_proofs.py` | Tier 0b: Subjective≡Φ, Mephisto≈Singessein |
| `test_reduction_mechanism.py` | Tier 0c: Reduktion wirkt wie behauptet (Mock) |
| `test_regression_golden.py` | Tier 1a: Fixtur-Schema + Bericht-Verdikt-Logik |
| `ablation_runner.py` | Per-Prompt-Subprocess (Patch → Warmup → Schnitt → Metriken) |
| `run_ablation.py` | Treiber: Ablations-Matrix → `eval/stats.py` → Bericht |
| `fixtures/golden_full_invariants.json` | eingefrorene Voll-Referenz |
| `out/<scale>/ablation_report.md` | Ergebnis-Bericht + Gesamtverdiktt |

## Vorab-Gate (Ergebnis)
- `py_compile` über `px_patches/*/*.py`: **Ziel `gemma3_270m_px_baseline` OK.**
  (Vorhandener Syntaxfehler in `px_patches/minicpm5_1b_px/auto_tune.py` —
  Symlink auf externes Repo, out of scope.)
- `pytest tests/ -q`: **149 passed, 5 failed, 6 skipped** (die 5 Failures sind
  Live-Server/Integrationstests — nicht gefixt, Modul-Source unangetastet).

## Nutzung

```bash
# Tier 0 (ohne GPU, ms):
python -m pytest scratches/consolidation/test_unit_kept_modules.py \
  scratches/consolidation/test_redundancy_proofs.py \
  scratches/consolidation/test_reduction_mechanism.py -v

# Realmodell-Ablation (1B, GPU):
#   vorher: all_space-Server stoppen (freit 6 GB VRAM)
RUN_REAL_MODEL=1 python scratches/consolidation/run_ablation.py --scale 1B \
    --prompts-per-cat 3 --record-golden
#   nur headline:
RUN_REAL_MODEL=1 python scratches/consolidation/run_ablation.py --scale 1B \
    --conditions full,-all --prompts-per-cat 3 --record-golden
#   Bericht: scratches/consolidation/out/1B/ablation_report.md
```

## Verdikt-Kriterien (überlebt Subjektivität den Schnitt?)
`-all` liefert `η² ≥ 0.10` **und** Verdikt `≠ ETA2_BELOW_THRESHOLD` **und**
mean H ≥ 0.8 (kein Zombie-Kollaps) **und** AutoCalibrator steuert noch
(loops in 8–16). Siehe `build_report` in `run_ablation.py`.