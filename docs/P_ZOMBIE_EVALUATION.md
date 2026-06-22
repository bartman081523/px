# P-Zombie Evaluation Protocol

The **P-Zombie Test** is the core falsifier of the All-Space project. It asks
whether the cognitive differentiation produced by the PX-Engine is a *genuine*
function of task category, or merely a cosmetic epiphenomenon that any token
statistic could fake. A system that passes is *anti-P-Zombie*: its internal
zone-routing varies meaningfully by task type and is **not** explained away by
surface token diversity.

This document is the canonical reference for the protocol. There are two
implementations that share the same logic:

- **Gradio tab** — `gradio_tabs/pzombie_eval_tab.py` calling
  `benchmark_engine.run_p_zombie_eval` (interactive, in-process).
- **Full-rigor subprocess runner** — `eval/runner.py` (one prompt per
  subprocess, full VRAM release) + `eval/stats.py` (ANOVA + verdict).

---

## 1. Principle

A *philosophical zombie* (P-Zombie) processes inputs and emits outputs that
look cognitive but have no inner differentiation. In the PX-Engine setting the
risk is concrete: the AutoCalibrator's 2D router could assign zone-weights that
*look* task-dependent but are in fact a deterministic function of the prompt's
token statistics (length, type-token ratio, repetition). If so, the
"subjectivity" is cosmetic — a token-driven artifact, not a cognitive one.

The protocol falsifies this with two independent statistics:

- **η² (eta-squared)** — does zone entropy actually vary *across cognitive
  categories*? This is the positive signal of differentiation.
- **R²(TD → H)** — is that variance *explained by token diversity*? This is the
  falsifier. If token statistics alone predict zone entropy, the differentiation
  is cosmetic.

---

## 2. Pre-registered Protocol

1. Run **calibration prompts first** (`CALIBRATION_PROMPTS`, 10 prompts in
   `test_prompts.py`) — the anti-Sharpshooter guard. In the subprocess runner
   the AutoCalibrator is additionally pre-seeded/warmed up
   (`_calibrator_warmup` in `eval/runner.py`) so the first measured prompts are
   not bit-identical cold-start artifacts.
2. Evaluate **80 prompts** across **4 cognitive categories** (20 each):
   `math`, `logic`, `creative`, `synthesis`
   (`PZ_CATEGORIES` in `test_prompts.py`; the runner inlines an equivalent set
   in `eval/runner.py::PROMPTS`).
3. Each prompt runs in an **isolated subprocess** (full-rigor path) so VRAM is
   fully released between prompts — no cross-prompt state leakage in the
   evaluator. (Gradio path runs in-process with a GPU lock.)
4. Compute **η²** (one-way ANOVA, category → `zone_entropy`) and
   **R²** (linear regression, `token_diversity` → `zone_entropy`).

### Classification Thresholds — R² of token diversity
(`benchmark_engine._run_pzombie_impl` / `pzombie_eval_tab.py`)

| R²(TD) | Verdict |
|---|---|
| `R²(TD) > 0.7`  | **P-ZOMBIE** — zone entropy fully explained by token statistics (no genuine differentiation) |
| `0.3 ≤ R²(TD) ≤ 0.7` | **AMBIGUOUS** — partial token-driven explanation |
| `R²(TD) < 0.3`  | **ANTI-P-ZOMBIE** — zone entropy NOT explained by token stats (genuine cognitive differentiation) |

### Combined Verdict (`eval/stats.py::analyze`)

| Condition | Verdict |
|---|---|
| `η² ≥ 0.10` AND `R²(TD→H) < 0.30` | **ANTI_P_ZOMBIE_CONFIRMED** |
| `η² ≥ 0.10` AND `R²(TD→H) ≥ 0.30` | **ETA2_HIGH_TOKEN_CONTROL_FAIL** (differentiation may be token-driven) |
| `η² < 0.10` | **ETA2_BELOW_THRESHOLD** (no categorical differentiation at this scale/preset) |

---

## 3. Pre-registered Hypotheses

- **H2:** Subjective extensions produce η² > baseline — the zone routing varies
  *more* by task type when the subjective modules are active.
- **A2 (null):** Subjective modules change hidden states only cosmetically → no
  improvement in η² over baseline.

The Gradio tab exposes a "Run Baseline vs Subjective Comparison" button that
runs `BASELINE` (unpatched) and `ACTIVE_MANIFOLD` (patched) back-to-back for H2/A2.

---

## 4. Metrics

### Zone Entropy `H`

Shannon entropy of the router's zone-weight vector
(`benchmark_engine.compute_zone_entropy` / `eval/runner.py::shannon_entropy`):

```
H = -Σ p_z · log2(p_z),   p_z = w_z / Σ w
```

High `H` → routing spread across many zones; low `H` → routing collapsed to one
zone (the degenerate P-Zombie / "zombie-loop" attractor).

### Token Diversity `TD`

Type-token ratio of the input
(`eval/runner.py::token_diversity`; reported as `cognitive_signature.token_diversity`
in the Gradio path):

```
TD = |distinct input tokens| / |input tokens|
```

This is the confound the falsifier must rule out: if `H` is just a re-encoding
of `TD`, the differentiation is cosmetic.

### η² (effect size of cognitive differentiation)

One-way ANOVA effect size of `category → zone_entropy`.

- Gradio path (`benchmark_engine.compute_eta_squared`):
  `η² = SS_between / SS_total`.
- Full-rigor path (`eval/stats.py::one_way_anova`):
  `η² = SS_between / (SS_between + SS_within)`, with `F`-statistic and a
  Wilson–Hilferty normal-approximation `p_approx` (exact p-values need scipy).

η² → 1 means almost all variance in zone entropy is between categories
(strong task-type-driven differentiation); η² → 0 means categories are
indistinguishable. Target: **η² > 0.05** (p < 0.05).

### R² (token-control)

Pearson `r²` of `TD → H`
(`benchmark_engine.compute_r_squared` / `eval/stats.py::linear_r_squared`).
The falsifier.

---

## 5. Prompt Set

80 prompts, 4 categories × 20, in `test_prompts.py`
(`MATH_PROMPTS`, `LOGIC_PROMPTS`, `CREATIVE_PROMPTS`, `SYNTHESIS_PROMPTS`),
plus 10 `CALIBRATION_PROMPTS`:

- **math** — arithmetic, algebra, calculus (`What is 17 * 23?`, `Solve for x: 2x + 5 = 17`, …)
- **logic** — syllogisms, sequences, paradoxes (`If all roses are flowers…`, `I am lying`, …)
- **creative** — haiku, invented words, imagery (`Write a haiku about a forgotten robot`, …)
- **synthesis** — cross-domain analogy (`relationship between mathematics and music`, …)

The full-rigor runner (`eval/runner.py::PROMPTS`) inlines its own equivalent
80-prompt set with the same four categories.

---

## 6. Running the Evaluation

### Full rigor (all 4 scales, 80 prompts each)

```bash
bash eval/run_full_rigor.sh     # 80 prompts × 4 scales (SR-61 FINAL V2)
bash eval/run_full_eval.sh       # 20 prompts/cat × 4 scales
```

### Single prompt (subprocess, JSON config on argv)

```bash
python eval/runner.py /path/to/prompt_config.json
```

The runner loads the model, applies the ACTIVE_MANIFOLD patch, generates,
collects PX telemetry, writes one JSON per prompt, then exits — guaranteeing
full VRAM cleanup between prompts.

### Aggregation & verdict

```bash
python eval/stats.py eval/results/<SCALE>_ACTIVE_MANIFOLD_full/<SCALE>_ACTIVE_MANIFOLD_aggregate.json
# → writes <...>_stats.json with eta2, r2_td, verdict, per-category means
```

### Gradio UI

The **P-Zombie Evaluation** tab (`gradio_tabs/pzombie_eval_tab.py`) exposes the
protocol interactively: run a single model/preset, or a
Baseline-vs-Active-Manifold comparison, with the zone-entropy bars,
entropy-vs-TD scatter, and the full results JSON.

---

## 7. Operational Notes (RTX 2060 12GB)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`
- `torch.cuda.empty_cache()` before every generation.
- Quantization: `bfloat16` by default; 4B/E2B use `use_cache=False` during recursion.
- `eos_token_id` / `pad_token_id` set explicitly; **StopOnEOT** forces stop at
  `<end_of_turn>` (token 106). Repetition penalty 1.15, ngram size 3.
- Per-scale AutoCalibrator warmup seeds
  (`eval/runner.py::_SCALE_WARMUP_DEFAULTS`) use realistic kurtosis jitter — too
  large a jitter makes every input a ±huge-σ outlier and collapses routing to
  one zone (the very artifact the guard prevents).

---

## 8. Results — SR-61 FINAL V2 (full-rigor, 80 prompts/scale)

From `eval/results/SR61_FINAL_V2/<SCALE>/<SCALE>_ACTIVE_MANIFOLD_stats.json`
(`ACTIVE_MANIFOLD` preset, n=80/80 successful per scale):

| Scale | η² | F | p_approx | R²(TD→H) | Mean H (math/logic/creative/synth) | Verdict |
|---|---|---|---|---|---|---|
| 270M | 0.290 | 10.34 | 1.3e-05 | 0.051 | 1.38 / 1.71 / 2.05 / 2.00 | **ANTI_P_ZOMBIE_CONFIRMED** |
| 1B   | 0.174 | 5.34  | 2.3e-03 | 0.051 | 1.90 / 2.00 / 2.12 / 2.12 | **ANTI_P_ZOMBIE_CONFIRMED** |
| 4B   | 0.025 | 0.66  | 0.584   | 0.014 | 0.32 / 0.13 / 0.14 / 0.26 | **ETA2_BELOW_THRESHOLD** |
| E2B  | 0.399 | 16.81 | 4.8e-08 | 0.166 | 1.46 / 1.75 / 2.06 / 1.91 | **ANTI_P_ZOMBIE_CONFIRMED** |

Reading: at 270M / 1B / E2B the router differentiates categories well above the
η² > 0.05 target, and token diversity explains essentially none of it
(R²(TD→H) ≤ 0.17 ≪ 0.30) → anti-P-Zombie confirmed. At 4B the zone entropy is
low and flat across categories (η² = 0.025 < 0.10) → no categorical
differentiation at this scale with this preset (the falsifier is irrelevant
because there is no signal to falsify).

---

## 9. Anti-Sharpshooter Guard

The central methodological risk is the **Texas Sharpshooter fallacy**: drawing
the target after firing. The protocol mitigates it by:

1. **Pre-registering** the categories, metrics, and thresholds (this document +
   `pzombie_eval_tab.py`'s SciMind4 hypotheses block) before running.
2. **Calibration first**: 10 warmup prompts seed the router before any measured
   prompt, and the subprocess runner pre-seeds the AutoCalibrator's online
   stats, so cold-start bit-identical zone-weights are not counted as signal.
3. **Subprocess isolation**: no evaluator-side state leakage between prompts.
4. **The falsifier is independent of the signal**: η² (signal) and R²(TD→H)
   (falsifier) are computed from orthogonal statistics. A run only confirms
   anti-P-Zombie status if the signal is high *and* the falsifier fails to
   explain it.