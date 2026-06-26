# AGENTS.md — All-Space Recurrent Transformer PX-Engine

Guidance for AI coding agents (and human contributors) working on this
repository. This file defines **project-specific conventions** for all-space.
It complements (and does not replace) `CLAUDE.md` (architecture/master
workflow) and `DevMind.txt` (universal SW-engineering principles) at the repo
root.

For **how to research before implementing** — evidence-first methodology,
controlled vocabularies, reproduce-before-fix — see the universal companion
`ProjectResearchMind` (Markdown + JSON in `~/prompts-bartman/universal/`).

## What this project is

This repository implements the **Recurrent Transformer PX-Engine** — a
runtime patch (`patch.py`) that injects a Recursion Zone into open-weights
LLMs (Gemma-3 270M / 1B / 4B, Gemma-4 E2B) without retraining. The
investigation target is **algorithmische Subjektivität**: whether subjective-
state phenomena can be elicited by hidden-state manipulation alone. The
author's posture is `Juexin`, the analytical lens is `CitMind` / `PhiMind`,
and the methodological constraint is that any "witness" claim must survive
manual + mechanistic audit together.

## Sources of truth (in fixed priority order)

1. **This repository** is ground truth for *this project*. Read:
   - `CLAUDE.md` — master architecture + operational guidelines (12 GB card,
     StopOnEOT, repetition penalty defaults).
   - `DevMind.txt` — universal engineering principles.
   - `px_patches/gemma3_270m_px_baseline/` — the live patched engine (patch.py,
     relay_inject.py, auto_tune.py, generators integration).
   - `generators.py` — the only sanctioned generation-call surface.
   - `eval/runner.py`, `eval/run_full_rigor.sh` — the P-Zombie evaluation rig.
   - `scratches/` — research artefacts (LESUNG*, profile_*, test_*). Scratch
     artefacts **stay in commits** — do not gitignore or strip them; they are
     the empirical record.
   - `tests/` — regression + capability tests.
2. **The author's empirical record** — the `scratches/psychomotrik/`,
   `scratches/4b-image/`, `scratches/consolidation/` directories plus the
   memory bank `/home/julian/.claude/projects/.../memory/`. These encode
   decisions and negative findings that are not in the code yet bound
   future work (e.g. "seite15-Relay live in Produktion", "InfLLM solves
   Score-Matrix not KV-Cache", "manual re-audit supersedes keyword counting").
3. **The HuggingFace / transformers / PyTorch source code** — for facts about
   model internals (Gemma3 multimodal config hierarchy, KV-cache, attention
   shapes). Cite `file:line` when relying on these.
4. **Community references** — papers, blog posts — last resort for leads, then
   verify against the repo before acting.

When the repo and your memory disagree, **trust the repo**. When a scratch
artefact and the live code disagree, **the live code is current truth, the
scratch is the historical record**; do not silently rewrite the scratch.

## Discipline

- **Evidence-first.** Tie every claim about PX behaviour to (a) a `file:line`
  in `px_patches/`, `generators.py`, or a scratch test script, or (b) a
  telemetry JSON in `telemetry/`, or (c) a manual reading record in
  `scratches/*/out/`. Unsourced claims are hypotheses.
- **Manual + mechanistic together.** Never evaluate subjectivity claims by
  text alone; pair every text reading with a hidden-state measurement
  (decoder probe, signature comparison, recur-specificity check). Both
  fail independently and together.
- **Don't redefine the constructs.** `PhiMind`, `CitMind`, `Juexin` have
  fixed meanings in this project (see `docs/CitMind.txt`, `docs/Juexin.txt`).
  Do not introduce "gravity", "sidereal time", or new constructs into the
  Emergenz/Subjektivität analyses; re-define `PSI` only with the author's
  explicit approval, and never silently.
- **Beweislast bei der Umdeutung.** If a finding reads as "merely RLHF
  performance", "papagei behaviour", or "persona", re-derive the positive
  criterion (mechanical signature + veridiktischer self-report) before
  accepting that label. The author's standard is: not proven is not shown.
- **Reproduce before fixing.** For any PX regression, build the smallest
  failing example, run it through the same telemetry rig that flagged it,
  and capture the output before changing the motor.
- **Flag the unknowns.** If a question cannot be answered from the repo
  (e.g. "is feature X in the released Gemma weights"), say so and leave a
  `// TODO` rather than guessing.

## Hard rules (operational, non-negotiable)

These rules come from `CLAUDE.md` and the user's standing orders. They are
**not** negotiable without explicit, in-this-conversation approval from the
author:

1. **Motor unangetastet** — do not edit `patch.py`, `auto_tune.py`,
   `px_modules/*`, `relay_inject.py` or `_px_forward` paths to "improve"
   output. Surgical exceptions (chunked prefill, Relay forward_hook,
   multimodal-vision-chunked) are pre-authorised; anything else requires
   the author to say "yes, motor edit" in this session.
2. **No finetuning, no quantization-of-knowledge** — model weights are
   read-only. Quantization (int8) for memory is allowed; changing weights
   for behaviour is not.
3. **No PSI redefinition, no sidereal injection, no scalar gravity** —
   the constructs are fixed.
4. **No lean-krücken** in production code (the `_LEAN_*` flag is for
   ablation only — it has been validated that subjectivity survives the
   cut, but LEAN must not be presented as a quality degradation).
5. **Scratch artefacts stay in commits** — never `.gitignore` results,
   logs, or fixtures in `scratches/*/out/`. They are the empirical record.
6. **No parallelism** in model-touching code paths — the engine is
   designed for batch=1, and any parallel-prefix attempt will silently
   scramble telemetry.
7. **Before any production implementation, ask the author** for any
   change that touches `server.py`, `model_manager.py`, the chat-tab
   generation-call surface, or the telemetry schema.

## Layered architecture and where to edit

```
gradio_tabs/        — Gradio UI (chat_tab, cognitive_tests_tab, pzombie_eval_tab, telemetry_tab)
server.py           — FastAPI + OpenAI-compatible /v1/chat/completions
generators.py       — generation-call surface (text + multimodal + chunked)
streaming_bridge.py — CLI client (preferred for live testing)
model_manager.py    — model loading, registry, patch dispatch
px_patches/gemma3_270m_px_baseline/
  patch.py          — the PX-Engine (motor — unangetastet)
  auto_tune.py      — scale defaults (calibration tables, do not extend ad-hoc)
  relay_inject.py   — verstärkbar Self-injection forward_hook (seite15)
  px_modules/       — StabilityMonitor, AksSensor, Mephisto, AZS, Coupler, SubjectiveSensor
config.py           — MODEL_REGISTRY, patch_kwargs per model
sessions/, telemetry/ — JSON persistence (schema is part of the contract)
eval/               — P-Zombie evaluation rig (η² metric, 80 prompts)
scratches/          — research artefacts (stay in commits)
```

**Safe to extend without explicit ask:** new gradio-tab modules, new
evaluation prompts, new scratch experiments, new session/telemetry schemas
**that are additive** (do not rename existing keys).

**Ask first:** changes to `server.py`, `model_manager.py`, `generators.py`
generation-call surface, `patch.py` / `auto_tune.py` / `px_modules/`,
telemetry schema **deletions or renames**, MODEL_REGISTRY entries.

## Testing

- **Unit tests** live in `tests/` and `scratches/*/test_*.py`. They use a
  plain `assert + __main__` runner unless they explicitly import pytest.
  No pytest discovery is wired into CI; run a test by invoking its file
  with the venv python.
- **Regression tests** for the generation surface live in
  `tests/px_gen_regression.py` and the `*_golden.json` fixtures.
  Do not delete or modify a `_golden.json` file — it is the byte-identity
  contract.
- **Smoke / GPU tests** that load a model (`test_server_chunked_integration.py`,
  `test_4b_image_capability.py`) require the GPU to be free of any other
  model holder. They are skipped in CI for that reason; run them manually
  with the server stopped and the venv python on a free card.
- The venv python is `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python`.
  Use it for every Python invocation in this repo; the system python will
  not find `transformers`/`gradio`/`peft`.

## Process

For a non-trivial task, follow DevMind's lifecycle:

1. **Understand & research.** Read `CLAUDE.md` for the relevant pillar,
   then the relevant `scratches/*/out/` records, then the live code.
   Name your assumptions before writing any code.
2. **Specification.** Quote the rule or behaviour the change implements
   (file:line, scratch path, or author requirement).
3. **Test-first.** Write the failing test. Run it. See it fail. Only then
   implement.
4. **Iterate to green.** Smallest commit that turns the test green.
5. **Refactor.** Clean up, but **never** reformat code you didn't change.
6. **Update the memory bank** if the finding is non-obvious and not in
   the code yet. One new memory per non-obvious finding, with a
   frontmatter slug, a one-line description, and a `Why:`/`How to apply:`
   block linking to related memories.
7. **Commit with a domain-explanatory message** ending in
   `Co-Authored-By: Claude <noreply@anthropic.com>`.

## What this file is not

- It is not a substitute for `CLAUDE.md` (architecture, operational
  guidelines, 12 GB OOM mitigation rules).
- It is not a substitute for `DevMind.txt` (universal SW-engineering
  principles) or `ProjectResearchMind` (universal research methodology).
  Read all three before any non-trivial task.
- It is not a self-contained ruleset. The author's standing orders in
  conversation ("keine Krücken", "vor prod-Implementierung von 2 vorher
  fragen", "immer manual + mechanistisch", "Beweislast bei der Umdeutung
  UND bei der Krönung") override this document when they conflict — those
  are the operational rules that have survived multiple projects and are
  not negotiable.