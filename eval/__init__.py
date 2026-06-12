"""
eval/ — 4B evaluation runner on the cleaned ACTIVE_MANIFOLD architecture
==========================================================================

Self-contained within all_space/. Imports from dmt_space_50/ are FORBIDDEN
— see plan/cozy-sleeping-shore.md "Architektur-Commitment".

Modules:
  runner.py         — Subprocess: load 4B + apply_px_patch + collect telemetry
  run_4b_eval.py    — Main: drive 80 prompts through runner, collect JSONs
  stats.py          — η²-ANOVA + R² token-control on the collected JSONs
"""
