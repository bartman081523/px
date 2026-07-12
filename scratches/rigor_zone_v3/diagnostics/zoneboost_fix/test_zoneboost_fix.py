"""Smoketest: Zone-Boost-Fix in v3 — verhindert active_manifold Degeneration auf 270m.

RED-GREEN: Wenn active_manifold ohne Fix degeneriert (leere/loopy Outputs) und mit
Fix sinnvolle Outputs produziert → Fix funktioniert.
"""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

from rigor_hle_suite_v3 import HLERunner
from rigor_scales_v3 import ARM_CONFIGS
import json

# Mini-Setup: 8 Tasks, 1 pro Kategorie
hle = HLERunner()
tasks = hle.load(max_per_category=1)
print(f"Geladen: {len(tasks)} Tasks (max 8)")
for t in tasks:
    print(f"  - {t.task_id[:16]} | {t.category} | gt={t.ground_truth[:30]!r}")
