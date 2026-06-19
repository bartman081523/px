"""sys.path-Bootstrap für scratches/consolidation (Pattern aus scratches/infinite_context/conftest.py).

Erlaubt Imports wie:
    from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, get_px_metrics
    from eval.runner import PROMPTS, shannon_entropy
ohne dass das Repo installiert werden muss. Keine Modul-Source wird angerührt.
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)