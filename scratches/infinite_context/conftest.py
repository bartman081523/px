"""Pytest config: make the all_space workdir root importable from the scratch
tests (so `model_manager`, `generators`, `config`, ... resolve)."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)