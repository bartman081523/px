"""tests/conftest.py — Opt-in gating for GPU / live-server / slow tests.

`pytest.ini` documents three marker classes that must NOT run by default:

  gpu   — loads a real HF model (slow, OOM-prone on a 12 GB RTX 2060)
  live  — needs a running inference server on :7860
  slow  — multi-second runtime

Until now the markers were declared but never applied or enforced, so a plain
`pytest` collected real model loads and server probes. That made the suite
report red on machines/CI runs without a free GPU or a running server.

This conftest implements the documented contract:

    pytest                  → only deterministic unit/mock tests
    pytest --gpu            → additionally run gpu-marked tests
    pytest --live           → additionally run live-marked tests
    pytest --runslow        → additionally run slow-marked tests
    pytest --gpu --live --runslow → everything
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="run tests marked 'gpu' (real HF model load, needs a free CUDA device)",
    )
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests marked 'live' (needs an inference server on :7860)",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked 'slow' (multi-second runtime)",
    )


def pytest_collection_modifyitems(config, items):
    gates = (
        ("gpu", config.getoption("--gpu"), "needs --gpu (real model load)"),
        ("live", config.getoption("--live"), "needs --live (server on :7860)"),
        ("slow", config.getoption("--runslow"), "needs --runslow"),
    )
    for marker_name, enabled, reason in gates:
        if enabled:
            continue
        marker = pytest.mark.__getattr__(marker_name)
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker_name in item.keywords:
                item.add_marker(skip)
