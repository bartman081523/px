"""tests/test_seed_determinism.py — Verify Seed Determinism via generators.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from schemas import ChatCompletionRequest
from generators import _apply_seed


def test_schema_accepts_seed():
    req = ChatCompletionRequest(
        model="gemma3-1b-it",
        messages=[{"role": "user", "content": "hi"}],
        seed=42,
    )
    assert req.seed == 42


def test_schema_default_seed_is_none():
    req = ChatCompletionRequest(
        model="gemma3-1b-it",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.seed is None


def test_apply_seed_deterministic_rng():
    _apply_seed(42)
    val1 = torch.randn(10)
    _apply_seed(42)
    val2 = torch.randn(10)
    assert torch.equal(val1, val2)

    _apply_seed(123)
    val3 = torch.randn(10)
    assert not torch.equal(val1, val3)


def test_apply_seed_zero_is_unseeded():
    # Calling _apply_seed(0) should NOT re-seed, meaning consecutive calls draw different numbers
    _apply_seed(42)
    _apply_seed(0)  # should not seed
    r1 = torch.randn(10)
    _apply_seed(0)  # should not seed
    r2 = torch.randn(10)
    assert not torch.equal(r1, r2)


def test_apply_seed_none_is_unseeded():
    _apply_seed(42)
    _apply_seed(None)
    r1 = torch.randn(10)
    _apply_seed(None)
    r2 = torch.randn(10)
    assert not torch.equal(r1, r2)
