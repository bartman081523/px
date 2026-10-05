"""tests/test_px_capture_flush.py — TT-B3 Capture-Cleanliness-Contract.

Prueft (CPU, kein Modell): (1) calculate_phi branchless == alte
Early-Return-Semantik fuer B=1 (G4-eager-aequivalent); (2) px_capture_guard
setzt/raeumt _PX_CAPTURE; (3) flush_px_capture replayt collect() +
Telemetrie mit den exakten Ring-Werten (chronologisch), kein Doppel-Flush,
Wraparound chronologisch, noop ohne Ring. Die Device-Seite (Ring-Write
unter Capture) wird im TT-B2a-PoC (GPU) bewiesen.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_px_capture_flush.py
"""
from __future__ import annotations
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402

from px_patches.ternary_bonsai_27b_px import patch as PX      # noqa: E402
from px_patches.ternary_bonsai_27b_px.px_modules import (     # noqa: E402
    StabilityMonitor)


class MockCal:
    def __init__(self):
        self.calls = []

    def collect(self, kurtosis, phi, token_diversity=None, token_len=1):
        self.calls.append((kurtosis, float(phi), token_diversity,
                           token_len))


def make_tm(ring_n=8, device=torch.device("cpu")):
    tm = types.SimpleNamespace()
    tm._px_calibrator = MockCal()
    tm._px_current_telemetry = []
    tm._task_kurtosis = 123.5
    tm._task_token_diversity = 0.42
    PX.ensure_capture_buffers(tm, n_tokens=ring_n, device=device)
    return tm


def _old_phi(h_new, h_old):
    """Vor-TT-B3-Semantik (Early-Return) — Referenz fuer Aequivalenz."""
    h_n = h_new.to(torch.float32)
    h_o = h_old.to(torch.float32)
    norm_n_raw = torch.norm(h_n, dim=-1, keepdim=True)
    norm_o_raw = torch.norm(h_o, dim=-1, keepdim=True)
    both_zero = (norm_n_raw < 1e-9) & (norm_o_raw < 1e-9)
    if both_zero.all():
        return torch.tensor(1.0, device=h_n.device, dtype=h_n.dtype)
    max_n = torch.max(torch.abs(h_n), dim=-1, keepdim=True)[0]
    max_o = torch.max(torch.abs(h_o), dim=-1, keepdim=True)[0]
    h_n_scaled = h_n / (max_n + 1e-35)
    h_o_scaled = h_o / (max_o + 1e-35)
    norm_n = torch.norm(h_n_scaled, dim=-1, keepdim=True)
    norm_o = torch.norm(h_o_scaled, dim=-1, keepdim=True)
    phi = (h_n_scaled * h_o_scaled).sum(dim=-1, keepdim=True) \
        / (norm_n * norm_o + 1e-9)
    return phi.mean()


class TestCalculatePhiEquivalence(unittest.TestCase):
    """Branchless-Fix darf B=1-Werte nicht aendern (G4-eager-aequivalent)."""

    def test_random_pairs_b1(self):
        gen = torch.Generator().manual_seed(42)
        for _ in range(120):
            a = torch.randn(1, 256, generator=gen) * 3.0
            b = torch.randn(1, 256, generator=gen) * 3.0
            if torch.rand(1, generator=gen).item() < 0.3:
                b = a  # korrelierte Paare
            old = float(_old_phi(a, b))
            new = float(StabilityMonitor.calculate_phi(a, b))
            self.assertAlmostEqual(old, new, places=6)

    def test_both_zero_is_one(self):
        z = torch.zeros(1, 256)
        self.assertEqual(float(StabilityMonitor.calculate_phi(z, z)), 1.0)

    def test_identical_vectors_is_one(self):
        h = torch.randn(1, 256)
        self.assertAlmostEqual(
            float(StabilityMonitor.calculate_phi(h, h)), 1.0, places=6)

    def test_one_side_zero_is_zero(self):
        h = torch.randn(1, 256)
        z = torch.zeros(1, 256)
        self.assertEqual(float(StabilityMonitor.calculate_phi(h, z)), 0.0)

    def test_extreme_values_stable(self):
        h = torch.full((1, 256), 1e12)
        h2 = torch.full((1, 256), -1e12)
        v = float(StabilityMonitor.calculate_phi(h, h2))
        self.assertTrue(-1.0 <= v <= 1.0)


class TestCaptureGuard(unittest.TestCase):

    def test_guard_sets_and_resets(self):
        self.assertFalse(PX._PX_CAPTURE["active"])
        with PX.px_capture_guard():
            self.assertTrue(PX._PX_CAPTURE["active"])
            self.assertFalse(PX._PX_CAPTURE["need_flush"])
        self.assertFalse(PX._PX_CAPTURE["active"])

    def test_guard_exception_resets_active(self):
        try:
            with PX.px_capture_guard():
                PX._PX_CAPTURE["need_flush"] = True
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertFalse(PX._PX_CAPTURE["active"])
        # need_flush bleibt True -> flush_px_capture holt nach
        PX._PX_CAPTURE["need_flush"] = False


class TestFlushPxCapture(unittest.TestCase):

    def test_flush_collects_and_resets(self):
        tm = make_tm()
        tm._px_phi_ring[0], tm._px_phi_ring[1], tm._px_phi_ring[2] = \
            0.12345, 0.2, 0.3
        tm._px_ring_idx[0] = 3
        PX._PX_CAPTURE["need_flush"] = True
        PX.flush_px_capture(tm)
        self.assertEqual(len(tm._px_calibrator.calls), 3)
        # Ring ist float32 -> f32-Präzision des ersten Werts
        self.assertEqual(tm._px_calibrator.calls[0][0], 123.5)
        self.assertAlmostEqual(tm._px_calibrator.calls[0][1], 0.12345,
                               places=5)
        self.assertEqual(tm._px_calibrator.calls[0][2], 0.42)
        self.assertEqual(tm._px_calibrator.calls[0][3], 1)
        got = [c[1] for c in tm._px_calibrator.calls]
        for want, have in zip([0.12345, 0.2, 0.3], got):
            self.assertAlmostEqual(want, have, places=5)
        self.assertEqual([d["t"] for d in tm._px_current_telemetry],
                         [0, 1, 2])
        self.assertAlmostEqual(tm._px_gen_phi_sum, 0.62345, places=5)
        self.assertEqual(tm._px_gen_phi_n, 3)
        self.assertFalse(PX._PX_CAPTURE["need_flush"])

    def test_no_double_flush(self):
        tm = make_tm()
        tm._px_phi_ring[0] = 0.5
        tm._px_ring_idx[0] = 1
        PX._PX_CAPTURE["need_flush"] = True
        PX.flush_px_capture(tm)
        self.assertEqual(len(tm._px_calibrator.calls), 1)
        PX.flush_px_capture(tm)                    # need_flush=False
        self.assertEqual(len(tm._px_calibrator.calls), 1)
        self.assertEqual(len(tm._px_current_telemetry), 1)

    def test_wraparound_chronological(self):
        tm = make_tm(ring_n=4)
        for i in range(6):                         # ring -> [4,5,2,3]
            tm._px_phi_ring[i % 4] = float(i)
        tm._px_ring_idx[0] = 6
        PX._PX_CAPTURE["need_flush"] = True
        PX.flush_px_capture(tm)
        # letzte 4 Eintraege chronologisch: 2,3,4,5
        self.assertEqual([c[1] for c in tm._px_calibrator.calls],
                         [2.0, 3.0, 4.0, 5.0])

    def test_noop_without_ring(self):
        tm = types.SimpleNamespace()
        tm._px_calibrator = MockCal()
        PX._PX_CAPTURE["need_flush"] = True
        PX.flush_px_capture(tm)                    # kein Ring, kein Crash
        self.assertEqual(len(tm._px_calibrator.calls), 0)
        self.assertFalse(PX._PX_CAPTURE["need_flush"])

    def test_lazy_flush_noop(self):
        tm = make_tm()
        PX.flush_px_capture(tm)                    # need_flush=False
        self.assertEqual(len(tm._px_calibrator.calls), 0)


class TestEnsureCaptureBuffers(unittest.TestCase):

    def test_creates_and_is_idempotent(self):
        tm = make_tm(ring_n=16)
        self.assertEqual(tm._px_phi_ring.shape, (16,))
        self.assertEqual(tm._px_phi_ring.dtype, torch.float32)
        self.assertEqual(tm._px_ring_idx.dtype, torch.long)
        r1, i1 = PX.ensure_capture_buffers(tm, n_tokens=32)
        self.assertIs(r1, tm._px_phi_ring)          # kein Replace
        self.assertEqual(r1.shape, (16,))
        self.assertIs(i1, tm._px_ring_idx)

    def test_no_device_probe_on_cpu_tm(self):
        # ohne device-Arg und ohne Parameter -> device=None waere Crash;
        # make_tm setzt device explizit, hier nur Guard-Doku
        tm = types.SimpleNamespace()
        PX.ensure_capture_buffers(tm, device=torch.device("cpu"))
        self.assertEqual(tm._px_phi_ring.device.type, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)