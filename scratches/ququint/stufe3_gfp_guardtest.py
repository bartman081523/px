#!/usr/bin/env python
"""Stufe-3d: GF(p)-Mask-Guard-Validation — RB % RC != 0 erzwingen.

(m, n) = (192, 1280): nb=10; GF5 RB=100 (100 % 16 = 4 -> leere Mask-Slots
im letzten Chunk), GF3 RB=70 (70 % 16 = 6). Der Guard-Bind-Normalfall
(rr >= RB -> m2 falsch, s=0) wird fuer beide Kerne erzwungen und gegen
die verlustfreie Q·S-Referenz (=> W@x) geprueft. Referenz f32,
Toleranz 1e-4 (Akkumulationsordnungs-Rauschen, wie Stufe-3b-Kerneltest).

Laufzeit < 20 s, < 0,5 GiB VRAM — laeuft neben dem Server.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ququint_quant"))
import gf5_gemv as G5          # noqa: E402  (pack_gfp_from_ternary, gfp_matvec)

torch.manual_seed(7)
m, n = 192, 1280
nb = n // 128

# fp16-exakte Skalare (PTQ1_0-Kreise): Zustand {−d, 0, +d} je Block
rng0 = np.random.default_rng(0)
rng1 = np.random.default_rng(1)
dvals = np.array([0.0625, 0.375, 1.5, 3.25], dtype=np.float16)
d = dvals[rng0.integers(0, len(dvals), size=(m, nb))].astype(np.float32)
sign = rng1.integers(-1, 2, size=(m, nb, 128))          # {-1,0,+1}
# f32-Signs erzwingen — f32*int64 wuerde NumPy auf float64 promovieren
# (addmv-Dtype-Fehler: "Float, Double, Float"); Werte exakt {-d,0,+d}.
W = (d[..., None] * sign.astype(np.float32)).reshape(m, n)                 # f32 ternar
W_gpu = torch.from_numpy(W).to("cuda")
x = torch.randn(n, dtype=torch.float32, device="cuda")
y_ref = W_gpu @ x
print(f"Material ({m},{n}) ternar, nb={nb} | GF5 RB={nb * G5.RPB[5]} "
      f"(RB % 16 = {nb * G5.RPB[5] % 16}), GF3 RB={nb * G5.RPB[3]} "
      f"(RB % 16 = {nb * G5.RPB[3] % 16})")

Wb = W.reshape(m, nb, 128)
for p in (5, 3):
    Q, u, Sb = G5.pack_gfp_from_ternary(W, p)
    # Pack-Superset (Clip nie aktiv): Q·S == W bitweise
    S16 = np.maximum(np.abs(Wb).max(axis=2, keepdims=True), 1e-12) \
        / G5.HALFV[p]
    Sfull = np.repeat(S16, 128, axis=2).reshape(m, n)
    assert np.array_equal(Q * Sfull, W), f"GF{p}: Pack nicht verlustfrei"
    u_dev = torch.from_numpy(u.view(np.int32).copy()).to("cuda")
    s_dev = torch.from_numpy(Sb).to("cuda")
    y = G5.gfp_matvec(u_dev, s_dev, x, m, n, p)
    err = float((y.float() - y_ref).abs().max())
    print(f"  GF{p}: RB={u.shape[0]:<3} max|Δ| vs W@x = {err:.6g}")
    assert err <= 1e-4, f"GF{p}: Guard-Pfad ungenau (max|Δ|={err})"

print("OK: GF(p)-Mask-Guard bindet korrekt bei RB %% RC != 0 (p=5, p=3)")