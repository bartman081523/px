#!/usr/bin/env python
"""Pre-Test fuer repack_gf3._pack_chunk: Digit-Layout <-> Kernel-Spaltenabbildung.

CPU-only. Baut ein synthetisches Ternary-W, packt ueber _pack_chunk und
dekodiert mit der Kernel-Abbildung (u[rr*M+r], col = b*128 + (rr%RPB)*ND + k)
zurueck. Muss max|Delta| = 0 liefern (Layout-Beweis VOR dem 12-min-Converter).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ququint_quant"))
import repack_gf3 as RG  # noqa: E402

rng = np.random.default_rng(11)
c, nb = 7, 5                                       # 7 Zeilen, 5 Bloecke
w = rng.integers(-1, 2, size=(c, nb, 128)).astype(np.float32)
w[0, 0, 0] = 0.0                                    # Nullen mixen
w[3, 4, 77] = 0.0
q = torch.from_numpy(np.sign(w))
s = torch.from_numpy(np.abs(w).max(axis=2))         # (c, nb)
u, s16 = RG._pack_chunk(torch.from_numpy(w), q, s)  # u (RB, c) int32 | s16 (nb, c)

print(f"u: {tuple(u.shape)} {u.dtype} | s16: {tuple(s16.shape)} {s16.dtype}")

uu = u.numpy().astype(np.uint32).astype(np.uint64)
s_np = s16.numpy().astype(np.float32).T             # (c, nb)
w2 = np.zeros((c, nb * 128), dtype=np.float32)
for row in range(c):                                # Kernel-Spaltenabbildung
    for rr in range(nb * RG.RPB):
        rec = int(uu[rr, row])
        b = rr // RG.RPB
        rin = rr % RG.RPB
        sb = s_np[row, b]
        for k in range(RG.ND):
            d = rec % 3
            rec //= 3
            j = rin * RG.ND + k
            if j < 128:
                w2[row, b * 128 + j] = (d - 1) * sb

w_ref = w.reshape(c, nb * 128)
diff = np.abs(w2 - w_ref).max()
print(f"max|Delta| = {diff:.6g}")
assert diff == 0.0, "Digit-Layout <-> Kernel-Abbildung NICHT verlustfrei"
print("OK: Layout bewiesen (LSB-Rang zuerst, Record rr = b*RPB + rin,"
      " Spalte b*128 + rin*ND + k, Pad digit=1 -> 0-Beitrag)")