#!/usr/bin/env python
"""Debug: GF(p)-Kernel mit Block-Alignment + Skalaren gegen CPU-Referenz."""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, str(REPO / "ququint_quant"))
import gf5_gemv as G  # noqa: E402

rng = np.random.default_rng(5)
for p in (5, 3):
    m, blk = 4, 128
    n = 128 * 2                        # nb = 2 Blöcke
    W = rng.integers(-1, 2, (m, n)).astype(np.float32) * 3.0
    W[:, 128:] *= 7.0                  # unterschiedliche Block-Skalare
    Q, u, S = G.pack_gfp_from_ternary(W, p, block=blk)
    x = torch.from_numpy(rng.standard_normal(n).astype(np.float32)).cuda()
    y = G.gfp_matvec(torch.from_numpy(u.view(np.int32)).cuda(),
                     torch.from_numpy(S).cuda(), x, m, n, p)
    y_ref = torch.from_numpy(W).cuda() @ x
    d = float((y - y_ref).abs().max())
    print(f"p={p}: RB={u.shape} S={S.shape} max|Δ|={d:.6f}")