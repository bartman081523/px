#!/usr/bin/env python
"""Debug: GF(p)-Pack + Kernel-Digit-Decode isoliert auf winziger Matrix."""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
rng = np.random.default_rng(3)

def pack_row(Q, p, nd, half):
    m, n = Q.shape
    rb = (n + nd - 1) // nd
    d = Q.astype(np.int32) + half
    d = np.concatenate([d, np.full((m, rb * nd - n), half, np.int32)], axis=1)
    pk = p ** np.arange(nd, dtype=np.int64)
    u = (d.reshape(m, rb, nd).astype(np.int64) * pk[None, None, :]).sum(axis=2)
    assert u.max() <= p**nd - 1, (p, u.max())
    return u

def unpack_row(u, m, n, p, nd, half):
    rb = (n + nd - 1) // nd
    uu = u.reshape(m, rb).astype(np.int64)
    digs = np.stack([(uu // p**k) % p - half for k in range(nd)], axis=2)
    return digs.reshape(m, rb * nd)[:, :n]

for p, nd, half in ((5, 13, 2), (3, 20, 1), (2, 32, 1)):
    m, n = 3, nd * 2 + 5
    Q = rng.integers(-half, half + 1, size=(m, n), dtype=np.int8)
    u = pack_row(Q, p, nd, half)
    got = unpack_row(u, m, n, p, nd, half)
    print(f"p={p}: u.max={u.max():>10} unpack-match={np.array_equal(got, Q)}")
    if not np.array_equal(got, Q):
        print("  expect[:8]", Q.ravel()[:8])
        print("  got[:8]   ", got.ravel()[:8])