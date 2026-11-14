#!/usr/bin/env python
"""GF(5)-GEMM (Prefill) — batched Triton-Kernel über das Kernel-Record-Layout.

Warum ein eigener Kernel: der Decode-Pfad (`gfp_matvec`) ist ein Matvec;
für Prefill (T > 1 Tokens) dequantisiert der Fallback bisher 278 Matrizen
in numpy (~20 s pro Prompt). Dieser Kernel rechnet direkt aus den gepackten
Records, ohne die Gewichte zu materialisieren.

Layout (identisch zu `gf5_gemv.py`):
    u: (RB, M) uint32/int32, RB = (N/128)·RPB, LSB-Digit zuerst
    s: (N/128, M) fp16, ein Skalar je (128er-Block, Ausgabezeile)

Mathematik je 128er-Block b und Record i:
    w[m, k] = q(u[b·RPB+i, m], k) · s[b, m],  k = 0..12
    acc    += X[:, b·128 + i·13 + k] · w[m, k]
Das sind 10 Dots mit K = 16 (13 Digits + 3 Null-Pads) pro Block — K ≥ 16 ist
die Tensor-Core-Untergrenze von `tl.dot`.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

ND = 13
RPB = 10
KPAD = 16          # 13 Digits + 3 Nullen (tl.dot braucht K >= 16)


@triton.jit
def _gfp_mm(u_ptr, s_ptr, x_ptr, out_ptr, kpow_ptr, M, N, T,
            ND: tl.constexpr, P: tl.constexpr, RPB: tl.constexpr,
            KPAD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rm < M
    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = rt < T

    HALF: tl.constexpr = (P - 1) // 2
    # 5^k kommt als Tensor herein: `P ** tl.arange(...)` kennt Triton nicht.
    kpow = tl.load(kpow_ptr + tl.arange(0, KPAD)).to(tl.int64)
    kvalid = tl.arange(0, KPAD) < ND

    acc = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
    nb = N // 128
    for b in range(nb):
        s = tl.load(s_ptr + b * M + rm, mask=rmask, other=0.0).to(tl.float32)
        for i in tl.static_range(RPB):
            rr = b * RPB + i
            u = tl.load(u_ptr + rr * M + rm, mask=rmask, other=0).to(tl.int64) & 0xFFFFFFFF
            q = ((u[:, None] // kpow[None, :]) % P) - HALF          # (BM, KPAD)
            q = tl.where(kvalid[None, :], q, 0)
            w = q.to(tl.float32) * s[:, None]                       # (BM, KPAD)
            cols = b * 128 + i * ND + tl.arange(0, KPAD)
            cmask = tmask[:, None] & (cols[None, :] < N)
            xs = tl.load(x_ptr + rt[:, None] * N + cols[None, :],
                         mask=cmask, other=0.0)                     # (BT, KPAD)
            acc += tl.dot(xs, tl.trans(w))

    tl.store(out_ptr + rt[:, None] * M + rm[None, :],
             acc.to(out_ptr.dtype.element_ty),
             mask=tmask[:, None] & rmask[None, :])


def gfp_matmul(u, s, x2d: torch.Tensor, m: int, n: int, p: int = 5,
               block_m: int = 64, block_t: int = 64) -> torch.Tensor:
    """x2d (T, n) → (T, m), direkt aus gepackten Records."""
    assert p == 5 and n % 128 == 0, "nur GF(5), n Vielfaches von 128"
    T = x2d.shape[0]
    x = x2d.contiguous()
    out = torch.empty((T, m), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(T, block_t))
    kpow = torch.tensor([5 ** k for k in range(KPAD)], dtype=torch.int64, device=x.device)
    _gfp_mm[grid](u, s, x, out, kpow, m, n, T,
                  ND=ND, P=p, RPB=RPB, KPAD=KPAD,
                  BLOCK_M=block_m, BLOCK_T=block_t)
    return out
