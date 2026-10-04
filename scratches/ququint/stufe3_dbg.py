#!/usr/bin/env python
"""GF3-Matvec-Debug: Bisekt der Kernel-Komponenten.

Stufe A: Ein-Record-Kernel (Grid-Achse 1 = rr direkt, kein RC/kein Guard)
          -> per-Record-Partials vs Hand-Dekode aus u-Bytes.
Stufe B: RC-Chunking mit SCALAR-IF-Guard (Variante des Runtime-Kernels).
Stufe C: RC-Chunking mit MASK-Guard  (m2 = rmask & (rr < RB)).
Stufe D: RC-Chunking, RB%-16==0-Material, SCALAR-IF (Guard tot).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
for pth in (ROOT / "ququint_quant", ROOT / "px_patches" / "ternary_bonsai_27b_px"):
    sys.path.insert(0, str(pth))
import repack_gf3 as RG
import gf3_quant as G3
import triton
import triton.language as tl


@triton.jit
def _dbg1(u_ptr, s_ptr, x_ptr, part_ptr, M, RB, XP,
          ND: tl.constexpr, RPB: tl.constexpr, BLOCK_R: tl.constexpr):
    # Grid: (cdiv(M, BLOCK_R), RB, mm)  — EIN Record je Programm
    pid_r = tl.program_id(0)
    rr = tl.program_id(1)
    mi = tl.program_id(2)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = r < M
    u = tl.load(u_ptr + rr * M + r, mask=rmask, other=0).to(tl.int64) & 0xFFFFFFFF
    s = tl.load(s_ptr + (rr // RPB) * M + r, mask=rmask, other=0).to(tl.float32)
    a2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
    uu = u
    base = rr * ND
    for k in tl.static_range(ND):
        dv = uu % 3
        uu = uu // 3
        xc = tl.load(x_ptr + mi * XP + base + k)
        qv = dv - 1
        a2 += tl.where(qv > 0, xc, tl.where(qv < 0, -xc, 0.0))
    tl.store(part_ptr + rr * M + r, s * a2, mask=rmask)


@triton.jit
def _dbg2(u_ptr, s_ptr, x_ptr, part_ptr, M, RB, CB, XP,
          ND: tl.constexpr, RPB: tl.constexpr, BLOCK_R: tl.constexpr,
          RC: tl.constexpr, USE_MASK_GUARD: tl.constexpr):
    # Grid: (cdiv(M, BLOCK_R), CB, mm); RC Records pro cb-Programm
    pid_r = tl.program_id(0)
    cb = tl.program_id(1)
    mi = tl.program_id(2)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = r < M
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    c0 = cb * RC
    for i in range(RC):
        rr = c0 + i
        if USE_MASK_GUARD:
            ok = rr < RB
            m2 = rmask & ok
        else:
            m2 = rmask
            ok = True
        u = tl.load(u_ptr + rr * M + r, mask=m2, other=0).to(tl.int64) & 0xFFFFFFFF
        s = tl.load(s_ptr + (rr // RPB) * M + r, mask=m2, other=0).to(tl.float32)
        a2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
        uu = u
        base = rr * ND
        for k in tl.static_range(ND):
            dv = uu % 3
            uu = uu // 3
            xc = tl.load(x_ptr + mi * XP + base + k)
            qv = dv - 1
            a2 += tl.where(qv > 0, xc, tl.where(qv < 0, -xc, 0.0))
        acc += s * a2
    tl.store(part_ptr + (mi * M + r) * CB + cb, acc, mask=rmask)


def mat_material(m, nb):
    rng = np.random.default_rng(3)
    w = rng.integers(-2, 3, size=(m, nb, 128)).astype(np.float32)
    w[w == 2] = 1.0
    w[w == -2] = -1.0
    d = np.resize(np.array([0.0625, 0.375, 1.5, 3.25]), nb).astype(np.float32)
    w *= d[None, :, None]
    return torch.from_numpy(w)


def run_nb(nb, n, m=192):
    torch.manual_seed(0)
    w = mat_material(m, nb)
    q = torch.sign(w)
    s = w.abs().amax(dim=2)
    u, s16 = RG._pack_chunk(w, q, s)
    u_gpu, s_gpu = u.to("cuda"), s16.to("cuda")
    rb = nb * RG.RPB
    x = torch.randn(1, n, dtype=torch.bfloat16, device="cuda")
    x_f32 = x.float()
    W_ref = (q * s16.t().to(torch.float32).unsqueeze(-1)).reshape(m, n).numpy()
    y_ref = F.linear(x_f32.cpu(), torch.from_numpy(W_ref)).numpy()

    # ------------- Stufe A: ein Record je Programm -------------
    xp = torch.zeros(1, rb * RG.ND, dtype=torch.float32, device="cuda")
    xp[0, :128 * nb] = x_f32[0]
    part = torch.empty(rb, m, dtype=torch.float32, device="cuda")
    _dbg1[(triton.cdiv(m, 128), rb, 1)](u_gpu, s_gpu, xp, part, m, rb, xp.stride(0),
                                        ND=RG.ND, RPB=RG.RPB, BLOCK_R=128)
    yA = part.sum(0).cpu().numpy()
    dA = np.abs(yA - y_ref[0]).max()

    # ------------- Handle-Dekode je Record (Referenz der Partials) -------------
    u_np = u.numpy().astype(np.uint32).astype(np.uint64)
    s_np = s16.numpy().astype(np.float32)
    part_ref = np.zeros((rb, m), dtype=np.float32)
    for rr in range(rb):
        b, rin = rr // RG.RPB, rr % RG.RPB
        for row in range(m):
            rec = int(u_np[rr, row])
            a2 = 0.0
            for k in range(RG.ND):
                dv = rec % 3
                rec //= 3
                j = rin * RG.ND + k
                if j < 128:
                    a2 += (dv - 1) * float(x_f32[0, b * 128 + j])
            part_ref[rr, row] = s_np[b, row] * a2
    dPart = np.abs(part.cpu().numpy() - part_ref).max()

    # ------------- Stufe B/C = Runtime-Kernel mit beiden Guards -------------
    xp2 = torch.zeros(2, rb * RG.ND, dtype=torch.float32, device="cuda")
    xp2[0, :128 * nb] = x_f32[0]
    res = {}
    for name, use_mask in (("scalar_if", False), ("mask_guard", True)):
        cb = triton.cdiv(rb, G3._RC)
        part2 = torch.empty(1, m, cb, dtype=torch.float32, device="cuda")
        _dbg2[(triton.cdiv(m, 128), cb, 1)](
            u_gpu, s_gpu, xp2, part2, m, rb, cb, xp2.stride(0),
            ND=RG.ND, RPB=RG.RPB, BLOCK_R=128, RC=G3._RC, USE_MASK_GUARD=use_mask)
        y2 = part2.sum(-1).cpu().numpy()
        res[name] = np.abs(y2[0] - y_ref[0]).max()

    # ------------- Stufe D: Runtime-Kernel unverändert -------------
    yR = G3._gf3_kernel_matvec(u_gpu, s_gpu, x_f32, m, n).cpu().numpy()
    dR = np.abs(yR[0] - y_ref[0]).max()

    print(f"nb={nb:<3} RB={rb:<4}: StufeA(1Rec)={dA:.6g}  Partials={dPart:.6g}  "
          f"scalar_if={res['scalar_if']:.6g}  mask_guard={res['mask_guard']:.6g}  "
          f"runtime={dR:.6g}")


run_nb(4, 512)
run_nb(16, 2048)