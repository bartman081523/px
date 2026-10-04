#!/usr/bin/env python
"""GF(p)-GEMV-Triton-Kernel (PLAN-Stufe 2) + Decode-Bench.

Konzept (gf5_quant.txt): Gewichte als Base-p-Records (13 Ququints/u32 für
GF5, 20 Trits/u32 für GF3), Digit-Extraktion LSB-zuerst über konstante
Division (Triton: / und % auf Konstante → multiply-shift), mul-freie
Akkumulation (q=0 → 0, ±1 → ±x, ±2 → ±(x+x) — Betrag nur über Add, das
Vorzeichen über Select).

Layout: (RB, M) record-major → koaleszierte 128-Breite-Loads über r;
Partials (CB, M) f32 + Summe — analog ptq10_matvec (runtime_qwen35_ptq).

Bench auf einem ECHTEN Bonsai-Tensor (down_proj 5120×17408, ternär):
GF5 und GF3 sind über den Superset-Satz VERLUSTFREI auf demselben Material,
dense bf16 und ptq10_matvec (PTQ1_0-Layout) dienen als Referenz.

Läuft neben dem Server (benötigt ~0,7 GiB VRAM auf der 12-GB-Karte).
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "px_patches" / "ternary_bonsai_27b_px"))
import runtime_qwen35_ptq as RT  # noqa: E402  (dequant_pack, ptq10_matvec)

HF_FILE = "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf/model.safetensors"

_spec = importlib.util.spec_from_file_location(
    "gf5q", REPO / "ququint_quant" / "gf5_llm_quantizer.py")
Q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q)

ND = {5: 13, 3: 20}          # Digits per Record (Base-5/Base-3, LSB zuerst)
HALFV = {5: 2, 3: 1}         # (p-1)//2 — Zustand {−HALF..+HALF}


# ── Pack: (m, n) ternär → u32 (RB, m) + Skalare, LSB-Rang zuerst, k=1 ─────
def pack_gfp_from_ternary(W, p, block=128):
    """W (m, n) f32 ∈ {−d,0,+d} je Block → Q, u32 (RB, m) record-major.

    Block-orientiertes Repacking: pro `block`-Block (Bonsai: 128, mit fp16-
    Skalar d je Zeile×Block — identisch zu dequant_pack) entsteht
    rpb = ceil(block/nd) Base-p-Records; Padding: Zustand 0 (digit=HALF).
    Nur so stimmt die Skalar-Granularität mit dem Ausgangsformat überein —
    verlustfrei auf ternärem Material (Superset: Clip wird nie aktiv).
    Rückgabe: Q (m, n), u uint32 (RB, m), Sb fp16 (nb, m) — RB = Nb·rpb.
    """
    m, n = W.shape
    nd = ND[p]
    nb = n // block
    rpb = -(-block // nd)                       # ceil(block/nd): 10 (GF5), 7 (GF3)
    Wb = W.reshape(m, nb, block)
    amax = np.maximum(np.abs(Wb).max(axis=2, keepdims=True), 1e-12)
    S16 = (amax / HALFV[p]).astype(np.float16)
    Qb = np.clip(np.rint(Wb / S16.astype(np.float32)), -HALFV[p], HALFV[p]).astype(np.int8)
    # Superset-Assert: Clip darf NIE aktiv werden (ternäre Zustände exakt)
    Qraw = np.rint(Wb / S16.astype(np.float32))
    assert np.array_equal(Qb, Qraw), f"GF{p}: Clip aktiviert → kein Superset"
    pad_w = rpb * nd - block                    # GF5: 2, GF3: 12 Pad-Gewichte
    d = Qb.astype(np.int32) + HALFV[p]
    d = np.concatenate([d, np.full((m, nb, pad_w), HALFV[p], np.int32)], axis=2)
    pk = p ** np.arange(nd, dtype=np.int64)
    u = (d.reshape(m, nb, rpb, nd).astype(np.int64)
         * pk[None, None, None, :]).sum(axis=3)              # (m, nb, rpb)
    assert u.max() <= p ** nd - 1, f"Base-{p}-Pack-Überlauf"
    # T → astype (uint32: echtes 4-Byte-Format) → copy (C-contiguous)
    u = u.reshape(m, nb * rpb).T.astype(np.uint32).copy()   # (RB, m)
    Sb = S16[:, :, 0].T.copy()                  # (nb, m) fp16 block-major
    return Qb.reshape(m, n), u, Sb


# ── Triton-Kernel ──────────────────────────────────────────────────────────
@triton.jit
def _gfp_mv(u_ptr, s_ptr, x_ptr, part_ptr, M, RB,
            ND: tl.constexpr, P: tl.constexpr, RPB: tl.constexpr,
            BLOCK_R: tl.constexpr, RC: tl.constexpr):
    """part[cb, r] = Σ_over_RC_records s_rr·( Σ_k q_k·x[rr·ND+k] ).

    Digits mul-frei (Betrag über x+x, Vorzeichen über Select); der
    Block-Skalar s (fp16, Bonsai-kompatibel) wird je Record einmal
    geladen (rr//RPB → Bonsai-Block) und EINmal multipliziert.
    """
    pid_r = tl.program_id(0)
    cb = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = r < M
    HALF = (P - 1) // 2
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    c0 = cb * RC
    for i in range(RC):
        rr = c0 + i
        # Mask-Guard statt Scalar-If: Records >= RB aus den Loads
        # ausgeschlossen (kein OOB); s=0 neutralisiert ihren Beitrag
        m2 = rmask & (rr < RB)
        # uint32-Bytes werden als int32-Zeiger übergeben; .to(int64)
        # sign-extendet → Maske holt die unsigned Digit-Welt zurück
        # (GF3-Digits erreichen 3^20−1 ≈ 3,49e9 > 2^31)
        u = tl.load(u_ptr + rr * M + r, mask=m2, other=0).to(tl.int64) & 0xFFFFFFFF
        s = tl.load(s_ptr + (rr // RPB) * M + r, mask=m2, other=0).to(tl.float32)
        a2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
        uu = u
        base = rr * ND
        for k in tl.static_range(ND):
            dv = uu % P
            uu = uu // P
            xc = tl.load(x_ptr + base + k)          # x ist vorgepadet
            qv = dv - HALF                          # {−2..2} / {−1..1}
            # mul-frei: Betrag (|q|=2 → x+x), Vorzeichen via Select
            two = (qv == 2) | (qv == -2)
            xs = tl.where(two, xc + xc, xc)
            a2 += tl.where(qv > 0, xs,
                           tl.where(qv < 0, -xs, 0.0))
        acc += s * a2
    tl.store(part_ptr + cb * M + r, acc, mask=rmask)


RPB = {5: 10, 3: 7}          # Records je Bonsai-128-Block


def gfp_matvec(u_gpu, s_gpu, x_f32, m, n, p, block_r=128, rc=16, x_pad=None):
    nb, rpb = n // 128, RPB[p]
    rb, cb = nb * rpb, triton.cdiv(nb * rpb, rc)
    nd = ND[p]
    # x auf das Record-Raster aufspannen: pro Block 128 Gewichte + pad_w
    # Null-Spalten — Digits rr·nd+k ↔ Gewicht (b·128+j) für j<128
    # (call-invariant → in einer Runtime gecacht; Bench übergibt x_pad)
    if x_pad is None:
        x_pad = torch.zeros(nb, rpb * nd, dtype=torch.float32, device="cuda")
        x_pad[:, :128] = x_f32.reshape(nb, 128)
        x_pad = x_pad.reshape(-1)
    part = torch.empty(cb, m, dtype=torch.float32, device="cuda")
    _gfp_mv[(triton.cdiv(m, block_r), cb)](
        u_gpu, s_gpu, x_pad, part, m, rb, ND=nd, P=p, RPB=rpb,
        BLOCK_R=block_r, RC=rc)
    return part.sum(0)


# ── Bench ──────────────────────────────────────────────────────────────────
def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def main():
    name = "model.layers.0.mlp.down_proj.weight"
    dev = "cuda"
    torch.cuda.init()
    free0, total = torch.cuda.mem_get_info()
    print(f"VRAM frei: {free0/2**30:.2f} GiB / {total/2**30:.2f} GiB")

    from safetensors import safe_open
    with safe_open(HF_FILE, framework="pt", device="cpu") as f:
        packed_u8 = f.get_tensor(name)
    w = RT.dequant_pack(packed_u8)                    # (m, Nb, 128) ∈ {−d,0,+d}
    m_, nb, bw = w.shape
    n_ = nb * bw
    W = w.reshape(m_, n_).contiguous().numpy().astype(np.float32)
    print(f"Material: {name}: {tuple(W.shape)} ({m_*n_/1e6:.1f} M Gewichte)")

    x = torch.randn(n_, dtype=torch.float32, device=dev)
    x_bf = x.to(torch.bfloat16)
    W_gpu = torch.from_numpy(W).to(dev)                          # f32 Referenz
    W_bf = W_gpu.to(torch.bfloat16)
    packed_u8_dev = packed_u8.to(dev)
    y_ref = W_gpu @ x                                            # f32 Ground-Truth

    Q5, u5, S5 = pack_gfp_from_ternary(W, 5)
    Q3, u3, S3 = pack_gfp_from_ternary(W, 3)
    # Ende-zu-Ende-Verlustfreiheit: Q·S (per Block aufgeweitet) == W exakt
    Wb = W.reshape(m_, n_ // 128, 128)
    for p, Qf, half in ((5, Q5, 2), (3, Q3, 1)):
        S16 = np.maximum(np.abs(Wb).max(axis=2, keepdims=True), 1e-12) / half
        Sfull = np.repeat(S16, 128, axis=2).reshape(m_, n_)
        assert np.array_equal(Qf * Sfull, W), \
            f"GF{p}: Requant-Roundtrip nicht exakt (kein Verlustfrei-Beweis)"
    # uint32-Landmine: Triton-Pointer als int32 (selbe 4 Bytes); die Maske
    # im Kernel holt Unsigned-Semantik zurück
    u5_dev = torch.from_numpy(u5.view(np.int32)).to(dev)
    u3_dev = torch.from_numpy(u3.view(np.int32)).to(dev)
    s5_dev = torch.from_numpy(S5).to(dev)
    s3_dev = torch.from_numpy(S3).to(dev)

    def y_dense():
        return torch.nn.functional.linear(x_bf, W_bf).float()

    def y_ptq10():
        return RT.ptq10_matvec(packed_u8_dev, x_bf).float()

    # call-invariant x-Pad einmal bauen (Runtime-Cache-Modell)
    nb_ = n_ // 128
    xpads = {}
    for p in (5, 3):
        xp = torch.zeros(nb_, RPB[p] * ND[p], dtype=torch.float32, device="cuda")
        xp[:, :128] = x.reshape(nb_, 128)
        xpads[p] = xp.reshape(-1)

    def y_gf5():
        return gfp_matvec(u5_dev, s5_dev, x, m_, n_, 5, x_pad=xpads[5])

    def y_gf3():
        return gfp_matvec(u3_dev, s3_dev, x, m_, n_, 3, x_pad=xpads[3])

    rows = []
    for nm, fn, bytes_mv in (
            ("dense_bf16", y_dense, m_ * n_ * 2),
            ("ptq10_u8", y_ptq10, m_ * nb * 28),
            ("gf5_u32", y_gf5, u5.nbytes + S5.nbytes),
            ("gf3_u32", y_gf3, u3.nbytes + S3.nbytes)):
        ms = bench(fn)
        y = fn().squeeze()
        err = float((y - y_ref).abs().max())
        gb = bytes_mv / ms / 1e6                          # GB/s effektiv
        rows.append({"fmt": nm, "bytes": bytes_mv, "ms": round(ms, 4),
                     "gbps": round(gb, 1), "max_abs_err": round(err, 4)})
        print(f"  {nm:<12} {bytes_mv/1e6:>7.1f} MB  {ms:>8.3f} ms  "
              f"{gb:>7.1f} GB/s  max|Δ|={err:.4f}")

    # Bit-Bilanz inkl. Skalare und Block-Align-Pad: pro 128er-Block je
    # rpb Records à 4 Byte + fp16-Skalar
    bits = {"dense_bf16": 16.0, "ptq10_u8": 28 / 128}
    for p in (5, 3):
        bits[f"gf{p}_u32"] = (RPB[p] * 4 * 8 + 16) / 128
    out = {"tensor": name, "shape": list(W.shape),
           "vram_free_gib": round(free0 / 2**30, 2), "rows": rows,
           "bits_weight": bits}
    dst = Path(__file__).parent / "results_kernel_v0_2.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"OK → {dst}")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()