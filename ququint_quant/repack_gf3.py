#!/usr/bin/env python
"""Stufe-3-Converter: PTQ1_0-ternary-Safetensors -> GF(3)-Artefakt.

Liest model.safetensors (5,98 GB, Tensor (m, Nb, 28) uint8, ID143-Layout
qs[0:24] + qh[24:26] + d fp16 @ 26:28) und repackt jeden Ternary-Tensor
VERLUSTFREI nach GF(3):

    Q = sign(w)            # w = dequant_pack -> {−d, 0, +d} (exakt)
    S = max|w| per Block   # = |d| desselben Blocks (fp16-exakt)

Q*S == w koerperlich (kein Requant-Rounding, Clip nie aktiv), deshalb ist
der GF3-Artefakt-Matvec gegen das urspruengliche PTQ1_0-Gewicht bit-treu
im Level-Sinn (max|Delta| 0.0000, wie im Stufe-2-Bench bewiesen).

Storage pro Ternary-Tensor W (m, n):
    <key>.gq3 : int32 (RB, m)   uint32-Records als Bit-Pattern (int32-View;
                                GF3-Records bis 3^20-1 > 2^31 — Kernel maskiert
                                & 0xFFFFFFFF nach int64-Sign-Extend)
    <key>.gs3 : fp16 (nb, m)    block-major Skalare (Kernel: rr // RPB)
RB = Nb * 7 (7 Records je 128-Block, 20 Digits je Record, 12 Pad-Digits
Nullzustand). Alle sonstigen Tensoren 1:1 kopiert.

GPU-Dequant, row-gechunkt (~1,4 GB Transient je 8192 Zeilen) — laeuft
NEBEN dem Server (freie VRAM reicht). Zielpfad default
~/.cache/huggingface/ternary-bonsai-2-27b-hf/gf3_model.safetensors
(ML4-Disk hat nur 12 GB frei -> /home).

    python ququint_quant/repack_gf3.py [--sample N] [--out P]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_pp = str(Path(__file__).resolve().parent.parent / "px_patches" /
          "ternary_bonsai_27b_px")
if _pp not in sys.path:
    sys.path.insert(0, _pp)
import runtime_qwen35_ptq as RT  # noqa: E402

HF_DIR = "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf"
ROW_CHUNK = 8192

ND = 20          # Digits pro uint32-Record (GF3, LSB zuerst)
RPB = 7          # Records je 128-Block
PAD = RPB * ND - 128   # 12 Pad-Digits (Nullzustand digit=1)
P3 = [3 ** k for k in range(ND)]


def _pack_chunk(w, q, s):
    """(c, nb, 128) f32-Ternary + Q + S -> (u (RB=c-chunk, m) int32, S16 (nb, m))."""
    c, nb, bw = q.shape
    dq = (q.to(torch.int32) + 1)                     # 0..2
    if PAD > 0:
        dq = torch.cat([dq, torch.full((c, nb, PAD), 1,
                                       dtype=torch.int32, device=dq.device)], dim=2)
    dq = dq.view(c, nb, RPB, ND).permute(0, 1, 2, 3)  # (c, nb, 7, 20)
    u = torch.zeros(c, nb, RPB, dtype=torch.int64, device=dq.device)
    for k in range(ND):
        u += dq[..., k].to(torch.int64) * P3[k]
    u = u.view(c, nb * RPB).transpose(0, 1).contiguous()   # (RB, c)
    assert int(u.max()) < 2**32                             # Record < u32
    # Bit-Pattern: int64 -> uint32 -> int32 (niederwertiges Wort, hi=0) —
    # torch Tensor.view(dtype) skaliert die LETZTE Achse und wuerde
    # (RB, c) -> (RB, c*2) verdoppeln; NumPy-view korrekt (u ist CUDA ->
    # zuerst .cpu())
    u32 = torch.from_numpy(
        u.cpu().numpy().astype(np.uint32).copy().view(np.int32))
    s16 = s.transpose(0, 1).to(torch.float16).contiguous().cpu()  # (nb, c)
    return u32, s16


def convert(src=HF_DIR + "/model.safetensors",
            dst=HF_DIR + "/gf3_model.safetensors",
            sample_check=8):
    from safetensors import safe_open
    from safetensors.torch import save_file

    t_start = time.perf_counter()
    with safe_open(src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        out = {}
        n_ternary = n_small = 0
        n_weights = 0
        checked = 0
        tern_keys = []
        for ki, key in enumerate(keys):
            t = f.get_tensor(key)
            # Ternary-Erkennung per FORM: u8 (m, nb, 28) — nur lm_head/embed
            # tragen den Suffix ".ternary", Layer-Gewichte heissen ".weight"
            is_tern = t.dtype == torch.uint8 and t.ndim == 3 and t.shape[-1] == 28
            if not is_tern:
                out[key] = t
                n_small += 1
                continue
            m_rows, nb, _ = t.shape
            us, ss = [], []
            for lo in range(0, m_rows, ROW_CHUNK):
                hi = min(lo + ROW_CHUNK, m_rows)
                chunk = t[lo:hi].to("cuda")
                w = RT.dequant_pack(chunk)               # (c, nb, 128) f32 ∈ {−d,0,+d}
                q = torch.sign(w)
                s = w.abs().amax(dim=-1)                 # (c, nb) = |d| exakt
                s16 = s.to(torch.float16)
                # Verlustfrei-Assert (bitweise): Q·S16 == w
                w_rec = (q * s16.to(torch.float32).unsqueeze(-1))
                if checked < sample_check or (ki % 50 == 0):
                    diff = (w_rec - w).abs().max().item()
                    assert diff == 0.0, f"{key}[{lo}:{hi}]: Q·S != w ({diff})"
                    checked += 1
                u_chunk, s_chunk = _pack_chunk(w, q, s)
                us.append(u_chunk)
                ss.append(s_chunk)
                del chunk, w, q, s, w_rec
            u = torch.cat(us, dim=1)                     # (RB, m) int32
            s16 = torch.cat(ss, dim=1)                   # (nb, m) fp16
            out[key + ".gq3"] = u
            out[key + ".gs3"] = s16
            n_ternary += 1
            n_weights += m_rows * nb * 128
            tern_keys.append(key)
            if ki % 25 == 0:
                el = time.perf_counter() - t_start
                print(f"  [{ki+1}/{len(keys)}] {key[:60]} m={m_rows} nb={nb} "
                      f"({el:.1f}s)", flush=True)
    # safetensors-Metadaten verlangen string-Werte ('int' object is not an
    # instance of 'str' — deshalb der Abbruch nach abgeschlossenem Packen)
    meta = {
        "format": "gf3", "p": "3", "nd": str(ND), "rpb": str(RPB),
        "block": "128",
        "source": "model.safetensors (PTQ1_0 ID143)",
        "lossless": "Q=sign(w), S=amax|w| (fp16-exakt) -> Q*S==w bitweise",
        "n_ternary_tensors": str(n_ternary), "n_small_tensors": str(n_small),
        "n_weights": str(int(n_weights)),
        "ternary_keys": json.dumps(tern_keys),
    }
    save_file(out, dst, metadata=meta)
    el = time.perf_counter() - t_start
    print(f"OK {dst} ({n_ternary} ternary, {n_small} copy, "
          f"{n_weights/1e9:.2f} Mrd Gewichte, {el:.0f}s)")
    return meta


def verify(dst=HF_DIR + "/gf3_model.safetensors", n_checks=6):
    """Sampled Roundtrip: GF3-artifact dekodiert == PTQ1_0-dequant (Zufallszeilen)."""
    from safetensors import safe_open
    rng = np.random.default_rng(0)
    t_start = time.perf_counter()
    with safe_open(dst, framework="pt", device="cpu") as fmeta:
        keys = json.loads((fmeta.metadata() or {})["ternary_keys"])
    with safe_open(HF_DIR + "/model.safetensors", framework="pt", device="cpu") as fsrc, \
         safe_open(dst, framework="pt", device="cpu") as fdst:
        pick = [keys[i] for i in rng.choice(len(keys), size=min(n_checks, len(keys)),
                                            replace=False)]
        for key in pick:
            src_t = fsrc.get_tensor(key)
            m_rows = src_t.shape[0]
            rid = int(rng.integers(0, m_rows))
            w_ref = RT.dequant_pack(src_t[rid:rid + 1])[0]   # (nb, 128) f32
            # gq3 ist (RB, m) record-major — Gewichtszeile rid = SPALTE rid
            u = fdst.get_tensor(key + ".gq3")[:, rid].numpy()       # (RB,) int32
            s16 = fdst.get_tensor(key + ".gs3")[:, rid].numpy()     # (nb,) fp16
            uu = u.astype(np.uint32).astype(np.uint64)
            nb = s16.shape[0]
            w2 = np.zeros((nb, 128), dtype=np.float32)
            for b in range(nb):
                for r in range(RPB):
                    rec = int(uu[b * RPB + r])
                    s = s16[b]
                    for k in range(ND):
                        d = rec % 3
                        rec //= 3
                        j = r * ND + k
                        if j < 128:
                            w2[b, j] = (d - 1) * s
            print(f"  {key[14:52]} row {rid}: max|Δ| "
                  f"{np.abs(w2 - w_ref.numpy()).max():.6g}")
            assert np.array_equal(w2, w_ref.numpy()), f"{key} roundtrip!="
    print(f"VERIFY OK ({len(pick)} Tensoren, {time.perf_counter()-t_start:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HF_DIR + "/gf3_model.safetensors")
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if args.verify_only:
        verify(args.out)
        return
    convert(dst=args.out, sample_check=args.sample)
    verify(args.out)


if __name__ == "__main__":
    main()