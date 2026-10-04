#!/usr/bin/env python
"""GF(p)-Quantisierungs-Familie v0.2 (PLAN-Stufe 1 + GF3/GF2-Ableitung).

GF(5)/Ququint (Stufe 1): 5 Zustände {−2,−1,0,1,2}, 13 Digits/u32.
Abgeleitet: GF(3)/Triquint: 3 Zustände {−1,0,+1}, 20 Digits/u32 — auf
ternärem Material SUPERSET (exakt, wie GF5). GF(2): BNN-Sign (±1),
32 Digits/u32 — extremes Bit-Format ohne Nullstufe (Qualität ehrlich
mitgemessen; XNOR/popcount-Kernel-Familie = PLAN-Stufe 2b, nicht hier).

Alle Formate: per-Block-Skalare (fp16-Ziel), LSB-zuerst-Pack, verlustfreier
Roundtrip inkl. Randfälle. Messung rel-L2 gegen ternär (BitNet-1.58-Stil)
und INT4-sym auf synthetischen Verteilungen und echtem Bonsai-Material.

Nur CPU (kein Triton/CUDA, kein VRAM) — läuft neben dem Server.

    python ququint_quant/gf5_llm_quantizer.py            # voll
    python ququint_quant/gf5_llm_quantizer.py --quick   # nur synthetisch
"""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
HF_FILE = "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf/model.safetensors"
KGRID = [round(0.50 + 0.05 * i, 2) for i in range(17)]   # 0.50 … 1.30

# Digits pro uint32-Record (LSB zuerst) — GF5: 5^13 ≤ 2^32, GF3: 3^20 ≤ 2^32,
# GF2: 32 Digits (2^32 Record-Maximum)
DIGITS_U32 = {5: 13, 3: 20, 2: 32}

# Nominal-Informationsbits (flat-pack; effektiv siehe eval_matrix)
NOMINAL_BITS = {"gf5": 32 / 13, "gf3": 32 / 20, "gf2": 1.0,
                "ter": math.log2(3), "int4": 4.0}
SCALE_BITS = lambda block: 16 / block


# ── GF(5)-Quantisierung ────────────────────────────────────────────────────
def block_view(W, block):
    """(m, n) → (m*n/block, block); n muss Vielfaches von block sein."""
    m, n = W.shape
    if n % block:
        raise ValueError(f"n={n} nicht teilbar durch Block {block}")
    return W.reshape((m * n) // block, block)


def quantize(Wb, qmax, ks):
    """Symmetrische Quantisierung auf q-Levels ±qmax; globaler k Fit.

    Wb: (nb, block) float32 → Q int8, S float32 (nb,1), k_opt, err².
    """
    amax = np.abs(Wb).max(axis=1, keepdims=True)
    amax = np.where(amax > 0, amax, 1.0)
    best = None
    for k in ks:
        S = np.maximum(k * amax / qmax, 1e-12)
        Q = np.clip(np.rint(Wb / S), -qmax, qmax).astype(np.int8)
        err = float((E2 := (Wb - (Q * S)) ** 2).sum())
        if best is None or err < best[3]:
            best = (Q, S.astype(np.float32), float(k), err)
    return best


def dequantize(Q, S):
    return (Q * S).ravel().astype(np.float32)


def rel_l2(W, Wh):
    return float(np.sqrt(((W - Wh) ** 2).sum() / (W ** 2).sum()))


def quantize_gfp(Wb, p, ks):
    """Familien-Quantisierung: GF(5)/(3) → quantize(qmax=(p−1)/2).

    GF(2) → BNN-Sign: Q = sign(W) (KEINE Nullstufe), nur der Skalar wird
    gefittet (S = k·mean|W|; Optimum liegt trivial bei k=1 — Grid deckt
    das ab). Das Vorzeichen ist fix, daher hoher erwartbarer Fehler.
    """
    if p == 2:
        S0 = np.maximum(np.abs(Wb).mean(axis=1, keepdims=True), 1e-12)
        Q = np.where(Wb >= 0, 1, -1).astype(np.int8)
        best = None
        for k in ks:
            S = np.maximum(k * S0, 1e-12)
            err = float(((Wb - (Q * S)) ** 2).sum())
            if best is None or err < best[3]:
                best = (Q, S.astype(np.float32), float(k), err)
        return best
    return quantize(Wb, (p - 1) // 2, ks)


# ── Base-p-Pack: 13/20/32 Digits pro uint32, LSB-Rang zuerst ──────────────
def pack_base_p(Q, p):
    """Q flatt → (u uint32-Array, n_orig); Padding: Digit des Nullzustands.

    GF(5)/(3): d = q + (p−1)/2 ∈ {0..p−1}, Pad = (p−1)/2 (Zustand 0).
    GF(2): d = (q+1)//2 ∈ {0,1} (decode q = 2d−1) — direkt q+1 wäre {0,2}
    und KEIN gültiges Basis-2-Digit (Überlauf-Beweis: kernel_debug.py);
    Pad bei GF2 willkürlich (es gibt keine Nullstufe) → 0.
    """
    nd = DIGITS_U32[p]
    if p == 2:
        d = (Q.astype(np.int32).ravel() + 1) // 2
        pad = 0
    else:
        pad = (p - 1) // 2
        d = Q.astype(np.int32).ravel() + pad
    n = d.size
    nw = (n + nd - 1) // nd
    dp = np.full(nw * nd, pad, dtype=np.int32)
    dp[:n] = d
    pk = p ** np.arange(nd, dtype=np.int64)
    u = (dp.reshape(nw, nd).astype(np.int64) * pk).sum(axis=1)
    assert u.max() <= p ** nd - 1, f"Base-{p}-Pack-Überlauf"
    return u.astype(np.uint32), n


def unpack_base_p(u, n, p):
    nd = DIGITS_U32[p]
    uu = u.astype(np.int64)
    out = np.empty((uu.size, nd), dtype=np.int8)
    for k in range(nd):
        d = (uu // p ** k) % p
        out[:, k] = (2 * d - 1) if p == 2 else (d - (p - 1) // 2)
    return out.ravel()[:n]


def test_roundtrip():
    rng = np.random.default_rng(0)
    for p in (5, 3, 2):
        nd = DIGITS_U32[p]
        qmax = 1 if p == 2 else (p - 1) // 2
        if p == 2:      # ±1-Dichte ohne Nullstufe (kein Zustand 0!)
            Q = (rng.integers(0, 2, size=10_000, dtype=np.int8) * 2 - 1).astype(np.int8)
        else:
            Q = rng.integers(-qmax, qmax + 1, size=10_000, dtype=np.int8)
        u, n = pack_base_p(Q, p)
        assert np.array_equal(unpack_base_p(u, n, p), Q), f"GF{p} roundtrip"
        for ext, u_expect in ((-qmax, 0), (qmax, p ** nd - 1)):  # Grenzfälle
            Q2 = np.full(nd, ext, dtype=np.int8)
            u2, n2 = pack_base_p(Q2, p)
            assert int(u2[0]) == u_expect, f"GF{p} Boundary {ext}: {u2[0]}"
            assert np.array_equal(unpack_base_p(u2, n2, p), Q2.ravel())
    return True


# ── Mess-Suite ─────────────────────────────────────────────────────────────
# Format-Tripletts: (key, p in GF-Familie oder 0, qmax der Referenzformate)
FAMILY = (("gf5", 5, None), ("gf3", 3, None), ("gf2", 2, None),
          ("ter", 0, 1), ("int4", 0, 7))


def eval_matrix(name, W, block, results):
    Wb = block_view(W, block)
    row = {"name": name, "shape": list(W.shape), "n": int(W.size), "block": block}
    for key, p, qmax in FAMILY:
        if p:                                    # GF-Familie (GF2 = BNN intern)
            Q, S, k_opt, _ = quantize_gfp(Wb, p, KGRID)
        else:                                    # Referenz: ternär / INT4-sym
            Q, S, k_opt, _ = quantize(Wb, qmax, KGRID if key == "ter" else [1.0])
        err = rel_l2(W, dequantize(Q, S).reshape(W.shape))
        sp = float((Q == 0).mean())
        row[f"{key}_fit"] = {"rel_l2": round(err, 6), "k": k_opt,
                             "sparsity": round(sp, 4)}
    # Pack + verlustfreier Roundtrip je GF-Familie
    for key, p, _ in FAMILY[:3]:
        Qg, *_ = quantize_gfp(Wb, p, KGRID)
        u, n_ = pack_base_p(Qg, p)
        assert np.array_equal(unpack_base_p(u, n_, p), Qg.ravel()), \
            f"GF{p}-Pack-Roundtrip verletzt"
        row[f"pack_{key}"] = {"bytes": int(u.nbytes),
                              "bits_effective": round(u.nbytes * 8 / max(n_, 1)
                                                      + SCALE_BITS(block), 4)}
    results.append(row)
    return row


def fmt_row(row):
    cells = [f"{row['name'][:24]:<24}", f"{row['n']:>9}"]
    for k in ("gf5", "gf3", "gf2", "ter", "int4"):
        cells.append(f"{row[f'{k}_fit']['rel_l2']:>9.4f}")
    for k in ("gf5", "gf3", "gf2"):
        cells.append(f"{row[f'pack_{k}']['bits_effective']:>7.2f}")
    return "  ".join(cells)


# ── Echtes Material ────────────────────────────────────────────────────────
def load_real_f32(n_layers=8):
    """linear_attn in_proj_{a,b} (F32, 48×5120 je Schicht) als echte Gewichte.

    Hybrid-Layout: nur 48 der 64 Layer haben linear_attn (die übrigen
    self_attn) — daher die Namen sammeln statt Layer-Indizes zu raten.
    """
    from safetensors import safe_open
    names = []
    with safe_open(HF_FILE, framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        for k in keys:
            # exakt: in_proj_{a,b} sind F32; in_proj_qkv/in_proj_z sind U8-ternär!
            if k.endswith((".linear_attn.in_proj_a.weight",
                           ".linear_attn.in_proj_b.weight")):
                names.append(k)
        names = names[: 2 * n_layers]
        if len(names) < 2 * n_layers:
            raise RuntimeError(f"nur {len(names)} in_proj-Tensoren gefunden")
        rows = [f.get_tensor(k).numpy() for k in names]
    return np.concatenate(rows, axis=0).astype(np.float32)


def load_runtime():
    path = REPO / "px_patches/ternary_bonsai_27b_px/runtime_qwen35_ptq.py"
    spec = importlib.util.spec_from_file_location("rtq", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_real_ternary(name):
    """Dequant Bonsai-Tensor → (m, 128·Nb) float32 — ternäre Zustände."""
    from safetensors import safe_open
    mod = load_runtime()
    with safe_open(HF_FILE, framework="pt", device="cpu") as f:
        packed = f.get_tensor(name)
    w = mod.dequant_pack(packed)                # (m, Nb, 128) ∈ {−d,0,+d}
    m, nb, b = w.shape
    return w.reshape(m, nb * b).numpy().astype(np.float32)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="nur synthetisch (ohne Gewichts-Load)")
    ap.add_argument("--block", type=int, default=128,
                    help="Blockgröße (128 = Bonsai-PTQ1_0-Konvention)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results_v0_2.json"))
    args = ap.parse_args()
    block = args.block
    rng = np.random.default_rng(7)

    results = []
    header = (f"{'Matrix':<24}  {'n':>9}  {'GF5fit':>9}  {'GF3fit':>9}  "
              f"{'GF2fit':>9}  {'TERfit':>9}  {'INT4':>9}"
              f"  {'bGF5':>7}  {'bGF3':>7}  {'bGF2':>7}")
    print(header)
    print("-" * len(header))

    # ── synthetisch ──
    W = rng.standard_normal((512, 512), dtype=np.float32)
    print(fmt_row(eval_matrix("synthetic_gauss512", W, block, results)))
    W = rng.laplace(0.0, 1.0, (512, 512)).astype(np.float32)
    print(fmt_row(eval_matrix("synthetic_laplace512", W, block, results)))
    out = rng.standard_normal((512, 512), dtype=np.float32)
    out[:, ::20] *= 8.0                      # 5 % Outlier-Kanäle (LLM-typisch)
    print(fmt_row(eval_matrix("synthetic_outliers512", out, block, results)))

    # ── Bonsai ternär → GF5/GF3 (Superset: erwartet exakt 0) ──
    assert test_roundtrip(), "Base-p-Roundtrip verlustfrei"
    wt = load_real_ternary("model.layers.0.mlp.down_proj.weight")
    rt = eval_matrix("real_ternary_superset", wt, 128, results)
    print(fmt_row(rt))
    assert rt["gf5_fit"]["rel_l2"] < 1e-9, "GF5 ⊉ ternär?! Superset verletzt"
    assert rt["gf3_fit"]["rel_l2"] < 1e-9, "GF3 ⊉ ternär?! Superset verletzt"

    # ── echtes F32 ──
    if not args.quick:
        wr = load_real_f32()
        print(fmt_row(eval_matrix("real_inproj_f32_8L", wr, block, results)))
        t0 = time.perf_counter()
        u, n = pack_base_p(Qg := (np.clip(np.rint(wr / (np.abs(wr).max() / 2)),
                                          -2, 2).astype(np.int8)), 5)
        t_pack = time.perf_counter() - t0
        t0 = time.perf_counter()
        unpack_base_p(u, n, 5)
        t_un = time.perf_counter() - t0
        pack_bench = {"n_weights": int(wr.size),
                      "pack_ms": round(t_pack * 1e3, 2),
                      "unpack_ms": round(t_un * 1e3, 2),
                      "bits": round(u.nbytes * 8 / wr.size, 4)}
        print(f"  pack-bench: {pack_bench}")

    print("-" * len(header))
    # Bits/Gewicht: flat (global aligniert, wenig Pad) vs block-aligniert
    # (Bonsai-128er-Blöcke: Pad-Verschwendung + fp16-Skalar einkalkuliert)
    flat = lambda p: 32 / DIGITS_U32[p]
    aligned = lambda p: (-(-128 // DIGITS_U32[p]) * 4 * 8 + 16) / 128
    print(f"Bits/Gewicht flat  : GF5 {flat(5):.4f} | GF3 {flat(3):.4f} | "
          f"GF2 {flat(2):.4f}")
    print(f"Bits/Gewicht 128blk: GF5 {aligned(5):.4f} | GF3 {aligned(3):.4f} | "
          f"GF2 {aligned(2):.4f} | Bonsai PTQ1_0 {28 * 8 / 128:.4f}")
    print(f"(Referenz inkl. fp16-Skalare bei block={block}: GF5 "
          f"{32/13 + 16/block:.3f} | GF3 {32/20 + 16/block:.3f} | "
          f"GF2 {1.0 + 16/block:.3f} | ternär {math.log2(3) + 16/block:.3f} bit)")

    payload = {
        "version": "0.2",
        "kgrid": KGRID,
        "digits_u32": DIGITS_U32,
        "roundtrip_lossless": True,
        "nominal_bits": NOMINAL_BITS,
        "results": results,
    }
    if not args.quick:
        payload["pack_bench"] = pack_bench
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"OK → {args.out}")


if __name__ == "__main__":
    main()