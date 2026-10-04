#!/usr/bin/env python
"""GF(5)/Ququint-Quantisierungs-Prototyp v0.1 (PLAN-Stufe 1).

Quantisiert Matrizen auf 5 Zustände {−2,−1,0,1,2} mit per-Block-Skalaren,
packt 13 Ququints pro uint32 (Basis 5, LSB zuerst) und misst den
Rekonstruktionsfehler gegen ternär (BitNet-1.58-Stil) und symmetrischem
INT4 — auf synthetischen Verteilungen und echten F32-Gewichten des
Ternary-Bonsai-2-27B. Zusätzlich: Superset-Nachweis (ternäre Zustände
werden von GF5 exakt repräsentiert).

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

# Nominal-Informationsbits + effektiv (inkl. fp16-Skalare, Blockgröße B)
NOMINAL_BITS = {"gf5": 32 / 13, "ter": math.log2(3), "int4": 4.0}
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


# ── Base-5-Pack: 13 Ququints / uint32, LSB-Rang zuerst ────────────────────
def pack_base5(Q):
    """Q flatt → (u uint32-Array, n_orig); Padding: q=0 (digit 2)."""
    d = Q.astype(np.int32).ravel() + 2
    n = d.size
    nw = (n + 12) // 13
    pad = np.full(nw * 13, 2, dtype=np.int32)
    pad[:n] = d
    p5 = 5 ** np.arange(13, dtype=np.int64)
    u = (pad.reshape(nw, 13).astype(np.int64) * p5).sum(axis=1)
    assert u.max() <= 5**13 - 1, "Base-5-Pack-Überlauf"
    return u.astype(np.uint32), n


def unpack_base5(u, n):
    u = u.astype(np.int64)
    out = np.empty((u.size, 13), dtype=np.int8)
    for k in range(13):
        out[:, k] = (u // 5**k) % 5 - 2
    return out.ravel()[:n]


def test_roundtrip():
    rng = np.random.default_rng(0)
    Q = rng.integers(-2, 3, size=10_000, dtype=np.int8)
    u, n = pack_base5(Q)
    assert np.array_equal(unpack_base5(u, n), Q)
    for ext, u_expect in ((-2, 0), (2, 5**13 - 1)):       # Grenzfälle
        Q2 = np.full(13, ext, dtype=np.int8)
        u2, n2 = pack_base5(Q2)
        assert int(u2[0]) == u_expect, f"Boundary {ext}: {u2[0]}"
        assert np.array_equal(unpack_base5(u2, n2), Q2.ravel())
    return True


# ── Mess-Suite ─────────────────────────────────────────────────────────────
def eval_matrix(name, W, block, results):
    Wb = block_view(W, block)
    row = {"name": name, "shape": list(W.shape), "n": int(W.size), "block": block}
    for key, qmax in (("gf5", 2), ("ter", 1), ("int4", 7)):
        for variant, ks in (("amax", [1.0]), ("fit", KGRID)):
            Q, S, k_opt, _ = quantize(Wb, qmax, ks)
            err = rel_l2(W, dequantize(Q, S).reshape(W.shape))
            sp = float((Q == 0).mean())
            row[f"{key}_{variant}"] = {"rel_l2": round(err, 6), "k": k_opt,
                                       "sparsity": round(sp, 4)}
    # Pack auf der echten GF5-Quantisierung (fit)
    Qg, *_ = quantize(Wb, 2, KGRID)
    u, n = pack_base5(Qg)
    assert np.array_equal(unpack_base5(u, n), Qg.ravel())
    row["packed_bytes"] = int(u.nbytes)
    row["bits_weight_effective"] = round(u.nbytes * 8 / max(n, 1)
                                         + SCALE_BITS(block), 4)
    results.append(row)
    return row


def fmt_row(row):
    cells = [f"{row['name'][:26]:<26}", f"{row['n']:>9}"]
    for k in ("gf5_amax", "gf5_fit", "ter_amax", "ter_fit", "int4_amax"):
        cells.append(f"{row[k]['rel_l2']:>9.4f}")
    for k in ("gf5_fit", "ter_fit"):
        cells.append(f"  z={row[k]['sparsity']:.2f}")
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
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results_v0_1.json"))
    args = ap.parse_args()
    block = args.block
    rng = np.random.default_rng(7)

    results = []
    header = (f"{'Matrix':<26}  {'n':>9}  {'GF5amax':>9}  {'GF5fit':>9}  "
              f"{'TERamax':>9}  {'TERfit':>9}  {'INT4':>9}")
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

    # ── Bonsai ternär → GF5 (Superset: erwartet exakt 0) ──
    assert test_roundtrip(), "Base-5-Roundtrip verlustfrei"
    wt = load_real_ternary("model.layers.0.mlp.down_proj.weight")
    rt = eval_matrix("real_ternary_superset", wt, 128, results)
    print(fmt_row(rt))
    assert rt["gf5_fit"]["rel_l2"] < 1e-9, "GF5 ⊉ ternär?! Superset verletzt"

    # ── echtes F32 ──
    if not args.quick:
        wr = load_real_f32()
        print(fmt_row(eval_matrix("real_inproj_f32_8L", wr, block, results)))
        t0 = time.perf_counter()
        u, n = pack_base5(Qg := (np.clip(np.rint(wr / (np.abs(wr).max() / 2)),
                                         -2, 2).astype(np.int8)))
        t_pack = time.perf_counter() - t0
        t0 = time.perf_counter()
        unpack_base5(u, n)
        t_un = time.perf_counter() - t0
        pack_bench = {"n_weights": int(wr.size),
                      "pack_ms": round(t_pack * 1e3, 2),
                      "unpack_ms": round(t_un * 1e3, 2),
                      "bits": round(u.nbytes * 8 / wr.size, 4)}
        print(f"  pack-bench: {pack_bench}")

    print("-" * len(header))
    print(f"blöcke à {block} Gewichte, fp16-Skalare: effektiv "
          f"GF5 {32/13 + 16/block:.3f} bit | ternär {math.log2(3) + 16/block:.3f} bit | "
          f"INT4 {4.0 + 16/block:.3f} bit  (nominal GF5 {32/13:.4f})")

    payload = {
        "version": "0.1",
        "kgrid": KGRID,
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