"""Formel-Beweis: official == mine[tbl]-Zusammensetzung? (array_equal)

Formeln (alle layer-invariant, 48 Einheiten):
  tbl[i] = (i//16) + 3*(i % 16)
  gate:      off_blk[i]  = mine_blk[tbl[i]]
  qkv:       off_blk[0..31] = mine_blk[0..31];
             off_blk[32+j]  = mine_blk[32 + tbl[j]]
  conv1d:    off_ch = qkv-Formel auf Kanaelen
  alpha/beta: off_row[i] = mine_row[tbl[i]]
  dt.bias:   off[i]     = mine[tbl[i]]
  ssm_a:     Diagnose (ident? tbl? Raushen)
"""
import sys
import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
                 "/scratches/ptq10_gguf")
from wdiv_sample import dequant  # noqa: E402
from htest_hadamard import OFFICIAL, RECON  # noqa: E402

TBL = np.array([(i // 16) + 3 * (i % 16) for i in range(48)], dtype=np.int64)


def t2(r, name):
    to = next(t for t in r.tensors if t.name == name)
    ne = [int(s) for s in to.shape]
    tt = int(to.tensor_type)
    raw = np.frombuffer(to.data, dtype=np.uint8).copy()
    if tt == 143:
        return dequant(raw).reshape(ne[1], ne[0]), ne, tt, raw
    if len(ne) == 1:
        dt = np.float32 if tt == 0 else np.uint16
        return np.frombuffer(raw, dtype=dt), ne, tt, raw
    if tt == 0:
        return np.frombuffer(raw, dtype=np.float32).reshape(ne[1], ne[0]), \
            ne, tt, raw
    if tt == 30:
        return np.frombuffer(raw, dtype=np.uint16).reshape(ne[1], ne[0]), \
            ne, tt, raw
    return np.frombuffer(raw, dtype=np.float16).reshape(ne[1], ne[0]), \
        ne, tt, raw


def qkv_formula(mr):
    nblk = mr.shape[0] // 128
    assert nblk == 80, nblk
    out = np.empty_like(mr)
    out[:32 * 128] = mr[:32 * 128]
    for j in range(48):
        src = 32 + TBL[j]
        out[(32 + j) * 128:(33 + j) * 128] = mr[src * 128:(src + 1) * 128]
    return out


def block_formula(mr):
    nblk = mr.shape[0] // 128
    assert nblk == 48, nblk
    out = np.empty_like(mr)
    for i in range(nblk):
        out[i * 128:(i + 1) * 128] = mr[TBL[i] * 128:(TBL[i] + 1) * 128]
    return out


def row_formula(mr):
    return mr[TBL]


def main():
    r_off = GGUFReader(OFFICIAL)
    r_rec = GGUFReader(RECON)
    layers = [0, 32, 62]

    tests = [
        ("attn_gate.weight", block_formula),
        ("attn_qkv.weight", qkv_formula),
        ("ssm_alpha.weight", row_formula),
        ("ssm_beta.weight", row_formula),
        ("ssm_dt.bias", row_formula),
    ]
    for suffix, fn in tests:
        for L in layers:
            mo, ne, _, _ = t2(r_off, f"blk.{L}.{suffix}")
            mr, ner, _, _ = t2(r_rec, f"blk.{L}.{suffix}")
            pred = fn(mr)
            ok = bool(np.array_equal(pred, mo))
            print(f"blk.{L}.{suffix:20s} FORMEL-ERGEBNIS = "
                  f"{'EXAKT' if ok else 'FEHLER'}  "
                  f"(max|d|={np.abs(pred - mo).max():.3g})")

    # conv1d: Kanalachse (Spalten) mit qkv-Formel
    for L in layers:
        mo, ne, tt, _ = t2(r_off, f"blk.{L}.ssm_conv1d.weight")
        mr, _, _, _ = t2(r_rec, f"blk.{L}.ssm_conv1d.weight")
        # (10240 Kanael x 4) bzw. Zeilen = Kanael? ne=[4, 10240],
        # reshape -> (ne1, ne0) = (4, 10240)? Klärung:
        print(f"blk.{L}.ssm_conv1d ne={ne} shape geladen={mo.shape} tt={tt}")
        # Kanal-Achse ist die 10240-Achse
        if mo.shape[1] == 10240:
            pred = np.empty_like(mr)
            pred[:, :32 * 128] = mr[:, :32 * 128]
            for j in range(48):
                src = 32 + TBL[j]
                pred[:, (32 + j) * 128:(33 + j) * 128] = \
                    mr[:, src * 128:(src + 1) * 128]
        else:                                    # Zeilen = Kanael
            pred = np.empty_like(mr)
            pred[:32 * 128] = mr[:32 * 128]
            for j in range(48):
                src = 32 + TBL[j]
                pred[(32 + j) * 128:(33 + j) * 128] = \
                    mr[src * 128:(src + 1) * 128]
        ok = bool(np.array_equal(pred, mo))
        print(f"blk.{L}.ssm_conv1d FORMEL = {'EXAKT' if ok else 'FEHLER'}")

    # ssm_a: Diagnose
    for L in layers:
        mo, ne, tt, rawo = t2(r_off, f"blk.{L}.ssm_a")
        mr, _, _, rawr = t2(r_rec, f"blk.{L}.ssm_a")
        print(f"\nblk.{L}.ssm_a ne={ne} ident={np.array_equal(rawo, rawr)}")
        for label, idx in (("ident", np.arange(48)), ("tbl", TBL)):
            d = np.abs(mo - mr[idx])
            print(f"  off vs mine[{label}]: max|d|={d.max():.6g}  "
                  f"gleich={np.array_equal(mo, mr[idx])}")
        print("  off[0:6] =", mo[:6].tolist())
        print("  mine[0:6]=", mr[:6].tolist())


if __name__ == "__main__":
    main()