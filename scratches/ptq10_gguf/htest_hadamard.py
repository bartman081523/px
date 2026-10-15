"""Hadamard-Fold-Beweis: Ist das offizielle GGUF存储 = W*D*H (folded),
das GF3/neuralworm-W = raw (unfolded)?

Test auf blk.0.ffn_up.weight (in = hidden 5120 = 5 H-Bloecke a 1024):
  fold(dequant(S_mine))  vs  dequant(S_off):  Trit-Vorhersage-Agreement
  (trit = 0 wenn |Wf| < d_off/2, sonst sign)
H = normalisierter Sylvester-Hadamard 1024 (natuerliche Ordnung),
D = Diagonal +/-1 aus prism.hadamard.sign_values (width 5120).
"""
import sys
import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")
from gguf import GGUFReader, GGUFValueType  # noqa: E402

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
                 "/scratches/ptq10_gguf")
from wdiv_sample import dequant  # noqa: E402

OFFICIAL = "/home/julian/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
RECON = "/home/julian/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0-recon.gguf"
BLOCK = 1024


def field_scalar(r, name):
    return int(r.fields[name].contents())


def field_string(r, name):
    return str(r.fields[name].contents())


def field_i32(r, name):
    return np.asarray(r.fields[name].contents(), dtype=np.int32)


def field_strings(r, name):
    return [str(s) for s in r.fields[name].contents()]


def sylvester_normalized(n):
    """(n, n) float32, H[i,j] = (-1)^popcount(i&j)/sqrt(n) — Sylvester-
    Konstruktion in natuerlicher Ordnung, normalisiert."""
    par = np.zeros((n, n), dtype=np.int8)
    step = 128
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        v = (np.arange(lo, hi, dtype=np.uint64)[:, None]
             & np.arange(n, dtype=np.uint64)[None, :])
        c = np.zeros(v.shape, dtype=np.uint8)
        w = v.copy()
        while w.any():
            c += (w & np.uint64(1)).astype(np.uint8)
            w >>= np.uint64(1)
        par[lo:hi] = np.where((c & 1) == 1, -1, 1).astype(np.int8)
    return par.astype(np.float32) / np.sqrt(n)


def main():
    r_off = GGUFReader(OFFICIAL)
    print("prism.hadamard.version =", field_scalar(r_off, "prism.hadamard.version"))
    print("block_size =", field_scalar(r_off, "prism.hadamard.block_size"))
    print("transform =", field_string(r_off, "prism.hadamard.transform"))
    print("sign_mode =", field_string(r_off, "prism.hadamard.sign_mode"))
    widths = field_i32(r_off, "prism.hadamard.sign_widths")
    vals = field_i32(r_off, "prism.hadamard.sign_values")
    names = field_strings(r_off, "prism.hadamard.weight_names")
    inv = field_strings(r_off, "prism.hadamard.inverse_weight_names")
    print("sign_widths =", widths.tolist())
    print("sign_values len =", vals.size, "= sum(widths):",
          int(widths.sum()), "OK" if vals.size == int(widths.sum()) else "MISMATCH")
    print("n weight_names =", len(names), "| erste:", names[:3], "| letzte:", names[-2:])
    print("inverse names =", inv)

    # D-Vektor fuer width 5120 (ffn_up-Eingang = hidden)
    dmap, off = {}, 0
    for w in widths:
        dmap[int(w)] = vals[off:off + int(w)].astype(np.float32)
        off += int(w)
    print("widths keys:", sorted(dmap), "| D5120: unique",
          np.unique(dmap[5120][:8]).tolist(), "…")

    H = sylvester_normalized(BLOCK)

    r_rec = GGUFReader(RECON)
    pick = "blk.0.ffn_up.weight"
    to = next(t for t in r_off.tensors if t.name == pick)
    tr = next(t for t in r_rec.tensors if t.name == pick)
    ne0, ne1 = int(to.shape[0]), int(to.shape[1])          # ne0 = in (schnellste)
    in_dim, out_dim = ne0, ne1
    wo = dequant(np.frombuffer(to.data, dtype=np.uint8).copy()) \
        .reshape(out_dim, in_dim)                          # Blöcke = K-Stripe
    wr = dequant(np.frombuffer(tr.data, dtype=np.uint8).copy()) \
        .reshape(out_dim, in_dim)
    print(f"\n{pick}: shape (out,in) = {out_dim} x {in_dim}, "
          f"H-Bloecke = {in_dim // BLOCK}")

    D = dmap[in_dim]                      # (in,)
    Wf = np.empty_like(wr)
    for b in range(0, in_dim, BLOCK):
        e = b + BLOCK
        Wf[:, b:e] = (wr[:, b:e] * D[None, b:e]) @ H
    # offizielle Skalen je 128-Block aus wo ableiten: d_off = max|wo| pro Blockzeile
    nb = wo.shape[1] // 128
    d_off = np.abs(wo.reshape(wo.shape[0], nb, 128)).max(-1)  # (out, nb)
    d_pred = np.repeat(d_off, 128, axis=1)                    # (out, in)
    trit_off = np.sign(wo) * (np.abs(wo) >= 0.5 * d_pred)
    trit_pred = np.sign(Wf) * (np.abs(Wf) >= 0.5 * d_pred)
    agree = (trit_off == trit_pred).mean()
    nz_off = (trit_off != 0).mean()
    print(f"fold(raw)-vs-official Trit-Agreement = {agree:.4%} "
          f"(nonzero off = {nz_off:.2%})")
    # Kontrolle: raw direkt gg. official (ohne Fold)
    agree_raw = (trit_off == (np.sign(wr) * (np.abs(wr) >= 0.5 * d_pred))).mean()
    print(f"raw-vs-official  Trit-Agreement = {agree_raw:.4%}  (Kontrolle)")

    # Gegenprobe: unfold(official) vs raw — Vorzeichen-Uebereinstimmung
    Wu = np.empty_like(wo)
    for b in range(0, in_dim, BLOCK):
        e = b + BLOCK
        Wu[:, b:e] = (wo[:, b:e] @ H) * D[None, b:e]
    trit_raw = np.sign(wr) != 0
    trit_unf = np.sign(Wu) != 0
    agree_u = (trit_raw == trit_unf).mean()
    print(f"unfold(official)-vs-raw nonzero-Agreement = {agree_u:.4%}")


if __name__ == "__main__":
    main()