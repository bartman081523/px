"""GDN-Layout-Test: Ist official(attn_gate/attn_qkv) eine Reihen-Permutation
meiner GF3-Variante (gdn_v_grouped-Layout) oder ein Hadamard-Fold?

Test je Tensor:
  1. row-multiset: sortierte Zeilen-Hashes identisch? -> reine Permutation
  2. fold-Test: fold(mine) mit D/H (Sign-Vektor) vs official — Agreement
"""
import sys
import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
                 "/scratches/ptq10_gguf")
from wdiv_sample import dequant  # noqa: E402
from htest_hadamard import (field_i32, sylvester_normalized,
                            OFFICIAL, RECON, BLOCK)  # noqa: E402


def load_mat(r, name):
    to = next(t for t in r.tensors if t.name == name)
    ne0, ne1 = int(to.shape[0]), int(to.shape[1])
    w = dequant(np.frombuffer(to.data, dtype=np.uint8).copy())
    # Blöcke liegen je Zeile kontiguell entlang ne0 (K-Richtung)
    return w.reshape(ne1, ne0) if w.size == ne0 * ne1 else w


def row_multiset_diff(a, b):
    """Anteil Zeilen, die im Multiset der anderen Datei fehlen."""
    ha = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape[0], -1)
    hb = np.ascontiguousarray(b).view(np.uint8).reshape(b.shape[0], -1)
    # Hash pro Zeile (SipHash via hashlib waere langsam — nutze sum/hash)
    from collections import Counter
    def keys(m):
        # quantisierte Werte sind ganzzahlig-skaliert: normalisiere je Zeile
        # auf Signatur: (Sign-Muster, Werte/d-Zeile) — Werte sind 0/±d
        s = np.sign(m)
        return Counter(map(lambda row: row.tobytes(), s.astype(np.int8)))
    ka, kb = keys(a), keys(b)
    # Multiset-Symmetriedifferenz auf SIGN-Mustern (skaleninvariant)
    diff = sum(((ka - kb) + (kb - ka)).values())
    return diff, a.shape[0]


def main():
    r_off = GGUFReader(OFFICIAL)
    r_rec = GGUFReader(RECON)
    widths = field_i32(r_off, "prism.hadamard.sign_widths")
    vals = field_i32(r_off, "prism.hadamard.sign_values")
    dmap, off = {}, 0
    for w in widths:
        dmap[int(w)] = vals[off:off + int(w)].astype(np.float32)
        off += int(w)
    H = sylvester_normalized(BLOCK)

    for pick in ("blk.0.attn_gate.weight", "blk.0.attn_qkv.weight"):
        print(f"\n=== {pick} ===")
        mo = load_mat(r_off, pick)
        mr = load_mat(r_rec, pick)
        print("shape official", mo.shape, "| mine", mr.shape,
              "| exakt gleich?" , bool(np.array_equal(mo, mr)))
        d, n = row_multiset_diff(mo, mr)
        print(f"Sign-Zeilen-Multiset-Differenz: {d} Zeilen von {n} "
              f"({'REINE PERMUTATION' if d == 0 else 'KEINE Permutation'})")
        # Fold-Gegenprobe auf Eingangs-Dim (falls in_width ein Sign-Width-Key)
        inw = mr.shape[1]
        if inw in dmap:
            D = dmap[inw]
            Wf = np.empty_like(mr)
            for bi in range(0, inw, BLOCK):
                Wf[:, bi:bi + BLOCK] = (mr[:, bi:bi + BLOCK]
                                        * D[None, bi:bi + BLOCK]) @ H
            # Vergleich im Quant-Raum: wo-Nonzero-sign vs Wf-sign (nur wo!=0)
            nz = np.abs(mo) > 0
            sgn_match = (np.sign(mo) == np.sign(Wf))[nz].mean()
            print(f"fold(mine)-vs-official sign-Agreement (auf official-nonzero):"
                  f" {sgn_match:.4%}")


if __name__ == "__main__":
    main()