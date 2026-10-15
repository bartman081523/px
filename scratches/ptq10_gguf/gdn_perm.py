"""Permutations-Mapping mine -> official fuer GDN-Tensoren je Layer 0.

Matcht Zeilen byte-exakt (Sign + Werte) und druckt die Abbildung
komprimiert, plus Tests fuer alpha/beta/conv1d/dt/A_log.
"""
import sys
import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
                 "/scratches/ptq10_gguf")
from wdiv_sample import dequant  # noqa: E402
from htest_hadamard import OFFICIAL, RECON  # noqa: E402


def load_bytes(r, name):
    to = next(t for t in r.tensors if t.name == name)
    return np.frombuffer(to.data, dtype=np.uint8).copy(), \
        [int(s) for s in to.shape], int(to.tensor_type)


def main():
    r_off = GGUFReader(OFFICIAL)
    r_rec = GGUFReader(RECON)

    for pick, lay in (("blk.0.attn_gate.weight", "PTQ1_0 row-perm"),
                      ("blk.0.attn_qkv.weight", "PTQ1_0 row-perm"),
                      ("blk.0.ssm_conv1d.weight", "F32 row-perm?")):
        bo, ne, tt = load_bytes(r_off, pick)
        br, ne_r, _ = load_bytes(r_rec, pick)
        print(f"\n=== {pick} type={tt} ne={ne} vs={ne_r} "
              f"bytes off={bo.size} mine={br.size} ident={np.array_equal(bo, br)}")
        if np.array_equal(bo, br):
            continue
        if tt == 143:
            m = dequant(bo).reshape(ne[1], ne[0])          # (out, in)
            m2 = dequant(br).reshape(ne_r[1], ne_r[0])
            n = ne[1]
        else:
            # F32/BF16: Werte als Zeilenmatrizen bei 2D-Signatur deuten
            dt = np.float32 if tt == 0 else np.uint16
            wo = np.frombuffer(bo.tobytes(), dtype=dt)
            wr = np.frombuffer(br.tobytes(), dtype=dt)
            if len(ne) == 2:
                m = wo.reshape(ne[1], ne[0])
                m2 = wr.reshape(ne_r[1], ne_r[0])
            else:
                print("  1D/3D — Zeilen-Mapping uebersprungen")
                continue
            n = ne[1]
        # Zeilen byte-exakt matchen (als gequantelte int32-Skalen normalisieren)
        keyo = np.ascontiguousarray(m).view(np.int8).reshape(n, -1)
        keyr = np.ascontiguousarray(m2).view(np.int8).reshape(n, -1)
        # Normalisiere je Zeile: Werte / max|Werte| (Skaleninvariante Signatur)
        def sig(k):
            z = k.astype(np.float32).reshape(len(k), -1)
            mz = np.abs(z).max(1, keepdims=True)
            return np.where(mz > 0, np.rint(z / np.maximum(mz, 1e-30) * 1e4),
                            0).astype(np.int32)
        so, sr = sig(keyo), sig(keyr)
        # Match via Sortierreihenfolge der Zeilenbytes
        idx_r = np.lexsort(sr.T[::-1])
        idx_o = np.lexsort(so.T[::-1])
        same = np.array_equal(so[idx_o], sr[idx_r])
        print(f"  sortierte Signaturzeilen identisch: {same}")
        if same:
            perm = np.empty(n, dtype=np.int64)
            perm[idx_o] = idx_r             # official-row i == mine-row perm[i]
            # Komprimierung: Stueckweise affine Muster ausgeben
            segs = []
            for i in range(n):
                if not segs or segs[-1][2] != "inv" and False:
                    pass
            # Muster drucken: erste 40 Eintraege + Segmentzusammenfassung
            print("  perm[:48] =", perm[:48].tolist())
            # Segmente gleicher Stuktur grob: diff von perm
            dp = np.diff(perm)
            jumps = np.nonzero((dp != 1) & (dp != -1))[0]
            print(f"  nicht-monotone Sprungstellen: {len(jumps)}, "
                  f"erste: {jumps[:20].tolist()}")
            print("  perm bei Spruengen:",
                  [(int(j), int(perm[j]), int(perm[j + 1])) for j in jumps[:8]])


if __name__ == "__main__":
    main()