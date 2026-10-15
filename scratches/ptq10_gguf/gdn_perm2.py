"""Block-Tabellen der GDN-Permutation (head_dim=128-Block-Struktur).

perm[off_zeile] = mine_zeile; Blocks: official-Block b <- mine-Block
print fuer alle Layer-0-Tensoren + Layer-Invarianz-Check auf blk.19.
"""
import sys
import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
                 "/scratches/ptq10_gguf")
from wdiv_sample import dequant  # noqa: E402
from htest_hadamard import OFFICIAL, RECON  # noqa: E402


def load_rows(r, name):
    to = next(t for t in r.tensors if t.name == name)
    ne0, ne1 = int(to.shape[0]), int(to.shape[1])
    tt = int(to.tensor_type)
    if tt == 143:
        w = dequant(np.frombuffer(to.data, dtype=np.uint8).copy()) \
            .reshape(ne1, ne0)
    elif tt == 0:
        w = np.frombuffer(to.data, dtype=np.float32).reshape(ne1, ne0)
    elif tt == 30:
        w = np.frombuffer(to.data, dtype=np.uint16).reshape(ne1, ne0) \
            .view(np.float16) if False else \
            np.frombuffer(to.data, dtype=np.uint16).astype(np.float32) \
            * 1.0  # bf16-Bits: Skalierung egal, Muster zaehlt
        w = np.frombuffer(to.data, dtype=np.uint16).reshape(ne1, ne0)
    return w


def block_perm(mo, mr, blk=128):
    n = mo.shape[0]
    idx_r = np.lexsort(np.ascontiguousarray(mr).view(np.int8)
                       .reshape(n, -1).T[::-1])
    idx_o = np.lexsort(np.ascontiguousarray(mo).view(np.int8)
                       .reshape(n, -1).T[::-1])
    ok = np.array_equal(mo[idx_o], mr[idx_r])
    if not ok:
        return None
    perm = np.empty(n, dtype=np.int64)
    perm[idx_o] = idx_r
    return perm[idx_r] if False else perm            # official-row i <- mine perm[i]


def main():
    r_off = GGUFReader(OFFICIAL)
    r_rec = GGUFReader(RECON)
    for pick in ("blk.0.attn_gate.weight", "blk.0.attn_qkv.weight",
                 "blk.19.attn_gate.weight", "blk.19.attn_qkv.weight"):
        mo, mr = load_rows(r_off, pick), load_rows(r_rec, pick)
        p = block_perm(mo, mr)
        nb = p.size // 128
        tbl = p[::128] // 128
        print(f"\n=== {pick} rows={p.size} blk-table (off<-mine) ===")
        print(tbl.tolist())
        # Segmentdetektion: laufende Blöcke mit Schrittweite s
        d = np.diff(tbl)
        print("  blk-Schritte:", np.unique(d, return_counts=True))


if __name__ == "__main__":
    main()