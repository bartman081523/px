"""GDN-Vollprovenienz: Layer-Invarianz + alle GDN-Nebentensoren.

1. GDN-Layer aus off.tensors sammeln (attn_gate vorhanden)
2. Block-Tabellen gate/qkv/conv1d auf 3 Layern — identisch?
3. alpha/beta (bf16-2D): Zeilen-Permutationstest + Tabelle
4. ssm_a / ssm_dt.bias (1D f32): Element-Permutation
"""
import sys
import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
                 "/scratches/ptq10_gguf")
from wdiv_sample import dequant  # noqa: E402
from htest_hadamard import OFFICIAL, RECON  # noqa: E402


def tensor(r, name):
    return next((t for t in r.tensors if t.name == name), None)


def load2d(r, name):
    """Gibt (matrix-f32-equiv, ne, tensor_type) oder (None, ne, tt)."""
    to = tensor(r, name)
    if to is None:
        return None, None, None
    ne = [int(s) for s in to.shape]
    tt = int(to.tensor_type)
    raw = np.frombuffer(to.data, dtype=np.uint8).copy()
    if tt == 143:
        w = dequant(raw).reshape(ne[1], ne[0])          # (out, in)
        return w, ne, tt
    if len(ne) == 1:
        dt = np.float32 if tt == 0 else np.uint16
        w = np.frombuffer(raw, dtype=dt)
        return w, ne, tt
    if tt == 0:
        w = np.frombuffer(raw, dtype=np.float32).reshape(ne[1], ne[0])
    elif tt == 30:                                       # bf16-Rohbits
        w = np.frombuffer(raw, dtype=np.uint16).reshape(ne[1], ne[0])
    else:
        w = np.frombuffer(raw, dtype=np.float16).reshape(ne[1], ne[0])
    return w, ne, tt


def rowperm(mo, mr):
    """official-Zeile i <- mine-Zeile perm[i]; None wenn keine Permutation."""
    n = mo.shape[0]
    ko = np.ascontiguousarray(mo).view(np.uint8).reshape(n, -1)
    kr = np.ascontiguousarray(mr).view(np.uint8).reshape(n, -1)
    so = np.lexsort(ko.T[::-1])
    sr = np.lexsort(kr.T[::-1])
    if not np.array_equal(ko[so], kr[sr]):
        return None
    perm = np.empty(n, np.int64)
    perm[so] = sr
    return perm


def blktbl(perm, w=128):
    """Block-Tabelle, wenn perm je Block konstant ist; sonst None."""
    n = perm.size
    tb = perm.reshape(-1, w)
    if not (tb == tb[:, :1]).all():
        return None
    return tb[:, 0] // w


def fmt(tbl):
    if tbl is None:
        return "KEINE Block-perm"
    d = np.diff(tbl)
    return (f"steps={np.unique(d, return_counts=True)} "
            f"tbl[:16]={tbl[:16].tolist()}")


def main():
    r_off = GGUFReader(OFFICIAL)
    r_rec = GGUFReader(RECON)

    gdn_layers = sorted(int(t.name.split(".")[1]) for t in r_off.tensors
                        if t.name.endswith(".attn_gate.weight"))
    print(f"GDN-Layer ({len(gdn_layers)}): {gdn_layers[:12]} … {gdn_layers[-3:]}")

    # 2) Layer-Invarianz auf gate/qkv/conv1d
    picks3 = [gdn_layers[0], gdn_layers[len(gdn_layers) // 2], gdn_layers[-1]]
    for suffix in ("attn_gate.weight", "attn_qkv.weight", "ssm_conv1d.weight"):
        tbls = {}
        for L in picks3:
            mo, ne, _ = load2d(r_off, f"blk.{L}.{suffix}")
            mr, ner, _ = load2d(r_rec, f"blk.{L}.{suffix}")
            ident = bool(np.array_equal(mo, mr))
            p = None if ident else rowperm(mo, mr)
            tbls[L] = "IDENT" if ident else fmt(blktbl(p) if p is not None
                                                else None)
            print(f"  blk.{L}.{suffix} ne={ne} ident={ident}")
            if not ident:
                print(f"    {tbls[L]}")
        same = len({str(v) for v in tbls.values()}) == 1
        print(f"  => {suffix}: Layer-invariant = {same}")

    # 3) alpha/beta (2D, rohbytes) auf allen drei Layern
    for suffix in ("ssm_alpha.weight", "ssm_beta.weight"):
        for L in picks3[:2]:
            mo, ne, tt = load2d(r_off, f"blk.{L}.{suffix}")
            mr, ner, _ = load2d(r_rec, f"blk.{L}.{suffix}")
            ident = bool(np.array_equal(mo, mr))
            print(f"  blk.{L}.{suffix} tt={tt} ne={ne} ident={ident}")
            if ident:
                continue
            p = rowperm(mo, mr)
            if p is None:
                print("    KEINE Zeilen-Permutation (numerisch?)")
                # Zeilen-diff-Statistik
                dmask = np.any(np.ascontiguousarray(mo).view(np.uint8)
                               .reshape(mo.shape[0], -1)
                               != np.ascontiguousarray(mr).view(np.uint8)
                               .reshape(mr.shape[0], -1), axis=1)
                print(f"    differing Zeilen: {int(dmask.sum())}/{len(dmask)}")
                continue
            print(f"    steps={np.unique(np.diff(p), return_counts=True)} "
                  f"p[:20]={p[:20].tolist()}")

    # 4) 1D ssm_a / ssm_dt.bias
    for pick in ("ssm_a", "ssm_dt.bias"):
        for L in picks3[:2]:
            mo, ne, tt = load2d(r_off, f"blk.{L}.{pick}")
            mr, _, _ = load2d(r_rec, f"blk.{L}.{pick}")
            ident = bool(np.array_equal(mo, mr))
            print(f"  blk.{L}.{pick} tt={tt} ne={ne} ident={ident}")
            if not ident:
                so = np.argsort(mo, kind="stable")
                sr = np.argsort(mr, kind="stable")
                perm = np.empty(len(mo), np.int64)
                perm[so] = sr
                print(f"    Multiset gleich: "
                      f"{np.array_equal(np.sort(mo), np.sort(mr))}")
                print(f"    perm={perm.tolist()}")


if __name__ == "__main__":
    main()