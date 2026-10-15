"""Per-Tensor-Byte-Diff: recon vs official PTQ1_0-GGUF.

Klassifiziert jede differierende Byte-Position im 28-B-Block:
  0-15  qs80  (5 Trit-Bytes, 80 Digit-Positionen)
  16-23 qs40  (8 Trit-Bytes, 40 Positionen)
  24-25 qh    (2 Bytes, 8 Positionen, LSB-Trit implizit 0)
  26-27 scale (fp16-Bits)
Prueft je Byte-Paar die Fold-Aequivalenz (decode_fold identisch?)
und vergleicht die Scale-Bits global.
"""
import sys
import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

OFFICIAL = "/home/julian/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
RECON = "/home/julian/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0-recon.gguf"


def fold(b, n):
    q = (np.uint16(b) * np.uint16(3 ** n)) & np.uint16(0xFF)
    return int((q * np.uint16(3)) >> np.uint16(8) & 0xFF)


def main():
    r_off = GGUFReader(OFFICIAL)
    r_rec = GGUFReader(RECON)
    assert len(r_off.tensors) == len(r_rec.tensors)
    total_diff = 0
    seg_hist = {k: 0 for k in ("qs80", "qs40", "qh", "scale")}
    pairs_seen = {}
    scale_mismatch = 0
    tensors_diff = []
    for i, (to, tr) in enumerate(zip(r_off.tensors, r_rec.tensors)):
        assert to.name == tr.name, (to.name, tr.name)
        a = np.frombuffer(to.data, dtype=np.uint8)
        b = np.frombuffer(tr.data, dtype=np.uint8)
        assert a.size == b.size, (to.name, a.size, b.size)
        dif = np.nonzero(a != b)[0]
        if dif.size == 0:
            continue
        tensors_diff.append(to.name)
        total_diff += int(dif.size)
        blk = dif // 28
        pos = dif % 28
        if pos.max() >= 26:
            scale_mismatch += int((pos >= 26).sum())
        for idx, p in zip(dif[:4000], pos % 28):
            seg = ("qs80" if p < 16 else "qs40" if p < 24
                   else "qh" if p < 26 else "scale")
            seg_hist[seg] += 1
            key = (int(a[idx]), int(b[idx]))
            eq = all(fold(key[0], n) == fold(key[1], n) for n in range(5))
            pairs_seen.setdefault(key, [0, eq])
            pairs_seen[key][0] += 1
        if len(tensors_diff) <= 6:
            print(f"[{i}] {to.name}: {dif.size} Diff-Bytes "
                  f"(erste an {dif[:5].tolist()}, "
                  f"Blockbytes {pos[:5].tolist()})")
        else:
            print(f"[{i}] {to.name}: {dif.size} Diff-Bytes")
    print(f"\nTOTAL diff bytes = {total_diff} in {len(tensors_diff)} Tensoren")
    print(f"Segment-Histogramm: {seg_hist}")
    print(f"Scale-Byte-Mismatches: {scale_mismatch}")
    print(f"\nByte-Paar-Klassen (official, mine) -> Fold-einheitlich?:")
    for (bo, bm), (n, eq) in sorted(pairs_seen.items(),
                                    key=lambda kv: -kv[1][0]):
        d = bo - bm
        v_o = [fold(bo, n) for n in range(5)]
        v_m = [fold(bm, n) for n in range(5)]
        print(f"  official={bo:3d} mine={bm:3d} delta={d:+d} x{n:6d} "
              f"fold-eq={eq} quints_o={v_o[:3]}...")

    # Fold-Preimage-Analyse: welche Bytes teilen einen Quint?
    if pairs_seen:
        print("\n--- Pack-Gegenprobe: kanonisches ceil-pack vs official ---")
        v_of_official = None
        # Fuer die haeufigsten Paare: welchen v packt pack( ) zu welchem Byte?
        P = {}
        for v in range(243):
            pb = ((v * 256 + 242) // 243)
            P.setdefault(pb, []).append(v)
        for (bo, bm), (n, eq) in sorted(pairs_seen.items(),
                                        key=lambda kv: -kv[1][0]):
            vo = P.get(bo, [])
            vm = P.get(bm, [])
            print(f"  official={bo}: pack^{-1}={vo[:4]}{'MULTI' if len(vo)>1 else ''}"
                  f" | mine={bm}: pack^{-1}={vm[:4]}")


if __name__ == "__main__":
    main()