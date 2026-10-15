"""Weight-Level-Divergenz: recon vs official auf Stichproben-Tensoren.

Frage (L1-Interpretation): Ist der prism-GGUF eine Raffinierung desselben
Modells (kleine Mittelwerte, max|dw| <= d) oder ein fremder Quantisierungs-
lauf (breite Abweichung)? Misst pro Tensor:
  frac_w_diff  Anteil Gewichte mit w_official != w_recon
  max|dw|/d    groesste relative Abweichung (skaliert)
"""
import sys
import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
                 "/scratches/ptq10_gguf")
from diff_recon_official import fold  # noqa: E402

OFFICIAL = "/home/julian/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
RECON = "/home/julian/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0-recon.gguf"


def dequant(buf):
    """(N, 28) uint8 -> (N, 128) f32 — exakt wie das C-dequant (keine
    id==0-Sonderzweig, reiner Fold)."""
    blk = buf.reshape(-1, 28)
    d = blk[:, 26:28].copy().view(np.float16).reshape(-1, 1).astype(np.float32)
    parts = []
    for off, nb_bytes, nsegs in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
        arr = []
        for n in range(nsegs):
            col = np.empty((blk.shape[0], nb_bytes), dtype=np.uint8)
            for i in range(nb_bytes):
                b = blk[:, off + i]
                q = (b.astype(np.uint16) * np.uint16(3 ** n)) \
                    & np.uint16(0xFF)
                col[:, i] = ((q * np.uint16(3)) >> np.uint16(8))
            arr.append(col)
        parts.append(np.concatenate(arr, axis=1))   # (N, nsegs*nb_bytes)
    # Position e = n*16 + i fuer qs80 (Byte i, Shift n), analog qs40/qh
    w = np.concatenate(parts, axis=1).astype(np.float32) - 1.0   # (N, 128)
    return w * d


def main():
    r_off = GGUFReader(OFFICIAL)
    r_rec = GGUFReader(RECON)
    pick = ["blk.0.attn_gate.weight", "blk.3.mlp.up_proj.weight",
            "blk.19.self_attn.q_proj.weight",
            "blk.5.linear_attn.out_proj.weight"]
    for to in r_off.tensors:
        if to.name not in pick:
            continue
        tr = next(t for t in r_rec.tensors if t.name == to.name)
        wo = dequant(np.frombuffer(to.data, dtype=np.uint8).copy())
        wr = dequant(np.frombuffer(tr.data, dtype=np.uint8).copy())
        same_pos = wo == wr
        frac = 1.0 - same_pos.mean()
        rel = np.abs(wo - wr).max() / max(np.abs(wo).max(), 1e-30)
        print(f"{to.name:44s} frac_w_diff={frac:8.4%} "
              f"max|dw|/wmax={rel:.3f}")


if __name__ == "__main__":
    main()