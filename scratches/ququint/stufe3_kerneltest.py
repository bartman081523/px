#!/usr/bin/env python
"""Stufe-3-Runtime-Kerneltest: gf3_quant.gf3_matvec_safe + gf3_deq_rows.

Packt synthetisch-ternaeres Material mit repack_gf3._pack_chunk (GLEICHE
Pack-Funktion wie der Konverter), dann:
  1. Kernel-Pfad  M=1 / M=5 / M=8  vs Dekodier-Referenz (erwartet max|Δ|=0)
  2. GEMM-Pfad    M=320            vs bf16-Dekodier-GEMM-Referenz (bitgleich)
  3. gf3_deq_rows direkt          vs Python-Dekodierung
  4. GF3Embedding-Gather-Pfad     (index_select + deq) vs Python-Dekodierung
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for pth in (ROOT / "ququint_quant", ROOT / "px_patches" / "ternary_bonsai_27b_px"):
    sys.path.insert(0, str(pth))
import repack_gf3 as RG          # noqa: E402
import gf3_quant as G3           # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(3)

m, n, nb = 192, 512, 4                            # K=512, RB = 4*7 = 28
w = rng.integers(-2, 3, size=(m, nb, 128)).astype(np.float32)
w[w == 2] = 1.0                                   # ternar {-1,0,+1} * d
w[w == -2] = -1.0
d = np.array([0.0625, 0.375, 1.5, 3.25], dtype=np.float32)   # fp16-exakt, wie PTQ1_0
w *= d[None, :, None]
q = torch.from_numpy(np.sign(w))
s = torch.from_numpy(np.abs(w).max(axis=2))
u, s16 = RG._pack_chunk(torch.from_numpy(w), q, s)      # u (RB, m) cpu int32
u_gpu, s_gpu = u.to("cuda"), s16.to("cuda")
W_dec = (q * s16.t().to(torch.float32).unsqueeze(-1)).reshape(m, n).numpy()
assert np.abs(W_dec - w.reshape(m, n)).max() == 0.0, "Pack nicht verlustfrei"
print(f"Material {w.shape}; u {tuple(u.shape)}, s {tuple(s16.shape)}; "
      f"Pack verlustfrei (CPU-Ziel = Quelle)")

x = torch.randn(10, n, dtype=torch.bfloat16, device="cuda")
x_f32 = x.to(torch.float32)
y_ref = F.linear(x_f32, torch.from_numpy(W_dec).to("cuda"))  # (10, m) f32

for mm in (1, 5, 8):
    sel = x[:mm]
    # (a) f32-Kern strikt: Vorstufe (f32-Partials) vs f32-Referenz — nur
    #     Akkumulationsordnungs-Rauschen (f32-GEMM ~1e-5, Toleranz 1e-4)
    y32 = G3._gf3_kernel_matvec(u_gpu, s_gpu, sel.to(torch.float32), m, n)
    dif = (y32 - y_ref[:mm]).abs().max().item()
    print(f"kernel  M={mm:<2}: f32-Kern max|Δ| vs Referenz = {dif:.6g}")
    assert dif <= 1e-4, f"Kernel-Pfad M={mm} zu ungenau"
    # (b) Wrapper: bf16-Ausgabe ist die exakte Rundung der f32-Vorstufe
    #     (deterministisch, bitweise)
    y = G3.gf3_matvec_safe(u_gpu, s_gpu, sel.to(torch.bfloat16), m, n)
    dif = (y.float() - y32.to(torch.bfloat16).float()).abs().max().item()
    print(f"          Wrapper bf16 bitweise-Rundung: max|Δ| = {dif:.6g}")
    assert dif == 0.0, f"Wrapper M={mm} rundet die Vorstufe nicht exakt"

mm = 320                                                # GEMM-Pfad (M > 8)
xg = torch.randn(mm, n, dtype=torch.bfloat16, device="cuda")
y = G3.gf3_matvec_safe(u_gpu, s_gpu, xg, m, n)
w_bf = torch.from_numpy(W_dec).to("cuda").to(torch.bfloat16)
y_ref2 = F.linear(xg, w_bf)                              # ptq10-aequivalent
dif = (y.float() - y_ref2.float()).abs().max().item()
print(f"GEMM    M=320: shape {tuple(y.shape)}  max|Δ| vs bf16-GEMM={dif:.6g}")
assert dif == 0.0, "GEMM-Pfad nicht bitgleich zur Referenz"

# Dekodierung direkt (Kernel) vs Python-Dekodierung
c = 48
u_sl = u_gpu[:, :c].contiguous()
s_sl = s_gpu[:, :c].contiguous()
wd = G3.gf3_deq_rows(u_sl, s_sl, out_dtype=torch.float32)
u_np = u[:, :c].numpy().astype(np.uint32).astype(np.uint64)
s_np = s16[:, :c].numpy().astype(np.float32)
w_ref = np.zeros((c, n), dtype=np.float32)
for row in range(c):
    for b in range(nb):
        for r in range(RG.RPB):
            rec = int(u_np[b * RG.RPB + r, row])
            for k in range(RG.ND):
                dd = rec % 3
                rec //= 3
                j = r * RG.ND + k
                if j < 128:
                    w_ref[row, b * 128 + j] = (dd - 1) * s_np[b, row]
dif = float(np.abs(wd.cpu().numpy() - w_ref).max())
print(f"deq-Kernel  : max|Δ| vs Python-Dekodierung = {dif:.6g}")
assert dif == 0.0, "deq-Kernel nicht exakt"

# Embedding-Gather: 5 zufaellige Token-IDs dekodieren. Referenz ist die
# Dekodier-Slice (c Zeilen) — IDs muessen darin liegen (% c, nicht % m)
ids = torch.tensor([0, 7, c - 1, 3, 100], dtype=torch.long, device="cuda") % c
emb = G3.GF3Embedding(u_gpu, s_gpu, m, n, unfold=lambda z: z)
z = emb(ids)
z_ref = w_ref[ids.cpu().numpy()]
dif = float(np.abs(z.cpu().numpy() - z_ref).max())
print(f"Embedding   : shape {tuple(z.shape)}  max|Δ| = {dif:.6g}")
assert (z.abs().max() > 0 or np.abs(z_ref).max() == 0), "Embedding leer (Fold-Identity)"
assert dif == 0.0, "Embedding-Gather nicht exakt"

print("OK: GF3-Runtime-Kernel exakt (Kernel-Decode, GEMM-Paritaet, Embedding-Gather)")