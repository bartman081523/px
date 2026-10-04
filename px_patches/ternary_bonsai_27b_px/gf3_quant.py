#!/usr/bin/env python
"""GF(3)/Triquint-Runtime (Stufe 3): matvec/Linear/Embedding/Loader.

Nimmt das GF3-Artefakt (gf3_model.safetensors, siehe
ququint_quant/repack_gf3.py: Q = sign(w), S = per-128-Block amax — verlustfrei
auf PTQ1_0-ternarem Material) und stellt dieselben Modulschnittstellen bereit,
die der ternary-Patch benutzt:

    GF3Linear(x)        wie PTQ10Linear: fold(x) -> matvec -> y.to(x.dtype)
    GF3Embedding(ids)   wie PTQ10Embedding: gather -> decode -> unfold

Layout (bewiesen: scratches/ququint/stufe3_packtest.py + Stufe-2-Bench
results_kernel_v0_2.json — GF3 max|Δ|=0,0000 END-TO-END DURCH den Kernel):
    u  int32 (RB, m)  uint32-Records bit-treu (3^20-1 > 2^31 -> Kernel:
                      .to(int64)-Sign-Extend + '& 0xFFFFFFFF')
    s  fp16  (nb, m)  block-major; Record rr liegt in Block rr // RPB
    Spalte je Digit:  col = b*128 + (rr % RPB)*ND + k; Pad-Digits (12/Block)
    sind Nullzustand (digit=1 -> q=0) und tragen nie bei.

Kernel-Familie:
    _gf3_mv  : Decode (M <= MMAX)   GF3-GEMV, f32-Partials, Lanes über
               Grid-Achse 2 (M=1 identisch zum Stufe-2-Bench: 0,240 ms
               gegen ptq10 0,972 ms bei max|Δ|=0,0000).
    _gf3_deq : Prefill (M > MMAX)   GF3 -> bf16-Dekodierung (row-gechunkt) +
               cuBLAS-GEMM (identische Numerik wie ptq10-_mv_dequant).

isfinite-Guard: gleiche Transienz-Rettung wie ptq10_matvec.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

GF3_DIR = os.environ.get(
    "GF3_ARTIFACT_DIR",
    "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf/gf3_model.safetensors")
ND = 20                                   # Digits je uint32-Record
RPB = 7                                   # Records je 128-Block
HALF = 1                                  # (p-1)/2 — Zustand {-1,0,+1}

# ------------------------------------------------------------ Matvec-Kernel ---
try:
    import triton
    import triton.language as tl
    _TRITON = triton is not None
except Exception:                                    # kein Triton -> GEMM-Pfad
    _TRITON = False

_MMAX = 8                                 # Kernel bis M=8 (ptq10-Parität)
_RC = 16                                  # Records je Programm (Bench-Wert)
_BLOCK_R = 128
_PART_BYTES = 96 * 2**20
_RESCUE_N = [0]
_RESCUE_PRINT_MAX = 10

# isfinite-Guard: ein Pro-Call-Sync (.item) serialisiert 400+ Matvecs/Token
# und drosselt Decode auf <1 tok/s. Modi:
#   "auto" (Default) — Logits-Kopf (head=True) prueft JEDE Ausgabe (der Sync
#                      faellt beim Sampling ohnehin an), alle uebrigen
#                      Kernel-Matvecs amortisiert alle _NANCHECK_EVERY Calls;
#   "1"              — Pro-Call-Check wie frueher (Tests/Debug);
#   "0"              — nie im Kernel-Pfad.
if os.environ.get("PX_GF3_NANCHECK", "auto") == "1":
    _NANCHECK_MODE = "percall"
elif os.environ.get("PX_GF3_NANCHECK", "auto") in ("0", "off"):
    _NANCHECK_MODE = "off"
else:
    _NANCHECK_MODE = "auto"
_NANCHECK_EVERY = 256
_CALL_N = [0]

if _TRITON:

    @triton.jit
    def _gf3_mv(u_ptr, s_ptr, x_ptr, part_ptr, M, RB, CB, XP,
                ND: tl.constexpr, RPB: tl.constexpr,
                BLOCK_R: tl.constexpr, RC: tl.constexpr):
        """part[mi, r, cb] = Σ_{rr in cb} s·Σ_k q_k·x[mi, rr*ND+k].

        M = m_out (Output-Zeilen), Grid-Achse 2 = Token-Lane (mi). Digits
        mul-frei (q ∈ {-1,0,+1}: Betrag nur über Select/Negation); uint32-
        Records liegen als int32-Bytes vor -> Sign-Extend + Maske.
        """
        pid_r = tl.program_id(0)
        cb = tl.program_id(1)
        mi = tl.program_id(2)
        r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < M
        acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
        c0 = cb * RC
        for i in range(RC):
            rr = c0 + i
            # Mask-Guard statt Scalar-If: Records >= RB werden von den
            # Loads ausgeschlossen (kein OOB); s=0 neutralisiert ihren
            # Beitrag (u=0 dekodiert qv=-1, aber s*a2 = 0)
            m2 = rmask & (rr < RB)
            u = tl.load(u_ptr + rr * M + r, mask=m2,
                        other=0).to(tl.int64) & 0xFFFFFFFF
            s = tl.load(s_ptr + (rr // RPB) * M + r, mask=m2,
                        other=0).to(tl.float32)
            a2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
            uu = u
            base = rr * ND
            for k in tl.static_range(ND):
                dv = uu % 3
                uu = uu // 3
                xc = tl.load(x_ptr + mi * XP + base + k)  # x vorgepadet
                qv = dv - 1
                a2 += tl.where(qv > 0, xc,
                               tl.where(qv < 0, -xc, 0.0))
            acc += s * a2
        tl.store(part_ptr + (mi * M + r) * CB + cb, acc, mask=rmask)

    @triton.jit
    def _gf3_deq(u_ptr, s_ptr, out_ptr, M, RB, N, NB,
                 ND: tl.constexpr, RPB: tl.constexpr, RPB_P2: tl.constexpr,
                 BLOCK_R: tl.constexpr, CBW: tl.constexpr):
        """out[r, b*128 + j] = digit(rin, j % ? −1)·s[b, r].

        r  = Output-Zeilen (M = c), b = 128-Block. pro Programm: BLOCK_R
        Zeilen × CBW Blöcke; Digits LSB-zuerst über %3 //3; Pad-Spalten
        (j >= 128) werden nie geschrieben.
        """
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = r < M
        for bb in tl.static_range(CBW):
            b = pid_c * CBW + bb
            # Mask-Guard statt Scalar-If: Blöcke >= NB werden von Loads und
            # Stores ausgeschlossen (kein OOB-Write in fremde Tensor-Spalten)
            m3 = rmask & (b < NB)
            # Record-Achse auf nächste 2er-Potenz (7 -> 8) aufgepadet;
            # recs >= RPB landen bei cn >= 140 > 128 und werden dort maskiert
            recs = tl.arange(0, RPB_P2)
            u = tl.load(u_ptr + (b * RPB + recs)[:, None] * M + r[None, :],
                        mask=m3[None, :] & (recs < RPB)[:, None], other=0
                        ).to(tl.int64) & 0xFFFFFFFF
            sb = tl.load(s_ptr + b * M + r, mask=m3,
                         other=0).to(tl.float32)
            uu = u
            for k in tl.static_range(ND):
                dv = uu % 3
                uu = uu // 3
                cn = recs * ND + k
                m2 = m3[None, :] & (cn < 128)[:, None]
                val = ((dv - 1).to(tl.float32) * sb[None, :]).to(
                    out_ptr.dtype.element_ty)
                tl.store(out_ptr + r[None, :] * N + b * 128 + cn[:, None],
                         val, mask=m2)


_XPAD = {}


def _x_pad_buffer(shape, device, lanes=_MMAX):
    """Padded-x-Buffer ((lanes, n_full) f32): je Block 128 echte + 12 0-Spalten.

    Reale Gewichtsspalte b*128+pos sitzt bei Record-Spalte b*140+pos; Pad-
    Digits sind Nullzustand und tragen unabhängig vom x-Wert nicht bei.
    Call-invariant -> einmal bauen, wiederverwenden.
    """
    key = (device, shape)
    if key not in _XPAD:
        nb = shape // 128
        xp = torch.zeros(lanes, nb * RPB * ND, dtype=torch.float32, device=device)
        _XPAD[key] = xp
    return _XPAD[key]


def _gf3_kernel_matvec(u, s, x2f, m_out, K):
    """Kernel-Pfad (M = x2f.shape[0] <= _MMAX); Rückgabe (M, m_out) f32."""
    M = x2f.shape[0]
    nb, rb = K // 128, (K // 128) * RPB
    cb = triton.cdiv(rb, _RC)
    xp = _x_pad_buffer(K, x2f.device)
    mch = 1 if M == 1 else max(1, min(M, _PART_BYTES // (m_out * cb * 4)))
    ys = []
    for m0 in range(0, M, mch):
        mm = min(mch, M - m0)
        xb = xp[:mm]
        # x in Record-Raster streuen: Block b belegt bei Record-Spalte
        # b*RPB*ND (= b*140), echte Spalte j bei b*140 + j. Strided-Setitem
        # in einem Op; Pad-Spalten (Digit 1 = Nullzustand) tragen nie bei —
        # kein Zeroing noetig.
        xb.view(mm, nb, RPB * ND)[:, :, :128] = x2f[m0:m0 + mm].view(mm, nb, 128)
        part = torch.empty(mm, m_out, cb, dtype=torch.float32, device=u.device)
        _gf3_mv[(triton.cdiv(m_out, _BLOCK_R), cb, mm)](
            u, s, xb, part, m_out, rb, cb, xb.stride(0),
            ND=ND, RPB=RPB, BLOCK_R=_BLOCK_R, RC=_RC)
        ys.append(part.sum(-1))
    return ys[0] if len(ys) == 1 else torch.cat(ys, 0)


def gf3_deq_rows(u_rows, s_rows, out_dtype=torch.bfloat16):
    """u (RB, c) int32 contiguous + s (nb, c) fp16 -> w (c, nb*128)."""
    rb, c = u_rows.shape
    nb = s_rows.shape[0]
    n = nb * 128
    out = torch.empty(c, n, dtype=out_dtype, device=u_rows.device)
    _gf3_deq[(triton.cdiv(c, 32), triton.cdiv(nb, 32))](
        u_rows, s_rows, out, c, rb, n, nb,
        ND=ND, RPB=RPB, RPB_P2=triton.next_power_of_2(RPB),
        BLOCK_R=32, CBW=32)
    return out


def _gf3_deq_matvec(u, s, x2, in_shape, m_out, K):
    """Prefill-Pfad: GF3 -> bf16 dekodieren (row-gechunkt) + GEMM (ptq10-Parität)."""
    out = u.shape[1]
    y = torch.empty(x2.shape[0], out, dtype=torch.bfloat16, device=x2.device)
    for lo in range(0, out, 8192):
        hi = min(lo + 8192, out)
        w = gf3_deq_rows(u[:, lo:hi].contiguous(), s[:, lo:hi])
        y[:, lo:hi] = F.linear(x2, w)
    return y.reshape(*in_shape[:-1], m_out)


def _nancheck_needed(head):
    """Amortisierter Check-Trigger (Modi siehe Block oben): "percall" immer,
    "off" nie, "auto": Logits-Kopf je Call (der Sync faellt beim Sampling
    ohnehin an), alle uebrigen Kernel-Matvecs alle _NANCHECK_EVERY Calls."""
    if _NANCHECK_MODE == "percall":
        return True
    if _NANCHECK_MODE == "off":
        return False
    if head:
        return True
    _CALL_N[0] += 1
    return _CALL_N[0] % _NANCHECK_EVERY == 0


def gf3_matvec_safe(u, s, x, m_out, K, head=False):
    """Interface-Analogon zu ptq10_matvec: Ausgabe x.dtype [..., m_out].

    isfinite-Guard amortisiert (Block oben): Kernel-Pfad checkt je Modus,
    Rettung wie immer Dequant+GEMM. Deq-Pfad (Prefill, M > MMAX): nur der
    Logits-Kopf checkt (kein zweiter Deq als Rettung vorhanden) — Best-
    Effort ueber nan_to_num; Transienten im Deq-Pfad (f32-Akkumulation
    ueber bf16-GEMM) sind praktisch ausgeschlossen.
    """
    in_shape = x.shape
    x2 = x.reshape(-1, in_shape[-1])
    M = x2.shape[0]
    if not _TRITON or M > _MMAX:
        y = _gf3_deq_matvec(u, s, x2, in_shape, m_out, K)
        if head and _NANCHECK_MODE != "off" and not bool(
                torch.isfinite(y).all().item()):
            _RESCUE_N[0] += 1
            if _RESCUE_N[0] <= _RESCUE_PRINT_MAX:
                print(f"[gf3] deq-Kopf-Rescue #{_RESCUE_N[0]}: nichtfinite "
                      f"Logits (out={m_out}) -> nan_to_num", flush=True)
            y = torch.nan_to_num(y, nan=0.0, posinf=3.0e38, neginf=-3.0e38)
        return y
    y = _gf3_kernel_matvec(u, s, x2.to(torch.float32), m_out, K)
    y = y.reshape(*in_shape[:-1], m_out).to(x.dtype)
    if _nancheck_needed(head) and not bool(torch.isfinite(y).all().item()):
        _RESCUE_N[0] += 1
        if _RESCUE_N[0] <= _RESCUE_PRINT_MAX:
            print(f"[gf3] matvec-Rescue #{_RESCUE_N[0]}: nichtfinite Ausgabe "
                  f"(out={m_out} K={K} head={head}) -> Dequant+GEMM",
                  flush=True)
        y = _gf3_deq_matvec(u, s, x2, in_shape, m_out, K)
    return y


# ---------------------------------------------------------------- Module ---
class GF3Linear(nn.Module):
    """Linear aus GF3-(u, s) — Interface-Parität zu PTQ10Linear.

    u/s als register_buffer int32/fp16 — model.to(device) (ohne dtype) castet
    Integer-Buffer nie; fp16-Skalare bleiben fp16 (Kernel dekodiert exakt).
    """

    def __init__(self, u, s, fold=None):
        super().__init__()
        K, m_out = s.shape[0] * 128, s.shape[1]
        self.in_features = K
        self.out_features = m_out
        self.fold = fold
        self.head = False        # Logits-Kopf -> je-Call-isfinite (Modus auto)
        self.register_buffer("u", u.contiguous())    # (RB, m_out) int32
        self.register_buffer("s", s.contiguous())    # (nb, m_out) fp16

    def forward(self, x):
        if self.fold is not None:
            x = self.fold(x)
        y = gf3_matvec_safe(self.u, self.s, x, self.out_features,
                            self.in_features, head=self.head)
        return y.to(x.dtype)


class GF3Embedding(nn.Module):
    """Latent-gepacktes Embedding: Gather -> Decode -> Unfold (PTQ10-Parität)."""

    def __init__(self, u, s, num_embeddings, embedding_dim, unfold):
        super().__init__()
        self.unfold = unfold
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("u", u.contiguous())    # (RB, V) int32
        self.register_buffer("s", s.contiguous())    # (nb, V) fp16

    def forward(self, input_ids):
        flat = input_ids.reshape(-1)
        u_g = self.u.index_select(1, flat)           # (RB, n_ids)
        s_g = self.s.index_select(1, flat)           # (nb, n_ids)
        z = gf3_deq_rows(u_g, s_g, out_dtype=torch.float32)
        z = z.reshape(*input_ids.shape, self.embedding_dim)
        return self.unfold(z)


# ---------------------------------------------------------------- Laden ---
def gf3_artifact_exists(dir_path=None):
    base = os.path.dirname(GF3_DIR)
    tgt = (dir_path.rstrip("/") + "/gf3_model.safetensors") if dir_path else GF3_DIR
    return os.path.exists(tgt), base


def build_and_load_gf3(fold, device="cuda", verbose=True, hf_dir=None):
    """Spiegel von runtime_qwen35_ptq.build_and_load auf das GF3-Artefakt.

    Schluessel = Originalschluessel + ".gq3"/".gs3" (Layer-Ternaries heissen
    "...weight", embed/lm_head "...ternary" — Konverter haengt gq3/gs3 an).
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from safetensors.torch import load_file

    assert hf_dir, "build_and_load_gf3 braucht hf_dir (Modell-Dir)"
    from transformers import AutoConfig as _AC
    cfg = _AC.from_pretrained(hf_dir)
    path = hf_dir + "/gf3_model.safetensors"
    if verbose:
        print(f"GF3 build: {path}")
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM._from_config(cfg)

    sd = load_file(path)
    dtype = torch.bfloat16

    def lin(key):
        return GF3Linear(sd[key + ".gq3"], sd[key + ".gs3"], fold)

    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        gdn = hasattr(layer, "linear_attn")
        lin_ = layer.linear_attn if gdn else layer.self_attn
        if gdn:
            for attr in ("in_proj_qkv", "in_proj_z", "out_proj"):
                setattr(lin_, attr, lin(p + f"linear_attn.{attr}.weight"))
        else:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(lin_, attr, lin(p + f"self_attn.{attr}.weight"))
        for attr in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, attr, lin(p + f"mlp.{attr}.weight"))
        for attr in ("input_layernorm", "post_attention_layernorm"):
            t = sd[p + attr + ".weight"].to(device, dtype)
            getattr(layer, attr).weight = nn.Parameter(t, requires_grad=False)
        if gdn:
            la = lin_
            la.norm.weight = nn.Parameter(
                sd[p + "linear_attn.norm.weight"].to(device, dtype),
                requires_grad=False)
            la.conv1d.weight = nn.Parameter(
                sd[p + "linear_attn.conv1d.weight"].to(device, dtype),
                requires_grad=False)
            la.A_log = nn.Parameter(sd[p + "linear_attn.A_log"].to(device),
                                    requires_grad=False)
            la.dt_bias = nn.Parameter(sd[p + "linear_attn.dt_bias"].to(device),
                                      requires_grad=False)
            la.in_proj_a.weight = nn.Parameter(
                sd[p + "linear_attn.in_proj_a.weight"].to(device, dtype),
                requires_grad=False)
            la.in_proj_b.weight = nn.Parameter(
                sd[p + "linear_attn.in_proj_b.weight"].to(device, dtype),
                requires_grad=False)
        else:
            for attr in ("q_norm", "k_norm"):
                t = sd[p + f"self_attn.{attr}.weight"].to(device, dtype)
                getattr(lin_, attr).weight = nn.Parameter(t, requires_grad=False)

    model.model.embed_tokens = GF3Embedding(
        sd["model.embed_tokens.ternary.gq3"], sd["model.embed_tokens.ternary.gs3"],
        cfg.vocab_size, cfg.hidden_size, fold.unfold)
    model.lm_head = GF3Linear(sd["lm_head.ternary.gq3"], sd["lm_head.ternary.gs3"],
                              fold)
    model.lm_head.head = True
    model.model.norm.weight = nn.Parameter(sd["model.norm.weight"].to(device, dtype),
                                           requires_grad=False)
    sd = None

    for name, mod in model.named_modules():
        if hasattr(mod, "inv_freq"):
            mod.__init__(model.config, device=torch.device(device))

    model.to(device)
    for n, par in model.named_parameters():
        assert not par.is_meta, f"Meta-Rest: {n}"
    for n, buf in model.named_buffers():
        assert not buf.is_meta, f"Meta-Rest (Buffer): {n}"
    model.eval()

    model._px_quant_format = "gf3"
    model._px_long_ctx = True
    if verbose:
        # GF3-Module tragen keine nn.Parameter (nur Buffer) -> Dicht-Aequi-
        # valent aus (in,out)-Produkten zaehlen, plus echter NN-Rest.
        dense = sum(m.out_features * m.in_features
                    for m in model.modules() if isinstance(m, GF3Linear))
        dense += sum(m.num_embeddings * m.embedding_dim
                     for m in model.modules() if isinstance(m, GF3Embedding))
        npar = dense + sum(p.numel() for p in model.parameters())
        vm = torch.cuda.max_memory_allocated() / 2**30
        print(f"GF3 geladen: {npar/1e9:.3f}B Parameter (dichte-Aequi., "
              f"davon GF3 {dense/1e9:.3f}B), VRAM-Spitze {vm:.2f} GiB")
    return model