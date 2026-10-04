#!/usr/bin/env python
"""PTQ1_0-Ternary-Runtime fuer Ternary-Bonsai-2-27B (HF-konvertiertes Format).

- torch-Dequant des gepackten ID143-Formats (C-Semantik zertifiziert gegen
  /tmp/ptq_verify.py: 28B = qs[0:24] + qh[24:26] + d(f16)@26:28).
- Hadamard-Aktivierungs-Fold: fold(x) = blockwise-normalisierter Sylvester-WHT
  (signs * x), block=1024, Zeichenvektor pro Eingangs-Achse
  (hidden 5120 / gdn-v 6144 / interm 17408). Gewichte sind im GGUF als
  W' = W (fold^-1)^T abgelegt -> y = fold(x) W'^T = x W^T exakt.
- token_embd ist LATENT (inverse-after-lookup): e = (z @ H_block) * s.
- Gewickelt in transformers-Qwen3_5ForCausalLM: nn.Linear ersetzt durch
  PTQ10Linear (gepackt) mit fuseriertem Triton-PTQ1_0-Matvec-Kernel
  (Decode M<=8: 0.28 s/token, 16x vs Dequant — D≡A bit-identisch gegen
  ds-Buffer-Orakel; Prefill M>8: Dequant+GEMM, M-unabhaengig),
  embed_tokens durch PTQ10Embedding (Gather + Unfold), lm_head ebenfalls
  gepackt (PTQ10Linear) — spart die 2.37 GiB bf16-Materialisierung
  (248320x5120x2) und liest im Decode nur noch 265 MB statt 2.37 GiB.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.environ.get("TERNARY_BONSAI_HF_DIR",
    "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf")
BLOCK = 1024

# ---------------------------------------------------------------- Dequant ---
_TPOW = {}


def _pow_tables(device):
    if device not in _TPOW:
        # Digit-Slot n <-> 3^n (wie Oracle POW3Q/POW3H, LSB-Rang zuerst)
        _TPOW[device] = (
            torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16, device=device),
            torch.tensor([1, 3, 9, 27], dtype=torch.int16, device=device),
        )
    return _TPOW[device]


def dequant_pack(pack):
    """pack: (..., Nb, 28) uint8 -> (..., Nb, 128) float32 (Werte {-d,0,+d})."""
    dev = pack.device
    p5, p4 = _pow_tables(dev)
    b = pack.to(torch.int16)
    qs, qh = b[..., 0:24], b[..., 24:26]
    d = pack[..., 26:28].contiguous().view(torch.float16).to(torch.float32)
    pre = tuple(pack.shape[:-1])
    t1 = ((qs[..., 0:16, None] * p5) & 0xFF) * 3 >> 8             # (..., Nb, 16, 5)
    t2 = ((qs[..., 16:24, None] * p5) & 0xFF) * 3 >> 8            # (..., Nb, 8, 5)
    t3 = ((qh[..., :, None] * p4) & 0xFF) * 3 >> 8                # (..., Nb, 2, 4)
    w = torch.empty(pre + (128,), dtype=torch.float32, device=dev)
    w[..., 0:80] = t1.transpose(-2, -1).reshape(pre + (80,)) - 1.0
    w[..., 80:120] = t2.transpose(-2, -1).reshape(pre + (40,)) - 1.0
    w[..., 120:128] = t3.transpose(-2, -1).reshape(pre + (8,)) - 1.0
    return w * d


# ------------------------------------------------------- Triton-Matvec ---
try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:                                  # kein Triton → Fallback
    _TRITON = False

if _TRITON:

    @triton.jit
    def _ptq10_mv(pk_ptr, x_ptr, part_ptr, out, nb, K, p5_ptr, p4_ptr,
                  BLOCK_R: tl.constexpr):
        """Ein Programm: RB out-Zeilen × 1 Record × 1 Token. Partials f32.

        Layout (zertifiziert, LSB-Rang zuerst): Record 28B = qs[0:24] |
        qh[24:26] | d[26:28] (fp16-LE). Trit-Positionen: t1 (qs 0..15,
        P5) → b + s*16; t2 (qs 16..23, P5) → 80 + b + s*8; t3 (qh 0..1,
        P4) → 120 + b + s*2. digit = ((byt·P)&0xFF)·3>>8 → trit = digit−1.
        P5 auf 8 Slots gepaddet (Pad → Trit 0 = null Contribution).
        """
        pid = tl.program_id(0)
        r = tl.program_id(1)
        m = tl.program_id(2)
        rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < out
        brow = rows.to(tl.int64) * (nb * 28) + r * 28
        p5 = tl.load(p5_ptr + tl.arange(0, 8))
        p4 = tl.load(p4_ptr + tl.arange(0, 4))
        s5v = tl.arange(0, 8) < 5
        e5 = tl.arange(0, 8)
        e4 = tl.arange(0, 4)

        # t1: qs[0:16] × P5 → w-pos b + s*16 (Trits 0..79)
        b16 = tl.load(pk_ptr + brow[:, None] + tl.arange(0, 16)[None, :],
                      mask=rmask[:, None], other=0).to(tl.int32)
        d1 = ((b16[:, :, None] * p5[None, None, :]) & 0xFF) * 3 >> 8
        d1 = tl.where(s5v[None, None, :], d1, 1)
        xc1 = tl.arange(0, 16)[:, None] + e5[None, :] * 16
        x1 = tl.load(x_ptr + m * K + r * 128 + xc1).to(tl.float32)
        p1 = tl.sum(tl.sum((d1 - 1).to(tl.float32) * x1[None, :, :], 2), 1)

        # t2: qs[16:24] × P5 → w-pos 80 + b + s*8 (Trits 80..119)
        b8 = tl.load(pk_ptr + brow[:, None] + 16 + tl.arange(0, 8)[None, :],
                     mask=rmask[:, None], other=0).to(tl.int32)
        d2 = ((b8[:, :, None] * p5[None, None, :]) & 0xFF) * 3 >> 8
        d2 = tl.where(s5v[None, None, :], d2, 1)
        xc2 = tl.arange(0, 8)[:, None] + e5[None, :] * 8
        x2 = tl.load(x_ptr + m * K + r * 128 + 80 + xc2).to(tl.float32)
        p2 = tl.sum(tl.sum((d2 - 1).to(tl.float32) * x2[None, :, :], 2), 1)

        # t3: qh[0:2] × P4 → w-pos 120 + b + s*2 (Trits 120..127)
        qh = tl.load(pk_ptr + brow[:, None] + 24 + tl.arange(0, 2)[None, :],
                     mask=rmask[:, None], other=0).to(tl.int32)
        d3 = ((qh[:, :, None] * p4[None, None, :]) & 0xFF) * 3 >> 8
        xc3 = tl.arange(0, 2)[:, None] + e4[None, :] * 2
        x3 = tl.load(x_ptr + m * K + r * 128 + 120 + xc3).to(tl.float32)
        p3 = tl.sum(tl.sum((d3 - 1).to(tl.float32) * x3[None, :, :], 2), 1)

        # d = fp16-Scale aus Bytes 26,27 (LE, Bitcast — Mini-Debug exakt)
        blo = tl.load(pk_ptr + brow + 26, mask=rmask, other=0).to(tl.int32)
        bhi = tl.load(pk_ptr + brow + 27, mask=rmask, other=0).to(tl.int32)
        u16 = (blo + (bhi << 8)).to(tl.int16)
        df = tl.cast(u16, tl.float16, bitcast=True).to(tl.float32)

        tl.store(part_ptr + m * out * nb + rows * nb + r,
                 (p1 + p2 + p3) * df, mask=rmask)

    _MV_POW = {}

    def _mv_pows(device):
        if device not in _MV_POW:
            _MV_POW[device] = (
                torch.tensor([1, 3, 9, 27, 81, 243, 729, 2187],
                             dtype=torch.int32, device=device),
                torch.tensor([1, 3, 9, 27], dtype=torch.int32, device=device))
        return _MV_POW[device]

_PART_BYTES = 96 * 2**20
_KERNEL_MMAX = 8            # decode/near-decode: Kernel; darueber: Dequant+GEMM
_RESCUE_PRINT_MAX = 10
_RESCUE_N = [0]


def _mv_dequant(packed, x2, in_shape):
    """Deterministischer Pfad: dequantisieren + GEMM (grosses M und Rescue).

    Row-gechunkt (8192 Ausgabereihen): haelt das f32-Dequant-Transient fest
    (lm_head 248320x40x128 waere sonst 4.75 GiB f32 gleichzeitig). Numerik
    identisch zum ungechunkten GEMM — dieselben Reihen, gleiche Skalen.
    """
    out = packed.shape[0]
    y = torch.empty(x2.shape[0], out, dtype=torch.bfloat16, device=x2.device)
    for lo in range(0, out, 8192):
        hi = min(lo + 8192, out)
        w = dequant_pack(packed[lo:hi]).reshape(hi - lo, -1).to(torch.bfloat16)
        y[:, lo:hi] = F.linear(x2, w)
    return y.reshape(*in_shape[:-1], out)


def ptq10_matvec(packed, x, BLOCK_R=32):
    """y = x @ W^T aus gepackten PTQ1_0-Bytes, ohne Gewichte-Materialisierung.

    Hybrid (gemessen): M<=_KERNEL_MMAX → Triton-Matvec (M=1: 16-20x schneller
    als Dequant, da kein bf16-W-Materialisieren); groesseres M (Prefill) →
    Dequant + GEMM (Dequant M-unabhaengig, GEMM bei cublas schneller als das
    M-fache W-Relesen des Kernels). Partials f32 [M,out,nb] auf ~96 MB
    begrenzt. Fallback ohne Triton: dequant_pack + F.linear.

    Guard (bewiesen per Catch-/Guard-Sonde): der Triton-Matvec liefert sehr
    selten eine komplett-nichtfinite Ausgabe bei finitem Input und sauberem
    Rerun (transiente Launch-Anomalie, Bursts nach Trajektorie). Solche
    Ausgaben werden hier einmalig deterministisch per Dequant+GEMM gerettet:
    gleiche Gewichte, gleicher (bereits gefalteter) Input → korrekt, nicht
    nur stabil. Gibt bf16 [.., out] zurueck.
    """
    in_shape = x.shape
    x2 = x.reshape(-1, in_shape[-1])
    M = x2.shape[0]
    if not _TRITON or M > _KERNEL_MMAX:
        return _mv_dequant(packed, x2, in_shape)
    dev = x.device
    p5, p4 = _mv_pows(dev)
    out, nb = packed.shape[0], packed.shape[1]
    x2f = x2.to(torch.float32).contiguous()
    mch = max(1, min(M, _PART_BYTES // (out * nb * 4)))
    ys = []
    for m0 in range(0, M, mch):
        xm = x2f[m0:m0 + mch]
        mm = xm.shape[0]
        part = torch.empty(mm, out, nb, dtype=torch.float32, device=dev)
        _ptq10_mv[(triton.cdiv(out, BLOCK_R), nb, mm)](
            packed, xm, part, out, nb, nb * 128, p5, p4, BLOCK_R=BLOCK_R)
        ys.append(part.sum(-1))
    y = ys[0] if len(ys) == 1 else torch.cat(ys, 0)
    y = y.reshape(*in_shape[:-1], out).to(torch.bfloat16)
    if not bool(torch.isfinite(y).all().item()):
        _RESCUE_N[0] += 1
        if _RESCUE_N[0] <= _RESCUE_PRINT_MAX:
            print(f"[ptq10] matvec-Rescue #{_RESCUE_N[0]}: nichtfinite Ausgabe "
                  f"(out={out} nb={nb}) -> Dequant+GEMM", flush=True)
        y = _mv_dequant(packed, x2.to(torch.bfloat16), in_shape)
    return y


# ---------------------------------------------------------------- Hadamard ---
def sylvester_hn(block=BLOCK, dtype=torch.float32, device="cuda"):
    """Normalisierte Sylvester-WHT (llama.cpp load_tensors: popcount-parity)."""
    r = torch.arange(block, device=device)
    p = (r[:, None] & r[None, :]).to(torch.int32)
    for s in (16, 8, 4, 2, 1):
        p = p ^ (p >> s)
    return torch.where((p & 1) != 0, -1.0, 1.0).to(dtype) / math.sqrt(block)


def _block_wh(x, hn):
    """Blockwise WHT entlang letzter Dim: (..., n) -> (..., n/1024, 1024) @ Hn."""
    shape = x.shape
    return x.view(*shape[:-1], shape[-1] // BLOCK, BLOCK) @ hn


class FoldOps:
    """Fold-Operationen fuer eine gewaehlte Kompositionsreihenfolge."""

    def __init__(self, signs, mode="signs_first", device="cuda"):
        self.hn = sylvester_hn(device=device)
        self.signs = {w: torch.as_tensor(v, dtype=torch.float32, device=device).view(1, 1, -1)
                      for w, v in signs.items()}
        self.mode = mode

    def fold(self, x):
        s = self.signs[x.shape[-1]]
        f32 = x.to(torch.float32)
        if self.mode == "signs_first":
            y = _block_wh(f32 * s, self.hn)
        else:
            y = _block_wh(f32, self.hn) * s
        return y.reshape(x.shape).to(x.dtype)

    def unfold(self, z, dtype=torch.bfloat16):
        """token_embd: inverse-after-lookup (zum Fold entgegengesetzte Ordnung)."""
        s = self.signs[z.shape[-1]]
        f32 = z.to(torch.float32)
        shape = z.shape
        if self.mode == "signs_first":
            y = _block_wh(f32, self.hn).reshape(shape) * s
        else:
            y = _block_wh(f32 * s, self.hn)
        return y.reshape(shape).to(dtype)

    __call__ = fold


# ---------------------------------------------------------------- Module ---
class PTQ10Linear(nn.Module):
    """Linearschicht aus gepackten ID143-Gewichten, Hybrid-Matvec (Triton/Dequant)."""

    def __init__(self, packed, fold=None, predequant=False):
        super().__init__()
        self.fold = fold
        self.out_features, nb = packed.shape[0], packed.shape[1]
        self.in_features = nb * 128
        if predequant:
            self.weight = nn.Parameter(dequant_pack(packed).to(torch.bfloat16),
                                       requires_grad=False)
            self.packed = None
        else:
            self.register_buffer("packed", packed.to(torch.uint8))

    def forward(self, x):
        if self.packed is not None:
            if self.fold is not None:
                x = self.fold(x)
            y = ptq10_matvec(self.packed, x)      # Hybrid: Triton-Kernel (decode) / Dequant+GEMM (prefill)
            return y.to(x.dtype)
        x = self.fold(x) if self.fold is not None else x
        return F.linear(x, self.weight)


class PTQ10Embedding(nn.Module):
    """Embedding aus latent gepacktem token_embd: Gather + Unfold pro Aufruf."""

    def __init__(self, packed, unfold):
        super().__init__()
        self.register_buffer("packed", packed.to(torch.uint8))          # (V, Nb, 28)
        self.unfold = unfold
        self.num_embeddings = packed.shape[0]
        self.embedding_dim = packed.shape[1] * 128

    def forward(self, input_ids):
        z = dequant_pack(self.packed[input_ids])
        z = z.reshape(*input_ids.shape, self.embedding_dim)             # (..., 5120)
        return self.unfold(z)


# ---------------------------------------------------------------- Laden ---
def build_and_load(fold, device="cuda", predequant=False, verbose=True, hf_dir=None):
    out = hf_dir or OUT_DIR
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from safetensors.torch import load_file

    cfg = AutoConfig.from_pretrained(out)          # -> Qwen3_5TextConfig
    sd = load_file(f"{out}/model.safetensors")
    dtype = torch.bfloat16                             # Normen/small tensors -> Modelldtype
    if verbose:
        print(f"Config: H={cfg.hidden_size} L={cfg.num_hidden_layers} "
              f"tie={cfg.tie_word_embeddings} rope={cfg.rope_parameters.get('rope_type')}")

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM._from_config(cfg)

    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        gdn = hasattr(layer, "linear_attn")
        lin = layer.linear_attn if gdn else layer.self_attn
        if gdn:
            for attr in ("in_proj_qkv", "in_proj_z", "out_proj"):
                setattr(lin, attr, PTQ10Linear(sd[p + f"linear_attn.{attr}.weight"], fold))
        else:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(lin, attr, PTQ10Linear(sd[p + f"self_attn.{attr}.weight"], fold))
        for attr in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, attr, PTQ10Linear(sd[p + f"mlp.{attr}.weight"], fold))
        for attr in ("input_layernorm", "post_attention_layernorm"):
            t = sd[p + attr + ".weight"].to(device, dtype)
            getattr(layer, attr).weight = nn.Parameter(t, requires_grad=False)
        if gdn:
            la = lin
            la.norm.weight = nn.Parameter(sd[p + "linear_attn.norm.weight"].to(device, dtype),
                                          requires_grad=False)
            la.conv1d.weight = nn.Parameter(sd[p + "linear_attn.conv1d.weight"].to(device, dtype),
                                            requires_grad=False)
            la.A_log = nn.Parameter(sd[p + "linear_attn.A_log"].to(device), requires_grad=False)
            la.dt_bias = nn.Parameter(sd[p + "linear_attn.dt_bias"].to(device), requires_grad=False)
            la.in_proj_a.weight = nn.Parameter(sd[p + "linear_attn.in_proj_a.weight"].to(device, dtype),
                                               requires_grad=False)
            la.in_proj_b.weight = nn.Parameter(sd[p + "linear_attn.in_proj_b.weight"].to(device, dtype),
                                               requires_grad=False)
        else:
            for attr in ("q_norm", "k_norm"):
                t = sd[p + f"self_attn.{attr}.weight"].to(device, dtype)
                getattr(lin, attr).weight = nn.Parameter(t, requires_grad=False)

    model.model.embed_tokens = PTQ10Embedding(sd["model.embed_tokens.ternary"], fold.unfold)
    model.lm_head = PTQ10Linear(sd["lm_head.ternary"], fold)
    model.model.norm.weight = nn.Parameter(sd["model.norm.weight"].to(device, dtype),
                                           requires_grad=False)
    sd = None                                                            # RAM freigeben

    # Rotary-Buffer neu erzeugen VOR .to() (Meta-Init vermuellt inv_freq)
    for name, mod in model.named_modules():
        if hasattr(mod, "inv_freq"):
            mod.__init__(model.config, device=torch.device(device))

    model.to(device)
    for n, par in model.named_parameters():
        assert not par.is_meta, f"Meta-Rest: {n}"
    for n, buf in model.named_buffers():
        assert not buf.is_meta, f"Meta-Rest (Buffer): {n}"

    model.eval()
    if verbose:
        npar = sum(p.numel() for p in model.parameters())
        vm = torch.cuda.max_memory_allocated() / 2**30
        print(f"Model geladen: {npar/1e9:.3f}B Parameter, VRAM-Spitze {vm:.2f} GiB")
    return model


# ---------------------------------------------------------------- Smoke ---
def load_signs():
    with open(f"{OUT_DIR}/config.json") as fh:
        c = json.load(fh)["prism_hadamard"]
    widths = [int(x) for x in c["sign_widths"]]
    vals = [int(x) for x in c["sign_values"]]
    signs, off = {}, 0
    for w in widths:
        signs[w] = torch.tensor(vals[off:off + w], dtype=torch.float32)
        off += w
    assert off == len(vals) == sum(widths)
    return signs, (c["version"], c.get("gdn_v_grouped"), c["folded_weight_names"][-1:])


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "signs_first"
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    signs, meta = load_signs()
    print("Signs:", list(signs), "meta:", meta)
    fold = FoldOps(signs, mode=variant)
    model = build_and_load(fold)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(OUT_DIR, fix_mistral_regex=True)
    msgs = [{"role": "user", "content": "Say hello and name one color."}]
    prompt_txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt_txt, return_tensors="pt").input_ids
    ids = ids.to(model.device)
    print("Prompt-Tokens:", ids.shape)
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=ntok, do_sample=False)
    dt = time.time() - t0
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    n = out.shape[1] - ids.shape[1]
    print(f"[{variant}] {ntok} tokens in {dt:.1f}s ({dt/max(n,1):.2f}s/token)")
    print(">>>", repr(txt))


if __name__ == "__main__":
    main()