#!/usr/bin/env python
"""GF(3)-Safetensors -> PTQ1_0-GGUF (Ternary-Bonsai-2-27B, Typ 143).

Rekonstruiert das GGUF byte-identisch aus dem GF3-Artefakt
(Q = sign(w), S = amax|w| fp16-exakt) plus dem offiziellen 11.120.992-Byte-Header.
Die Bijektion GF3<->PTQ1_0 ist exakt:

  GF3 digit (record-major: 20 Digits/Record LSB-first, 7 Records pro 128-Block,
  12 Pad-Digits j = 128..139 verworfen) == PTQ1_0 digit (trit + 1).
  Scale d (fp16, amax|w|) == gs3 (fp16) bitweise.
  Encode: MSB-first 5-Trit-Byte  v = sum_s digit(b + 16*s) * 3^(4-s),
      byt = (v*256 + 242) // 243
      e < 80:      byte b = e & 15,        slot s = e >> 4
      e < 120:     byte b = 16 + ((e-80) & 7),  s = (e-80) >> 3
      e < 128:     qh-Byte b = e & 1,      slot s = (e-120) >> 1,
          Gewichte [81,27,9,3] — das q *= 3 des C-Encoders (ggml-quants.c)
          verschiebt den ersten Trit auf 3^4, LSB-Trit implizit 0.
  Decode-Fold: digit_s = ((byt * 3^s) & 0xFF) * 3 >> 8.

Gates:
  G-P1    243-Wert-Pack/Fold-Roundtrip (self-test) + je Tensor digit/scale-
          Bitidentitaet gg. GF3 ueber den Mirror-Codec (max|delta| = 0).
  G-P1.5  GElOEST 2026-10-15 (gdn_formula.py, max|d| = 0 fuer alle Formeln):
          der prism-GGUF ist mit dem GF3-Artefakt wertidentisch — die 662 M
          Differenz-Bytes waren KEINE eigene Quantisierung, sondern:
          (1) GDN-Layout-Permutation (GF3/PX = Head-major, je Head 3 Rollen;
              prism = roll-major Streifen): Einheit i <- Einheit
              tbl[i] = (i // 16) + 3 * (i % 16) — auf 128-Block-Einheiten bei
              attn_gate (48 Bloecke) / attn_qkv-Tail (80 Bloecke, 32 ident) /
              ssm_conv1d-Kanaelen; auf Zeilen/Elementen bei ssm_alpha,
              ssm_beta, ssm_dt.bias (48 Einheiten). Layer-invariant, bit-exakt.
          (2) ssm_a: official = -exp(A_log)[tbl] — f32-Exp-ULPs ~3e-8 bleiben.
          (3) RMS-Norm-Gains: official = GF3-Offset + 1.0 (attn_norm,
              post_attention_norm, attn_q_norm, attn_k_norm, output_norm;
              ssm_norm byte-identisch, kein Offset).
          Der Konverter wendet alle drei an. G-P1 bleibt GF3-lossless: die
          Permutation ist werttreu, Exp/+1 sind dokumentierte Adaptionen.
          Erwartung nach Fix: byte-identisch ausser ssm_a-ULPs (~9 KB).

    python convert_gf3_to_gguf.py [--selftest] [--no-verify] [--full-sha256]
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/llama.cpp/gguf-py")

GGUF_DIR = "/home/julian/models/bonsai"
GF3 = ("~/.cache/huggingface/hub/models--neuralworm--ternary-bonsai-2-27b-hf/"
       "snapshots/ac4233470a41c6866a2b34c45f5ea28a1786bf24/gf3_model.safetensors")
HEADER = ("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/"
          "ptq10_gguf/official_PTQ1_0_header.partial")
OUT = f"{GGUF_DIR}/Ternary-Bonsai-2-27B-PTQ1_0-recon.gguf"
OFFICIAL = f"{GGUF_DIR}/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
EXPECT_OFFICIAL = 5_946_648_928

DATA_START = 11_120_992          # offset des ersten Tensors (output.weight)

# --- GF3-Layout -------------------------------------------------------------
ND, RPB = 20, 7                  # Digits/Record, Records pro 128-Block (12 Pad)
P5U8 = np.array([81, 27, 9, 3, 1], dtype=np.uint8)   # slot s -> 3^(4-s)
P4U8 = np.array([81, 27, 9, 3], dtype=np.uint8)      # qh (LSB-Trit implizit 0)
ROW_CHUNK = 4096
TT_PTQ1_0 = 143

# --- GGUF-Suffix -> (GF3-Template, kind); Template-Präfix: model.layers.N. ---
SUFFIX_RULES = {
    "attn_norm.weight": ("input_layernorm.weight", "f32"),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", "f32"),
    "ffn_gate.weight": ("mlp.gate_proj.weight", "tern"),
    "ffn_up.weight": ("mlp.up_proj.weight", "tern"),
    "ffn_down.weight": ("mlp.down_proj.weight", "tern"),
    "attn_q.weight": ("self_attn.q_proj.weight", "tern"),
    "attn_k.weight": ("self_attn.k_proj.weight", "tern"),
    "attn_v.weight": ("self_attn.v_proj.weight", "tern"),
    "attn_output.weight": ("self_attn.o_proj.weight", "tern"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "f32"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "f32"),
    "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", "tern"),
    "attn_gate.weight": ("linear_attn.in_proj_z.weight", "tern"),
    "ssm_out.weight": ("linear_attn.out_proj.weight", "tern"),
    "ssm_a": ("linear_attn.A_log", "f32"),
    "ssm_a.weight": ("linear_attn.A_log", "f32"),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "f32"),
    "ssm_dt.bias": ("linear_attn.dt_bias", "f32"),
    "ssm_norm.weight": ("linear_attn.norm.weight", "f32"),
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "bf16"),
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "bf16"),
}
NONLAYER = {
    "output.weight": ("lm_head.ternary", "tern"),
    "token_embd.weight": ("model.embed_tokens.ternary", "tern"),
    "output_norm.weight": ("model.norm.weight", "f32"),
}

# --- GDN-Layout-Adaption (2026-10-15, bewiesen bit-exakt in gdn_formula.py) --
# GF3/PX = Head-major (Head h: Rollen r0, r1, r2 in Interleave),
# prism-GGUF = roll-major (3 Streifen à 16 Einheiten). 48 GDN-Layer.
TBL_GDN = np.array([(i // 16) + 3 * (i % 16) for i in range(48)],
                   dtype=np.int64)
SUFFIX_TRANSFORMS = {
    "attn_gate.weight": "tern_perm",          # 48 128-Block-Einheiten
    "attn_qkv.weight": "tern_perm",           # 80 Bloecke: 32 ident + TBL
    "ssm_conv1d.weight": "conv_perm",         # Kanaele, wie qkv (80 Bloecke)
    "ssm_alpha.weight": "rows_48",            # bf16, 48 Zeilen
    "ssm_beta.weight": "rows_48",
    "ssm_dt.bias": "elem_48",                 # f32, 48 Elemente
    "ssm_a": "ssm_a",                         # -exp(A_log) + TBL
    "ssm_a.weight": "ssm_a",
    "attn_norm.weight": "plus1",              # Gain y = Offset + 1.0
    "post_attention_norm.weight": "plus1",
    "attn_q_norm.weight": "plus1",
    "attn_k_norm.weight": "plus1",
    "output_norm.weight": "plus1",
}


def suffix_of(name):
    m = re.match(r"blk\.\d+\.(.+)$", name)
    return m.group(1) if m else name


def gdn_row_src(m):
    """Zeilen-Quellindex src: offizielle Zeile i liest GF3-Zeile src[i].

    128-Zeilen-Bloecke: qkv/conv1d 80 Bloecke (32 ident, dann TBL auf dem
    Tail), gate 48 Bloecke (durchgehend TBL). Werttreu (nur Reihenfolge).
    """
    nblk = m // 128
    assert m % 128 == 0, f"gdn_row_src: {m} nicht 128-teilig"
    blk = np.arange(nblk, dtype=np.int64)
    if nblk == 80:
        blk[32:80] = 32 + TBL_GDN
    elif nblk == 48:
        blk[:] = TBL_GDN
    else:
        raise AssertionError(f"gdn_row_src: unerwartete Blockzahl {nblk}")
    return np.repeat(blk, 128) * 128 + np.tile(np.arange(128), nblk)


def resolve(gguf_name):
    """-> (gf3_key, kind) mit gf3_key = voller GF3-Key der Basis."""
    if gguf_name in NONLAYER:
        return NONLAYER[gguf_name]
    m = re.match(r"blk\.(\d+)\.(.+)$", gguf_name)
    assert m, f"GGUF-Name ohne Regel: {gguf_name}"
    tpl, kind = SUFFIX_RULES[m.group(2)]
    return (f"model.layers.{m.group(1)}.{tpl}", kind)


# --------------------------------------------------------------- GF3 decode -
def gf3_digits(gq3):
    """(RB, c) int32-Bitpattern -> (c, nb, 128) Digits 0..2 (uint8).

    Record r des Blocks b = Zeile b*7 + r; j = r*20 + k; j >= 128 = Pad.
    """
    rec = gq3.view(np.uint32)                              # (RB, c)
    digs = np.empty(rec.shape + (ND,), dtype=np.uint8)
    tmp = rec.copy()
    for k in range(ND):
        digs[:, :, k] = (tmp % 3).astype(np.uint8)
        tmp //= 3
    c = digs.shape[1]
    nb = digs.shape[0] // RPB
    d = digs.transpose(1, 0, 2).reshape(c, nb, RPB * ND)
    return d[:, :, :128].copy()


def gf3_weights(gq3, gs3):
    """(RB, c) + (nb, c) f16 -> (c, nb, 128) float32 == GF3-Unpack-Oracle."""
    q = gf3_digits(gq3)
    s = gs3.astype(np.float32).T                           # (c, nb)
    return (q.astype(np.float32) - 1.0) * s[:, :, None]


# --------------------------------------------------------------- PTQ1_0 enc -
def ceil_pack(v):
    """uint8-Byte: (v*256 + 242) // 243, v 0..242 (uint8)."""
    return ((v.astype(np.uint32) * 256 + 242) // 243).astype(np.uint8)


def encode_blocks(q, gs3_chunk):
    """(c, nb, 128) uint8-Digits + (nb, c) f16-Scales -> (c, nb, 28) uint8."""
    c, nb, _ = q.shape
    out = np.empty((c, nb, 28), dtype=np.uint8)
    # e < 80:  e = s*16 + b  -> reshape (s, b), transpose (b, s)
    q80 = q[:, :, 0:80].reshape(c, nb, 5, 16).transpose(0, 1, 3, 2)
    v80 = (q80 * P5U8).sum(-1, dtype=np.uint8)             # 0..242, kein Overflow
    out[:, :, 0:16] = ceil_pack(v80)
    # 80 <= e < 120:  e - 80 = s*8 + b
    q40 = q[:, :, 80:120].reshape(c, nb, 5, 8).transpose(0, 1, 3, 2)
    v40 = (q40 * P5U8).sum(-1, dtype=np.uint8)
    out[:, :, 16:24] = ceil_pack(v40)
    # 120 <= e < 128:  e - 120 = s*2 + b, LSB-Trit entfaellt (q *= 3 im C)
    qh = q[:, :, 120:128].reshape(c, nb, 4, 2).transpose(0, 1, 3, 2)
    vh = (qh * P4U8).sum(-1, dtype=np.uint8)
    out[:, :, 24:26] = ceil_pack(vh)
    scale = gs3_chunk.view(np.uint8).reshape(nb, c, 2).transpose(1, 0, 2)
    out[:, :, 26:28] = scale
    return out


def decode_fold(byt, n):
    """ggml-Codec: q = (byt*3^n) & 0xFF; digit = (q*3) >> 8."""
    q = (byt.astype(np.uint16) * np.uint16(3 ** n)) & np.uint16(0xFF)
    return ((q * np.uint16(3)) >> np.uint16(8)).astype(np.uint8)


def decode_mirror(blk):
    """EXAKTER numpy-Mirror von dequantize_row_ptq1_0 (ggml-quants.c).

    blk: (N, 28) uint8 -> (N, 128) float32.
    """
    d = blk[:, 26:28].copy().view(np.float16).reshape(-1, 1).astype(np.float32)
    t1 = decode_fold(blk[:, 0:16, None], np.arange(5))     # (N, 16, 5) [b, s]
    w80 = t1.transpose(0, 2, 1).reshape(-1, 80)            # e = s*16 + b
    t2 = decode_fold(blk[:, 16:24, None], np.arange(5))    # (N, 8, 5)
    w40 = t2.transpose(0, 2, 1).reshape(-1, 40)            # e = s*8 + b
    t3 = decode_fold(blk[:, 24:26, None], np.arange(4))    # (N, 2, 4)
    wh = t3.transpose(0, 2, 1).reshape(-1, 8)              # e = s*2 + b
    w = np.concatenate([w80, w40, wh], axis=1).astype(np.float32) - 1.0
    return w * d


def self_test_pack():
    """G-P1-Kern: alle 243 Werte MSB-first-Pack <-> Fold-Decode-Roundtrip."""
    v = np.arange(243, dtype=np.uint32)
    # MSB-first digits: digit_s = (v // 3^(4-s)) % 3
    dig = ((v[:, None] // (3 ** np.arange(4, -1, -1, dtype=np.uint32))[None, :])
           % 3).astype(np.uint8)                            # (243, 5)
    byt = ceil_pack(np.arange(243, dtype=np.uint8))
    got = np.stack([decode_fold(byt, n) for n in range(5)], axis=1)
    assert np.array_equal(got, dig), (
        "Pack/Fold verletzt: "
        + str(np.nonzero((got != dig).any(1))[0][:10].tolist()))
    # qh-Variante: 4 Trits, Werte immer Vielfache von 3
    rng = np.random.default_rng(0)
    d4 = rng.integers(0, 3, size=(4096, 4)).astype(np.uint8)
    vh = (d4 * P4U8).sum(-1, dtype=np.uint8)
    bh = ceil_pack(vh)
    for n in range(4):
        assert np.array_equal(decode_fold(bh, n), d4[:, n]), f"qh digit {n}"
    print("  pack/fold self-test OK (243 Werte + qh 4096 Zufallsblocks)")


# -------------------------------------------------------------- safetensors -
_HDRL = {}


def st_hdr_len(path):
    p = os.path.expanduser(path)
    if p not in _HDRL:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        _HDRL[p] = n
    return _HDRL[p]


def st_header(path):
    with open(os.path.expanduser(path), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def st_memmap(path, val, dt):
    base = 8 + st_hdr_len(path) + val["data_offsets"][0]
    return np.memmap(os.path.expanduser(path), dtype=dt, mode="r",
                     offset=base, shape=tuple(val["shape"]))


def st_bytes(path, val):
    n = val["data_offsets"][1] - val["data_offsets"][0]
    with open(os.path.expanduser(path), "rb") as f:
        f.seek(8 + st_hdr_len(path) + val["data_offsets"][0])
        buf = f.read(n)
    assert len(buf) == n
    return buf


# ------------------------------------------------------------------- GGUF ---
def read_official_tensors(path):
    """[(name, ttype, ne-Liste, offset, n_bytes)] via gguf-py (Clone)."""
    from gguf import GGUFReader
    r = GGUFReader(os.path.expanduser(path))
    out = []
    for t in r.tensors:
        out.append((t.name, int(t.tensor_type),
                    [int(s) for s in t.shape],
                    int(t.data_offset), int(t.n_bytes)))
    return out


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ main ----
def convert(args):
    t0 = time.perf_counter()
    self_test_pack()
    gf3h = st_header(args.gf3)
    meta = gf3h.pop("__metadata__", {})
    official = read_official_tensors(args.official)
    total_off = max(off + nb for (_, _, _, off, nb) in official)
    hdr_full = open(os.path.expanduser(args.official), "rb").read(DATA_START)
    assert len(hdr_full) == DATA_START
    out_path = os.path.expanduser(args.out)
    with open(out_path, "wb") as f0:
        f0.truncate(total_off)                       # sparse Null-Prefill
    fout = open(out_path, "r+b")
    fout.write(hdr_full)
    n_tern_ok, n_small_ok = 0, 0
    cursor = DATA_START
    for idx, (name, ttype, ne, off, n_bytes) in enumerate(official):
        assert off >= cursor, f"{name}: offset {off} < cursor {cursor}"
        cursor = off + n_bytes
        fout.seek(off)
        basis, kind = resolve(name)
        if kind == "tern":
            assert ttype == TT_PTQ1_0, (name, ttype)
            m = ne[1]
            nb = ne[0] // 128
            xf = SUFFIX_TRANSFORMS.get(suffix_of(name))
            src_rows = gdn_row_src(m) if xf == "tern_perm" else None
            if src_rows is not None:
                print(f"  {name}: GDN-Layout-Perm (128-Block-TBL)")
            gq3v = gf3h[basis + ".gq3"]
            gs3v = gf3h[basis + ".gs3"]
            RBv, mv = gq3v["shape"]
            assert mv == m and RBv == nb * RPB, (name, gq3v["shape"], ne)
            assert gs3v["shape"] == [nb, m], (name, gs3v["shape"], (nb, m))
            gq3_mm = st_memmap(args.gf3, gq3v, "<i4")
            gs3_mm = st_memmap(args.gf3, gs3v, "<f2")
            chunk = 128 if src_rows is not None else ROW_CHUNK
            for lo in range(0, m, chunk):
                hi = min(lo + chunk, m)
                c = hi - lo
                if src_rows is None:
                    gq3 = np.array(gq3_mm[:, lo:hi])    # (RB, c)
                    gs3 = np.array(gs3_mm[:, lo:hi])    # (nb, c)
                else:
                    rows = src_rows[lo:hi]              # offizielle Reihenfolge
                    gq3 = np.ascontiguousarray(gq3_mm[:, rows])
                    gs3 = np.ascontiguousarray(gs3_mm[:, rows])
                q = gf3_digits(gq3)                  # (c, nb, 128) uint8
                blk = encode_blocks(q, gs3)          # (c, nb, 28)
                if args.verify:
                    back = decode_mirror(blk.reshape(c * nb, 28))
                    w_gf3 = gf3_weights(gq3, gs3)
                    assert np.array_equal(
                        back.reshape(c, nb, 128), w_gf3), name
                fout.write(blk.reshape(c * nb * 28))
            n_tern_ok += 1
        else:
            assert ttype in (0, 30), (name, ttype)
            val = gf3h.get(basis)
            assert val is not None, basis
            assert val["dtype"] == "F32", (name, val["dtype"])
            n_el = int(np.prod(ne))
            buf = st_bytes(args.gf3, val)
            assert len(buf) == n_el * 4, (name, len(buf), n_el * 4)
            xf = SUFFIX_TRANSFORMS.get(suffix_of(name))
            if xf == "rows_48":                      # bf16-Zeilenperm (48)
                u32 = np.frombuffer(buf, dtype="<u4")
                assert int((u32 & np.uint32(0xFFFF)).max()) == 0, (
                    f"{basis}: F32 nicht bf16-exakt")
                out = (u32 >> np.uint32(16)).astype("<u2") \
                    .reshape(-1, ne[0])[TBL_GDN]
                fout.write(np.ascontiguousarray(out).tobytes())
            elif xf == "conv_perm":                  # Kanael-Perm (80 Bloecke)
                m2 = np.frombuffer(buf, dtype="<f4").reshape(ne[1], ne[0])
                fout.write(np.ascontiguousarray(
                    m2[gdn_row_src(ne[1])]).tobytes())
            elif xf == "elem_48":                    # dt.bias-Elemperm (48)
                v = np.frombuffer(buf, dtype="<f4")
                fout.write(v[TBL_GDN].tobytes())
            elif xf == "ssm_a":                      # A = -exp(A_log)[TBL]
                v = np.frombuffer(buf, dtype="<f4")
                fout.write((-np.exp(v[TBL_GDN])).tobytes())
            elif xf == "plus1":                      # Gain = Offset + 1.0
                v = np.frombuffer(buf, dtype="<f4")
                fout.write((v + np.float32(1.0)).tobytes())
            elif kind == "bf16":
                u32 = np.frombuffer(buf, dtype="<u4")
                assert int((u32 & np.uint32(0xFFFF)).max()) == 0, (
                    f"{basis}: F32 nicht bf16-exakt")
                fout.write((u32 >> np.uint32(16)).astype("<u2").tobytes())
            else:
                fout.write(buf)
            n_small_ok += 1
        if idx % 50 == 49:
            print(f"  … {idx + 1}/851 Tensoren "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    fout.close()
    used_tern = sum(1 for (_, t, _, _, _) in official if t == TT_PTQ1_0)
    assert n_tern_ok == used_tern == int(meta["n_ternary_tensors"]), (
        n_tern_ok, used_tern, meta["n_ternary_tensors"])
    assert n_small_ok == int(meta["n_small_tensors"]), (
        n_small_ok, meta["n_small_tensors"])
    # Bijektion: keine GF3-Referenz unkonsumiert
    seen = set()
    for (name, _, _, _, _) in official:
        basis, kind = resolve(name)
        if kind == "tern":
            seen.add(basis + ".gq3")
            seen.add(basis + ".gs3")
        else:
            seen.add(basis)
    missing = set(gf3h) - seen
    assert not missing, f"GF3-Keys ohne GGUF-Ziel ({len(missing)}): " \
        f"{sorted(missing)[:5]}"
    print(f"  Bijektion OK: {used_tern} ternary-Paare + {n_small_ok} kleine "
          f"Tensoren ({len(gf3h)} GF3-Keys komplett)")

    # --- G-P1.5: Vollvergleich gg. offiziellem GGUF -------------------------
    # GElOEST 2026-10-15: prism ist wertidentisch mit GF3 (keine eigene
    # Quantisierung). Nach den drei Adaptionen (GDN-TBL-Perm, ssm_a-Exp,
    # Norm-+1) ist Byte-Identitaet die ERWARTUNG — Residuum nur ssm_a-Exp-ULPs
    # (48 Tensoren, ~9 KB, semantisch exakt auf 3e-8). Sha-Abweichung wird
    # per per-Tensor-Diff (diff_recon_official.py) lokalisiert.
    if args.full_sha256:
        off_path = os.path.expanduser(args.official)
        dl = os.path.getsize(off_path)
        h1, h2 = sha_file(out_path), sha_file(off_path)
        ident = h1 == h2
        print(f"  G-P1.5 (beide {dl} B): recon={h1[:16]}… "
              f"official={h2[:16]}… identisch = {ident}")
        if not ident:
            print("  HINWEIS: Rest-Differenzen lokalisiert pruefen "
                  "(Erwartung: nur ssm_a-Exp-ULPs)")
    dt = time.perf_counter() - t0
    print(f"OK {out_path} ({n_tern_ok} ternary + {n_small_ok} klein, {dt:.0f}s)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gf3", default=GF3)
    ap.add_argument("--header", default=HEADER)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--official", default=OFFICIAL)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--full-sha256", action="store_true")
    args = ap.parse_args()
    args.verify = not args.no_verify
    if args.selftest:
        self_test_pack()
        return
    rc = convert(args)
    if args.full_sha256 and rc == 0:
        print("SHA256", sha_file(os.path.expanduser(args.out)))
    sys.exit(rc)


if __name__ == "__main__":
    main()