#!/usr/bin/env python
"""GF(5)-Quantisierung für gemma-4-E2B — Ende-zu-Ende (Pack + Lauf).

Aufbau
------
1. `pack`  quantisiert jeden 2-D-Gewichtstensor der **Sprachpfade** blockweise
           auf GF(5) {−2,−1,0,1,2} (per-Block-fp16-Skalar, 13 Ququints/u32)
           und legt das Ergebnis als safetensors im HF-Cache ab.
           Nicht quantisiert (bitgenau bf16): Vision-/Audio-Tower (für Text
           ungenutzt, Multimodalität bleibt unangetastet), Norms, Skalare und
           `embed_tokens` (geteilt mit dem lm_head → wird als volle Matrix
           gebraucht).
2. `eval`  lädt die GF5-Variante, ersetzt Linear- und Embedding-Module durch
           gepackte Pendants und generiert Text.

Speicher-/Rechenmodell (ehrlich)
--------------------------------
Gepackt ist ein **Speicherformat**. Der Forward dequantisiert pro Aufruf:
- `GF5Linear` dequantisiert seine Matrix einmal pro Forward-Pass (bei
  Prefill amortisiert, bei Decode pro Token).
- `GF5Embedding` dequantisiert nur die tatsächlich nachgeschlagenen Zeilen
  (Lookup-Tabellen brauchen nie die ganze Matrix).
Der fused GF(5)-GEMV (`gf5_gemv.py`, gemessen 2,37× auf echtem Tensor) ist
hier **nicht** eingebunden: sein Record-Layout (10 Records je 128er-Block,
Skalar je (Block,Rohr)) weicht vom Packformat ab. Ohne Kernel ist GF5 ein
Größen-Gewinn, kein Speed-Gewinn — das wird auch so berichtet.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

ND = 13          # Ququints je uint32 (5^13 = 1.22e9 ≤ 2^32)
P = 5
QMAX = 2
SKIP_PREFIXES = ("model.vision_tower.", "model.audio_tower.", "lm_head.")
# `embed_tokens` ist mit dem lm_head geteilt (tie_word_embeddings=True) und wird
# als volle Matrix gebraucht -> bf16. `embed_tokens_per_layer` ist dagegen ein
# reiner Lookup und wird quantisiert.
PACK_CHUNK_BYTES = 256 * 1024 * 1024   # float32-Peak je Chunk


# ── Kernmathematik ───────────────────────────────────────────────────────
def _pick_block(n: int) -> int:
    for b in (128, 64, 32, 16, 8):
        if n % b == 0:
            return b
    return 0


def pack_array(W: np.ndarray, block: int, k: float = 1.0):
    """(m,n) float32 → (u uint32, S fp16 (nb,)). Erwartet m*n % block == 0."""
    nb = (W.size) // block
    Wb = W.reshape(nb, block)
    amax = np.abs(Wb).max(axis=1, keepdims=True)
    S = np.maximum(k * amax / QMAX, 1e-12).astype(np.float32)
    Q = np.clip(np.rint(Wb / S), -QMAX, QMAX).astype(np.int8)
    d = (Q.astype(np.int64) + QMAX)
    nw = (d.size + ND - 1) // ND
    dp = np.full(nw * ND, QMAX, dtype=np.int64)
    dp[:d.size] = d.reshape(-1)
    pk = (P ** np.arange(ND, dtype=np.int64))
    u = (dp.reshape(nw, ND) * pk).sum(axis=1).astype(np.uint32)
    return u, S.astype(np.float16).reshape(nb)


def _blocks_per_chunk(block: int, max_bytes: int = PACK_CHUNK_BYTES) -> int:
    """Chunk-Größe in 128er-Blöcken, Vielfaches von ND (Digit-Alignment)."""
    return max(ND, (max_bytes // (block * 4)) // ND * ND)


def pack_rows(t: torch.Tensor, block: int, k: float = 1.0):
    """Lookup-Tabellen zeilenweise packen: jede Zeile beginnt auf einem
    Record-Rand (Pad auf ein Vielfaches von ND Digits). Nur so lassen sich
    einzelne Zeilen entpacken, ohne den globalen Digit-Strom zu verschieben.

    Rückgabe: u (num_rows*rec_per_row uint32), S (num_rows, blocks_per_row) fp16,
              rec_per_row
    """
    rows, dim = t.shape
    bpr = dim // block
    rec_per_row = -(-dim // ND)
    S_all = torch.empty((rows, bpr), dtype=torch.float16)
    u_all = torch.empty(rows * rec_per_row, dtype=torch.int64)
    rows_per_chunk = max(1, (PACK_CHUNK_BYTES // (dim * 4)))
    for start in range(0, rows, rows_per_chunk):
        end = min(rows, start + rows_per_chunk)
        W = t[start:end].float().numpy()
        m = end - start
        Wb = W.reshape(m * bpr, block)
        amax = np.abs(Wb).max(axis=1, keepdims=True)
        Sc = np.maximum(k * amax / QMAX, 1e-12).astype(np.float32)
        Q = np.clip(np.rint(Wb / Sc), -QMAX, QMAX).astype(np.int8)
        S_all[start:end] = torch.from_numpy(Sc.reshape(m, bpr).astype(np.float16))
        d = (Q.astype(np.int64) + QMAX).reshape(m, dim)
        dp = np.full((m, rec_per_row * ND), QMAX, dtype=np.int64)
        dp[:, :dim] = d
        pk = (P ** np.arange(ND, dtype=np.int64))
        u_all[start * rec_per_row:end * rec_per_row] = torch.from_numpy(
            (dp.reshape(m * rec_per_row, ND) * pk).sum(axis=1))
    return u_all.numpy().astype(np.uint32), S_all.numpy(), rec_per_row


def unpack_rows(u: np.ndarray, S: np.ndarray, row_ids: np.ndarray,
                dim: int, block: int, rec_per_row: int) -> np.ndarray:
    """Dequantisiert nur die angeforderten Zeilen."""
    bpr = dim // block
    n = len(row_ids)
    recs = (row_ids[:, None] * rec_per_row
            + np.arange(rec_per_row)[None, :]).reshape(-1)
    uu = u[recs].astype(np.int64)
    out = np.empty((uu.size, ND), dtype=np.int8)
    for j in range(ND):
        out[:, j] = ((uu // (P ** j)) % P).astype(np.int8) - QMAX
    digits = out.reshape(n, rec_per_row * ND)[:, :dim]
    Sc = S[row_ids].astype(np.float32)                       # (n, bpr)
    W = digits.astype(np.float32).reshape(n, bpr, block) * Sc[:, :, None]
    return W.reshape(n, dim)


def pack_tensor_chunked(t: torch.Tensor, block: int, k: float = 1.0,
                        max_bytes: int = PACK_CHUNK_BYTES):
    """Chunked, damit auch 4,7-GB-Embedding-Tabellen ohne float32-Peak passen.

    Wichtig: Chunks müssen ein Vielfaches von ND Digits umfassen, sonst
    entstünden zwischen den Chunks Padding-Digits und der globale
    Digit-Strom wäre beim Entpacken verschoben. block=128 und ND=13 sind
    teilerfremd -> Chunk = Vielfaches von 13 Blöcken.
    """
    m, n = t.shape
    nb_total = (m * n) // block
    blocks_per_chunk = _blocks_per_chunk(block, max_bytes)
    u_parts, s_parts = [], []
    for start in range(0, nb_total, blocks_per_chunk):
        end = min(nb_total, start + blocks_per_chunk)
        flat = t.reshape(-1)[start * block:end * block].float().numpy()
        u, S = pack_array(flat, block, k)
        u_parts.append(u)
        s_parts.append(S)
    return np.concatenate(u_parts), np.concatenate(s_parts)


def unpack_array(u: np.ndarray, S: np.ndarray, block: int) -> np.ndarray:
    """u32 + Skalare → float32 (flach)."""
    uu = u.astype(np.int64)
    out = np.empty((uu.size, ND), dtype=np.int8)
    for j in range(ND):
        out[:, j] = ((uu // (P ** j)) % P).astype(np.int8) - QMAX
    nb = S.size
    Q = out.reshape(-1)[: nb * block].astype(np.float32).reshape(nb, block)
    return (Q * S.astype(np.float32).reshape(nb, 1)).reshape(-1)


def rel_l2_tensor(t: torch.Tensor, u, S, block: int) -> float:
    m, n = t.shape
    nb = S.size
    num = 0.0
    den = 0.0
    blocks_per_chunk = _blocks_per_chunk(block)
    for start in range(0, nb, blocks_per_chunk):
        end = min(nb, start + blocks_per_chunk)
        ref = t.reshape(-1)[start * block:end * block].float().numpy()
        # Blockindex -> Recordindex (Chunks sind ND-aligned, s. o.)
        r0 = start * block // ND
        r1 = min(u.size, -(-end * block // ND))   # ceil: letzter Chunk ist gepaddet
        rec = unpack_array(u[r0:r1], S[start:end], block)
        num += float(((ref - rec) ** 2).sum())
        den += float((ref ** 2).sum())
    return float((num / max(den, 1e-30)) ** 0.5)


# ── Packen ───────────────────────────────────────────────────────────────
def cmd_pack(a):
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, out = a.src, a.out
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    packed, rest, manifest = {}, {}, {}
    t_start = time.time()
    n_q = 0
    total_q_bytes = 0

    for fn in files:
        with safe_open(os.path.join(src, fn), framework="pt") as f:
            keys = list(f.keys())
            for i, key in enumerate(keys):
                t = f.get_tensor(key)
                is_tied_embed = key.endswith("embed_tokens.weight")
                skipped = (key.startswith(SKIP_PREFIXES) or is_tied_embed
                           or t.ndim != 2 or min(t.shape) <= 1)
                block = 0 if skipped else _pick_block(t.shape[1])
                if skipped or block == 0:
                    rest[key] = t.contiguous()
                    manifest[key] = {"format": "bf16", "shape": list(t.shape)}
                    continue
                is_lookup = "embed_tokens_per_layer" in key
                if is_lookup:
                    u, S, rec_per_row = pack_rows(t, block, a.k)
                else:
                    u, S = pack_tensor_chunked(t, block, a.k)
                    rec_per_row = 0
                if a.stages >= 2 and not is_lookup:
                    # Residual-Stufe: R = W - dequant(u,S); R wird erneut GF5-gepackt.
                    flat = t.reshape(-1)
                    nb = S.size
                    bpc = _blocks_per_chunk(block)
                    r2u, r2s = [], []
                    for _s in range(0, nb, bpc):
                        _e = min(nb, _s + bpc)
                        ref = flat[_s * block:_e * block].float().numpy()
                        rec = unpack_array(u[_s * block // ND: min(u.size, -(-_e * block // ND))],
                                           S[_s:_e], block)
                        _u2, _s2 = pack_array(ref - rec, block, a.k)
                        r2u.append(_u2); r2s.append(_s2)
                    packed[key + ".gf5b_u32"] = torch.from_numpy(np.concatenate(r2u))
                    packed[key + ".gf5b_scale"] = torch.from_numpy(np.concatenate(r2s))
                packed[key + ".gf5_u32"] = torch.from_numpy(u)
                packed[key + ".gf5_scale"] = torch.from_numpy(S).reshape(-1) \
                    if not is_lookup else torch.from_numpy(S)
                if is_lookup:
                    idx = np.linspace(0, t.shape[0] - 1,
                                      min(4096, t.shape[0])).astype(np.int64)
                    ref = t[idx].float().numpy()
                    rec = unpack_rows(u, S, idx, t.shape[1], block, rec_per_row)
                    rel = float((((ref - rec) ** 2).sum() / (ref ** 2).sum()) ** 0.5)
                else:
                    rel = rel_l2_tensor(t, u, S, block)
                manifest[key] = {"format": "gf5", "shape": list(t.shape),
                                 "block": block, "k": a.k,
                                 "layout": "rows" if is_lookup else "flat",
                                 "stages": 1 if (is_lookup or a.stages < 2) else 2,
                                 "rec_per_row": rec_per_row,
                                 "rel_l2": round(rel, 5),
                                 "bits": round(32.0 / ND + 16.0 / block, 3)}
                n_q += 1
                total_q_bytes += u.nbytes + S.nbytes
                del t
                if n_q % 100 == 0:
                    print(f"  [{n_q:4d} quantisiert] {key} "
                          f"({time.time()-t_start:.0f}s, {total_q_bytes/1e9:.2f} GB gepackt)",
                          flush=True)

    for extra in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "processor_config.json",
                  "generation_config.json", "special_tokens_map.json"):
        p = os.path.join(src, extra)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, extra))

    save_file(packed, os.path.join(out, "gf5_packed.safetensors"))
    save_file(rest, os.path.join(out, "gf5_rest.safetensors"))

    rels = [v["rel_l2"] for v in manifest.values() if "rel_l2" in v]
    lm = [v["rel_l2"] for k, v in manifest.items()
          if "language_model.layers" in k and "rel_l2" in v]
    manifest["_meta"] = {
        "src": src, "k": a.k, "stages": a.stages, "digits_u32": ND, "qmax": QMAX,
        "n_quantized": n_q,
        "packed_bytes": sum(v.numel() * v.element_size() for v in packed.values()),
        "rest_bytes": sum(v.numel() * v.element_size() for v in rest.values()),
        "rel_l2_mean": round(float(np.mean(rels)), 5) if rels else None,
        "rel_l2_p95": round(float(np.percentile(rels, 95)), 5) if rels else None,
        "rel_l2_max": round(float(max(rels)), 5) if rels else None,
        "rel_l2_language_mean": round(float(np.mean(lm)), 5) if lm else None,
        "seconds": round(time.time() - t_start, 1),
    }
    json.dump(manifest, open(os.path.join(out, "gf5_manifest.json"), "w"), indent=1)

    m = manifest["_meta"]
    src_bytes = sum(os.path.getsize(os.path.join(src, f)) for f in files)
    print(f"\nquantisierte Tensoren: {m['n_quantized']}")
    print(f"Quelle:   {src_bytes/1e9:.2f} GB")
    print(f"GF5:      {m['packed_bytes']/1e9:.2f} GB + Rest {m['rest_bytes']/1e9:.2f} GB "
          f"= {(m['packed_bytes']+m['rest_bytes'])/1e9:.2f} GB "
          f"({src_bytes/(m['packed_bytes']+m['rest_bytes']):.2f}× kleiner)")
    print(f"rel-L2: mean {m['rel_l2_mean']} | p95 {m['rel_l2_p95']} | max {m['rel_l2_max']} "
          f"| language_model {m['rel_l2_language_mean']}")
    print(f"geschrieben: {out}  ({m['seconds']}s)")


# ── Module ───────────────────────────────────────────────────────────────
class GF5Linear(torch.nn.Module):
    """Lineare Schicht mit GF(5)-Pack **im RAM**; dequantisiert pro Aufruf.

    Kein persistenter Dequant-Cache: alle 278 Matrizen gleichzeitig
    dequantisiert wären 8,5 GB VRAM — genau der Speicher, den das Format
    einsparen soll. Pro Aufruf fällt nur eine Matrix an (~19 MB bei
    1536x6144), die sofort wieder freigegeben wird.
    """

    def __init__(self, u, S, out_features, in_features, block, bias=None):
        super().__init__()
        self.register_buffer("u32", u.to(torch.int32), persistent=False)
        self.register_buffer("scale", S.to(torch.float16), persistent=False)
        self.out_features = out_features
        self.in_features = in_features
        self.block = block
        self.bias = bias

    def dequantized(self, dtype=torch.bfloat16, device="cpu"):
        u = self.u32.detach().cpu().numpy().view(np.uint32)
        S = self.scale.detach().float().cpu().numpy()
        W = unpack_array(u, S, self.block)
        return torch.from_numpy(W.reshape(self.out_features, self.in_features)).to(device, dtype)

    def forward(self, x):
        W = self.dequantized(x.dtype, x.device)
        if getattr(self, "u32b", None) is not None:
            ub = self.u32b.detach().cpu().numpy().view(np.uint32)
            Sb = self.scaleb.detach().float().cpu().numpy()
            W = W + torch.from_numpy(
                unpack_array(ub, Sb, self.block).reshape(self.out_features, self.in_features)
            ).to(x.device, x.dtype)
        out = torch.nn.functional.linear(x, W, self.bias)
        del W
        return out

    def packed_bytes(self):
        n = self.u32.numel() * 4 + self.scale.numel() * 2
        if getattr(self, "u32b", None) is not None:
            n += self.u32b.numel() * 4 + self.scaleb.numel() * 2
        return n


class GF5Embedding(torch.nn.Module):
    """Lookup-Tabelle im zeilen-alignten GF5-Layout (nur angefragte Zeilen
    werden dequantisiert)."""

    def __init__(self, u, S, num_embeddings, embedding_dim, block, rec_per_row):
        super().__init__()
        self.register_buffer("u32", u.to(torch.int32), persistent=False)
        self.register_buffer("scale", S.to(torch.float16), persistent=False)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.block = block
        self.rec_per_row = rec_per_row

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        rows = unpack_rows(self.u32.detach().cpu().numpy().view(np.uint32),
                           self.scale.detach().float().cpu().numpy(),
                           uniq.cpu().numpy(), self.embedding_dim,
                           self.block, self.rec_per_row)
        t = torch.from_numpy(rows).to(ids.device)
        return t[inv].reshape(*ids.shape, self.embedding_dim)

    def packed_bytes(self):
        return self.u32.numel() * 4 + self.scale.numel() * 2


class _DtypeCastLinear(torch.nn.Module):
    """Kapselt eine Linearschicht und castet die Aktivierung auf ihr Gewicht.

    Nötig, weil der Modell-Forward intern float32-Buffer (RoPE inv_freq) hält
    und die Hidden States dadurch fp32 werden, während lm_head bf16 ist.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x.to(self.inner.weight.dtype))


def load_gf5_model(out_dir: str, device: str = "cuda", dtype=torch.bfloat16):
    """Baut das Modell aus der Config (keine bf16-Gewichte laden) und ersetzt
    Linears/Embeddings durch die gepackten GF5-Module."""
    from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig
    from safetensors.torch import load_file

    manifest = json.load(open(os.path.join(out_dir, "gf5_manifest.json")))
    print("[gf5] instantiating model from config (no bf16 weights) ...", flush=True)
    cfg = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_config(
        cfg, trust_remote_code=True, torch_dtype=dtype)
    try:
        model.generation_config = GenerationConfig.from_pretrained(out_dir)
    except Exception:
        pass

    packed = load_file(os.path.join(out_dir, "gf5_packed.safetensors"))
    rest = load_file(os.path.join(out_dir, "gf5_rest.safetensors"))

    replaced_l = replaced_e = 0
    for name, mod in list(model.named_modules()):
        key = name + ".weight"
        uk, sk = key + ".gf5_u32", key + ".gf5_scale"
        if uk not in packed:
            continue
        meta = manifest[key]
        if isinstance(mod, torch.nn.Linear):
            new = GF5Linear(packed[uk], packed[sk], meta["shape"][0], meta["shape"][1],
                            meta["block"], bias=mod.bias)
            if uk.replace(".gf5_u32", ".gf5b_u32") in packed:
                new.register_buffer(
                    "u32b", packed[uk.replace(".gf5_u32", ".gf5b_u32")].to(torch.int32),
                    persistent=False)
                new.register_buffer(
                    "scaleb", packed[uk.replace(".gf5_u32", ".gf5b_scale")].to(torch.float16),
                    persistent=False)
            replaced_l += 1
        elif isinstance(mod, torch.nn.Embedding):
            new = GF5Embedding(packed[uk], packed[sk], meta["shape"][0], meta["shape"][1],
                               meta["block"], meta["rec_per_row"])
            replaced_e += 1
        else:
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], new)
    print(f"[gf5] replaced {replaced_l} Linear + {replaced_e} Embedding", flush=True)

    sd = model.state_dict()
    keep = {k: v for k, v in rest.items() if k in sd and sd[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"[gf5] rest tensors loaded: {len(keep)} | missing: {len(missing)} "
          f"| unexpected: {len(unexpected)}", flush=True)

    import gc
    gc.collect()
    _pb = sum(p.numel() * p.element_size() for p in model.parameters())
    _bb = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"[gf5] vor .to(device): params {_pb/1e9:.2f} GB, buffers {_bb/1e9:.2f} GB "
          f"(davon gepackt {sum(m.packed_bytes() for m in model.modules() if isinstance(m, (GF5Linear, GF5Embedding)))/1e9:.2f} GB)",
          flush=True)
    # Tied embeddings + einheitlicher dtype (from_config baut fp32)
    try:
        model.tie_weights()
    except Exception:
        pass
    for _p in model.parameters():
        if _p.is_floating_point():
            _p.data = _p.data.to(dtype)
    for _b in model.buffers():
        if _b.is_floating_point():
            _b.data = _b.data.to(dtype)
    model = model.to(dtype=dtype)
    if hasattr(model, "lm_head"):
        model.lm_head = _DtypeCastLinear(model.lm_head)
    model = model.to(device)
    for _m in model.modules():
        if isinstance(_m, (GF5Linear, GF5Embedding)):
            _m.u32 = _m.u32.cpu()
            _m.scale = _m.scale.cpu()
    model.eval()
    del packed, rest, sd, keep
    gc.collect()
    torch.cuda.empty_cache()
    return model


def cmd_eval(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.out)
    model = load_gf5_model(a.out, device=a.device)
    packed_bytes = sum(m.packed_bytes() for m in model.modules()
                       if isinstance(m, (GF5Linear, GF5Embedding)))
    live = sum(p.numel() * p.element_size() for p in model.parameters())
    vram = torch.cuda.memory_allocated() / 1e9 if a.device == "cuda" else 0.0
    print(f"[gf5] gepackt: {packed_bytes/1e9:.2f} GB | Parameter live: {live/1e9:.2f} GB | "
          f"VRAM: {vram:.2f} GB")

    prompts = [
        "Write a Python function `add(a, b)` that returns the sum of two numbers. "
        "Return only one ```python code block.",
        "Write a Python function `is_even(n)` returning True for even n. Code only.",
        "Explain in two sentences why the sky is blue.",
    ]
    res = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        dt = time.time() - t0
        n_new = gen.shape[1] - ids["input_ids"].shape[1]
        txt = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        res.append({"prompt": p, "text": txt, "wall_s": round(dt, 2),
                    "tok_s": round(n_new / dt, 2) if dt else None})
        print(f"\n--- {dt:.1f}s ({n_new/dt:.1f} tok/s) ---\n{txt[:400]}", flush=True)

    json.dump({"device": a.device, "packed_bytes": packed_bytes,
               "live_param_bytes": live, "vram_bytes": vram, "results": res},
              open(os.path.join(a.out, "gf5_eval.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(a.out, 'gf5_eval.json')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pack")
    p1.add_argument("--src", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--stages", type=int, default=2,
                    help="1 = GF5, 2 = GF5 + GF5-Residual (GF5², rel-L2 ~0.114 @ 5.17 bit)")
    p1.add_argument("--k", type=float, default=0.6,
                    help="Skalar-Fit k (S = k*amax/qmax); 0.6 ist auf echten "
                         "gemma-4-Gewichten gemessen optimal (rel-L2 0.316 vs 0.451 bei k=1.0)")
    p1.set_defaults(func=cmd_pack)
    p2 = sub.add_parser("eval")
    p2.add_argument("--out", required=True)
    p2.add_argument("--device", default="cuda")
    p2.add_argument("--max-new", type=int, default=120)
    p2.set_defaults(func=cmd_eval)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
