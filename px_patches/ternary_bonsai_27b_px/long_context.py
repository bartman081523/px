#!/usr/bin/env python
"""KV-4bit-Cache + Chunked-Prefill-Generator (Stufe 3c) — 128k+ Kontext.

Geometrie (real gelesen: ternary-bonsai-2-27b config.json):
    64 Layer = 48x linear_attention (GDN, rekurrent — KEIN KV-Cache)
             + 16x full_attention (num_key_value_heads=4 x head_dim=256
               -> 2048 K+V-Elemente je Token/Layer).
    bf16-KV @131072: 16 x 131072 x 2048 x 2 B = 8,0 GiB — neben ~6,7 GiB
    (GF3-)Gewichten nicht auf die 12-GB-Karte. Also 4-Bit-KV:
    132 B je (Token, KV-Head) (128 B Nibbles + 2x2 B Skalen) ->
    16 x 131072 x 4 x 132 B = 1,11 GiB (~7,2x kleiner).
Dequant-Transient je Update: KV in bf16 = L x 2048 x 2 B = 512 MB @131k
(PyTorch-Allokator wiederverwendet den Block; Peak-Budget ~10,8 GiB).
Prefill-Dekor-Gesamtkosten O(n^2/chunk): ~350 GB Traffic @131k ~ 1-3 s —
gegen 8-11 min GEMM-Prefill vernachlaessigbar.

Aufbau (Interface-Beweise: transformers 5.13 cache_utils.py):
    QuantKV4Layer   — DynamicLayer-Ident-Oberflaeche, speichert u8+f16.
                      KEIN layer_type-Attribut -> keine Registry-Mutation
                      (__init_subclass__ registriert nur Klassen MIT
                      explizitem layer_type; Cache-Zeile 45). Muster-Vorlage
                      ist transformers-eigen: QuantizedLayer(DynamicLayer)
                      (KIVI: gepackt lagern, dekodieren bei update).
    make_qwen35_cache — DynamicCache(config) + Layer-Swap je full_attention;
                      GDN-Layer unberuehrt (update_conv_state /
                      update_recurrent_state laufen wie zuvor).
                      Cache.get_seq_length() OHNE layer_idx springt von
                      linear auf die erste CacheLayerMixin-Layer (Zeile
                      1315-1335: LinearAttentionLayer erbt NICHT von
                      CacheLayerMixin) -> past_seen in px_forward stimmt.
    chunked_prefill  — Forward je 2048-Chunks ueber model.model (px_forward
                      aktiv), lm_head NUR auf der letzten Hidden-Spalte des
                      letzten Chunks (logits je Chunk waeren 1,0 GB bf16).
                      _px_is_decode (patch.py): Chunk 1 detektiert Prefill
                      (shape>1 und past=0) und setzt den Telemetrie-Trace
                      zurueck; px-Metriken stammen vom letzten Forward.
    decode_loop      — Autoregression je Token (lm_head 248320x1), Sampling
                      (temp/top_k/top_p/rep_pen), EOS-Break, HF-Streamer.
    generate_long    — Orchestrator; Env PX_KV_MODE (kv4|bf16) /
                      PX_PREFILL_CHUNK (2048).

Nicht halten: keys/values (DynamicLayer-Symbole) bleiben None — die
dekodierte Rueckgabe ist ein Transient, damit @131k KEINE 8 GiB bf16-KV
materialisieren (sonst genau das, was der Layer vermeidet).
"""
import math
import os

import torch
from transformers.cache_utils import DynamicLayer

_KV_SCALE_MAX = 7          # symmetrisches int4: q in {-7..+7}
_KV_HALF = 8               # Nibble-Bias (q + 8 in {1..15})


def _resolve_text_model(model):
    """Text-Modell aus ForCausalLM/Multimodal-Wrapper (patch.py-Zeile 53).

    Wird als lokale Spiegelung statt Import gepflegt — patch.py nutzt
    relative Imports (from .anti_zombie_sensor import ...), kann also
    nicht als Flat-Modul neben gf3_quant geladen werden.
    """
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    return model


# ------------------------------------------------------------- Pack/Dequant ---
def kv4_pack(x):
    """(B,H,L,256) -> (u8 (B,H,L,128), f16 (B,H,L,2)), symmetrisch int4.

    s = amax|128-Block| / 7 (fp16), q = round(x/s) in {-7..7}, Nibble
    = q+8 in {1..15}; lo-Block (Dims 0..127) im Low-Nibble, hi-Block im
    High-Nibble. Pack-Transient bezieht sich nur auf NEUE Tokens (klein).
    """
    B, H, L, D = x.shape
    xb = x.reshape(B, H, L, 2, D // 2)
    amax = xb.abs().amax(dim=-1).to(torch.float16)                    # (B,H,L,2)
    s = (amax.to(torch.float32) / _KV_SCALE_MAX).clamp_min(1e-8).to(
        torch.float16)
    q = torch.round(xb / s.to(torch.float32).unsqueeze(-1)).clamp(
        -_KV_SCALE_MAX, _KV_SCALE_MAX)
    q = (q.to(torch.int16) + _KV_HALF).to(torch.uint8)                # {1..15}
    u8 = (q[..., 0, :].to(torch.int16)
          | (q[..., 1, :].to(torch.int16) << 4)).to(torch.uint8)
    return u8, s


def kv4_dequant(u8, s, dtype):
    """(B,H,L,128) u8 + (B,H,L,2) f16 -> (B,H,L,256) dtype."""
    B, H, L, _ = u8.shape
    k = torch.empty(B, H, L, 2 * u8.shape[-1], dtype=dtype, device=u8.device)
    lo = (u8 & 0x0F).to(torch.int16) - _KV_HALF                       # {-7..7}
    hi = ((u8.to(torch.int16) >> 4) & 0x0F) - _KV_HALF
    k[..., : u8.shape[-1]] = (lo.to(torch.float16)
                              * s[..., 0:1]).to(dtype)
    k[..., u8.shape[-1]:] = (hi.to(torch.float16)
                             * s[..., 1:2]).to(dtype)
    return k


# ----------------------------------------------------------------- Layer ------
class QuantKV4Layer(DynamicLayer):
    """4-Bit-KV-Cache-Layer mit DynamicLayer-Ident-Oberflaeche.

    Vererbung ist PFLICHT: Cache.get_seq_length() (ohne layer_idx) springt
    zur ersten CacheLayerMixin-Instanz (Zeile 1315-1335) — ein nacktes
    nn.Module wuerde alle 16 KV4-Layer ueberspringen und StopIteration
    werfen. QuantizedLayer(DynamicLayer) ist transformers' eigenes Muster.

    Speichert u8 (B,H,L,D/2) + f16-Skalen (B,H,L,2) statt bf16 (B,H,L,D);
    update() konkatiniert gepackt und gibt DEKODIERTE K/V zurueck (wie
    transformers QuantizedLayer, aber ohne Residual-Buffer).

    KEIN layer_type-Klassenattribut -> __init_subclass__ in
    CacheLayerMixin schreibt die Klasse NICHT in LAYER_TYPE_CACHE_MAPPING
    (keine globale Registry-Mutation). Wird nur per Layer-Swap in
    make_qwen35_cache an full_attention-Slots gehaengt.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.ku8 = None
        self.ks = None
        self.vu8 = None
        self.vs = None

    # --- Basis-Einrichtung (DynamicLayer-Spiegel ohne leere Tensoren) ---
    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    # --- Kern ---
    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        u8, s = kv4_pack(key_states)
        v8, w = kv4_pack(value_states)
        if self.ku8 is None:
            self.ku8, self.ks, self.vu8, self.vs = u8, s, v8, w
        else:
            self.ku8 = torch.cat([self.ku8, u8], dim=-2)
            self.ks = torch.cat([self.ks, s], dim=-2)
            self.vu8 = torch.cat([self.vu8, v8], dim=-2)
            self.vs = torch.cat([self.vs, w], dim=-2)
        return (self.kv4_keys(), self.kv4_values())

    def kv4_keys(self):
        return kv4_dequant(self.ku8, self.ks, self.dtype)

    def kv4_values(self):
        return kv4_dequant(self.vu8, self.vs, self.dtype)

    # --- Lese-Oberflaeche (Cache/px_forward/mask-Hebel) ---
    def get_seq_length(self) -> int:
        return 0 if self.ku8 is None else self.ku8.shape[-2]

    def get_mask_sizes(self, query_length: int) -> tuple:
        return self.get_seq_length() + query_length, 0

    def get_max_length(self) -> int:
        return -1

    # --- Struktur-Operationen (mixin-Muster gespiegelt) ---
    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        for name in ("ku8", "ks", "vu8", "vs"):
            setattr(self, name, getattr(self, name)[..., :max_length, :])

    def reset(self) -> None:
        self.ku8 = self.ks = self.vu8 = self.vs = None
        self.is_initialized = False

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() > 0:
            for name in ("ku8", "ks", "vu8", "vs"):
                setattr(self, name, getattr(self, name).repeat_interleave(
                    repeats, dim=0))

    def batch_select_indices(self, indices) -> None:
        if self.get_seq_length() > 0:
            for name in ("ku8", "ks", "vu8", "vs"):
                setattr(self, name, getattr(self, name)[indices, ...])

    def reorder_cache(self, beam_idx) -> None:
        if self.get_seq_length() > 0:
            for name in ("ku8", "ks", "vu8", "vs"):
                setattr(self, name, getattr(self, name).index_select(
                    0, beam_idx.to(self.ku8.device)))

    def offload(self) -> None:
        if self.is_initialized and self.ku8 is not None:
            for name in ("ku8", "ks", "vu8", "vs"):
                setattr(self, name, getattr(self, name).to("cpu",
                                                           non_blocking=True))

    def prefetch(self) -> None:
        if self.is_initialized and self.ku8 is not None \
                and self.ku8.device != self.device:
            for name in ("ku8", "ks", "vu8", "vs"):
                setattr(self, name, getattr(self, name).to(
                    self.device, non_blocking=True))


# -------------------------------------------------------------- Cache-Bau -----
def make_qwen35_cache(config, kv_mode=None):
    """DynamicCache(config) + Layer-Swap: full_attention -> QuantKV4Layer.

    kv_mode: "kv4" (Standard) | "bf16" (reine DynamicCache). GDN-Layer
    (linear_attention -> LinearAttentionLayer) unberuehrt.
    """
    from transformers.cache_utils import DynamicCache

    kv_mode = kv_mode or os.environ.get("PX_KV_MODE", "kv4")
    text = config.get_text_config()          # DynamicCache macht dasselbe
    cache = DynamicCache(config=text)
    if kv_mode == "kv4":
        n_swap = 0
        for i, lt in enumerate(text.layer_types):
            if lt == "full_attention":
                cache.layers[i] = QuantKV4Layer()
                n_swap += 1
        cache._px_kv_mode = kv_mode
        cache._px_n_kv4_layers = n_swap
    elif kv_mode not in ("bf16", "none", None):
        raise ValueError(f"unbekannter PX_KV_MODE: {kv_mode!r}")
    return cache


def kv4_size_estimate(config, seq_len):
    """KV-Bytes je Modus -> (kv4_gib, bf16_gib) — Budget-Ausweis."""
    text = config.get_text_config()
    n_full = sum(1 for t in text.layer_types if t == "full_attention")
    h_kv, d = text.num_key_value_heads, text.head_dim
    kv4 = n_full * seq_len * h_kv * (d // 2 + 4)
    bf16 = n_full * seq_len * h_kv * 2 * d * 2
    return kv4 / 2**30, bf16 / 2**30


# ------------------------------------------------------- Chunked Prefill ------
def chunked_prefill(model, input_ids, cache, chunk=None, verbose=False):
    """Prefill (1, T) in Chunks ueber das px-gepatchte Text-Modell.

    lm_head NUR auf die letzte Hidden-Spalte des letzten Chunks — die
    ForCausalLM-Zeile wuerde sonst je Chunk (chunk x 248320 x 2 B) bf16-
    Logits materialisieren (1,0 GB je 2048er-Chunk).
    Rueckgabe: Logits (1, 1, V) float32.
    """
    chunk = int(chunk or os.environ.get("PX_PREFILL_CHUNK", 2048))
    base = _resolve_text_model(model)
    T = input_ids.shape[1]
    n_chunks = math.ceil(T / chunk)
    h_last = None
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        out = base(input_ids=input_ids[:, lo:hi], past_key_values=cache,
                   use_cache=True)
        h_last = out.last_hidden_state[:, -1:]
        if verbose:
            print(f"    prefill {hi}/{T} (Chunk {hi // chunk}/{n_chunks})",
                  flush=True)
    return model.lm_head(h_last).float()


# ------------------------------------------------------------ Sampling --------
def _ban_repeat_ngrams(logits, generated_ids, ngram):
    """HF-äquivalentes no_repeat_ngram_size: Continuation-Token je
    wiederholtem (n-1)-Präfix sperren. Wird nur angewendet, wenn mind.
    ein Token ungesperrt bleibt (verhindert all--inf-Softmax-NaN)."""
    g, n = generated_ids, int(ngram)
    if n < 2 or len(g) < n - 1:
        return
    prefix = tuple(g[len(g) - (n - 1):])
    banned = {g[i + n - 1] for i in range(len(g) - n + 1)
              if tuple(g[i:i + n - 1]) == prefix}
    if not banned:
        return
    cand = torch.tensor(sorted(banned), device=logits.device,
                        dtype=torch.long)
    keep = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    keep[cand] = False
    # Guard: bannt nur, wenn ein endlicher Nicht-Block-Token bleibt
    # (all--inf wäre Softmax-NaN).
    if keep.numel() and torch.isfinite(logits[keep]).any():
        logits[cand] = float("-inf")


def _apply_logit_warps(logits, generated_ids, prompt_uniq, temperature,
                       top_k, top_p, repetition_penalty,
                       no_repeat_ngram_size=0):
    """HF-konforme Reihenfolge: rep_penalty -> ngram -> temp -> top_k -> top_p."""
    if repetition_penalty != 1.0 and (prompt_uniq is not None
                                      or generated_ids):
        parts = []
        if prompt_uniq is not None and prompt_uniq.numel():
            parts.append(prompt_uniq.to(logits.device))
        if generated_ids:
            parts.append(torch.tensor(generated_ids, device=logits.device,
                                      dtype=torch.long))
        if parts:
            ids = torch.unique(torch.cat(parts))
            sel = logits.index_select(0, ids)
            logits[ids] = torch.where(
                sel < 0, sel * repetition_penalty,
                sel / repetition_penalty)
    # ngram-Block vor temp/top_*: HF maskiert vor dem Sampling (Warpers
    # laufen bei HF in fixer Reihenfolge; ngram-Ban ist kein Warp) — hier
    # in der Warp-Funktion, weil die ngram-Sperre vor temp wirkt.
    _ban_repeat_ngrams(logits, generated_ids, no_repeat_ngram_size)
    if temperature and temperature > 0:
        logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        order = torch.argsort(logits, descending=True)
        probs = torch.softmax(logits[order], dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cut = cum - probs >= top_p                 # Kern (inclusive) behalten
        logits[order[cut]] = float("-inf")
    return logits


def decode_loop(model, cache, logits, max_new_tokens, temperature=0.7,
                top_k=40, top_p=0.95, repetition_penalty=1.15,
                eos_token_ids=(), streamer=None, max_total_seq=None,
                prompt_uniq=None, no_repeat_ngram_size=0, verbose=False):
    """Autoregression auf existierendem Cache. Rueckgabe: (ids (1, n), steps).

    logits: (1, 1, V) f32 vom Prefill-Ende; der Loop ruft model(...) je
    Token (ForCausalLM.forward -> px-gepatchtes Text-Modell interno).
    """
    gen = []
    cur = logits[0, -1] if logits.ndim == 3 else logits[0]
    steps = 0
    for step in range(max_new_tokens):
        lg = _apply_logit_warps(cur.clone(), gen, prompt_uniq, temperature,
                                top_k, top_p, repetition_penalty,
                                no_repeat_ngram_size)
        tok = int(torch.multinomial(
            torch.softmax(lg.float(), dim=-1), 1).item())
        gen.append(tok)
        steps = step + 1
        if streamer is not None:
            streamer.put(torch.tensor([[tok]], device=cur.device))
        if tok in eos_token_ids:
            break
        if max_total_seq and cache.get_seq_length() + 1 >= max_total_seq:
            if verbose:
                print(f"    Kontext-Limit {max_total_seq} erreicht (step {step})",
                      flush=True)
            break
        nxt = torch.tensor([[tok]], device=cur.device)
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        cur = out.logits[0, -1].float()
    ids = torch.tensor([gen], dtype=torch.long, device=cur.device)
    if streamer is not None:
        streamer.end()
    return ids, steps


# ------------------------------------------------------------ Orchestrator ----
def generate_long(model, input_ids, max_new_tokens=1024, chunk=None,
                  kv_mode=None, streamer=None, eos_token_ids=(),
                  max_total_seq=None, verbose=False, **sample_cfg):
    """128k+-Pfad: KV-4bit-Cache + Chunked-Prefill + Decode-Loop.

    model: px-gepatchtes Qwen3_5ForCausalLM (ForCausalLM.forward ruft das
    gepatchte model.model fuer Text und lm_head fur Logits).
    Rueckgabe: dict(generated=ids (1, n), steps=n, kv_mode=..., chunk=...).
    """
    kv_mode = kv_mode or os.environ.get("PX_KV_MODE", "kv4")
    chunk = int(chunk or os.environ.get("PX_PREFILL_CHUNK", 2048))
    # Nur bekannte Sampling-Keys weiterreichen (do_sample & Co. aus
    # model.generate-Welt laufen decode_loop ins Leere → TypeError).
    s_cfg = {k: v for k, v in sample_cfg.items()
             if k in ("temperature", "top_k", "top_p",
                      "repetition_penalty", "no_repeat_ngram_size")}
    dropped = sorted(set(sample_cfg) - set(s_cfg))
    if dropped and verbose:
        print(f"[long] ignorierte Sample-Keys: {dropped}", flush=True)
    cache = make_qwen35_cache(model.config, kv_mode)
    kv4_gib, bf16_gib = kv4_size_estimate(model.config, 131072)
    if verbose:
        print(f"[long] KV-Modus {kv_mode}: @131k kv4={kv4_gib:.2f} GiB "
              f"(bf16 waere {bf16_gib:.2f} GiB), chunk={chunk}", flush=True)
    logits = chunked_prefill(model, input_ids, cache, chunk, verbose=verbose)
    prompt_uniq = torch.unique(input_ids.reshape(-1)) \
        if repetition_penalty_wanted(s_cfg) else None
    ids, steps = decode_loop(model, cache, logits, max_new_tokens,
                             streamer=streamer, eos_token_ids=eos_token_ids,
                             max_total_seq=max_total_seq,
                             prompt_uniq=prompt_uniq, verbose=verbose,
                             **s_cfg)
    return {"generated": ids, "steps": steps, "kv_mode": kv_mode,
            "chunk": chunk, "cache": cache}


def repetition_penalty_wanted(sample_cfg):
    """rep_pen aktiv? (decode_loop-Default 1.15 -> Prompt-Ids noetig)."""
    return sample_cfg.get("repetition_penalty", 1.15) != 0