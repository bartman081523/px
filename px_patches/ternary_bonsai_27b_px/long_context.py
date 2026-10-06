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
import importlib
import math
import os
import sys
import traceback

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

    def __init__(self, config=None, static_capacity=0):
        super().__init__(config)
        self.ku8 = None
        self.ks = None
        self.vu8 = None
        self.vs = None
        # TT-B1a: Static-Capacity-Modus (CUDA-Graph-Decode, PX_KV_STATIC).
        # capacity>0 -> Buffers werden bei lazy_initialization full-capacity
        # zero-init angelegt und per index_copy_ am device-Positions-Zaehler
        # beschrieben (kein torch.cat). Default 0 = cat-Pfad unveraendert.
        self.static_capacity = int(static_capacity)
        self._written = 0

    def _is_capturing(self) -> bool:
        # Capture-Erkennung robust auf CPU/ohne Kontext (Unit-Tests).
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    # --- Basis-Einrichtung (DynamicLayer-Spiegel ohne leere Tensoren) ---
    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        if self.static_capacity:
            B, H, _, D = key_states.shape
            dev = key_states.device
            cap = self.static_capacity
            self.ku8 = torch.zeros((B, H, cap, D // 2), dtype=torch.uint8,
                                   device=dev)
            self.ks = torch.zeros((B, H, cap, 2), dtype=torch.float16,
                                  device=dev)
            self.vu8 = torch.zeros((B, H, cap, D // 2), dtype=torch.uint8,
                                   device=dev)
            self.vs = torch.zeros((B, H, cap, 2), dtype=torch.float16,
                                  device=dev)
            # Device-Positions-Zaehler: advance im Graph via replay
            # (self.pos += n ist geräteseitig — kein Host-Sync).
            self.pos = torch.zeros((), dtype=torch.long, device=dev)
            self._written = 0
        self.is_initialized = True

    # --- Kern ---
    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        u8, s = kv4_pack(key_states)
        v8, w = kv4_pack(value_states)
        if self.static_capacity:
            n = u8.shape[-2]
            idx = torch.arange(n, device=self.ku8.device) + self.pos
            self.ku8.index_copy_(-2, idx, u8)
            self.ks.index_copy_(-2, idx, s)
            self.vu8.index_copy_(-2, idx, v8)
            self.vs.index_copy_(-2, idx, w)
            self.pos += n
            self._written += n
            if self._is_capturing():
                # Capture: kein Python-Slicing (Laenge aendert sich je
                # Schritt) -> Full-Capacity-Dequant; Gueltigkeit enforce
                # der Graph-Decoder via Maske arange(cap) < pos.
                return (self.kv4_keys(), self.kv4_values())
            # Eager: dequant nur ueber den geschriebenen Bereich (Slice-
            # View) — identische Bytes wie der cat-Pfad (G4-exakt).
            return (self.kv4_written_keys(), self.kv4_written_values())
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

    def kv4_written_keys(self):
        w = self._written
        return kv4_dequant(self.ku8[..., :w, :], self.ks[..., :w, :],
                           self.dtype)

    def kv4_written_values(self):
        w = self._written
        return kv4_dequant(self.vu8[..., :w, :], self.vs[..., :w, :],
                           self.dtype)

    # --- Lese-Oberflaeche (Cache/px_forward/mask-Hebel) ---
    def get_seq_length(self) -> int:
        if self.static_capacity:
            return self._written
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
def make_qwen35_cache(config, kv_mode=None, static_capacity=0):
    """DynamicCache(config) + Layer-Swap: full_attention -> QuantKV4Layer.

    kv_mode: "kv4" (Standard) | "bf16" (reine DynamicCache). GDN-Layer
    (linear_attention -> LinearAttentionLayer) unberuehrt.

    static_capacity > 0 (TT-B1a): QuantKV4Layer alloziert full-capacity
    Buffers statt cat-Wachstum (CUDA-Graph-Voraussetzung; Positions- und
    Gueltigkeits-Buchhaltung ueber device-Zaehler, siehe QuantKV4Layer).
    """
    from transformers.cache_utils import DynamicCache

    kv_mode = kv_mode or os.environ.get("PX_KV_MODE", "kv4")
    text = config.get_text_config()          # DynamicCache macht dasselbe
    cache = DynamicCache(config=text)
    if kv_mode == "kv4":
        n_swap = 0
        for i, lt in enumerate(text.layer_types):
            if lt == "full_attention":
                cache.layers[i] = QuantKV4Layer(
                    static_capacity=static_capacity)
                if n_swap == 0:
                    # TT-B2a: Uhr-Markierer — der erste Static-Layer ist der
                    # Device-Takt (pos vor Update) fuer Maske/Positionen des
                    # gecaptureden Decode-Forwards (patch.py Capture-Branch).
                    cache._px_kv4_layer = cache.layers[i]
                n_swap += 1
        cache._px_kv_mode = kv_mode
        cache._px_n_kv4_layers = n_swap
    elif kv_mode not in ("bf16", "none", None):
        raise ValueError(f"unbekannter PX_KV_MODE: {kv_mode!r}")
    if static_capacity:
        cache._px_static_capacity = int(static_capacity)
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


# ------------------------------------------------- TT-B2a Decode-Graph -------
class _DecodeGraph:
    """CUDA-Graph fuer den Decode-Schritt (TT-B2a, PX_DECODE_GRAPH=1).

    Capturiert den px-gepatchten Text-Forward SAMT allen PX-Hooks
    (patch.py Capture-Branch, TT-B3) + lm_head in einen statischen
    Logit-Buffer; Sampling/Streamer/Logit-Warps bleiben host-seitig
    identisch zum Eager-Pfad. Replays zaehlen den KV-Positionstakt
    device-seitig weiter (layer.pos); die Host-Buchhaltung (_written-
    Mirror) macht der Controller je Schritt.

    Warmup laeuft auf einem THROWAWAY bf16-Cache (eigene GDN-Zustaende,
    eigene cat-Layer) — der Live-Static-Cache bleibt exakt Praefill-
    frisch (pos/_written unveraendert); aufgewaermt werden nur die
    cache-unabhaengigen Workspaces (cuBLAS, Triton-Autotune, T=1-Shapes).
    Der px_capture_guard unterdrueckt waehrend Warmup+Capture Calibrator-
    collect und Host-Telemetrie; die Phi-Ring-Writes des Warmups (auf
    bf16-Cache ohne KV4-Slot auch Device-Ops) werden durch Ring-Reset
    verworfen, damit flush_px_capture NUR die echten Replay-Schritte
    replayt.

    Jeder Fehler beim Capturen (OOM, Sync-Hazard) propagiert — decode_loop
    faellt davon transparent auf Eager zurueck.
    """

    def __init__(self, model, cache, device, verbose=False):
        self.base = _resolve_text_model(model)
        # Modul-Alignment (Defekt 9349-vs-3653): model_manager legt
        # px_patches/ auf sys.path und laedt den Patch als
        # "ternary_bonsai_27b_px.patch" (config.py patch_dir ohne Praefix)
        # — deferred px_patches.*-Imports an dieser Stelle erzeugen ein
        # ZWEITES Modul-Objekt derselben Datei: px_capture_guard setzte
        # active=True im fremden Dict, der gebundene _px_forward sah den
        # Guard nie → else-Zweig baute die Maske auf written+1, waehrend
        # der KV4-Layer unter Capture Full-Capacity-Keys dequantisiert →
        # sdpa-Dim-3-Crash im ersten Decode-Schritt (5942 vs 3893). Auf-
        # loesung immer ueber die bereits geladenen/bundenen Kopien.
        px = self.px = _px_patch_module_for(self.base)
        gf3 = self.gf3 = _loaded_module(
            "ternary_bonsai_27b_px.gf3_quant",
            "px_patches.ternary_bonsai_27b_px.gf3_quant")
        if gf3 is None:
            from px_patches.ternary_bonsai_27b_px import gf3_quant as gf3fb
            gf3 = self.gf3 = gf3fb
        self.model, self.device = model, torch.device(device)
        self.cache = cache
        tm = self.tm = self.base
        px = self.px
        px.ensure_capture_buffers(tm, n_tokens=int(
            os.environ.get("PX_PHI_RING", "4096")), device=self.device)
        tm._px_ring_idx.zero_()            # Ring-Index generationen-frisch
        self.static_tok = torch.zeros(1, 1, dtype=torch.long,
                                      device=self.device)
        vocab = int(model.config.get_text_config().vocab_size)
        self.static_logits = torch.zeros(1, 1, vocab, dtype=torch.float32,
                                         device=self.device)
        self._written0 = int(cache._px_kv4_layer._written)   # T nach Prefill

        # --- Warmup: throwaway Caches, live cache unberuehrt ---------------
        # (a) bf16: cuBLAS/Triton-GEMV + GDN-Decode-Kernels. (b) kv4-Static
        # (klein): KV4-Full-Cap-Dequant + SDPA-maske + patch.py Masken-Branch
        # — T=1/KV4-Formen werden im Produktionsszenario NUR hier warm, denn
        # Prefill compiliert nur Prefill-Shapes und Triton-JIT/-Autotune
        # unter Capture illegal ist (PoC-Befund, Breaker #3).
        warm = make_qwen35_cache(model.config, "bf16")
        warm4 = make_qwen35_cache(model.config, "kv4", static_capacity=4)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad(), px.px_capture_guard():
            for _ in range(3):
                out = self.base(input_ids=self.static_tok,
                                past_key_values=warm, use_cache=True)
                self.static_h = out.last_hidden_state[:, -1:]
            for _ in range(2):
                self.base(input_ids=self.static_tok,
                          past_key_values=warm4, use_cache=True)
        torch.cuda.current_stream().wait_stream(s)
        del warm, warm4
        tm._px_ring_idx.zero_()            # Warmup-Ring-Writes verwerfen
        px._PX_CAPTURE["need_flush"] = False

        # --- Capture: Live-Static-Cache + lm_head + Logit-Buffer ----------
        # Nancheck-Rescue (.item()-Syncs) unter Capture illegal: das betrifft
        # nicht nur gf3_quant, sondern auch den Runtime-eigenen lm_head-Guard
        # (ptq10_matvec head=True → isfinite je Call; PoC-Befund, Breaker
        # #2). Beide Modi nur hier "off", danach restauriert.
        torch.cuda.synchronize()
        nm_old = gf3._NANCHECK_MODE
        try:
            rt = _loaded_module("ternary_bonsai_27b_px.runtime_qwen35_ptq",
                                "px_patches.ternary_bonsai_27b_px"
                                ".runtime_qwen35_ptq")
            if rt is None:
                rt = importlib.import_module(
                    "px_patches.ternary_bonsai_27b_px.runtime_qwen35_ptq")
            nm_old_rt = rt._NANCHECK_MODE
            rt._NANCHECK_MODE = "off"
            gf3._NANCHECK_MODE = "off"
            try:
                g = self.graph = torch.cuda.CUDAGraph()
                with px.px_capture_guard(), torch.no_grad():
                    try:
                        # TT-B2a / Matrix-4-Befund (capture_debug_matrix2/4):
                        # der `torch.cuda.graph(g)`-ctx-Manager invalidiert an
                        # diesem Body (pctx — Fehler schon beim ersten
                        # Body-Kernel, fold/unfold@runtime:278), waehrend
                        # manual capture_begin auf frischem Side-Stream
                        # denselben Body vollstaendig rekordet (debug full3/
                        # mglobal OK). Produktion nutzt die bewiesene
                        # manual-Form: Side-Stream + wait_stream +
                        # capture_begin(explizit "global"). Begin ausserhalb
                        # des inneren try (bei Begin-Fehler laeuft kein
                        # dangling capture_end); Body+End gepaart wie im
                        # ctx-__exit__ (End wirft bei invalidiertem Capture
                        # selbst auf und verdraengt die Ursprungs-Exception —
                        # identisches Verhalten wie torch.cuda.graph).
                        s2 = torch.cuda.Stream()
                        s2.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s2):
                            g.capture_begin(capture_error_mode="global")
                            try:
                                out = self.base(input_ids=self.static_tok,
                                                past_key_values=cache,
                                                use_cache=True)
                                self.static_logits.copy_(model.lm_head(
                                    out.last_hidden_state[:, -1:]).float())
                            finally:
                                g.capture_end()
                        torch.cuda.current_stream().wait_stream(s2)
                    except BaseException:
                        # TT-B2a / Round-4-Befund: ein gescheiterter Capture-
                        # Pass hat Python-Buchhaltung AUSGEFUEHRT (_written += 1
                        # je Static-Layer), waehrend die Device-Ops nur
                        # rekordet statt ausgefuehrt wurden (pos unveraendert).
                        # Ohne Restore laeuft der Eager-Fallback (decode_loop
                        # dg="fail") auf getaintetem Cache (_written=74 vs
                        # pos=73), dequantisiert einen Garbage-Slot — sichtbar
                        # als deterministischer phi-Drift 0.0033 bei
                        # byte-identischer Greedy-Sequenz (PoC-Log poc4).
                        self.reset_written(self._written0)
                        px._PX_CAPTURE["need_flush"] = False
                        raise
            finally:
                rt._NANCHECK_MODE = nm_old_rt
        finally:
            gf3._NANCHECK_MODE = nm_old
        # Der Capture-Pass fuehrt Python-Buchhaltung aus (_written += 1 je
        # Static-Layer), aber KEINE Device-Ops (pos/idx nur rekorded) —
        # Host-Mirror zurueck auf den Praefill-Stand.
        self.reset_written(self._written0)
        self._steps = 0
        if verbose:
            print(f"[graph] decode captured: cap={cache._px_static_capacity} "
                  f"pos0={self._written0}", flush=True)

    def reset_written(self, value):
        for L in self.cache.layers:
            if getattr(L, "static_capacity", 0):
                L._written = int(value)

    def step(self, tok):
        """Ein Graph-Schritt; Rueckgabe Logits (V,) f32 (Buffer-View)."""
        self.static_tok.fill_(int(tok))
        self.graph.replay()
        self._steps += 1
        for L in self.cache.layers:        # Host-Mirror je Replay
            if getattr(L, "static_capacity", 0):
                L._written += 1
        return self.static_logits[0, -1]

    def close(self):
        """Telemetrie/Calibrator der Replay-Schritte aus dem Device-Phi-Ring
        nachtraeglich replayen (TT-B3 flush; der eine Sync ist EOS-eager)."""
        self.px.flush_px_capture(self.tm)


def _loaded_module(*names):
    """Erster in sys.modules bereits geladener Key gewinnt; sonst None.

    Doppelt geladene Modul-Objekte derselben Datei (Server-Key
    "ternary_bonsai_27b_px.*" vs Package-Key "px_patches.
    ternary_bonsai_27b_px.*") sind der Defekt 9349-vs-3653 — daher hier
    nur LESEN, niemals einen zweiten Key erzeugen."""
    for n in names:
        mod = sys.modules.get(n)
        if mod is not None:
            return mod
    return None


def _px_patch_module_for(text_model):
    """Patch-Modul DORT aufloesen, wo der gebundene _px_forward lebt.

    Der gebundene Forward liest _PX_CAPTURE aus seinen Funktions-Globals;
    px_capture_guard / ensure_capture_buffers / flush_px_capture muessen
    in ebendiesem Modul greifen, sonst bleibt unsichtbar, was der Guard
    im anderen Modul-Dict setzt. Fallback (kein gebundener px-Forward
    erkennbar): bereits geladene Kopie; sonst deferred Package-Import.
    """
    fwd = getattr(text_model, "forward", None)
    fn = getattr(fwd, "__func__", fwd)
    glob = getattr(fn, "__globals__", None) if fn is not None else None
    if glob is not None and glob.get("_PX_CAPTURE") is not None:
        name = glob.get("__name__")
        mod = sys.modules.get(name) if name else None
        if (mod is not None
                and getattr(mod, "_PX_CAPTURE", None) is glob["_PX_CAPTURE"]):
            return mod
    mod = _loaded_module("ternary_bonsai_27b_px.patch",
                         "px_patches.ternary_bonsai_27b_px.patch")
    if mod is not None and hasattr(mod, "_PX_CAPTURE"):
        return mod
    return _px_patch_module()


def _px_patch_module():
    """Deferred patch-Import (Vermeidet Zirkelimport beim Modul-Laden)."""
    import importlib
    return importlib.import_module(
        "px_patches.ternary_bonsai_27b_px.patch")


def _decode_graph_enabled(cache):
    """TT-B2a-Gate: Env an, CUDA da, Static-KV4-Cache, nicht B>1."""
    return (os.environ.get("PX_DECODE_GRAPH", "0") == "1"
            and torch.cuda.is_available()
            and int(getattr(cache, "_px_static_capacity", 0)) > 0)


def decode_loop(model, cache, logits, max_new_tokens, temperature=0.7,
                top_k=40, top_p=0.95, repetition_penalty=1.15,
                eos_token_ids=(), streamer=None, max_total_seq=None,
                prompt_uniq=None, no_repeat_ngram_size=0, verbose=False):
    """Autoregression auf existierendem Cache. Rueckgabe: (ids (1, n), steps).

    logits: (1, 1, V) f32 vom Prefill-Ende; der Loop ruft model(...) je
    Token (ForCausalLM.forward -> px-gepatchtes Text-Modell interno).

    TT-B2a: wenn PX_DECODE_GRAPH=1 und ein Static-KV4-Cache vorliegt,
    laeuft der Modell-Schritt als CUDA-Graph-Replay (PX-Hooks inklusive,
    siehe _DecodeGraph); Sampling/Streamer/Warps bleiben host-seitig
    identisch zum Eager-Pfad. Jeder Capture-Fehler (OOM/Hazard) faellt
    transparent auf Eager zurueck.
    """
    gen = []
    cur = logits[0, -1] if logits.ndim == 3 else logits[0]
    steps = 0
    dg = None
    try:
        for step in range(max_new_tokens):
            lg = _apply_logit_warps(cur.clone(), gen, prompt_uniq,
                                    temperature,
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
                    print(f"    Kontext-Limit {max_total_seq} erreicht "
                          f"(step {step})", flush=True)
                break
            if dg is None and _decode_graph_enabled(cache):
                try:
                    dg = _DecodeGraph(model, cache, cur.device,
                                      verbose=verbose)
                    if verbose:
                        print("[long] Decode-Graph aktiv (TT-B2a)",
                              flush=True)
                except Exception as e:
                    # TT-B2a: Capture-Fehler melden async — der Traceback
                    # zeigt, WO der Fehler aufgetreten ist (nicht den Op, der
                    # die Invalidierung verursacht hat). Ohne Frame-Info war
                    # die PoC-Fehlersuche (Rounds 1-6) nicht möglich.
                    traceback.print_exc()
                    print(f"[long] Decode-Graph fallback eager: {e}",
                          flush=True)
                    dg = "fail"        # Sentinel: keine zweiten Versuche
            if dg is not None and dg != "fail":
                cur = dg.step(tok).float()
                continue
            nxt = torch.tensor([[tok]], device=cur.device)
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            cur = out.logits[0, -1].float()
    finally:
        if dg is not None and dg != "fail":
            dg.close()                 # Phi-Flush (TT-B3) nach EOS/Break
    ids = torch.tensor([gen], dtype=torch.long, device=cur.device)
    if streamer is not None:
        streamer.end()
    return ids, steps


# ------------------------------------------------------------ Orchestrator ----
def _static_capacity(T_prompt, max_new_tokens, chunk):
    """TT-B1a: Kapazität der Static-KV4-Buffers, wenn PX_KV_STATIC=1.

    Prompt + genehmigte Generierung + chunk-Puffer; sonst 0 (cat-Pfad).
    """
    if os.environ.get("PX_KV_STATIC", "0") != "1":
        return 0
    total = int(T_prompt) + int(max(max_new_tokens, 0) or 0) \
        + int(chunk or 0) + 1
    return max(total, 1)


def _consume_streamer_skip(streamer):
    """Stream-Skip-Guard konsumieren, bevor der erste Decode-Token fließt.

    Der Long-Pfad pusht NIE die Prompt-Ids in den Streamer (decode_loop
    schreibt nur generierte Tokens, Zeile ~353). Ein frisch konstruierter
    TextIteratorStreamer(skip_prompt=True) (generators.py ~910, chat_tab.
    py ~609) frisst mit next_tokens_are_prompt=True deshalb das ERSTE
    generierte Token — Live-Bug f31eff3e idx17/idx19 ("elen Dank" statt
    "Vielen", " nehme" statt "Ich"; nur Turns mit T > Long-CTX-Schwelle,
    denn der HF-Plain-Pfad konsumiert den Skip selbst via streamer.put(
    input_ids), transformers generation/utils.py:2575-2576).
    """
    if streamer is not None and getattr(streamer,
                                        "next_tokens_are_prompt", None):
        streamer.next_tokens_are_prompt = False


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
    _consume_streamer_skip(streamer)
    if verbose:
        # A2: effektive Decode-Settings ausweisen (Beweisquelle für
        # Repetitions-/Sampling-Diagnosen — Repro f31eff3e/RTPF).
        print(f"[long] Decode-Settings: T={input_ids.shape[1]} "
              f"temperature={s_cfg.get('temperature')} "
              f"top_k={s_cfg.get('top_k', 40)} "
              f"top_p={s_cfg.get('top_p', 0.95)} "
              f"rep_p={s_cfg.get('repetition_penalty', 1.15)} "
              f"ngram={s_cfg.get('no_repeat_ngram_size', 0)}",
              flush=True)
    cache = make_qwen35_cache(model.config, kv_mode,
                              static_capacity=_static_capacity(
                                  input_ids.shape[1], max_new_tokens,
                                  chunk))
    _sc = getattr(cache, "_px_static_capacity", 0)
    if verbose and _sc:
        print(f"[long] KV-static: capacity={_sc}", flush=True)
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