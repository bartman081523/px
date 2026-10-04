#!/usr/bin/env python
"""Stufe-3f-Diagnose am 27b: (A) Kernel-Konstanten, (B) Matvec-A/B mit und
ohne pro-Call-isfinite-Sync, (C) chunked Prefill kv4/bf16 (3 Chunks, sdpa-Spy
+ KV-Slots + get_mask_sizes-Trace → 6144/4096-Crash-Chunks), (D) Decode-M=1-
Timing ohne/mit Guard.

Ein Modell-Load (~10-15 s), alles danach warm. Kein Netz, kein HF generate.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

HF_DIR = "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf"

import torch                                                     # noqa: E402

RT = __import__("px_patches.ternary_bonsai_27b_px.runtime_qwen35_ptq",
                fromlist=("x"))
GF3 = __import__("px_patches.ternary_bonsai_27b_px.gf3_quant", fromlist=("x"))
LC = __import__("px_patches.ternary_bonsai_27b_px.long_context",
                fromlist=("x"))
PATCH = __import__("px_patches.ternary_bonsai_27b_px.patch", fromlist=("x"))

print("== A. Kernel-Konstanten ==", flush=True)
for n in ("_TRITON", "_MMAX", "_RC", "_BLOCK_R", "_PART_BYTES", "ND", "RPB"):
    print(f"   {n} = {getattr(GF3, n, '<fehlt>')!r}", flush=True)

# ---------------------------------------------------------------- sdpa-Spy ---
import transformers.integrations.sdpa_attention as SDPA            # noqa
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS    # noqa

_orig_sdpa = SDPA.sdpa_attention_forward
_calls = {"n": 0}


def _spy(module, q, k, v, mask, **kw):
    """Druckt Prefill-Shape-Zeilen (ersten 96 Calls) + CRASH-Zeile + reraise."""
    n = _calls["n"]
    _calls["n"] += 1
    mn = getattr(module, "layer_idx", "?")
    if (n < 16) or (q.shape[2] > 1 and n < 200):
        try:
            print(f"   [sdpa #{n}] L{mn}: q{tuple(q.shape)} k{tuple(k.shape)} "
                  f"v{tuple(v.shape)} "
                  f"mask={tuple(mask.shape) if mask is not None else None} "
                  f"causal={kw.get('is_causal', '?')}", flush=True)
        except Exception:
            pass
    try:
        return _orig_sdpa(module, q, k, v, mask, **kw)
    except Exception as exc:
        print(f"   [sdpa CRASH #{n}] L{mn} q{tuple(q.shape)} k{tuple(k.shape)} "
              f"v{tuple(v.shape)} "
              f"mask={tuple(mask.shape) if mask is not None else None}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        raise


SDPA.sdpa_attention_forward = _spy
try:
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _spy
    print("   [spy] Registry gepatcht", flush=True)
except Exception as exc:
    print(f"   [spy] Registry-Patch ueber: {exc!r}", flush=True)

# ------------------------------------------------------------------- Load ----
print("== Load ==", flush=True)
t0 = time.perf_counter()
fold = RT.FoldOps(RT.load_signs()[0], mode="signs_first")
model = GF3.build_and_load_gf3(fold, hf_dir=HF_DIR)
PATCH.apply_px_patch(model, "ACTIVE_MANIFOLD")
print(f"   geladen + gepatcht in {time.perf_counter()-t0:.0f}s "
      f"(VRAM {torch.cuda.memory_allocated()/2**30:.2f} GiB)", flush=True)

# ------------------------------------------------------------ B. Matvec A/B --
print("== B. Matvec-Bench (warm, f32-M=1) ==", flush=True)
shaped = {}
for name, mod in model.named_modules():
    if isinstance(mod, GF3.GF3Linear):
        shaped.setdefault(f"{mod.out_features}x{mod.in_features}", name)


def _bench(lin, n=12):
    x = torch.randn(1, lin.in_features, device=lin.u.device,
                    dtype=torch.bfloat16)
    for _ in range(3):
        lin(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        lin(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


_orig_safe = GF3.gf3_matvec_safe


def _safe_fast(u, s, x, m_out, K, head=False):
    """Kopie von gf3_matvec_safe OHNE pro-Call-isfinite-Sync."""
    in_shape = x.shape
    x2 = x.reshape(-1, in_shape[-1])
    M = x2.shape[0]
    if not GF3._TRITON or M > GF3._MMAX:
        return GF3._gf3_deq_matvec(u, s, x2, in_shape, m_out, K)
    y = GF3._gf3_kernel_matvec(u, s, x2.to(torch.float32), m_out, K)
    return y.reshape(*in_shape[:-1], m_out).to(x.dtype)


for tag in sorted(shaped):
    mod = model.get_submodule(shaped[tag])
    t_guard = _bench(mod)
    GF3.gf3_matvec_safe = _safe_fast
    t_noguard = _bench(mod)
    GF3.gf3_matvec_safe = _orig_safe
    print(f"   {tag}: guard {t_guard*1e3:.2f} ms/call -> ohne isfinite "
          f"{t_noguard*1e3:.2f} ms/call", flush=True)

GF3.gf3_matvec_safe = _safe_fast    # ab hier ohne Guard weitermessen
print("   [ab jetzt: gf3_matvec_safe ohne pro-Call-isfinite]", flush=True)

# ---------------------------------------------- get_mask_sizes-Instrument ----
from transformers.cache_utils import CacheLayerMixin               # noqa

_kv4cls = getattr(LC, "QuantKV4Layer", None)
_gms_orig = None
if _kv4cls is not None and hasattr(_kv4cls, "get_mask_sizes"):
    _gms_orig = _kv4cls.get_mask_sizes

    def _gms(self, query_length):
        r = _gms_orig(self, query_length)
        li = getattr(self, "_layer_idx", getattr(self, "layer_idx", "?"))
        print(f"   [gms] kv4 L{li} S={self.get_seq_length()} "
              f"q={query_length} -> {r}", flush=True)
        return r

    _kv4cls.get_mask_sizes = _gms


def _dump(cache, tag):
    try:
        kv4 = [i for i, l in enumerate(cache.layers)
               if isinstance(l, LC.QuantKV4Layer)]
        seqs = " ".join(f"{i}:{cache.get_seq_length(i)}" for i in kv4)
        kinds = {}
        for l in cache.layers:
            kinds[type(l).__name__] = kinds.get(type(l).__name__, 0) + 1
        first_ccl = next((i for i, l in enumerate(cache.layers)
                          if isinstance(l, CacheLayerMixin)), "kein")
        print(f"   [{tag}] CacheLayerMixins: {kinds}, erster@{first_ccl}, "
              f"global_seq={cache.get_seq_length()}", flush=True)
        print(f"   [{tag}] kv4-Slots: {seqs}", flush=True)
    except Exception as exc:
        print(f"   [{tag}] dump-Fehler: {exc!r}", flush=True)

# --------------------------------------------------------- C. Chunked Prefill
ids = torch.randint(0, 240000, (1, 2048), device=model.device)
base = LC._resolve_text_model(model)
print(f"== C. chunked Prefill 3×2048 == (Text-Modell {type(base).__name__})",
      flush=True)


def _three_chunks(cache, mode):
    with torch.no_grad():
        for ch in range(3):
            _dump(cache, f"{mode} chunk{ch+1} vor")
            t = time.perf_counter()
            try:
                out = base(input_ids=ids, past_key_values=cache, use_cache=True)
            except Exception:
                import traceback
                traceback.print_exc()
                print(f"   {mode} chunk{ch+1} >>> CRASH <<< nach "
                      f"{time.perf_counter()-t:.1f}s", flush=True)
                return False
            print(f"   {mode} chunk{ch+1} OK ({time.perf_counter()-t:.1f}s)",
                  flush=True)
            _dump(cache, f"{mode} chunk{ch+1} nach")
    return True


cache = LC.make_qwen35_cache(model.config, "kv4")
kv4_ok = _three_chunks(cache, "kv4")
print(f"== C2. bf16-Isolation ==" if not kv4_ok else "== C2. bf16-Vergleich ==",
      flush=True)
cache_b = LC.make_qwen35_cache(model.config, "bf16")
bf16_ok = _three_chunks(cache_b, "bf16")

# --------------------------------------------------- D. Decode M=1-Timing ----
print("== D. Decode 10 Steps (M=1) ==", flush=True)
use_cache = cache_b if (not kv4_ok and bf16_ok) else cache
try:
    with torch.no_grad():
        tok = torch.randint(0, 240000, (1, 1), device=model.device)
        for _ in range(2):            # warm
            out = base(input_ids=tok, past_key_values=use_cache, use_cache=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            out = base(input_ids=tok, past_key_values=use_cache, use_cache=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 10
        print(f"   OHNE Guard: {dt*1e3:.0f} ms/Token ({1/dt:.2f} tok/s)",
              flush=True)
        GF3.gf3_matvec_safe = _orig_safe
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            out = base(input_ids=tok, past_key_values=use_cache, use_cache=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 10
        print(f"   MIT Guard:  {dt*1e3:.0f} ms/Token ({1/dt:.2f} tok/s)",
              flush=True)
        GF3.gf3_matvec_safe = _safe_fast
except Exception:
    import traceback
    traceback.print_exc()

print("OK: Diagnose abgeschlossen", flush=True)