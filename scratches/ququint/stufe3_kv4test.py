#!/usr/bin/env python
"""Stufe-3d: KV4-Cache + Chunked-Prefill — Tiny-Modell-Paritaetstest.

TINY-Qwen3_5 (4 Layer = 2x linear_attention + 2x full_attention). Beweise:
  1. kv4_pack -> kv4_dequant: |err| <= s/2 je Block        (Roundtrip-Bound)
  2. chunked_prefill (bf16-Cache, chunk 16, T=37) == single-pass Logits
     (bf16-tolerant, T nicht chunk-teilig)
  3. KV4-Swap: _px_n_kv4_layers == 2; get_seq_length()-Auto-Skip ueber
     layer 0 (linear, kein CacheLayerMixin) -> 37; Logits ~ bf16-Cache
     (KV-Quant-Rauschen, grosszuegig, SMOKE-Level)
  4. decode_loop: deterministisch geseedet — ids in Vocab, Cache waechst
     je Token; max_total_seq-Break; EOS-Break via One-hot-Logits
  5. crop + reset: crop(20)/crop(-5), Prefill nach crop, reset -> exakte
     Wiederholbarkeit vs frischer Caches
  6. px-Fusion: apply_px_patch(ACTIVE_MANIFOLD) + chunked_prefill durch
     px_forward mit KV4-Cache + generate_long End-to-End + px-Metriken
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from px_patches.ternary_bonsai_27b_px import long_context as LC  # noqa: E402

torch.manual_seed(123)
dev = "cuda"

# ---------------------------------------------------- 1. Roundtrip-Bound -----
x = (torch.randn(2, 4, 37, 128, dtype=torch.float32, device=dev) * 3
     ).to(torch.bfloat16)
u8, s = LC.kv4_pack(x)
k = LC.kv4_dequant(u8, s, x.dtype)
err = (k.float() - x.float()).abs().max().item()
bound = (s.float().max() / 2).item()
print(f"1. kv4-Roundtrip : max|err|={err:.6g} , s-max/2={bound:.6g}")
assert err <= bound * 1.05, "KV4-Roundtrip ueberschreitet s/2-Bound"

# ------------------------------------------------ 2. TINY-Modell + Paritaet --
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

cfg = Qwen3_5TextConfig(
    vocab_size=101, hidden_size=64, intermediate_size=96,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, linear_conv_kernel_dim=4, linear_key_head_dim=32,
    linear_value_head_dim=32, linear_num_key_heads=2,
    linear_num_value_heads=2, max_position_embeddings=512,
    tie_word_embeddings=False,
    layer_types=["linear_attention", "full_attention",
                 "linear_attention", "full_attention"])
model = Qwen3_5ForCausalLM(cfg).to(dev, torch.bfloat16).eval()
ids = torch.randint(0, 101, (1, 37), device=dev)

out = model(input_ids=ids, use_cache=True)
lg_full = out.logits.float()                                  # (1, 37, 101)

cache_b = LC.make_qwen35_cache(cfg, kv_mode="bf16")
lg_b = LC.chunked_prefill(model, ids, cache_b, chunk=16)
d_b = (lg_b - lg_full[:, -1:]).abs().max().item()
print(f"2. chunked vs single (bf16): max|d| = {d_b:.6g}")
assert d_b <= 0.1, f"chunked-Prefill divergiert ({d_b})"

# -------------------------------------------------------------- 3. KV4-Swap --
cache4 = LC.make_qwen35_cache(cfg, kv_mode="kv4")
assert cache4._px_n_kv4_layers == 2, "Swap-Zaehl"
from transformers.cache_utils import DynamicLayer  # noqa: E402
assert isinstance(cache4.layers[1], LC.QuantKV4Layer)
assert isinstance(cache4.layers[3], LC.QuantKV4Layer)
assert not isinstance(cache4.layers[0], LC.QuantKV4Layer)
lg4 = LC.chunked_prefill(model, ids, cache4, chunk=16)
seq = cache4.get_seq_length()                       # Auto-Skip ueber layer 0
d_4 = (lg4 - lg_b).abs().max().item()
print(f"3. KV4-Cache     : seq={seq} (Auto-Skip), max|d| vs bf16 = {d_4:.6g}")
assert seq == 37, "get_seq_length-Auto-Skip"
assert cache4.layers[3].get_seq_length() == 37
assert cache4.layers[1].get_max_length() == -1
assert cache4.layers[3].keys is None, "kein bf16-Halten!"
assert d_4 <= 2.0, f"KV4-Rauschen ausser SMOKE-Bound ({d_4})"

# ----------------------------------------------------------- 4. decode_loop --
torch.manual_seed(42)
ids_dec, steps = LC.decode_loop(model, cache4, lg4, 12, temperature=0.0,
                                top_k=0, top_p=1.0, repetition_penalty=1.0)
seq_after = cache4.get_seq_length()
print(f"4. decode_loop   : steps={steps}, ids {tuple(ids_dec.shape)}, "
      f"seq {seq} -> {seq_after}")
assert steps == 12 and ids_dec.shape == (1, 12)
assert int(ids_dec.min()) >= 0 and int(ids_dec.max()) < 101
assert seq_after == 37 + 12 and torch.isfinite(ids_dec.float()).all()

# max_total_seq-Break
cache_x = LC.make_qwen35_cache(cfg, kv_mode="kv4")
lgx = LC.chunked_prefill(model, ids, cache_x, chunk=16)
torch.manual_seed(43)
_, steps_x = LC.decode_loop(model, cache_x, lgx, 10, temperature=0.0,
                            top_k=0, top_p=1.0, repetition_penalty=1.0,
                            max_total_seq=40)
print(f"   max_total_seq  : break bei step {steps_x} "
      f"(seq {cache_x.get_seq_length()})")
# Break VOR dem Forward des letzten Tokens: je Step wächst die Cache erst
# im nachgelagerten Forward — seq = 37 + (steps - 1) = 39 bei steps 3.
assert steps_x <= 3 and cache_x.get_seq_length() == 37 + steps_x - 1

# EOS-Break via One-hot-Logits (deterministisch)
cache_e = LC.make_qwen35_cache(cfg, kv_mode="kv4")
lg_e = LC.chunked_prefill(model, ids, cache_e, chunk=16)
lg_e[0, 0, :] = -1e4
lg_e[0, 0, 50] = 1e4
_, steps_e = LC.decode_loop(model, cache_e, lg_e, 8, temperature=0.0,
                            top_k=0, top_p=1.0, repetition_penalty=1.0,
                            eos_token_ids=(50,))
print(f"   EOS-Break      : steps={steps_e}, seq {cache_e.get_seq_length()}")
assert steps_e == 1 and cache_e.get_seq_length() == 37

# ---------------------------------------------------------- 5. crop + reset --
cache_e.crop(20)
assert cache_e.get_seq_length() == 20
cache_e.crop(-5)
assert cache_e.get_seq_length() == 15
LC.chunked_prefill(model, ids[:, :8], cache_e, chunk=16)
assert cache_e.get_seq_length() == 23
cache_e.reset()
assert cache_e.get_seq_length() == 0
lg_r = LC.chunked_prefill(model, ids, cache_e, chunk=16)
same = torch.equal(lg_r, lg4)
print(f"5. crop + reset  : 20/15/23 -> 0, frischer Prefill == Original: "
      f"{same}")
assert same, "reset-Prefill nicht reproduzierbar"

# ------------------------------------------------------------ 6. px-Fusion ---
from px_patches.ternary_bonsai_27b_px.patch import (  # noqa: E402
    apply_px_patch, get_px_metrics)

ok = apply_px_patch(model, "ACTIVE_MANIFOLD", relay_layer=2, relay_sign=0,
                    n_loops=4, recur_start=1, recur_end=2, bimodal_hub=2)
assert ok, "apply_px_patch lieferte False"
cache_p = LC.make_qwen35_cache(cfg, kv_mode="kv4")
lgp = LC.chunked_prefill(model, ids, cache_p, chunk=16)
d_p = (lgp - lg_b).abs().max().item()
assert cache_p.get_seq_length() == 37
assert d_p <= 2.0, f"px_forward mit KV4-Cache divergiert ({d_p})"
gen = LC.generate_long(model, ids, max_new_tokens=8, chunk=16, kv_mode="kv4")
assert gen["steps"] == 8 and gen["kv_mode"] == "kv4"
assert gen["cache"].get_seq_length() == 37 + 8
mm = get_px_metrics(model)
print(f"6. px-Fusion     : patch OK, chunked seq=37 (max|d|={d_p:.6g}), "
      f"generate_long steps={gen['steps']}")
print(f"   px-Metriken   : {sorted(str(k) for k in mm)[:8]}")

print("OK: KV4-Cache + Chunked-Prefill + decode_loop + px-Fusion paritaetisch")