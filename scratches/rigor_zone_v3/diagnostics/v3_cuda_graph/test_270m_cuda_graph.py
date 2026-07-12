"""Test: CUDA-Graph für 270m generate. RTX 2060 (Turing, CC 7.5).

Strategie: 
- Statische Shape: (B, 1) input_ids
- Pre-capturiere KV-Cache updates
- Replay in Python-Loop ohne Python-Dispatch
"""
import os
os.environ["PYTHONPATH"] = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
import sys
import time
import subprocess
import threading
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-3-270m-it"
device = "cuda"
print(f"Lade {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device).eval()
print(f"  loaded, {sum(p.numel() for p in m.parameters())/1e6:.0f}M params")

# CUDA-Graph-Capture eines einzelnen Decoding-Steps (T=1, B=8)
# Pattern: NVIDIA RNN-T / PyTorch make_graphed_callables
print("\n=== Setup CUDA-Graph für Decode-Step ===")
B, T = 8, 1
H = m.config.hidden_size
V = m.config.vocab_size
n_layers = m.config.num_hidden_layers
n_heads = m.config.num_attention_heads
n_kv = m.config.num_key_value_heads
head_dim = H // n_heads

# Static input: token IDs der Länge 1
# Wir allokieren fixe KV-Cache pro Layer: (B, n_kv, MAX_SEQ, head_dim)
MAX_SEQ = 256
print(f"  Static: B={B}, T={T}, MAX_SEQ={MAX_SEQ}, n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}")

# 1) Prefill: Token 0-15 in den Cache, dann 16: static cache
tokenizer = tok
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
prompts = [f"What is 2+{i}?" for i in range(B)]
tpl = [tokenizer.apply_chat_template(
    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
) for p in prompts]
enc = tokenizer(tpl, padding="longest", return_tensors="pt", add_special_tokens=False)
ids = enc["input_ids"].to(device)
am = enc["attention_mask"].to(device)
print(f"  prompt shape: {ids.shape}")

# Prefill
print("\nPrefill...")
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    out = m.generate(input_ids=ids, attention_mask=am, max_new_tokens=2, do_sample=False, use_cache=True)
dt = time.perf_counter() - t0
print(f"  prefill+2 tokens: {dt*1000:.0f}ms")

# 2) Bauen wir den CUDA-Graph für den DECODE-STEP allein
# Pattern: stream-capture mit fixen tensor-pointer
# Wir hooken uns in m.model.forward (textModel forward) und capturen das

# Allocate static input
static_input_ids = torch.zeros((B, T), dtype=torch.long, device=device)
static_pos_ids = torch.zeros((B, T), dtype=torch.long, device=device)
# Attention mask muss 4D sein für SDPA
# Allocate static KV cache buffers per layer
print("\nBuilding static KV cache...")

# Approach: Use m.model directly with our own static-cache implementation
# Actually, easier: use generate with past_key_values already populated
# und mache per-step die Python-Loop

# Test first: pure m.generate baseline (no cuda graph)
print("\n=== Baseline m.generate bs=8, 50 tokens ===")
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    out = m.generate(input_ids=ids, attention_mask=am, max_new_tokens=50, do_sample=False, use_cache=True)
dt = time.perf_counter() - t0
print(f"  generate: {dt*1000:.0f}ms = {50/dt:.1f} tok/s = {50/(dt*B):.1f} tok/s/seq")

# Approach 2: Single text-model forward (no LM head), pro token
print("\n=== Pure forward B=8 T=1, ohne generate wrapper ===")
# Setup: muss den past-key-values populated haben
# Erst Prefill mit generate, dann holen wir past
with torch.inference_mode():
    out_prefill = m.generate(input_ids=ids, attention_mask=am, max_new_tokens=1, do_sample=False, use_cache=True, return_dict_in_generate=True)
past = out_prefill.past_key_values
print(f"  prefill+1 done, past type={type(past).__name__}")
# Decode-step
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(50):
        next_input = out_prefill.sequences[:, -1:]  # (B, 1)
        o = m(input_ids=next_input, past_key_values=past, use_cache=True)
        next_token = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out_prefill.sequences = torch.cat([out_prefill.sequences, next_token], dim=1)
dt = time.perf_counter() - t0
print(f"  pure forward 50x: {dt*1000:.0f}ms = {50/dt:.1f} tok/s = {50/(dt*B):.1f} tok/s/seq")

# Approach 3: CUDA-Graph für den Forward-Step
print("\n=== CUDA-Graph capture: 1 Decode-Step B=8 T=1 ===")
# Wir capturen die next_token = m(input).logits.argmax() Pipeline
# Static input buffer (muss befüllt werden vor replay)
static_input = torch.zeros((B, T), dtype=torch.long, device=device)

# Warmup runs
print("  warmup 3x...")
with torch.inference_mode():
    for _ in range(3):
        # fresh past each time
        with torch.no_grad():
            o = m(input_ids=static_input, past_key_values=past, use_cache=True)
            next_token = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)

# Capture
print("  capturing graph...")
g = torch.cuda.CUDAGraph()
# Static input/output buffers
static_out_logits = torch.zeros((B, T, m.config.vocab_size), dtype=torch.bfloat16, device=device)
# Wir allokieren den past-key-values als static. ABER: m.update() returned neue tensors
# die nicht in der Graph-Region sein können. Wir müssen PyTorch's Cache-API nutzen
# die statisch allokiert.
# Alternative: Wir forwarden ohne past, also T=prefill_len, was aber cuda-graph bricht.
# Daher: hack — wir bauen ein nn.Module das past in-place mutiert.

# Simpler approach: capture the per-step call on a FRESH past per replay
# (will be slow due to cache updates, but let's see the dispatch overhead)
torch.cuda.synchronize()
try:
    with torch.inference_mode():
        with torch.cuda.graph(g):
            o = m(input_ids=static_input, past_key_values=past, use_cache=True)
            next_token = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    print(f"  graph captured! size: {g.pool().__sizeof__() if hasattr(g, 'pool') else '?'}")
    
    # Replay
    print("  replay 50x...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(50):
            g.replay()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  graph replay 50x: {dt*1000:.0f}ms = {50/dt:.1f} replay/s = {50/dt*B:.1f} tok/s")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
