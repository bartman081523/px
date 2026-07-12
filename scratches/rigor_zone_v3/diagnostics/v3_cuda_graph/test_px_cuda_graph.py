"""Test: PX-Forward in CUDA-Graph.
Wir laden das Model, wenden den PX-Patch an, und capturen den Decode-Step.
"""
import os
os.environ["PYTHONPATH"] = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
import sys
import time
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from px_patches_v3.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from px_patches_v3 import patch as v3_patch

MODEL = "google/gemma-3-270m-it"
device = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

# Apply PX patch (ACTIVE_MANIFOLD, T=1-Skip)
v3_patch.apply_px_patch(m, "ACTIVE_MANIFOLD")
print("PX-Patch applied")

# Setup runner (use a smaller MAX_SEQ since we're capturing PX forward)
prompts = [f"What is 2+{i}?" for i in range(8)]
tpl = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
enc = tok(tpl, padding="longest", return_tensors="pt", add_special_tokens=False)
ids = enc["input_ids"].to("cuda")
am = enc["attention_mask"].to("cuda")

try:
    cfg = CUDAGraphRunnerConfig(batch_size=8, max_seq_len=64)
    runner = CUDAGraphRunner(m, cfg)
    first = runner.setup(ids, am)
    print(f"Setup OK, first_token shape={first.shape}")
    # 20 Replays
    t0 = time.time()
    for _ in range(20):
        nxt = runner.step()
        runner.append(nxt)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"20 Replays: {dt*1000:.0f}ms = {20/dt:.1f} steps/s = {20*8/dt:.0f} tok/s")
    # Decode
    tokens = [first]
    for _ in range(20):
        nxt = runner.step()
        runner.append(nxt)
        tokens.append(nxt)
    out = torch.cat(tokens, dim=1)
    text = tok.batch_decode(out, skip_special_tokens=True)
    print("Generated:")
    for t in text[:3]:
        print(f"  '{t[:80]}'")
except Exception as e:
    import traceback
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
