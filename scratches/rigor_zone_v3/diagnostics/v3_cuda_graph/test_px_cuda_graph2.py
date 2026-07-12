"""Debug: was passiert im CUDA-Graph-Setup."""
import os
os.environ["PYTHONPATH"] = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin"
import sys
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from px_patches_v3.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from px_patches_v3 import patch as v3_patch

MODEL = "google/gemma-3-270m-it"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"
v3_patch.apply_px_patch(m, "ACTIVE_MANIFOLD")
print("Auto-detected config:")
print(f"  cfg: {CUDAGraphRunnerConfig()}")
prompts = [f"What is 2+{i}?" for i in range(8)]
tpl = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
enc = tok(tpl, padding="longest", return_tensors="pt", add_special_tokens=False)
ids = enc["input_ids"].to("cuda")
am = enc["attention_mask"].to("cuda")
print(f"Prompt shape: {ids.shape}")
cfg = CUDAGraphRunnerConfig(batch_size=8, max_seq_len=64)
runner = CUDAGraphRunner(m, cfg)
print(f"After init:")
print(f"  cfg.n_layers={runner.cfg.n_layers}, head_dim={runner.cfg.head_dim}, n_kv={runner.cfg.n_kv_heads}")
print(f"  layer_types: {runner._layer_types[:5]}")
print(f"  expected head_dim 256, got {runner.cfg.head_dim}")

# Pre-call (PX prefill via m.model)
from transformers.cache_utils import DynamicCache
past = DynamicCache(config=m.config)
with torch.inference_mode():
    o = runner.text_model(input_ids=ids, attention_mask=am, past_key_values=past, use_cache=True)
print(f"After PX-prefill:")
print(f"  past.layers[0].keys.shape: {past.layers[0].keys.shape}")
print(f"  past.get_seq_length(): {past.get_seq_length()}")
