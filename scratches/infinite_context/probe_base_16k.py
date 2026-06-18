"""Decisive probe: does the BASE gemma3-1b-it (no PX patch) OOM on a 16k prefill?
And which SDPA backend is actually used? Separates 'flash not engaging' vs 'PX recursion'.
"""
import torch, time, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
from transformers import AutoModelForCausalLM, AutoTokenizer

print("SDPA backends:", "flash=", torch.backends.cuda.flash_sdp_enabled(),
      "mem_eff=", torch.backends.cuda.mem_efficient_sdp_enabled(),
      "math=", torch.backends.cuda.math_sdp_enabled(),
      "flex=", getattr(torch.backends.cuda, 'flex_attention_sdp_enabled', lambda: '?')())

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16, device_map="cuda")
tc = model.config.text_config if hasattr(model.config, "text_config") else model.config
print("attn_impl:", getattr(tc, "_attn_implementation", "?"), "head_dim:", getattr(tc, "head_dim", "?"),
      "num_attention_heads:", tc.num_attention_heads, "sliding_window:", getattr(tc, "sliding_window", None))

for T in (2000, 8000, 16000):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    ids = torch.randint(10, tok.vocab_size, (1, T), device="cuda")
    try:
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=4, do_sample=False)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"T={T}: OK peak={peak:.2f}GB time={dt:.1f}s out_len={out.shape[1]}")
    except torch.OutOfMemoryError as e:
        free, total = torch.cuda.mem_get_info()
        print(f"T={T}: OOM (free={free/1e9:.2f}GB) {str(e)[:120]}")
        torch.cuda.empty_cache()