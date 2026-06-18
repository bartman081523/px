"""Probe flex_attention on base gemma3-1b-it at 16k. Lossless (full causal) + memory-efficient (tiled)."""
import torch, time, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flex_attention")
tc = model.config.text_config if hasattr(model.config, "text_config") else model.config
print("attn_impl:", getattr(tc, "_attn_implementation", "?"))

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
    except Exception as e:
        print(f"T={T}: ERR {type(e).__name__}: {str(e)[:160]}")
        torch.cuda.empty_cache()