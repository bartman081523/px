#!/usr/bin/env python
"""VRAM-Audit ternary-bonsai-27b: Wo liegen die GiB (named params+buffers),
Peak nach Load, Peak-Delta bei kurzem Generate. Laeuft baseline (vor Edits)
und nach den VRAM-Edits (lm_head packed + row-chunked _mv_dequant) erneut.
Exit 0 immer (Mess-Skript)."""
import sys
import time

ROOT = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/px_patches")

import torch
from transformers import AutoTokenizer

import ternary_bonsai_27b_px.runtime_qwen35_ptq as rt


def main():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pre = torch.cuda.memory_allocated() / 2**30

    signs, _ = rt.load_signs()
    fold = rt.FoldOps(signs, mode="signs_first")
    model = rt.build_and_load(fold, verbose=False)
    loaded = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[audit] pre={pre:.3f} | after-load alloc={loaded:.2f} GiB | "
          f"load-peak={peak:.2f} GiB", flush=True)

    tens = []
    for n, p in model.named_parameters():
        tens.append((p.numel() * p.element_size(), "param", n))
    for n, b in model.named_buffers():
        tens.append((b.numel() * b.element_size(), "buf", n))
    tens.sort(key=lambda t: -t[0])
    tot = sum(t[0] for t in tens)
    print(f"[audit] named tensors total: {tot / 2**30:.2f} GiB", flush=True)
    for sz, kind, n in tens[:12]:
        print(f"[audit]   {sz / 2**30:6.3f} GiB {kind}  {n}", flush=True)
    rest = tot - sum(t[0] for t in tens[:12])
    print(f"[audit]   (Rest ~{rest / 2**30:.3f} GiB ueber {len(tens) - 12} Tensoren)",
          flush=True)

    tok = AutoTokenizer.from_pretrained(rt.OUT_DIR, fix_mistral_regex=True)
    msgs = [{"role": "user", "content": "Hello."}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=24,
                             pad_token_id=tok.eos_token_id if tok.eos_token_id else 0)
    dt = time.time() - t0
    n = out.shape[1] - ids.shape[1]
    print(f"[audit] generate 24max: {n} tok in {dt:.1f}s ({dt / max(n, 1):.3f}s/t) | "
          f"gen-peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    tail = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"[audit] tail: {tail[-90:]!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())