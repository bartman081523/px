"""B2 — Chunked Prefill für gemma-4 (eigener, minimaler Loop).

Idee: statt eines Prefills über T Tokens (Maske T×T) wird der Prompt in
Chunks der Größe C geschoben; jeder Chunk sieht den akkumulierten KV-Cache.
Die Maske pro Chunk ist dann (C, T_so_far) statt (T, T).

Gemessen wird: Peak-VRAM, Zeit, Needle-Recall — bei 8k / 16k / 32k.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("GF5_DIR", "."))
from transformers import AutoTokenizer  # noqa: E402

FILLER = ("The archive contains routine maintenance notes about ventilation, "
          "lighting schedules and inventory counts. ")
NEEDLE = "The secret maintenance code is KESTREL-7741."


def chunked_prefill(model, ids, chunk=512):
    """Schiebt ids in Chunks durch das Modell; gibt Logits der letzten Position."""
    T = ids.shape[1]
    past = None
    pos = 0
    out = None
    ctx = torch.no_grad()      # ohne no_grad halten die Chunks ihre Autograd-Graphen
    ctx.__enter__()
    while pos < T:
        end = min(T, pos + chunk)
        chunk_ids = ids[:, pos:end]
        cache_position = torch.arange(pos, end, device=ids.device)
        out = model(input_ids=chunk_ids, past_key_values=past, use_cache=True,
                    cache_position=cache_position, logits_to_keep=1
                    if end == T else 0)
        past = out.past_key_values
        pos = end
        if end < T:
            del out
            out = None
            torch.cuda.empty_cache()
    ctx.__exit__(None, None, None)
    return out.logits[:, -1:, :], past


def main(model_loader, tok, lengths, chunk=512, max_new=16):
    for chars in lengths:
        doc = (FILLER * (chars // len(FILLER) + 1))[:chars]
        pos = len(doc) // 2
        text = (doc[:pos] + "\n\n" + NEEDLE + "\n\n" + doc[pos:]
                + "\n\nReply with the secret code only.")
        ids = tok(text, return_tensors="pt").to("cuda")
        T = ids["input_ids"].shape[1]
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            t0 = time.time()
            with torch.no_grad():
                logits, past = chunked_prefill(model, ids["input_ids"], chunk)
            t_pre = time.time() - t0
            # Greedy-Decode aus dem Cache
            gen = [int(logits[0, -1].argmax())]
            eos = tok.eos_token_id
            for _ in range(max_new - 1):
                step = torch.tensor([[gen[-1]]], device="cuda")
                cp = torch.arange(T + len(gen) - 1, T + len(gen), device="cuda")
                with torch.no_grad():
                    o = model(input_ids=step, past_key_values=past, use_cache=True,
                              cache_position=cp)
                past = o.past_key_values
                nxt = int(o.logits[0, -1].argmax())
                gen.append(nxt)
                if nxt == eos:
                    break
            dt = time.time() - t0
            ans = tok.decode(gen, skip_special_tokens=True)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"T={T:6d} chunk={chunk}: prefill {t_pre:6.1f}s total {dt:6.1f}s "
                  f"peak={peak:5.2f}GB hit={'7741' in ans} -> {ans[:24]!r}", flush=True)
            del logits, past
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"T={T:6d} chunk={chunk}: ERROR {type(e).__name__}: {str(e)[:110]}",
                  flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    from gf5_runtime import load_gf5_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get("GF5_DIR", "."))
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[60_000, 130_000, 260_000])
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.dir)
    t0 = time.time()
    model = load_gf5_model(a.dir, device="cuda", dtype=torch.float16)
    print(f"[load] {time.time()-t0:.0f}s | VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB",
          flush=True)
    main(None, tok, a.lengths, chunk=a.chunk)
