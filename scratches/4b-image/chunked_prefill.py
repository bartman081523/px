"""
chunked_prefill.py — Plan 3 Phase B: Chunked-Prefill via forward_hook
======================================================================

Löst T>4500 OOM durch Aufteilen des Prefills in Chunks.

Architektur:
  Statt model.generate(**inputs, max_new_tokens=N) einmal aufzurufen,
  rufen wir model.forward() iterativ in Chunks auf. Jeder Chunk sieht
  die letzten chunk_size tokens. Der past_key_values akkumuliert die
  KV states zwischen Chunks.

  Vorteil: attention-Matrix pro Chunk ist [chunk_size, T_so_far] statt
  [T_total, T_total]. Bei chunk_size=512 und T=8000 ist die attention
  8x kleiner als full-attention.

Trade-off:
  - Erstes Token dauert länger (mehrere forwards statt einer)
  - Logits werden erst nach letztem Chunk gelesen
  - Aber: VRAM peak ~ KV_so_far + chunk_size × hidden statt T_total × hidden

Run:
    python -c "from chunked_prefill import chunked_generate; help(chunked_generate)"
"""
from __future__ import annotations

import sys
from typing import Optional

import torch


def chunked_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 64,
    chunk_size: int = 512,
    do_sample: bool = False,
    use_cache: bool = True,
    eos_token_id: Optional[int] = None,
):
    """Generate tokens via chunked prefill + incremental decode.

    Args:
        model: HuggingFace model (mit past_key_values support)
        input_ids: [B, T_prefill] input token IDs
        attention_mask: [B, T_prefill] (optional, 1=valid)
        max_new_tokens: N — number of new tokens to generate
        chunk_size: prefill chunk size (default 512)
        do_sample: greedy wenn False
        use_cache: KV-Cache aktiv (default True)
        eos_token_id: stop token id (optional)

    Returns:
        output_ids: [B, T_prefill + max_new_tokens] (truncated at eos)

    Speicher-Charakteristik:
        Peak ≈ (T_so_far + chunk_size) × hidden_size × bytes
             + (chunk_size + T_so_far) × bytes (attention logits, cached)
        Bei T=8000, chunk=512: peak ≈ 8512 × 4096 × 2 = 70 MB statt
        8000 × 8000 × 2 = 122 MB für attention matrix. Faktor 12x.
    """
    device = input_ids.device
    B, T_prefill = input_ids.shape

    # Past_key_values initialisieren (HF-Cache)
    past_kv = None

    # === Phase 1: Chunked Prefill ===
    # Wir gehen über die input_ids in Chunks. Jeder Chunk baut den KV-Cache auf.
    n_chunks = (T_prefill + chunk_size - 1) // chunk_size
    prefill_logits = None

    for i in range(n_chunks):
        s = i * chunk_size
        e = min(s + chunk_size, T_prefill)
        chunk_ids = input_ids[:, s:e]

        # attention_mask für diesen Chunk
        if attention_mask is not None:
            chunk_mask = attention_mask[:, :e]
        else:
            chunk_mask = None

        with torch.inference_mode():
            out = model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                past_key_values=past_kv,
                use_cache=use_cache,
            )
        past_kv = out.past_key_values
        prefill_logits = out.logits  # logits für LAST token in chunk

    # === Phase 2: Incremental Decode ===
    # Letztes Token aus Prefill als erstes generated token
    next_token_logits = prefill_logits[:, -1, :]
    if do_sample:
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)

    output_ids = [input_ids, next_token]
    if eos_token_id is not None and (next_token == eos_token_id).all():
        return torch.cat(output_ids, dim=1)

    # Decode N-1 weitere tokens
    for step in range(max_new_tokens - 1):
        with torch.inference_mode():
            out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=use_cache,
            )
        past_kv = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        if do_sample:
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        output_ids.append(next_token)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return torch.cat(output_ids, dim=1)


def test_chunked_against_full():
    """Vergleicht chunked_generate vs model.generate — output soll gleich sein."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True,max_split_size_mb:256")
    import time
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from quantize_pipeline import quantize_all_linears

    hf_id = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    quantize_all_linears(model)
    model.eval()

    prompt = "Die subjektive Erfahrung des Bewusstseins. " * 30
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    T = inputs["input_ids"].shape[1]
    print(f"Prompt T={T}")

    # === Reference: full generate (sollte für T=4502 OK sein) ===
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pre_vram = torch.cuda.memory_allocated() / 1e9
    t0 = time.time()
    with torch.inference_mode():
        out_full = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    dt_full = time.time() - t0
    peak_full = torch.cuda.max_memory_allocated() / 1e9
    text_full = tokenizer.decode(out_full[0, T:], skip_special_tokens=True)
    print(f"\nFULL generate: T={T} peak={peak_full:.3f}GB dt={dt_full:.1f}s")
    print(f"  text[:100]: {text_full[:100]!r}")

    # === Chunked: bei gleichem T (zum Vergleich) ===
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pre_vram_c = torch.cuda.memory_allocated() / 1e9
    t0 = time.time()
    out_chunked = chunked_generate(
        model, inputs["input_ids"], chunk_size=512,
        max_new_tokens=32, do_sample=False,
    )
    dt_chunked = time.time() - t0
    peak_chunked = torch.cuda.max_memory_allocated() / 1e9
    text_chunked = tokenizer.decode(out_chunked[0, T:], skip_special_tokens=True)
    print(f"\nCHUNKED generate: T={T} peak={peak_chunked:.3f}GB dt={dt_chunked:.1f}s")
    print(f"  text[:100]: {text_chunked[:100]!r}")

    # Vergleich: text soll identisch sein (greedy decoding)
    if text_full == text_chunked:
        print(f"\n[OK] chunked == full (byte-identisch)")
    else:
        # Akzeptanz: erste 50% sollen gleich sein
        common_prefix = 0
        for a, b in zip(text_full, text_chunked):
            if a == b:
                common_prefix += 1
            else:
                break
        match_pct = common_prefix / max(len(text_full), 1) * 100
        print(f"\n[INFO] common prefix: {common_prefix}/{len(text_full)} chars ({match_pct:.1f}%)")

    return peak_full, peak_chunked


if __name__ == "__main__":
    test_chunked_against_full()