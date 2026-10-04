#!/usr/bin/env python
"""Warme Decode-Rate: fla-recurrent vs torch-recurrent (GDN-Decode-Pfad).
Der Audit-Generate zeigte 0.664 s/t — davon koennte die fla-fused_recurrent
auf sm75 signifikant langsamer sein als der torch-Fallback. Hier: nach
ausdruecklichem Warmup (alle JIT-Compiles weg) 24-tok greedy generieren in
beiden Konfigurationen und s/t vergleichen. Exit 0 immer (Mess-Skript).
"""
import sys
import time

ROOT = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/px_patches")

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

import ternary_bonsai_27b_px.runtime_qwen35_ptq as rt


def gen24(model, ids, tok, n=24):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=n,
                             pad_token_id=tok.eos_token_id if tok.eos_token_id else 0)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return out, dt


def main():
    signs, _ = rt.load_signs()
    fold = rt.FoldOps(signs, mode="signs_first")
    model = rt.build_and_load(fold, verbose=False)
    tok = AutoTokenizer.from_pretrained(rt.OUT_DIR, fix_mistral_regex=True)
    msgs = [{"role": "user", "content": "Name three colors."}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)

    gdn = [(i, layer.linear_attn) for i, layer in enumerate(model.model.layers)
           if getattr(layer, "linear_attn", None) is not None]
    orig = [(la.recurrent_gated_delta_rule, la.chunk_gated_delta_rule) for _, la in gdn]

    # Warmup: prefill-Chunk-Compile + recurrent-Compile wegraeumen
    _, _ = gen24(model, ids, tok, n=8)
    print("[bench] warmup fertig", flush=True)

    # --- fla (wie verdrahtet) ---
    out_fla, dt_fla = gen24(model, ids, tok)
    txt_fla = tok.decode(out_fla[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"[bench] fla-recurrent: {dt_fla / 24:.3f} s/t | {txt_fla[:70]!r}", flush=True)

    # --- torch fallback recurrent ---
    for (rec0, _), (_, la) in zip(orig, gdn):
        la.recurrent_gated_delta_rule = mq.torch_recurrent_gated_delta_rule
    _, _ = gen24(model, ids, tok, n=8)  # warmup (falls torch-Pfad eigener code)
    out_t, dt_t = gen24(model, ids, tok)
    txt_t = tok.decode(out_t[0, ids.shape[1]:], skip_special_tokens=True)
    for (rec0, _), (_, la) in zip(orig, gdn):
        la.recurrent_gated_delta_rule = rec0
    print(f"[bench] torch-recurrent: {dt_t / 24:.3f} s/t | {txt_t[:70]!r}", flush=True)
    print(f"[bench] delta: fla - torch = {dt_fla - dt_t:+.1f} s / 24 tok "
          f"({(dt_fla - dt_t) / 24:+.3f} s/t)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())