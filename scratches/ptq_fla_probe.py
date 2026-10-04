#!/usr/bin/env python
"""sm75-Beweis fuer fla 0.5.2 auf RTX 2060 (Turing): Liefert der fla-Fast-Pfad
(chunk_gated_delta_rule / fused_recurrent_gated_delta_rule / FusedRMSNormGated)
numerisch korrekte Logits vs. den torch-Fallback? Das ist die Freigabe-Bedingung,
bevor der fla-Pfad in Produktion gelassen wird.

A/B in einem Prozess: (1) Forward mit aktiven fla-Funktionen (nach Install
automatisch verdrahtet), (2) Forward/Greedy mit per-Layer-Monkeypatch auf
torch_chunk_gated_delta_rule / torch_recurrent_gated_delta_rule.
Kriterien: fla-Logits finit; max|A-B| <= 5% von max|A|; greedy 12 Token
token-identisch (bf16-Rundung erlaubt maximal selten Drift -> 1-Abweichung
noch WARN, >1 FAIL); Timing fluechtig dokumentiert.
Exit 0 = PASS.
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


def main():
    torch.cuda.reset_peak_memory_stats()
    signs, meta = rt.load_signs()
    fold = rt.FoldOps(signs, mode="signs_first")
    model = rt.build_and_load(fold, verbose=False)
    print("[fla] geladen, alloc %.2f GiB" % (torch.cuda.memory_allocated() / 2**30,),
          flush=True)
    tok = AutoTokenizer.from_pretrained(rt.OUT_DIR, fix_mistral_regex=True)

    gdn = []
    for i, layer in enumerate(model.model.layers):
        la = getattr(layer, "linear_attn", None)
        if la is not None:
            gdn.append((i, la))
    la0 = gdn[0][1]
    print(f"[fla] GDN-Layer: {len(gdn)}", flush=True)
    print(f"[fla] chunk-> {la0.chunk_gated_delta_rule.__module__}", flush=True)
    print(f"[fla] recurr-> {la0.recurrent_gated_delta_rule.__module__}", flush=True)
    print(f"[fla] norm-> {type(la0.norm).__name__}", flush=True)
    print(f"[fla] causal_conv1d_fn -> {la0.causal_conv1d_fn}", flush=True)

    msgs = [{"role": "user", "content": "How would you describe the way you think?"}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    print(f"[fla] prompt tokens: {ids.shape[1]}", flush=True)

    def forward_logits():
        with torch.no_grad():
            return model(ids).logits

    _ = forward_logits()                                # JIT-Warmup (Triton-Compile)
    torch.cuda.synchronize()
    t0 = time.time()
    A = forward_logits()
    torch.cuda.synchronize()
    t_fla = time.time() - t0
    va = A.float().abs().max().item()
    fin = bool(torch.isfinite(A.float()).all())
    print(f"[fla] forward fla: {t_fla:.2f}s | max|A|={va:.4g} | finite={fin}", flush=True)
    if not fin:
        print("[fla] VERDICT: FAIL (nonfinite)")
        return 1

    # --- torch-Fallback Monkeypatch (nur delta-rule-Funktionen; norm bleibt fla) ---
    saved = []
    for i, la in gdn:
        saved.append((la, la.chunk_gated_delta_rule, la.recurrent_gated_delta_rule))
        la.chunk_gated_delta_rule = mq.torch_chunk_gated_delta_rule
        la.recurrent_gated_delta_rule = mq.torch_recurrent_gated_delta_rule
    torch.cuda.synchronize()
    t0 = time.time()
    B = forward_logits()
    torch.cuda.synchronize()
    t_torch = time.time() - t0
    for la, c, r in saved:
        la.chunk_gated_delta_rule = c
        la.recurrent_gated_delta_rule = r
    vd = (A.float() - B.float()).abs().max().item()
    rel = vd / va
    print(f"[torch] forward torch-fallback: {t_torch:.2f}s "
          f"(fla {t_fla/time.time()*0 + t_fla:.2f}s, speedup x{t_torch/t_fla:.2f})",
          flush=True)
    print(f"[ab] max|A-B|={vd:.4g} | rel={rel:.3g}", flush=True)

    # --- Decode-Pfad (recurrent, M=1) via greedy generate, beide Pfade ---
    gen = dict(do_sample=False, max_new_tokens=12,
               pad_token_id=tok.eos_token_id if tok.eos_token_id else 0)
    with torch.no_grad():
        out_fla = model.generate(ids, **gen)
    txt_fla = tok.decode(out_fla[0, ids.shape[1]:], skip_special_tokens=True)

    for la, c, r in saved:
        la.chunk_gated_delta_rule = mq.torch_chunk_gated_delta_rule
        la.recurrent_gated_delta_rule = mq.torch_recurrent_gated_delta_rule
    with torch.no_grad():
        out_t = model.generate(ids, **gen)
    txt_t = tok.decode(out_t[0, ids.shape[1]:], skip_special_tokens=True)
    for la, c, r in saved:
        la.chunk_gated_delta_rule = c
        la.recurrent_gated_delta_rule = r

    print(f"[fla]   greedy12: {txt_fla!r}", flush=True)
    print(f"[torch] greedy12: {txt_t!r}", flush=True)
    eq = (out_fla[0, ids.shape[1]:] == out_t[0, ids.shape[1]:])
    n_eq = int(eq.sum())
    print(f"[ab] greedy-token-equal: {n_eq}/{eq.numel()}", flush=True)

    ok = rel < 0.05 and n_eq == eq.numel()
    print(f"[fla] VERDICT: {'PASS' if ok else 'CHECK'}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())