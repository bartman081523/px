"""vLLM-Probe: lädt gemma-4-E2B und misst Durchsatz.

vLLM startet einen Engine-Subprozess, der das Hauptskript erneut ausführt —
deshalb MUSS das ein echtes File sein (kein `python - <<EOF`).

Aufruf:
    python bench/probes/vllm_probe.py [--max-model-len 2048] [--max-tokens 64]
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--dtype", default="float16",
                    help="float16|bfloat16 (vLLM lehnt fp16 für gemma3 ab)")
    ap.add_argument("--out", default="bench/results/vllm_probe.json")
    a = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    print(f"[vllm] torch {torch.__version__} | cc {torch.cuda.get_device_capability()}",
          flush=True)
    t0 = time.time()
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_util, enforce_eager=True,
              trust_remote_code=True)
    load_s = time.time() - t0
    print(f"[vllm] loaded in {load_s:.0f}s | VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB",
          flush=True)

    prompts = [
        "Write a Python function add(a, b). Code only.",
        "Explain in two sentences why the sky is blue.",
    ]
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0)
    t0 = time.time()
    out = llm.generate(prompts, sp)
    gen_s = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in out)
    res = {
        "model": a.model, "dtype": a.dtype, "load_s": round(load_s, 1),
        "gen_s": round(gen_s, 2), "tokens": n_tok,
        "tok_s": round(n_tok / gen_s, 2) if gen_s else None,
        "outputs": [o.outputs[0].text[:160] for o in out],
    }
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
