"""
test_cuda_graph_search_variance.py — Regression-Test: CUDA-Graph hält bei variable-length search=True Prompts
================================================================================================================
RED-Phase: Test prüft dass CUDA-Graph NICHT in den Fallback-Pfad fällt wenn Suchergebnisse die
Prompt-Länge variieren.
"""
import os
import sys
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from px_patches_v3.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig

MODEL = "google/gemma-3-270m-it"


def test_cuda_graph_with_varied_prompt_lengths():
    """Prompts mit stark unterschiedlichen Längen (search-augmented) brechen CUDA-Graph nicht."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    # Prompts: 4 lang + 4 kurz (simuliert search-on vs search-off)
    long_prompts = [
        "What is the largest planet? " + " ".join(["saturn jupiter"] * 20) for _ in range(4)
    ]
    short_prompts = [f"Q: {i}+{i}?" for i in range(4)]
    prompts = long_prompts + short_prompts
    tpl = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(tpl, padding="longest", return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to("cuda")
    am = enc["attention_mask"].to("cuda")
    print(f"  Input shape: {ids.shape}")

    cfg = CUDAGraphRunnerConfig(batch_size=8, max_seq_len=ids.shape[1] * 2 + 300 + 64)
    runner = CUDAGraphRunner(m, cfg)
    try:
        first = runner.setup(ids, am)
        # 5 decode steps
        for _ in range(5):
            nxt = runner.step()
            runner.append(nxt)
        print(f"  ✓ CUDA-Graph survives varied prompt lengths (max_seq={cfg.max_seq_len})")
    except RuntimeError as e:
        if "size of tensor" in str(e).lower():
            print(f"  ✗ CUDA-Graph FAILED with size mismatch: {e}")
            raise AssertionError(f"Tensor-mismatch bei variable-length prompts: {e}")
        raise
    finally:
        del m, tok, runner
        torch.cuda.empty_cache()


def test_cuda_graph_holds_with_max_in_times_2():
    """Mit max_in*2 Buffer überlebt CUDA-Graph auch bei search=True-Batches."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    # Simuliere search-augmented: 8 Prompts mit ~500 input tokens
    long_prompts = [
        " ".join([f"fact{i}: some long context " * 5 for i in range(10)])
        for _ in range(8)
    ]
    tpl = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in long_prompts]
    enc = tok(tpl, padding="longest", return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to("cuda")
    am = enc["attention_mask"].to("cuda")
    n_in = ids.shape[1]
    max_seq = n_in * 2 + 64  # wie v3.5 harness
    print(f"  Input shape: {ids.shape}, max_seq={max_seq}")

    cfg = CUDAGraphRunnerConfig(batch_size=8, max_seq_len=max_seq)
    runner = CUDAGraphRunner(m, cfg)
    try:
        first = runner.setup(ids, am)
        for _ in range(10):
            nxt = runner.step()
            runner.append(nxt)
        print(f"  ✓ CUDA-Graph mit max_in*2 + 64 Buffer: stabil")
    finally:
        del m, tok, runner
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("CUDA-Graph Search-Variance Regression-Test (v3.5 RIGOR)")
    print("=" * 70)
    tests = [
        ("test_cuda_graph_with_varied_prompt_lengths", test_cuda_graph_with_varied_prompt_lengths),
        ("test_cuda_graph_holds_with_max_in_times_2", test_cuda_graph_holds_with_max_in_times_2),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"Ergebnis: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
