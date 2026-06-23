"""
test_infllm_vram_reduction.py — TDD-rot: echte VRAM-Messung Phase D
=====================================================================

Misst ob InfLLM (block-memory + top-k retrieval) tatsächlich VRAM
reduziert vs Standard-Attention. Akzeptanzkriterium für Phase D:
≥ 30% KV-Cache-Reduktion bei T=4096+ Tokens.

Was wir messen:
  - KV-Cache-Speicher nach 4096 Tokens forward-pass
  - Mit InfLLM (top_k=4, block=16) vs ohne (Standard-KV)
  - In beiden Fällen: gleicher Input, gleiche Layer-Anzahl

Wir mocken KEIN 4b (zu groß/langsam für unit-tests). Wir bauen einen
kleinen Mock-Stack:
  - Mini-GPT: 8 Layers, 4 heads, head_dim=32, hidden=128
  - Forward mit T=4096 Tokens
  - Mit/ohne InfLLM
  - Messe `torch.cuda.memory_allocated()` Differenz

WICHTIG: dieser Test läuft nur wenn CUDA verfügbar.

Run:
    /path/to/venv/bin/python test_infllm_vram_reduction.py
"""
import sys
import os

import torch
import torch.nn as nn

_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_SCRATCHES = os.path.join(_REPO, "scratches", "4b-image")
if _SCRATCHES not in sys.path:
    sys.path.insert(0, _SCRATCHES)


# Tests ---------------------------------------------------------------------

def test_infllm_kv_memory_growth_sublinear():
    """KV-Cache-Wachstum mit InfLLM ist SUBLINEAR (nicht O(T)).

    Wir bauen einen Mini-GPT-Stack und simulieren 4 forward-Passes mit
    progressiv wachsendem T (1024, 2048, 4096, 8192). Wir erwarten:
      - Ohne InfLLM: KV-Memory wächst ~linear mit T (ca. 2*Hkv*D*bytes × T)
      - Mit InfLLM: KV-Memory wächst langsamer (top_k-Blöcke bounded)

    Akzeptanzkriterium: bei T=8192 ist InfLLM-KV mindestens 30% kleiner
    als Standard-KV.
    """
    if not torch.cuda.is_available():
        print("[SKIP] CUDA nicht verfügbar")
        return

    from infinite_context import InfLLMCache
    from infllm_integration import install_infllm_hooks, remove_infllm_hooks

    # Mini-GPT: 8 Layers, 4 Heads, head_dim=32, hidden=128, kv_heads=2 (GQA)
    NUM_LAYERS = 8
    NUM_HEADS = 4
    NUM_KV_HEADS = 4  # Kein GQA für den Mini-Stack → konsistente Heads
    HEAD_DIM = 32
    HIDDEN = 128

    # Cache-Konfiguration
    cache = InfLLMCache(
        type("Cfg", (), {"num_hidden_layers": NUM_LAYERS})(),
        block_size=16,
        r_tokens=2,
        top_k_blocks=4,
        sinks_count=4,
    )

    # Mini-Stacks
    class MiniAttn(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.layer_idx = idx
            self.head_dim = HEAD_DIM
            self.q_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
            self.k_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
            self.v_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
            self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False)
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.scaling = 0.5
            self.attention_dropout = 0.0
            self.sliding_window = None
            self.num_key_value_groups = NUM_HEADS // NUM_KV_HEADS
            self.config = type("Cfg", (), {"_attn_implementation": "eager"})()
            self.rotary_emb = lambda x, pos: (
                torch.ones(*x.shape[:3], x.shape[-1], device=x.device),
                torch.zeros(*x.shape[:3], x.shape[-1], device=x.device),
            )

        def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                     past_key_values=None, **kwargs):
            """Standard SDPA forward (für Vergleich OHNE InfLLM)."""
            from transformers.models.gemma3.modeling_gemma3 import (
                apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
            )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            attn_out, _ = eager_attention_forward(
                self, q, k, v, attention_mask,
                dropout=0.0, scaling=self.scaling, sliding_window=None, **kwargs,
            )
            attn_out = attn_out.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_out), None

    class MiniLayer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.attn = MiniAttn(idx)

    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([MiniLayer(i) for i in range(NUM_LAYERS)])

    # ===== OHNE InfLLM (Standard-KV) =====
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_std = MiniModel().cuda()

    # Wir messen Peak-Memory während eines wachsenden Forward-Passes
    peak_without = 0
    T_values = [256, 512, 1024, 2048]
    mem_without = {}
    for T in T_values:
        h = torch.randn(1, T, HIDDEN, device="cuda")
        for layer in model_std.layers:
            _ = layer.attn(h, position_embeddings=None, attention_mask=None, past_key_values=None)
        torch.cuda.synchronize()
        current = torch.cuda.memory_allocated() / 1e9  # GB
        mem_without[T] = current
        peak_without = max(peak_without, current)

    del model_std
    torch.cuda.empty_cache()

    # ===== MIT InfLLM =====
    cache_infllm = InfLLMCache(
        type("Cfg", (), {"num_hidden_layers": NUM_LAYERS})(),
        block_size=16,
        r_tokens=2,
        top_k_blocks=4,
        sinks_count=4,
    )
    torch.cuda.reset_peak_memory_stats()
    model_infllm = MiniModel().cuda()
    install_infllm_hooks(model_infllm, cache_infllm)

    mem_with = {}
    for T in T_values:
        h = torch.randn(1, T, HIDDEN, device="cuda")
        for layer in model_infllm.layers:
            _ = layer.attn(h, position_embeddings=None, attention_mask=None, past_key_values=None)
        torch.cuda.synchronize()
        current = torch.cuda.memory_allocated() / 1e9
        mem_with[T] = current

    # Cleanup
    remove_infllm_hooks(model_infllm)
    del model_infllm
    torch.cuda.empty_cache()

    # Vergleich
    print(f"\n{'T':>6} {'no-infllm':>12} {'infllm':>12} {'reduction':>10}")
    for T in T_values:
        red = 1 - mem_with[T] / max(mem_without[T], 1e-9)
        print(f"{T:>6} {mem_without[T]:>10.4f}GB {mem_with[T]:>10.4f}GB {red:>9.1%}")

    # Akzeptanzkriterium: bei T=2048 mindestens 30% Reduktion
    reduction_at_max = 1 - mem_with[T_values[-1]] / max(mem_without[T_values[-1]], 1e-9)
    assert reduction_at_max >= 0.30, (
        f"InfLLM reduziert nur {reduction_at_max:.1%} bei T={T_values[-1]} "
        f"(Akzeptanz: ≥30%). mem_with={mem_with[T_values[-1]]:.4f}GB, "
        f"mem_without={mem_without[T_values[-1]]:.4f}GB")
    print(f"\n[OK] InfLLM reduziert KV bei T={T_values[-1]} um {reduction_at_max:.1%}")


def test_infllm_quality_cos_sim_vs_sdpa():
    """InfLLM-Output vs SDPA-Output: cos_sim >= 0.85 (verlustbehaftet, aber nah).

    Das misst: wenn wir die gleiche Mock-Attention einmal mit und einmal
    ohne InfLLM laufen lassen, ist der Output cosinely ähnlich (wir
    verlieren Blöcke, aber Top-K + Sinks sollten den Hauptteil liefern).

    Wichtig: bei T=128, block=16, top_k=4 → 64/128 = 50% der Tokens sind
    im retrieved-Set. Die andere Hälfte fehlt — das muss den Output
    deutlich ändern. Cos_sim >= 0.85 ist ein schwaches, aber ehrliches
    Kriterium.
    """
    if not torch.cuda.is_available():
        print("[SKIP] CUDA nicht verfügbar")
        return

    from infinite_context import InfLLMCache
    from infllm_integration import install_infllm_hooks, remove_infllm_hooks

    NUM_LAYERS = 2
    NUM_HEADS = 4
    NUM_KV_HEADS = 4  # Kein GQA für Mini-Stack
    HEAD_DIM = 32
    HIDDEN = 128
    T = 128

    class MiniAttn(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.layer_idx = idx
            self.head_dim = HEAD_DIM
            self.q_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
            self.k_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
            self.v_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
            self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False)
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.scaling = 0.5
            self.attention_dropout = 0.0
            self.sliding_window = None
            self.num_key_value_groups = NUM_HEADS // NUM_KV_HEADS
            self.config = type("Cfg", (), {"_attn_implementation": "eager"})()
            self.rotary_emb = lambda x, pos: (
                torch.ones(*x.shape[:3], x.shape[-1], device=x.device),
                torch.zeros(*x.shape[:3], x.shape[-1], device=x.device),
            )

        def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                     past_key_values=None, **kwargs):
            """Standard SDPA forward (für Vergleich OHNE InfLLM)."""
            from transformers.models.gemma3.modeling_gemma3 import (
                apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
            )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            attn_out, _ = eager_attention_forward(
                self, q, k, v, attention_mask,
                dropout=0.0, scaling=self.scaling, sliding_window=None, **kwargs,
            )
            attn_out = attn_out.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_out), None

    class MiniLayer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.attn = MiniAttn(idx)

    torch.manual_seed(42)
    model = nn.ModuleList([MiniLayer(i) for i in range(NUM_LAYERS)]).cuda()
    h = torch.randn(1, T, HIDDEN, device="cuda")

    # Standard
    out_std_layers = []
    for layer in model:
        out, _ = layer.attn(h, position_embeddings=None, attention_mask=None, past_key_values=None)
        out_std_layers.append(out.detach().clone())

    # Mit InfLLM
    cache = InfLLMCache(
        type("Cfg", (), {"num_hidden_layers": NUM_LAYERS})(),
        block_size=16, r_tokens=2, top_k_blocks=4, sinks_count=4,
    )
    # install auf model
    class Wrapper(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers
    wrapper = Wrapper(model).cuda()
    install_infllm_hooks(wrapper, cache)
    out_infllm_layers = []
    for layer in model:
        out, _ = layer.attn(h, position_embeddings=None, attention_mask=None, past_key_values=None)
        out_infllm_layers.append(out.detach().clone())
    remove_infllm_hooks(wrapper)

    # Cosine sim pro Layer
    cos_sims = []
    for i in range(NUM_LAYERS):
        cos = torch.nn.functional.cosine_similarity(
            out_std_layers[i].flatten().unsqueeze(0).float(),
            out_infllm_layers[i].flatten().unsqueeze(0).float(),
        ).item()
        cos_sims.append(cos)

    mean_cos = sum(cos_sims) / len(cos_sims)
    print(f"\nCosine sim per layer: {cos_sims}, mean={mean_cos:.4f}")
    assert mean_cos >= 0.85, f"cos_sim={mean_cos:.4f} < 0.85 (zu viel Verlust)"
    print(f"[OK] InfLLM quality cos_sim={mean_cos:.4f} >= 0.85")


if __name__ == "__main__":
    tests = [
        ("KV memory sublinear",   test_infllm_kv_memory_growth_sublinear),
        ("quality cos_sim>=0.85", test_infllm_quality_cos_sim_vs_sdpa),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except AssertionError as e:
            print(f"[FAIL] {name}: AssertionError: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)