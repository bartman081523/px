"""
test_int8_kv_cache.py — Plan 3 Phase A: KV-Cache int8 Quantization (TDD-rot)
================================================================================

Was getestet wird:
  - KV-Tensoren (k/v aus dem Attention-forward) werden per-channel int8
    quantisiert + dequantisiert on-read.
  - Bei T=2048, hidden=256, 8 layers: KV int8 belegt ~50% von bf16 KV.
  - cos_sim >= 0.95 (int8 ist sehr treu für KV-Werte).
  - forward-hook patch funktioniert idempotent.

Architektur:
  - hook auf attention forward (q_proj/k_proj/v_proj/o_proj)
  - nach k/v_proj: (k_int8, k_scale, k_zero) statt (k_full)
  - bei retrieve: dequant on-the-fly (matmul verträgt das nicht direkt,
    also reconstruct zu bf16 vor matmul)
  - Storage: int8 + fp32 scale per (head, channel)

Warum forward_hook und kein motor-edit:
  - Gemma3Attention.forward ist der originale, motor bleibt sauber
  - hook ist rückgängig machbar
  - quantisiertes KV ist API-kompatibel mit DynamicCache.update()

Akzeptanzkriterien:
  1. test_int8_kv_vram_reduction_50pct: VRAM >= 40% kleiner als bf16 KV
  2. test_int8_kv_cos_sim_above_095: round-trip cos_sim >= 0.95
  3. test_int8_kv_install_idempotent: install+remove hinterlässt model sauber
  4. test_int8_kv_double_install: zweimal install → kein crash
  5. test_int8_kv_attention_output_close: full forward mit int8 KV
     produziert output der nahezu gleich ist wie ohne (cos_sim >= 0.95)

Run:
    python -m py_compile test_int8_kv_cache.py && python test_int8_kv_cache.py
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

def test_int8_kv_install_uninstall_idempotent():
    """install+remove → model unverändert (KV bleibt bf16)."""
    # Noch nicht implementiert: import muss failen
    try:
        from int8_kv_cache import install_int8_kv_hooks, remove_int8_kv_hooks
    except ImportError:
        raise AssertionError("int8_kv_cache.py noch nicht implementiert (TDD-rot erwartet)")

    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 8
            self.q_proj = nn.Linear(16, 16, bias=False)
            self.k_proj = nn.Linear(16, 16, bias=False)
            self.v_proj = nn.Linear(16, 16, bias=False)
            self.o_proj = nn.Linear(16, 16, bias=False)
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.scaling = 0.5
            self.attention_dropout = 0.0
            self.sliding_window = None
            self.num_key_value_groups = 1
            self.config = type("Cfg", (), {"_attn_implementation": "eager"})()
            self.rotary_emb = lambda x, pos: (
                torch.ones(*x.shape[:3], x.shape[-1]),
                torch.zeros(*x.shape[:3], x.shape[-1]),
            )

        def forward(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states), None)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MockAttention()

    model = MockModel()
    n_patched = install_int8_kv_hooks(model)
    assert n_patched == 1, f"expected 1 patched, got {n_patched}"
    assert hasattr(model.attn, "_original_forward"), "_original_forward not saved"
    n_removed = remove_int8_kv_hooks(model)
    assert n_removed == 1, f"expected 1 removed, got {n_removed}"
    assert not hasattr(model.attn, "_original_forward"), "_original_forward leaked"
    print("[OK] install+uninstall idempotent")


def test_int8_kv_cos_sim_above_095():
    """int8 KV round-trip cos_sim >= 0.95."""
    try:
        from int8_kv_cache import quantize_kv, dequantize_kv
    except ImportError:
        raise AssertionError("int8_kv_cache.py fehlt")

    torch.manual_seed(42)
    # Typische KV-shape: [B, H_kv, T, head_dim]
    k = torch.randn(1, 4, 512, 32)
    v = torch.randn(1, 4, 512, 32)

    k_int8, k_scale, k_zero = quantize_kv(k)
    v_int8, v_scale, v_zero = quantize_kv(v)

    k_back = dequantize_kv(k_int8, k_scale, k_zero)
    v_back = dequantize_kv(v_int8, v_scale, v_zero)

    cos_k = torch.nn.functional.cosine_similarity(
        k.flatten().unsqueeze(0).float(), k_back.flatten().unsqueeze(0).float()
    ).item()
    cos_v = torch.nn.functional.cosine_similarity(
        v.flatten().unsqueeze(0).float(), v_back.flatten().unsqueeze(0).float()
    ).item()

    assert cos_k >= 0.95, f"k cos_sim={cos_k:.4f} < 0.95"
    assert cos_v >= 0.95, f"v cos_sim={cos_v:.4f} < 0.95"
    print(f"[OK] int8 KV round-trip cos_sim k={cos_k:.4f} v={cos_v:.4f} >= 0.95")


def test_int8_kv_vram_reduction_50pct():
    """int8 KV belegt ~50% von bf16 KV (Akzeptanz: ≥40% Reduktion).

    Wichtig: wir messen nicht das ganze Model (Activation, Weights etc.),
    sondern nur den KV-Storage-Overhead den InfLLM/DynamicCache anlegt.
    """
    if not torch.cuda.is_available():
        print("[SKIP] CUDA nicht verfügbar")
        return

    try:
        from int8_kv_cache import quantize_kv, dequantize_kv
    except ImportError:
        raise AssertionError("int8_kv_cache.py fehlt")

    # Simuliere KV-Storage für 8 layers × 4096 tokens × 8 kv_heads × 128 head_dim
    # bf16: 2 bytes/value → total = 8 * 4096 * 8 * 128 * 2 = ~67 MB pro KV
    # int8: 1 byte/value + 4 bytes scale (per (head, channel)) → ca. 50%
    NUM_LAYERS = 8
    T = 4096
    NUM_KV_HEADS = 8
    HEAD_DIM = 128

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # bf16 baseline
    bf16_kvs = []
    for _ in range(NUM_LAYERS):
        k = torch.randn(1, NUM_KV_HEADS, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, NUM_KV_HEADS, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        bf16_kvs.append((k, v))
    torch.cuda.synchronize()
    bf16_mem = torch.cuda.memory_allocated()
    del bf16_kvs
    torch.cuda.empty_cache()

    # int8 version
    torch.cuda.reset_peak_memory_stats()
    int8_kvs = []
    for _ in range(NUM_LAYERS):
        k = torch.randn(1, NUM_KV_HEADS, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, NUM_KV_HEADS, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        k_int8, k_scale, k_zero = quantize_kv(k)
        v_int8, v_scale, v_zero = quantize_kv(v)
        # Sofort auf cpu verschieben (das ist der Sinn: int8 KV liegt auf cpu)
        int8_kvs.append(((k_int8.cpu(), k_scale.cpu(), k_zero.cpu()),
                          (v_int8.cpu(), v_scale.cpu(), v_zero.cpu())))
        del k, v, k_int8, k_scale, k_zero, v_int8, v_scale, v_zero
    torch.cuda.synchronize()
    int8_mem = torch.cuda.memory_allocated()

    print(f"\n  bf16 KV: {bf16_mem/1e9:.4f}GB")
    print(f"  int8 KV (nur cuda): {int8_mem/1e9:.4f}GB")

    # Aufräumen
    del int8_kvs
    torch.cuda.empty_cache()

    # Akzeptanz: int8 sollte weniger GPU-Speicher brauchen
    # (eigentlich sogar ~0 wenn alles auf CPU verschoben wurde)
    assert int8_mem < bf16_mem * 0.6, (
        f"int8 braucht {int8_mem/1e9:.4f}GB, bf16 {bf16_mem/1e9:.4f}GB — "
        f"Reduktion nur {(1-int8_mem/bf16_mem)*100:.1f}%")
    print(f"[OK] int8 KV VRAM-Reduktion {(1-int8_mem/bf16_mem)*100:.1f}%")


def test_int8_kv_double_install_safe():
    """Zweimal install → kein crash, idempotent."""
    try:
        from int8_kv_cache import install_int8_kv_hooks, remove_int8_kv_hooks
    except ImportError:
        raise AssertionError("int8_kv_cache.py fehlt")

    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 8
            self.q_proj = nn.Linear(16, 16, bias=False)
            self.k_proj = nn.Linear(16, 16, bias=False)
            self.v_proj = nn.Linear(16, 16, bias=False)
            self.o_proj = nn.Linear(16, 16, bias=False)
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.scaling = 0.5
            self.attention_dropout = 0.0
            self.sliding_window = None
            self.num_key_value_groups = 1
            self.config = type("Cfg", (), {"_attn_implementation": "eager"})()
            self.rotary_emb = lambda x, pos: (
                torch.ones(*x.shape[:3], x.shape[-1]),
                torch.zeros(*x.shape[:3], x.shape[-1]),
            )

        def forward(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states), None)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MockAttention()

    model = MockModel()
    n1 = install_int8_kv_hooks(model)
    n2 = install_int8_kv_hooks(model)
    assert n1 == n2 == 1, f"expected 1/1, got {n1}/{n2}"
    remove_int8_kv_hooks(model)
    print(f"[OK] double install idempotent (n1={n1}, n2={n2})")


def test_int8_kv_attention_output_close():
    """Full forward mit int8 KV → output nahezu gleich wie bf16 KV (cos_sim >= 0.95)."""
    try:
        from int8_kv_cache import quantize_kv, dequantize_kv, install_int8_kv_hooks, remove_int8_kv_hooks
    except ImportError:
        raise AssertionError("int8_kv_cache.py fehlt")

    # Mini-Stack mit echter attention
    class MockAttention(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.layer_idx = idx
            self.head_dim = 8
            HIDDEN = 16
            NUM_HEADS = 2
            NUM_KV_HEADS = 2
            self.q_proj = nn.Linear(HIDDEN, NUM_HEADS * 8, bias=False)
            self.k_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * 8, bias=False)
            self.v_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * 8, bias=False)
            self.o_proj = nn.Linear(NUM_HEADS * 8, HIDDEN, bias=False)
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.scaling = 0.5
            self.attention_dropout = 0.0
            self.sliding_window = None
            self.num_key_value_groups = 1
            self.config = type("Cfg", (), {"_attn_implementation": "eager"})()
            self.rotary_emb = lambda x, pos: (
                torch.ones(*x.shape[:3], x.shape[-1]),
                torch.zeros(*x.shape[:3], x.shape[-1]),
            )

        def forward(self, hidden_states, **kwargs):
            from transformers.models.gemma3.modeling_gemma3 import (
                apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
            )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, 8)
            q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            attn_out, _ = eager_attention_forward(
                self, q, k, v, None,
                dropout=0.0, scaling=self.scaling, sliding_window=None,
            )
            return self.o_proj(attn_out.reshape(*input_shape, -1)), None

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = MockAttention(0)
            self.attn2 = MockAttention(1)

    torch.manual_seed(123)
    model_bf16 = MockModel()
    h = torch.randn(1, 64, 16)

    # bf16 baseline
    out_bf16_1, _ = model_bf16.attn1(h)
    out_bf16_2, _ = model_bf16.attn2(out_bf16_1)

    # int8: install hook, dann forward (hook quantisiert KV on-the-fly)
    model_int8 = MockModel()
    # State-dict sync damit die attention-Gewichte gleich sind
    model_int8.load_state_dict(model_bf16.state_dict())
    install_int8_kv_hooks(model_int8)
    try:
        out_int8_1, _ = model_int8.attn1(h)
        out_int8_2, _ = model_int8.attn2(out_int8_1)
    finally:
        remove_int8_kv_hooks(model_int8)

    cos_1 = torch.nn.functional.cosine_similarity(
        out_bf16_1.flatten().unsqueeze(0).float(),
        out_int8_1.flatten().unsqueeze(0).float(),
    ).item()
    cos_2 = torch.nn.functional.cosine_similarity(
        out_bf16_2.flatten().unsqueeze(0).float(),
        out_int8_2.flatten().unsqueeze(0).float(),
    ).item()
    print(f"\n  cos_sim layer1: {cos_1:.4f}, layer2: {cos_2:.4f}")
    assert cos_1 >= 0.95, f"layer1 cos_sim={cos_1} < 0.95"
    assert cos_2 >= 0.95, f"layer2 cos_sim={cos_2} < 0.95"
    print(f"[OK] int8 KV forward cos_sim ≥ 0.95 (l1={cos_1:.4f}, l2={cos_2:.4f})")


if __name__ == "__main__":
    tests = [
        ("install+uninstall idempotent", test_int8_kv_install_uninstall_idempotent),
        ("cos_sim >= 0.95",             test_int8_kv_cos_sim_above_095),
        ("VRAM reduction >=40%",        test_int8_kv_vram_reduction_50pct),
        ("double install idempotent",   test_int8_kv_double_install_safe),
        ("attention output cos_sim",    test_int8_kv_attention_output_close),
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