"""
test_infllm_integration.py — TDD-rot: Forward-Hook-Integration (Phase C)
==========================================================================

Phase C pinnt:
  - test_infllm_install_uninstall_idempotent: install + remove hinterlässt
    model unverändert (forward == original)
  - test_infllm_install_returns_count: install_infllm_hooks returnt die Anzahl
    der gepatchten Attention-Layer
  - test_infllm_double_install_safe: zweimal install → kein Crash, idempotent
  - test_infllm_attention_forward_returns_shape: einzelner Aufruf der
    forward_with_infllm Methode produziert Output mit erwarteter Shape
  - test_infllm_uses_cache_when_provided: wenn past_key_values=None ist,
    wird der InfLLM-Cache als past_key_values genutzt (in unserem Patch)

Diese Tests mocken ein Gemma3Attention-Modul (kein echtes 4b-Download).
Wir testen die INTEGRATION-LOGIK, nicht die Modell-Korrektheit.

Run:
    /path/to/venv/bin/python test_infllm_integration.py
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

def test_infllm_install_uninstall_idempotent():
    """install + remove → model unverändert (forward == original)."""
    from infllm_integration import install_infllm_hooks, remove_infllm_hooks

    # Mock-Model mit einem Gemma3Attention-ähnlichen Modul
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

        def original_forward(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states), None)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MockAttention()

    model = MockModel()
    # Snapshot: vor Patch ist m.forward = m._call_impl → wir merken uns das.
    # nn.Module.__getattr__ macht jedes Mal ein neues bound method, daher
    # können wir nicht `m.forward is original_forward` prüfen. Stattdessen:
    # (a) das Vorhandensein von `_original_forward` vor dem Patch
    # (b) das Verschwinden nach uninstall

    # Cache mock (es muss `prepare_reattention` haben — wir geben ihm eine no-op)
    class MockCache:
        def __init__(self):
            self.calls = 0
        def prepare_reattention(self, q, k, v, layer_idx, rotary_emb_module, **kwargs):
            self.calls += 1
            return q, k, v

    cache = MockCache()

    # Install
    n_patched = install_infllm_hooks(model, cache)
    assert n_patched == 1, f"expected 1 patched, got {n_patched}"
    # _original_forward wurde gespeichert
    assert hasattr(model.attn, "_original_forward"), "_original_forward not saved"

    # Uninstall
    n_removed = remove_infllm_hooks(model)
    assert n_removed == 1, f"expected 1 removed, got {n_removed}"
    # _original_forward muss weg sein
    assert not hasattr(model.attn, "_original_forward"), "_original_forward leaked"
    # forward muss noch funktionieren (nn.Module-default)
    try:
        x = torch.randn(2, 3)
        # nn.Module default forward ruft _forward_unimplemented auf bei
        # unbekanntem arg — wir testen nur dass das Attribut exisiert.
        _ = model.attn.forward
    except Exception as e:
        raise AssertionError(f"forward not callable after restore: {e}")

    print("[OK] install+uninstall idempotent (forward restored, no leak)")


def test_infllm_install_returns_count():
    """install_infllm_hooks returnt Anzahl der gepatchten Layer."""
    from infllm_integration import install_infllm_hooks, remove_infllm_hooks

    class MockAttention(nn.Module):
        def __init__(self, idx=0):
            super().__init__()
            self.layer_idx = idx
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

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = MockAttention(0)
            self.attn2 = MockAttention(1)
            # Non-attention submodule (should NOT be patched)
            self.lin = nn.Linear(16, 16)

    model = MockModel()

    class MockCache:
        def prepare_reattention(self, q, k, v, layer_idx, **kwargs):
            return q, k, v

    cache = MockCache()
    n = install_infllm_hooks(model, cache)
    assert n == 2, f"expected 2 patched, got {n}"
    # Non-attention submodule untouched
    assert not hasattr(model.lin, "_original_forward"), "non-attn got patched"
    remove_infllm_hooks(model)
    print(f"[OK] install returned n={n} (2 attn + 1 lin; only attn patched)")


def test_infllm_double_install_safe():
    """Zweimal install → kein Crash, idempotent (Registry dedupliziert)."""
    from infllm_integration import install_infllm_hooks, remove_infllm_hooks

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

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MockAttention()

    model = MockModel()
    cache = type("C", (), {"prepare_reattention": lambda s, q, k, v, **kw: (q, k, v)})()
    n1 = install_infllm_hooks(model, cache)
    n2 = install_infllm_hooks(model, cache)
    assert n1 == n2 == 1, f"expected 1/1, got {n1}/{n2}"
    # Cleanup
    remove_infllm_hooks(model)
    print(f"[OK] double install idempotent (n1={n1}, n2={n2})")


def test_infllm_attention_forward_returns_shape():
    """Ein forward-Call der patched Attention produziert Output mit erwarteter Shape.

    Wir mocken _alles_: q_proj/k_proj/v_proj/o_proj sind echt, rotary_emb
    ist trivial, past_key_values=None → Cache wird genutzt."""
    from infllm_integration import install_infllm_hooks, remove_infllm_hooks

    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 8  # head_dim
            # Gemma3Attention: q_proj → num_heads * head_dim, k/v_proj → num_kv_heads * head_dim
            # Für mock vereinfachen wir: alle projizieren auf 16.
            self.q_proj = nn.Linear(32, 16, bias=False)  # 2 heads * 8 head_dim
            self.k_proj = nn.Linear(32, 16, bias=False)
            self.v_proj = nn.Linear(32, 16, bias=False)
            self.o_proj = nn.Linear(16, 32, bias=False)
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

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MockAttention()

    model = MockModel()

    # Echter InfLLMCache (aber trivial)
    from infinite_context import InfLLMCache
    class _Cfg:
        num_hidden_layers = 1
    cache = InfLLMCache(_Cfg(), block_size=4, r_tokens=2, sinks_count=2)

    install_infllm_hooks(model, cache)
    try:
        hidden_states = torch.randn(1, 5, 32)  # [B, T, hidden]
        out, attn_weights = model.attn(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None)
        # Output shape muss [B, T, hidden] sein (= input hidden_states.shape)
        assert out.shape == hidden_states.shape, (
            f"output shape {out.shape} != input {hidden_states.shape}")
        # InfLLM-Cache wurde durch prepare_reattention gefüllt: Sinks existieren
        assert cache.sinks_k[0] is not None, "cache.sinks_k[0] not set after first call"
        assert cache.sinks_k[0].shape[-2] == 2, (
            f"cache.sinks_k[0] shape {cache.sinks_k[0].shape}, expected 2 sinks")
        print(f"[OK] forward produces shape {out.shape}, cache.sinks filled")
    finally:
        remove_infllm_hooks(model)


def test_infllm_uses_cache_when_provided():
    """Wenn past_key_values=None ist, wird der Cache genutzt (NICHT ein
    neuer Cache erstellt). Wir pinnen: der SELBE Cache wird 2 Calls
    hintereinander gefüllt, NICHT ein neuer pro Call."""
    from infllm_integration import install_infllm_hooks, remove_infllm_hooks

    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = 8
            self.q_proj = nn.Linear(32, 16, bias=False)
            self.k_proj = nn.Linear(32, 16, bias=False)
            self.v_proj = nn.Linear(32, 16, bias=False)
            self.o_proj = nn.Linear(16, 32, bias=False)
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

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MockAttention()

    model = MockModel()
    from infinite_context import InfLLMCache
    class _Cfg:
        num_hidden_layers = 1
    # block_size=8 damit nicht sofort archiviert wird
    cache = InfLLMCache(_Cfg(), block_size=8, r_tokens=2, sinks_count=2)

    install_infllm_hooks(model, cache)
    try:
        # Zwei aufeinanderfolgende Calls (2 tokens je call = 4 total, < 8)
        h = torch.randn(1, 2, 32)
        out1, _ = model.attn(h, position_embeddings=None, attention_mask=None, past_key_values=None)
        buf_after_first = sum(t.size(-2) for t in cache.buffer_k[0])
        out2, _ = model.attn(h, position_embeddings=None, attention_mask=None, past_key_values=None)
        buf_after_second = sum(t.size(-2) for t in cache.buffer_k[0])

        # Cache-Buffer muss wachsen (jeder Call append'ed 2 tokens in den Buffer)
        assert buf_after_second > buf_after_first, (
            f"buffer wächst nicht: {buf_after_first} → {buf_after_second}")
        print(f"[OK] cache buffer wächst: {buf_after_first} → {buf_after_second}")
    finally:
        remove_infllm_hooks(model)


def test_infllm_remove_returns_zero_when_not_installed():
    """remove_infllm_hooks auf ungepatchtem model → return 0, kein Crash."""
    from infllm_integration import remove_infllm_hooks, is_infllm_installed

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

    model = MockModel()
    assert not is_infllm_installed(model), "is_infllm_installed sollte False sein"
    n = remove_infllm_hooks(model)
    assert n == 0, f"expected 0, got {n}"
    print("[OK] remove_infllm_hooks on unpatched model returns 0 safely")


if __name__ == "__main__":
    tests = [
        ("install+uninstall idempotent", test_infllm_install_uninstall_idempotent),
        ("install returns count",       test_infllm_install_returns_count),
        ("double install idempotent",   test_infllm_double_install_safe),
        ("forward shape preserved",     test_infllm_attention_forward_returns_shape),
        ("cache grows over calls",      test_infllm_uses_cache_when_provided),
        ("remove unpatched safe",       test_infllm_remove_returns_zero_when_not_installed),
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