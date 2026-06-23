"""
test_infllm_api_gaps.py — TDD-rot: API-Lücken in InfLLMCache
==============================================================

Phase A (Inventory + API-Re-Check) für Plan 2 (InfLLM/ReAttention).

Was diese Tests pinnen (per PLAN_INFINITE_CONTEXT.md Phase A):
  - test_infllm_cache_initial_state_empty: neue Cache hat leere buffers, sinks=None
  - test_infllm_cache_archive_block_atomicity: archive ist atomar
  - test_infllm_cache_topk_selection_stable: gleiche Keys → gleiche top-k (deterministic)
  - test_sinks_persist_after_eviction: LTM-Block-Eviction darf sinks nicht antasten
  - test_infllm_cache_from_kv_cache: API `from_kv_cache(k_cache, v_cache)` (FEHLT)
  - test_infllm_cache_evict_block: API `evict_block(idx)` (FEHLT)
  - test_infllm_cache_serialize_deserialize: API `serialize()/deserialize()` (FEHLT)

Die letzten drei schlagen fehl (TDD-rot) bis wir die API implementieren.
Erste vier sollten schon grün sein (per existierender Implementierung).

Run:
    /path/to/venv/bin/python test_infllm_api_gaps.py
"""
import sys
import os

import torch

_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class _StubConfig:
    num_hidden_layers = 4


# Tests ---------------------------------------------------------------------

def test_infllm_cache_initial_state_empty():
    """Neue Cache hat leere buffers, sinks=None, seen_tokens=0."""
    from infinite_context import InfLLMCache

    c = InfLLMCache(_StubConfig(), block_size=8, r_tokens=2)
    for layer in range(_StubConfig.num_hidden_layers):
        assert c.ltm_k[layer] == [], f"layer {layer} ltm_k nicht leer"
        assert c.ltm_v[layer] == [], f"layer {layer} ltm_v nicht leer"
        assert c.ltm_rk[layer] == [], f"layer {layer} ltm_rk nicht leer"
        assert c.buffer_k[layer] == [], f"layer {layer} buffer_k nicht leer"
        assert c.buffer_v[layer] == [], f"layer {layer} buffer_v nicht leer"
        assert c.sinks_k[layer] is None, f"layer {layer} sinks_k nicht None"
        assert c.sinks_v[layer] is None, f"layer {layer} sinks_v nicht None"
    assert c.seen_tokens == 0, f"seen_tokens={c.seen_tokens}"
    print("[OK] initial state empty (ltm/buffer/sinks all empty)")


def test_infllm_cache_archive_block_atomicity():
    """Archive ist atomar: entweder ein voller Block drin oder keiner.

    Bei block_size=4, T=2 (kleiner als block) DARF kein partial-Block im LTM
    sein. Nach exakt 4 Tokens muss GENAU 1 Block da sein, nach 5 Tokens immer
    noch 1 (Rest im Buffer)."""
    from infinite_context import InfLLMCache

    c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2)

    B, H, D = 1, 2, 8
    torch.manual_seed(42)

    def _rotary(x, pos):
        return torch.ones(*x.shape[:3], x.shape[-1], device=x.device), \
               torch.zeros(*x.shape[:3], x.shape[-1], device=x.device)

    # 1 token (kein archive)
    q = torch.randn(B, H, 1, D); k = torch.randn(B, H, 1, D); v = torch.randn(B, H, 1, D)
    c.prepare_reattention(q, k, v, 0, _rotary)
    assert len(c.ltm_k[0]) == 0, f"nach 1 token: ltm_k={len(c.ltm_k[0])} (erwartet 0)"

    # 3 tokens (total 4 → archive fires because buffer_len >= block_size)
    for _ in range(3):
        q = torch.randn(B, H, 1, D); k = torch.randn(B, H, 1, D); v = torch.randn(B, H, 1, D)
        c.prepare_reattention(q, k, v, 0, _rotary)
    # Bei _archive_block werden ALLE Buffer-Einträge verarbeitet: ein voller
    # Block (4 tokens) → archive, der leere Rest (0 tokens) bleibt im Buffer.
    assert len(c.ltm_k[0]) == 1, f"nach 4 tokens: ltm_k={len(c.ltm_k[0])} (erwartet 1)"
    # LTM-Block muss VOLL sein (4 Tokens)
    assert c.ltm_k[0][0].shape[-2] == 4, f"ltm_block size={c.ltm_k[0][0].shape[-2]} (erwartet 4)"
    # Rest muss LEER sein (0 tokens)
    assert sum(x.size(-2) for x in c.buffer_k[0]) == 0, (
        f"buffer_rest={sum(x.size(-2) for x in c.buffer_k[0])} (erwartet 0)")

    # 1 token mehr (5 total → buffer=0+1=1 < block=4 → kein archive)
    q = torch.randn(B, H, 1, D); k = torch.randn(B, H, 1, D); v = torch.randn(B, H, 1, D)
    c.prepare_reattention(q, k, v, 0, _rotary)
    assert len(c.ltm_k[0]) == 1, f"nach 5 tokens: ltm_k={len(c.ltm_k[0])} (erwartet 1, kein neuer Block)"
    assert sum(x.size(-2) for x in c.buffer_k[0]) == 1, (
        f"buffer_rest={sum(x.size(-2) for x in c.buffer_k[0])} (erwartet 1)")

    # 3 weitere tokens (8 total → buffer=1+3=4 ≥ block=4 → archive)
    for _ in range(3):
        q = torch.randn(B, H, 1, D); k = torch.randn(B, H, 1, D); v = torch.randn(B, H, 1, D)
        c.prepare_reattention(q, k, v, 0, _rotary)
    assert len(c.ltm_k[0]) == 2, f"nach 8 tokens: ltm_k={len(c.ltm_k[0])} (erwartet 2)"

    print("[OK] archive atomicity (no partial blocks; full-block or nothing)")


def test_infllm_cache_topk_selection_stable():
    """Gleiche Keys → gleiche top-k Selektion (deterministic)."""
    from infinite_context import InfLLMCache

    c1 = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2, top_k_blocks=2)
    c2 = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2, top_k_blocks=2)

    B, H, T, D = 1, 2, 4, 8
    torch.manual_seed(7)
    # 4 archive-blocks mit unterschiedlichen representative keys
    for _ in range(2):
        q = torch.randn(B, H, 4, D); k = torch.randn(B, H, 4, D); v = torch.randn(B, H, 4, D)

        def _rotary(x, pos):
            return torch.ones(*x.shape[:3], x.shape[-1]), torch.zeros(*x.shape[:3], x.shape[-1])

        c1.prepare_reattention(q, k, v, 0, _rotary)
        c2.prepare_reattention(q, k, v, 0, _rotary)

    # Beide Caches sollten identische ltm_rk haben (gleiche seed → gleiche Normen)
    for i in range(len(c1.ltm_rk[0])):
        assert torch.allclose(c1.ltm_rk[0][i], c2.ltm_rk[0][i]), (
            f"ltm_rk[{i}] divergiert zwischen identischen seeds")

    # retrieve sollte deterministisch gleiche Indizes liefern
    q = torch.randn(B, H, 1, D)

    def _rotary(x, pos):
        return torch.ones(*x.shape[:3], x.shape[-1]), torch.zeros(*x.shape[:3], x.shape[-1])

    # Direkter _retrieve-Test
    ret1 = c1._retrieve(0, q)
    ret2 = c2._retrieve(0, q)
    # Vergleiche ret_k (cosine sim der konkatenierten Blöcke)
    assert ret1[0].shape == ret2[0].shape, "retrieve shape divergiert"
    # Inhalt sollte ähnlich sein (gleiche Selektion)
    if ret1[0].numel() > 0:
        cos = torch.nn.functional.cosine_similarity(
            ret1[0].flatten().unsqueeze(0).float(),
            ret2[0].flatten().unsqueeze(0).float()
        ).item()
        assert cos > 0.99, f"retrieve divergiert: cos_sim={cos}"

    print("[OK] topk selection deterministic across runs")


def test_sinks_persist_after_eviction():
    """Sinks bleiben nach Block-Archive/Eviction unverändert."""
    from infinite_context import InfLLMCache

    c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2, sinks_count=2)
    B, H, D = 1, 2, 8

    def _rotary(x, pos):
        return torch.ones(*x.shape[:3], x.shape[-1]), torch.zeros(*x.shape[:3], x.shape[-1])

    # Erste Call: 8 tokens → 1 Block archiviert, 2 Sinks da
    q = torch.randn(B, H, 8, D); k = torch.randn(B, H, 8, D); v = torch.randn(B, H, 8, D)
    c.prepare_reattention(q, k, v, 0, _rotary)
    sinks_before_k = c.sinks_k[0].clone()
    sinks_before_v = c.sinks_v[0].clone()

    # Mehr Calls: 16 weitere Tokens → 2+ Blöcke archiviert
    for _ in range(16):
        q = torch.randn(B, H, 1, D); k = torch.randn(B, H, 1, D); v = torch.randn(B, H, 1, D)
        c.prepare_reattention(q, k, v, 0, _rotary)

    # Sinks müssen byte-identisch sein
    assert torch.equal(c.sinks_k[0], sinks_before_k), "sinks_k nach eviction verändert"
    assert torch.equal(c.sinks_v[0], sinks_before_v), "sinks_v nach eviction verändert"
    print(f"[OK] sinks persist (ltm_blöcke={len(c.ltm_k[0])}, sinks={c.sinks_k[0].shape[-2]})")


def test_infllm_cache_from_kv_cache():
    """API `from_kv_cache(k_cache, v_cache)` füllt Cache aus existierendem KV.

    k_cache/v_cache: [n_layers][B, H, T, D] (z.B. aus HF transformers Cache).
    Erwartet: nach Aufruf ist ltm_k mit den Daten gefüllt (oder Buffer,
    was auch immer sinnvoll ist). Wichtig: KEIN Datenverlust."""
    from infinite_context import InfLLMCache

    c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2)
    n_layers = _StubConfig.num_hidden_layers

    # n_layers × [B=1, H=2, T=16, D=8]
    k_cache = [torch.randn(1, 2, 16, 8) for _ in range(n_layers)]
    v_cache = [torch.randn(1, 2, 16, 8) for _ in range(n_layers)]

    # API muss existieren
    assert hasattr(c, "from_kv_cache"), "InfLLMCache fehlt from_kv_cache() API"
    c.from_kv_cache(k_cache, v_cache)
    # Nach Aufruf: alle Schichten müssen gefüllt sein
    total_tokens = 0
    for layer in range(n_layers):
        # Wir akzeptieren entweder ltm ODER buffer (Implementation-Wahl)
        n = len(c.ltm_k[layer]) * c.block_size + sum(x.size(-2) for x in c.buffer_k[layer])
        total_tokens += n
    expected = n_layers * 16
    assert total_tokens == expected, (
        f"erwartet {expected} tokens ({n_layers} layer × 16), gefunden {total_tokens}")
    print(f"[OK] from_kv_cache loaded {total_tokens} tokens across {n_layers} layers")


def test_infllm_cache_evict_block():
    """API `evict_block(layer_idx, block_idx)` entfernt einen LTM-Block."""
    from infinite_context import InfLLMCache

    c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2)

    B, H, D = 1, 2, 8
    def _rotary(x, pos):
        return torch.ones(*x.shape[:3], x.shape[-1]), torch.zeros(*x.shape[:3], x.shape[-1])

    # 16 tokens → 4 Blöcke
    for _ in range(4):
        q = torch.randn(B, H, 4, D); k = torch.randn(B, H, 4, D); v = torch.randn(B, H, 4, D)
        c.prepare_reattention(q, k, v, 0, _rotary)

    n_before = len(c.ltm_k[0])
    assert n_before == 4, f"erwartet 4 Blöcke, gefunden {n_before}"

    # API muss existieren
    assert hasattr(c, "evict_block"), "InfLLMCache fehlt evict_block() API"
    c.evict_block(layer_idx=0, block_idx=1)
    n_after = len(c.ltm_k[0])
    assert n_after == n_before - 1, f"nach evict: {n_after} (erwartet {n_before - 1})"
    print(f"[OK] evict_block removed block (ltm {n_before} → {n_after})")


def test_infllm_cache_serialize_deserialize():
    """API `serialize()/deserialize()` muss Cache-Zustand runden.

    serialize → bytes/string; deserialize → identischer Cache-State.
    ltm_k, ltm_v, ltm_rk, buffer_k, buffer_v, sinks müssen (qualitativ)
    gleich sein."""
    from infinite_context import InfLLMCache

    c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2, sinks_count=2)

    B, H, D = 1, 2, 8
    def _rotary(x, pos):
        return torch.ones(*x.shape[:3], x.shape[-1]), torch.zeros(*x.shape[:3], x.shape[-1])

    # Zustand aufbauen: 12 tokens (2 Blöcke + 4 im Buffer)
    q = torch.randn(B, H, 8, D); k = torch.randn(B, H, 8, D); v = torch.randn(B, H, 8, D)
    c.prepare_reattention(q, k, v, 0, _rotary)  # 8 tokens: 1 Block + Sinks
    q = torch.randn(B, H, 4, D); k = torch.randn(B, H, 4, D); v = torch.randn(B, H, 4, D)
    c.prepare_reattention(q, k, v, 0, _rotary)  # +4 tokens: 1 Block + 4 im Buffer

    # APIs müssen existieren
    assert hasattr(c, "serialize"), "InfLLMCache fehlt serialize() API"
    assert hasattr(c, "deserialize"), "InfLLMCache fehlt deserialize() API"

    blob = c.serialize()
    c2 = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2, sinks_count=2)
    c2.deserialize(blob)

    # Sinks byte-identisch
    assert c2.sinks_k[0].shape == c.sinks_k[0].shape, "sinks_k shape divergiert"
    assert torch.allclose(c2.sinks_k[0], c.sinks_k[0]), "sinks_k values divergieren"
    assert torch.allclose(c2.sinks_v[0], c.sinks_v[0]), "sinks_v values divergieren"
    # Anzahl LTM-Blöcke gleich
    assert len(c2.ltm_k[0]) == len(c.ltm_k[0]), (
        f"ltm_k count divergiert: {len(c2.ltm_k[0])} vs {len(c.ltm_k[0])}")
    assert len(c2.ltm_rk[0]) == len(c.ltm_rk[0]), (
        f"ltm_rk count divergiert")
    # Buffer-Inhalt: Token-Count gleich
    n_buf_orig = sum(x.size(-2) for x in c.buffer_k[0])
    n_buf_new = sum(x.size(-2) for x in c2.buffer_k[0])
    assert n_buf_orig == n_buf_new, f"buffer count divergiert: {n_buf_new} vs {n_buf_orig}"
    print(f"[OK] serialize/deserialize roundtrip clean (ltm={len(c2.ltm_k[0])}, buf={n_buf_new})")


if __name__ == "__main__":
    tests = [
        ("initial state empty",          test_infllm_cache_initial_state_empty),
        ("archive atomicity",            test_infllm_cache_archive_block_atomicity),
        ("topk selection deterministic", test_infllm_cache_topk_selection_stable),
        ("sinks persist eviction",       test_sinks_persist_after_eviction),
        ("from_kv_cache API",            test_infllm_cache_from_kv_cache),
        ("evict_block API",              test_infllm_cache_evict_block),
        ("serialize/deserialize",        test_infllm_cache_serialize_deserialize),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except AssertionError as e:
            print(f"[FAIL] {name}: AssertionError: {e}")
            failed += 1
        except AttributeError as e:
            print(f"[FAIL] {name}: AttributeError: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)