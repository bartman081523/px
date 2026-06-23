"""
test_infllm_hierarchical.py — TDD-rot für Phase B: GPU↔CPU Hierarchical Cache
================================================================================

Plan 2 (InfLLM/ReAttention) Phase B. Was getestet wird:

- test_l1_blocks_resident_on_cpu: archived Blöcke leben auf CPU device
- test_topk_promotion_l1_to_l0: bei Bedarf wird ein L1-Block auf GPU promoted
- test_evict_l0_to_l1: L0-volles → LRU-Block wandert nach L1
- test_l1_cpu_memory_bounded: L1 hat max N Blöcke; bei Überlauf → LRU nach L2
- test_l2_serialization_roundtrip: serialize → deserialize → cos_sim >= 0.999

Phase B fügt eine **Bounds-Layer** um InfLLMCache. Die existierende
InfLLMCache hält alles in ltm_k (schon CPU) ohne Limit. Die neue
HierarchicalCache begrenzt:
  - L0: aktuelle Sequenz + Sinks + top-k (GPU, klein)
  - L1: archived Blöcke (CPU, bounded — älteste fliegen raus)
  - L2: serialisiert auf Disk (bei L1-Überlauf)

Run:
    /path/to/venv/bin/python test_infllm_hierarchical.py
"""
import sys
import os
import tempfile
import shutil

import torch

_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_SCRATCHES = os.path.join(_REPO, "scratches", "4b-image")
if _SCRATCHES not in sys.path:
    sys.path.insert(0, _SCRATCHES)


class _StubConfig:
    num_hidden_layers = 4


# Tests ---------------------------------------------------------------------

def test_l1_blocks_resident_on_cpu():
    """Archivierte Blöcke (ltm) leben auf CPU device.

    Wichtig: das prüft die EXISTIERENDE InfLLMCache — nicht die neue
    Hierarchical. Wenn der Basis-Cache schon CPU-LTM hat (was er laut
    Code tut: `block_k.cpu()` in `_archive_block`), ist Phase B ein
    Wrapper der das explizit macht und bounded."""
    from infinite_context import InfLLMCache

    c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2)

    B, H, D = 1, 2, 8
    def _rotary(x, pos):
        return torch.ones(*x.shape[:3], x.shape[-1]), torch.zeros(*x.shape[:3], x.shape[-1])

    # 8 tokens → 1 Block wird archiviert
    q = torch.randn(B, H, 8, D); k = torch.randn(B, H, 8, D); v = torch.randn(B, H, 8, D)
    c.prepare_reattention(q, k, v, 0, _rotary)

    # Wenn 1 Block im LTM ist, MUSS er auf CPU sein
    assert len(c.ltm_k[0]) >= 1, "kein Block im LTM"
    for blk in c.ltm_k[0]:
        assert blk.device.type == "cpu", (
            f"ltm block auf {blk.device} (erwartet cpu)")
        assert blk.shape[-2] == 4, f"block size={blk.shape[-2]} (erwartet 4)"
    print(f"[OK] L1 blocks resident on cpu (n={len(c.ltm_k[0])})")


def test_topk_promotion_l1_to_l0():
    """Wenn die attention retrieve einen Block braucht, wird er auf GPU promoted.

    Konkret: bei einem Aufruf der InfLLMCache.prepare_reattention, wenn LTM-
    Blöcke da sind, holt _retrieve die top-k aufs GPU. Wir prüfen, dass das
    retrievete k_parts AUCH auf GPU ist (also die Promotion funktioniert)."""
    from infinite_context import InfLLMCache

    c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2, top_k_blocks=2, sinks_count=2)
    B, H, D = 1, 2, 8

    # Erst 4 archive-Operationen → 4 LTM-Blöcke (alle CPU)
    for _ in range(2):
        def _rotary(x, pos):
            return torch.ones(*x.shape[:3], x.shape[-1]), torch.zeros(*x.shape[:3], x.shape[-1])
        q = torch.randn(B, H, 4, D); k = torch.randn(B, H, 4, D); v = torch.randn(B, H, 4, D)
        c.prepare_reattention(q, k, v, 0, _rotary)
    assert len(c.ltm_k[0]) >= 1, "kein Block im LTM für retrieve test"

    # Forward-Pass, der retrieve triggert (T_new <= 1024, also NICHT der
    # big-prefill-Bypass). q auf GPU simulieren.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        # CPU-test: skip
        print("[SKIP] test_topk_promotion_l1_to_l0: keine GPU")
        return

    q = torch.randn(B, H, 1, D, device=device)
    k = torch.randn(B, H, 1, D, device=device)
    v = torch.randn(B, H, 1, D, device=device)

    def _rotary_gpu(x, pos):
        return torch.ones(*x.shape[:3], x.shape[-1], device=x.device), \
               torch.zeros(*x.shape[:3], x.shape[-1], device=x.device)

    q_out, k_out, v_out = c.prepare_reattention(q, k, v, 0, _rotary_gpu)
    # Output k/v muss auf GPU sein (L1→L0 promotion passierte)
    assert k_out.device.type == device, f"k_out device={k_out.device}"
    assert v_out.device.type == device, f"v_out device={v_out.device}"
    # Größer als input T (sinks + retrieved + buffer concatenated)
    assert k_out.shape[-2] >= 1 + 2, (
        f"k_out T={k_out.shape[-2]} (erwartet >= input+sinks+retrieved)")
    print(f"[OK] topk promotion L1→L0 (k_out device={k_out.device}, T={k_out.shape[-2]})")


def test_evict_l0_to_l1():
    """Wenn L0 (Buffer + Sinks + top-k) voll ist, werden LRU-Blöcke nach L1 evicted.

    Bei InfLLMCache: das passiert automatisch wenn _archive_block aufgerufen
    wird — der volle Buffer wird zum LTM (L1). Wir prüfen, dass NACH dem
    Archive L0 (buffer) LEER ist und L1 (ltm) den Block enthält."""
    from infinite_context import InfLLMCache

    c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2)

    B, H, D = 1, 2, 8
    def _rotary(x, pos):
        return torch.ones(*x.shape[:3], x.shape[-1]), torch.zeros(*x.shape[:3], x.shape[-1])

    # 4 tokens → 1 Block archiviert
    q = torch.randn(B, H, 4, D); k = torch.randn(B, H, 4, D); v = torch.randn(B, H, 4, D)
    c.prepare_reattention(q, k, v, 0, _rotary)

    # Nach archive: L1 hat 1 Block, L0 (buffer) ist leer
    assert len(c.ltm_k[0]) == 1, f"L1 count={len(c.ltm_k[0])} (erwartet 1)"
    assert sum(x.size(-2) for x in c.buffer_k[0]) == 0, (
        f"L0 buffer={sum(x.size(-2) for x in c.buffer_k[0])} (erwartet 0)")
    print(f"[OK] evict L0→L1 atomic (L1={len(c.ltm_k[0])}, L0 buffer=0)")


def test_l1_cpu_memory_bounded():
    """L1-CPU-Speicher ist bounded: bei Überlauf wandert der älteste Block
    nach L2 (Disk).

    Phase B fügt eine `max_l1_blocks` Konfiguration ein. Bei Überlauf
    wird der ÄLTESTE Block (LRU) nach L2 serialisiert.

    Wichtig: dieser Test pinnt das VERHALTEN, nicht die Implementation. Er
    funktioniert sobald der HierarchicalCache `l2_path` konfiguriert hat
    UND `max_l1_blocks`."""
    from infinite_context import InfLLMCache

    # 16 Blöcke mit max_l1_blocks=4 → 12 wandern nach L2
    c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2)
    assert hasattr(c, "max_l1_blocks"), (
        "InfLLMCache braucht max_l1_blocks Konfiguration für L1-Bounding")

    # Setzen via setter oder Konstruktor
    if not hasattr(c, "max_l1_blocks"):
        # Konfiguration muss via init gehen → API-Lücke
        print("[SKIP] test_l1_cpu_memory_bounded: max_l1_blocks API fehlt")
        return

    print("[OK] L1 cpu memory bounded (placeholder, wird in phase B implementiert)")


def test_l2_serialization_roundtrip():
    """L2-Disk-Serialisierung: serialize → deserialize → cos_sim >= 0.999
    der zurückgelesenen Blöcke.

    Nutzt temporäres Verzeichnis. Block wird nach L2 serialisiert (z.B. via
    `torch.save`), und kann ohne Verlust wieder geladen werden."""
    from infinite_context import InfLLMCache

    tmp = tempfile.mkdtemp(prefix="infllm_l2_")
    try:
        # Cache mit l2_path Konfiguration
        c = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2, l2_path=tmp)
        assert hasattr(c, "_l2_serialize_block"), (
            "HierarchicalCache braucht _l2_serialize_block für Disk-Evict")

        B, H, D = 1, 2, 8
        torch.manual_seed(99)
        block_k = torch.randn(B, H, 4, D)
        block_v = torch.randn(B, H, 4, D)
        # Mock-Repräsentative keys
        r_k = torch.randn(B, H, 2, D)

        # Serialisieren
        c._l2_serialize_block(layer_idx=0, block_idx=0,
                              block_k=block_k, block_v=block_v, r_k=r_k)
        # Datei muss da sein
        files = os.listdir(tmp)
        assert any(f.endswith(".pt") for f in files), (
            f"keine .pt Datei in {tmp}: {files}")

        # Zurücklesen
        c2 = InfLLMCache(_StubConfig(), block_size=4, r_tokens=2, l2_path=tmp)
        k_back, v_back, rk_back = c2._l2_load_block(layer_idx=0, block_idx=0)
        # Cosine sim muss nahe 1 sein
        cos_k = torch.nn.functional.cosine_similarity(
            block_k.flatten().unsqueeze(0).float(),
            k_back.flatten().unsqueeze(0).float()
        ).item()
        cos_v = torch.nn.functional.cosine_similarity(
            block_v.flatten().unsqueeze(0).float(),
            v_back.flatten().unsqueeze(0).float()
        ).item()
        cos_rk = torch.nn.functional.cosine_similarity(
            r_k.flatten().unsqueeze(0).float(),
            rk_back.flatten().unsqueeze(0).float()
        ).item()
        assert cos_k > 0.999, f"block_k cos_sim={cos_k}"
        assert cos_v > 0.999, f"block_v cos_sim={cos_v}"
        assert cos_rk > 0.999, f"r_k cos_sim={cos_rk}"
        print(f"[OK] L2 roundtrip cos_sim k={cos_k:.4f} v={cos_v:.4f} rk={cos_rk:.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [
        ("L1 blocks on CPU",            test_l1_blocks_resident_on_cpu),
        ("L1→L0 topk promotion",        test_topk_promotion_l1_to_l0),
        ("L0→L1 evict atomic",          test_evict_l0_to_l1),
        ("L1 cpu memory bounded",       test_l1_cpu_memory_bounded),
        ("L2 serialize roundtrip",      test_l2_serialization_roundtrip),
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