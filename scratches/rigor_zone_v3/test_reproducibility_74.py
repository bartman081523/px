"""
test_reproducibility_74.py — TDD-Tests für run_reproducibility_phase
====================================================================
Prüft: 74 echte Repro-Runs werden erzeugt, jeder mit echtem Model-Output (kein Placeholder).
"""
import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")


def test_reproducibility_phase_writes_real_outputs():
    """run_reproducibility_phase erzeugt n_repro echte Output-Files (kein Placeholder)."""
    from rigor_harness_v3 import run_reproducibility_phase, WorkItem, PreprocessContext, DEFAULT_REPRO_SEED, ARM_CONFIGS

    # Setup: 4 WorkItems mit seed=42 (DEFAULT_REPRO_SEED - 1 = 42)
    items = []
    for i in range(4):
        items.append(WorkItem(
            task_id=f"task_{i:03d}",
            arm_name="active_manifold",
            arm_preset="ACTIVE_MANIFOLD",
            rigor_mephisto=False,
            mephisto_scale=None,
            with_search=False,
            seed=DEFAULT_REPRO_SEED - 1,  # 42
            category="math",
            ground_truth="42",
            prompt=f"What is {i}+{i}?",
        ))
    ctx = MagicMock()
    ctx.work_items = items
    ctx.batch_size = 1
    ctx.max_new_tokens = 50
    ctx.model_cache = {}
    ctx.out_dir = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out"
    ctx.search_cache = MagicMock()

    # Mock run_gpu_loop → returnt deterministische Outputs
    with patch("rigor_harness_v3.run_gpu_loop") as mock_gpu:
        mock_gpu.side_effect = lambda *a, **kw: [
            {"text": f"repro_output_for_{kw.get('seed', '?')}_{a[3][0][:20] if a[3] else '?'}",
             "n_input_tokens": 10, "n_output_tokens": 5}
        ]
        repro_results = run_reproducibility_phase(ctx, n_repro=4)

    # Assertions
    assert len(repro_results) == 4, f"Erwartet 4 Repro-Outputs, bekam {len(repro_results)}"
    for rec in repro_results:
        assert "placeholder" not in rec.get("output_text", ""), f"Placeholder gefunden: {rec}"
        assert rec.get("seed") == DEFAULT_REPRO_SEED
        assert "output_text" in rec
        assert "output_hash" in rec
    print(f"  ✓ 4 echte Repro-Outputs (kein Placeholder)")


def test_reproducibility_files_written():
    """Reproducibility-Files werden in out/reproducibility/ geschrieben."""
    from rigor_harness_v3 import run_reproducibility_phase, WorkItem, PreprocessContext, DEFAULT_REPRO_SEED, ARM_CONFIGS

    out_dir = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out"
    repro_dir = f"{out_dir}/reproducibility"

    # Cleanup
    import shutil
    if os.path.exists(repro_dir):
        shutil.rmtree(repro_dir)

    items = [
        WorkItem(
            task_id=f"repro_task_{i:03d}",
            arm_name="active_manifold",
            arm_preset="ACTIVE_MANIFOLD",
            rigor_mephisto=False,
            mephisto_scale=None,
            with_search=False,
            seed=DEFAULT_REPRO_SEED - 1,
            category="math",
            ground_truth="42",
            prompt=f"Repro test {i}",
        )
        for i in range(3)
    ]
    ctx = MagicMock()
    ctx.work_items = items
    ctx.batch_size = 1
    ctx.max_new_tokens = 50
    ctx.model_cache = {}
    ctx.out_dir = out_dir
    ctx.search_cache = MagicMock()

    with patch("rigor_harness_v3.run_gpu_loop") as mock_gpu:
        mock_gpu.return_value = [
            {"text": "real output", "n_input_tokens": 5, "n_output_tokens": 2}
        ]
        results = run_reproducibility_phase(ctx, n_repro=3)

    files = os.listdir(repro_dir)
    assert len(files) == 3, f"Erwartet 3 Files, bekam {len(files)}: {files}"
    for fn in files:
        with open(f"{repro_dir}/{fn}") as f:
            data = json.load(f)
        assert data["output_text"] == "real output"
        assert "placeholder" not in data
    shutil.rmtree(repro_dir)  # cleanup
    print(f"  ✓ 3 Repro-Files geschrieben: {[f[:40] for f in files]}")


def test_reproducibility_handles_no_primary_items():
    """Wenn keine primary Items vorhanden, fallback auf erste n_repro."""
    from rigor_harness_v3 import run_reproducibility_phase, WorkItem, PreprocessContext, DEFAULT_REPRO_SEED

    # Items mit seed=99 (nicht 42), fallback wird genutzt
    items = [
        WorkItem(
            task_id=f"fb_task_{i}",
            arm_name="baseline",
            arm_preset="BASELINE",
            rigor_mephisto=False,
            mephisto_scale=None,
            with_search=False,
            seed=99,  # ≠ 42
            category="math",
            ground_truth="42",
            prompt=f"Fallback {i}",
        )
        for i in range(2)
    ]
    ctx = MagicMock()
    ctx.work_items = items
    ctx.batch_size = 1
    ctx.max_new_tokens = 50
    ctx.model_cache = {}
    ctx.out_dir = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out"
    ctx.search_cache = MagicMock()
    with patch("rigor_harness_v3.run_gpu_loop") as mock_gpu:
        mock_gpu.return_value = [{"text": "fb", "n_input_tokens": 0, "n_output_tokens": 0}]
        results = run_reproducibility_phase(ctx, n_repro=2)
    assert len(results) == 2
    print(f"  ✓ Fallback auf erste n_repro Items funktioniert")


def main():
    print("=" * 70)
    print("Reproducibility-Phase 74 TDD-Tests (v3.5 RIGOR)")
    print("=" * 70)
    tests = [
        ("test_reproducibility_phase_writes_real_outputs", test_reproducibility_phase_writes_real_outputs),
        ("test_reproducibility_files_written", test_reproducibility_files_written),
        ("test_reproducibility_handles_no_primary_items", test_reproducibility_handles_no_primary_items),
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
