"""
test_hypothesis_in_main.py — TDD-Test: main() ruft Hypothesentests
==================================================================
Prüft: nach postprocess() werden H1-H4 aus rigor_stats.py aufgerufen
und Ergebnisse in out/hypothesis_results.json geschrieben.
"""
import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v1")


def test_hypothesis_suite_called_in_main():
    """main() ruft test_h1..h4 und schreibt hypothesis_results.json."""
    from rigor_harness_v3 import main

    with patch("rigor_harness_v3.preprocess") as mock_pp, \
         patch("rigor_harness_v3.run_gpu_loop") as mock_gpu, \
         patch("rigor_harness_v3.postprocess") as mock_post, \
         patch("rigor_stats.test_h1_loop_reduction") as mock_h1, \
         patch("rigor_stats.test_h2_task_specific") as mock_h2, \
         patch("rigor_stats.test_h3_search_effect") as mock_h3, \
         patch("rigor_stats.test_h4_reproducibility") as mock_h4, \
         patch("rigor_harness_v3.parse_args") as mock_args:
        # Setup mocks
        mock_ctx = MagicMock()
        mock_ctx.work_items = []
        mock_ctx.batch_size = 1
        mock_ctx.max_new_tokens = 50
        mock_ctx.model_cache = {}
        mock_ctx.out_dir = MagicMock()
        mock_ctx.search_cache.get = MagicMock(return_value=[])
        mock_pp.return_value = (mock_ctx, MagicMock())
        mock_gpu.return_value = []
        mock_post.return_value = {"n_outputs": 0}
        mock_h1.return_value = {"status": "PASS", "delta": 0.05}
        mock_h2.return_value = {"status": "PASS", "eta2_math": 0.5}
        mock_h3.return_value = {"status": "PASS", "search_effect": 0.1}
        mock_h4.return_value = {"status": "PASS", "pass_rate": 1.0}
        mock_args_obj = MagicMock()
        mock_args_obj.smoke = False
        mock_args_obj.no_write = True
        mock_args_obj.n_reproducibility = 0
        mock_args_obj.enable_tools = ""
        mock_args_obj.max_tool_iterations = 5
        mock_args.return_value = mock_args_obj

        # ACT
        try:
            main()
        except SystemExit as e:
            pass

        # ASSERT: alle 4 Hypothesis-Tests wurden aufgerufen
        assert mock_h1.called, "test_h1_loop_reduction nicht aufgerufen"
        assert mock_h2.called, "test_h2_task_specific nicht aufgerufen"
        assert mock_h3.called, "test_h3_search_effect nicht aufgerufen"
        assert mock_h4.called, "test_h4_reproducibility nicht aufgerufen"
        print(f"  ✓ H1-H4 alle aufgerufen")


def test_hypothesis_results_json_written():
    """main() schreibt out/hypothesis_results.json mit 4 Hypothesentests."""
    from rigor_harness_v3 import main

    out_dir = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out"
    hyp_path = f"{out_dir}/hypothesis_results.json"
    # Cleanup
    if os.path.exists(hyp_path):
        os.unlink(hyp_path)

    with patch("rigor_harness_v3.preprocess") as mock_pp, \
         patch("rigor_harness_v3.run_gpu_loop") as mock_gpu, \
         patch("rigor_harness_v3.postprocess") as mock_post, \
         patch("rigor_stats.test_h1_loop_reduction") as mock_h1, \
         patch("rigor_stats.test_h2_task_specific") as mock_h2, \
         patch("rigor_stats.test_h3_search_effect") as mock_h3, \
         patch("rigor_stats.test_h4_reproducibility") as mock_h4, \
         patch("rigor_harness_v3.parse_args") as mock_args:
        mock_ctx = MagicMock()
        mock_ctx.work_items = []
        mock_ctx.batch_size = 1
        mock_ctx.max_new_tokens = 50
        mock_ctx.model_cache = {}
        # Use real Path for out_dir
        from pathlib import Path as _P
        real_out = _P("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out")
        mock_ctx.out_dir = real_out
        mock_ctx.search_cache.get = MagicMock(return_value=[])
        mock_pp.return_value = (mock_ctx, MagicMock())
        mock_gpu.return_value = []
        mock_post.return_value = {"n_outputs": 0}
        mock_h1.return_value = {"status": "BESTÄTIGT", "delta": 0.05}
        mock_h2.return_value = {"status": "BESTÄTIGT", "eta2_math": 0.68}
        mock_h3.return_value = {"status": "WIDERLEGT", "search_effect": 0.02}
        mock_h4.return_value = {"status": "BESTÄTIGT", "pass_rate": 1.0}
        mock_args_obj = MagicMock()
        mock_args_obj.smoke = False
        mock_args_obj.no_write = True
        mock_args_obj.n_reproducibility = 0
        mock_args_obj.enable_tools = ""
        mock_args_obj.max_tool_iterations = 5
        mock_args.return_value = mock_args_obj

        try:
            main()
        except SystemExit:
            pass

        assert os.path.exists(hyp_path), f"hypothesis_results.json fehlt: {hyp_path}"
        with open(hyp_path) as f:
            data = json.load(f)
        assert "H1" in data, f"H1 fehlt: {list(data.keys())}"
        assert "H2" in data
        assert "H3" in data
        assert "H4" in data
        assert data["H1"]["status"] == "BESTÄTIGT"
        assert data["H2"]["eta2_math"] == 0.68
        print(f"  ✓ hypothesis_results.json geschrieben: H1={data['H1']['status']}, H2={data['H2']['eta2_math']}, H3={data['H3']['status']}, H4={data['H4']['status']}")


def main():
    print("=" * 70)
    print("Hypothesentests in main() TDD-Test (v3.5 RIGOR)")
    print("=" * 70)
    tests = [
        ("test_hypothesis_suite_called_in_main", test_hypothesis_suite_called_in_main),
        ("test_hypothesis_results_json_written", test_hypothesis_results_json_written),
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
