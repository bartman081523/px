"""test_harness_no_tools_falls_back.py — TDD-Test für Fallback-Pfad.

Wenn --enable-tools '' (leer) oder arm_name == "baseline":
→ run_tool_loop wird NICHT aufgerufen
→ Stattdessen model.generate() (CUDA-Graph-Pfad oder baseline.generate)

SciMind 5.0: Regression-Schutz für existierende CUDA-Graph-Pipeline.
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))

from rigor_harness_v3 import run_gpu_loop  # noqa: E402


def test_run_gpu_loop_baseline_uses_model_generate():
    """arm_name=baseline → model.generate(), KEIN run_tool_loop."""
    mock_model = MagicMock()
    mock_model.device = "cuda"
    mock_model.generate = MagicMock(return_value=MagicMock(shape=[1, 30]))
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.apply_chat_template = MagicMock(return_value="<t>")
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.batch_decode = MagicMock(return_value=["baseline output"])
    mock_tokenizer.padding_side = "left"

    with patch("toolchain.run_tool_loop") as mock_rtl:
        with patch("rigor_harness_v3._get_model", return_value=(mock_model, mock_tokenizer, None)):
            results = run_gpu_loop(
                arm_name="baseline",
                arm_preset="BASELINE",
                patch_kwargs=None,
                prompts=["task"],
                max_new_tokens=10,
                enable_tools=[],   # ← LEER
                max_tool_iterations=75,
            )

    assert not mock_rtl.called, \
        f"run_tool_loop aufgerufen obwohl enable_tools=[]: {mock_rtl.call_args}"
    # Baseline-Pfad: model.generate wurde aufgerufen
    assert mock_model.generate.called, "model.generate nicht aufgerufen im Baseline-Pfad"
    print(f"  ✓ test_run_gpu_loop_baseline_uses_model_generate: baseline nutzt model.generate")


def test_run_gpu_loop_empty_tools_falls_back():
    """arm=active_manifold + enable_tools=[] → KEIN run_tool_loop, sondern CUDA-Graph."""
    mock_model = MagicMock()
    mock_model.device = "cuda"
    mock_model.generate = MagicMock(return_value=MagicMock(shape=[1, 30]))
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.apply_chat_template = MagicMock(return_value="<t>")
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.batch_decode = MagicMock(return_value=["px output"])
    mock_tokenizer.padding_side = "left"

    # Mock CUDA-Graph-Runner (PX-Pfad)
    mock_runner = MagicMock()
    mock_runner.setup = MagicMock(return_value=MagicMock(shape=[1, 1]))
    mock_runner.step = MagicMock(return_value=MagicMock(shape=[1, 1]))
    mock_runner.append = MagicMock()

    with patch("toolchain.run_tool_loop") as mock_rtl:
        with patch("rigor_harness_v3._get_model", return_value=(mock_model, mock_tokenizer, None)):
            with patch("rigor_harness_v3.CUDAGraphRunner", return_value=mock_runner, create=True) as mock_cgr:
                results = run_gpu_loop(
                    arm_name="active_manifold",
                    arm_preset="ACTIVE_MANIFOLD",
                    patch_kwargs=None,
                    prompts=["task"],
                    max_new_tokens=10,
                    enable_tools=[],   # ← LEER
                    max_tool_iterations=75,
                )

    assert not mock_rtl.called, "run_tool_loop aufgerufen obwohl enable_tools=[]"
    print(f"  ✓ test_run_gpu_loop_empty_tools_falls_back: enable_tools=[] → direktes generate")


def main() -> int:
    print("="*70)
    print("TDD test_harness_no_tools_falls_back (v3.5g)")
    print("="*70)
    tests = [
        ("test_run_gpu_loop_baseline_uses_model_generate", test_run_gpu_loop_baseline_uses_model_generate),
        ("test_run_gpu_loop_empty_tools_falls_back", test_run_gpu_loop_empty_tools_falls_back),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR ({type(e).__name__}): {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*70}")
    print(f"Tests: {len(tests) - failed}/{len(tests)} grün")
    print(f"{'='*70}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
