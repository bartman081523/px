"""test_harness_uses_toolchain.py — TDD-Test für Toolchain-Integration in v3.5g.

SciMind 5.0 + DevMind 1.0: RED-first.
Test prüft: rigot_harness_v3.py:run_gpu_loop() ruft toolchain.run_tool_loop()
mit den korrekten Parametern auf, wenn:
  - args.enable_tools gesetzt
  - arm_name != "baseline" (Baseline nutzt weiterhin model.generate)
  - max_iterations=args.max_tool_iterations

Test ist isoliert: monkeypatched toolchain.run_tool_loop, mocked Model.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# venv-Pfad
_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)

# Repo-Pfade
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))

# Importiere nach Path-Setup
from rigor_harness_v3 import run_gpu_loop  # noqa: E402


def test_run_gpu_loop_uses_toolchain_when_enabled():
    """Wenn enable_tools gesetzt + non-baseline arm → toolchain.run_tool_loop aufgerufen."""
    # Mock-Modell + Tokenizer
    mock_model = MagicMock()
    mock_model.device = "cuda"
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.apply_chat_template = MagicMock(return_value="<template>")
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.batch_decode = MagicMock(return_value=["OK"])
    mock_tokenizer.padding_side = "left"

    # Mock-ToolLoopResult
    mock_result = MagicMock()
    mock_result.final_answer = "X"
    mock_result.n_iterations = 2
    mock_result.transcript = []
    mock_result.tool_calls_made = ["web_search"]
    mock_result.stuck = False
    mock_result.max_hit = False

    # Patche run_tool_loop im toolchain-Modul (lokal im Harness importiert)
    with patch("toolchain.run_tool_loop", return_value=mock_result) as mock_rtl:
        with patch("rigor_harness_v3._get_model", return_value=(mock_model, mock_tokenizer, None)):
            prompts = ["What is 2+2?"]
            results = run_gpu_loop(
                arm_name="active_manifold",
                arm_preset="ACTIVE_MANIFOLD",
                patch_kwargs=None,
                prompts=prompts,
                max_new_tokens=200,
                enable_tools=["web_search"],
                max_tool_iterations=5,
            )

    # Assert: run_tool_loop wurde aufgerufen
    assert mock_rtl.called, "run_tool_loop wurde NICHT aufgerufen, obwohl enable_tools gesetzt"
    call_kwargs = mock_rtl.call_args.kwargs
    assert call_kwargs["max_iterations"] == 5, f"max_iterations falsch: {call_kwargs.get('max_iterations')}"
    assert call_kwargs["max_new_tokens"] == 200
    assert call_kwargs["user_prompt"] == "What is 2+2?"
    assert "system_prompt" in call_kwargs
    # System-Prompt sollte TOOLCHAIN_DEFINITION enthalten
    assert "web_search" in call_kwargs["system_prompt"] or "<tool_call>" in call_kwargs["system_prompt"], \
        f"System-Prompt ohne Toolchain-Definition: {call_kwargs['system_prompt'][:200]}"

    # Result ist das ToolLoopResult.final_answer
    assert len(results) == 1
    assert results[0]["text"] == "X", f"text falsch: {results[0]}"
    print(f"  ✓ test_run_gpu_loop_uses_toolchain_when_enabled: run_tool_loop aufgerufen mit max_iterations=5")
    print(f"    result: text='X', n_iterations=2")


def test_run_gpu_loop_uses_toolchain_with_max_75():
    """Spezialfall: --max-tool-iterations 75 wird durchgereicht."""
    mock_model = MagicMock()
    mock_model.device = "cuda"
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.apply_chat_template = MagicMock(return_value="<template>")
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.batch_decode = MagicMock(return_value=["OK"])
    mock_tokenizer.padding_side = "left"

    mock_result = MagicMock()
    mock_result.final_answer = "X"
    mock_result.n_iterations = 75
    mock_result.transcript = []
    mock_result.tool_calls_made = ["web_search"] * 74 + ["final_answer"]
    mock_result.stuck = False
    mock_result.max_hit = True

    with patch("toolchain.run_tool_loop", return_value=mock_result) as mock_rtl:
        with patch("rigor_harness_v3._get_model", return_value=(mock_model, mock_tokenizer, None)):
            results = run_gpu_loop(
                arm_name="rigor_mid",
                arm_preset="RIGOR",
                patch_kwargs={"rigor_mephisto": True, "rigor_mephisto_scale": 0.5},
                prompts=["task"],
                max_new_tokens=300,
                enable_tools=["web_search", "execute_python"],
                max_tool_iterations=75,
            )

    assert mock_rtl.called
    call_kwargs = mock_rtl.call_args.kwargs
    assert call_kwargs["max_iterations"] == 75, f"max_iterations falsch: {call_kwargs.get('max_iterations')}"
    print(f"  ✓ test_run_gpu_loop_uses_toolchain_with_max_75: max_iterations=75 durchgereicht")


def main() -> int:
    print("="*70)
    print("TDD test_harness_uses_toolchain (v3.5g)")
    print("="*70)
    tests = [
        ("test_run_gpu_loop_uses_toolchain_when_enabled", test_run_gpu_loop_uses_toolchain_when_enabled),
        ("test_run_gpu_loop_uses_toolchain_with_max_75", test_run_gpu_loop_uses_toolchain_with_max_75),
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
