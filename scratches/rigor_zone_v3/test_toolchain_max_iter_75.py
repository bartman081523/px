"""test_toolchain_max_iter_75.py — TDD-Test für max_iterations=75 in toolchain.run_tool_loop.

SciMind 5.0: Bei 75-Iter-Cap bricht der Loop korrekt ab (max_hit=True),
auch wenn das Modell 80× in Folge tool_calls produziert ohne final_answer.
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))

from toolchain import run_tool_loop, ToolLoopResult, _TOOL_REGISTRY  # noqa: E402


def _make_mock_model_iter_tool_call(n_iterations: int = 80):
    """Mock-Modell, das n-mal einen tool_call für web_search produziert."""
    mock_model = MagicMock()
    mock_model.device = "cuda"

    # Token-IDs simulieren: jeder generate() returnt (1, 5) token
    mock_ids = MagicMock()
    mock_ids.shape = [1, 5]
    call_count = [0]

    def fake_generate(**kwargs):
        call_count[0] += 1
        out = MagicMock()
        out.shape = [1, 5]
        out.__getitem__ = MagicMock(return_value=mock_ids)
        return out

    mock_model.generate = fake_generate
    return mock_model, call_count


def test_run_tool_loop_max_iter_75_hits_cap():
    """75-Iter-Cap: Modell produziert 80× tool_call, Loop bricht bei 75 ab."""
    mock_model, call_count = _make_mock_model_iter_tool_call()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value="<t>")
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.decode = MagicMock(
        return_value='<tool_call>{"name": "web_search", "arguments": {"query": "x"}}\n</tool_call>'
    )

    result = run_tool_loop(
        model=mock_model,
        tokenizer=mock_tokenizer,
        system_prompt="You have tools.",
        user_prompt="What is 2+2?",
        max_iterations=75,
        max_new_tokens=200,
        enable_tools=["web_search"],
    )

    # Assert: max_hit=True, n_iterations=75
    assert result.max_hit, f"max_hit=False, erwartet True (Loop lief nicht 75 Iter)"
    assert result.n_iterations == 75, f"n_iterations={result.n_iterations}, erwartet 75"
    assert result.final_answer is None, f"final_answer={result.final_answer}, sollte None sein"
    # Tool wurde 75× aufgerufen (nicht 80, weil Cap bei 75)
    assert call_count[0] == 75, f"generate wurde {call_count[0]}× aufgerufen, erwartet 75"
    print(f"  ✓ test_run_tool_loop_max_iter_75_hits_cap: cap stoppt bei 75, max_hit=True")


def test_run_tool_loop_max_iter_5_default():
    """Default max_iterations=5: Loop stoppt bei 5."""
    mock_model, call_count = _make_mock_model_iter_tool_call()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value="<t>")
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.decode = MagicMock(
        return_value='<tool_call>{"name": "web_search", "arguments": {"query": "x"}}\n</tool_call>'
    )

    result = run_tool_loop(
        model=mock_model, tokenizer=mock_tokenizer,
        system_prompt="You have tools.", user_prompt="x",
        max_iterations=5, max_new_tokens=200,
    )
    assert result.n_iterations == 5
    assert result.max_hit
    print(f"  ✓ test_run_tool_loop_max_iter_5_default: default 5 cap funktioniert")


def test_run_tool_loop_finishes_before_cap():
    """Wenn Modell < max_iterations final_answer produziert → Loop endet vor Cap."""
    mock_model = MagicMock()
    mock_model.device = "cuda"
    mock_model.generate = MagicMock(return_value=MagicMock(shape=[1, 5]))
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value="<t>")
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])

    # 1. Aufruf: tool_call; 2. Aufruf: final_answer
    outputs = [
        '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>',
        "<final_answer>42</final_answer>",
    ]
    mock_tokenizer.decode = MagicMock(side_effect=outputs)

    result = run_tool_loop(
        model=mock_model, tokenizer=mock_tokenizer,
        system_prompt="You have tools.", user_prompt="x",
        max_iterations=75, max_new_tokens=200,
    )
    assert result.n_iterations == 2, f"n_iterations={result.n_iterations}, erwartet 2"
    assert not result.max_hit
    assert result.final_answer == "42"
    print(f"  ✓ test_run_tool_loop_finishes_before_cap: bricht ab bei final_answer, n_iterations=2")


def main() -> int:
    print("="*70)
    print("TDD test_toolchain_max_iter_75 (v3.5g)")
    print("="*70)
    tests = [
        ("test_run_tool_loop_max_iter_75_hits_cap", test_run_tool_loop_max_iter_75_hits_cap),
        ("test_run_tool_loop_max_iter_5_default", test_run_tool_loop_max_iter_5_default),
        ("test_run_tool_loop_finishes_before_cap", test_run_tool_loop_finishes_before_cap),
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
