"""
test_toolchain_run_loop.py — TDD-Tests für run_tool_loop
=========================================================
Coverage:
    1. Mock-Model: 1× web_search, 1× final_answer → n_iterations=2
    2. Mock-Model: kein Tool, kein final_answer → stuck=True
    3. Mock-Model: 5 Iterationen ohne final_answer → max_hit=True
    4. Mock-Model: Tool dispatch erhält korrekte Args
    5. Mock-Model: Transcript enthält alle Iterationen
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")

from toolchain import run_tool_loop, ToolLoopResult  # noqa: E402


class MockTokenizer:
    """Mock-Tokenizer mit deterministischem Verhalten."""
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.call_count = 0

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):
        return f"PROMPT[{len(conversation)}]"

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = MagicMock()
        ids.to = MagicMock(return_value=ids)
        ids.__getitem__ = MagicMock(return_value=MagicMock(shape=[1, 1]))
        ids.input_ids = MagicMock(shape=[1, 1])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        if self.call_count >= len(self.responses):
            return ""  # keine Antwort mehr
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


class MockModel:
    """Mock-Model mit deterministischer generate()."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.device = "cuda"
        self.generate_calls = 0

    def generate(self, **kwargs):
        self.generate_calls += 1
        # Return mock output_ids: [BOS, response_token_42]
        ids = MagicMock()
        ids.__getitem__ = MagicMock(return_value=MagicMock(shape=[1, 2]))
        return ids


def test_run_tool_loop_two_iterations():
    """Iter 1: web_search, Iter 2: final_answer → n_iterations=2."""
    tok = MockTokenizer([
        '<tool_call>\n{"name": "web_search", "arguments": {"query": "12+34"}}\n</tool_call>',
        '<final_answer>46</final_answer>',
    ])
    model = MockModel(tok)
    result = run_tool_loop(
        model, tok,
        system_prompt="System",
        user_prompt="What is 12+34?",
        max_iterations=5,
    )
    assert isinstance(result, ToolLoopResult)
    assert result.final_answer == "46", f"final_answer: {result.final_answer!r}"
    assert result.n_iterations == 2, f"n_iterations: {result.n_iterations}"
    assert result.tool_calls_made == ["web_search"], f"tools: {result.tool_calls_made}"
    assert result.stuck is False
    assert result.max_hit is False
    # Transcript: iter1=assistant (tool_call), iter1=user (tool_result), iter2=assistant (final_answer) = 3
    assert len(result.transcript) == 3, f"transcript len: {len(result.transcript)}"
    print(f"  ✓ 2-iter loop: final={result.final_answer}, tools={result.tool_calls_made}")


def test_run_tool_loop_stuck():
    """Model gibt weder Tool noch final_answer → stuck=True."""
    tok = MockTokenizer(["Just some text without structure."])
    model = MockModel(tok)
    result = run_tool_loop(model, tok, "System", "Hi", max_iterations=5)
    assert result.stuck is True
    assert result.final_answer == "Just some text without structure."
    assert result.n_iterations == 1
    print(f"  ✓ stuck: final={result.final_answer!r}, stuck={result.stuck}")


def test_run_tool_loop_max_iter():
    """5 Iterationen ohne final_answer → max_hit=True."""
    # Endlose Tool-Calls (z.B. immer web_search mit gleicher Query)
    responses = [
        '<tool_call>\n{"name": "web_search", "arguments": {"query": "loop"}}\n</tool_call>'
    ] * 10
    tok = MockTokenizer(responses)
    model = MockModel(tok)
    result = run_tool_loop(model, tok, "System", "Hi", max_iterations=5)
    assert result.max_hit is True
    assert result.final_answer is None
    assert result.n_iterations == 5
    assert len(result.tool_calls_made) == 5
    print(f"  ✓ max_hit after 5 iters, tools_made={len(result.tool_calls_made)}")


def test_run_tool_loop_multiple_tools():
    """Iter 1: execute_python, Iter 2: write_file, Iter 3: final_answer."""
    tok = MockTokenizer([
        '<tool_call>\n{"name": "execute_python", "arguments": {"code": "print(2+2)"}}\n</tool_call>',
        '<tool_call>\n{"name": "write_file", "arguments": {"path": "/tmp/result.txt", "content": "4"}}\n</tool_call>',
        '<final_answer>The result is 4</final_answer>',
    ])
    model = MockModel(tok)
    result = run_tool_loop(model, tok, "System", "Compute 2+2", max_iterations=5)
    assert result.n_iterations == 3
    assert result.tool_calls_made == ["execute_python", "write_file"]
    assert result.final_answer == "The result is 4"
    assert result.stuck is False
    print(f"  ✓ multi-tool: {result.tool_calls_made}, final={result.final_answer!r}")


def test_run_tool_loop_tool_not_in_registry():
    """Tool-Call mit unbekanntem Tool → wie stuck behandelt."""
    tok = MockTokenizer([
        '<tool_call>\n{"name": "unknown_tool", "arguments": {}}\n</tool_call>',
    ])
    model = MockModel(tok)
    result = run_tool_loop(model, tok, "System", "Hi", max_iterations=5)
    assert result.stuck is True
    assert result.tool_calls_made == []  # wurde nicht ausgeführt
    print(f"  ✓ unknown tool: stuck, no execution")


def main():
    print("=" * 70)
    print("run_tool_loop TDD-Tests (v3.5 RIGOR)")
    print("=" * 70)
    tests = [
        ("test_run_tool_loop_two_iterations", test_run_tool_loop_two_iterations),
        ("test_run_tool_loop_stuck", test_run_tool_loop_stuck),
        ("test_run_tool_loop_max_iter", test_run_tool_loop_max_iter),
        ("test_run_tool_loop_multiple_tools", test_run_tool_loop_multiple_tools),
        ("test_run_tool_loop_tool_not_in_registry", test_run_tool_loop_tool_not_in_registry),
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
