"""
test_toolchain_parse.py — TDD-Tests für Hermes-Format-Parser (v3.5 RIGOR)
========================================================================
RED-PHASE: Tests werden vor toolchain.py geschrieben.
GREEN-PHASE: toolchain.py macht sie grün.

Coverage:
    1. parse_tool_call: Standard-Tool-Call mit web_search
    2. parse_tool_call: komplexes JSON-Argument
    3. parse_tool_call: kein Tool → None
    4. parse_final_answer: Standard-Antwort
    5. parse_final_answer: keine Antwort → None
    6. parse_tool_call: malformed JSON → None
"""
import os
import sys

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")

from toolchain import parse_tool_call, parse_final_answer, ToolCall  # noqa: E402


def test_parse_tool_call_web_search():
    """Standard-tool_call mit web_search wird korrekt geparst."""
    text = (
        'Ich werde die Web-Suche nutzen.\n'
        '<tool_call>\n'
        '{"name": "web_search", "arguments": {"query": "Python fakultät"}}\n'
        '</tool_call>'
    )
    tc = parse_tool_call(text)
    assert tc is not None, "Tool-Call wurde nicht erkannt"
    assert tc.name == "web_search", f"Name: {tc.name}"
    assert tc.arguments == {"query": "Python fakultät"}, f"Args: {tc.arguments}"
    print(f"  ✓ web_search: name={tc.name}, args={tc.arguments}")


def test_parse_tool_call_complex_args():
    """execute_python mit code-Argument (mehrzeilig)."""
    text = (
        '<tool_call>\n'
        '{"name": "execute_python", "arguments": {"code": "import math\\nprint(math.factorial(12))"}}\n'
        '</tool_call>'
    )
    tc = parse_tool_call(text)
    assert tc is not None
    assert tc.name == "execute_python"
    assert "math.factorial" in tc.arguments["code"]
    print(f"  ✓ execute_python: code={tc.arguments['code'][:40]}...")


def test_parse_tool_call_no_tool():
    """Text ohne tool_call → None."""
    text = "Das ist eine normale Antwort ohne Tool."
    tc = parse_tool_call(text)
    assert tc is None, f"Erwartet None, bekam {tc}"
    print(f"  ✓ kein tool_call → None")


def test_parse_final_answer_standard():
    """Standard <final_answer>...</final_answer>."""
    text = "Nach langer Überlegung:\n<final_answer>479001600</final_answer>"
    fa = parse_final_answer(text)
    assert fa == "479001600", f"Bekam: {fa!r}"
    print(f"  ✓ final_answer: {fa}")


def test_parse_final_answer_none():
    """Text ohne final_answer → None."""
    text = "Ich weiß es nicht."
    fa = parse_final_answer(text)
    assert fa is None
    print(f"  ✓ kein final_answer → None")


def test_parse_tool_call_malformed_json():
    """Malformed JSON in tool_call → None (kein Crash)."""
    text = (
        '<tool_call>\n'
        '{"name": "web_search", "arguments": {"query": "unclosed quote}\n'
        '</tool_call>'
    )
    tc = parse_tool_call(text)
    assert tc is None, f"Erwartet None bei malformed JSON, bekam {tc}"
    print(f"  ✓ malformed JSON → None (graceful)")


def test_parse_tool_call_whitespace_handling():
    """Whitespace um JSON wird toleriert."""
    text = '<tool_call>   \n\n   {"name": "read_file", "arguments": {"path": "/tmp/x"}}   \n</tool_call>'
    tc = parse_tool_call(text)
    assert tc is not None
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "/tmp/x"}
    print(f"  ✓ whitespace-tolerant: {tc.name}")


def main():
    print("=" * 70)
    print("Toolchain Parser TDD-Tests (v3.5 RIGOR / Hermes-Format)")
    print("=" * 70)
    tests = [
        ("test_parse_tool_call_web_search", test_parse_tool_call_web_search),
        ("test_parse_tool_call_complex_args", test_parse_tool_call_complex_args),
        ("test_parse_tool_call_no_tool", test_parse_tool_call_no_tool),
        ("test_parse_final_answer_standard", test_parse_final_answer_standard),
        ("test_parse_final_answer_none", test_parse_final_answer_none),
        ("test_parse_tool_call_malformed_json", test_parse_tool_call_malformed_json),
        ("test_parse_tool_call_whitespace_handling", test_parse_tool_call_whitespace_handling),
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
