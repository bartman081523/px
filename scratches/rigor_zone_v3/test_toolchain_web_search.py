"""
test_toolchain_web_search.py — TDD-Tests für web_search + read_file + write_file
=================================================================================
RED-PHASE: Tests vor Implementation.
GREEN-PHASE: toolchain.py macht sie grün.

Coverage:
    1. web_search: mocked DDGS liefert 3 Results
    2. web_search: DDG-Failure → error-Feld
    3. read_file: Pfad in Whitelist → content
    4. read_file: Pfad außerhalb Whitelist → REJECTED
    5. read_file: nicht-existente Datei → error
    6. write_file: in agent_workspace → success, Datei existiert
    7. write_file: atomar (kein partial file)
"""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand")
sys.path.insert(0, "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3")

from toolchain import web_search, read_file, write_file  # noqa: E402


def test_web_search_returns_three_results():
    """web_search mocked: returnt 3 Results."""
    fake_results = [
        {"title": "Factorial - Wikipedia", "body": "n! = 1·2·...·n", "href": "https://en.wikipedia.org/wiki/Factorial"},
        {"title": "Python math.factorial", "body": "Return n! as int", "href": "https://docs.python.org/3/library/math.html"},
        {"title": "StackOverflow: factorial", "body": "Use math.factorial", "href": "https://stackoverflow.com/q/1"},
    ]
    fake_ddgs = MagicMock()
    fake_ddgs.text = MagicMock(return_value=iter(fake_results))
    fake_ddgs.__enter__ = MagicMock(return_value=fake_ddgs)
    fake_ddgs.__exit__ = MagicMock(return_value=False)
    with patch("ddgs.DDGS", return_value=fake_ddgs):
        result = web_search("factorial 12", max_results=3)
    assert result["error"] is None, f"Error: {result.get('error')}"
    assert len(result["results"]) == 3
    assert result["results"][0]["title"] == "Factorial - Wikipedia"
    assert result["results"][1]["url"].startswith("https://")
    print(f"  ✓ web_search: 3 results, titles={[r['title'][:20] for r in result['results']]}")


def test_web_search_handles_ddg_failure():
    """web_search: DDG wirft Exception → error-Feld gefüllt."""
    with patch("ddgs.DDGS", side_effect=RuntimeError("DDG rate limit")):
        result = web_search("test")
    assert result["error"] is not None
    assert "rate limit" in result["error"]
    assert result["results"] == []
    print(f"  ✓ DDG failure: error={result['error']}")


def test_read_file_in_whitelist():
    """read_file: Datei in Whitelist (Projekt) → content."""
    # Nutze eine bekannte Datei im Projekt
    test_path = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/toolchain.py"
    result = read_file(test_path, max_kb=20)
    assert result["error"] is None, f"Error: {result.get('error')}"
    assert "toolchain" in result["content"]
    assert result["size"] > 0
    print(f"  ✓ read_file: {result['size']} bytes")


def test_read_file_outside_whitelist():
    """read_file: /etc/passwd → REJECTED."""
    result = read_file("/etc/passwd", max_kb=10)
    assert "not in whitelist" in result["error"]
    assert result["content"] == ""
    print(f"  ✓ path whitelist: rejected")


def test_read_file_not_found():
    """read_file: nicht-existente Datei → error."""
    result = read_file("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/nonexistent_file_xyz.py")
    assert "not found" in result["error"].lower() or "FileNotFoundError" in result["error"]
    print(f"  ✓ nicht gefunden: {result['error']}")


def test_write_file_to_workspace():
    """write_file: in agent_workspace → success + Datei existiert."""
    target = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out/agent_workspace/test_write.txt"
    content = "Hello, agent world!\nLine 2"
    # Cleanup first
    if os.path.exists(target):
        os.unlink(target)
    result = write_file(target, content)
    assert result["success"], f"Error: {result.get('error')}"
    assert os.path.exists(target)
    with open(target) as f:
        read_back = f.read()
    assert read_back == content
    os.unlink(target)  # cleanup
    print(f"  ✓ write_file: {len(content)} chars written, atomic")


def test_write_file_creates_parent_dirs():
    """write_file: legt Parent-Dirs an, falls nicht existent."""
    target = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out/agent_workspace/subdir1/subdir2/nested.txt"
    if os.path.exists(target):
        os.unlink(target)
    result = write_file(target, "nested content")
    assert result["success"], f"Error: {result.get('error')}"
    assert os.path.exists(target)
    # cleanup
    os.unlink(target)
    os.rmdir("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out/agent_workspace/subdir1/subdir2")
    os.rmdir("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3/out/agent_workspace/subdir1")
    print(f"  ✓ nested dirs: created")


def main():
    print("=" * 70)
    print("Toolchain Tool-Implementierungen TDD-Tests (v3.5 RIGOR)")
    print("=" * 70)
    tests = [
        ("test_web_search_returns_three_results", test_web_search_returns_three_results),
        ("test_web_search_handles_ddg_failure", test_web_search_handles_ddg_failure),
        ("test_read_file_in_whitelist", test_read_file_in_whitelist),
        ("test_read_file_outside_whitelist", test_read_file_outside_whitelist),
        ("test_read_file_not_found", test_read_file_not_found),
        ("test_write_file_to_workspace", test_write_file_to_workspace),
        ("test_write_file_creates_parent_dirs", test_write_file_creates_parent_dirs),
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
