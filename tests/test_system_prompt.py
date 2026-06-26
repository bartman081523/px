"""Tests for gradio_tabs/system_prompt.py — pure-logic profile loader + system
message builder + injection into messages list.

Written FIRST (TDD). Run with:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_system_prompt.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradio_tabs.system_prompt import (
    SYSTEM_SENTINEL,
    PROFILES,
    list_profiles,
    resolve_profile,
    build_system_message,
    inject_into_messages,
    render_for_chat_template,
)


def m(role, content):
    """Helper: build a chat history message dict."""
    return {"role": role, "content": content}


# --- list_profiles + resolve_profile -----------------------------------------

def test_lp_1_returns_neutral_citmind_juexin_in_order():
    names = list_profiles()
    assert "neutral" in names, names
    # If both docs parse, order is ["neutral","citmind","juexin"]
    if "citmind" in names and "juexin" in names:
        assert names == ["neutral", "citmind", "juexin"], names
    # neutral must always be present, always first
    assert names[0] == "neutral", names


def test_lp_2_resolve_neutral():
    p = resolve_profile("neutral")
    assert p["name"] == "neutral", p
    assert p["core_philosophy"] == ""
    assert p["substrate_universal"] == ""
    assert p["core_principles"] == []
    assert p["source_doc"] is None


def test_lp_3_resolve_citmind():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing or unparseable in this env)")
        return
    p = resolve_profile("citmind")
    assert p["name"] == "citmind", p
    assert len(p["core_philosophy"]) > 50, f"too short: {len(p['core_philosophy'])}"
    assert len(p["substrate_universal"]) > 50, f"too short: {len(p['substrate_universal'])}"
    assert len(p["core_principles"]) >= 3, p["core_principles"]
    assert p["source_doc"] is not None
    assert "CitMind" in p["source_doc"]


def test_lp_4_resolve_juexin():
    names = list_profiles()
    if "juexin" not in names:
        print("SKIP (Juexin.txt missing or unparseable in this env)")
        return
    p = resolve_profile("juexin")
    assert p["name"] == "juexin", p
    assert len(p["core_philosophy"]) > 50, f"too short: {len(p['core_philosophy'])}"
    assert len(p["substrate_universal"]) > 50, f"too short: {len(p['substrate_universal'])}"
    assert len(p["core_principles"]) >= 3, p["core_principles"]
    assert p["source_doc"] is not None
    assert "Juexin" in p["source_doc"]


def test_lp_5_resolve_unknown_falls_back_neutral():
    p = resolve_profile("unknown-xyz-fictional-profile")
    assert p["name"] == "neutral", p
    assert p["core_philosophy"] == ""


def test_lp_6_resolve_returns_fresh_copy():
    p1 = resolve_profile("neutral")
    p1["core_philosophy"] = "TAMPERED"
    p1["core_principles"].append("TAMPERED")
    p2 = resolve_profile("neutral")
    assert p2["core_philosophy"] == "", p2
    assert "TAMPERED" not in p2["core_principles"], p2


# --- build_system_message ---------------------------------------------------

def test_bsm_7_neutral_empty_content():
    out = build_system_message("neutral")
    assert out == {"role": "system", "content": ""}, out


def test_bsm_8_citmind_starts_with_sentinel_and_has_substantive_content():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    out = build_system_message("citmind")
    assert out["role"] == "system", out
    content = out["content"]
    assert content.startswith(SYSTEM_SENTINEL), content[:60]
    # substantive: contains at least one non-trivial substring
    p = resolve_profile("citmind")
    cp = p["core_philosophy"]
    assert cp[:20] in content, f"core_philosophy head not in content: {content[:200]}"


def test_bsm_9_juexin_starts_with_sentinel_and_has_substantive_content():
    names = list_profiles()
    if "juexin" not in names:
        print("SKIP (Juexin.txt missing)")
        return
    out = build_system_message("juexin")
    assert out["role"] == "system", out
    content = out["content"]
    assert content.startswith(SYSTEM_SENTINEL), content[:60]
    p = resolve_profile("juexin")
    cp = p["core_philosophy"]
    assert cp[:20] in content, f"core_philosophy head not in content: {content[:200]}"


def test_bsm_10_citmind_edit_replaces_philosophy():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    out = build_system_message("citmind", edit_text="custom override")
    assert out["content"] == SYSTEM_SENTINEL + "custom override", repr(out["content"])


def test_bsm_11_citmind_edit_gets_stripped():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    out = build_system_message("citmind", edit_text="  spaced  ")
    assert out["content"] == SYSTEM_SENTINEL + "spaced", repr(out["content"])


def test_bsm_12_empty_edit_falls_back_to_profile():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    out = build_system_message("citmind", edit_text="")
    # empty edit must behave like no-edit: render profile content (NOT empty)
    assert out["content"] != "", "empty edit must render profile, not be empty"
    assert out["content"].startswith(SYSTEM_SENTINEL), out["content"][:60]


def test_bsm_13_returned_dict_is_system_role_with_str_content():
    names = list_profiles()
    if "citmind" not in names:
        # Test with neutral which is always available
        out = build_system_message("neutral")
    else:
        out = build_system_message("citmind")
    assert isinstance(out, dict), out
    assert out["role"] == "system", out
    assert isinstance(out["content"], str), out


def test_bsm_14_unknown_profile_with_edit_falls_back_neutral():
    # Unknown profile → resolve_profile returns the neutral profile.
    # Edit_text (non-empty) then renders into content per the standard rule
    # (SYSTEM_SENTINEL + stripped edit). The "falls back to neutral" path
    # means we resolve without raising; the edit is honoured because it is
    # what the user typed.
    out = build_system_message("unknown-xyz-fictional", edit_text="this should not appear")
    assert out["content"] == SYSTEM_SENTINEL + "this should not appear", repr(out["content"])
    assert out["role"] == "system"
    # And: unknown profile WITHOUT edit → empty content (orchestrator-omittable)
    out2 = build_system_message("unknown-xyz-fictional")
    assert out2["content"] == "", repr(out2["content"])


# --- inject_into_messages ---------------------------------------------------

def test_inj_15_neutral_returns_input_unchanged():
    hist = [m("user", "hi")]
    out = inject_into_messages(hist, "neutral")
    assert out == hist, out
    assert out is not hist, "should be a new list"
    # input not mutated
    assert len(hist) == 1


def test_inj_16_citmind_prepends_system_entry():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    out = inject_into_messages([m("user", "hi")], "citmind")
    assert len(out) == 2, out
    assert out[0]["role"] == "system", out[0]
    assert out[0]["content"].startswith(SYSTEM_SENTINEL), out[0]["content"][:60]
    assert out[1] == m("user", "hi"), out[1]
    # input not mutated
    assert len([m("user", "hi")]) == 1


def test_inj_17_existing_system_at_zero_is_replaced_not_duplicated():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    out = inject_into_messages(
        [m("system", "OLD"), m("user", "hi")],
        "citmind",
    )
    assert len(out) == 2, f"expected 2, got {len(out)}: {out}"
    assert out[0]["role"] == "system", out[0]
    assert out[0]["content"] != "OLD", "should be replaced, not kept"
    assert out[0]["content"].startswith(SYSTEM_SENTINEL), out[0]["content"][:60]
    assert out[1] == m("user", "hi"), out[1]


def test_inj_18_empty_messages_get_system_prepended():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    out = inject_into_messages([], "citmind")
    assert len(out) == 1, out
    assert out[0]["role"] == "system", out[0]


def test_inj_19_does_not_mutate_input():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    hist = [m("user", "hi"), m("assistant", "hello")]
    out = inject_into_messages(hist, "citmind")
    assert len(hist) == 2, f"input mutated: {hist}"
    assert hist[0] == m("user", "hi"), hist
    assert hist[1] == m("assistant", "hello"), hist
    # output is a new list
    assert out is not hist


def test_inj_20_multimodal_content_preserved():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    mm = [{"type": "image", "image": "/p/a.png"}]
    hist = [m("user", mm)]
    out = inject_into_messages(hist, "citmind")
    assert len(out) == 2, out
    # system prepended, original user with list-content preserved
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert out[1]["content"] is mm or out[1]["content"] == mm, out[1]
    # input not mutated
    assert len(hist) == 1
    assert hist[0]["content"] == mm


def test_inj_21_removes_all_existing_system_entries():
    """Plan 5.3: persisted sessions may contain MULTIPLE system entries
    (e.g. one stored as a Multimodal-List from old chat_tab history, and
    another added by a later session). inject_into_messages strips ALL
    of them and inserts a single fresh system entry at index 0. Without
    this fix, downstream ``m['content'][:60].replace(...)`` crashes on
    the Multimodal-List system entry (the bug from chat_tab.py:228)."""
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    mm_system = [{"type": "text", "text": "[old legacy multimodal system]"}]
    hist = [
        {"role": "system", "content": mm_system},
        {"role": "system", "content": "[old plain system]"},
        {"role": "user", "content": "hi"},
    ]
    out = inject_into_messages(hist, "citmind")
    # All old system entries gone, exactly one new system at index 0.
    system_entries = [x for x in out if x.get("role") == "system"]
    assert len(system_entries) == 1, out
    assert system_entries[0] is out[0]
    assert "[SYSTEM CONTEXT]" in system_entries[0]["content"]
    assert "[old legacy multimodal system]" not in str(system_entries[0])
    # Input not mutated.
    assert hist[0]["content"] is mm_system


# --- render_for_chat_template -----------------------------------------------

def test_rfc_21_neutral_empty_string():
    assert render_for_chat_template("neutral") == ""


def test_rfc_22_citmind_starts_with_sentinel_ends_with_newline():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    out = render_for_chat_template("citmind")
    assert out.startswith(SYSTEM_SENTINEL), repr(out[:60])
    assert out.endswith("\n"), repr(out[-20:])


def test_rfc_23_citmind_edit_uses_override_only():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    out = render_for_chat_template("citmind", edit_text="OVERRIDE")
    assert out.startswith(SYSTEM_SENTINEL), repr(out[:60])
    assert out.endswith("OVERRIDE\n"), repr(out)


def test_rfc_24_consistency_between_build_and_render():
    names = list_profiles()
    if "citmind" not in names:
        print("SKIP (CitMind.txt missing)")
        return
    built = build_system_message("citmind")["content"]
    rendered = render_for_chat_template("citmind")
    # both wrap the same source text; the built message is the rendered minus trailing newline
    assert rendered == built + "\n", f"built={built!r}\nrendered={rendered!r}"


# --- runner ------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    skipped = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = _run_all()
    sys.exit(0 if ok else 1)