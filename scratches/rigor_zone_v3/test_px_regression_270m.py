"""TDD Regression-Test: PX-Engine auf 270m darf die Baseline-Matches nicht verlieren.

Hintergrund: Auf gemma3-270m-it (HS=640) testen wir, dass
- ACTIVE_MANIFOLD die korrekten MCQ-Antworten produziert, die BASELINE produziert
- ACTIVE_MANIFOLD_LEAN ebenso
- Keine zusätzlichen Loopy-Outs auf den 5 lösbaren HLE-Tasks

WICHTIG: Dies ist ein TDD-Test gegen failende Regression. Wir verifizieren die
5 Tasks, die alle 3 Presets lösen (Humanities D, Physics 3, Biology B, Engineering B, Other 6).
"""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch
from rigor_hle_suite_v3 import load_hle_suite

# Diese 5 Tasks werden von allen 3 Presets (BASELINE, ACTIVE_MANIFOLD, LEAN) korrekt gelöst.
EXPECTED_MATCHES = {
    "668825f80a": ("Humanities/Social Science", "d"),
    "66b827b9b6": ("Physics", "3"),
    "66e88728ba": ("Biology/Medicine", "b"),
    "66ec02c52e": ("Engineering", "b"),
    "66e8d47362": ("Other", "6"),
}


def _load_suite():
    source, suite = load_hle_suite(max_per_category=4)
    return {(tid[:10]): (cat, prompt, gt) for tid, cat, prompt, gt in suite}


def _get_model(preset):
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", dtype=torch.bfloat16).to("cuda").eval()
    if preset != "BASELINE":
        apply_px_patch(model.model, config_preset=preset)
    return tok, model


def _generate(tok, model, prompt, max_new=200):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def _check_match(text, gt):
    """Match-Logik: Multi-Choice oder Substring."""
    text_l = text.lower()
    gt_l = gt.lower().strip()
    if len(gt_l) <= 3 and gt_l.replace(" ", "").isalpha():
        if f"**{gt_l.upper()}" in text or f"answer is {gt_l.upper()}" in text_l or f"answer: {gt_l.upper()}" in text_l:
            return True
        if f"\n{gt_l.upper()}. " in text or f"{gt_l.upper()}. " in text[:50]:
            return True
    else:
        if gt_l in text_l:
            return True
    return False


def _analyze(text):
    if not text.strip():
        return "EMPTY"
    words = text.split()
    for plen in [3, 4]:
        for i in range(len(words) - plen*3):
            phrase = " ".join(words[i:i+plen])
            n_repeats = sum(1 for j in range(i+plen, len(words)-plen+1, plen)
                          if " ".join(words[j:j+plen]) == phrase)
            if n_repeats >= 3:
                return f"LOOPY"
    return "OK"


@pytest.mark.parametrize("preset", ["ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN"])
def test_px_preserves_baseline_matches(preset):
    """TDD: PX-Engine muss die 5 Baseline-Matches erhalten."""
    suite = _load_suite()
    tok, model = _get_model(preset)
    try:
        for tid_short, (cat, expected_gt) in EXPECTED_MATCHES.items():
            # Find task
            matching = [v for k, v in suite.items() if k.startswith(tid_short)]
            assert matching, f"Task {tid_short} nicht in Suite"
            actual_cat, prompt, actual_gt = matching[0]
            assert actual_gt.lower().strip() == expected_gt, f"GT Mismatch: {actual_gt} vs {expected_gt}"

            text = _generate(tok, model, prompt, max_new=200)
            status = _analyze(text)
            match = _check_match(text, actual_gt)
            assert status == "OK", f"Degeneriert: {tid_short} preset={preset} status={status} | {text[:100]!r}"
            assert match, f"Match verloren: {tid_short} preset={preset} | {text[:100]!r}"
    finally:
        del model
        torch.cuda.empty_cache()


def test_px_does_not_increase_loopy_rate_on_math():
    """TDD: PX darf Math-Task 66b2c7c979 nicht in Loopy drängen (Regression-Guard)."""
    tid_short = "66b2c7c979"
    suite = _load_suite()
    matching = [v for k, v in suite.items() if k.startswith(tid_short)]
    cat, prompt, gt = matching[0]

    # BASELINE produziert OK
    tok, model = _get_model("BASELINE")
    try:
        baseline_text = _generate(tok, model, prompt, max_new=200)
    finally:
        del model
        torch.cuda.empty_cache()
    baseline_status = _analyze(baseline_text)
    assert baseline_status == "OK", f"BASELINE selbst ist LOOPY?! {baseline_text[:100]}"

    # ACTIVE_MANIFOLD muss auch OK sein
    tok, model = _get_model("ACTIVE_MANIFOLD")
    try:
        am_text = _generate(tok, model, prompt, max_new=200)
    finally:
        del model
        torch.cuda.empty_cache()
    am_status = _analyze(am_text)
    assert am_status == "OK", (
        f"REGRESSION: ACTIVE_MANIFOLD macht Math {tid_short} LOOPY. "
        f"BASELINE OK, AM LOOPY. | {am_text[:100]!r}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("TDD Regression-Test: PX auf 270m muss Baseline-Matches erhalten")
    print("=" * 70)
    # Run without pytest
    for preset in ["ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN"]:
        print(f"\n[Test] {preset}")
        try:
            test_px_preserves_baseline_matches(preset)
            print(f"  ✓ {preset} preserves all 5 baseline matches")
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
    print(f"\n[Test] ACTIVE_MANIFOLD darf Math 66b2c7c979 nicht loopy machen")
    try:
        test_px_does_not_increase_loopy_rate_on_math()
        print(f"  ✓ ACTIVE_MANIFOLD Math 66b2c7c979 ist OK (keine Regression)")
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
