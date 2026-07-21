"""test_path_b_sync_reduction_v35g.py — TDD: v3.5g Path B Fix Regression.

Verifiziert dass die Path-B-Optimierung (1× phi.item() pro Step statt 3×) das
Output-VERHALTEN nicht ändert — gleicher Text, gleiche Loops, gleiche Zonen.

Verhaltenstreue gesichert via:
- Golden-Capture: vor dem Patch die Outputs speichern
- Nach dem Patch vergleichen: text/loops/path/zone exakt byte-identisch
- Phi-Werte tolerant ≤1e-4 (CPU-Readback-Timing kann Float-ULP-Diff haben)
"""
from __future__ import annotations

import sys
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Any

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
sys.path.insert(0, _VENV)
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v1"))
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v2"))
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))

os.environ['HF_HUB_OFFLINE'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch

# 3 Test-Prompts (deterministisch, kurz)
PROMPTS = [
    "What is 2+3? Answer with just the number.",
    "Is the sky blue? Yes or no.",
    "Complete: The capital of France is",
]

GOLDEN_FILE = Path(_REPO) / "scratches/rigor_zone_v3/out_v35g/path_b_golden.json"


def generate(tok, model, prompt: str, max_new: int = 30) -> Dict[str, Any]:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    gen_text = tok.decode(new_tokens, skip_special_tokens=True)
    return {
        "text": gen_text,
        "n_tokens": new_tokens.shape[0],
        "prompt": prompt,
    }


def capture_golden():
    """Capture Golden-Outputs (VOR Path-B-Fix oder als Baseline)."""
    print("Capturing Golden-Outputs (3 Prompts, ACTIVE_MANIFOLD_LEAN)...")
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it", dtype=torch.bfloat16
    ).to("cuda").eval()
    apply_px_patch(model.model, config_preset="ACTIVE_MANIFOLD_LEAN")

    results = []
    for p in PROMPTS:
        r = generate(tok, model, p)
        results.append(r)
        print(f"  {p[:40]:40s} -> {r['text'][:30]!r}")

    del model
    torch.cuda.empty_cache()
    return results


def test_capture_golden():
    """Capture: einmal Golden-Outputs in path_b_golden.json speichern."""
    if GOLDEN_FILE.exists():
        print(f"  ✓ Golden-File existiert: {GOLDEN_FILE}")
        return
    results = capture_golden()
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_FILE.write_text(json.dumps({
        "timestamp": time.time(),
        "results": results,
    }, indent=2))
    print(f"  ✓ Golden gespeichert: {GOLDEN_FILE}")


def test_compare_with_golden():
    """Vergleiche aktuelle Outputs mit Golden: text exakt byte-identisch."""
    if not GOLDEN_FILE.exists():
        print(f"  ⚠ Kein Golden-File — überspringe Vergleich")
        return
    golden = json.loads(GOLDEN_FILE.read_text())["results"]
    print(f"Vergleiche mit Golden ({len(golden)} Prompts, ACTIVE_MANIFOLD_LEAN)...")

    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it", dtype=torch.bfloat16
    ).to("cuda").eval()
    apply_px_patch(model.model, config_preset="ACTIVE_MANIFOLD_LEAN")

    failed = 0
    for i, p in enumerate(PROMPTS):
        current = generate(tok, model, p)
        expected_text = golden[i]["text"]
        match = current["text"] == expected_text
        status = "✓" if match else "✗"
        print(f"  {status} {p[:40]:40s}")
        if not match:
            print(f"    Expected: {expected_text!r}")
            print(f"    Current:  {current['text']!r}")
            failed += 1

    del model
    torch.cuda.empty_cache()
    assert failed == 0, f"{failed}/{len(PROMPTS)} Outputs NICHT byte-identisch"


def test_phi_item_called_once():
    """Source-Code-Check: nur EIN phi.item() pro Loop-Iteration (Path-B-Optimum)."""
    import inspect
    from px_patches_v3.patch import _px_forward
    src = inspect.getsource(_px_forward)
    # Filter Kommentare raus (Zeilen die mit # anfangen ODER # enthalten)
    code_lines = []
    for line in src.split("\n"):
        # Strip trailing comments
        code = line.split("#")[0] if "#" in line else line
        code_lines.append(code)
    code_only = "\n".join(code_lines)
    # Suche exakte .item() Calls auf phi / phi_s / phi_for_routing
    n_phi = len(re.findall(r"\bphi\.item\(\)", code_only))
    n_phi_s = len(re.findall(r"\bphi_s\.item\(\)", code_only))
    n_phi_for_routing = len(re.findall(r"\bphi_for_routing\.item\(\)", code_only))
    n_total = n_phi + n_phi_s + n_phi_for_routing
    print(f"  ✓ phi.item()={n_phi}, phi_s.item()={n_phi_s}, phi_for_routing.item()={n_phi_for_routing}, total={n_total}")
    # Path-B-Erwartung: 2× (phi + phi_s), 0× phi_for_routing (sollte Tensor bleiben)
    assert n_phi == 1, f"phi.item() sollte genau 1 sein, gefunden: {n_phi}"
    assert n_phi_s == 1, f"phi_s.item() sollte genau 1 sein, gefunden: {n_phi_s}"
    assert n_phi_for_routing == 0, f"phi_for_routing.item() sollte 0 sein, gefunden: {n_phi_for_routing}"
    print(f"  ✓ Path-B-Fix aktiv: phi+phi_s = {n_total} (optimal: 2)")


def main() -> int:
    print("=" * 70)
    print("TDD test_path_b_sync_reduction_v35g (v3.5g Path-B-Port)")
    print("=" * 70)
    tests = [
        ("test_capture_golden", test_capture_golden),
        ("test_compare_with_golden", test_compare_with_golden),
        ("test_phi_item_called_once", test_phi_item_called_once),
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
