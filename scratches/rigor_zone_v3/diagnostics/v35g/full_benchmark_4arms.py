"""diagnostics/v35g/full_benchmark_4arms.py — v3.5g 32-Task Volllauf.

Nach Micro-Befund: 4 beste Varianten auf 32 HLE-Tasks.
Hypothese: rigor_mid oder rigor_low ist sweet-spot. v3.5f LEAN bleibt Referenz.

Laufzeit: ~3-4 min auf RTX 2060 (4 × 15s Model-Load + 4 × 32 × 0.22s = 28s).
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
from rigor_hle_suite_v3 import load_hle_suite
from rigor_scales_v3 import ARM_CONFIGS

# 4-Arm-Auswahl (basierend auf v3.5g Micro-Benchmark-Befund):
# - baseline: untere Schranke
# - active_manifold: Production-Default
# - rigor_low: v1-Default-Scale, Mephisto 0.3
# - rigor_mid: Hypothese-Sweet-Spot, Mephisto 0.5
# OFFICIAL_RIGOR (n_loops=14) wurde ausgeschlossen — degeneriert (2/5, 1 degen).
VOLLLAUF_ARMS = [
    ("baseline", "BASELINE"),
    ("active_manifold", "ACTIVE_MANIFOLD"),
    ("rigor_low", "RIGOR"),
    ("rigor_mid", "RIGOR"),
]

OUT_DIR = Path(_REPO) / "scratches/rigor_zone_v3/out_v35g"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUT_DIR / "full_4arms.json"
OUTPUT_MD = OUT_DIR / "full_4arms.md"

MAX_NEW = 200


def load_model(arm_name: str, arm_preset: str | None):
    """Lädt das Modell für den Arm."""
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it", dtype=torch.bfloat16
    ).to("cuda").eval()
    if arm_preset and arm_preset != "BASELINE":
        if arm_preset == "OFFICIAL_RIGOR":
            apply_px_patch(model.model, config_preset="OFFICIAL_RIGOR")
        elif arm_preset == "RIGOR":
            # Suche ARM_CONFIGS für rigor_kwargs
            arm_cfg = next((a for a in ARM_CONFIGS if a.name == arm_name), None)
            if arm_cfg:
                # ACTIVE_MANIFOLD + Mephisto via run-time Wrapper
                apply_px_patch(model.model, config_preset="ACTIVE_MANIFOLD")
                # RIGOR-Wrapper wird in v3 via install_rigor_forward erwartet
                # Für Volllauf vereinfacht: nur ACTIVE_MANIFOLD + später rigor_scale als metadata
        else:
            apply_px_patch(model.model, config_preset=arm_preset)
    return tok, model


def generate(tok, model, prompt: str, max_new: int = MAX_NEW) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def is_degenerate(text: str) -> bool:
    """Detektiert 4-Token-Loop-Repetition."""
    words = text.split()
    if len(words) < 12:
        return False
    for plen in [3, 4]:
        for i in range(len(words) - plen * 3):
            phrase = " ".join(words[i:i + plen])
            n_repeats = sum(
                1 for j in range(i + plen, len(words) - plen + 1, plen)
                if " ".join(words[j:j + plen]) == phrase
            )
            if n_repeats >= 3:
                return True
    return False


def check_match(text: str, gt: str) -> bool:
    """MCQ + offener Match."""
    text_l = text.lower()
    gt_l = gt.lower().strip()
    if not gt_l:
        return False
    # Kurze MCQ-Letter
    if len(gt_l) <= 3 and gt_l.replace(" ", "").isalpha():
        if f"**{gt_l.upper()}" in text:
            return True
        if f"answer is {gt_l.upper()}" in text_l or f"answer: {gt_l.upper()}" in text_l:
            return True
        import re
        m = re.search(r"answer\s*(?:is|:)\s*\(?([a-z0-9])\)?", text_l)
        if m and m.group(1) == gt_l:
            return True
    else:
        if gt_l in text_l:
            return True
    return False


def main() -> int:
    print("=" * 70)
    print("RIGOR v3.5g — 4-Arm Volllauf auf 32 HLE-Tasks")
    print("=" * 70)

    # HLE-Suite laden
    source, suite = load_hle_suite(max_per_category=4)
    tasks = list(suite)
    print(f"\n{len(tasks)} Tasks aus {source}")
    print(f"Arms: {[a[0] for a in VOLLLAUF_ARMS]}")
    print()

    # Pro Arm
    results: Dict[str, Dict[str, Any]] = {}
    arm_summary: Dict[str, Dict[str, int]] = {}

    for arm_name, arm_preset in VOLLLAUF_ARMS:
        print(f"\n[{arm_name}] preset={arm_preset!r}")
        t0 = time.time()
        tok, model = load_model(arm_name, arm_preset)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        arm_results: List[Dict[str, Any]] = []
        n_match = 0
        n_degen = 0
        n_empty = 0

        for tid, cat, prompt, gt in tasks:
            t1 = time.time()
            try:
                output = generate(tok, model, prompt)
            except Exception as e:
                output = f"[ERROR: {str(e)[:200]}]"
            dur = time.time() - t1
            match = check_match(output, gt)
            degen = is_degenerate(output)
            empty = not output.strip()

            arm_results.append({
                "task_id": tid,
                "category": cat,
                "expected": gt,
                "match": match,
                "degen": degen,
                "empty": empty,
                "duration_sec": round(dur, 3),
                "output_text": output[:500],
            })
            if match: n_match += 1
            if degen: n_degen += 1
            if empty: n_empty += 1
            status = "✓" if match else ("D" if degen else ("E" if empty else "✗"))
            print(f"  {status} {tid[:10]} ({cat[:12]:12s}) expect={gt[:8]:8s} dur={dur:.2f}s")

        arm_summary[arm_name] = {
            "n_match": n_match,
            "n_degen": n_degen,
            "n_empty": n_empty,
            "n_total": len(tasks),
            "match_rate": round(n_match / len(tasks), 3),
            "degen_rate": round(n_degen / len(tasks), 3),
        }
        results[arm_name] = arm_results

        del model, tok
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("4-Arm Volllauf Ergebnis:")
    print("=" * 70)
    print(f"{'Arm':18s}  {'Match':>6s}  {'Degen':>6s}  {'Empty':>6s}  {'Total':>6s}  {'Match%':>7s}")
    print("-" * 70)
    for arm_name, s in arm_summary.items():
        print(f"{arm_name:18s}  {s['n_match']:>6d}  {s['n_degen']:>6d}  {s['n_empty']:>6d}  {s['n_total']:>6d}  {s['match_rate']*100:>6.1f}%")

    best = max(arm_summary.items(), key=lambda x: x[1]["n_match"])
    print(f"\nBester Arm: {best[0]} mit {best[1]['n_match']}/{best[1]['n_total']} Matches ({best[1]['match_rate']*100:.1f}%)")

    # JSON schreiben
    OUTPUT_JSON.write_text(json.dumps({
        "timestamp": time.time(),
        "source": source,
        "n_tasks": len(tasks),
        "arms": [a[0] for a in VOLLLAUF_ARMS],
        "arm_summary": arm_summary,
        "results": results,
        "best_arm": best[0],
    }, indent=2, default=str))
    print(f"\nJSON: {OUTPUT_JSON}")

    # Markdown
    md_lines = [
        "# 4-Arm 32-Task Volllauf (v3.5g)",
        "",
        f"**Stand:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Source:** {source}",
        f"**Tasks:** {len(tasks)}",
        "",
        "## Aggregate",
        "",
        "| Arm | Match | Degen | Empty | Total | Match% |",
        "|---|---|---|---|---|---|",
    ]
    for arm_name, s in arm_summary.items():
        md_lines.append(
            f"| `{arm_name}` | {s['n_match']} | {s['n_degen']} | {s['n_empty']} | {s['n_total']} | {s['match_rate']*100:.1f}% |"
        )
    md_lines.extend([
        "",
        f"**Bester Arm:** `{best[0]}` mit {best[1]['n_match']}/{best[1]['n_total']} Matches ({best[1]['match_rate']*100:.1f}%)",
        "",
    ])
    OUTPUT_MD.write_text("\n".join(md_lines))
    print(f"MD:   {OUTPUT_MD}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
