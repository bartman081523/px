"""diagnostics/v35g/micro_benchmark_8arms.py — v3.5g Micro-Benchmark.

Schneller Vergleich: 8 Arms × 5 Baseline-Match-HLE-Tasks = 40 Runs.
Misst welche rigor_* Variante die 5 Baseline-Matches erhält.

Hypothese: rigor_mid (Mephisto scale=0.5) oder rigor_low (0.3) ist sweet-spot.
v3.5f-Fix (phi-Gate für Reflector-Injection) sollte verhindern, dass rigor_full degeneriert.

Laufzeit: ~2.5 min auf RTX 2060 (8 × 15s Model-Load + 8 × 5 × 0.22s = 1s).
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# venv-Check
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

# 5 Baseline-Match-Tasks (aus v3.5f-Befund, ECHTE 24-stellige IDs)
# Humanities D, Physics 3, Biology B, Engineering B, Other 6
# Diese löste BASELINE korrekt, jetzt testen wir ob rigor_* sie auch löst.
BASELINE_MATCH_TASKS = [
    # (task_id, category, expected_letter)
    ("668825f80a642802bdfeadfa", "Humanities/Social Science", "D"),   # Weak Non-Sadism
    ("66b827b9b64deaedfbb997a2", "Physics", "3"),                       # ?
    ("66e88728ba7d8bc0d5806f3a", "Biology/Medicine", "B"),              # Only pi nucleotide diversity
    ("66ec02c52ec65d6153428744", "Engineering", "B"),                  # ?
    ("66e8d4736299517dd7a5da8c", "Other", "6"),                         # ?
]

OUT_DIR = Path(_REPO) / "scratches/rigor_zone_v3/out_v35g"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUT_DIR / "micro_8arms.json"
OUTPUT_MD = OUT_DIR / "micro_8arms.md"

MAX_NEW = 200


def load_model(arm_name: str, arm_preset: str | None):
    """Lädt das Modell für den Arm (mit PX-Patch wenn preset != BASELINE)."""
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it", dtype=torch.bfloat16
    ).to("cuda").eval()
    if arm_preset and arm_preset != "BASELINE":
        if arm_preset == "OFFICIAL_RIGOR":
            apply_px_patch(model.model, config_preset="OFFICIAL_RIGOR")
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


def check_match(text: str, expected: str) -> bool:
    """Sehr simpler MCQ-Match: suche expected als **LETTER** oder 'answer is LETTER'."""
    text_l = text.lower()
    exp_l = expected.lower().strip()
    if not exp_l:
        return False
    # **A**, **B**, etc.
    if f"**{exp_l.upper()}" in text:
        return True
    # "answer is A" / "answer: A"
    if f"answer is {exp_l.upper()}" in text_l or f"answer: {exp_l.upper()}" in text_l:
        return True
    # Plain letter (e.g. "A" am Anfang nach "**")
    if exp_l.isalpha() and len(exp_l) == 1:
        # Suche nach "the answer is X" pattern
        import re
        m = re.search(r"answer\s*(?:is|:)\s*\(?([a-z0-9])\)?", text_l)
        if m and m.group(1) == exp_l:
            return True
    return False


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


def main() -> int:
    print("=" * 70)
    print("RIGOR v3.5g — 8-Arm Micro-Benchmark (5 Baseline-Match-HLE-Tasks)")
    print("=" * 70)

    # HLE-Suite laden
    source, suite = load_hle_suite(max_per_category=10)
    all_tasks = {tid: (cat, prompt, gt) for tid, cat, prompt, gt in suite}

    # Filter auf 5 Baseline-Match-Tasks
    tasks = []
    for tid, cat, exp in BASELINE_MATCH_TASKS:
        if tid in all_tasks:
            full_cat, prompt, gt = all_tasks[tid]
            tasks.append((tid, full_cat, prompt, exp))
        else:
            print(f"  [WARN] Task {tid} nicht in Suite")

    print(f"\n{len(tasks)} Tasks × 8 Arms = {len(tasks) * 8} Runs")
    print(f"Tasks: {[t[0] for t in tasks]}")
    print()

    # Pro Arm: lade Modell einmal, generiere für alle 5 Tasks
    results: List[Dict[str, Any]] = []
    arm_summary: Dict[str, Dict[str, int]] = {}

    for arm_cfg in ARM_CONFIGS:
        arm_name = arm_cfg.name
        arm_preset = arm_cfg.preset
        # OFFICIAL_RIGOR wird als 9. Arm getestet
        if arm_name not in [a.name for a in ARM_CONFIGS]:
            continue

        print(f"\n[{arm_name}] preset={arm_preset!r} mephisto={arm_cfg.rigor_mephisto} scale={arm_cfg.mephisto_scale}")
        t0 = time.time()
        tok, model = load_model(arm_name, arm_preset)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        arm_results = []
        n_match = 0
        n_degen = 0
        n_empty = 0

        for tid, cat, prompt, exp in tasks:
            t1 = time.time()
            try:
                output = generate(tok, model, prompt)
            except Exception as e:
                output = f"[ERROR: {str(e)[:200]}]"
            dur = time.time() - t1
            match = check_match(output, exp)
            degen = is_degenerate(output)
            empty = not output.strip()

            arm_results.append({
                "arm": arm_name,
                "task_id": tid,
                "category": cat,
                "expected": exp,
                "match": match,
                "degen": degen,
                "empty": empty,
                "duration_sec": round(dur, 3),
                "output_text": output[:500],  # truncate
            })
            if match: n_match += 1
            if degen: n_degen += 1
            if empty: n_empty += 1
            status = "✓" if match else ("D" if degen else ("E" if empty else "✗"))
            print(f"  {status} {tid[:10]} ({cat:12s}) expect={exp} dur={dur:.2f}s")

        arm_summary[arm_name] = {
            "n_match": n_match, "n_degen": n_degen, "n_empty": n_empty,
            "n_total": len(tasks),
        }
        results.extend(arm_results)

        # Modell aus GPU entfernen
        del model, tok
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("8-Arm Micro-Benchmark Ergebnis:")
    print("=" * 70)
    print(f"{'Arm':18s}  {'Match':>6s}  {'Degen':>6s}  {'Empty':>6s}  {'Total':>6s}")
    print("-" * 70)
    for arm_name, s in arm_summary.items():
        print(f"{arm_name:18s}  {s['n_match']:>6d}  {s['n_degen']:>6d}  {s['n_empty']:>6d}  {s['n_total']:>6d}")

    # Best Arm identifizieren
    best = max(arm_summary.items(), key=lambda x: x[1]["n_match"])
    print(f"\nBester Arm: {best[0]} mit {best[1]['n_match']}/{best[1]['n_total']} Matches")

    # JSON schreiben
    OUTPUT_JSON.write_text(json.dumps({
        "timestamp": time.time(),
        "tasks": [{"task_id": t[0], "category": t[1], "expected": t[3]} for t in tasks],
        "arm_summary": arm_summary,
        "results": results,
        "best_arm": best[0],
    }, indent=2, default=str))
    print(f"\nJSON: {OUTPUT_JSON}")

    # Markdown schreiben
    md_lines = [
        "# 8-Arm Micro-Benchmark (v3.5g)",
        "",
        f"**Stand:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Tasks:** {len(tasks)} Baseline-Match-HLE-Tasks (5 Repräsentanten)",
        "",
        "## Aggregate",
        "",
        "| Arm | Match | Degen | Empty | Total |",
        "|---|---|---|---|---|",
    ]
    for arm_name, s in arm_summary.items():
        md_lines.append(f"| `{arm_name}` | {s['n_match']} | {s['n_degen']} | {s['n_empty']} | {s['n_total']} |")
    md_lines.extend([
        "",
        f"**Bester Arm:** `{best[0]}` mit {best[1]['n_match']}/{best[1]['n_total']} Matches",
        "",
        "## Per-Task Detail",
        "",
    ])
    for arm_name, _ in arm_summary.items():
        arm_res = [r for r in results if r["arm"] == arm_name]
        md_lines.append(f"### `{arm_name}`")
        md_lines.append("")
        for r in arm_res:
            status = "✓" if r["match"] else ("D" if r["degen"] else ("E" if r["empty"] else "✗"))
            md_lines.append(f"- {status} `{r['task_id'][:10]}` ({r['category']:12s}) expect={r['expected']} dur={r['duration_sec']}s")
        md_lines.append("")

    OUTPUT_MD.write_text("\n".join(md_lines))
    print(f"MD:   {OUTPUT_MD}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
