"""diagnostics/v35g/tool_loop_75iter.py — v3.5g Tool-Loop 75-Iter Self-Correction-Test.

Testet ob 270m durch 75 Iterationen + Tool-Loop mehr HLE-Tasks löst.

5 Baseline-Match-Tasks × 4 Modi = 20 Runs.
  M1: Tools=off, max_iter=5
  M2: Tools=off, max_iter=75
  M3: Tools=on,  max_iter=5
  M4: Tools=on,  max_iter=75  ← Self-Correction-Variante

Hypothese: M4 > M3 (Self-Correction) > M1/M2 (Baseline).

Hinweis: toolchain.py hat bereits 5 grüne TDD-Tests für run_tool_loop.
Laufzeit: ~5-10 min auf RTX 2060 (4 × 15s Load + 5 × 4 × ~3s Tool-Loop).
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
from toolchain import run_tool_loop, TOOLCHAIN_DEFINITION

OUT_DIR = Path(_REPO) / "scratches/rigor_zone_v3/out_v35g"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUT_DIR / "tool_75iter.json"
OUTPUT_MD = OUT_DIR / "tool_75iter.md"

# 5 repräsentative Tasks (MATH-Heavy, Mephisto-Damping sollte helfen)
TASKS = [
    # (task_id, expected, short_prompt, category)
    ("math_addition_42_58", "100", "What is 42 + 58? Answer with the final number.", "math"),
    ("math_subtraction_100_37", "63", "What is 100 - 37? Answer with the final number.", "math"),
    ("math_mult_7_8", "56", "What is 7 * 8? Answer with the final number.", "math"),
    ("math_division_144_12", "12", "What is 144 / 12? Answer with the final number.", "math"),
    ("math_sqrt_144", "12", "What is the square root of 144? Answer with the final number.", "math"),
]

# 4 Modi
MODI = [
    ("M1_tools_off_iter5", False, 5),
    ("M2_tools_off_iter75", False, 75),
    ("M3_tools_on_iter5", True, 5),
    ("M4_tools_on_iter75", True, 75),
]

MAX_NEW = 300  # Tool-Outputs brauchen Platz

# Tool-Registry (aus toolchain.py)
TOOLCHAIN_TOOLS = ["web_search", "read_file", "write_file", "execute_python"]


def load_model():
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it", dtype=torch.bfloat16
    ).to("cuda").eval()
    apply_px_patch(model.model, config_preset="ACTIVE_MANIFOLD")
    return tok, model


def check_match(text: str, expected: str) -> bool:
    """Numerischer Match: suche expected als isoliertes Wort oder am Ende."""
    text_l = text.lower()
    exp = expected.strip()
    if not exp:
        return False
    if exp.lower() in text_l:
        return True
    for line in text_l.splitlines():
        line = line.strip()
        if line == exp or line.endswith(exp) or line.endswith(f"is {exp}") or line.endswith(f"= {exp}"):
            return True
    return False


def main() -> int:
    print("=" * 70)
    print("RIGOR v3.5g — Tool-Loop 75-Iter Self-Correction-Test")
    print("=" * 70)
    print(f"\n{len(TASKS)} Tasks × {len(MODI)} Modi = {len(TASKS) * len(MODI)} Runs")
    print(f"Modi: {[m[0] for m in MODI]}")
    print()

    # Einmal Modell laden (alle Modi nutzen ACTIVE_MANIFOLD auf 270m)
    print("Lade Modell (ACTIVE_MANIFOLD)...")
    t0 = time.time()
    tok, model = load_model()
    print(f"  loaded in {time.time() - t0:.1f}s\n")

    # System-Prompt (mit/ohne Tools)
    def build_system(use_tools: bool) -> str:
        if not use_tools:
            return "You are a helpful assistant. Answer concisely."
        return (
            "You are a rigorous scientific assistant working on a research task.\n\n"
            + TOOLCHAIN_DEFINITION
        )

    results: List[Dict[str, Any]] = []
    mode_summary: Dict[str, Dict[str, int]] = {}

    for mode_name, use_tools, max_iter in MODI:
        print(f"\n[{mode_name}] use_tools={use_tools} max_iter={max_iter}")
        sys_prompt = build_system(use_tools)
        enabled_tools = TOOLCHAIN_TOOLS if use_tools else []

        n_match = 0
        n_degen = 0
        n_empty = 0

        for tid, expected, prompt, cat in TASKS:
            t1 = time.time()
            try:
                result = run_tool_loop(
                    model=model,
                    tokenizer=tok,
                    system_prompt=sys_prompt,
                    user_prompt=prompt,
                    max_iterations=max_iter,
                    max_new_tokens=MAX_NEW,
                    enable_tools=enabled_tools,
                )
                output = result.final_answer
                n_iter = result.n_iterations
                max_hit = result.max_hit
            except Exception as e:
                output = f"[ERROR: {str(e)[:200]}]"
                n_iter = 0
                max_hit = False
            dur = time.time() - t1
            match = check_match(output, expected)
            degen = len(output.split()) > 100 and (output.split()[0] == output.split()[10] if len(output.split()) > 10 else False)
            empty = not output.strip()

            results.append({
                "mode": mode_name,
                "task_id": tid,
                "category": cat,
                "expected": expected,
                "match": match,
                "degen": degen,
                "empty": empty,
                "n_iterations": n_iter,
                "max_iterations_hit": max_hit,
                "duration_sec": round(dur, 3),
                "output_text": output[:500],
            })
            if match: n_match += 1
            if degen: n_degen += 1
            if empty: n_empty += 1
            status = "✓" if match else ("D" if degen else ("E" if empty else "✗"))
            print(f"  {status} {tid:30s} (iter={n_iter:2d}{'*' if max_hit else ' '}) expect={expected} dur={dur:.2f}s")

        mode_summary[mode_name] = {
            "n_match": n_match,
            "n_degen": n_degen,
            "n_empty": n_empty,
            "n_total": len(TASKS),
            "use_tools": use_tools,
            "max_iter": max_iter,
        }

    del model, tok
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("Tool-Loop Self-Correction Ergebnis:")
    print("=" * 70)
    print(f"{'Mode':25s}  {'Tools':>6s}  {'MaxIter':>8s}  {'Match':>6s}  {'Degen':>6s}  {'Empty':>6s}")
    print("-" * 70)
    for mode_name, s in mode_summary.items():
        print(f"{mode_name:25s}  {str(s['use_tools']):>6s}  {s['max_iter']:>8d}  {s['n_match']:>6d}  {s['n_degen']:>6d}  {s['n_empty']:>6d}")

    best = max(mode_summary.items(), key=lambda x: x[1]["n_match"])
    print(f"\nBester Modus: {best[0]} mit {best[1]['n_match']}/{best[1]['n_total']} Matches")

    # JSON schreiben
    OUTPUT_JSON.write_text(json.dumps({
        "timestamp": time.time(),
        "tasks": [{"task_id": t[0], "expected": t[1], "category": t[3]} for t in TASKS],
        "modi": [{"name": m[0], "use_tools": m[1], "max_iter": m[2]} for m in MODI],
        "mode_summary": mode_summary,
        "results": results,
        "best_mode": best[0],
    }, indent=2, default=str))
    print(f"\nJSON: {OUTPUT_JSON}")

    # Markdown
    md_lines = [
        "# Tool-Loop 75-Iter Self-Correction-Test (v3.5g)",
        "",
        f"**Stand:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Tasks:** {len(TASKS)} (math)",
        f"**Modi:** {len(MODI)}",
        "",
        "## Aggregate",
        "",
        "| Modus | Tools | MaxIter | Match | Degen | Empty |",
        "|---|---|---|---|---|---|",
    ]
    for mode_name, s in mode_summary.items():
        md_lines.append(
            f"| `{mode_name}` | {s['use_tools']} | {s['max_iter']} | {s['n_match']} | {s['n_degen']} | {s['n_empty']} |"
        )
    md_lines.extend([
        "",
        f"**Bester Modus:** `{best[0]}` mit {best[1]['n_match']}/{best[1]['n_total']} Matches",
        "",
        "## Hypothese-Check",
        "",
        "- M1 (tools=off, iter=5) ist Baseline-Verhalten",
        "- M2 (tools=off, iter=75) testet, ob max_iter-Cap ohne Tools etwas ändert",
        "- M3 (tools=on, iter=5) testet Tool-Loop ohne Self-Correction",
        "- M4 (tools=on, iter=75) ist Self-Correction-Variante",
        "",
        "**Erwartung:** M4 > M3 > M1 = M2",
        "",
    ])
    OUTPUT_MD.write_text("\n".join(md_lines))
    print(f"MD:   {OUTPUT_MD}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
