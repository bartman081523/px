"""
HLE 5-Arm-Benchmark für RIGOR-Zone.

5 Arms × 5 HLE-Tasks × 2 Search-Settings = 50 Runs.

WICHTIG: KEINE Production-Änderung. RIGOR-Overrides passieren via
rigor_zone_manifold.py (Dicts + AutoCalibrator-Method-Patches) und
rigor_zone_forward.py (Mephisto-Damping-Wrapper) — beides Monkey-Patches
zur Laufzeit.

Hermes-Suche: wenn `hermes` binary im PATH nicht gefunden wird,
ist with_search=True ein NoOp (Run mit search=True hat gleichen Output
wie mit search=False). Report dokumentiert das.
"""
import os
import sys
import json
import time
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

# Sicherstellen dass das Repo-Root im Path ist (für model_manager etc.)
_REPO_ROOT = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Eigene Imports (triggert die Monkey-Patches in rigor_zone_manifold)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rigor_zone_config import (
    HLE_ARMS, HLE_TASKS, OUT_DIR_NAME, RESULTS_MD_NAME,
    RIGOR_MEPHISTO_DEFAULT_SCALE,
)
from rigor_zone_manifold import _at  # noqa: F401  (triggert Patches)
from rigor_zone_forward import install_rigor_forward, restore_rigor_forward


# ═══════════════════════════════════════════════════════════════════════════════
# Hermes-Tool-Detection
# ═══════════════════════════════════════════════════════════════════════════════

def find_hermes() -> Optional[str]:
    """Sucht nach Hermes-Agent binary. Returnt Pfad oder None."""
    # 1. PATH
    try:
        result = subprocess.run(
            ["which", "hermes"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    # 2. Häufige Pfade
    for path in [
        "/usr/local/bin/hermes",
        "/usr/bin/hermes",
        os.path.expanduser("~/.local/bin/hermes"),
        os.path.expanduser("~/bin/hermes"),
    ]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


HERMES_PATH = find_hermes()
HERMES_AVAILABLE = HERMES_PATH is not None
if HERMES_AVAILABLE:
    print(f"[HLE-Harness] Hermes Agent gefunden: {HERMES_PATH}")
else:
    print(f"[HLE-Harness] Hermes Agent NICHT gefunden.")

# ddgs (DuckDuckGo) als direkter Web-Such-Pfad
try:
    from ddgs import DDGS as _DDGS_PROBE
    WEB_SEARCH_AVAILABLE = True
    print(f"[HLE-Harness] Web-Suche via ddgs: verfügbar")
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print(f"[HLE-Harness] Web-Suche via ddgs: NICHT verfügbar — with_search=True läuft als NoOp")


# ═══════════════════════════════════════════════════════════════════════════════
# Web-Suche (DuckDuckGo via ddgs, kein API-Key)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Direkter DDG-Scrape ist die einfachste robuste Lösung:
# - kein API-Key, kein Rate-Limit-Token
# - block-frei (kein LLM im Loop wie bei hermes chat -q)
# - liefert Titel + Snippet + URL pro Result
# User-direktive 2026-07-11: Hermes Agent installiert, wir nutzen aber
# den direkten Web-Such-Pfad weil hermes chat -q den LLM-Loop triggert
# (würde unsere PX-Engine umgehen).

def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """DDG-Web-Suche, returnt Liste von {title, href, body}."""
    if not WEB_SEARCH_AVAILABLE:
        return []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [
                {"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")}
                for r in results
            ]
    except Exception as e:
        return [{"error": str(e)}]


def augment_prompt_with_search(prompt: str, with_search: bool) -> str:
    """Wenn with_search=True, ergänze Prompt mit DDG-Suchergebnissen.
    Extrahiert 1-2 Keywords aus dem Prompt (heuristisch: längste Wörter).
    """
    if not with_search:
        return prompt
    # Einfache Keyword-Extraktion: 2 längste Wörter (>=5 chars)
    words = [w.strip(".,!?;:\"'()[]{}") for w in prompt.split() if len(w.strip(".,!?;:\"'()[]{}")) >= 5]
    if not words:
        return prompt
    # Sortiere nach Länge desc, nimm 2 längste
    keywords = sorted(set(words), key=len, reverse=True)[:2]
    query = " ".join(keywords)
    results = web_search(query, max_results=3)
    if not results or "error" in results[0]:
        return prompt
    # Formatiere als Kontext-Block
    context_lines = ["\n[Web search results for context — DO NOT repeat these verbatim, use as inspiration:]"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")[:200]
        href = r.get("href", "")
        context_lines.append(f"  [{i}] {title}")
        if body:
            context_lines.append(f"      {body}")
        if href:
            context_lines.append(f"      {href}")
    context_lines.append("")  # trailing newline
    return prompt + "\n".join(context_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Model-Loading
# ═══════════════════════════════════════════════════════════════════════════════

# Wir nutzen model_manager._load_model direkt (sync), nicht async get_model
# → vermeidet asyncio-Setup-Overhead
def load_model_for_arm(arm_name: str, preset: Optional[str], patch_kwargs: Optional[Dict]):
    """Lade gemma3-270m-it via ModelManager._load_model.

    WICHTIG: RIGOR-Override wird NACH apply_px_patch via
    install_rigor_forward() aktiviert.
    """
    from model_manager import ModelManager, _migrate_preset

    manager = ModelManager()
    model_id = "gemma3-270m-it"
    effective_preset = _migrate_preset(preset) if preset else "BASELINE"
    print(f"  → Loading {model_id} with preset={effective_preset} (arm={arm_name})...")

    # 1. Load model + apply patch
    entry = manager._load_model(
        model_id,
        px_subjective=True,  # Standard für aktive_manifold und rigor
        px_config_preset=effective_preset,
    )

    # 2. RIGOR-Override installieren (nur für RIGOR-Arms)
    if preset == "RIGOR" and patch_kwargs:
        # Resolve text_model für die Forward-Patches
        tm = manager._resolve_text_model(entry["model"])
        rigor_info = install_rigor_forward(
            tm,
            config_preset="RIGOR",
            **(patch_kwargs or {}),
        )
        print(f"  → RIGOR-Override installiert: {rigor_info}")

    return entry, manager


# ═══════════════════════════════════════════════════════════════════════════════
# Run-Arm (PX-Engine)
# ═══════════════════════════════════════════════════════════════════════════════

def run_arm_px(
    arm_name: str, preset: str, patch_kwargs: Optional[Dict],
    with_search: bool, task_name: str, task_prompt: str,
) -> Dict[str, Any]:
    """Run a single PX-Arm × Task × Search-Setting."""
    start_ts = time.time()
    result = {
        "arm": arm_name,
        "preset": preset,
        "patch_kwargs": patch_kwargs,
        "with_search": with_search,
        "task_name": task_name,
        "task_prompt": task_prompt,
        "timestamp": start_ts,
    }
    try:
        # 1. Web-Suche (DDG via ddgs)
        augmented_prompt = augment_prompt_with_search(task_prompt, with_search)
        result["web_search_invoked"] = with_search and WEB_SEARCH_AVAILABLE
        result["search_context_chars"] = len(augmented_prompt) - len(task_prompt)

        # 2. Load model
        entry, manager = load_model_for_arm(arm_name, preset, patch_kwargs)
        model = entry["model"]
        tokenizer = entry["tokenizer"]
        device = next(model.parameters()).device

        # 3. Prepare prompt
        messages = [{"role": "user", "content": augmented_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        n_input_tokens = inputs["input_ids"].shape[1]

        # 4. Generate
        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,  # Greedy wie Greedy-Decoding-Default
                temperature=0.7,
            )
        n_new_tokens = outputs.shape[1] - n_input_tokens
        text = tokenizer.decode(
            outputs[0][n_input_tokens:], skip_special_tokens=True,
        ).strip()

        # 5. Metrics (PX-spezifisch wenn verfügbar)
        tm = manager._resolve_text_model(model)
        zone = getattr(tm, "_px_zone", "UNKNOWN")

        result.update({
            "success": True,
            "output": text,
            "n_input_tokens": n_input_tokens,
            "n_new_tokens": n_new_tokens,
            "px_zone": zone,
            "duration_sec": time.time() - start_ts,
        })

        # 6. Cleanup RIGOR-Override
        if preset == "RIGOR":
            restore_rigor_forward(tm)

    except Exception as e:
        result.update({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "duration_sec": time.time() - start_ts,
        })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Run-Arm (minimaxm3:cloud via Ollama)
# ═══════════════════════════════════════════════════════════════════════════════

def run_arm_ollama_cloud(
    with_search: bool, task_name: str, task_prompt: str,
) -> Dict[str, Any]:
    """Run a single cloud-Arm × Task × Search via Ollama subprocess."""
    start_ts = time.time()
    result = {
        "arm": "minimaxm3_cloud",
        "preset": None,
        "patch_kwargs": None,
        "with_search": with_search,
        "task_name": task_name,
        "task_prompt": task_prompt,
        "timestamp": start_ts,
    }
    try:
        # 1. Web-Suche
        augmented_prompt = augment_prompt_with_search(task_prompt, with_search)
        result["web_search_invoked"] = with_search and WEB_SEARCH_AVAILABLE
        result["search_context_chars"] = len(augmented_prompt) - len(task_prompt)

        # 2. Ollama-Subprocess
        proc = subprocess.run(
            ["ollama", "run", "minimaxm3:cloud", augmented_prompt],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ollama failed (rc={proc.returncode}): {proc.stderr[:200]}")

        text = proc.stdout.strip()
        result.update({
            "success": True,
            "output": text,
            "duration_sec": time.time() - start_ts,
        })
    except Exception as e:
        result.update({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "duration_sec": time.time() - start_ts,
        })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(_HERE) / OUT_DIR_NAME
    out_dir.mkdir(exist_ok=True)

    print("="*60)
    print(f"HLE 5-Arm-Benchmark")
    print(f"5 Arms × 5 HLE-Tasks × 2 Search-Settings = {len(HLE_ARMS)*len(HLE_TASKS)*2} Runs")
    print(f"Output: {out_dir}")
    print("="*60)
    print(f"\nArms:")
    for arm_name, preset, kw in HLE_ARMS:
        print(f"  - {arm_name:18s} preset={preset!s:18s} kwargs={kw}")
    print(f"\nHLE-Tasks: {len(HLE_TASKS)}")
    print(f"Search-Settings: 2 (False, True)")
    print(f"Hermes verfügbar: {HERMES_AVAILABLE} ({HERMES_PATH or 'N/A'})")
    print("="*60)

    results: List[Dict[str, Any]] = []
    start_total = time.time()

    for arm_name, preset, patch_kwargs in HLE_ARMS:
        for with_search in [False, True]:
            for task_name, task_prompt in HLE_TASKS:
                run_idx = len(results) + 1
                print(f"\n[Run {run_idx:2d}/50] arm={arm_name:18s} search={with_search} task={task_name}")
                if arm_name == "minimaxm3_cloud":
                    res = run_arm_ollama_cloud(with_search, task_name, task_prompt)
                else:
                    res = run_arm_px(arm_name, preset, patch_kwargs, with_search, task_name, task_prompt)

                # Speichere pro-Run JSON
                fname = f"arm_{arm_name}_{task_name}_search{int(with_search)}.json"
                out_path = out_dir / fname
                out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
                results.append(res)

                # Kurze Status-Zeile
                if res.get("success"):
                    preview = (res.get("output", "")[:80] + "...") if len(res.get("output", "")) > 80 else res.get("output", "")
                    print(f"  ✓ {res.get('duration_sec', 0):.1f}s | zone={res.get('px_zone', 'N/A'):10s} | {preview}")
                else:
                    print(f"  ✗ {res.get('duration_sec', 0):.1f}s | ERROR: {res.get('error', '?')[:100]}")

    total_duration = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"Alle 50 Runs abgeschlossen in {total_duration/60:.1f} min")
    print(f"Output: {out_dir}")
    print(f"Schreibe Report: {RESULTS_MD_NAME}")
    print(f"{'='*60}")

    # Report generieren
    write_report(results, total_duration)


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown-Report
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(results: List[Dict[str, Any]], total_duration: float):
    """Schreibt hle_results.md mit Vergleichstabelle + Aggregaten."""
    md_path = Path(_HERE) / RESULTS_MD_NAME
    arms_names = [a[0] for a in HLE_ARMS]
    task_names = [t[0] for t in HLE_TASKS]

    # Output-Grouper
    by_run = {}  # (arm, task, search) → result
    for r in results:
        key = (r["arm"], r["task_name"], r["with_search"])
        by_run[key] = r

    lines = []
    lines.append("# HLE 5-Arm-Benchmark — RIGOR Zone v1")
    lines.append("")
    lines.append(f"**Stand:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Duration:** {total_duration/60:.1f} min")
    lines.append(f"**Web-Suche (ddgs):** {'verfügbar' if WEB_SEARCH_AVAILABLE else 'NICHT verfügbar — with_search=True ist NoOp'}")
    lines.append(f"**Hermes Agent:** {'verfügbar ('+HERMES_PATH+')' if HERMES_AVAILABLE else 'NICHT gefunden'}")
    lines.append("")
    lines.append("## 5 Arms")
    lines.append("")
    lines.append("| Arm | Preset | Patch-Kwargs |")
    lines.append("|---|---|---|")
    for arm, preset, kw in HLE_ARMS:
        lines.append(f"| `{arm}` | `{preset or 'N/A'}` | `{kw or '{}'}` |")
    lines.append("")

    # Pro-Task: 5 Spalten × 2 Search
    for search_setting in [False, True]:
        lines.append(f"## Search = {search_setting}")
        lines.append("")
        lines.append(f"| Task | " + " | ".join(arms_names) + " |")
        lines.append("|" + "---|" * (len(arms_names) + 1))
        for task_name, _ in HLE_TASKS:
            row = [f"**{task_name}**"]
            for arm in arms_names:
                key = (arm, task_name, search_setting)
                r = by_run.get(key, {})
                if r.get("success"):
                    out = r.get("output", "")
                    duration = r.get("duration_sec", 0)
                    zone = r.get("px_zone", "N/A")
                    preview = out[:60].replace("\n", " ").replace("|", "\\|")
                    row.append(f"✓ {duration:.1f}s<br>zone={zone}<br>{preview}...")
                else:
                    err = r.get("error", "?")[:40].replace("|", "\\|")
                    row.append(f"✗ {err}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Aggregate
    lines.append("## Aggregate (Success-Rate + Avg-Duration)")
    lines.append("")
    lines.append("| Arm | Success (10 Runs) | Avg Duration | Avg Output-Length |")
    lines.append("|---|---|---|---|")
    for arm in arms_names:
        arm_results = [r for r in results if r["arm"] == arm]
        n_success = sum(1 for r in arm_results if r.get("success"))
        avg_dur = sum(r.get("duration_sec", 0) for r in arm_results) / max(len(arm_results), 1)
        avg_len = sum(len(r.get("output", "")) for r in arm_results if r.get("success")) / max(n_success, 1)
        lines.append(f"| `{arm}` | {n_success}/10 | {avg_dur:.1f}s | {avg_len:.0f} chars |")
    lines.append("")

    # Vollständige Outputs
    lines.append("## Vollständige Outputs")
    lines.append("")
    for arm in arms_names:
        lines.append(f"### Arm: {arm}")
        lines.append("")
        for search_setting in [False, True]:
            lines.append(f"**search={search_setting}**")
            lines.append("")
            for task_name, task_prompt in HLE_TASKS:
                key = (arm, task_name, search_setting)
                r = by_run.get(key, {})
                lines.append(f"- **{task_name}**")
                if r.get("success"):
                    out = r.get("output", "(leer)")
                    lines.append(f"  - duration: {r.get('duration_sec', 0):.1f}s")
                    lines.append(f"  - zone: {r.get('px_zone', 'N/A')}")
                    lines.append(f"  - output:")
                    for line in out.split("\n"):
                        lines.append(f"    > {line}")
                else:
                    lines.append(f"  - ERROR: {r.get('error', '?')[:200]}")
                lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report geschrieben: {md_path}")


if __name__ == "__main__":
    main()
