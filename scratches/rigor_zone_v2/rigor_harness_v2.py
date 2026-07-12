"""RIGOR v2 — Epistemischer Haupt-Harness (Worker + Decoder 2-Stage).

SciMind 5.0 — Core Mandates (gemini-nightly + hermes-agent port):
  1. Incomplete Suggestion: Hypothesen H1-H4 sind in README pre-registriert
  2. Empirical Verification: Wir prüfen stdout/stderr + JSON-Output, nicht nur exit codes
  3. Falsificationism: Wir suchen aktiv nach Daten, die Hypothesen widerlegen
  4. Pipeline Integrity: set -o pipefail in allen Shell-Wrappern
  5. Zero-Trust Environment: Pre-Lauf-Probes für ollama, hermes, ddgs, datasets
  6. Atomic & Secured: Jeder Run schreibt JSON VOR dem nächsten (kein Verlust bei Crash)
  7. Anti-Embedding: Worker-Output (gemma3-270m) wird NIE als Tool-Call gerundet;
     strikte 2-Stage-Trennung: Worker generiert → Decoder verifiziert (anderes Modell).

Architektur (4 Schichten):
  - Schicht 1 (Daten): rigor_hle_suite.py + rigor_scales_v2.py
  - Schicht 2 (Verifikation + Stats): rigor_verify.py + rigor_stats.py
  - Schicht 3 (Harness): diese Datei
  - Schicht 4 (Report): rigor_report.py

Grid (8 Arm × 30 Tasks × 2 Search × 1 Seed = 480 Runs, + 80 Reproducibility)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Repo-Root
_REPO_ROOT = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO_ROOT)

# Local v2 modules
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from rigor_hle_suite import load_hle_suite
from rigor_scales_v2 import ARM_CONFIGS
from rigor_verify import (
    verify_by_category, compute_self_report_flags,
    compute_self_report_flags_batch, verify_by_category_batch,
    build_decoder_prompt, build_judge_prompt, parse_judge_verdict,
    aggregate_judge_votes,
)
from rigor_stats import hash_output, reproducibility_check

# v1 reuse (nur die rigor_forward, NICHT v1-Harness)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scratches/rigor_zone_v1"))
from rigor_zone_manifold import _at  # noqa: F401  (triggert Monkey-Patches)
from rigor_zone_forward import install_rigor_forward, restore_rigor_forward


# ═══════════════════════════════════════════════════════════════════════════════
# SciMind Pre-Lauf-Probes (Zero-Trust Environment)
# ═══════════════════════════════════════════════════════════════════════════════

def pre_run_probes() -> Dict[str, bool]:
    """Verifiziert alle Dependencies (SciMind 5.0: Zero-Trust)."""
    probes = {}

    # 1. ollama (für minimaxm3:cloud falls verfügbar, sonst skip)
    try:
        r = subprocess.run(["which", "ollama"], capture_output=True, text=True, timeout=5)
        probes["ollama"] = r.returncode == 0
    except Exception:
        probes["ollama"] = False

    # 2. ddgs (Web-Suche)
    try:
        import ddgs  # noqa: F401
        probes["ddgs"] = True
    except ImportError:
        probes["ddgs"] = False

    # 3. datasets (HLE-Suite)
    try:
        import datasets  # noqa: F401
        probes["datasets"] = True
    except ImportError:
        probes["datasets"] = False

    # 4. torch + CUDA
    try:
        import torch
        probes["torch_cuda"] = torch.cuda.is_available()
    except ImportError:
        probes["torch_cuda"] = False

    # 5. hermes
    try:
        r = subprocess.run(["which", "hermes"], capture_output=True, text=True, timeout=5)
        probes["hermes"] = r.returncode == 0
    except Exception:
        probes["hermes"] = False

    return probes


# ═══════════════════════════════════════════════════════════════════════════════
# Web-Suche (DDG via ddgs, v1-Pattern wiederverwendet)
# ═══════════════════════════════════════════════════════════════════════════════

def web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """DDG-Web-Suche via ddgs. Returnt leere Liste wenn nicht verfügbar."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [
                {"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")}
                for r in results
            ]
    except Exception:
        return []


def augment_prompt_with_search(prompt: str, with_search: bool) -> str:
    """Wenn with_search=True, ergänze Prompt mit DDG-Suchergebnissen."""
    if not with_search:
        return prompt
    words = [w.strip(".,!?;:\"'()[]{}") for w in prompt.split() if len(w.strip(".,!?;:\"'()[]{}")) >= 5]
    if not words:
        return prompt
    keywords = sorted(set(words), key=len, reverse=True)[:2]
    query = " ".join(keywords)
    results = web_search(query, max_results=3)
    if not results:
        return prompt
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
    context_lines.append("")
    return prompt + "\n".join(context_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Worker (gemma3-270m-it mit apply_px_patch)
# ═══════════════════════════════════════════════════════════════════════════════

# Globales Model-Cache: GPU-Cache zwischen Runs nicht clearen
_MODEL_CACHE: Dict[str, Any] = {}


def load_model_for_arm(arm_name: str, preset: str, patch_kwargs: Optional[Dict]):
    """Lade gemma3-270m-it via ModelManager. Cached pro Preset."""
    from model_manager import ModelManager, _migrate_preset

    cache_key = f"{preset}_{json.dumps(patch_kwargs or {}, sort_keys=True)}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    manager = ModelManager()
    model_id = "gemma3-270m-it"
    effective_preset = _migrate_preset(preset) if preset else "BASELINE"
    print(f"    → Loading {model_id} with preset={effective_preset} (arm={arm_name})...")

    entry = manager._load_model(
        model_id,
        px_subjective=(preset != "BASELINE"),
        px_config_preset=effective_preset,
    )

    # RIGOR-Override installieren (nur für RIGOR-Arms)
    if preset == "RIGOR" and patch_kwargs:
        tm = manager._resolve_text_model(entry["model"])
        rigor_info = install_rigor_forward(
            tm,
            config_preset="RIGOR",
            **(patch_kwargs or {}),
        )
        print(f"    → RIGOR-Override installiert: {rigor_info}")

    result = (entry, manager)
    _MODEL_CACHE[cache_key] = result
    return result


def run_worker(
    arm_name: str, preset: str, patch_kwargs: Optional[Dict],
    prompt: str, max_new_tokens: int = 300, seed: int = 42,
) -> Dict[str, Any]:
    """Generiert Output via gemma3-270m-it (Worker-Stage 1)."""
    import torch
    start_ts = time.time()
    result = {
        "arm": arm_name,
        "preset": preset,
        "patch_kwargs": patch_kwargs,
        "seed": seed,
        "timestamp": start_ts,
    }
    try:
        # Set seed (SciMind: Reproduzierbarkeit)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        entry, manager = load_model_for_arm(arm_name, preset, patch_kwargs)
        model = entry["model"]
        tokenizer = entry["tokenizer"]
        device = next(model.parameters()).device

        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        n_input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy (SciMind: deterministisch)
                temperature=1.0,
            )
        n_new_tokens = outputs.shape[1] - n_input_tokens
        text = tokenizer.decode(
            outputs[0][n_input_tokens:], skip_special_tokens=True,
        ).strip()

        # PX-Zone (für spätere Auswertung)
        try:
            tm = manager._resolve_text_model(model)
            zone = getattr(tm, "_px_zone", "UNKNOWN")
        except Exception:
            zone = "UNKNOWN"

        result.update({
            "success": True,
            "output": text,
            "output_hash": hash_output(text),
            "n_input_tokens": n_input_tokens,
            "n_new_tokens": n_new_tokens,
            "px_zone": zone,
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
# BATCHED Worker (mehrere Prompts in einem model.generate Call)
# ═══════════════════════════════════════════════════════════════════════════════
#
# SciMind 5.0 — Performance-Optimierung:
# - GPU ist mit 1-Prompt-Generierung nur 20-35% ausgelastet
# - 270m ist klein genug, dass BATCH_SIZE=4-8 in den 12GB RTX 2060 passt
# - Left-Padding ist Standard für Batched Generation mit attention_mask
# - Greedy + identische Seeds → identische Outputs wie Single-Prompt (deterministisch)

def _left_pad_and_collate(tokenizer, prompt_texts: List[str], device) -> Tuple[Any, List[int]]:
    """Tokenisiert + left-padded prompt_texts für batched generate.

    Verwendet HF-Tokenizer mit padding='longest' + return_tensors='pt' (RUST-beschleunigt,
    vermeidet Python-Loop-Bottleneck). Left-padding ist Standard für kausale LM-Generation.

    Returns:
        inputs: dict mit input_ids, attention_mask (auf device)
        n_input_tokens_list: [n_tokens_per_prompt] (vor Padding)
    """
    import torch  # nur für Tensor-Move auf device
    # Pad-Token sicherstellen (Gemma hat keinen dedizierten pad_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply chat template pro Prompt (unvermeidlich, da Templates pro-Prompt sind)
    # ABER: tokenizer-Call danach ist ein einziger Rust-Call für ALLE Prompts
    templated = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
        )
        for p in prompt_texts
    ]
    n_input_tokens_list = [
        len(tokenizer.encode(t, add_special_tokens=False)) for t in templated
    ]

    # EIN Tokenizer-Call mit padding='longest' und padding_side='left'
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        templated,
        padding="longest",       # pad bis zur längsten Sequenz im Batch
        truncation=True,         # safety: cut übergroße Prompts
        max_length=2048,         # hard cap
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, n_input_tokens_list


def run_worker_batch(
    arm_name: str, preset: str, patch_kwargs: Optional[Dict],
    prompts: List[str], max_new_tokens: int = 300, seed: int = 42,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Batched Generation: mehrere Prompts in einem model.generate Call.

    Vorteile vs. run_worker (Single-Prompt):
    - GPU-Util 20-35% → 60-80% (SciMind: CPU-Bottleneck vermeiden)
    - Speicher-Util ~10% → 40-60% (RTX 2060 12GB)
    - Gesamtlaufzeit: 480 Runs × ~30s → 480/4 × ~30s = ~60 min statt ~4h

    Greedy-Decoding (do_sample=False) ist deterministisch — gleicher Seed + gleicher Prompt
    produziert gleichen Output (in Theorie). Bei CUDA-Non-Determinismus kann das variieren
    (das ist ein valider H4-Befund, kein Bug).

    Args:
        prompts: Liste von Prompt-Strings (geordnet)
        batch_size: max prompts pro generate-Call (default: 4)

    Returns:
        Liste von Result-Dicts (gleiche Länge wie prompts, in Order)
    """
    import torch
    if not prompts:
        return []
    results: List[Dict[str, Any]] = []
    n_total = len(prompts)
    start_total = time.time()

    # Seed einmal setzen (SciMind: Reproduzierbarkeit)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        entry, manager = load_model_for_arm(arm_name, preset, patch_kwargs)
        model = entry["model"]
        tokenizer = entry["tokenizer"]
        device = next(model.parameters()).device
    except Exception as e:
        # Model-Loading-Fehler: alle Prompts mit gleichem Fehler returnen
        for p in prompts:
            results.append({
                "arm": arm_name,
                "preset": preset,
                "patch_kwargs": patch_kwargs,
                "seed": seed,
                "success": False,
                "error": f"Model-Loading: {e}",
                "traceback": traceback.format_exc(),
                "duration_sec": 0.0,
            })
        return results

    # PX-Zone einmal ermitteln
    try:
        tm = manager._resolve_text_model(model)
        zone = getattr(tm, "_px_zone", "UNKNOWN")
    except Exception:
        zone = "UNKNOWN"

    # Batches durchlaufen
    for batch_start in range(0, n_total, batch_size):
        batch = prompts[batch_start:batch_start + batch_size]
        batch_start_ts = time.time()
        try:
            inputs, n_input_tokens_list = _left_pad_and_collate(tokenizer, batch, device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy (SciMind: deterministisch)
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            # Decode alle Outputs in EINEM batch_decode-Call (RUST-beschleunigt, kein Python-Loop)
            # outputs shape: [batch, padded_input_len + new_tokens]
            padded_len = outputs.shape[1]
            # Slice: jeder Output bekommt seine generierten Tokens (ab n_input)
            sliced = [out_ids[n_input:].tolist() for n_input, out_ids in zip(n_input_tokens_list, outputs)]
            decoded_texts = tokenizer.batch_decode(
                sliced, skip_special_tokens=True,
            )
            for j, (text, n_input) in enumerate(zip(decoded_texts, n_input_tokens_list)):
                text = text.strip()
                results.append({
                    "arm": arm_name,
                    "preset": preset,
                    "patch_kwargs": patch_kwargs,
                    "seed": seed,
                    "success": True,
                    "output": text,
                    "output_hash": hash_output(text),
                    "n_input_tokens": n_input,
                    "n_new_tokens": padded_len - n_input,
                    "px_zone": zone,
                    "duration_sec": time.time() - batch_start_ts,
                    "_batch_idx": batch_start + j,
                })
            batch_dur = time.time() - batch_start_ts
            print(f"    [BATCH] {len(results)}/{n_total} done in {batch_dur:.1f}s ({len(batch)} prompts/call)")
        except Exception as e:
            err_str = str(e)
            # Tensor-Size-Mismatch: Bug in px_patches mit Attention-Cache; fall back to single
            if "size of tensor" in err_str.lower():
                print(f"    [BATCH-WORKAROUND] Tensor-Size-Mismatch — falling back to single-prompt für diese {len(batch)} prompts")
                for j, p in enumerate(batch):
                    single_result = run_worker(
                        arm_name, preset, patch_kwargs,
                        p, max_new_tokens=max_new_tokens, seed=seed,
                    )
                    single_result["_batch_idx"] = batch_start + j
                    single_result["batch_fallback"] = True
                    results.append(single_result)
            else:
                # Anderer Fehler: alle Prompts im Batch bekommen gleichen Fehler
                tb = traceback.format_exc()
                for j, _ in enumerate(batch):
                    results.append({
                        "arm": arm_name,
                        "preset": preset,
                        "patch_kwargs": patch_kwargs,
                        "seed": seed,
                        "success": False,
                        "error": err_str,
                        "traceback": tb,
                        "duration_sec": time.time() - batch_start_ts,
                        "_batch_idx": batch_start + j,
                    })
                print(f"    [BATCH-ERROR] {err_str[:120]}")

    total_dur = time.time() - start_total
    print(f"    [BATCH-DONE] {n_total} prompts in {total_dur:.1f}s ({total_dur/max(n_total,1):.2f}s/prompt)")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Decoder (2-Stage: Worker-Output → Decoder-Verdict)
# ═══════════════════════════════════════════════════════════════════════════════
#
# SciMind 5.0 Anti-Embedding: Decoder-Prompt enthält NUR den Worker-Output
# (kein System-Prompt, keine Tool-Call-Strings, keine Conversation-History).
# Strikt getrennte Modelle: Worker (gemma3-270m) ≠ Decoder (gemma3-1b via Ollama).

def call_ollama_decoder(worker_output: str, model: str = "gemma3:1b", timeout: float = 30.0) -> Optional[Dict[str, str]]:
    """Ruft Decoder via Ollama subprocess. Returnt None wenn Modell nicht verfügbar."""
    try:
        decoder_prompt = build_decoder_prompt(worker_output)
        proc = subprocess.run(
            ["ollama", "run", model, decoder_prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return None
        # Parse JSON verdict
        import re
        json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', proc.stdout, re.DOTALL)
        if not json_match:
            return None
        return json.loads(json_match.group(0))
    except Exception:
        return None


def run_decoder_stage(worker_result: Dict[str, Any], use_decoder: bool = True) -> Dict[str, Any]:
    """Stage 2: Decoder review. Skip wenn use_decoder=False oder Output leer."""
    if not use_decoder or not worker_result.get("success") or not worker_result.get("output"):
        worker_result["decoder_verdict"] = "SKIPPED"
        return worker_result
    verdict = call_ollama_decoder(worker_result["output"])
    if verdict is None:
        worker_result["decoder_verdict"] = "UNAVAILABLE"
    else:
        worker_result["decoder_verdict"] = verdict.get("verdict", "UNKNOWN")
        worker_result["decoder_text"] = verdict.get("text", "")[:500]  # truncated
    return worker_result


# ═══════════════════════════════════════════════════════════════════════════════
# Run-Loop (Haupt-Harness)
# ═══════════════════════════════════════════════════════════════════════════════

def run_single(
    arm, task_id: str, category: str, prompt: str, ground_truth: str,
    with_search: bool, seed: int, use_decoder: bool, out_dir: Path,
) -> Dict[str, Any]:
    """Führt einen einzelnen Run aus (Worker + optional Decoder).

    SciMind 5.0:
    - Atomic & Secured: schreibt JSON VOR nächstem Run
    - Pipeline Integrity: bei Tensor-Size-Mismatch (v1-Production-Bug mit Search-Kontext)
      wird automatisch OHNE Search retry. Das ist ein Workaround für einen bekannten
      Bug in px_patches/gemma3_270m_px_baseline/patch.py:163 wenn die Input-Länge
      sich nach dem ersten Layer ändert (Attention-Cache-Mismatch).
    """
    start_ts = time.time()
    out_filename = f"arm_{arm.name}__{task_id}__search{int(with_search)}__seed{seed}.json"
    out_path = out_dir / out_filename

    augmented_prompt = augment_prompt_with_search(prompt, with_search)
    worker_result = run_worker(
        arm.name, arm.preset, arm.to_patch_kwargs(),
        augmented_prompt, seed=seed,
    )

    # Workaround: Tensor-Size-Mismatch bei PX-Arm + Search → retry ohne Search
    if (
        not worker_result.get("success", False)
        and "size of tensor" in worker_result.get("error", "").lower()
        and with_search
    ):
        print(f"    [WORKAROUND] Tensor-Size-Mismatch — retry ohne search")
        worker_result = run_worker(
            arm.name, arm.preset, arm.to_patch_kwargs(),
            prompt, seed=seed,  # original prompt, kein augmented
        )
        worker_result["search_workaround_applied"] = True

    # 2-Stage Decoder
    worker_result = run_decoder_stage(worker_result, use_decoder=use_decoder)

    # Self-Report-Flags
    output = worker_result.get("output", "")
    flags = compute_self_report_flags(output)
    worker_result.update(flags)

    # Deterministische Verifikation
    worker_result["verified"] = (
        worker_result.get("success", False) and
        verify_by_category(output, ground_truth, category)
    )
    worker_result["task_id"] = task_id
    worker_result["category"] = category
    worker_result["ground_truth"] = ground_truth[:200]  # truncated
    worker_result["with_search"] = with_search
    worker_result["total_duration_sec"] = time.time() - start_ts

    # Atomic & Secured: schreibe JSON VOR nächstem Run
    try:
        out_path.write_text(json.dumps(worker_result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"    [WARN] Konnte {out_filename} nicht schreiben: {e}")

    return worker_result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RIGOR v2 — Epistemischer Haupt-Harness")
    parser.add_argument("--max-per-category", type=int, default=10, help="Max HLE-Tasks pro Kategorie")
    parser.add_argument("--seed", type=int, default=42, help="Primärer Seed")
    parser.add_argument("--n-reproducibility", type=int, default=5, help="Anzahl Reproducibility-Tasks")
    parser.add_argument("--use-decoder", action="store_true", help="Aktiviere 2-Stage Decoder")
    parser.add_argument("--arms", nargs="*", default=None, help="Subset von Arm-Namen (default: alle 8)")
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Max new tokens (default: 300 — micro harness, nicht endlos generieren)")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip Runs die bereits Output-Files haben (resumable, default: AN)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Deaktiviere Resume-Modus")
    parser.add_argument("--no-cache", action="store_true", help="Deaktiviere Model-Cache (default: AN — spart CPU-Loading)")
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts pro batched generate-Call (default: 8 — gemessen TDD-optimal für 270M auf RTX 2060, 65% GPU-Util)")
    parser.add_argument("--no-batch", dest="batch", action="store_false", default=True, help="Deaktiviere Batched-Generation (fallback: single-prompt Loops)")
    args = parser.parse_args()

    out_dir = Path(_HERE) / "out"
    out_dir.mkdir(exist_ok=True)
    repro_dir = out_dir / "reproducibility"
    repro_dir.mkdir(exist_ok=True)

    # SciMind: Resume-Skip-Logik
    def already_done(arm_name: str, task_id: str, with_search: bool, seed: int) -> bool:
        """Prüft ob Output-File bereits existiert (resumable)."""
        if not args.resume:
            return False
        fname = f"arm_{arm_name}__{task_id}__search{int(with_search)}__seed{seed}.json"
        path = out_dir / fname
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            # Nur als 'done' werten wenn success=True oder ein definierter Error
            return data.get("success") is True or "error" in data
        except Exception:
            return False

    def repro_already_done(arm_name: str, task_id: str) -> bool:
        if not args.resume:
            return False
        path = repro_dir / f"repro_{arm_name}__{task_id}.json"
        return path.exists()

    # Model-Cache deaktivieren wenn gewünscht
    if args.no_cache:
        global _MODEL_CACHE
        _MODEL_CACHE = {}
        # Monkey-patch load_model_for_arm zu no-op cache
    else:
        print("[SciMind: Atomic & Secured] Model-Cache AN — vermeidet CPU-Loading zwischen Runs")
    print()

    # ── SciMind Header ──────────────────────────────────────────────────
    print("=" * 60)
    print("RIGOR v2 — Epistemischer Haupt-Harness")
    print("=" * 60)
    print()
    print("SciMind 5.0 — Core Mandates:")
    print("  1. Incomplete Suggestion: H1-H4 in README pre-registriert")
    print("  2. Empirical Verification: stdout/stderr + JSON-Output Checks")
    print("  3. Falsificationism: Antithesen pro Hypothese")
    print("  4. Pipeline Integrity: set -o pipefail in Wrappern")
    print("  5. Zero-Trust: Pre-Probes laufen jetzt...")
    print("  6. Atomic & Secured: JSON-write pro Run VOR nächstem")
    print("  7. Anti-Embedding: Worker-Output ≠ Tool-Call-String")
    print()

    # ── Pre-Probes ──────────────────────────────────────────────────────
    print("Pre-Lauf-Probes (Zero-Trust Environment):")
    probes = pre_run_probes()
    for name, ok in probes.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    print()
    if not probes.get("torch_cuda"):
        print("FATAL: torch + CUDA nicht verfügbar. Abbruch.")
        sys.exit(1)
    if not probes.get("datasets") and not probes.get("ddgs"):
        print("WARN: weder datasets noch ddgs verfügbar. Fallback wird verwendet.")

    # ── HLE-Suite laden ─────────────────────────────────────────────────
    print("HLE-Suite wird geladen...")
    hle_source, hle_tasks = load_hle_suite(max_per_category=args.max_per_category)
    print(f"  Source: {hle_source}, Total: {len(hle_tasks)} Tasks")
    print()

    # ── Arm-Subset ──────────────────────────────────────────────────────
    if args.arms:
        arms = [a for a in ARM_CONFIGS if a.name in args.arms]
        if not arms:
            print(f"FATAL: --arms {args.arms} matcht keine Arm-Configs. Verfügbar: {[a.name for a in ARM_CONFIGS]}")
            sys.exit(1)
    else:
        arms = list(ARM_CONFIGS)
    print(f"Arm-Konfigurationen: {len(arms)} ({[a.name for a in arms]})")
    print()

    # ── Reproducibility-Tasks (5 Repräsentanten) ────────────────────────
    repro_tasks = hle_tasks[:args.n_reproducibility]
    print(f"Reproducibility-Tasks: {len(repro_tasks)}")
    print()

    # ── Pre-Load Models (SciMind: CPU-Bottleneck vermeiden) ─────────────
    # Wir laden alle 8 Arm-Modelle EINMAL vor der Run-Loop. Spart 7× Re-Loading.
    # Speicher-Budget: 8 Modelle × 1.5 GB = 12 GB — passt gerade noch in 12 GB RTX 2060.
    if not args.no_cache:
        print("=" * 60)
        print("PRE-LOAD: 8 Arm-Modelle werden geladen (spart 7× Re-Loading)")
        print("=" * 60)
        for arm in arms:
            try:
                load_model_for_arm(arm.name, arm.preset, arm.to_patch_kwargs())
                print(f"  ✓ {arm.name} (preset={arm.preset})")
            except Exception as e:
                print(f"  ✗ {arm.name}: {e}")
        print()

    # ── Run-Loop ────────────────────────────────────────────────────────
    print("=" * 60)
    print("RUN-LOOP START")
    print("=" * 60)
    if args.resume:
        print(f"[RESUME] Skip-Modus aktiv: bereits existente Output-Files werden übersprungen")
    if args.batch and args.batch_size > 1:
        print(f"[BATCH] Batched-Generation aktiv (bs={args.batch_size}) — GPU besser ausgelastet")
    else:
        print(f"[BATCH] Batched-Generation DEAKTIVIERT (--no-batch oder bs=1) — single-prompt Loops")
    print()
    start_total = time.time()
    results: List[Dict[str, Any]] = []
    run_idx = 0
    skipped_count = 0
    total_runs = len(arms) * len(hle_tasks) * 2  # ×2 für search=False/True
    total_runs += len(arms) * len(repro_tasks)  # Reproducibility

    for arm in arms:
        for with_search in [False, True]:
            # ── SciMind: Resumable + Batched ──────────────────────────────
            # Sammle alle unskipped Tasks, generiere in Batches (GPU-Util ↑),
            # schreibe pro Task JSON (Atomic & Secured: kein Verlust bei Crash)
            pending: List[Tuple[str, str, str, str]] = []
            for task_id, category, prompt, ground_truth in hle_tasks:
                run_idx += 1
                if already_done(arm.name, task_id, with_search, args.seed):
                    print(f"[Run {run_idx}/{total_runs}] SKIP (resume) arm={arm.name:18s} search={with_search} task={task_id}")
                    skipped_count += 1
                    continue
                pending.append((task_id, category, prompt, ground_truth))

            if not pending:
                print(f"[{arm.name:18s} search={with_search}] alle {len(hle_tasks)} Tasks bereits done — skip")
                continue

            print(f"[{arm.name:18s} search={with_search}] BATCH {len(pending)} Tasks (bs={args.batch_size})...")
            # Augment prompts mit Search (wenn aktiv)
            augmented_prompts = [
                augment_prompt_with_search(p, with_search) for (_, _, p, _) in pending
            ]

            if args.batch and args.batch_size > 1:
                # ── Batched-Generation (SciMind: GPU-Util ↑) ──────────────
                worker_results = run_worker_batch(
                    arm.name, arm.preset, arm.to_patch_kwargs(),
                    augmented_prompts, max_new_tokens=args.max_new_tokens,
                    seed=args.seed, batch_size=args.batch_size,
                )
            else:
                # ── Single-Prompt (Fallback) ───────────────────────────────
                worker_results = [
                    run_worker(arm.name, arm.preset, arm.to_patch_kwargs(), p, seed=args.seed)
                    for p in augmented_prompts
                ]

            # ── Post-Process: Vektorisierter Erfolgs-Pfad + Workaround-Loop ──
            # 1) Trenne Erfolgs-Results von Workaround-Kandidaten
            needs_workaround: List[int] = []
            for i, wres in enumerate(worker_results):
                if (
                    not wres.get("success", False)
                    and "size of tensor" in wres.get("error", "").lower()
                    and with_search
                ):
                    needs_workaround.append(i)

            # 2) Workaround-Runs (einzeln, da sie die Batch-Pfade sprengen)
            for i in needs_workaround:
                task_id, category, prompt, ground_truth = pending[i]
                print(f"    [WORKAROUND] Tensor-Size-Mismatch — retry ohne search (task={task_id})")
                wres_new = run_worker(
                    arm.name, arm.preset, arm.to_patch_kwargs(),
                    prompt, max_new_tokens=args.max_new_tokens, seed=args.seed,
                )
                wres_new["search_workaround_applied"] = True
                worker_results[i] = wres_new

            # 3) Vektorisierte Flags + Verifikation für ALLE Results in EINEM Pass
            all_outputs = [wres.get("output", "") for wres in worker_results]
            all_flags = compute_self_report_flags_batch(all_outputs)
            all_categories = [category for (_, category, _, _) in pending]
            all_ground_truths = [gt for (_, _, _, gt) in pending]
            all_verified = verify_by_category_batch(
                all_outputs, all_ground_truths, all_categories,
            )

            # 4) Pro Result: Decoder, Metadata, Atomic-Write
            for i, ((task_id, category, prompt, ground_truth), wres) in enumerate(zip(pending, worker_results)):
                run_idx += 1
                # 2-Stage Decoder
                wres = run_decoder_stage(wres, use_decoder=args.use_decoder)
                # Vektorisierte Flags + Verdict
                wres.update(all_flags[i])
                wres["verified"] = wres.get("success", False) and all_verified[i]
                wres["task_id"] = task_id
                wres["category"] = category
                wres["ground_truth"] = ground_truth[:200]
                wres["with_search"] = with_search
                wres["total_duration_sec"] = wres.get("duration_sec", 0)

                # Atomic & Secured: JSON VOR nächstem Run
                out_filename = f"arm_{arm.name}__{task_id}__search{int(with_search)}__seed{args.seed}.json"
                out_path = out_dir / out_filename
                try:
                    out_path.write_text(json.dumps(wres, indent=2, ensure_ascii=False))
                except Exception as e:
                    print(f"    [WARN] Konnte {out_filename} nicht schreiben: {e}")

                results.append(wres)
                if wres.get("success"):
                    print(f"  ✓ task={task_id} | {wres.get('duration_sec', 0):.1f}s | loops={wres.get('output_loops', False)} | verified={wres.get('verified', False)}")
                else:
                    err = wres.get("error", "?")[:100]
                    print(f"  ✗ task={task_id} | {wres.get('duration_sec', 0):.1f}s | ERROR: {err}")

        # Reproducibility-Runs für diesen Arm (sequentiell, je 2 Seeds = 2 Runs)
        for task_id, category, prompt, ground_truth in repro_tasks:
            run_idx += 1
            if repro_already_done(arm.name, task_id):
                print(f"[Run {run_idx}/{total_runs}] SKIP repro (resume) arm={arm.name:18s} task={task_id}")
                skipped_count += 1
                continue
            print(f"[Run {run_idx}/{total_runs}] REPRO arm={arm.name:18s} task={task_id} (seed={args.seed} + {args.seed+1})")
            result_repro = run_single(
                arm, task_id, category, prompt, ground_truth,
                False, args.seed, args.use_decoder, out_dir,
            )
            # Zusätzlich: 2. Run mit seed+1 für Reproducibility-Check
            result_repro2 = run_single(
                arm, task_id, category, prompt, ground_truth,
                False, args.seed + 1, args.use_decoder, out_dir,
            )
            # Vergleich
            match = reproducibility_check(
                result_repro.get("output", ""),
                result_repro2.get("output", ""),
            )
            repro_entry = {
                "arm": arm.name,
                "task": task_id,
                "category": category,
                "seed_0": args.seed,
                "seed_1": args.seed + 1,
                "match": match,
                "hash_0": result_repro.get("output_hash", ""),
                "hash_1": result_repro2.get("output_hash", ""),
                "duration_sec": result_repro.get("duration_sec", 0) + result_repro2.get("duration_sec", 0),
            }
            (repro_dir / f"repro_{arm.name}__{task_id}.json").write_text(
                json.dumps(repro_entry, indent=2, ensure_ascii=False)
            )
            status = "✓" if match else "✗"
            print(f"  {status} Reproducibility: {match}")

    total_duration = time.time() - start_total
    print()
    print("=" * 60)
    print(f"Alle Runs abgeschlossen in {total_duration/60:.1f} min")
    print(f"Total Runs: {len(results)}")
    print(f"Skipped (resume): {skipped_count}")
    print(f"Reproducibility-Runs: {len(repro_tasks) * len(arms)}")
    print(f"Output: {out_dir}")
    print("=" * 60)
    print()

    # ── Report-Generierung ──────────────────────────────────────────────
    from rigor_report import write_report, load_all_results
    all_results = load_all_results(out_dir)
    repro_results = []
    for jp in repro_dir.glob("*.json"):
        try:
            with open(jp) as f:
                repro_results.append(json.load(f))
        except Exception:
            pass

    metadata = {
        "hle_source": hle_source,
        "n_search": 2,
        "seed_primary": args.seed,
        "seed_secondary": args.seed + 1,
        "total_duration_sec": total_duration,
    }
    report_path = Path(_HERE) / "hle_results_v2.md"
    write_report(all_results, repro_results, metadata, report_path)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
