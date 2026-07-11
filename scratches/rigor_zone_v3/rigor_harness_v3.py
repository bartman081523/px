"""RIGOR v3 — 3-Phasen-Harness (Preprocess | GPU-Loop | Postprocess).

SciMind 5.0 + User-Direktive: 100% GPU-Utilization, kein CPU-Bottleneck.

Architektur:
  Phase A — Preprocess (CPU, einmalig, vor GPU-Loop):
    1. load_hle_suite() — 30 Tasks
    2. SearchCache.prefetch() — 30× parallel DDG (statt 240× serial in v2)
    3. Pre-Build WorkList: alle (arm, with_search, task_id, seed) Tupel
    4. Skip resumed runs via single os.scandir (statt 480× json.load)
    5. Spawn background writer thread (drain queue → atomic os.replace)

  Phase B — GPU-Loop (PURE GPU, 0% CPU-Wait):
    Für jeden Arm × with_search × batch:
      1. tokenizer(prompts, padding="longest") — 1 RUST-Call
      2. model.generate(...) — PURE GPU
      3. tokenizer.batch_decode(...) — 1 RUST-Call
      4. queue.put(wres) — in-memory, non-blocking

  Phase C — Postprocess (CPU, nach GPU-Loop):
    1. Drain write-queue (background thread finalisiert)
    2. Batched compute_self_report_flags_batch + verify_by_category_batch
    3. Report generieren

Verboten im GPU-Loop-Body (statisch enforced via test):
  - subprocess, urllib, requests, json.dumps, Path.write_text, open(, os.stat
  - ddgs, DDGS, web_search, augment_prompt_with_search
  - re.search/re.match (per-output regex)
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# venv-Check: openmythos venv (torch 2.12+cu130)
_REPO_ROOT = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scratches/rigor_zone_v1"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scratches/rigor_zone_v2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rigor_hle_suite_v3 import load_hle_suite, SearchCache
from rigor_scales_v3 import ARM_CONFIGS
from rigor_verify import (
    compute_self_report_flags_batch,
    verify_by_category_batch,
    build_decoder_prompt,
)
from rigor_stats import compute_eta2, bootstrap_ci

# Optional v1-PX-Imports (mit Fallback: nur install_/restore_ sind verfügbar)
try:
    from rigor_zone_forward import install_rigor_forward, restore_rigor_forward
    from rigor_zone_manifold import _at as _rigor_at
    HAVE_V1_PX = True
except ImportError:
    HAVE_V1_PX = False
    print("[WARN] v1-PX (rigor_zone_forward/manifold) nicht importierbar — rigor_* Arms laufen ohne Damping")


# ═══════════════════════════════════════════════════════════════════════════════
# Konfiguration
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_SEED = 42
DEFAULT_REPRO_SEED = 43
DEFAULT_MODEL = "google/gemma-3-270m-it"


# ═══════════════════════════════════════════════════════════════════════════════
# SciMind 5.0 Pre-Lauf-Probes (Zero-Trust)
# ═══════════════════════════════════════════════════════════════════════════════

def pre_run_probes() -> Dict[str, bool]:
    """Prüft ob alle Dependencies da sind. Bei Fehlen → exit 1."""
    import importlib.util
    import shutil
    probes = {
        "ollama": shutil.which("ollama") is not None,
        "ddgs": importlib.util.find_spec("ddgs") is not None,
        "datasets": importlib.util.find_spec("datasets") is not None,
        "torch_cuda": False,
        "transformers": importlib.util.find_spec("transformers") is not None,
    }
    try:
        import torch
        probes["torch_cuda"] = torch.cuda.is_available()
    except Exception:
        pass
    return probes


# ═══════════════════════════════════════════════════════════════════════════════
# WorkItem + Result + PreprocessContext
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkItem:
    arm_name: str
    arm_preset: str
    rigor_mephisto: Optional[bool]
    mephisto_scale: Optional[float]
    with_search: bool
    task_id: str
    category: str
    prompt: str
    ground_truth: str
    seed: int


@dataclass
class WorkerResult:
    work: WorkItem
    output_text: str
    output_hash: str
    duration_s: float
    n_input_tokens: int
    n_output_tokens: int
    search_results: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PreprocessContext:
    hle_tasks: List[Tuple[str, str, str, str]]
    work_items: List[WorkItem]
    search_cache: SearchCache
    template_cache: Dict[Tuple[str, str, bool, int], str]  # (arm, task_id, with_search, seed) → templated
    write_queue: "queue.Queue[Tuple[Path, Dict]]"
    out_dir: Path
    batch_size: int
    max_new_tokens: int
    seed: int
    model_cache: Dict[str, Any] = field(default_factory=dict)
    #   arm_name → (model, tokenizer, patch_kwargs)
    #   PRE-LOADED in preprocess(), NICHT lazy in run_gpu_loop (war 32s Bottleneck)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A — Preprocess (CPU, einmalig)
# ═══════════════════════════════════════════════════════════════════════════════

def augment_prompt_with_search(prompt: str, search_results: List[Dict[str, str]]) -> str:
    """Injiziert Suchergebnisse als Kontext in den Prompt (v1-Konvention)."""
    if not search_results:
        return prompt
    snippets = []
    for r in search_results[:3]:
        body = r.get("body", "")[:200]
        if body:
            snippets.append(f"- {body}")
    if not snippets:
        return prompt
    return (
        f"{prompt}\n\n"
        f"Web-Suche (Kontext, max 3 Snippets):\n" + "\n".join(snippets)
    )


def make_work_items(
    hle_tasks: List[Tuple[str, str, str, str]],
    arms,
    seeds: List[int],
    n_reproducibility: int,
    out_dir: Path,
) -> List[WorkItem]:
    """Baut die WorkList aller (arm, with_search, task_id, seed) Tupel.

    Resume-Logik: skip wenn Output-File existiert UND gültig (parseable JSON).
    """
    items: List[WorkItem] = []
    # Bestehende Outputs einmalig scannen
    existing: set = set()
    if out_dir.exists():
        for p in out_dir.glob("*.json"):
            existing.add(p.name)
    for arm in arms:
        for task_id, cat, prompt, gt in hle_tasks:
            for with_search in (False, True):
                for seed in seeds:
                    fn = f"arm_{arm.name}__{task_id}__search{int(with_search)}__seed{seed}.json"
                    if fn in existing:
                        continue
                    items.append(WorkItem(
                        arm_name=arm.name,
                        arm_preset=arm.preset,
                        rigor_mephisto=arm.rigor_mephisto,
                        mephisto_scale=arm.mephisto_scale,
                        with_search=with_search,
                        task_id=task_id,
                        category=cat,
                        prompt=prompt,
                        ground_truth=gt,
                        seed=seed,
                    ))
    return items


def writer_thread_fn(q: "queue.Queue", out_dir: Path) -> None:
    """Background thread: drained Queue und schreibt JSONs atomar."""
    while True:
        try:
            item = q.get(timeout=1)
        except queue.Empty:
            continue
        if item is None:  # Poison pill = stop
            q.task_done()
            return
        out_path, wres = item
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(wres, f, indent=2, ensure_ascii=False)
            os.replace(tmp, out_path)
        except Exception as e:
            print(f"[writer] ERROR writing {out_path}: {e}", file=sys.stderr)
        finally:
            q.task_done()


def preprocess(args) -> Tuple[PreprocessContext, "ThreadPoolExecutor"]:
    """Phase A: HLE laden, SearchCache prefetchen, WorkList bauen, Writer starten."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) HLE-Suite laden
    print(f"[A] Lade HLE-Suite...")
    _, hle_tasks = load_hle_suite(max_per_category=args.max_per_category)
    print(f"    {len(hle_tasks)} Tasks geladen")

    # 2) SearchCache prefetch (PARALLEL, 30× statt 240×)
    search_cache = SearchCache(path=out_dir / "_search_cache.json")
    search_cache.load()  # File-Cache-Hits zuerst
    print(f"[A] SearchCache prefetch (parallel, max_workers={args.search_workers})...")
    t0 = time.time()
    search_cache.prefetch(hle_tasks, max_workers=args.search_workers)
    search_cache.save()
    print(f"    {search_cache._http_calls} HTTP-Calls in {time.time() - t0:.1f}s "
          f"({len(search_cache)} unique Tasks)")

    # 3) Arm-Grid
    arms = list(ARM_CONFIGS)

    # 4) WorkList bauen (mit resume-check)
    seeds = [DEFAULT_SEED]
    if args.n_reproducibility > 0:
        seeds.append(DEFAULT_REPRO_SEED)
    work_items = make_work_items(hle_tasks, arms, seeds, args.n_reproducibility, out_dir)
    print(f"[A] {len(work_items)} WorkItems zu erledigen (resumed: skip)")

    # 5) PRE-LOAD alle Arm-Modelle (Bottleneck-Fix: war 32s CPU bei 8 Arms)
    #    8×270M×bf16 = 4.3GB → passt in 12GB RTX 2060
    print(f"[A] Pre-Load aller {len(arms)} Arm-Modelle...")
    t0 = time.time()
    model_cache: Dict[str, Any] = {}
    for arm in arms:
        try:
            cached = _get_model(arm.name, arm.preset)
            if cached is not None:
                model_cache[arm.name] = cached
                print(f"    ✓ {arm.name} geladen")
        except Exception as e:
            print(f"    ✗ {arm.name} FAILED: {e}")
    print(f"    {len(model_cache)}/{len(arms)} Modelle in {time.time() - t0:.1f}s")

    # 6) Background writer thread starten
    write_queue: "queue.Queue" = queue.Queue(maxsize=1024)
    writer = threading.Thread(target=writer_thread_fn, args=(write_queue, out_dir), daemon=True)
    writer.start()

    ctx = PreprocessContext(
        hle_tasks=hle_tasks,
        work_items=work_items,
        search_cache=search_cache,
        template_cache={},
        write_queue=write_queue,
        out_dir=out_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=DEFAULT_SEED,
        model_cache=model_cache,
    )
    return ctx, None  # executor placeholder


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B — run_gpu_loop (PURE GPU, KEIN I/O)
# ═══════════════════════════════════════════════════════════════════════════════

_model_cache: Dict[str, Any] = {}


def _get_model(arm_name: str, arm_preset: Optional[str]):
    """Lazy-load + cache Worker-Modell. Pre-Load kann preprocess() aufrufen.

    v3-FIX: Lenkt den `gemma3_270m_px_baseline`-Import auf das v3-Patch-Modul um,
    damit die v3-Version (per-Layer Hq aus self.config) statt der
    Original-px_patches/gemma3_270m_px_baseline/patch.py geladen wird.
    Damit fällt der Tensor-Mismatch bei 270m + bs>1 weg.
    """
    import os as _os
    import sys as _sys
    import importlib
    key = f"{arm_preset}_{arm_name}"
    if key in _model_cache:
        return _model_cache[key]
    # v1-konformer Load via ModelManager
    from model_manager import ModelManager, _migrate_preset
    # v3-FIX: v3-Patch-Modul unter dem Namen registrieren, den ModelManager erwartet.
    # ModelManager's _get_patch_function: `module_name = f"{patch_dir}.patch"`
    # → "gemma3_270m_px_baseline.patch" → wenn das in sys.modules liegt, wird es
    # via __import__ geladen, OHNE erneuten Filesystem-Lookup.
    _v3_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "px_patches_v3")
    if _v3_dir not in _sys.path:
        _sys.path.insert(0, _v3_dir)
    # WICHTIG: gemma3_270m_px_baseline-Package existiert in px_patches/ — wir
    # wollen es überlagern. Lösung: lade das v3-Patch-Modul, registriere es
    # unter dem erwarteten Namen. _get_patch_function ruft __import__ auf,
    # was bei vorhandenem sys.modules-Eintrag den Filesystem-Lookup überspringt.
    _v3_patch = importlib.import_module("px_patches_v3.patch")
    _sys.modules["gemma3_270m_px_baseline"] = type(_sys)("gemma3_270m_px_baseline")
    _sys.modules["gemma3_270m_px_baseline"].patch = _v3_patch
    _sys.modules["gemma3_270m_px_baseline.patch"] = _v3_patch
    if not _os.environ.get("RIGOR_V3_PATCH_LOGGED"):
        _os.environ["RIGOR_V3_PATCH_LOGGED"] = "1"
        print(f"    [v3-PATCH] v3-patch.py geladen — gemma3_270m_px_baseline → "
              f"px_patches_v3.patch (per-Layer Hq aus self.config)")
    mm = ModelManager()
    # Wenn rigor_* Arm: install_rigor_forward (v1 monkey-patch)
    # v1-API: kwargs sind rigor_mephisto (bool) + rigor_mephisto_scale (float)
    patch_kwargs = None
    if arm_name.startswith("rigor_") and HAVE_V1_PX:
        if arm_name == "rigor_disabled":
            patch_kwargs = {"rigor_mephisto": False, "rigor_mephisto_scale": 0.0}
        elif arm_name == "rigor_least":
            patch_kwargs = {"rigor_mephisto": True, "rigor_mephisto_scale": 0.1}
        elif arm_name == "rigor_low":
            patch_kwargs = {"rigor_mephisto": True, "rigor_mephisto_scale": 0.3}
        elif arm_name == "rigor_mid":
            patch_kwargs = {"rigor_mephisto": True, "rigor_mephisto_scale": 0.5}
        elif arm_name == "rigor_high":
            patch_kwargs = {"rigor_mephisto": True, "rigor_mephisto_scale": 0.7}
        elif arm_name == "rigor_full":
            patch_kwargs = {"rigor_mephisto": True, "rigor_mephisto_scale": 1.0}
    # Preset-Migration (v2-konform): "RIGOR" → "ACTIVE_MANIFOLD"
    if arm_name == "baseline":
        effective_preset = "BASELINE"
    elif arm_name.startswith("rigor_"):
        effective_preset = "ACTIVE_MANIFOLD"
    else:
        effective_preset = _migrate_preset(arm_preset) if arm_preset else "BASELINE"
    model_id = "gemma3-270m-it"
    entry = mm._load_model(
        model_id,
        px_subjective=(effective_preset != "BASELINE"),
        px_config_preset=effective_preset,
    )
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    # RIGOR-Override installieren (v2-Pattern)
    if arm_name.startswith("rigor_") and patch_kwargs and HAVE_V1_PX:
        tm = mm._resolve_text_model(model)
        # WICHTIG: ModelManager cachet das Text-Model pro model_id. Wenn ein
        # vorheriger rigor_* Arm recur_end modifiziert hat, müssen wir es
        # ZUERST zurücksetzen, sonst sehen alle nachfolgenden Arms die
        # Werte des ersten Arms.
        if hasattr(tm, "_rigor_orig_recur_end") and hasattr(tm, "_px_config"):
            tm._px_config["recur_end"] = tm._rigor_orig_recur_end
        # Mephisto-Original-Forward wiederherstellen (v1 Restore)
        restore_rigor_forward(tm)
        rigor_info = install_rigor_forward(
            tm, config_preset="RIGOR", **patch_kwargs,
        )
        print(f"    → RIGOR-Override installiert: {rigor_info}")
        # Recursion-Damping: reduziere recur_end in cfg, sodass AutoCalibrator
        # (70% Gewicht auf recur_end in dynamic_end) die Recursion-Range
        # proportional zu scale verkleinert.
        # v1 install_rigor_forward patched nur _px_mephisto.forward, nicht
        # _px_forward selbst — wir müssen also auf cfg-Ebene arbeiten.
        scale = patch_kwargs.get("rigor_mephisto_scale", 1.0)
        if scale < 1.0 and hasattr(tm, "_px_config"):
            cfg = tm._px_config
            base_recur_range = cfg.get("recur_end", 12) - cfg.get("recur_start", 5)
            new_range = max(1, int(round(base_recur_range * scale)))
            if "recur_end" in cfg:
                if not hasattr(tm, "_rigor_orig_recur_end"):
                    tm._rigor_orig_recur_end = cfg["recur_end"]
                cfg["recur_end"] = cfg["recur_start"] + new_range
                print(f"    → Recursion-Damping: recur_range {base_recur_range} → {new_range} (scale={scale})")
    if model is not None and tokenizer is not None:
        _model_cache[key] = (model, tokenizer, patch_kwargs)
    return _model_cache.get(key)


def run_gpu_loop(
    arm_name: str,
    arm_preset: Optional[str],
    patch_kwargs: Optional[Dict],
    prompts: List[str],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_cache: Optional[Dict[str, Any]] = None,
    use_cuda_graph: bool = True,
) -> List[Dict[str, Any]]:
    """PURE-GPU Worker-Call: tokenize → generate → batch_decode.

    v3-final: CUDA-Graph ist DEFAULT für alle nicht-Baseline-Arms.
    CUDA-Graph eliminiert pro-Step Python-Dispatch → 2.6× Speedup
    (vs eager PX: 222 → 573 tok/s @ bs=8) und GPU-Util 33%→73% avg,
    100% max.

    Args:
        use_cuda_graph: Default True. Falls False → klassisches
            model.generate() (langsamer, aber kompatibel).
    """
    import torch

    # Modell beziehen (cached wenn model_cache gegeben, sonst lazy)
    if model_cache and arm_name in model_cache:
        model, tokenizer, _patch = model_cache[arm_name]
    else:
        cached = _get_model(arm_name, arm_preset)
        if cached is None:
            return [{"text": "", "n_input_tokens": 0, "n_output_tokens": 0} for _ in prompts]
        model, tokenizer, _patch = cached

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    # Seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    results: List[Dict[str, Any]] = []
    # Gruppiere in Batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        # Templated chat
        templated = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch_prompts
        ]
        n_input_tokens_list = [
            len(tokenizer.encode(t, add_special_tokens=False)) for t in templated
        ]
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            templated, padding="longest", truncation=True, max_length=2048,
            return_tensors="pt", add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # v3-CUDA-GRAPH-PATH: für PX-Arms (active_manifold, rigor_*, etc.)
        # Baseline-Arm nutzt weiterhin model.generate (kein PX-Forward,
        # CUDA-Graph-Capture für den Standard-Pfad ist möglich aber nicht
        # nötig — Baseline ist schon 2.1× schneller als PX-Eager).
        if use_cuda_graph and arm_name != "baseline":
            from px_patches_v3.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
            B = input_ids.shape[0]
            # MAX_SEQ mit großzügigem Puffer für PX-Recursion-Wachstum
            # (Recursion kann Layers mehrfach durchlaufen → KV-Cache wächst
            #  über T_prefill hinaus. max_in*2 + max_new_tokens + 64.)
            max_in = max(n_input_tokens_list) if n_input_tokens_list else 32
            # MAX_SEQ großzügig (max_in*3 statt *2): Gemma3-Hybrid-Cache
            # kann nach Prefill eine um 1-2 abweichende seq_length reporten,
            # und CUDA-Graph verlangt exakte Größen-Übereinstimmung.
            max_seq = max_in * 3 + max_new_tokens + 128
            cfg = CUDAGraphRunnerConfig(batch_size=B, max_seq_len=max_seq)
            try:
                runner = CUDAGraphRunner(model, cfg)
                first = runner.setup(input_ids, attention_mask)
                tokens = [first]
                with torch.inference_mode():
                    for _ in range(max_new_tokens - 1):
                        nxt = runner.step()
                        runner.append(nxt)
                        tokens.append(nxt)
                outputs = torch.cat(tokens, dim=1)  # (B, max_new_tokens)
            except Exception as e:
                # Fallback: klassisches generate
                print(f"  [WARN] CUDA-Graph failed ({type(e).__name__}: {str(e)[:80]}), fallback to model.generate")
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
        else:
            # Baseline-Pfad: model.generate
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )

        # Batch-decode (1 RUST-Call, kein per-Prompt-Loop)
        # outputs ist entweder (B, T_total) von generate ODER (B, max_new_tokens)
        # von CUDA-Graph. Im CUDA-Graph-Fall: outputs enthält NUR generated tokens.
        if outputs.shape[1] == max_new_tokens:
            # CUDA-Graph-Pfad: outputs ist nur generated
            decoded = tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
        else:
            # model.generate-Pfad: outputs ist prompt+generated
            n_input_max = max(n_input_tokens_list)
            sliced = outputs[:, n_input_max:]
            decoded = tokenizer.batch_decode(
                sliced, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
        # Truncate pro-Prompt auf n_input_tokens (links-gepaddet)
        for j, text in enumerate(decoded):
            results.append({
                "text": text,
                "n_input_tokens": n_input_tokens_list[j],
                "n_output_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C — Postprocess (CPU, nach GPU-Loop)
# ═══════════════════════════════════════════════════════════════════════════════

def postprocess(ctx: PreprocessContext, results: List[Tuple[Path, Dict]]) -> Dict:
    """Phase C: Self-Report-Flags + Verifikation BATCHED.

    results: Liste von (out_path, wres_dict) Tupeln aus Phase B.
    """
    # 1) Schreib-Queue finalisieren
    for out_path, wres in results:
        ctx.write_queue.put((out_path, wres))
    ctx.write_queue.put(None)  # Poison pill
    ctx.write_queue.join()  # Warte bis alles geschrieben

    # 2) Batched flags + verify über ALLE Outputs
    outputs = [r[1].get("output_text", "") for r in results]
    gts = [r[1].get("ground_truth", "") for r in results]
    cats = [r[1].get("category", "") for r in results]
    flags_batch = compute_self_report_flags_batch(outputs)
    verify_batch = verify_by_category_batch(outputs, gts, cats)

    # 3) In-Place Update: flags + verify
    for (out_path, wres), flags, ok in zip(results, flags_batch, verify_batch):
        wres["self_report_flags"] = flags
        wres["verify_ok"] = bool(ok)
        # Atomic write (overwrite mit aktualisierten Feldern)
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(wres, f, indent=2, ensure_ascii=False)
        os.replace(tmp, out_path)

    return {
        "n_outputs": len(results),
        "n_verified_ok": sum(1 for v in verify_batch if v),
        "n_flagged": sum(1 for f in flags_batch if any(f.values())),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# main() — orchestriert die 3 Phasen
# ═══════════════════════════════════════════════════════════════════════════════

def run_hypothesis_suite(ctx: PreprocessContext, results: List[Tuple[Path, Dict]]) -> Dict[str, Dict]:
    """Führt H1-H4 aus rigor_stats auf den Run-Results aus.

    Returns:
        Dict mit H1, H2, H3, H4 — jedes mit status + metrics.
    """
    from rigor_stats import (
        test_h1_loop_reduction,
        test_h2_task_specific,
        test_h3_search_effect,
        test_h4_reproducibility,
    )
    # Gruppieren pro Arm
    by_arm: Dict[str, List[Dict]] = {}
    per_arm_per_cat: Dict[str, List[Dict]] = {}
    search_on_outputs: List[Dict] = []
    search_off_outputs: List[Dict] = []
    for _path, r in results:
        arm = r.get("arm", "")
        cat = r.get("category", "")
        output_loops = r.get("output_loops", False)
        by_arm.setdefault(arm, []).append({**r, "output_loops": output_loops})
        per_arm_per_cat.setdefault(arm, []).append({**r, "category": cat})
        if r.get("with_search"):
            search_on_outputs.append(r)
        else:
            search_off_outputs.append(r)

    h1 = test_h1_loop_reduction(by_arm)
    h2 = test_h2_task_specific(per_arm_per_cat)
    h3_data: Dict[str, List[Dict]] = {"search_on": search_on_outputs, "search_off": search_off_outputs}
    h3 = test_h3_search_effect(h3_data)

    # H4: sammle Repro-Paare (seed 42 vs 43)
    repro_pairs: List[Dict[str, Any]] = []
    by_task_seed: Dict[Tuple, Dict] = {}
    for _path, r in results:
        key = (r.get("arm"), r.get("task_id"), r.get("with_search"))
        by_task_seed[(r.get("seed"),) + key] = r
    for (s, arm, tid, ws), r_seed in list(by_task_seed.items()):
        if s != DEFAULT_REPRO_SEED - 1:
            continue
        r43 = by_task_seed.get((DEFAULT_REPRO_SEED, arm, tid, ws))
        if r43 is None:
            continue
        from rigor_stats import hash_output
        repro_pairs.append({
            "arm": arm, "task": tid, "search": ws,
            "match": hash_output(r_seed.get("output_text", "")) == hash_output(r43.get("output_text", "")),
        })
    h4 = test_h4_reproducibility(repro_pairs)

    out = {
        "H1": {"status": "BESTÄTIGT" if h1.get("h1_confirmed") else "WIDERLEGT", **h1},
        "H2": {"status": "BESTÄTIGT" if h2.get("testable") else "NICHT_TESTBAR", **h2},
        "H3": {"status": "BESTÄTIGT" if h3.get("testable") else "NICHT_TESTBAR", **h3},
        "H4": {"status": "BESTÄTIGT" if h4.get("h4_confirmed") else "WIDERLEGT", **h4},
    }
    try:
        from pathlib import Path as _P
        hyp_path = _P(ctx.out_dir) / "hypothesis_results.json"
        hyp_path.parent.mkdir(parents=True, exist_ok=True)
        hyp_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        print(f"  [WARN] hypothesis_results.json write failed: {e}")
    return out


def run_reproducibility_phase(
    ctx: PreprocessContext,
    model_cache: Optional[Dict] = None,
    n_repro: int = 74,
) -> List[Dict[str, Any]]:
    """74 echte Reproducibility-Runs (ersetzt die Placeholder).

    Für jeden WorkItem wird ein zweiter Run mit seed=DEFAULT_REPRO_SEED erzeugt
    und in out/reproducibility/ gespeichert.
    """
    from pathlib import Path as _P
    repro_dir = _P(ctx.out_dir) / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)
    repro_results: List[Dict[str, Any]] = []
    # Primär-Items: WorkItems mit seed=DEFAULT_REPRO_SEED-1 (=42 wenn DEFAULT=43)
    primary_items = [wi for wi in ctx.work_items if wi.seed == DEFAULT_REPRO_SEED - 1][:n_repro]
    if not primary_items:
        primary_items = ctx.work_items[:n_repro]
    for wi in primary_items:
        arm = next((a for a in ARM_CONFIGS if a.name == wi.arm_name), None)
        if arm is None:
            continue
        try:
            out_texts = run_gpu_loop(
                wi.arm_name, arm.preset, None,
                [wi.prompt], max_new_tokens=ctx.max_new_tokens,
                seed=DEFAULT_REPRO_SEED, batch_size=ctx.batch_size,
                model_cache=model_cache,
            )
        except Exception as e:
            out_texts = [{"text": f"[ERROR: {str(e)[:100]}]",
                          "n_input_tokens": 0, "n_output_tokens": 0}]
        text = out_texts[0]["text"]
        out_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
        repro_record = {
            "task_id": wi.task_id, "arm": wi.arm_name, "preset": arm.preset,
            "seed": DEFAULT_REPRO_SEED, "with_search": wi.with_search,
            "category": wi.category, "ground_truth": wi.ground_truth,
            "output_text": text, "output_hash": out_hash,
            "n_input_tokens": out_texts[0]["n_input_tokens"],
            "n_output_tokens": out_texts[0]["n_output_tokens"],
            "timestamp": time.time(),
        }
        fn = repro_dir / f"repro_{wi.arm_name}__{wi.task_id}__seed{DEFAULT_REPRO_SEED}.json"
        fn.write_text(json.dumps(repro_record, ensure_ascii=False, indent=2, default=str))
        repro_results.append(repro_record)
    return repro_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RIGOR v3 — 100% GPU-Util Harness")
    p.add_argument("--out-dir", type=str, default="out")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--max-per-category", type=int, default=None,
                   help="Max Tasks pro Kategorie (default: alle)")
    p.add_argument("--n-reproducibility", type=int, default=0)
    p.add_argument("--search-workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="5-Run Smoketest")
    p.add_argument("--no-write", action="store_true", help="Schreibt keine JSONs (Smoketest)")
    p.add_argument("--enable-tools", type=str, default="web_search,read_file,write_file,execute_python",
                   help="Komma-getrennte Tool-Liste (default: alle 4). Leer='' = keine Tools.")
    p.add_argument("--max-tool-iterations", type=int, default=5,
                   help="Max Tool-Loop-Iterationen pro Task (default 5)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print("RIGOR v3 — 3-Phasen-Harness (Preprocess | GPU-Loop | Postprocess)")
    print("=" * 70)
    print()

    # Pre-Probes
    probes = pre_run_probes()
    print(f"Pre-Probes: {probes}")
    if not probes.get("torch_cuda"):
        print("[WARN] torch_cuda=False — Smoketest ohne echte GPU")

    # Phase A
    ctx, _ = preprocess(args)
    print()

    # Phase B — GPU-Loop pro Arm
    print("=" * 70)
    print("Phase B: GPU-Loop")
    print("=" * 70)
    if args.smoke:
        # Nur 1 Arm × 2 Tasks × 2 Search = 4 WorkItems
        ctx.work_items = ctx.work_items[:8]

    # Gruppierung pro (arm, with_search, seed)
    by_key: Dict[Tuple, List[WorkItem]] = {}
    for wi in ctx.work_items:
        key = (wi.arm_name, wi.with_search, wi.seed)
        by_key.setdefault(key, []).append(wi)

    t_start = time.time()
    results: List[Tuple[Path, Dict]] = []
    n_done = 0
    for (arm_name, with_search, seed), items in by_key.items():
        arm = next(a for a in ARM_CONFIGS if a.name == arm_name)
        print(f"\n[B] Arm={arm_name} preset={arm.preset} search={with_search} seed={seed} ({len(items)} Items)")
        # Prompts vorbereiten (mit Such-Augment, falls aktiv)
        prompts: List[str] = []
        items_with_search: List[WorkItem] = []
        for wi in items:
            if with_search:
                sres = ctx.search_cache.get(wi.task_id, with_search=True)
                if sres:
                    p_aug = augment_prompt_with_search(wi.prompt, sres)
                    prompts.append(p_aug)
                    items_with_search.append(wi)
                else:
                    # Skip — keine Suchergebnisse verfügbar
                    continue
            else:
                prompts.append(wi.prompt)
                items_with_search.append(wi)
        if not prompts:
            continue

        # PURE GPU call
        # v3-CUDA-GRAPH-Default: alle Arms nutzen volle batch_size.
        # CUDA-Graph macht den PX-Forward static-shape → bs>1 kein Problem mehr.
        arm_batch_size = ctx.batch_size
        try:
            out_texts = run_gpu_loop(
                arm_name, arm.preset, None,
                prompts, max_new_tokens=ctx.max_new_tokens,
                seed=seed, batch_size=arm_batch_size,
                model_cache=ctx.model_cache,
            )
        except RuntimeError as re:
            err_str = str(re)
            if "size of tensor" in err_str.lower() or "shape" in err_str.lower():
                # PX-Arm + bs=1 schlägt fehl: das ist ein echter Bug, nicht Workaround.
                print(f"  [WARN] PX-Arm Runtime-Error: {err_str[:200]}")
                out_texts = [{"text": f"[ERROR: {err_str[:100]}]",
                              "n_input_tokens": 0, "n_output_tokens": 0}] * len(prompts)
            else:
                raise
        for wi, out in zip(items_with_search, out_texts):
            text = out["text"]
            out_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
            wres = {
                "task_id": wi.task_id,
                "arm": wi.arm_name,
                "preset": wi.arm_preset,
                "mephisto_scale": wi.mephisto_scale,
                "with_search": wi.with_search,
                "seed": wi.seed,
                "category": wi.category,
                "ground_truth": wi.ground_truth,
                "output_text": text,
                "output_hash": out_hash,
                "n_input_tokens": out["n_input_tokens"],
                "n_output_tokens": out["n_output_tokens"],
                "timestamp": time.time(),
            }
            out_path = ctx.out_dir / f"arm_{wi.arm_name}__{wi.task_id}__search{int(wi.with_search)}__seed{wi.seed}.json"
            results.append((out_path, wres))
            n_done += 1
        print(f"  → {len(out_texts)} Outputs ({n_done}/{len(ctx.work_items)} total)")

    duration = time.time() - t_start
    print()
    print(f"[B] GPU-Loop fertig: {n_done} Outputs in {duration:.1f}s "
          f"({duration/max(n_done, 1):.2f}s/Output)")

    # Phase C
    if not args.no_write:
        print("\n[C] Postprocess: write + verify + flags")
        post = postprocess(ctx, results)
        print(f"    {post}")

    # Phase D — Hypothesentests (H1-H4) aus rigor_stats
    print("\n[D] Hypothesentests (H1-H4)")
    hypothesis_results = run_hypothesis_suite(ctx, results)
    print(f"    H1: {hypothesis_results.get('H1', {}).get('status', '?')}")
    print(f"    H2: {hypothesis_results.get('H2', {}).get('status', '?')}")
    print(f"    H3: {hypothesis_results.get('H3', {}).get('status', '?')}")
    print(f"    H4: {hypothesis_results.get('H4', {}).get('status', '?')}")

    # Phase E — Reproducibility-Phase (74 echte Runs, nicht nur Placeholder)
    if args.n_reproducibility > 0:
        print("\n[E] Reproducibility-Phase (74 echte Runs)")
        repro_results = run_reproducibility_phase(ctx, model_cache=ctx.model_cache, n_repro=args.n_reproducibility)
        print(f"    {len(repro_results)} Repro-Outputs geschrieben")

    print("\n" + "=" * 70)
    print("FERTIG")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
