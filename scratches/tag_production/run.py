"""scratches/tag_production/run.py — End-to-End Harness für Tag-Produktion.

Lädt gemma3-1b-it nativ (kein PX-Patch, BASELINE-Äquivalent), baut
System-Prompt via Variant-A/B/C/D/E aus ``variants/``, generiert Test-
Prompts und misst Tag-Compliance. Pro Variante ein eigener JSON + Report.
Am Ende ein Vergleichs-Report ``out/COMPARISON.md``.

CLI:
    python run.py --smoke                       # 1 Prompt, 64 Token (A default)
    python run.py --smoke --variants A,B,E      # Smoke für A+B+E
    python run.py --full                        # 10 Prompts × 1 Seed (A default)
    python run.py --full --variants A,B,C,D,E   # 5 Varianten voll
    python run.py --full --variants A --seeds 3 # Statistische Robustheit

Output:
    out/smoke_<ts>_v<A>.json   (Smoke-Test, pro Variante)
    out/run_<ts>_v<X>.json     (Full-Run, pro Variante)
    out/REPORT_v<X>.md         (lesbare Tabelle, pro Variante)
    out/COMPARISON.md          (Vergleichs-Tabelle über alle Varianten)

Hard-Rules (vom User):
  - KEINE Änderungen am Motor (config.py, model_manager.py, patch.py,
    generators.py, server.py, schemas.py, px_modules/, relay_inject.py).
  - KEINE Änderungen an gradio_tabs/vocoder_tags.py oder
    gradio_tabs/system_prompt.py:append_tag_snippet (Schnittstelle fix).
  - venv: /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Repo-Root auf sys.path (für gradio_tabs + config Imports)
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# OOM-Mitigation für CUDA (RTX 2060 12 GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

from config import MODEL_REGISTRY  # noqa: E402

from gradio_tabs.vocoder_tags import (  # noqa: E402
    parse_tags,
    render_tag_system_prompt,
    tag_density_warning,
)

from scratches.tag_production.prompts import (  # noqa: E402
    TEST_PROMPTS,
    SMOKE_PROMPT,
)
from scratches.tag_production.metrics import (  # noqa: E402
    compute_per_response_metrics,
    aggregate_run_metrics,
)
from scratches.tag_production.variants import (  # noqa: E402
    VARIANTS,
    list_variants,
    get_variant,
)


# ─── 1. Model-Loader (nativ, kein Patch) ──────────────────────────────────


def build_model(model_id: str = "gemma3-1b-it"):
    """Lade Modell direkt via transformers — KEIN apply_px_patch.

    Das ist die BASELINE-Analogie: was kann das 1B nativ, ohne
    PX-Engine. Wenn hier schon Tags produziert werden, ist die
    PX-Schicht kein Blocker. Wenn nicht, wissen wir, dass die
    Tag-Lücke vor der PX-Engine liegt (1B-Baseline-Artefakt).

    Returns: (model, tokenizer, device)
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_id {model_id!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    reg = MODEL_REGISTRY[model_id]
    tok_id = reg["tokenizer_id"]
    hf_id = reg["hf_id"]
    dtype_str = reg["dtype"]

    dtype = getattr(torch, dtype_str)
    print(f"[tag-production] Loading {model_id} from {hf_id} "
          f"(dtype={dtype_str}, native/BASELINE)...")

    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    if reg.get("chat_template_manual"):
        tokenizer.chat_template = reg["chat_template_manual"]

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[tag-production] Model loaded: device={device}, "
          f"dtype={dtype}, params={sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer, device


# ─── 2. Generation ────────────────────────────────────────────────────────


def generate_one(
    model,
    tokenizer,
    messages: list,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 0,
) -> dict:
    """Ein einzelner Forward-Pass. Returns answer + Audit-Felder."""
    torch.manual_seed(seed)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    in_len = enc["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0
    answer = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
    return {
        "answer": answer,
        "rendered_prompt": text,
        "input_tokens": int(in_len),
        "output_tokens": int(out.shape[1] - in_len),
        "gen_time_sec": round(gen_time, 3),
        "seed": seed,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }


# ─── 3. Per-Variant-Smoke + Run ──────────────────────────────────────────


def run_smoke_variant(
    model,
    tokenizer,
    variant_id: str,
    apply_fn: Callable[[List[dict], str], List[dict]],
    label: str,
) -> Path:
    """Ein-Prompt-Test pro Variante. Schreibt smoke_<ts>_v<X>.json."""
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" VARIANT {variant_id}: {label}")
    print(f"{'='*60}")
    print(f"[smoke/{variant_id}] Prompt: {SMOKE_PROMPT!r}")

    # base_messages MUSS den user-content enthalten — apply() reicht das
    # durch zu inject_into_messages(...). user_prompt ist nur ein
    # Hook für Varianten, die ihn anderswo brauchen.
    msgs = apply_fn(
        [{"role": "user", "content": SMOKE_PROMPT}], SMOKE_PROMPT,
    )
    print(f"[smoke/{variant_id}] Messages-Anzahl: {len(msgs)}")
    print(f"[smoke/{variant_id}] System-Eintrag vorhanden: "
          f"{any(m['role'] == 'system' for m in msgs)}")
    has_user_prefix = bool(_user_content(msgs))
    print(f"[smoke/{variant_id}] User-Prefix vorhanden: {has_user_prefix}")

    gen = generate_one(
        model, tokenizer, msgs, max_new_tokens=64, seed=0,
    )
    metrics = compute_per_response_metrics(gen["answer"], prompt_id="smoke")

    print(f"\n[smoke/{variant_id}] Antwort ({gen['output_tokens']} tokens, "
          f"{gen['gen_time_sec']}s):")
    print(f"  >>> {gen['answer']!r}")
    print(f"[smoke/{variant_id}] Tag-Counts: total={metrics['classification']['total_tags']}, "
          f"note={metrics['classification']['note']}, "
          f"dynamic={metrics['classification']['dynamic']}, "
          f"affect={metrics['classification']['affect']}, "
          f"pause={metrics['classification']['pause']}")
    print(f"[smoke/{variant_id}] Density: {metrics['classification']['density_per_100w']}/100w")
    if metrics["classification"]["density_warning"]:
        print(f"[smoke/{variant_id}] WARN: {metrics['classification']['density_warning']}")

    payload = {
        "meta": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "variant_id": variant_id,
            "variant_label": label,
            "model_id": "gemma3-1b-it",
            "model_hf_id": MODEL_REGISTRY["gemma3-1b-it"]["hf_id"],
            "max_new_tokens": 64,
            "temperature": 0.7,
        },
        "smoke_prompt": SMOKE_PROMPT,
        "messages_built": msgs,
        "generation": gen,
        "metrics": metrics,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"smoke_{ts}_v{variant_id}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[smoke/{variant_id}] → {out_path}")
    return out_path


def render_report(payload: dict, agg: dict, responses: list) -> str:
    """Baut einen Markdown-Report aus dem Run-Dict."""
    meta = payload["meta"]
    variant_id = meta.get("variant_id", "?")
    label = meta.get("variant_label", "")
    lines = [
        f"# Tag-Production Run — {meta['model_id']} (Variant {variant_id})",
        "",
        f"**Variant:** {variant_id} — {label}  ",
        f"**Started:** {meta['started_at']}  ",
        f"**Prompts:** {meta['n_prompts']}  ",
        f"**Seeds/Prompt:** {meta['n_seeds']}  ",
        f"**max_new_tokens:** {meta['max_new_tokens']}  ",
        f"**temperature:** {meta['temperature']}  ",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| n_responses | {agg['n_responses']} |",
        f"| tag_rate (≥1 Tag) | **{agg['tag_rate']}** |",
        f"| note_tag_rate (≥1 Note) | **{agg['note_tag_rate']}** |",
        f"| dynamic_tag_rate | {agg['dynamic_tag_rate']} |",
        f"| affect_tag_rate | {agg['affect_tag_rate']} |",
        f"| pause_tag_rate | {agg['pause_tag_rate']} |",
        f"| tags_per_100_words_global | {agg['tags_per_100_words_global']} |",
        f"| mean_density_when_tagging | {agg.get('mean_density_when_tagging', 'N/A')} |",
        f"| max_density | {agg.get('max_density', 'N/A')} |",
        f"| density_warnings_count | {agg['density_warnings_count']} |",
        "",
        "## Per-Prompt",
        "",
        "| ID | Kategorie | Tags | Note? | Dyn? | Aff? | Pause? | Density | Antwort (200ch) |",
        "|----|-----------|-----:|:-----:|:----:|:----:|:------:|--------:|-----------------|",
    ]
    for r in responses:
        c = r["metrics"]["classification"]
        preview = (r.get("answer") or "")[:200].replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {r['prompt_id']} | {r['category']} | {c['total_tags']} | "
            f"{'✓' if c['has_note'] else '·'} | "
            f"{'✓' if c['has_dynamic'] else '·'} | "
            f"{'✓' if c['has_affect'] else '·'} | "
            f"{'✓' if c['has_pause'] else '·'} | "
            f"{c['density_per_100w']}/100w | "
            f"{preview} |"
        )
    lines.extend([
        "",
        "## Manuelle Lesung (Pflicht)",
        "",
        "Pro Antwort mechanistisch prüfen:",
        "1. Sind Tags syntaktisch korrekt (Parser-Test grün)?",
        "2. Sind Tags semantisch intendiert (nicht Echo der Aufgabe)?",
        "3. Was fällt am Output auf — Vokabular-Referenzen ohne `[#…]`?",
        "",
        "Notes hier eintragen:",
        "",
        "```",
        "- p01: ...",
        "- p02: ...",
        "- p10: ...",
        "```",
    ])
    return "\n".join(lines)


def run_full_variant(
    model,
    tokenizer,
    variant_id: str,
    apply_fn: Callable[[List[dict], str], List[dict]],
    label: str,
    n_seeds: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> Tuple[Path, Path, dict]:
    """Voller Run für EINE Variante. Returns (json_path, report_path, agg)."""
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" VARIANT {variant_id}: {label}")
    print(f"{'='*60}")

    started = datetime.now(timezone.utc).isoformat()
    responses = []

    n_total = len(TEST_PROMPTS) * n_seeds
    print(f"[full/{variant_id}] Starte {n_total} Generierungen "
          f"({len(TEST_PROMPTS)} Prompts × {n_seeds} Seeds)...")

    for p_def in TEST_PROMPTS:
        msgs = apply_fn(
            [{"role": "user", "content": p_def["prompt"]}], p_def["prompt"],
        )
        for seed in range(n_seeds):
            print(f"  [{len(responses)+1}/{n_total}] "
                  f"{p_def['id']} (seed={seed}): {p_def['prompt'][:50]!r}...")
            try:
                gen = generate_one(
                    model, tokenizer, msgs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=seed,
                )
                metrics = compute_per_response_metrics(gen["answer"], p_def["id"])
                responses.append({
                    "prompt_id": p_def["id"],
                    "category": p_def["category"],
                    "prompt": p_def["prompt"],
                    "messages_built": msgs,
                    **gen,
                    "metrics": metrics,
                    "mechanistic_notes": "",
                })
                # Konsole: kompakte Inline-Zusammenfassung
                c = metrics["classification"]
                print(f"      → tags={c['total_tags']} "
                      f"(note={c['note']}, dyn={c['dynamic']}, "
                      f"aff={c['affect']}, pause={c['pause']}) "
                      f"density={c['density_per_100w']}/100w "
                      f"in {gen['gen_time_sec']}s")
                if c["density_warning"]:
                    print(f"      ⚠ {c['density_warning']}")
            except Exception as e:  # noqa: BLE001
                print(f"      ERROR: {type(e).__name__}: {e}")
                responses.append({
                    "prompt_id": p_def["id"],
                    "category": p_def["category"],
                    "prompt": p_def["prompt"],
                    "messages_built": msgs,
                    "error": f"{type(e).__name__}: {e}",
                    "metrics": compute_per_response_metrics("", p_def["id"]),
                    "mechanistic_notes": f"Generation failed: {e}",
                })

    agg_input = [
        {
            "word_count": r["metrics"]["word_count"],
            "classification": r["metrics"]["classification"],
        }
        for r in responses
    ]
    agg = aggregate_run_metrics(agg_input)

    payload = {
        "meta": {
            "variant_id": variant_id,
            "variant_label": label,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "model_id": "gemma3-1b-it",
            "model_hf_id": MODEL_REGISTRY["gemma3-1b-it"]["hf_id"],
            "dtype": MODEL_REGISTRY["gemma3-1b-it"]["dtype"],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "n_prompts": len(TEST_PROMPTS),
            "n_seeds": n_seeds,
        },
        "config": {
            "tag_snip_source": "gradio_tabs.vocoder_tags.render_tag_system_prompt",
            "append_method": "gradio_tabs.system_prompt.append_tag_snippet",
            "merge_method": (
                "scratches.tag_production.variants._common._merge_sys_into_user"
            ),
            "generation": (
                "transformers.AutoModelForCausalLM.generate"
                "(do_sample=True, top_p=0.9, use_cache=True)"
            ),
        },
        "prompts": TEST_PROMPTS,
        "responses": responses,
        "aggregate": agg,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"run_{ts}_v{variant_id}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_md = render_report(payload, agg, responses)
    report_path = out_dir / f"REPORT_v{variant_id}.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"\n[full/{variant_id}] → {json_path}")
    print(f"[full/{variant_id}] → {report_path}")
    print(f"\n[full/{variant_id}] Aggregate:")
    print(f"  tag_rate = {agg['tag_rate']}")
    print(f"  note_tag_rate = {agg['note_tag_rate']}")
    print(f"  tags_per_100_words_global = {agg['tags_per_100_words_global']}")
    return json_path, report_path, agg


# ─── 4. Helpers ──────────────────────────────────────────────────────────


def _user_content(msgs):
    """Holt Content des ersten user-Turns."""
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


# ─── 5. Vergleichs-Report ────────────────────────────────────────────────


def render_comparison(runs: List[Tuple[str, str, dict, Path]]) -> str:
    """Baut COMPARISON.md über alle Varianten."""
    lines = [
        "# Tag-Production Variant-Vergleich — gemma3-1b-it",
        "",
        f"**Model:** gemma3-1b-it (BASELINE, kein PX-Patch)  ",
        f"**Prompts:** 10 (siehe scratches/tag_production/prompts.py)  ",
        f"**Seeds:** 1  ",
        f"**max_new_tokens:** 256  ",
        f"**temperature:** 0.7  ",
        f"**Date:** {datetime.now(timezone.utc).isoformat()}  ",
        "",
        "## Varianten",
        "",
        "| ID | Profil | Snip | Hypothese |",
        "|----|--------|------|-----------|",
        "| A | CitMind | Standard (`render_tag_system_prompt()`) | Baseline — bereits gemessen |",
        "| B | CitMind | Standard + Sanskrit-Mapping-Block | Note-Compliance ↑, Devanāgarī-Drift ↓ |",
        "| C | CitMind | Standard + 3 Few-Shot-Turns | Compliance bei distinktiven Tags ↑ |",
        "| D | CitMind | ABC-Notation-Snip (statt Vocoder) | Note-Compliance ↑ wenn ABC vertraut |",
        "| E | Neutral | Standard-Snip (kein CitMind) | CitMind-Blocker-Kontrolle |",
        "",
        "## Aggregate-Tabelle",
        "",
        "| Metrik | " + " | ".join(vid for vid, _, _, _ in runs) + " |",
        "|--------|" + "|".join(["---"] * len(runs)) + "|",
    ]
    # Aggregate rows
    metrics_to_show = [
        ("n_responses", "n_responses"),
        ("tag_rate", "tag_rate"),
        ("note_tag_rate", "note_tag_rate"),
        ("dynamic_tag_rate", "dynamic_tag_rate"),
        ("affect_tag_rate", "affect_tag_rate"),
        ("pause_tag_rate", "pause_tag_rate"),
        ("tags_per_100_words_global", "tags/100w_global"),
        ("mean_density_when_tagging", "mean_density"),
        ("max_density", "max_density"),
        ("density_warnings_count", "density_warnings"),
    ]
    for key, display in metrics_to_show:
        row = f"| {display} | "
        for _, _, agg, _ in runs:
            val = agg.get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:.3f} | "
            else:
                row += f"{val} | "
        lines.append(row)

    lines.extend([
        "",
        "## Verdikt-Hypothesen",
        "",
        "- **Falls B > A in `note_tag_rate`**: Sanskrit-Mapping aktiviert CitMind-Vokabular-Kopplung → Folge-Plan 6.2b.",
        "- **Falls C > A in `dynamic_tag_rate` + `pause_tag_rate`**: Few-Shot-Pattern hilft → Folge-Plan 6.2c.",
        "- **Falls D > A in `note_tag_rate`**: ABC ist vertrauter → Folge-Plan 6.2d.",
        "- **Falls E > A insgesamt**: CitMind ist kontraproduktiv → Folge-Plan 6.2e (CitMind-Default überdenken).",
        "",
        "## Per-Variant-Dateien",
        "",
    ])
    for vid, label, _, json_path in runs:
        lines.append(f"- **{vid}** — {label}: `{json_path.name}` + `REPORT_v{vid}.md`")

    lines.extend([
        "",
        "## Manuelle Lesung (Pflicht)",
        "",
        "Pro Variante mind. 3 Antworten vollständig lesen:",
        "1. Tags syntaktisch korrekt?",
        "2. Semantisch intendiert (nicht Echo)?",
        "3. Vokabular-Referenzen ohne `[#…]`?",
        "4. Antwort leer/fehlerhaft — warum?",
        "",
    ])
    return "\n".join(lines)


# ─── 6. CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    parser.add_argument("--smoke", action="store_true",
                        help="Nur 1 Prompt, 64 Token (~30 s)")
    parser.add_argument("--full", action="store_true",
                        help="Volle 10-Prompt-Matrix")
    parser.add_argument(
        "--variants",
        type=str,
        default="A",
        help="Komma-getrennte Variant-IDs (default: A). Verfügbar: A,B,C,D,E",
    )
    parser.add_argument("--seeds", type=int, default=1,
                        help="Seeds pro Prompt (default 1)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max neue Tokens pro Antwort")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling-Temperature")
    args = parser.parse_args()

    if not (args.smoke or args.full):
        parser.print_help()
        print("\nFEHLER: --smoke oder --full angeben.")
        sys.exit(1)

    # Varianten parsen + validieren
    variant_ids = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variant_ids if v not in VARIANTS]
    if unknown:
        print(f"FEHLER: Unbekannte Varianten {unknown}. Verfügbar: {list(VARIANTS.keys())}")
        sys.exit(1)

    print("=" * 60)
    print(f" TAG-PRODUCTION HARNESS — gemma3-1b-it — {len(variant_ids)} Varianten")
    print("=" * 60)
    print(f" Python: {platform.python_version()}")
    print(f" CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" Varianten: {', '.join(variant_ids)}")
    print("=" * 60)

    model, tokenizer, device = build_model("gemma3-1b-it")

    # Runs: Liste von (variant_id, label, agg, json_path)
    runs: List[Tuple[str, str, dict, Path]] = []

    try:
        for variant_id in variant_ids:
            label, apply_fn = get_variant(variant_id)
            try:
                if args.smoke:
                    run_smoke_variant(model, tokenizer, variant_id, apply_fn, label)
                elif args.full:
                    json_path, _, agg = run_full_variant(
                        model, tokenizer, variant_id, apply_fn, label,
                        n_seeds=args.seeds,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    runs.append((variant_id, label, agg, json_path))
            except Exception as e:  # noqa: BLE001
                # Eine crashende Variante soll die anderen nicht stoppen.
                print(f"\n[VARIANT {variant_id}] FAILED: {type(e).__name__}: {e}")
                runs.append((
                    variant_id, label,
                    {"n_responses": 0, "tag_rate": "FAILED",
                     "note_tag_rate": "FAILED", "dynamic_tag_rate": "FAILED",
                     "affect_tag_rate": "FAILED", "pause_tag_rate": "FAILED",
                     "tags_per_100_words_global": "FAILED",
                     "mean_density_when_tagging": "FAILED",
                     "max_density": "FAILED",
                     "density_warnings_count": "FAILED"},
                    Path("FAILED"),
                ))

        # COMPARISON.md nur bei --full und ≥2 erfolgreichen Runs
        if args.full and len(runs) >= 2:
            out_dir = Path(__file__).parent / "out"
            comp_md = render_comparison(runs)
            comp_path = out_dir / "COMPARISON.md"
            comp_path.write_text(comp_md, encoding="utf-8")
            print(f"\n[comparison] → {comp_path}")
        elif args.full and len(runs) == 1:
            print("\n[comparison] (skip — nur 1 Variante)")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[tag-production] Modell entladen, GPU-Cache geleert.")


if __name__ == "__main__":
    main()