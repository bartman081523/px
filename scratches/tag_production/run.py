"""scratches/tag_production/run.py — End-to-End Harness für Tag-Produktion.

Lädt gemma3-1b-it nativ (kein PX-Patch, BASELINE-Äquivalent), baut
System-Prompt mit CitMind-Profil + Standard-Tag-Snip via
append_tag_snippet, generiert 10 Test-Prompts und misst Tag-Compliance.

CLI:
    python run.py --smoke           # 1 Prompt, 64 Token, ~30 s
    python run.py --full            # 10 Prompts × 1 Seed, ~3-15 Min
    python run.py --full --seeds 3  # für statistische Robustheit

Output:
    out/smoke_<ts>.json   (Smoke-Test)
    out/run_<ts>.json     (Full-Run, Rohdaten + Aggregate)
    out/REPORT.md         (lesbare Tabelle + manuelle Notes-Sektion)

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
from typing import Optional

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
from gradio_tabs.system_prompt import (  # noqa: E402
    inject_into_messages,
    merge_system_into_first_user,
    append_tag_snippet,
    list_profiles,
    build_system_message,
)

from scratches.tag_production.prompts import (  # noqa: E402
    TEST_PROMPTS,
    SMOKE_PROMPT,
)
from scratches.tag_production.metrics import (  # noqa: E402
    compute_per_response_metrics,
    aggregate_run_metrics,
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


# ─── 2. System-Prompt-Bau (identisch zu streaming_bridge.py:209-227) ─────


def build_messages(
    profile_name: str,
    user_prompt: str,
    use_tag_snip: bool = True,
) -> list:
    """Baut die messages-Liste 1:1 wie der Server-Pfad.

    WORKAROUND (Plan 6.1): ``merge_system_into_first_user`` rendert das
    System aus ``profile_name`` neu via ``render_for_chat_template`` —
    der Tag-Snip, der VOR dem Merge via ``append_tag_snippet`` an die
    System-Message angehängt wurde, geht dabei verloren. Wir mergen
    deshalb VOR ``merge_system_into_first_user`` manuell: kopiere den
    vorhandenen System-Inhalt (inkl. Snip) als user-turn-Prefix, entferne
    das System-Item, und übergebe eine NEUE user-Liste an
    ``merge_system_into_first_user`` mit ``profile_name="neutral"``
    (kein Re-Render). So bleibt der Snip erhalten.

    Siehe streaming_bridge.py:209-227 für die Streaming-Bridge-Variante
    (gleicher Bug, gleicher Workaround wäre dort nötig — ist hier nicht
    in scope, weil wir die Webapp/CLI nicht anfassen).
    """
    base = [{"role": "user", "content": user_prompt}]
    msgs = inject_into_messages(base, profile_name=profile_name)

    if use_tag_snip:
        msgs = append_tag_snippet(msgs, render_tag_system_prompt())

    # WORKAROUND: System-Inhalt aus messages extrahieren (inkl. Snip),
    # als user-turn-Prefix setzen, System-Item droppen.
    sys_idx = next(
        (i for i, m in enumerate(msgs) if m.get("role") == "system"),
        None,
    )
    if sys_idx is not None:
        sys_content = msgs[sys_idx].get("content") or ""
        if isinstance(sys_content, str) and sys_content:
            # User-turn finden, Prefix setzen, System droppen.
            user_idx = next(
                (i for i, m in enumerate(msgs) if m.get("role") == "user"),
                None,
            )
            new_msgs = [m for m in msgs if m.get("role") != "system"]
            if user_idx is not None:
                # user_idx hat sich durch drop um 1 verschoben, neu suchen:
                user_idx_new = next(
                    (i for i, m in enumerate(new_msgs)
                     if m.get("role") == "user"),
                    None,
                )
                if user_idx_new is not None:
                    old_user_content = new_msgs[user_idx_new].get("content") or ""
                    new_msgs[user_idx_new] = {
                        **new_msgs[user_idx_new],
                        "content": sys_content + "\n\n" + old_user_content,
                    }
                else:
                    new_msgs.insert(0, {"role": "user", "content": sys_content})
            else:
                new_msgs.insert(0, {"role": "user", "content": sys_content})
            msgs = new_msgs

    # Jetzt hat msgs KEINEN system-Eintrag mehr. merge_system_into_first_user
    # mit neutral → keine Re-Injection (nur system-strip, was wir schon
    # gemacht haben). Ergebnis ist sauber für Gemma3-Chat-Template.
    msgs = merge_system_into_first_user(msgs, profile_name="neutral")
    return msgs


# ─── 3. Generation ────────────────────────────────────────────────────────


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


# ─── 4. Smoke-Test ────────────────────────────────────────────────────────


def run_smoke(model, tokenizer) -> Path:
    """Ein-Prompt-Test: validiert dass die ganze Pipeline integer ist."""
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[smoke] Prompt: {SMOKE_PROMPT!r}")
    msgs = build_messages("citmind", SMOKE_PROMPT, use_tag_snip=True)
    print(f"[smoke] Messages-Anzahl: {len(msgs)}")
    print(f"[smoke] System-Eintrag vorhanden: "
          f"{any(m['role'] == 'system' for m in msgs)}")
    sys_content = next((m["content"] for m in msgs if m["role"] == "system"), None)
    if isinstance(sys_content, str):
        print(f"[smoke] System-Content-Länge: {len(sys_content)} Zeichen")
        print(f"[smoke] Tag-Snip vorhanden: "
              f"{'VOCODER-TAG-SYSTEM' in sys_content}")

    gen = generate_one(
        model, tokenizer, msgs, max_new_tokens=64, seed=0,
    )
    metrics = compute_per_response_metrics(gen["answer"], prompt_id="smoke")

    print(f"\n[smoke] Antwort ({gen['output_tokens']} tokens, "
          f"{gen['gen_time_sec']}s):")
    print(f"  >>> {gen['answer']!r}")
    print(f"[smoke] Tag-Counts: total={metrics['classification']['total_tags']}, "
          f"note={metrics['classification']['note']}, "
          f"dynamic={metrics['classification']['dynamic']}, "
          f"affect={metrics['classification']['affect']}, "
          f"pause={metrics['classification']['pause']}")
    print(f"[smoke] Density: {metrics['classification']['density_per_100w']}/100w")
    if metrics["classification"]["density_warning"]:
        print(f"[smoke] WARN: {metrics['classification']['density_warning']}")

    payload = {
        "meta": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model_id": "gemma3-1b-it",
            "model_hf_id": MODEL_REGISTRY["gemma3-1b-it"]["hf_id"],
            "profile": "citmind",
            "use_tag_snip": True,
            "max_new_tokens": 64,
            "temperature": 0.7,
        },
        "smoke_prompt": SMOKE_PROMPT,
        "messages_built": msgs,
        "generation": gen,
        "metrics": metrics,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"smoke_{ts}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[smoke] → {out_path}")
    return out_path


# ─── 5. Volllauf ──────────────────────────────────────────────────────────


def render_report(payload: dict, agg: dict, responses: list) -> str:
    """Baut einen Markdown-Report aus dem Run-Dict."""
    meta = payload["meta"]
    lines = [
        f"# Tag-Production Run — {meta['model_id']}",
        "",
        f"**Started:** {meta['started_at']}  ",
        f"**Profile:** `{meta['profile']}`  ",
        f"**Tag-Snip:** {meta['use_tag_snip']}  ",
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
        preview = r["answer"][:200].replace("\n", " ").replace("|", "\\|")
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
        "Notes hier eintragen (Scratch-Konvention: Edits in den Outputs erlaubt):",
        "",
        "```",
        "- p01: ...",
        "- p02: ...",
        "- p10: ...",
        "```",
        "",
        "## Hypothese vs Befund",
        "",
        "**Erwartung**: bei Standard-Variante A vermutlich `tag_rate ≤ 0.2`, "
        "`note_tag_rate ≈ 0.0` (Snip-Beispiel enthält keine Noten, "
        "CitMind ist musik-neutral).",
        "",
        "**Falls `note_tag_rate ≥ 0.5`**: Variante A funktioniert — "
        "nächster Schritt ist Plan 6.2 (Variant B/C/D Vergleich).",
        "",
        "**Falls `note_tag_rate < 0.3`**: Variante B (musik-erweiterter "
        "Snip mit Sanskrit) als nächstes testen.",
        "",
    ])
    return "\n".join(lines)


def run_full(model, tokenizer, n_seeds: int = 1, max_new_tokens: int = 256,
             temperature: float = 0.7, out_tag: Optional[str] = None) -> Path:
    """Voller Run: 10 Prompts × n_seeds. Schreibt JSON + REPORT.md."""
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Profil prüfen — wenn citmind nicht parsbar ist, fallback auf neutral
    profiles = list_profiles()
    profile = "citmind" if "citmind" in profiles else "neutral"
    print(f"[full] Profile: {profile} (available: {profiles})")

    sys_msg = build_system_message(profile)
    print(f"[full] System-Content-Länge: {len(sys_msg['content'])} Zeichen")

    started = datetime.now(timezone.utc).isoformat()
    responses = []

    n_total = len(TEST_PROMPTS) * n_seeds
    print(f"[full] Starte {n_total} Generierungen "
          f"({len(TEST_PROMPTS)} Prompts × {n_seeds} Seeds)...")

    for p_def in TEST_PROMPTS:
        msgs = build_messages(profile, p_def["prompt"], use_tag_snip=True)
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

    # Aggregate über ALLE Antworten (auch Errors zählen als 0-Tag).
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
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "model_id": "gemma3-1b-it",
            "model_hf_id": MODEL_REGISTRY["gemma3-1b-it"]["hf_id"],
            "dtype": MODEL_REGISTRY["gemma3-1b-it"]["dtype"],
            "profile": profile,
            "use_tag_snip": True,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "n_prompts": len(TEST_PROMPTS),
            "n_seeds": n_seeds,
        },
        "config": {
            "tag_snip_source": "gradio_tabs.vocoder_tags.render_tag_system_prompt",
            "system_message_source": (
                "gradio_tabs.system_prompt.build_system_message('citmind')"
            ),
            "append_method": "gradio_tabs.system_prompt.append_tag_snippet",
            "merge_method": (
                "gradio_tabs.system_prompt.merge_system_into_first_user"
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

    suffix = f"_{out_tag}" if out_tag else ""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"run_{ts}{suffix}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # REPORT.md schreiben
    report_md = render_report(payload, agg, responses)
    report_path = out_dir / "REPORT.md"
    # Bestehender Report wird überschrieben (latest = last run)
    report_path.write_text(report_md, encoding="utf-8")

    print(f"\n[full] → {json_path}")
    print(f"[full] → {report_path}")
    print(f"\n[full] Aggregate:")
    print(f"  tag_rate = {agg['tag_rate']}")
    print(f"  note_tag_rate = {agg['note_tag_rate']}")
    print(f"  tags_per_100_words_global = {agg['tags_per_100_words_global']}")
    return json_path


# ─── 6. CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    parser.add_argument("--smoke", action="store_true",
                        help="Nur 1 Prompt, 64 Token (~30 s)")
    parser.add_argument("--full", action="store_true",
                        help="Volle 10-Prompt-Matrix")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Seeds pro Prompt (default 1)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max neue Tokens pro Antwort")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling-Temperature")
    parser.add_argument("--out-tag", type=str, default=None,
                        help="Suffix für Output-Filename (z.B. 'baseline')")
    args = parser.parse_args()

    if not (args.smoke or args.full):
        parser.print_help()
        print("\nFEHLER: --smoke oder --full angeben.")
        sys.exit(1)

    print("=" * 60)
    print(" TAG-PRODUCTION HARNESS — gemma3-1b-it + CitMind + Tag-Snip")
    print("=" * 60)
    print(f" Python: {platform.python_version()}")
    print(f" CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    model, tokenizer, device = build_model("gemma3-1b-it")
    try:
        if args.smoke:
            run_smoke(model, tokenizer)
        elif args.full:
            run_full(
                model, tokenizer,
                n_seeds=args.seeds,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                out_tag=args.out_tag,
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[tag-production] Modell entladen, GPU-Cache geleert.")


if __name__ == "__main__":
    main()