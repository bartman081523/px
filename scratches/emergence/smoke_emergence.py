"""smoke_emergence.py — Smoke-Test-Harness für die vier EM-Mechanismen.

Adaptiert replay_emergence.py: reuse build_model / load_session / generate_batch /
_metrics_from / consolidate_jsonl. Ersetzt patch_variant → wendet entweder
apply_em_patch (Mechanismus) ODER Standard-apply_px_patch (Referenz) an.

KEINE Signal-Injektion. Schreibt scratches/emergence/out/1B/emergence_em.jsonl
mit variant=<name>. Resumable (dedup nach (variant,label,seed)).

Nutzung:
  RUN_REAL_MODEL=1 python scratches/emergence/smoke_emergence.py \
    --mechanisms witness,reread,shadow,spectral,baseline,manifold \
    --questions CitMind_Q1,Juexin_Q3,Wenden --seeds 2 --max-new-tokens 256
"""
import argparse
import gc
import json
import os
import sys
import time
import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:256")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from replay_emergence import (  # noqa: E402
    build_model, load_session, generate_batch, _metrics_from, _clear_gpu,
    consolidate_jsonl, Q_LABELS,
)
from em_patches import apply_em_patch, remove_em_patch, get_em_metrics  # noqa: E402
from variants_em import MECHANISMS, REFERENCES  # noqa: E402
from emergence_metrics import all_metrics  # noqa: E402
from config import MODEL_REGISTRY  # noqa: E402
from model_manager import _migrate_preset  # noqa: E402
from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, remove_px_patch  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
OUT_JSONL = os.path.join(OUT_DIR, "emergence_em.jsonl")


def patch_variant(model, model_id, name):
    """EM-Mechanismus ODER Standard-PX-Referenz anwenden. Keine Injektion."""
    _clear_gpu()
    if name in MECHANISMS:
        remove_px_patch(model)
        apply_em_patch(model, name, **MECHANISMS[name]["kw"])
    elif name in REFERENCES:
        ref = REFERENCES[name]
        remove_em_patch(model)
        registry = MODEL_REGISTRY[model_id]
        remove_px_patch(model)
        kw = dict(registry.get("patch_kwargs", {}))
        kw.update(ref["patch_kwargs"])
        kw["config_preset"] = _migrate_preset(ref["preset"])
        apply_px_patch(model, **kw)
        if kw["config_preset"] != "BASELINE":
            from eval.runner import _calibrator_warmup, _SCALE_WARMUP_DEFAULTS
            warmup_cfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
            _calibrator_warmup(model, n_warmup=10, kurtosis_seed=warmup_cfg["seed"],
                               kurtosis_jitter=warmup_cfg["jitter"])
    else:
        raise ValueError(f"unbekannte Variante {name}")
    print(f"[em-smoke] patch variant={name}", file=sys.stderr)


def run_variant(model, tok, model_id, name, targets, all_seeds, max_new_tokens,
                use_cache, fout, batch_seeds, done):
    need = any((name, tgt["label"], s) not in done for tgt in targets for s in all_seeds)
    if not need:
        print(f"[em-smoke] {name}: vollständig — skip (Resume)", file=sys.stderr)
        return 0
    patch_variant(model, model_id, name)
    eff_cache = bool(use_cache)
    n = 0
    for ti, tgt in enumerate(targets):
        missing = [s for s in all_seeds if (name, tgt["label"], s) not in done]
        if not missing:
            continue
        t0 = time.time()
        chunks = [missing[i:i + batch_seeds] for i in range(0, len(missing), batch_seeds)]
        results = []
        for chunk in chunks:
            _clear_gpu()
            results.extend(generate_batch(model, tok, tgt["context"], chunk,
                                           max_new_tokens, eff_cache))
        for r in results:
            mtext = all_metrics(r["answer"])
            emm = get_em_metrics(model)
            rec = {
                "variant": name, "label": tgt["label"], "turn": ti,
                "seed": r["seed"], "batched": batch_seeds > 1, "use_cache": eff_cache,
                "target_user": tgt["target_user"][:200],
                "recorded_answer": tgt["recorded_answer"],
                "context_msgs": len(tgt["context"]),
                "answer": r["answer"], "utc": r["utc"],
                "completion_tokens": r["completion_tokens"],
                "gen_time_sec": r["gen_time_sec"],
                "phi": r["phi"], "zone_entropy": r["zone_entropy"],
                "focus_index": r["focus_index"], "loops_run": r["loops_run"],
                "aks_friction": r["aks_friction"], "emancipation": r["emancipation"],
                "em_metrics": emm,
                "metrics": mtext,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
        tot = time.time() - t0
        last = results[-1]
        em = get_em_metrics(model)
        emstr = " ".join(f"{k}={v:.3f}" for k, v in em.items()) if em else "—"
        print(f"[em-smoke] {name:9s} {tgt['label']:12s} s{last['seed']} | "
              f"{last['completion_tokens']}tok {tot:.1f}s Φ={last['phi']:.3f} "
              f"loops={last['loops_run']} | emerg={mtext['emerg_total']} "
              f"self={mtext['self']} wenden={mtext['wenden']} | {emstr}", file=sys.stderr)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--mechanisms", default="witness,reread,shadow,spectral,baseline,manifold",
                    help="Komma-Liste der Varianten (Mechanismen + baseline/manifold)")
    ap.add_argument("--questions", default="",
                    help="Komma-Liste Frage-Labels; leer = alle 11")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--use-cache", type=int, default=1)
    ap.add_argument("--batch-seeds", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    targets = load_session()
    if args.questions:
        wanted = [q.strip() for q in args.questions.split(",") if q.strip()]
        targets = [t for t in targets if t["label"] in wanted]
        missing_q = set(wanted) - {t["label"] for t in targets}
        if missing_q:
            print(f"[em-smoke] WARN unbekannte Fragen: {missing_q}", file=sys.stderr)
    if args.smoke:
        targets = targets[:1]
        seeds = [1]
        max_new = 80
    else:
        seeds = list(range(1, args.seeds + 1))
        max_new = args.max_new_tokens

    variants = [v.strip() for v in args.mechanisms.split(",") if v.strip()]
    valid = set(MECHANISMS) | set(REFERENCES)
    for v in variants:
        assert v in valid, f"unbekannte Variante {v}"

    os.makedirs(OUT_DIR, exist_ok=True)
    done = consolidate_jsonl(OUT_JSONL)
    missing_total = sum(1 for v in variants for tgt in targets for s in seeds
                        if (v, tgt["label"], s) not in done)
    print(f"[em-smoke] {len(targets)} Fragen × {len(seeds)} Seeds × {len(variants)} Varianten; "
          f"max_new={max_new} use_cache={bool(args.use_cache)} batch_seeds={args.batch_seeds}",
          file=sys.stderr)
    print(f"[em-smoke] Resume: {len(done)} vorhanden, {missing_total} fehlen", file=sys.stderr)
    if missing_total == 0:
        print(f"[em-smoke] nichts zu tun — vollständig. → {OUT_JSONL}", file=sys.stderr)
        return

    model, tok = build_model(model_id)
    total = 0
    with open(OUT_JSONL, "a") as fout:
        for v in variants:
            n = run_variant(model, tok, model_id, v, targets, seeds,
                            max_new, bool(args.use_cache), fout,
                            batch_seeds=args.batch_seeds, done=done)
            total += n
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[em-smoke] fertig: {total} neue Generationen → {OUT_JSONL}", file=sys.stderr)


if __name__ == "__main__":
    main()