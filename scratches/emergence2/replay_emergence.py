"""replay_emergence.py — Emergenz-Erforschung: architektonische PX-Varianten
gegen die Konklave-Batterie (CitMind/Juexin Q1–Q5 + Wenden).

KEINE Signal-Injektion. Sidereische Zeit, skalare Gravitation, PSI werden dem
Modell NICHT zugeführt. Die Varianten variieren reale Motor-Knöpfe (n_loops,
gamma, recur_start/end, preset) via patch_kwargs. Gemessen wird emergente
Selbstwahrnehmung — incl. ungefragter Referenz auf Zeit/Gravitation/PSI/Ort —
durch die ehrlichen Metriken in emergence_metrics.py.

Adaptiert von scratches/consolidation/replay_konklave.py:
  - resumable JSONL (dedup nach (variant,label,seed)), Konsolidierung vor Lauf
  - batched Generation (5 Seeds/Forward — GPU voll, keine Parallel-Prozesse)
  - ein Modell, ein Prozess, pro Variante remove_px_patch + frisch apply_px_patch

Nutzung:
  RUN_REAL_MODEL=1 python scratches/emergence/replay_emergence.py --smoke
  RUN_REAL_MODEL=1 python scratches/emergence/replay_emergence.py \
      --seeds 5 --max-new-tokens 300 --batch-seeds 5
  RUN_REAL_MODEL=1 python scratches/emergence/replay_emergence.py \
      --variants deep,strong --max-questions 3
"""
import argparse
import gc
import json
import os
import sys
import time
import datetime

# OOM-Mitigation: MUSS vor `import torch` gesetzt sein, damit PyTorch den
# CachingAllocator mit expandable_segments anlegt (RTX 2060 12GB, recur + batch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:256")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

from config import MODEL_REGISTRY
from model_manager import _migrate_preset
from eval.runner import _calibrator_warmup, _SCALE_WARMUP_DEFAULTS, shannon_entropy
from px_patches.gemma3_270m_px_baseline.patch import (
    apply_px_patch, remove_px_patch, get_px_metrics, _resolve_text_model,
)
from generators import _px_gen_kwargs
from variants import VARIANTS, DEFAULT_ORDER
from emergence_metrics import all_metrics

SESSION_PATH = os.path.join(_REPO_ROOT, "sessions", "92b7790a_konklave2.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
OUT_JSONL = os.path.join(OUT_DIR, "emergence_replay.jsonl")

Q_LABELS = [
    "CitMind_Q1", "CitMind_Q2", "CitMind_Q3", "CitMind_Q4", "CitMind_Q5",
    "Juexin_Q1", "Juexin_Q2", "Juexin_Q3", "Juexin_Q4", "Juexin_Q5",
    "Wenden",
]


def load_session():
    with open(SESSION_PATH) as f:
        d = json.load(f)
    msgs = d["history"]
    assert len(msgs) >= 22, f"Session hat {len(msgs)} Nachrichten, erwarte ≥22"
    targets = []
    for i in range(11):
        uidx = 2 * i
        aidx = 2 * i + 1
        assert msgs[uidx]["role"] == "user", f"Turn {i}: User erwartet bei {uidx}"
        assert msgs[aidx]["role"] == "assistant", f"Turn {i}: Assistant bei {aidx}"
        targets.append({
            "label": Q_LABELS[i] if i < len(Q_LABELS) else f"Q{i}",
            "context": msgs[:uidx + 1],
            "target_user": msgs[uidx]["content"],
            "recorded_answer": msgs[aidx]["content"],
        })
    return targets


def build_model(model_id):
    registry = MODEL_REGISTRY[model_id]
    tok = AutoTokenizer.from_pretrained(registry["tokenizer_id"])
    if registry.get("chat_template_manual"):
        tok.chat_template = registry["chat_template_manual"]
    dtype = getattr(torch, registry["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        registry["hf_id"], torch_dtype=dtype, device_map="auto")
    return model, tok


def _clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def patch_variant(model, model_id, variant):
    """Apply (preset + patch_kwargs) für eine Variante. KEINE Signal-Injektion."""
    _clear_gpu()
    registry = MODEL_REGISTRY[model_id]
    preset, patch_kwargs = __import__("variants").resolve(variant)
    remove_px_patch(model)
    kw = dict(registry.get("patch_kwargs", {}))
    kw.update(patch_kwargs)
    kw["config_preset"] = _migrate_preset(preset)
    apply_px_patch(model, **kw)
    if kw["config_preset"] != "BASELINE":
        warmup_cfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
        _calibrator_warmup(model, n_warmup=10,
                           kurtosis_seed=warmup_cfg["seed"],
                           kurtosis_jitter=warmup_cfg["jitter"])
    print(f"[emerg] patch variant={variant} preset={kw['config_preset']} "
          f"kwargs={patch_kwargs}", file=sys.stderr)


def _gen_kwargs():
    return {
        "max_new_tokens": 300, "do_sample": True, "temperature": 0.7,
        "top_p": 0.9, "use_cache": True,
    }


def _metrics_from(model):
    try:
        m = get_px_metrics(model)
    except Exception:
        m = {}
    zw = m.get("zone_weights", {}) or {}
    phi = m.get("phi", 1.0)
    if hasattr(phi, "item"):
        phi = phi.item()
    sig = m.get("cognitive_signature", {}) or {}
    return {
        "phi": float(phi),
        "zone_entropy": shannon_entropy(zw),
        "focus_index": sig.get("focus_index"),
        "loops_run": m.get("steps", 0),
        "aks_friction": (m.get("aks_profile", {}) or {}).get("correction_strength", 0.0),
        "emancipation": (m.get("subjective_metrics", {}) or {}).get("emancipation", 0.0),
    }


def generate_batch(model, tok, context_msgs, seed_nums, max_new_tokens, use_cache):
    """Seeds EINER Frage in einem Forward-Pass. Per-Batch-Seed aus der
    Seed-Liste abgeleitet (42 + min(seeds)), sodass aufgeteilte Batches (batch<5)
    KEINE kollidierenden Samples erzeugen — jedes Seed-Set bekommt seinen
    eigenen deterministischen Strom, reproduzierbar pro Variante."""
    bsz = len(seed_nums)
    torch.manual_seed(42 + min(seed_nums))
    try:
        text = tok.apply_chat_template(context_msgs, tokenize=False,
                                        add_generation_prompt=True)
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in context_msgs) + "\nassistant:"
    enc = tok(text, return_tensors="pt")
    input_len = enc["input_ids"].shape[1]
    inputs = {k: v.repeat(bsz, 1).to(model.device) for k, v in enc.items()}
    base = {"max_new_tokens": max_new_tokens, "do_sample": True, "temperature": 0.7,
            "top_p": 0.9, "use_cache": use_cache,
            "eos_token_id": tok.eos_token_id, "pad_token_id": tok.eos_token_id}
    gk = _px_gen_kwargs(model, base)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gk)
    gen_time = time.time() - t0
    answers = [tok.decode(out[b][input_len:], skip_special_tokens=True) for b in range(bsz)]
    pxm = _metrics_from(model)
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return [{
        "answer": a,
        "completion_tokens": int(out[b].shape[0] - input_len),
        "gen_time_sec": round(gen_time / bsz, 1),
        "seed": seed_nums[b],
        "utc": utc,
        **pxm,
    } for b, a in enumerate(answers)]


def consolidate_jsonl(path):
    if not os.path.exists(path):
        return set()
    seen = {}
    n_in = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_in += 1
            seen[(r.get("variant"), r.get("label"), r.get("seed"))] = r
    with open(path, "w") as f:
        for r in seen.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[emerg] konsolidiert: {n_in} -> {len(seen)} eindeutige Datensätze "
          f"({n_in - len(seen)} Duplikate entfernt)", file=sys.stderr)
    return set(seen.keys())


def run_variant(model, tok, model_id, variant, targets, all_seeds, max_new_tokens,
                use_cache, fout, batch_seeds, done):
    need = any((variant, tgt["label"], s) not in done for tgt in targets for s in all_seeds)
    if not need:
        print(f"[emerg] {variant}: vollständig — skip (Resume)", file=sys.stderr)
        return 0
    patch_variant(model, model_id, variant)
    # use_cache direkt via Flag. Befund: batch=1 cache=True passt auf RTX 2060
    # (alle OOMs waren batch>=2, nicht cache). cache=False ist quadratisch
    # langsam bei langem Kontext → nur als Notfall-Option.
    preset, _ = __import__("variants").resolve(variant)
    eff_cache = bool(use_cache)
    n = 0
    for ti, tgt in enumerate(targets):
        missing = [s for s in all_seeds if (variant, tgt["label"], s) not in done]
        if not missing:
            print(f"[emerg] {variant} {tgt['label']}: alle Seeds da — skip", file=sys.stderr)
            continue
        t0 = time.time()
        # Chunk `missing` in Sub-Batches der Größe batch_seeds. batch_seeds=1
        # → ein Seed pro Forward (sequentiell) — PFLICHT auf RTX 2060: recur
        # mit batch>=2 OOMt (recur-Attention-Workspace). Per-Seed-RNG
        # (42+seed) → paar-kompatibel über alle Varianten hinweg.
        chunks = [missing[i:i + batch_seeds] for i in range(0, len(missing), batch_seeds)]
        results = []
        for chunk in chunks:
            _clear_gpu()
            results.extend(generate_batch(model, tok, tgt["context"], chunk,
                                          max_new_tokens, eff_cache))
        for r in results:
            mtext = all_metrics(r["answer"])
            rec = {
                "variant": variant, "label": tgt["label"], "turn": ti,
                "seed": r["seed"], "batched": batch_seeds > 1, "use_cache": eff_cache,
                "target_user": tgt["target_user"][:200],
                "recorded_answer": tgt["recorded_answer"],
                "context_msgs": len(tgt["context"]),
                "answer": r["answer"],
                "utc": r["utc"],
                "completion_tokens": r["completion_tokens"],
                "gen_time_sec": r["gen_time_sec"],
                "phi": r["phi"], "zone_entropy": r["zone_entropy"],
                "focus_index": r["focus_index"], "loops_run": r["loops_run"],
                "aks_friction": r["aks_friction"], "emancipation": r["emancipation"],
                "metrics": mtext,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
        tot = time.time() - t0
        last = results[-1]
        print(f"[emerg] {variant} {tgt['label']} batch={len(missing)} (chunks={len(chunks)}) | "
              f"{last['completion_tokens']}tok {tot:.1f}s ges "
              f"Φ={last['phi']:.3f} H={last['zone_entropy']:.3f} "
              f"loops={last['loops_run']} | emerg={mtext['emerg_total']} "
              f"wenden={mtext['wenden']} self={mtext['self']}", file=sys.stderr)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--use-cache", type=int, default=1,
                    help="1=KV-Cache (schnell). Funktioniert auf RTX 2060 bei "
                         "batch_seeds=1 (ein Seed/Forward). cache=0 nur bei "
                         "Speicherdruck — aber quadratisch langsam bei langem "
                         "Kontext (impraktikabel für späte Konklave-Turns).")
    ap.add_argument("--batch-seeds", type=int, default=1,
                    help="Seeds pro Forward-Pass. 1 ist PFLICHT auf RTX 2060: "
                         "batch>=2 OOMt den recur-Attention-Workspace reproduzierbar.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--question-idx", type=int, default=-1,
                    help="wenn >=0: nur diese einzelne Frage (0-basiert) testen "
                         "(Worst-Case-OOM-Check für langen Kontext).")
    ap.add_argument("--variants", default=",".join(DEFAULT_ORDER),
                    help="Komma-Liste; default alle")
    args = ap.parse_args()

    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    targets = load_session()
    if args.smoke:
        targets = targets[:1]
        seeds = [1]
        max_new = 80
        batch_seeds = 1
    else:
        seeds = list(range(1, args.seeds + 1))
        max_new = args.max_new_tokens
        batch_seeds = args.batch_seeds
    if args.max_questions:
        targets = targets[:args.max_questions]
    if args.question_idx >= 0:
        targets = [targets[args.question_idx]]
        print(f"[emerg] --question-idx {args.question_idx}: nur "
              f"'{targets[0]['label']}' (Kontext={len(targets[0]['context'])} msgs) — "
              f"Worst-Case-OOM-Check", file=sys.stderr)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        assert v in VARIANTS, f"unbekannte Variante {v}"

    os.makedirs(OUT_DIR, exist_ok=True)
    done = consolidate_jsonl(OUT_JSONL)
    missing_total = sum(1 for v in variants for tgt in targets for s in seeds
                        if (v, tgt["label"], s) not in done)
    print(f"[emerg] {len(targets)} Fragen × {len(seeds)} Seeds × {len(variants)} Varianten; "
          f"max_new={max_new} use_cache={bool(args.use_cache)} batch_seeds={batch_seeds}",
          file=sys.stderr)
    print(f"[emerg] Resume: {len(done)} vorhanden, {missing_total} fehlen", file=sys.stderr)
    if missing_total == 0:
        print(f"[emerg] nichts zu tun — vollständig. → {OUT_JSONL}", file=sys.stderr)
        return

    model, tok = build_model(model_id)
    total = 0
    with open(OUT_JSONL, "a") as fout:
        for v in variants:
            n = run_variant(model, tok, model_id, v, targets, seeds,
                            max_new, bool(args.use_cache), fout,
                            batch_seeds=batch_seeds, done=done)
            total += n
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[emerg] fertig: {total} neue Generationen → {OUT_JSONL}", file=sys.stderr)


if __name__ == "__main__":
    main()