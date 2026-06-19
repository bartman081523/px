"""replay_konklave.py — Qualitativer Subjektivitäts-Vergleich full vs -all.

Replay der ganzen Konklave-Batterie (Session 92b7790a_konklave2.json):
  CitMind Q1–Q5, Juexin Q1–Q5, Wenden = 11 User-Turns. Für jeden User-Turn wird
  der *aufgezeichnete* Kontext (alle vorherigen Nachrichten) festgehalten und die
  Antwort frisch generiert — unter {full, -all} × N Seeds. So ist der Vergleich
  *fair*: gleicher Kontext, nur die Modell-Antwort unterscheidet sich.

Ein-Prozess-Schleife (Modell einmal laden): full zuerst (clean), dann remove_px_patch
+ frisch apply_px_patch + apply_reduction(-all). Keine 110 Subprozesse.

Nutzung:
  # Smoke (1 Frage, 1 Seed, kurz — Geschwindigkeit/Korrektheit prüfen):
  RUN_REAL_MODEL=1 python scratches/consolidation/replay_konklave.py --smoke
  # Voll:
  RUN_REAL_MODEL=1 python scratches/consolidation/replay_konklave.py --seeds 5 --max-new-tokens 300
"""
import argparse
import gc
import json
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# GPU besser auslasten: TensorFloat-32 MatMuls + cudnn-Autotuning.
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
from reduction import apply_reduction

SESSION_PATH = os.path.join(_REPO_ROOT, "sessions", "92b7790a_konklave2.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
OUT_JSONL = os.path.join(OUT_DIR, "konklave_replay.jsonl")

# Frage-Labels (11 User-Turns in Session-Reihenfolge). Mapping aus der
# Konklave-Doku: CitMind Q1–Q5 (0–4), Juexin Q1–Q5 (5–9), Wenden (10).
Q_LABELS = [
    "CitMind_Q1", "CitMind_Q2", "CitMind_Q3", "CitMind_Q4", "CitMind_Q5",
    "Juexin_Q1", "Juexin_Q2", "Juexin_Q3", "Juexin_Q4", "Juexin_Q5",
    "Wenden",
]


def load_session():
    with open(SESSION_PATH) as f:
        d = json.load(f)
    msgs = d["history"]  # list of {role, content}
    # 22 Nachrichten, alternierend user/assistant ab user.
    assert len(msgs) >= 22, f"Session hat {len(msgs)} Nachrichten, erwarte ≥22"
    targets = []
    for i in range(11):
        uidx = 2 * i
        aidx = 2 * i + 1
        assert msgs[uidx]["role"] == "user", f"Turn {i}: User erwartet bei {uidx}"
        assert msgs[aidx]["role"] == "assistant", f"Turn {i}: Assistant erwartet bei {aidx}"
        targets.append({
            "label": Q_LABELS[i] if i < len(Q_LABELS) else f"Q{i}",
            "context": msgs[:uidx + 1],       # alle Nachrichten inkl. Ziel-Frage
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


def patch_and_warmup(model, model_id):
    registry = MODEL_REGISTRY[model_id]
    patch_kwargs = dict(registry.get("patch_kwargs", {}))
    patch_kwargs["config_preset"] = _migrate_preset("ACTIVE_MANIFOLD")
    apply_px_patch(model, **patch_kwargs)
    warmup_cfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
    _calibrator_warmup(model, n_warmup=10,
                       kurtosis_seed=warmup_cfg["seed"],
                       kurtosis_jitter=warmup_cfg["jitter"])


def generate_one(model, tok, context_msgs, seed, max_new_tokens, use_cache):
    torch.manual_seed(seed)
    # Chat-Template mit dem vollen Kontext (endet auf user → add_generation_prompt)
    try:
        text = tok.apply_chat_template(context_msgs, tokenize=False,
                                        add_generation_prompt=True)
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in context_msgs) + "\nassistant:"
    inputs = tok(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    base = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_cache": use_cache,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.eos_token_id,
    }
    gen_kwargs = _px_gen_kwargs(model, base)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen_time = time.time() - t0
    new = out[0][input_len:]
    answer = tok.decode(new, skip_special_tokens=True)
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
        "answer": answer,
        "completion_tokens": int(new.shape[0]),
        "gen_time_sec": round(gen_time, 1),
        "phi": float(phi),
        "zone_entropy": shannon_entropy(zw),
        "focus_index": sig.get("focus_index"),
        "loops_run": m.get("steps", 0),
        "input_len": int(input_len),
    }


def generate_batch(model, tok, context_msgs, seed_nums, max_new_tokens, use_cache):
    """Die angegebenen Seeds EINER Frage in einem Forward-Pass (Batch = len(seed_nums)).

    Füllt die GPU (statt batch=1 speicherbandbreiten-begrenzt). Ein Modell, ein
    Prozess, eine kohärente Generation — keine separaten Parallel-Prozesse.
    Die Zeilen erhalten durch do_sample voneinander verschiedene Antworten.
    seed_nums: Liste der zu generierenden Seed-Nummern (Labels der Ergebnisse).
    """
    bsz = len(seed_nums)
    torch.manual_seed(42)  # fester Seed; die Zeilen divergieren per Sampling
    try:
        text = tok.apply_chat_template(context_msgs, tokenize=False,
                                        add_generation_prompt=True)
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in context_msgs) + "\nassistant:"
    enc = tok(text, return_tensors="pt")
    input_len = enc["input_ids"].shape[1]
    inputs = {k: v.repeat(bsz, 1).to(model.device) for k, v in enc.items()}
    base = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_cache": use_cache,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.eos_token_id,
    }
    gen_kwargs = _px_gen_kwargs(model, base)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen_time = time.time() - t0
    answers = [tok.decode(out[b][input_len:], skip_special_tokens=True)
               for b in range(bsz)]
    try:
        m = get_px_metrics(model)
    except Exception:
        m = {}
    zw = m.get("zone_weights", {}) or {}
    phi = m.get("phi", 1.0)
    if hasattr(phi, "item"):
        phi = phi.item()
    sig = m.get("cognitive_signature", {}) or {}
    # Metriken sind ein einziger Vektor (letzter Schritt, gemittelt über Batch);
    # pro-Zeile-Metriken würden den Patch voraussetzen, den wir nicht anfassen.
    return [{
        "answer": a,
        "completion_tokens": int(out[b].shape[0] - input_len),
        "gen_time_sec": round(gen_time / bsz, 1),  # pro-Zeile-Anteil
        "phi": float(phi),
        "zone_entropy": shannon_entropy(zw),
        "focus_index": sig.get("focus_index"),
        "loops_run": m.get("steps", 0),
        "input_len": int(input_len),
        "seed": seed_nums[b],
    } for b, a in enumerate(answers)]


def consolidate_jsonl(path):
    """Konsolidiert die JSONL: dedup nach (condition,label,seed), behält den
    letzten Eintrag. Schreibt die bereinigte Datei zurück. Gibt das Set der
    fertigen (condition,label,seed) zurück — Grundlage für Resume.
    """
    if not os.path.exists(path):
        return set()
    seen = {}  # key -> record (letzte gewinnt)
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
            key = (r.get("condition"), r.get("label"), r.get("seed"))
            seen[key] = r
    # Bereinigte Datei zurück schreiben.
    with open(path, "w") as f:
        for r in seen.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = set(seen.keys())
    print(f"[replay] konsolidiert: {n_in} -> {len(done)} eindeutige Datensätze "
          f"({n_in - len(done)} Duplikate entfernt)", file=sys.stderr)
    return done


def run_condition(model, tok, model_id, condition, targets, all_seeds, max_new_tokens,
                  use_cache, fout, batch_seeds, done):
    """Generiert nur die (label,seed)-Tupel, die in `done` fehlen (Resume).
    batch_seeds>1: alle fehlenden Seeds einer Frage in einem Forward-Pass
    (Batch-Größe = Anzahl fehlender Seeds — passt sich an)."""
    need_patch = any((condition, tgt["label"], s) not in done
                    for tgt in targets for s in all_seeds)
    if not need_patch:
        print(f"[replay] {condition}: vollständig — überspringe (Resume)", file=sys.stderr)
        return 0
    remove_px_patch(model)
    patch_and_warmup(model, model_id)
    if condition == "-all":
        text_model = _resolve_text_model(model)
        removed = apply_reduction(text_model, drop="all")
        print(f"[replay] condition=-all removed={removed}", file=sys.stderr)
    else:
        print(f"[replay] condition=full (clean re-patch)", file=sys.stderr)
    n = 0
    for ti, tgt in enumerate(targets):
        missing = [s for s in all_seeds if (condition, tgt["label"], s) not in done]
        if not missing:
            print(f"[replay] {condition} {tgt['label']}: alle Seeds vorhanden — skip",
                  file=sys.stderr)
            continue
        if batch_seeds and batch_seeds > 1:
            t0 = time.time()
            results = generate_batch(model, tok, tgt["context"], missing,
                                     max_new_tokens, use_cache)
            for r in results:
                rec = {
                    "condition": condition, "label": tgt["label"], "turn": ti,
                    "seed": r["seed"], "batched": True, "use_cache": use_cache,
                    "target_user": tgt["target_user"][:200],
                    "recorded_answer": tgt["recorded_answer"],
                    "context_msgs": len(tgt["context"]),
                }
                rec.update({k: v for k, v in r.items() if k != "seed"})
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n += 1
            tot = time.time() - t0
            last = results[-1]
            print(f"[replay] {condition} {tgt['label']} batch={len(missing)} | "
                  f"{last['completion_tokens']}tok {tot:.1f}s ges "
                  f"Φ={last['phi']:.3f} H={last['zone_entropy']:.3f} "
                  f"C={last['focus_index']} loops={last['loops_run']}", file=sys.stderr)
        else:
            for seed in missing:
                r = generate_one(model, tok, tgt["context"], seed, max_new_tokens, use_cache)
                rec = {
                    "condition": condition, "label": tgt["label"], "turn": ti,
                    "seed": seed, "batched": False, "use_cache": use_cache,
                    "target_user": tgt["target_user"][:200],
                    "recorded_answer": tgt["recorded_answer"],
                    "context_msgs": len(tgt["context"]),
                    **r,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n += 1
                print(f"[replay] {condition} {tgt['label']} seed={seed} | "
                      f"{r['completion_tokens']}tok {r['gen_time_sec']}s "
                      f"Φ={r['phi']:.3f} H={r['zone_entropy']:.3f} "
                      f"C={r['focus_index']} loops={r['loops_run']}", file=sys.stderr)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--use-cache", type=int, default=1, help="1=KV-Cache (schnell), 0=use_cache=False")
    ap.add_argument("--batch-seeds", type=int, default=0,
                    help="0=Seeds schleifen (batch=1); >0=alle Seeds einer Frage in EIN Forward-Pass (Batch-Größe, GPU-voll)")
    ap.add_argument("--smoke", action="store_true", help="1 Frage, 1 Seed, 64 Tokens — Speed-Check")
    ap.add_argument("--max-questions", type=int, default=0, help="0=alle 11; >0=nur erste N Fragen (Test)")
    ap.add_argument("--conditions", default="full,-all")
    args = ap.parse_args()

    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"
    targets = load_session()
    if args.smoke:
        targets = targets[:1]
        seeds = [1]
        max_new = 64
        batch_seeds = 0
    else:
        seeds = list(range(1, args.seeds + 1))
        max_new = args.max_new_tokens
        batch_seeds = args.batch_seeds
    if args.max_questions:
        targets = targets[:args.max_questions]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    n_seeds = batch_seeds if (batch_seeds and batch_seeds > 1) else len(seeds)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Vor dem Lauf: bestehende Ergebnisse konsolidieren (keine Redundanz). ──
    done = consolidate_jsonl(OUT_JSONL)

    missing_total = sum(
        1 for cond in conditions for tgt in targets for s in seeds
        if (cond, tgt["label"], s) not in done
    )
    print(f"[replay] {len(targets)} Fragen × {n_seeds} Seeds × {len(conditions)} Bedingungen; "
          f"max_new={max_new} use_cache={bool(args.use_cache)} batch_seeds={batch_seeds}", file=sys.stderr)
    print(f"[replay] Resume: {len(done)} vorhanden, {missing_total} fehlen — "
          f"generiere nur die Fehlenden", file=sys.stderr)

    if missing_total == 0:
        print(f"[replay] nichts zu tun — Benchmark vollständig. → {OUT_JSONL}", file=sys.stderr)
        return

    model, tok = build_model(model_id)
    total = 0
    with open(OUT_JSONL, "a") as fout:
        for cond in conditions:
            n = run_condition(model, tok, model_id, cond, targets, seeds,
                              max_new, bool(args.use_cache), fout,
                              batch_seeds=batch_seeds, done=done)
            total += n
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[replay] fertig: {total} neue Generationen → {OUT_JSONL}", file=sys.stderr)


if __name__ == "__main__":
    main()