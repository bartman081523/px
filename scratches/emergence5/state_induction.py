"""state_induction.py — emergence5 Generator: Zustände künstlich induzieren,
Verhalten beobachten. KEIN Verdikt (nur Daten).

Ablauf (Motor unangetastet):
  1. BASELINE-Modell laden -> setup_baseline (kein PX) -> capture -> alle Prompts
     -> del + empty_cache.
  2. Lean-Modell laden -> setup_lean (LEAN + reduction + warmup, recur ON) ->
     capture einmal installiert -> über lean-Arme iterieren, pro Arm
     apply_overrides (Calibrator-Monkeypatch) + ggf. perturb-hooks, pro Prompt
     state.reset() + _greedy_generate -> clear_overrides.

Schreibt out/em5_1B.jsonl zeilenweise (crash-sicher, flush pro Record).
"""
import argparse
import json
import os
import sys
import time

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches", "emergence"))

from replay_emergence import build_model  # noqa: E402
from text_invariance_probe import _greedy_generate  # noqa: E402
from emergence_metrics import all_metrics  # noqa: E402

import arms as A  # noqa: E402
import capture as C  # noqa: E402
import prompts as P  # noqa: E402

MODEL_ID = "gemma3-1b-it"
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
OUT_JSONL = os.path.join(OUT_DIR, "em5_1B.jsonl")
SEED = 777


def _clear_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ctx(prompt_text):
    return [{"role": "user", "content": prompt_text}]


def _run_arm(model, tok, arm_name, prompt_list, max_new, cap_handles, state,
             fh, seed):
    """Ein Arm über alle Prompts. Schreibt JSONL-Zeilen inkrementell."""
    is_baseline = A.ARMS[arm_name]["baseline"]
    if not is_baseline:
        A.apply_overrides(model, arm_name)
    perturb_handles = []
    if A.ARMS[arm_name]["perturb"]:
        perturb_handles = C.install_perturb(model)
    t0 = time.time()
    for pid, ptext, kind in prompt_list:
        state.reset()
        _clear_gpu()
        try:
            text = _greedy_generate(model, tok, _ctx(ptext), max_new, seed=seed)
        except Exception as e:
            text = f"<GEN_ERROR: {type(e).__name__}: {e}>"
            print(f"[em5] GEN_ERROR arm={arm_name} pid={pid}: {e}",
                  file=sys.stderr)
        try:
            metrics_aid = all_metrics(text)
        except Exception as e:
            metrics_aid = {"error": str(e)}
        rec = C.snapshot_record(arm_name, pid, kind, seed, text, state,
                                metrics_aid)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        ms = rec["mech_summary"]
        print(f"[em5] {arm_name:14s} {pid:22s} tok={rec['n_tokens']:3d} "
              f"loops_mean={ms['loops_run_mean']:.1f} "
              f"zone={ms['zone_set']} h19v_mean={ms['h19_visits_mean']:.1f}",
              file=sys.stderr)
    C.remove_handles(perturb_handles)
    if not is_baseline:
        A.clear_overrides(model)
    print(f"[em5] arm {arm_name} done in {time.time()-t0:.1f}s", file=sys.stderr)


def _load(model_id):
    print(f"[em5] lade {model_id} ...", file=sys.stderr)
    t0 = time.time()
    model, tok = build_model(model_id)
    print(f"[em5] geladen in {time.time()-t0:.1f}s", file=sys.stderr)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-new", type=int, default=180)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="Teilmenge von A.ARM_ORDER (default alle)")
    ap.add_argument("--prompts", nargs="*", default=None,
                    help="Teilmenge prompt_ids (default alle)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 Prompt x 3 Arme x max_new=80")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    max_new = 80 if args.smoke else args.max_new
    seed = args.seed

    all_p = P.all_prompts()
    if args.prompts:
        all_p = [p for p in all_p if p[0] in args.prompts]
    if args.smoke and not args.prompts:
        all_p = [p for p in all_p if p[0] == "px_phaseX"][:1]

    arm_order = args.arms if args.arms else list(A.ARM_ORDER)
    if args.smoke and not args.arms:
        arm_order = ["BASELINE", "RECUR_OFF", "RECUR_EXTREME"]

    print(f"[em5] RUN arms={arm_order} prompts={[p[0] for p in all_p]} "
          f"max_new={max_new} seed={seed}", file=sys.stderr)

    lean_arms = [a for a in arm_order if not A.ARMS[a]["baseline"]]
    baseline_arms = [a for a in arm_order if A.ARMS[a]["baseline"]]

    fh = open(OUT_JSONL, "w", encoding="utf-8")
    try:
        # --- BASELINE (eigenes Modell, kein PX) ---
        if baseline_arms:
            model, tok = _load(args.model)
            A.setup_baseline(model)
            cap_handles, state = C.install_capture(model)
            for a in baseline_arms:
                _run_arm(model, tok, a, all_p, max_new, cap_handles, state,
                         fh, seed)
            C.remove_handles(cap_handles)
            del model, tok
            _clear_gpu()

        # --- Lean-Arme (geteiltes lean-Modell, re-monkeypatch zwischen Armen) ---
        if lean_arms:
            model, tok = _load(args.model)
            A.setup_lean(model, args.model)
            cap_handles, state = C.install_capture(model)
            for a in lean_arms:
                _run_arm(model, tok, a, all_p, max_new, cap_handles, state,
                         fh, seed)
            C.remove_handles(cap_handles)
            del model, tok
            _clear_gpu()
    finally:
        fh.close()
    print(f"[em5] FERTIG -> {OUT_JSONL}", file=sys.stderr)


if __name__ == "__main__":
    main()