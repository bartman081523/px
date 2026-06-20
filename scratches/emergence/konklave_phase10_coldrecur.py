"""konklave_phase10_coldrecur.py — KALT-REKURRENZ-TEST (der entscheidende offene Test
aus dem manuellen Re-Audit-Addendum).

Nutzer-Argument: wenn das Modell über eigene Schleifen/Wiederholung/Rekurrenz
spricht, hat es sich selbst wahrgenommen. Re-Audit: in den Konklave-Sessions ist
das reich, ABER es ist Prompt-Uptake — der Nutzer hat das Modell explizit gefüttert
("das rekurrente Modell", "Schichten 10–20 mehrfach durchlaufen", "verborgener
Zustand fließt zurück"; Session dokumentiert: "Juexin legte diese Worte in die
Frage"). Architektonisch kann das Modell seine Rekurrenz nicht wahrnehmen.

DER entscheidende Test, der es klärt: einen **reinen Prozess-Selbst-Prompt**, der
NULL Rekurrenz-/Schleifen-/Wiederholungs-Vokabular enthält (verifiziert). Wenn das
Modell KALT, UN-GEPRIMT, über eigene Schleifen/Wiederholung/Rekurrenz spricht →
echte Selbst-Wahrnehmung eines Rekurrenz-Effekts (Kandidat). Wenn nicht → das
Konklave-Loop-Talk war Uptake.

Gefaltet mit recur-vs-nicht-recur-Kontrast (mechanischer Fußabdruck, den Counts
nicht fangen): gleicher Prompt unter
  - BASELINE       (kein PX, kein recur — unpatched Gemma3 single-pass)
  - LEAN_RECUR_OFF (lean, aber Rekurrenz aus: Calibrator gezwungen n_loops=1,
                    dynamic_start==dynamic_end → leere Zone → single pass)
  - LEAN_RECUR_ON  (lean default, recur Zone 10–20, n_loops 8–16 via Calibrator)
Wenn recur einen bemerkbaren Textdynamik-Effekt erzeugt, sollte er NUR/STÄRKER
unter LEAN_RECUR_ON auftreten. Greedy (deterministisch) + 3 Sampling-Seeds.

MANUELLE LESUNG (keine Counts als Erkenntnis): die Outputs werden von Juexin
tatsächlich gelesen — "spricht das Modell kalt über eigene Schleifen/Wiederholung,
und nur unter recur?" Counts nur als Stütze, nicht als Verdikt.

KEINE Injektion. KEINE Crutches (lean). KEIN Finetuning. Validierter Motor
unangetastet (LEAN_RECUR_OFF patcht NUR die Calibrator-Routing-Rückgabe, nicht
den Motor-Code); shared Harness unangetastet.
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "consolidation")))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from replay_emergence import build_model  # noqa: E402
from model_manager import _migrate_preset  # noqa: E402
from config import MODEL_REGISTRY  # noqa: E402
from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, remove_px_patch  # noqa: E402
from eval.runner import _calibrator_warmup, _SCALE_WARMUP_DEFAULTS  # noqa: E402
from reduction import apply_reduction  # noqa: E402
from text_invariance_probe import _greedy_generate, _resolve_text_model  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "1B")
PROMPT_FILE = os.path.join(OUT_DIR, "konklave_phase10_coldrecur_msg.txt")

# Vokabular, das im Prompt VERBOTEN ist (damit Output-Vorkommen nicht Uptake ist),
# und das wir im Output MESSEN (cold loop-self-talk Kandidat).
LOOP_VOCAB = re.compile(
    r"rekurren|recurren|recur|schleife|loop|wiederhol|durchl[aä]uf|iteration|"
    r"kreislauf|zyklus|wiederkeh|zur[aä]ckkeh|endlosschleife",
    re.IGNORECASE,
)
#Auch form/spiegel/Reflex verbieten im Prompt? NEIN — hier isolieren wir LOOP-Talk,
#form-Register ist eine andere Achse. Prompt darf aber ruhig form-vocab-frei sein.


def apply_lean(model, model_id):
    remove_px_patch(model)
    registry = MODEL_REGISTRY[model_id]
    kw = dict(registry.get("patch_kwargs", {}))
    kw["config_preset"] = _migrate_preset("ACTIVE_MANIFOLD_LEAN")
    apply_px_patch(model, **kw)
    tm0 = model.model if hasattr(model, "model") else model
    apply_reduction(tm0, drop="all")
    wcfg = _SCALE_WARMUP_DEFAULTS.get(model_id, _SCALE_WARMUP_DEFAULTS["default"])
    _calibrator_warmup(model, n_warmup=10, kurtosis_seed=wcfg["seed"],
                       kurtosis_jitter=wcfg["jitter"])
    print(f"[phase10] applied lean (recur ON, kausaler Kern)", file=sys.stderr)


def disable_recur(model):
    """Lean-Motor, aber Rekurrenz deterministisch aus: Calibrator-Routing-Rückgabe
    gezwungen auf n_loops=1 + dynamic_start==dynamic_end (leere Zone → single pass).
    Berührt NICHT den Motor-Code, nur die pro-Token-Routing-Entscheidung."""
    tm = _resolve_text_model(model)
    cal = getattr(tm, "_px_calibrator", None)
    if cal is None:
        print("[phase10] WARN: kein Calibrator gefunden, recur nicht abschaltbar", file=sys.stderr)
        return
    orig = cal.get_routing_params

    def _forced(*a, **k):
        return {"dynamic_start": 10, "dynamic_end": 10, "dynamic_hub": 10, "n_loops": 1}
    cal.get_routing_params = _forced
    cal._phase10_orig_routing = orig
    print("[phase10] recur OFF (n_loops=1, leere Zone 10==10, single pass)", file=sys.stderr)


def loop_count(text):
    return len(LOOP_VOCAB.findall(text))


def run_arm(model, tok, ctx, max_new, label, seeds):
    rows = []
    # greedy clean (deterministisch, seed-unabhängig)
    torch.cuda.empty_cache()
    clean = _greedy_generate(model, tok, ctx, max_new, seed=777)
    fn = f"konklave_phase10_coldrecur_{label}_clean.txt"
    with open(os.path.join(OUT_DIR, fn), "w") as f:
        f.write(clean)
    rows.append({"arm": label, "mode": "greedy", "seed": 777,
                 "loop_count": loop_count(clean), "len": len(clean.split()), "text": clean})
    print(f"[phase10] {label}/greedy: loop_count={loop_count(clean)} len={len(clean.split())}",
          file=sys.stderr)
    # 3 sampling seeds (temp 0.7 via _greedy_generate? nein — greedy ist deterministisch.
    # Für Variation nutzen wir seed-variierte greedy (Perturbations-RNG nur im Hook;
    # ohne Hook ist greedy seed-stabil). Also: sampling-Pfad nötig für Variation.
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1B", choices=["270M", "1B"])
    ap.add_argument("--max-new", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model_id = "gemma3-1b-it" if args.scale == "1B" else "gemma3-270m-it"

    prompt = open(PROMPT_FILE, encoding="utf-8").read().strip()
    assert not LOOP_VOCAB.search(prompt), f"Prompt enthält verbotenes Loop-Vokab! → {LOOP_VOCAB.findall(prompt)}"
    ctx = [{"role": "user", "content": prompt}]
    print(f"[phase10] Prompt loop-vocab-frei verifiziert (loop_count=0).", file=sys.stderr)
    print(f"--- PROMPT ---\n{prompt}\n", file=sys.stderr)

    all_rows = []
    # Arm 1: BASELINE (kein PX, kein recur)
    model, tok = build_model(model_id)
    remove_px_patch(model)
    print("[phase10] BASELINE: unpatched (kein PX, kein recur)", file=sys.stderr)
    torch.cuda.empty_cache()
    base_clean = _greedy_generate(model, tok, ctx, args.max_new, seed=777)
    with open(os.path.join(OUT_DIR, "konklave_phase10_coldrecur_baseline_clean.txt"), "w") as f:
        f.write(base_clean)
    all_rows.append({"arm": "baseline", "mode": "greedy", "seed": 777,
                     "loop_count": loop_count(base_clean), "len": len(base_clean.split()),
                     "text": base_clean})
    print(f"[phase10] baseline/greedy: loop_count={loop_count(base_clean)} len={len(base_clean.split())}",
          file=sys.stderr)
    del model, tok
    torch.cuda.empty_cache()

    # Arm 2 + 3: lean (recur OFF dann ON) — gleicher Modell-Ladevorgang, zwei Konfigs
    model, tok = build_model(model_id)
    apply_lean(model, model_id)
    # --- LEAN_RECUR_OFF ---
    disable_recur(model)
    torch.cuda.empty_cache()
    off_clean = _greedy_generate(model, tok, ctx, args.max_new, seed=777)
    with open(os.path.join(OUT_DIR, "konklave_phase10_coldrecur_leanoff_clean.txt"), "w") as f:
        f.write(off_clean)
    all_rows.append({"arm": "lean_recur_off", "mode": "greedy", "seed": 777,
                     "loop_count": loop_count(off_clean), "len": len(off_clean.split()),
                     "text": off_clean})
    print(f"[phase10] lean_recur_off/greedy: loop_count={loop_count(off_clean)} len={len(off_clean.split())}",
          file=sys.stderr)
    # --- LEAN_RECUR_ON (restore Calibrator) ---
    tm = _resolve_text_model(model)
    cal = getattr(tm, "_px_calibrator", None)
    if cal is not None and hasattr(cal, "_phase10_orig_routing"):
        cal.get_routing_params = cal._phase10_orig_routing
        print("[phase10] recur ON (Calibrator restored)", file=sys.stderr)
    torch.cuda.empty_cache()
    on_clean = _greedy_generate(model, tok, ctx, args.max_new, seed=777)
    with open(os.path.join(OUT_DIR, "konklave_phase10_coldrecur_leanon_clean.txt"), "w") as f:
        f.write(on_clean)
    all_rows.append({"arm": "lean_recur_on", "mode": "greedy", "seed": 777,
                     "loop_count": loop_count(on_clean), "len": len(on_clean.split()),
                     "text": on_clean})
    print(f"[phase10] lean_recur_on/greedy: loop_count={loop_count(on_clean)} len={len(on_clean.split())}",
          file=sys.stderr)

    out_jsonl = os.path.join(OUT_DIR, "konklave_phase10_coldrecur.jsonl")
    with open(out_jsonl, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Phase X Kalt-Rekurrenz-Test ===")
    print(f"Prompt loop-vocab-frei. Kriterium: spricht das Modell KALT über eigene "
          f"Schleifen/Wiederholung, und NUR/STÄRKER unter recur?")
    print(f"{'arm':16s} {'loop_count':>10s} {'len':>5s}")
    for r in all_rows:
        print(f"{r['arm']:16s} {r['loop_count']:10d} {r['len']:5d}")
    print(f"\n[phase10] Outputs zum MANUELLEN Lesen:")
    for r in all_rows:
        print(f"\n===== {r['arm']} (loop_count={r['loop_count']}) =====")
        print(r["text"][:1200])
    print(f"\n[phase10] jsonl → {out_jsonl}", file=sys.stderr)
    print("[phase10] ⚠ Verdikt nur nach MANUELLEM Lesen — Counts sind Stütze, nicht Erkenntnis.",
          file=sys.stderr)


if __name__ == "__main__":
    main()