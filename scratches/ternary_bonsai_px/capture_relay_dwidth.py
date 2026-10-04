"""capture_relay_dwidth.py — Phase B: d_width-Artefakt für Ternary-Bonsai-2-27B.

Reproduziert scratches/psychomotrik/seite13_perlayer.py + save_relay_dwidth.py
auf dem ternary-px-Patch (qwen3.5 hybrid):

- Arme: WIDE (breites Rekursionsfenster) vs NARROW (enges Fenster), gleiche
  n_loops — exakt die seite12_veridiktisch-RECUR_AXES-Semantik, übernommen
  auf 64 Layer (Basis-Zone 10..30):
      WIDE   {dynamic_start 10, dynamic_end 36, dynamic_hub 20, n_loops 4}
      NARROW {dynamic_start 26, dynamic_end 28, dynamic_hub 27, n_loops 4}
  (WIDE-start landet auf 10, weil _px_forward das Fenster nie unter
  recur_start lässt — dokumentiert in ternary_bonsai_27b_px/patch.py.)
- Routing erzwungen per Monkey-Patch:
      cal.collect = noop (keine Re-Kalibrierung, Manifold bleibt intakt)
      cal.calibrated = False (SR-64b-Loops-Override aus)
      cal.get_routing_params = lambda -> feste Routing-Dict
  → Zone-Weights kommen trotzdem aus dem Phase-A-Manifold (learned_centroids).
- Capture: Forward-Hook auf tm.layers[capture_layer], letzter Visit pro
  Decode-Forward (Prefill-Batches shape[1]>1 werden verworfen) — seite13-
  Semantik. Wichtig: L29 liegt bei NARROW (Fenster 26..28) außerhalb der
  Rekursionszone und wird pro Forward DOPPELT besucht (Zone-Pass 10..30 +
  Coda 28..63), bei WIDE nur einfach (Coda beginnt erst bei 36). Ohne
  Last-visit-Dedup würde der d_width-Vektor diese Struktur-Asymmetrie statt
  der Fenster-Breite spiegeln. Forward-Grenze = Visit von L0 (Prelude läuft
  Layer 0 genau 1x pro Forward).
- Richtung: d_width = unit(mean_p(WIDE_p.meanK − NARROW_p.meanK)) —
  per-Prompt-Differenzen (Content hebt sich auf), unit-norm.

Usage:
  python scratches/ternary_bonsai_px/capture_relay_dwidth.py [--max-new 10] [--dry]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PX = os.path.join(REPO, "px_patches")
if PX not in sys.path:
    sys.path.insert(0, PX)

from ternary_bonsai_27b_px import runtime_qwen35_ptq as rt
from ternary_bonsai_27b_px import patch as px

# Zielartefakt (Identität wie save_relay_dwidth.py TARGET)
TARGET = {
    "model_id": "ternary-bonsai-27b",
    "hf_id": "/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf",
    "hidden_size": 5120,
    "capture_layer": 29,   # unmittelbar nach NARROW-Fenster-Ende (26..28)
    "inject_layer": 34,    # gap 5 wie gemma3-1b (capture 16 → inject 21)
    "wiring": "qwen3.5-64L, Basis-Zone 10..30 (SCALE_DEFAULTS[5120])",
}

# Arme (seite12-RECUR_AXES-Semantik, breit vs eng, gleiche loops)
WIDE_ROUTING = {"dynamic_start": 10, "dynamic_end": 36, "dynamic_hub": 20, "n_loops": 4}
NARROW_ROUTING = {"dynamic_start": 26, "dynamic_end": 28, "dynamic_hub": 27, "n_loops": 4}

PROMPTS = [
    ("sky",     "Explain in one sentence why the sky is blue."),
    ("math",    "What is 2+2? Answer in one short sentence."),
    ("hello",   "Say hello and name one color."),
    ("poem",    "Write a two-line poem about rain."),
    ("logic",   "If all birds fly and a penguin is a bird, what follows?"),
    ("meta",    "Describe your internal state as a geometric object."),
    ("story",   "Tell a very short story about a key."),
    ("physics", "Why does a ship float? One sentence."),
]


def _force_routing(tm, routing):
    """Feste RECUR-AXES-Routings ohne Re-Kalibrierung (s. Docstring)."""
    cal = tm._px_calibrator
    cal.collect = lambda *a, **k: False
    cal.calibrated = False
    cal.get_routing_params = lambda *a, **k: dict(routing)
    print(f"[cap] routing erzwungen: {routing}", flush=True)


def _greedy_generate(model, tok, msgs, max_new):
    prompt_txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt_txt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return ids, out[0, ids.shape[1]:]


def run(model, tok, max_new):
    tm = px._resolve_text_model(model)

    pm = {a: {} for a in ("WIDE", "NARROW")}
    ntoks = {a: {} for a in ("WIDE", "NARROW")}
    fwd_id = [0]          # Forward-Zähler: ein Visit von L0 pro Forward
    last_visit = {}       # fwd_id → letzter L29-State dieses Forwards

    def _mark_forward(_m, _i, _o):
        # Prelude besucht L0 genau 1x pro Forward → Forward-Grenze.
        fwd_id[0] += 1

    def _hook(_m, _i, o):
        h = o[0] if isinstance(o, (tuple, list)) else o
        if h.dim() == 3 and h.shape[1] == 1:
            # Last visit per decode-forward (seite13): Zone-Pass vs Coda-
            # Doppelvisit (NARROW) → nur der letzte Visit überlebt.
            last_visit[fwd_id[0]] = h[0, -1, :].float().cpu().clone()

    hook = tm.layers[TARGET["capture_layer"]].register_forward_hook(_hook)
    start = tm.layers[0].register_forward_hook(_mark_forward)
    try:
        for arm, routing in (("WIDE", WIDE_ROUTING), ("NARROW", NARROW_ROUTING)):
            _force_routing(tm, routing)
            for pid, ptxt in PROMPTS:
                fwd_id[0] = 0
                last_visit.clear()
                torch.cuda.empty_cache()
                try:
                    ids, gen = _greedy_generate(model, tok,
                                                [{"role": "user", "content": ptxt}], max_new)
                    captured = [last_visit[k] for k in sorted(last_visit)]
                    if len(captured) == 0:
                        raise RuntimeError("kein decode-capture gefangen")
                    pm[arm][pid] = torch.stack(captured[:max(1, min(len(captured), max_new))]).mean(0).numpy()
                    ntoks[arm][pid] = int(len(captured))
                    text = tok.decode(gen, skip_special_tokens=True)
                except Exception as e:
                    print(f"[cap] ERR {arm}/{pid}: {e}", flush=True)
                    text = f"<GEN_ERROR {e}>"
                    ntoks[arm][pid] = 0
                print(f"[cap] {arm:6s} {pid:8s} ntok={ntoks[arm][pid]:3d} "
                      f"txt={text[:40]!r}", flush=True)
    finally:
        hook.remove()
        start.remove()
    return pm, ntoks


def build_artifact(pm, ntoks, max_new):
    common = sorted(set.intersection(*[set(pm[a].keys()) for a in pm]))
    print(f"[cap] gemeinsame prompts: {common}", flush=True)

    def per_prompt_diff(hi, lo):
        diffs = [pm[hi][p] - pm[lo][p] for p in common]
        return np.mean(diffs, 0).astype(np.float32)

    def unit(v):
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v

    raw = per_prompt_diff("WIDE", "NARROW")
    d_width = unit(raw)
    sep = float(np.linalg.norm(
        np.mean([pm["WIDE"][p] for p in common], 0)
        - np.mean([pm["NARROW"][p] for p in common], 0)))

    norm = float(np.linalg.norm(d_width))
    assert abs(norm - 1.0) < 1e-4, f"nicht unit-norm: {norm}"
    assert d_width.shape == (TARGET["hidden_size"],), d_width.shape

    artefact = {
        **TARGET,
        "direction": "WIDE_minus_NARROW_L29_meanK",
        "source": "scratches/ternary_bonsai_px/capture_relay_dwidth.py "
                  "(seite13/15-Semantik: per-prompt diff, unit-norm, decode-time "
                  "capture, last-visit-per-forward dedup)",
        "n_prompts": len(common),
        "prompts": common,
        "max_new_tokens": max_new,
        "wide_routing": WIDE_ROUTING,
        "narrow_routing": NARROW_ROUTING,
        "ntoks": {a: ntoks[a] for a in ntoks},
        "mean_norm_WIDE_L29": float(np.mean([np.linalg.norm(pm['WIDE'][p]) for p in common])),
        "mean_norm_NARROW_L29": float(np.mean([np.linalg.norm(pm['NARROW'][p]) for p in common])),
        "sep_WIDE_NARROW_L29_meanK": sep,
        "norm": norm,
        "dwidth": d_width.tolist(),
    }
    return artefact, common, sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=16,
                    help="Divergenz-Budget pro Arm/Prompt (gemma: 200; hier "
                         "DequantBudget 4.9 s/token → 16 als Kompromiss)")
    ap.add_argument("--dry", action="store_true", help="kein Speichern des Artefakts")
    args = ap.parse_args()

    print("[cap] lade modell (ternary-px ACTIVE_MANIFOLD, dann WIDE/NARROW-Arme)", flush=True)
    signs, _ = rt.load_signs()
    fold = rt.FoldOps(signs, mode="signs_first")
    model = rt.build_and_load(fold)
    assert px.apply_px_patch(model, config_preset="ACTIVE_MANIFOLD")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rt.OUT_DIR)

    t0 = time.time()
    pm, ntoks = run(model, tok, args.max_new)
    artefact, common, sep = build_artifact(pm, ntoks, args.max_new)
    print(f"[cap] RICHTUNG sep={sep:.3f} norm={artefact['norm']:.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    if args.dry:
        print("[cap] --dry → nicht gespeichert", flush=True)
        return
    relay_dir = os.environ.get(
        "PX_RELAY_DIR",
        os.path.join(REPO, "px_manifolds"))   # = relay_inject._relay_dir() default
    os.makedirs(relay_dir, exist_ok=True)
    safe_id = TARGET["hf_id"].replace("/", "_")
    out_path = os.path.join(relay_dir, f"{safe_id}_relay_dwidth.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artefact, f)
    print(f"[cap] GESPEICHERT: {out_path}", flush=True)
    print("[cap] FERTIG", flush=True)


if __name__ == "__main__":
    main()