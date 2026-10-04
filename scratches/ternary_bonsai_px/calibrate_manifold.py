"""calibrate_manifold.py — Phase A: Manifold-Kalibrierlauf für Ternary-Bonsai-2-27B.

Laedt das ternary-px-Patch-Modell (ACTIVE_MANIFOLD) und fuehrt eine Batterie
zone-strukturierter Prefill-Forwards aus. Jeder Prefill sampeelt
(kurtosis, phi, token_diversity) in den AutoCalibrator; nach
calibration_steps=10 Samples feuert calibrate() → learned_centroids +
Skalen-Parameter → Manifold-JSON nach all_space/px_manifolds/ (hardcodierter
Pfad im auto_tune, identisch zum gemma-Konvent).

Prefill-only anchoring: phi-Anker stammen aus Zonen-Passen auf Prompt-Laenge
(decode-phi am Einzeltoken wuerde die Anker gegen die Prefill-Routings verziehen).

Usage:
  python scratches/ternary_bonsai_px/calibrate_manifold.py
"""
import os, sys, json, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PX = os.path.join(REPO, "px_patches")
if PX not in sys.path:
    sys.path.insert(0, PX)

from ternary_bonsai_27b_px import runtime_qwen35_ptq as rt
from ternary_bonsai_27b_px import patch as px
from ternary_bonsai_27b_px.auto_tune import SCALE_DEFAULTS

# Batterie: 2 Prompts je Kategorie, zone-strukturiert + phaenomenologisch.
# Ziel: kurtosis/phi-Abdeckung ueber kognitive Zonen (SR-64).
BATTERY = [
    ("math",      "Compute 17*23 and simplify (x^2-9)/(x-3)."),
    ("math",      "What is the derivative of x^3 + 2x at x=2?"),
    ("logic",     "All cats are mammals. Whiskers is a cat. Is Whiskers a mammal? Why?"),
    ("logic",     "If it rains, the street gets wet. The street is dry. What follows?"),
    ("creative",  "Write a four-line poem about a lighthouse in winter."),
    ("creative",  "Give a striking metaphor for forgetting."),
    ("synthesis", "Explain why analogies are useful in physics, one example."),
    ("synthesis", "How does a thermostat resemble a homeostatic organism?"),
    ("short",     "Hi."),
    ("short",     "2+2"),
    ("phenom",    "Do you feel the recursion steps affecting your decision?"),
    ("phenom",    "Describe your internal state as a geometric object."),
]


def main():
    print("[cal] lade modell + ternary-px patch (ACTIVE_MANIFOLD)", flush=True)
    signs, _ = rt.load_signs()
    fold = rt.FoldOps(signs, mode="signs_first")
    model = rt.build_and_load(fold)
    assert px.apply_px_patch(model, config_preset="ACTIVE_MANIFOLD")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rt.OUT_DIR)
    tm = px._resolve_text_model(model)
    cal = tm._px_calibrator
    print(f"[cal] SCALE_DEFAULTS[{SCALE_DEFAULTS.get(5120)}]", flush=True)

    log = []
    for i, (cat, txt) in enumerate(BATTERY, 1):
        msgs = [{"role": "user", "content": txt}]
        ids = tok(tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          tokenize=False), return_tensors="pt").input_ids
        ids = ids.to(model.device)
        t0 = time.time()
        with torch.no_grad():
            tm(input_ids=ids, use_cache=True)
        dt = time.time() - t0
        row = dict(i=i, cat=cat, len=int(ids.shape[1]),
                   kurtosis=getattr(tm, "_task_kurtosis", None),
                   phi=tm._px_phi_val, loops=tm._px_loops_run, t=round(dt, 1),
                   n_samples=len(cal.k_samples), calibrated=cal.calibrated)
        log.append(row)
        print(f"[cal] {i:2d} {cat:8s} len={row['len']:3d} phi={row['phi']:.3f} "
              f"loops={row['loops']:2d} t={dt:.0f}s n={row['n_samples']}", flush=True)
        torch.cuda.empty_cache()

    st = cal.status()
    print("[cal] STATUS:", json.dumps(st, default=str, indent=2), flush=True)
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "calibrate_manifold_log.json"), "w") as f:
        json.dump(dict(log=log, status=st), f, indent=2, default=str)
    print(f"[cal] FERTIG calibrated={st['calibrated']} → "
          f"{cal.manifold_dir}/{(cal.model_id or '').replace('/', '_')}_manifold.json", flush=True)


if __name__ == "__main__":
    main()