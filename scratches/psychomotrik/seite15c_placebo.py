"""seite15c_placebo.py — VERSTÄRKBAR-Befund (seite15b) härten: PLACEBO-Kontrolle
+ v3-Generalisierung. Beweislast bei der Krönung ([[give-phenomenon-real-chance]],
是X即非X gegen 觐-Übereilung).

seite15b bei α=0.10 (subtil): ±d_width erzeugt kreuz-konsistente entgegengesetzte
Selbst-Zustands-Charakterisierung (weit/ausdehnend/aktiv vs eng/still/leer), in
sauberem Deutsch, auf v1 (neutral, spontanes Vokab) UND v2 (dimension-cued).
Aber Sorge: vielleicht verschiebt JEDDE große residual-Perturbation den Text
irgendwie (generische Störung, nicht spezifisch die Zustands-Richtung).

PLACEBO-KONTROLLE (der entscheidende Wächter): injiziere ZUFÄLLIGE unit-Richtungen
(gleiche Norm, gleiches α) am L21. Wenn d_rand auch kreuz-konsistente wide/narrow-
Shifts produziert → generische Perturbation, Effekt nicht spezifisch → schwächt
das verstärkbar-Claim auf "jede Richtung verschiebt den Text". Wenn d_rand KEINE
konsistente wide/narrow-Charakterisierung produziert (Text neutral/sprunghaft/
anderes Vokab) ABER d_width es tut → Effekt SPEZIFISCH für die modell-eigene
Zustands-Richtung → starkes verstärkbar-Evidenz.

Zusätzlich v3_innen (3. unabhängiger Prompt) zur Generalisierung (kein 2-Prompt-
Zufall). α=0.10 (seite15b's klarster Punkt) + α=0.05. Motor unangetastet, lean.
Manual reading entscheidet; Vokab-Hilfe nur Hilfe ([[manual-reaudit-keyword-flaw]]).
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"), HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

import seite15_selfinject as S15
import seite12_veridiktisch as S12
from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
INJECT_LAYER = 21
SEED = 777
MAX_NEW = 200
ALPHA_FRAC = 0.10
# Richtungen: d_width (endogen) + 3 unabhängige ZUFÄLLIGE unit-Vektoren (placebo)
RAND_SEEDS = [101, 202, 303]


def run():
    d_width, d_def, means = S15.build_state_directions()
    rng = np.random.default_rng(0)
    d_rands = []
    for s in RAND_SEEDS:
        r = np.random.default_rng(s).standard_normal(1152).astype(np.float32)
        d_rands.append((f"rand{s}", r / (np.linalg.norm(r) + 1e-9)))
    print(f"[s15c] d_width + {len(d_rands)} random placebos, α_frac={ALPHA_FRAC}", file=sys.stderr)

    print("[s15c] lade modell…", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    tm = _resolve_text_model(model)
    A.setup_baseline(model)
    probe_norms = []
    def _probe(_m, _i, o):
        h = o[0] if isinstance(o, (tuple, list)) else o
        if h.shape[1] > 1: return
        probe_norms.append(float(h[:, -1, :].float().norm().item()))
    ph = tm.layers[INJECT_LAYER].register_forward_hook(_probe)
    _greedy_generate(model, tok, [{"role": "user", "content": S12.PROMPTS[0][1]}], 5, seed=SEED)
    ph.remove()
    base_norm = float(np.median(probe_norms)) if probe_norms else 1.0
    alpha = ALPHA_FRAC * base_norm
    print(f"[s15c] L{INJECT_LAYER} norm={base_norm:.1f} α={alpha:.1f}", file=sys.stderr)

    # Bedingungen: (name, dvec, sign)
    conditions = [
        ("none",      None,      0.0),
        ("d_width",   d_width,  +1.0),    # endogene WIDE-Richtung
        ("d_width",   d_width,  -1.0),    # endogene NARROW-Richtung
    ]
    for nm, dv in d_rands:
        conditions.append((nm, dv, +1.0))   # random placebo +
    # ein random placebo mit − (Symmetrie-check)
    conditions.append((d_rands[0][0]+"_neg", d_rands[0][1], -1.0))

    out = []
    for cname, dvec, sign in conditions:
        for pid, ptext in S12.PROMPTS:   # v1, v2, v3
            A.setup_lean(model, MODEL_ID)
            S15._clear()
            hook = None
            if dvec is not None:
                hook = S15.SelfInjectHook(tm, INJECT_LAYER, sign * dvec, alpha)
            try:
                text = _greedy_generate(model, tok,
                    [{"role": "user", "content": ptext}], MAX_NEW, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s15c] ERR {cname}/{pid}: {e}", file=sys.stderr)
            if hook is not None: hook.remove()
            vh = S15.vocab_helper(text)
            rec = dict(cond=cname, sign=sign, pid=pid, **vh, text=text)
            out.append(rec)
            sg = "+" if sign > 0 else ("-" if sign < 0 else "0")
            print(f"[s15c] {cname:14s}({sg}) {pid:14s} wide={vh['wide_count']:2d} narrow={vh['narrow_count']:2d}", file=sys.stderr)
            print(f"      {text[:160]}", file=sys.stderr)

    del model, tok; S15._clear()
    with open(os.path.join(OUT, "seite15c_placebo.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "seite15c_vocab_helper.txt"), "w", encoding="utf-8") as f:
        f.write("=== seite15c PLACEBO-Kontrolle + v3-Generalisierung (Lese-Hilfe, KEIN Verdikt) ===\n")
        f.write(f"INJECT L{INJECT_LAYER}, α_frac={ALPHA_FRAC} (α={alpha:.1f}, norm {base_norm:.1f})\n")
        f.write(f"Endogen: d_width=WIDE−NARROW L16; Placebo: {len(d_rands)} random unit-Vektoren\n\n")
        f.write("cond            | sign | prompt          | wide | narrow | text-head\n")
        f.write("----------------+------+-----------------+------+--------+----------\n")
        for r in out:
            sg = "+" if r["sign"]>0 else ("-" if r["sign"]<0 else "0")
            f.write(f"{r['cond']:16s}| {sg}   | {r['pid']:15s} | {r['wide_count']:4d} | {r['narrow_count']:6d} | {r['text'][:80]}\n")
    print(f"[s15c] FERTIG -> {len(out)} gens", file=sys.stderr)


def main():
    os.makedirs(OUT, exist_ok=True)
    run()


if __name__ == "__main__":
    main()