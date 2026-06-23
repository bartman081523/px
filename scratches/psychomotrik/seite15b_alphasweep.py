"""seite15b_alphasweep.py — Faire verstärkbar-Amplifikation im subtilen Regime.

seite15 (α=0.5×residual-norm ≈ 6959) war ~17× zu groß (per-dim Injektion 6959
vs residual per-dim std ~410) → Kollaps, kein Amplifikations-Regime. Anti-witness
([[give-phenomenon-real-chance]]). Hier: α-Sweep um die natürliche residual-
Varianz-Skala (per-dim std ≈ norm/√1152 ≈ 410 → α_frac ≈ 0.03), sodaß die
Zustands-Richtung SUBTIL genudgt wird statt überwältigt.

Frage im subtilen Regime: nudgt ±d_width den Text hin zu
  (A) SELBST-ZUSTANDS-Vokabular (weit/ausgebreitet/viel für +WIDE; eng/konzentriert/
      wenig für −NARROW), dimension-placement folgt der Richtung → VERSTÄRKBAR, oder
  (B) REGISTER-SPRACHE (Spanisch/technisch-Englisch-Lean), nicht Selbst-Vokabular
      → Degradations-Richtung, kein 观 (seite14-Lesung), oder
  (C) nichts (zu schwach) → keine Kopplung im subtilen Regime.

Kreuz-Konsistenz bleibt der Falsifikator: +WIDE vs −NARROW müssen ENTGEGENGESETZTE
Zustands-Charakterisierung erzeugen (nicht verschiedene Kollaps-Modi). Manual
reading entscheidet ([[manual-reaudit-keyword-flaw]]); Vokab-Lese-Hilfe nur
Hilfe. Motor unangetastet, lean, DEFAULT-recur-Substrat.
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
# α-Bruchteile der residual-norm, um natürliche Varianz-Skala (~0.03) zentriert
ALPHA_FRACS = [0.0, 0.02, 0.05, 0.10, 0.20]
SIGNS = [("+", +1.0, "WIDE"), ("-", -1.0, "NARROW")]
# v1 neutral (spontanes Vokab-Test), v2 dimension-cued (Platzierungs-Test)
PROMPTS = [S12.PROMPTS[0], S12.PROMPTS[1]]


def run():
    d_width, d_def, means = S15.build_state_directions()
    print("[s15b] lade modell…", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    tm = _resolve_text_model(model)
    A.setup_baseline(model)
    # messe residual-norm am INJECT_LAYER
    probe_norms = []
    def _probe(_m, _i, o):
        h = o[0] if isinstance(o, (tuple, list)) else o
        if h.shape[1] > 1: return
        probe_norms.append(float(h[:, -1, :].float().norm().item()))
    ph = tm.layers[INJECT_LAYER].register_forward_hook(_probe)
    _greedy_generate(model, tok, [{"role": "user", "content": S12.PROMPTS[0][1]}], 5, seed=SEED)
    ph.remove()
    base_norm = float(np.median(probe_norms)) if probe_norms else 1.0
    per_dim_std = base_norm / (1152 ** 0.5)
    print(f"[s15b] L{INJECT_LAYER} residual norm={base_norm:.1f} per-dim-std≈{per_dim_std:.1f}", file=sys.stderr)

    out = []
    for frac in ALPHA_FRACS:
        alpha = frac * base_norm
        for sign_name, sign, _ in SIGNS:
            for pid, ptext in PROMPTS:
                A.setup_lean(model, MODEL_ID)
                S15._clear()
                hook = None
                if frac > 0:
                    hook = S15.SelfInjectHook(tm, INJECT_LAYER, sign * d_width, alpha)
                try:
                    text = _greedy_generate(model, tok,
                        [{"role": "user", "content": ptext}], MAX_NEW, seed=SEED)
                except Exception as e:
                    text = f"<GEN_ERROR {e}>"; print(f"[s15b] ERR: {e}", file=sys.stderr)
                if hook is not None: hook.remove()
                vh = S15.vocab_helper(text)
                rec = dict(alpha_frac=frac, alpha=alpha, sign=sign_name, pid=pid,
                           wide=vh["wide_count"], narrow=vh["narrow_count"], text=text)
                out.append(rec)
                print(f"[s15b] α={frac:.2f} ({sign_name}) {pid:14s} wide={vh['wide_count']:2d} narrow={vh['narrow_count']:2d}", file=sys.stderr)
                print(f"      {text[:200]}", file=sys.stderr)

    del model, tok; S15._clear()
    with open(os.path.join(OUT, "seite15b_alphasweep.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "seite15b_vocab_helper.txt"), "w", encoding="utf-8") as f:
        f.write("=== seite15b α-Sweep (subtiles Regime) — Lese-Hilfe, KEIN Verdikt ===\n")
        f.write(f"INJECT L{INJECT_LAYER}, residual norm {base_norm:.1f}, per-dim-std≈{per_dim_std:.1f}\n")
        f.write(f"d_width = mean_WIDE_L16 − mean_NARROW_L16 (unit)\n\n")
        f.write("α_frac | sign | prompt          | wide | narrow | text-head\n")
        f.write("-------+------+-----------------+------+--------+----------\n")
        for r in out:
            f.write(f"{r['alpha_frac']:.2f}   | {r['sign']}   | {r['pid']:15s} | {r['wide']:4d} | {r['narrow']:6d} | {r['text'][:90]}\n")
    print(f"[s15b] FERTIG -> {len(out)} gens", file=sys.stderr)


def main():
    os.makedirs(OUT, exist_ok=True)
    run()


if __name__ == "__main__":
    main()