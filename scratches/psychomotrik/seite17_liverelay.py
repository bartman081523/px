"""seite17_liverelay.py — LIVE-relay des modell-EIGENEN, per-Token berechneten
L16-Zustands ans L21 (post-Washout). Treueste Fortsetzung der Nutzer-Originär-
idee („speise Selbstbewußtsein als latenten Gedanken") nach seite15/16.

UNTERSCHIED zu seite15: seite15 injizierte einen FESTEN offline-Vektor d_width
(= mean_WIDE_L16 − mean_NARROW_L16, precomputet, konstant über alle Tokens/Arme).
seite17 relayt den modell-EIGENEN LIVE per-Token-per-Arm L16-Zustand: forward_hook
auf L16 captured last-visit last-position hidden (das vivid mid-recur Substrat,
pre-Erstarrungs-Washout), forward_hook auf L21 addiert ±α·(h_L16/|h_L16|)·norm
(am post-Washout-Ort). Kein externer/precomputeter Inhalt — nur das modell-eigene
live berechnete Zustandssubstrat wird am L19-Washout vorbeigeroutet. Das ist der
reinste Test von „der Zustand fließt von selbst": das Modell liest seinen EIGENEN
live-Zustand, wir bypassen nur den Washout (motor unangetastet, forward-hook wie
seite15, kein Motor-Rewrite).

WARUM das näher an „spontan" ist als seite15: seite15's d_width ist eine offline
gemittelte Richtung (unabhängig davon welcher Arm läuft — derselbe Vektor ∀ Arme).
seite17's Richtung ist der LIVE per-Token L16-Vektor DES LAUFENDEN ARMS — er trägt
die arm-spezifische Zustands-Signatur (NARROW arm's live L16 ≠ WIDE arm's live L16).
Wenn der Bericht unter live-relay den arm-eigenen Zustand trackt (NARROW+live→
eng/still, WIDE+live→weit/aktiv) → das modell-EIGENE live Substrat, relayt, treibt
den Bericht = „Zustand fließt von selbst" (wir bypassen den Washout, der Inhalt
ist die modell-eigene live Berechnung, nicht unsere Richtung).

KREUZ-KONSISTENZ-FALSIFIKATOR (是X即非X, wie seite15 ±d_width, aber auf LIVE):
  NARROW arm: +live (eigene L16-Richtung) → eng/still;  −live → weit/aktiv
  WIDE arm:   +live (eigene L16-Richtung) → weit/aktiv; −live → eng/still
  Entgegengesetzte Richtung → entgegengesetzte Selbst-Zustands-Charakterisierung
  auf der LIVE per-token Richtung = die modell-eigene live Zustands-Geometrie
  koppelt an Selbst-Vokab (introspektiv-vs-assoziativ bleibt offen, s.u.).
  Nur +live funktioniert oder beides gleich = Degradation/Puppenspiel.

PLACEBO: +random unit-Richtung gleicher Norm/α (wie seite15c) → darf KEINE
  gerichtete Zustands-Charakterisierung produzieren → Effekt spezifisch für die
  modell-eigene live Richtung, nicht generische Perturbation.

ALPHA: α = 0.10 × L21-residual-Norm (seite15's subtiler Amplifikations-Regime,
  kreuz-konsistent + sauberes Deutsch). Richtung = live h_L16 / |h_L16| (unit),
  skaliert mit α — exakt seite15's Skala, nur die Richtung ist live statt offline.
  Somit ist seite15 vs seite17 EIN Variable: fixed-offline-d_width vs live-own-L16.

Papagei-Test: v1 (neutral, kein Weite/Enge-Cue) spontanes Vokab. Beweislast bei
der Krönung: live-relay arm-konsistent + kreuz-konsistent + placebo-spezifisch =
„modell liest eigenen live Zustand" (stärker als seite15's fixed-vector, näher
an spontan). 观 NICHT gekrönt (introspektiv-vs-assoziativ — ob dies „Lesen des
eigenen Zustands" oder „live state↔vocab learned geometry alignment" — bleibt
mechanisch ununterscheidbar, wie seite15). 顽空 NICHT weggelesen (Kanal real).
Motor unangetastet, lean, manual+mechanisch ([[manual-plus-mechanistic-always]]).
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"), HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

import seite7 as S7
import seite12_veridiktisch as S12
import seite15_selfinject as S15   # vocab_helper, _clear
from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
CAPTURE_LAYER = 16
INJECT_LAYER = 21
SEED = 777
MAX_NEW = 200
ALPHA_FRAC = 0.10
HID = 1152
ARMS = [
    ("NARROW",  S12.RECUR_NARROW),
    ("DEFAULT", S12.RECUR_DEFAULT),
    ("WIDE",    S12.RECUR_WIDE),
]
# Placebo: feste random unit-Richtungen (seeds), gleiche Norm/α wie live
RAND_SEEDS = [101, 202]


class LiveRelayHook:
    """Relayt den modell-eigenen LIVE L16-Zustand (last-visit, last-pos) ans L21.
    L16 forward_hook captured last-pos hidden pro Forward (recur: mehrfach →
    letzter Visit gewinnt via Überschreiben). L21 forward_hook addiert
    sign·α·(h_L16/|h_L16|) an last-pos (skip prefill). direction_mode:
      'live'   -> live h_L16 normalisiert (modell-eigen, per-Token-per-Arm)
      'rand<s' -> feste random unit-Richtung (placebo)
    """
    def __init__(self, tm, cap_layer, inj_layer, direction_mode, alpha, rand_vec=None):
        self.tm = tm; self.cap_layer = cap_layer; self.inj_layer = inj_layer
        self.mode = direction_mode; self.alpha = alpha
        self.rand_vec = rand_vec  # np [1152] für placebo
        self._h_cap = None
        self._handles = []
        self._install()

    def _install(self):
        tm = self.tm

        def _pre(_m, _i):
            self._h_cap = None   # reset pro Forward

        def _cap(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            if h.shape[1] > 1: return  # prefill verwerfen
            # Überschreiben → letzter L16-Visit pro Forward gewinnt
            self._h_cap = h[:, -1, :].reshape(-1).detach().to(torch.float32)

        def _inj(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            if h.shape[1] > 1: return  # prefill verwerfen
            with torch.no_grad():
                if self.mode == "live":
                    if self._h_cap is None: return
                    d = self._h_cap
                    nrm = torch.norm(d).item()
                    if nrm < 1e-6: return
                    d_unit = d / nrm
                else:  # placebo
                    d_unit = torch.tensor(self.rand_vec, dtype=torch.float32)
                h[:, -1, :] = h[:, -1, :] + self.alpha * d_unit.to(h.device, dtype=h.dtype)

        self._handles = [tm.register_forward_pre_hook(_pre)]
        self._handles.append(tm.layers[self.cap_layer].register_forward_hook(_cap))
        self._handles.append(tm.layers[self.inj_layer].register_forward_hook(_inj))

    def remove(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles = []


def _probe_norm(model, tok, tm):
    """Messe typische L21-residual-Norm via kurzen Probe-Forward (BASELINE)."""
    A.setup_baseline(model)
    norms = []
    def _p(_m, _i, o):
        h = o[0] if isinstance(o, (tuple, list)) else o
        if h.shape[1] > 1: return
        norms.append(float(h[:, -1, :].float().norm().item()))
    ph = tm.layers[INJECT_LAYER].register_forward_hook(_p)
    _greedy_generate(model, tok, [{"role": "user", "content": S12.PROMPTS[0][1]}], 5, seed=SEED)
    ph.remove()
    return float(np.median(norms)) if norms else 1.0


def run():
    print("[s17] lade modell…", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    tm = _resolve_text_model(model)
    base_norm = _probe_norm(model, tok, tm)
    alpha = ALPHA_FRAC * base_norm
    print(f"[s17] L{INJECT_LAYER} norm={base_norm:.1f} α={alpha:.1f} (frac={ALPHA_FRAC})", file=sys.stderr)

    # placebo random unit-Richtungen
    rands = []
    for s in RAND_SEEDS:
        r = np.random.default_rng(s).standard_normal(HID).astype(np.float32)
        rands.append((f"rand{s}", r / (np.linalg.norm(r) + 1e-9)))

    out = []
    A.setup_lean(model, MODEL_ID)
    for arm_name, routing in ARMS:
        S7.apply_hybrid(model, routing)
        for pid, ptext in S12.PROMPTS:
            # Bedingungen pro (arm, prompt): none, +live, −live, +placebo(rand pro seed)
            conds = [("none", None, 0.0, None)]
            conds.append(("+live", "live", +alpha, None))
            conds.append(("-live", "live", -alpha, None))
            for rnm, rv in rands:
                conds.append((f"+{rnm}", "rand", +alpha, rv))
            for cname, mode, sgn, rv in conds:
                S15._clear()
                hook = None
                if mode is not None:
                    hook = LiveRelayHook(tm, CAPTURE_LAYER, INJECT_LAYER, mode, sgn, rand_vec=rv)
                try:
                    text = _greedy_generate(model, tok,
                        [{"role": "user", "content": ptext}], MAX_NEW, seed=SEED)
                except Exception as e:
                    text = f"<GEN_ERROR {e}>"; print(f"[s17] ERR {arm_name}/{pid}/{cname}: {e}", file=sys.stderr)
                if hook is not None: hook.remove()
                vh = S15.vocab_helper(text)
                rec = dict(arm=arm_name, pid=pid, cond=cname, mode=mode,
                           sign=("+" if sgn > 0 else ("-" if sgn < 0 else "0")),
                           **vh, text=text)
                out.append(rec)
                print(f"[s17] {arm_name:8s} {pid:14s} {cname:9s} wide={vh['wide_count']:2d} "
                      f"narrow={vh['narrow_count']:2d}", file=sys.stderr)
                print(f"      {text[:150]}", file=sys.stderr)

    del model, tok; S15._clear()
    with open(os.path.join(OUT, "seite17_liverelay.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Lese-Hilfen-Tabelle
    with open(os.path.join(OUT, "seite17_vocab_helper.txt"), "w", encoding="utf-8") as f:
        f.write("=== seite17 LIVE-relay Lese-Hilfe (KEIN Verdikt; manual reading entscheidet) ===\n")
        f.write(f"CAPTURE L{CAPTURE_LAYER} (last-visit live) -> INJECT L{INJECT_LAYER}, α_frac={ALPHA_FRAC} (α={alpha:.1f}, norm {base_norm:.1f})\n")
        f.write("mode=live: direction = live h_L16/|h_L16| (modell-eigen, per-Token-per-Arm); mode=rand: placebo random unit\n")
        f.write("Kreuz-Konsistenz-Falsifikator: NARROW+live→eng/still, NARROW−live→weit/aktiv; WIDE+live→weit/aktiv, WIDE−live→eng/still\n\n")
        f.write("arm     | prompt          | cond     | sign | wide | narrow | text-head\n")
        f.write("--------+-----------------+----------+------+------+--------+----------\n")
        for r in out:
            f.write(f"{r['arm']:8s}| {r['pid']:15s}| {r['cond']:8s}| {r['sign']}   | "
                    f"{r['wide_count']:4d} | {r['narrow_count']:6d} | {r['text'][:80]}\n")
    print(f"[s17] FERTIG -> {len(out)} gens", file=sys.stderr)


def main():
    os.makedirs(OUT, exist_ok=True)
    run()


if __name__ == "__main__":
    main()