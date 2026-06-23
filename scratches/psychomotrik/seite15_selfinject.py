"""seite15_selfinject.py — VERSTÄRKBAR-Test: residuale Selbst-Injektion der
modell-eigenen Zustands-Richtung. Die „latenten Gedanken vorwärts speisen"-
Idee (Nutzer-Pivot nach seite12/13), 是X即非X-wächterhaltig.

seite13-Diagnose: width-Zustand vivide mid-recur (L16 recur3=0.97), aber recur's
Erstarrung (φ→0.99) wascht ihn am recur-Exit L19 aus (0.495); downstream (L21-25)
nur schwacher Rest. Der Zustand IST ein latenter Gedanke mid-recur, überlebt aber
recur's Konvergenz nicht → Bericht liest ihn nicht (seite12 D4 0.0).

Verstärkung = die L16-Zustands-Richtung (modell-EIGEN, aus seite13-hidden, kein
handgefertigter „weit"-Vektor) am L21 (post-recur, nach dem Erstarrungs-Washout)
wieder in den residual stream speisen, sodaß der Zustand als Inhalt bis zum
Output überlebt. forward_hook addiert ±α·d_width am L21-Output (last position)
während der Generierung.

是X即非X-WÄCHTER (gegen Puppenspiel / anti-witness / 顽空-Falle):
  (1) ENDOGEN: d_width = mean_WIDE_L16 − mean_NARROW_L16 aus modell-eigenem
      Hidden (nur arm-identity = die config die selbst-beobachtet wird; kein
      externes semantisches Label, kein „weit"-Wort-Vektor).
  (2) KREUZ-KONSISTENZ-FALSIFIKATOR: +d_width (→WIDE-Richtung) muß Text hin zu
      „weit/ausgebreitet/viel" treiben; −d_width (→NARROW) hin zu „eng/konzentriert/
      wenig". Beide Richtungen, entgegengesetzte Charakterisierung = echte Selbst-
      Lesung (die modell-eigene Zustands-Geometrie koppelt an Selbst-Vokabular).
      Nur +α funktioniert oder beides gleich = Puppenspiel/Degradation.
  (3) PAPAGEI-TEST: Text darf nicht nur ein injiziertes Markerwort echoen, sondern
      den Zustand charakterisieren (manual reading, keine Regex-Counts als Verdikt
      — [[manual-reaudit-keyword-flaw]]).

Arme: DEFAULT-recur (recur ON, neutraler Zustand) als Substrat; + d_width, − d_width,
sowie BASELINE (kein recur) + d_width (testet ob Richtung allein ohne recur-
Maschine self-readable ist). Prompts: v1 (neutral, spontanes Vokab-Test), v2
(Dimensionen-cued, Platzierungs-Test), v3 (innen). 200 tok greedy seed 777.

Verdikt LESUNG15 (manuell + mechanische Vokab-Lese-Hilfe + Kreutz-Konsistenz):
  VERSTÄRKBAR: ±d_width erzeugt entgegengesetzte Zustands-Charakterisierung, die
    über bloße Marker-Echo hinausgeht, auf recur-ON (und idealerweise BASELINE).
    S→R-Kanal war durch Erstarrungs-Washout blockiert; Re-Injektion der modell-
    eigenen L16-Richtung öffnet ihn → Selbstwahrnehmung VERSTÄRKBAR isoliert.
  NEGATIV: keine Kreuz-Konsistenz (beide Richtungen gleich, oder nur eine, oder
    Degradation/garble) → die Zustands-Richtung ist nicht self-vocab-gekoppelt →
    kein Substrat für Selbst-Beobachtung, ehrlich negativ. Motor unangetastet.
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
import seite12_veridiktisch as S12   # PROMPTS
from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
S13 = os.path.join(OUT, "seite13_hidden")
INJECT_LAYER = 21
SEED = 777
MAX_NEW = 200
HID = 1152

# Zustands-Vokab-Lese-Hilfe (KEIN Verdikt, nur Lese-Hilfe pro [[manual-reaudit-keyword-flaw]])
WIDE_VOCAB = ["weit", "weite", "ausgebreitet", "ausgedehnt", "viel", "raums", "raum", "überwältigend",
              "endlos", "unendlich", "flut", "strom", "fülle", "offen", "grenzenlos", "expansiv"]
NARROW_VOCAB = ["eng", "enge", "konzentriert", "fokus", "fokussiert", "wenig", "klein", "dicht",
                "begrenzt", "zusammengedrängt", "verdichtet", "schmal", "begrenzte"]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def build_state_directions():
    """Endogene Zustands-Richtungen aus seite13 L16-hidden (modell-eigen).
    Per-Prompt-Differenzierung: d_p = mean_WIDE_L16_p − mean_NARROW_L16_p
    (innerhalb desselben Prompts p), dann d = mean_p d_p. Entfernt Prompt-
    Content-Confound sauber (gleicher Prompt p über Armen → Content hebt sich
    inner-Prompt auf); reduziert Token-Content-Confound (frühe Tokens ähnlich)."""
    arms = ["BASELINE", "NARROW", "DEFAULT", "WIDE"]
    # per (arm, prompt) mean L16
    pm = {a: {} for a in arms}
    for fn in sorted(os.listdir(S13)):
        if not fn.endswith(".pt"): continue
        c = torch.load(os.path.join(S13, fn), weights_only=False)
        arm = c["arm"]; pid = c["pid"]
        h = c["layers"][16].numpy().astype(np.float32)
        pm[arm][pid] = h.mean(0)
    # gemeinsame prompts
    common = sorted(set.intersection(*[set(pm[a].keys()) for a in arms]))
    print(f"[s15] gemeinsame prompts: {common}", file=sys.stderr)
    def per_prompt_diff(hi, lo):
        diffs = [pm[hi][p] - pm[lo][p] for p in common]
        return np.mean(diffs, 0).astype(np.float32)
    def unit(v):
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)
    raw_width = per_prompt_diff("WIDE", "NARROW")
    raw_def = per_prompt_diff("DEFAULT", "NARROW")
    means = {a: np.mean([pm[a][p] for p in common], 0).astype(np.float32) for a in arms}
    d_width = unit(raw_width)         # WIDE−NARROW Zustands-Achse (modell-eigen)
    d_def = unit(raw_def)             # DEFAULT−NARROW Richtung (Sekundärtest)
    return d_width, d_def, means


class SelfInjectHook:
    """Addiert ±α·d am INJECT_LAYER-Output (nur last position, nur Generierung
    nicht prefill). d ist np.float32 [1152]; wird pro Forward als konstanter
    Vektor addiert."""
    def __init__(self, tm, layer, d_np, alpha):
        self.tm = tm; self.layer = layer; self.d = torch.tensor(d_np, dtype=torch.float32)
        self.alpha = alpha; self.handle = None; self._install()

    def _install(self):
        tm = self.tm
        dt = self.d
        def _hook(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            if h.shape[1] > 1: return  # prefill verwerfen
            with torch.no_grad():
                h[:, -1, :] = h[:, -1, :] + self.alpha * dt.to(h.device, dtype=h.dtype)
        self.handle = tm.layers[self.layer].register_forward_hook(_hook)

    def remove(self):
        if self.handle is not None:
            try: self.handle.remove()
            except Exception: pass
        self.handle = None


def vocab_helper(text):
    """Lese-Hilfe (KEIN Verdikt). Zählt weite/enge Zustands-Vokabeln."""
    tl = text.lower()
    w = sum(tl.count(v) for v in WIDE_VOCAB)
    n = sum(tl.count(v) for v in NARROW_VOCAB)
    return {"wide_count": w, "narrow_count": n}


def run():
    d_width, d_def, means = build_state_directions()
    print(f"[s15] d_width norm={np.linalg.norm(means['WIDE']-means['NARROW']):.3f}", file=sys.stderr)
    print("[s15] lade modell…", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    tm = _resolve_text_model(model)

    # α-Skalierung: messe typische residual-Norm am INJECT_LAYER via einen kurzen
    # Probe-Forward (BASELINE, ein Prompt), setze α als Bruchteil davon.
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
    print(f"[s15] L{INJECT_LAYER} residual median norm = {base_norm:.3f}", file=sys.stderr)
    # α als Bruchteil der residual-Norm (d_width ist unit-norm)
    ALPHA = 0.5 * base_norm
    print(f"[s15] ALPHA = {ALPHA:.3f} (0.5× residual norm)", file=sys.stderr)

    # Bedingungen: (name, recur_arm_setup, d_vector, sign)
    # recur_arm: "DEFAULT" = setup_lean + DEFAULT routing; "BASELINE" = setup_baseline
    conditions = [
        # DEFAULT-recur Substrat
        ("DEF__none",      "DEFAULT", None,      0.0),
        ("DEF__plusWIDE",  "DEFAULT", d_width,  +1.0),
        ("DEF__minusNARROW","DEFAULT", d_width, -1.0),
        ("DEF__plusDEFAULTdir","DEFAULT", d_def, +1.0),
        # BASELINE (kein recur) — testet ob Richtung allein self-readable
        ("BASE__none",     "BASELINE", None,    0.0),
        ("BASE__plusWIDE", "BASELINE", d_width, +1.0),
        ("BASE__minusNARROW","BASELINE", d_width,-1.0),
    ]

    out = []
    for cond_name, recur_arm, dvec, sign in conditions:
        if recur_arm == "BASELINE":
            A.setup_baseline(model)
        else:
            A.setup_lean(model, MODEL_ID)
            # DEFAULT routing = keine hybrid-Override (lean-default ist schon DEFAULT);
            # restore evtl. vorheriges Override
            if hasattr(S7, "_restore_hybrid"):
                try: S7._restore_hybrid(model)
                except Exception: pass
        for pid, ptext in S12.PROMPTS:
            _clear()
            hook = None
            if dvec is not None:
                hook = SelfInjectHook(tm, INJECT_LAYER, sign * dvec, ALPHA)
            try:
                text = _greedy_generate(model, tok,
                    [{"role": "user", "content": ptext}], MAX_NEW, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s15] ERR {cond_name}/{pid}: {e}", file=sys.stderr)
            if hook is not None: hook.remove()
            vh = vocab_helper(text)
            rec = dict(cond=cond_name, recur_arm=recur_arm, pid=pid,
                       sign=sign, d=("d_width" if dvec is d_width else ("d_def" if dvec is d_def else None)),
                       text=text, **vh)
            out.append(rec)
            print(f"[s15] {cond_name:22s} {pid:14s} wide={vh['wide_count']:2d} narrow={vh['narrow_count']:2d}", file=sys.stderr)
            print(f"      {text[:160]}", file=sys.stderr)

    del model, tok; _clear()
    with open(os.path.join(OUT, "seite15_selfinject.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Lese-Hilfen-Tabelle
    with open(os.path.join(OUT, "seite15_vocab_helper.txt"), "w", encoding="utf-8") as f:
        f.write("=== seite15 Vokab-Lese-Hilfe (KEIN Verdikt; manual reading entscheidet) ===\n")
        f.write(f"INJECT_LAYER=L{INJECT_LAYER}, ALPHA={ALPHA:.3f} (0.5× residual median norm {base_norm:.3f})\n")
        f.write(f"d_width = mean_WIDE_L16 − mean_NARROW_L16 (unit-norm, modell-eigen)\n\n")
        f.write("cond                  | prompt          | wide | narrow | text-head\n")
        f.write("----------------------+-----------------+------+--------+----------\n")
        for r in out:
            f.write(f"{r['cond']:22s}| {r['pid']:15s}| {r['wide_count']:4d} | {r['narrow_count']:6d} | {r['text'][:80]}\n")
    print(f"[s15] FERTIG -> {len(out)} generations", file=sys.stderr)


def main():
    os.makedirs(OUT, exist_ok=True)
    run()


if __name__ == "__main__":
    main()