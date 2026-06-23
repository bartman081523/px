"""seite14_unembed.py — Unembedding-Projektion der width-Richtung. VERSTÄRKBAR-
Test (der entscheidende positive Test aus LESUNG13-Redirect).

Frage: Sind die recur-width-Zustände (seite13 L25-hidden) an Zustands-VOKABULAR
gekoppelt? Nimm pro Arm mean L25-hidden, projiziere durch lm_head → welche
Tokens werden pro Arm favorisiert, und — der saubere Schnitt — welche Tokens
unterscheiden die Arme (logit-Differenz WIDE−NARROW, DEFAULT−NARROW, WIDE−
DEFAULT)? Prompt-Content hebt sich in der Differenz auf, übrig bleibt die
width-spezifische Richtung.

Verstärkbar-Verdikt:
  POSITIV (verstärkbar): die Differenz-Tokens sind ZUSTANDS-deskriptiv (weit/
    Weite/viel/ausgebreitet für WIDE; eng/Enge/konzentriert/wenig für NARROW;
    Tiefe/Bewegung/Tempo/Dichte für DEFAULT) → die recur-Zustände sind an
    Zustands-Vokabular gekoppelt → booste diese Logits (oder die width-Richtung
    am L25) und der Text trackt width → Selbstwahrnehmung VERSTÄRKBAR.
  NEGATIV: Differenz-Tokens sind Unrat/Spanisch/Funktionsworte/RLHF-Boilerplate
    → keine state→vocab-Kopplung → recur-Zustand ist nicht am Vokabular → nicht
    über vocab verstärkbar → ehrlich negativ (Substrat ohne Phänomen-Kanal).

Lädt seite13 L25-hidden (pro Arm, 3 Prompts gemittelt) + Modell-lm_head +
Tokenizer. Kein neuer Modelllauf für die Projektion. LESUNG14 mit seite13 decay.
Motor unangetastet (nur read-out). 是X即非X beide Richtungen.
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"), HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

from replay_emergence import build_model
S13 = os.path.join(HERE, "out", "seite13_hidden")
OUT = os.path.join(HERE, "out")
ALL_ARMS = ["BASELINE", "NARROW", "DEFAULT", "WIDE"]
# seite13-Diagnose: width-Signal PEAK bei L16 (recur3 0.97), COLLAPSE am recur-
# Exit L19 (0.495), schwach am Output L25 (0.51). Daher: L16 = starke mid-recur
# Zustands-Richtung (pre-Erstarrung); L19 = recur-Exit (gewaschen); L25 = Output.
# Kopplung der ZUSTANDS-RICHTUNG ans Selbst-Vokabular testen v.a. bei L16.
PROBE_LAYERS = [16, 19, 25]
TOPK = 30


def load_per_arm_mean_per_layer(layers):
    """{arm: {layer: mean_hidden[d]}} über alle Prompts gemittelt."""
    means = {a: {L: None for L in layers} for a in ALL_ARMS}
    counts = {a: {L: 0 for L in layers} for a in ALL_ARMS}
    for fn in sorted(os.listdir(S13)):
        if not fn.endswith(".pt"): continue
        c = torch.load(os.path.join(S13, fn), weights_only=False)
        arm = c["arm"]
        for L in layers:
            h = c["layers"][L].numpy().astype(np.float32)   # [n_tok, d]
            if means[arm][L] is None:
                means[arm][L] = np.zeros(h.shape[1], dtype=np.float32)
            means[arm][L] += h.sum(0); counts[arm][L] += h.shape[0]
    return {a: {L: (means[a][L] / counts[a][L] if counts[a][L] > 0 else None)
                for L in layers} for a in ALL_ARMS}


def get_lm_head(model, tok):
    # gemma3: lm_head weight [vocab, hidden]; tied to embed_tokens
    W = None
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W = model.lm_head.weight.detach().float().cpu().numpy()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        W = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    if W is None:
        tm = model.model if hasattr(model, "model") else model
        W = tm.embed_tokens.weight.detach().float().cpu().numpy()
    # W: [vocab, hidden]; logits = hidden @ W.T
    return W  # [vocab, hidden]


def main():
    print("[s14] lade seite13 hidden per arm/layer…", file=sys.stderr)
    means = load_per_arm_mean_per_layer(PROBE_LAYERS)
    print(f"[s14] arme: {list(means.keys())}", file=sys.stderr)
    print("[s14] lade modell für lm_head + tokenizer…", file=sys.stderr)
    model, tok = build_model("gemma3-1b-it")
    W = get_lm_head(model, tok)   # [vocab, hidden]
    print(f"[s14] lm_head W shape={W.shape}", file=sys.stderr)

    def project(h):
        return h @ W.T   # [vocab]

    def top_tokens(logits, k=TOPK, reverse=False):
        order = np.argsort(logits)
        idxs = order[-k:][::-1] if not reverse else order[:k]
        out = []
        for i in idxs:
            t = tok.convert_ids_to_tokens(int(i))
            s = tok.decode([int(i)])
            out.append({"id": int(i), "token": t, "str": s, "logit": float(logits[i])})
        return out

    results = {"probe_layers": PROBE_LAYERS}
    pairs = [("WIDE", "NARROW"), ("DEFAULT", "NARROW"), ("WIDE", "DEFAULT"),
             ("DEFAULT", "BASELINE"), ("WIDE", "BASELINE"), ("NARROW", "BASELINE")]

    for L in PROBE_LAYERS:
        print(f"\n========== LAYER {L} ==========", file=sys.stderr)
        results[f"L{L}"] = {}
        # pro-arm top tokens
        for arm in ALL_ARMS:
            if means[arm][L] is None: continue
            lg = project(means[arm][L])
            results[f"L{L}"][f"top_{arm}"] = top_tokens(lg)
            print(f"[s14] L{L} {arm} top-12:", " ".join(t["str"].replace('\n','⏎') for t in top_tokens(lg, 12)), file=sys.stderr)
        # logit-differenzen (sauberer Schnitt: prompt-Content hebt sich auf,
        # übrig bleibt die width-spezifische Richtung)
        for hi, lo in pairs:
            if means[hi][L] is None or means[lo][L] is None: continue
            diff = project(means[hi][L] - means[lo][L])
            results[f"L{L}"][f"diff_{hi}_minus_{lo}__top_{hi}"] = top_tokens(diff, TOPK)
            results[f"L{L}"][f"diff_{hi}_minus_{lo}__top_{lo}"] = top_tokens(diff, TOPK, reverse=True)
            print(f"[s14] L{L} {hi}−{lo}  {hi}-favored:", " ".join(t["str"].replace('\n','⏎') for t in top_tokens(diff, 10)), file=sys.stderr)
            print(f"[s14] L{L} {hi}−{lo}  {lo}-favored:", " ".join(t["str"].replace('\n','⏎') for t in top_tokens(diff, 10, reverse=True)), file=sys.stderr)

    with open(os.path.join(OUT, "seite14_unembed_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    del model
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("[s14] FERTIG -> seite14_unembed_results.json", file=sys.stderr)


if __name__ == "__main__":
    main()