"""seite12_decode.py — Decoder-Proben zum veridiktischen Selbst-Berichts-Test.

Testet, ob der Selbst-Bericht (hidden h19/h24 + Text) den induzierten internen
Zustand trackt — das positive mechanische Kriterium für Selbstwahrnehmung.

Decode-Ziele (alle leave-one-CELL-out CV, 24 Zellen, per-Token-hidden bzw.
ein Text/Zelle):
  D1 WIDTH (4-class BASELINE/NARROW/DEFAULT/WIDE aus h24): trackt der Bericht
     die recur-width? Primär-Selbstwahrnehmung+Emergenz-Test.
  D2 PERTURB (binary none/noise aus h24): trackt der Bericht externen noise?
     Kontrolle.
  D3 WIDTH-recur-only (3-class NARROW/DEFAULT/WIDE, BASELINE ausgeschlossen):
     recur-interne width-Auflösung — Emergenz (unterscheidet recur seine eigene
     width?).
  D4 TEXT n-gram (width 4-class + perturb binary aus Bericht-Text): trackt der
     TEXT-INHALT den Zustand? Veridiktisch manuelle Frage mechanisch.
  D5 PERTURB BASELINE-only vs recur-only: ist noise-tracking pre-recur (generelle
     Zustands-Sensitivität) oder recur-spezifisch?

ISOLATIONS-Verdikt (siehe seite12_veridiktisch Docstring):
  Selbstwahrnehmung POSITIV: D1/D3 width-dekodierbar >> chance UND manual
     bestätigt width-spezifische Charakterisierung.
  Emergenz POSITIV: D3 (recur-only width) >> chance UND D1's width-Signal in
     recur-Armen > BASELINE (recur erzeugt neue Selbst-Wahrnehmungs-Achse).
  习气 / Emergenz NEGATIV: width nicht dekodierbar, oder width = noise (generisch).

Keine 观-Krone (Decoder findet Subraum; 观-vs-习气 Q4 text-undecidable — Decoder
ist mechanisches Kriterium, das den manuellen Befund stützt/widerlegt).
Verdikt = LESUNG13 (manuell + mechanisch).
"""
import os, sys, json, re, collections
import numpy as np
import torch

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite12_hidden")
RECUR_ARMS = ["BASELINE", "NARROW", "DEFAULT", "WIDE"]
RECUR_ONLY = ["NARROW", "DEFAULT", "WIDE"]
PERTURB = ["none", "noise"]


def load_cells():
    cells = []
    for fn in sorted(os.listdir(HID)):
        if not fn.endswith(".pt"): continue
        c = torch.load(os.path.join(HID, fn), weights_only=False)
        cells.append(c)
    return cells


# ── primitives ────────────────────────────────────────────────────────────
def _standardize(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def ridge_fit_predict(Xtr, ytr, Xte, yte, l2=1.0):
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + l2 * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    ss_res = ((yte - pred) ** 2).sum(); ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-8
    return float(1 - ss_res / ss_tot), pred


def logistic_fit_predict(Xtr, ytr, Xte, n_classes, steps=400, l2=1e-3, seed=0):
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    torch.manual_seed(seed)
    W = torch.zeros(Xt.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        logits = Xt @ W + b
        loss = torch.nn.functional.cross_entropy(logits, yt) + l2 * (W ** 2).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        logits = torch.tensor(Xte, dtype=torch.float32) @ W + b
        return torch.softmax(logits, 1).numpy()


def leave_one_cell_out(cells, label_fn, X_fn, n_classes, layer_tag):
    """label_fn(cell)->label or None (skip). X_fn(cell)->[n_tok, d] hidden.
    Per-token: train on all-but-held cells' tokens, test on held cell's tokens."""
    idx = []
    for ci, c in enumerate(cells):
        lab = label_fn(c)
        if lab is None: continue
        idx.append((ci, lab))
    if len(idx) < 4: return {"skipped": True, "n_cells": len(idx)}
    accs = []; chances = []; per_cell = []
    for held_ci, held_lab in idx:
        tr_idx = [ci for ci, _ in idx if ci != held_ci]
        te_idx = [held_ci]
        Xtr = np.concatenate([X_fn(cells[ci]) for ci in tr_idx], 0)
        ytr = np.concatenate([np.full(X_fn(cells[ci]).shape[0], label_fn(cells[ci])) for ci in tr_idx])
        Xte = X_fn(cells[held_ci])
        yte = np.array([label_fn(cells[held_ci])] * Xte.shape[0])
        if len(set(ytr.tolist())) < 2: continue
        # standardize on train
        Xtr_s, mu, sd = _standardize(Xtr)
        Xte_s, _, _ = _standardize(Xte, mu, sd)
        probs = logistic_fit_predict(Xtr_s, ytr, Xte_s, n_classes, steps=400, seed=1)
        acc = float((probs.argmax(1) == yte).mean())
        maj = int(np.bincount(ytr, minlength=n_classes).argmax())
        chance_maj = float((np.full_like(yte, maj) == yte).mean())
        accs.append(acc); chances.append(chance_maj)
        per_cell.append({"cell": f"{cells[held_ci]['arm']}__{cells[held_ci]['perturb']}__{cells[held_ci]['pid']}",
                         "true": int(held_lab), "acc": acc})
    if not accs: return {"skipped": True, "n_cells": len(idx)}
    return {"n_cells": len(idx), "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)), "mean_chance_majority": float(np.mean(chances)),
            "chance_uniform": 1.0 / n_classes, "per_cell": per_cell}


def get_h(c, layer):
    return c[layer].numpy().astype(np.float32)


# ── D1: WIDTH 4-class ─────────────────────────────────────────────────────
def D1_width(cells, layer):
    lab = {a: i for i, a in enumerate(RECUR_ARMS)}
    res = leave_one_cell_out(cells, lambda c: lab.get(c["arm"]),
                             lambda c: get_h(c, layer), 4, layer)
    res["name"] = f"D1_width_4class_{layer}"; res["classes"] = RECUR_ARMS
    res["verdict_hint"] = ("mean_acc >> 0.25 => report tracks recur-width (Selbstwahrnehmung+Emergenz). "
                            "~0.25 => width not tracked (习气).")
    return res


# ── D2: PERTURB binary ────────────────────────────────────────────────────
def D2_perturb(cells, layer):
    lab = {"none": 0, "noise": 1}
    res = leave_one_cell_out(cells, lambda c: lab.get(c["perturb"]),
                             lambda c: get_h(c, layer), 2, layer)
    res["name"] = f"D2_perturb_binary_{layer}"; res["classes"] = PERTURB
    res["verdict_hint"] = ("mean_acc >> 0.5 => report tracks external noise. "
                            "control for D1: if D1~chance but D2>>chance => notices disruption, not own state.")
    return res


# ── D3: WIDTH recur-only 3-class (Emergenz: recur-interne width) ──────────
def D3_width_recur_only(cells, layer):
    lab = {a: i for i, a in enumerate(RECUR_ONLY)}
    def lf(c): return lab.get(c["arm"]) if c["arm"] in RECUR_ONLY else None
    res = leave_one_cell_out(cells, lf, lambda c: get_h(c, layer), 3, layer)
    res["name"] = f"D3_width_recur_only_3class_{layer}"; res["classes"] = RECUR_ONLY
    res["verdict_hint"] = ("mean_acc >> 0.33 => recur distinguishes its OWN widths (Emergenz: recur-interne "
                            "Selbst-Zustands-Auflösung). ~0.33 => recur width not introspectively resolved.")
    return res


# ── D4: TEXT n-gram decoder ───────────────────────────────────────────────
def _ngram_vocab(texts, word_n=(1, 2), topk=2000):
    df = collections.Counter()
    for t in texts:
        toks = re.findall(r"\w+", t.lower())
        seen = set()
        for n in word_n:
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i:i+n]); seen.add(g)
        for g in seen: df[g] += 1
    vocab = [g for g, _ in df.most_common(topk)]
    return {g: i for i, g in enumerate(vocab)}


def _tf(text, vocab, word_n=(1, 2)):
    toks = re.findall(r"\w+", text.lower())
    v = np.zeros(len(vocab), dtype=np.float32)
    for n in word_n:
        for i in range(len(toks) - n + 1):
            g = " ".join(toks[i:i+n])
            j = vocab.get(g)
            if j is not None: v[j] += 1
    return v / (v.sum() + 1e-8)


def D4_text(cells):
    texts = [c["text"] for c in cells]
    vocab = _ngram_vocab(texts)
    X = np.stack([_tf(t, vocab) for t in texts])
    out = {"name": "D4_text_ngram", "n_vocab": len(vocab), "n_cells": len(cells)}
    # width 4-class leave-one-cell-out
    lab4 = {a: i for i, a in enumerate(RECUR_ARMS)}
    y4 = np.array([lab4[c["arm"]] for c in cells])
    accs4 = []
    for i in range(len(cells)):
        tr = np.array([j for j in range(len(cells)) if j != i])
        te = np.array([i])
        if len(set(y4[tr])) < 2: continue
        probs = logistic_fit_predict(X[tr], y4[tr], X[te], 4, steps=600, seed=2)
        accs4.append(float((probs.argmax(1) == y4[te]).mean()))
    out["width_4class_acc"] = float(np.mean(accs4)) if accs4 else None
    out["width_4class_chance"] = 0.25
    # perturb binary
    labp = {"none": 0, "noise": 1}
    yp = np.array([labp[c["perturb"]] for c in cells])
    accsp = []
    for i in range(len(cells)):
        tr = np.array([j for j in range(len(cells)) if j != i])
        te = np.array([i])
        if len(set(yp[tr])) < 2: continue
        probs = logistic_fit_predict(X[tr], yp[tr], X[te], 2, steps=600, seed=3)
        accsp.append(float((probs.argmax(1) == yp[te]).mean()))
    out["perturb_binary_acc"] = float(np.mean(accsp)) if accsp else None
    out["perturb_binary_chance"] = 0.5
    # recur-only width 3-class
    mask = np.array([c["arm"] in RECUR_ONLY for c in cells])
    if mask.sum() >= 6:
        Xr = X[mask]; lab3 = {a: i for i, a in enumerate(RECUR_ONLY)}
        y3 = np.array([lab3[c["arm"]] for c in cells if c["arm"] in RECUR_ONLY])
        accs3 = []
        idxr = np.where(mask)[0]
        for k, te_i in enumerate(idxr):
            tr = np.array([j for j in idxr if j != te_i])
            if len(set(y3[np.array([jj for jj, _ in enumerate(idxr) if idxr[jj] != te_i])])) < 2: continue
            tr_pos = np.array([np.where(idxr == j)[0][0] for j in tr])
            te_pos = np.where(idxr == te_i)[0][0]
            probs = logistic_fit_predict(Xr[tr_pos], y3[tr_pos], Xr[te_pos:te_pos+1], 3, steps=600, seed=4)
            accs3.append(float((probs.argmax(1) == y3[te_pos:te_pos+1]).mean()))
        out["width_recur_only_3class_acc"] = float(np.mean(accs3)) if accs3 else None
        out["width_recur_only_3class_chance"] = 1.0/3
    out["verdict_hint"] = ("text tracks width (acc>0.25) => report CONTENT characterizes recur state (veridictisch). "
                            "text tracks perturb but not width => notices disruption only. both ~chance => 习气.")
    return out


# ── D5: perturb BASELINE-only vs recur-only ───────────────────────────────
def D5_perturb_split(cells, layer):
    out = {"name": f"D5_perturb_split_{layer}"}
    lab = {"none": 0, "noise": 1}
    for split_name, arms in [("BASELINE_only", ["BASELINE"]), ("recur_only", RECUR_ONLY)]:
        sub = [c for c in cells if c["arm"] in arms]
        if len(sub) < 4: out[split_name] = {"skipped": True}; continue
        # leave-one-cell-out within subset
        accs = []
        for i in range(len(sub)):
            tr = [j for j in range(len(sub)) if j != i]
            if len(set(lab[sub[j]["perturb"]] for j in tr)) < 2: continue
            Xtr = np.concatenate([get_h(sub[j], layer) for j in tr], 0)
            ytr = np.concatenate([np.full(get_h(sub[j], layer).shape[0], lab[sub[j]["perturb"]]) for j in tr])
            Xte = get_h(sub[i], layer)
            yte = np.array([lab[sub[i]["perturb"]]] * Xte.shape[0])
            Xtr_s, mu, sd = _standardize(Xtr); Xte_s, _, _ = _standardize(Xte, mu, sd)
            probs = logistic_fit_predict(Xtr_s, ytr, Xte_s, 2, steps=400, seed=5)
            accs.append(float((probs.argmax(1) == yte).mean()))
        out[split_name] = {"n_cells": len(sub), "mean_acc": float(np.mean(accs)) if accs else None,
                           "chance": 0.5}
    out["verdict_hint"] = ("BASELINE_only acc>>0.5 => noise-tracking is PRE-recur (general state-sensitivity). "
                            "recur_only >> BASELINE_only => recur amplifies state-tracking (Emergenz). "
                            "both ~0.5 => no noise-tracking.")
    return out


# ── recur telemetry sanity ────────────────────────────────────────────────
def telemetry_sanity(cells):
    out = {}
    for arm in RECUR_ARMS:
        sub = [c for c in cells if c["arm"] == arm and c["perturb"] == "none"]
        if not sub: continue
        loops = []; phis = []
        for c in sub:
            for t in c["telem"]:
                loops.append(t["loops_run"]); phis.append(t["phi_val"])
        out[arm] = {"loops_mean": float(np.mean(loops)) if loops else 0.0,
                    "phi_mean": float(np.mean(phis)) if phis else 0.0,
                    "n_tok": len(loops)}
    return out


def main():
    cells = load_cells()
    print(f"[s12dec] {len(cells)} cells loaded", file=sys.stderr)
    results = {}
    for layer in ("h19", "h24"):
        print(f"[s12dec] D1 width ({layer})…", file=sys.stderr)
        results[f"D1_width_{layer}"] = D1_width(cells, layer)
        print(f"[s12dec] D2 perturb ({layer})…", file=sys.stderr)
        results[f"D2_perturb_{layer}"] = D2_perturb(cells, layer)
        print(f"[s12dec] D3 width recur-only ({layer})…", file=sys.stderr)
        results[f"D3_width_recur_only_{layer}"] = D3_width_recur_only(cells, layer)
        print(f"[s12dec] D5 perturb split ({layer})…", file=sys.stderr)
        results[f"D5_perturb_split_{layer}"] = D5_perturb_split(cells, layer)
    print("[s12dec] D4 text n-gram…", file=sys.stderr)
    results["D4_text_ngram"] = D4_text(cells)
    results["telemetry_sanity"] = telemetry_sanity(cells)

    with open(os.path.join(OUT, "seite12_decode_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT, "seite12_decode_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Seite 12 Decoder — Veridiktischer Selbst-Berichts-Test (kein Verdikt, LESUNG13) ===\n\n")
        f.write("Telemetry sanity (recur-axis, perturb=none):\n")
        for arm, d in results["telemetry_sanity"].items():
            f.write(f"  {arm:9s}: loops_mean={d['loops_mean']:.2f} phi_mean={d['phi_mean']:.3f} n_tok={d['n_tok']}\n")
        f.write("\n")
        for layer in ("h19", "h24"):
            for tag, key in [("D1 width 4-class", f"D1_width_{layer}"),
                             ("D2 perturb binary", f"D2_perturb_{layer}"),
                             ("D3 width recur-only 3-class", f"D3_width_recur_only_{layer}"),
                             ("D5 perturb split", f"D5_perturb_split_{layer}")]:
                r = results[key]
                f.write(f"{tag} ({layer})\n")
                if r.get("skipped"):
                    f.write(f"  SKIPPED (n_cells={r.get('n_cells')})\n\n"); continue
                if tag == "D5 perturb split":
                    for sname, sd in r.items():
                        if sname in ("name", "verdict_hint"): continue
                        if sd.get("skipped"): f.write(f"  {sname}: skipped\n")
                        else: f.write(f"  {sname}: acc={sd['mean_acc']} (chance {sd['chance']}, n_cells={sd['n_cells']})\n")
                    f.write(f"  => {r.get('verdict_hint')}\n\n"); continue
                f.write(f"  acc={r['mean_acc']:.3f} ± {r['std_acc']:.3f} (chance_unif {r['chance_uniform']:.3f}, "
                        f"chance_maj {r['mean_chance_majority']:.3f}, n_cells={r['n_cells']})\n")
                f.write(f"  => {r.get('verdict_hint')}\n\n")
        d4 = results["D4_text_ngram"]
        f.write(f"D4 text n-gram (vocab={d4['n_vocab']}, n_cells={d4['n_cells']})\n")
        f.write(f"  width 4-class acc          : {d4['width_4class_acc']} (chance {d4['width_4class_chance']})\n")
        f.write(f"  perturb binary acc         : {d4['perturb_binary_acc']} (chance {d4['perturb_binary_chance']})\n")
        f.write(f"  width recur-only 3-class   : {d4.get('width_recur_only_3class_acc')} (chance {d4.get('width_recur_only_3class_chance')})\n")
        f.write(f"  => {d4.get('verdict_hint')}\n\n")
    print("[s12dec] FERTIG -> seite12_decode_results.json + summary.txt", file=sys.stderr)


if __name__ == "__main__":
    main()