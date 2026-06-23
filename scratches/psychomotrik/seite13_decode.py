"""seite13_decode.py — Per-Layer width-decodability-decay Profil.

Für jede Schicht in [0,5,10,13,16,19,21,24,25]: width 4-class (BASELINE/
NARROW/DEFAULT/WIDE) + recur-only 3-class (NARROW/DEFAULT/WIDE), leave-one-
cell-out. Zeigt WO die width-Info lebt und stirbt → lokalisierter introspection-
failure (LESUNG14): info-loss (stirbt vorm Output) vs vocab-bottleneck (bis
Output da, aber Text nicht). Verwendet PCA-Reduktion auf 256 dims für Speed.
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite13_hidden")
LAYERS = [0, 5, 10, 13, 16, 19, 21, 24, 25]
ALL_ARMS = ["BASELINE", "NARROW", "DEFAULT", "WIDE"]
RECUR_ONLY = ["NARROW", "DEFAULT", "WIDE"]


def load_cells():
    cells = []
    for fn in sorted(os.listdir(HID)):
        if not fn.endswith(".pt"): continue
        cells.append(torch.load(os.path.join(HID, fn), weights_only=False))
    return cells


def _standardize(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def _pca_reduce(Xtr, Xte, k=256):
    # fit PCA on train, apply to test (unsupervised, no label leakage)
    mu = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    comps = Vt[:k].T  # [d, k]
    return (Xtr - mu) @ comps, (Xte - mu) @ comps


def logistic_fit_predict(Xtr, ytr, Xte, n_classes, steps=250, l2=1e-3, seed=0):
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


def leave_one_cell_out(cells, H_fn, label_fn, n_classes):
    idx = [(ci, label_fn(c)) for ci, c in enumerate(cells)]
    idx = [(ci, l) for ci, l in idx if l is not None]
    if len(idx) < 4: return None
    accs = []
    for held_ci, _ in idx:
        tr = [ci for ci, _ in idx if ci != held_ci]
        Xtr = np.concatenate([H_fn(cells[ci]) for ci in tr], 0)
        ytr = np.concatenate([np.full(H_fn(cells[ci]).shape[0], label_fn(cells[ci])) for ci in tr])
        Xte = H_fn(cells[held_ci])
        yte = np.full(Xte.shape[0], label_fn(cells[held_ci]))
        if len(set(ytr.tolist())) < 2: continue
        Xtr_s, mu, sd = _standardize(Xtr)
        Xte_s, _, _ = _standardize(Xte, mu, sd)
        Xtr_p, Xte_p = _pca_reduce(Xtr_s, Xte_s, k=256)
        probs = logistic_fit_predict(Xtr_p, ytr, Xte_p, n_classes, steps=250, seed=1)
        accs.append(float((probs.argmax(1) == yte).mean()))
    return float(np.mean(accs)) if accs else None


def main():
    cells = load_cells()
    print(f"[s13dec] {len(cells)} cells", file=sys.stderr)
    lab4 = {a: i for i, a in enumerate(ALL_ARMS)}
    lab3 = {a: i for i, a in enumerate(RECUR_ONLY)}
    lf4 = lambda c: lab4.get(c["arm"])
    lf3 = lambda c: lab3.get(c["arm"]) if c["arm"] in RECUR_ONLY else None

    results = {"layers": LAYERS, "width_4class_acc": {}, "recur_only_3class_acc": {}}
    for L in LAYERS:
        H_fn = lambda c, L=L: c["layers"][L].numpy().astype(np.float32)
        a4 = leave_one_cell_out(cells, H_fn, lf4, 4)
        a3 = leave_one_cell_out(cells, H_fn, lf3, 3)
        results["width_4class_acc"][str(L)] = a4
        results["recur_only_3class_acc"][str(L)] = a3
        print(f"[s13dec] L{L:2d}: width4={a4} recur3={a3}", file=sys.stderr)

    with open(os.path.join(OUT, "seite13_decode_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT, "seite13_decode_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Seite 13 Per-Layer width-decodability-decay ===\n")
        f.write("(chance: width4=0.25, recur3=0.333; PCA256, leave-one-cell-out)\n\n")
        f.write("layer | width_4class | recur_only_3class\n")
        f.write("------+--------------+-------------------\n")
        for L in LAYERS:
            a4 = results["width_4class_acc"][str(L)]
            a3 = results["recur_only_3class_acc"][str(L)]
            f.write(f"  L{L:2d} | {a4 if a4 is not None else 'skipped':>12} | {a3 if a3 is not None else 'skipped':>17}\n")
        f.write("\nLesart:\n")
        f.write("  width4 hoher Wert über alle Schichten → config-fingerprint global.\n")
        f.write("  decay vor L24/L25 → info-loss (Selbst-Zustand stirbt vorm Output).\n")
        f.write("  bleibt hoch bis L25, aber seite12 D4 text=0 → vocab-bottleneck (Info am Output da, Text nicht).\n")
    print("[s13dec] FERTIG", file=sys.stderr)


if __name__ == "__main__":
    main()