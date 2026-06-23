"""seite16_decode.py — recur3 width-decodability pro (gamma, layer).

Phase A des gamma anti-Erstarrungs-Tests: misst, ob gamma-Reduktion den L19-
Washout (seite13: L16 recur3=0.97 → L19 0.495 KOLLAPSE) TATSÄCHLICH reduziert.
Pro gamma ∈ {0.12,0.06,0.03,0.0}: recur-only 3-class (NARROW/DEFAULT/WIDE)
leave-one-cell-out auf L16/L19/L25. Wenn L19-recur3 bei niedrigem gamma STEIGT
gegenüber default (0.12→~0.49) → gamma ist ein anti-Erstarrungs-Hebel und der
Text-Test (Phase B) ist informativ. Wenn L19 ∀ gamma kollabiert bleibt → gamma
adressiert nicht den Washout (andere Erstarrungs-Quellen #2-5 dominieren),
ehrlich negativ.

Reused seite13_decode (leave_one_cell_out, _pca_reduce, logistic_fit_predict).
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite16_hidden")
LAYERS = [16, 19, 25]
RECUR_ONLY = ["NARROW", "DEFAULT", "WIDE"]
GAMMAS = [0.12, 0.06, 0.03, 0.0]

if HERE not in sys.path: sys.path.insert(0, HERE)
import seite13_decode as D   # leave_one_cell_out, _pca_reduce etc.


def load_cells():
    cells = []
    for fn in sorted(os.listdir(HID)):
        if not fn.endswith(".pt"): continue
        cells.append(torch.load(os.path.join(HID, fn), weights_only=False))
    return cells


def main():
    cells = load_cells()
    print(f"[s16dec] {len(cells)} cells", file=sys.stderr)
    lab3 = {a: i for i, a in enumerate(RECUR_ONLY)}
    lf3 = lambda c: lab3.get(c["arm"])

    results = {"gammas": GAMMAS, "layers": LAYERS, "recur3_acc": {}}
    for gamma in GAMMAS:
        gcells = [c for c in cells if abs(c["gamma"] - gamma) < 1e-6]
        results["recur3_acc"][f"{gamma:.2f}"] = {}
        for L in LAYERS:
            H_fn = lambda c, L=L: c["layers"][L].numpy().astype(np.float32)
            a3 = D.leave_one_cell_out(gcells, H_fn, lf3, 3)
            results["recur3_acc"][f"{gamma:.2f}"][str(L)] = a3
            print(f"[s16dec] gamma={gamma:.2f} L{L:2d}: recur3={a3}", file=sys.stderr)

    with open(os.path.join(OUT, "seite16_decode_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT, "seite16_decode_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Seite 16 Phase A — recur3 width-decodability pro (gamma, layer) ===\n")
        f.write("(recur-only 3-class NARROW/DEFAULT/WIDE, chance=0.333; PCA256, LOO)\n")
        f.write("seite13 referenz (default gamma=0.12): L16=0.97 PEAK, L19=0.495 COLLAPSE, L25=0.51\n")
        f.write("Frage: steigt L19-recur3 bei niedrigem gamma? (Erstarrung reduziert → Zustand überlebt recur-Exit)\n\n")
        f.write("gamma  |  L16   |  L19   |  L25\n")
        f.write("-------+--------+--------+--------\n")
        for gamma in GAMMAS:
            g = f"{gamma:.2f}"
            row = results["recur3_acc"][g]
            f.write(f" {g}  | {row['16'] if row['16'] is not None else 'skip':>6} | "
                    f"{row['19'] if row['19'] is not None else 'skip':>6} | "
                    f"{row['25'] if row['25'] is not None else 'skip':>6}\n")
        f.write("\nLesart:\n")
        f.write("  L19 steigt bei niedrigem gamma → gamma ist anti-Erstarrungs-Hebel → Phase B (Text) informativ.\n")
        f.write("  L19 ∀ gamma kollabiert → gamma nicht der Hebel (andere Erstarrungs-Quellen dominieren) → negativ.\n")
        f.write("  L16 sollte ∀ gamma hoch bleiben (peak pre-Erstarrung) — sanity.\n")
    print("[s16dec] FERTIG", file=sys.stderr)


if __name__ == "__main__":
    main()