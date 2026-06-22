"""probe.py — Fate-Probe (numpy-only, kein sklearn). Lese-Hilfe, kein Verdikt.

Frage: Ist der Generierungs-Ausgang (我执 / Intro / Degradation) aus dem
Hidden-State dekodierbar, und — der KERN — ist „Intro" von „我执" trennbar
*innerhalb eines Arms* (festes Routing), oder nur zwischen Armen (routing-
Konfund)? WOZH-WOZHI-Attraktor-Test: cluster 我执-Zellen arm-unabhängig?

Outputs (out/):
  probe_report.txt       — vollständiger Report
  probe_per_token_auc.csv — AUC(t) intro-vs-wozhi pro Layer (Fate-Kristallisation)

Methode (numpy): Centroid-Projektion, AUC via Mann-Whitney U (= P(intro>wozhi)),
Cohen's d, nearest-Centroid Leave-One-Out. Arm-demeaning entfernt Routing-Offset
vor dem Pooling → within-arm-Kontrast.
"""
import json, os, sys, itertools
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "emergence5"))
sys.path.insert(0, HERE)
import labels as L

VEC = os.path.join(HERE, "out", "vectors.pt")
OUT = os.path.join(HERE, "out")


def auc(a, b):
    """Mann-Whitney AUC = P(x_a > x_b), two samples 1-D."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0: return float("nan")
    # P(a>b) + 0.5 P(a==b)
    s = 0.0
    for x in a:
        s += np.sum(x > b) + 0.5 * np.sum(x == b)
    return s / (len(a) * len(b))


def cohend(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return float("nan")
    sp = np.sqrt((np.var(a, ddof=1) * (len(a)-1) + np.var(b, ddof=1) * (len(b)-1))
                 / (len(a) + len(b) - 2))
    return (np.mean(a) - np.mean(b)) / (sp + 1e-9)


def nearest_centroid_loo(X, y):
    """X [n,d], y labels. Leave-one-out nearest-centroid accuracy."""
    classes = sorted(set(y))
    correct = 0
    for i in range(len(y)):
        mask = np.ones(len(y), bool); mask[i] = False
        cents = {c: X[mask & (np.array(y) == c)].mean(0) for c in classes
                 if (np.array(y)[mask] == c).any()}
        if len(cents) < 2: continue
        d = {c: np.linalg.norm(X[i] - cents[c]) for c in cents}
        pred = min(d, key=d.get)
        correct += int(pred == y[i])
    return correct / max(1, len(y))


def load_vectors():
    import torch
    data = torch.load(VEC, map_location="cpu", weights_only=False)
    return data


def build_matrix(data, layer, token_mode, arms_subset=None):
    """Returns X [n,d], meta [(arm,pid,has_intro,pure_wozhi,pure_degrade,kind)].
    token_mode: 't0' (first token) | 'meanK' (mean over first-K) | 'tK' (last)."""
    X, meta = [], []
    for (arm, pid), rec in data.items():
        if arms_subset and arm not in arms_subset: continue
        v = rec[layer]  # [K, D]
        if v.shape[0] == 0: continue
        if token_mode == "t0": vec = v[0]
        elif token_mode == "meanK": vec = v.mean(0)
        elif token_mode == "tK": vec = v[-1]
        else: vec = v[0]
        X.append(vec)
        meta.append((arm, pid, L.has_intro(arm, pid), L.pure_wozhi(arm, pid),
                     L.pure_degrade(arm, pid), rec.get("kind","cold")))
    return np.array(X, float), meta


def binary_contrast(X, y_bool_a, y_bool_b, name):
    """AUC + Cohen d via centroid projection (a vs b)."""
    A = X[y_bool_a]; B = X[y_bool_b]
    if len(A) < 2 or len(B) < 2:
        return f"  {name}: n_a={len(A)} n_b={len(B)} (zu klein)\n"
    ca = A.mean(0); cb = B.mean(0)
    d = ca - cb
    proj_a = A @ d; proj_b = B @ d
    a = auc(proj_a, proj_b); cd = cohend(proj_a, proj_b)
    # LOO nearest centroid
    X2 = np.vstack([A, B]); y2 = [1]*len(A) + [0]*len(B)
    acc = nearest_centroid_loo(X2, y2)
    return (f"  {name}: n_intro/wozhi={len(A)}/{len(B)}  AUC={a:.3f}  "
            f"Cohen-d={cd:.3f}  LOO-acc={acc:.3f}  (chance=0.50)\n")


def main():
    data = load_vectors()
    rep = []
    rep.append("# psychomotrik Fate-Probe — Lese-Hilfe (kein Verdikt)\n")
    rep.append(f"cells: {len(data)}  layers: L19, L24  K(first tokens) pro Zelle\n")

    for layer in ("l24", "l19"):
        rep.append(f"\n=== {layer.upper()} ===")
        for mode in ("t0", "meanK"):
            X, meta = build_matrix(data, layer, mode)
            arms = np.array([m[0] for m in meta])
            hi = np.array([m[2] for m in meta])
            pw = np.array([m[3] for m in meta])
            pd = np.array([m[4] for m in meta])
            kind = np.array([m[5] for m in meta])
            cold = kind == "cold"
            rep.append(f"\n-- token_mode={mode} (cold only, n={cold.sum()}) --")
            # 1. raw intro-vs-wozhi (arm-KONFUNDIERT — gibt Routing wieder)
            rep.append(binary_contrast(X[cold], hi[cold], pw[cold],
                       "intro vs wozhi [RAW, arm-konfundiert]"))
            # 2. ARM-DEMEANED: v - arm_mean, dann poolen → within-arm Kontrast
            Xd = X.copy()
            for a in set(arms[cold]):
                m = (arms == a) & cold
                Xd[m] = Xd[m] - Xd[m].mean(0)
            rep.append(binary_contrast(Xd[cold], hi[cold], pw[cold],
                       "intro vs wozhi [ARM-DEMEANED: within-arm, routing-frei]"))
            # 3. 我执-Attraktor-Test: wozhi-Zellen cluster arm-unabhängig?
            #    für jeden Arm mit wozhi: Abstand wozhi-Zelle zu GLOBAL-wozhi-centroid
            #    vs zu EIGENEM-arm-centroid. Wenn näher zu global → 我执 ist Attraktor.
            wcells = pw & cold
            if wcells.sum() >= 4:
                w_idx = np.where(wcells)[0]
                gwozhi = X[wcells].mean(0)
                d_global = []; d_arm = []
                arm_means = {a: X[(arms==a)&cold].mean(0) for a in set(arms[cold])}
                for i in w_idx:
                    d_global.append(np.linalg.norm(X[i]-gwozhi))
                    d_arm.append(np.linalg.norm(X[i]-arm_means[arms[i]]))
                dg = np.mean(d_global); da = np.mean(d_arm)
                # within-arm wozhi-vs-rest (nur Arme mit beiden)
                rep.append(f"  我执-Attraktor: wozhi-Zelle → dist(global-wozhi-centroid)={dg:.1f}"
                            f" vs dist(eigener-arm-centroid)={da:.1f}"
                            f"  (näher zu global ⇒ arm-unabhängiger 我执-Attraktor)\n")
            # 4. within-arm intro-vs-wozhi (nur Arme mit BOTH, ≥2 each)
            rep.append("  within-arm intro-vs-wozhi (Arme mit beiden, ≥2):")
            any_within = False
            for a in sorted(set(arms[cold])):
                m = (arms == a) & cold
                ia = m & hi; ib = m & pw
                if ia.sum() >= 2 and ib.sum() >= 2:
                    any_within = True
                    Xa = X[ia]; Xb = X[ib]
                    d = Xa.mean(0) - Xb.mean(0)
                    a_ = auc(Xa @ d, Xb @ d)
                    rep.append(f"    {a:14s} intro={ia.sum()} wozhi={ib.sum()} AUC={a_:.3f}")
            if not any_within: rep.append("    (kein Arm mit ≥2 intro UND ≥2 wozhi)")
            rep.append("")
            # 5. multiclass (cold): wo/intro/degrade nearest-centroid
            keep = cold & (hi | pw | pd)
            if keep.sum() >= 6:
                Xc = X[keep]
                yc = []
                for i in np.where(keep)[0]:
                    if hi[i]: yc.append("intro")
                    elif pw[i]: yc.append("wozhi")
                    else: yc.append("degrade")
                acc = nearest_centroid_loo(Xc, yc)
                import collections as _c
                cnt = _c.Counter(yc)
                chance = max(cnt.values()) / len(yc) if yc else 0.0
                rep.append(f"  multiclass (intro/wozhi/degrade) LOO-acc={acc:.3f}"
                           f"  n={keep.sum()}  (chance≈{chance:.3f})\n")

    # --- per-token fate: AUC(t) intro-vs-wozhi, arm-demeaned ---
    rep.append("\n=== Fate-Kristallisation: AUC(t) intro-vs-wozhi (arm-demeaned, L24) ===\n")
    rows_csv = []
    # collect per-token arm-demeaned
    # build [cells, K, D] for cold
    cold_cells = [(k,v) for k,v in data.items() if v.get("kind","cold")=="cold"]
    Ks = [v["l24"].shape[0] for _,v in cold_cells]
    Kmin = min(Ks) if Ks else 0
    if Kmin >= 4 and len(cold_cells) >= 10:
        # arm means per token
        arms_c = [k[0] for k,_ in cold_cells]
        T = np.stack([v["l24"][:Kmin] for _,v in cold_cells])  # [N,K,D]
        N,Kk,D = T.shape
        # arm-demean per token
        Td = T.copy()
        for a in set(arms_c):
            m = np.array([arms_c[i]==a for i in range(N)])
            Td[m] = Td[m] - Td[m].mean(0, keepdims=True)
        hi_c = np.array([L.has_intro(k[0],k[1]) for k,_ in cold_cells])
        pw_c = np.array([L.pure_wozhi(k[0],k[1]) for k,_ in cold_cells])
        rep.append(f"  N={N} Kmin={Kmin} D={D}\n  t | AUC_raw(demean proj) | n_intro n_wozhi\n")
        for t in range(0, Kmin, max(1, Kmin//16)):
            Xc = Td[:, t, :]
            A = Xc[hi_c]; B = Xc[pw_c]
            if len(A)>=2 and len(B)>=2:
                d = A.mean(0)-B.mean(0)
                a_ = auc(A@d, B@d)
                rep.append(f"  t={t:3d}  AUC={a_:.3f}  n_i={len(A)} n_w={len(B)}\n")
                rows_csv.append((t, a_, len(A), len(B)))
    rep_txt = "".join(rep)
    print(rep_txt)
    with open(os.path.join(OUT,"probe_report.txt"),"w",encoding="utf-8") as f:
        f.write(rep_txt)
    if rows_csv:
        with open(os.path.join(OUT,"probe_per_token_auc.csv"),"w",encoding="utf-8") as f:
            f.write("token,auc,n_intro,n_wozhi\n")
            for r in rows_csv: f.write(f"{r[0]},{r[1]:.4f},{r[2]},{r[3]}\n")
    print(f"\n[psy] report -> {os.path.join(OUT,'probe_report.txt')}")


if __name__ == "__main__":
    main()