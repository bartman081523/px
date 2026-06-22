"""seite9_decode.py — Linear-Decoder-Proben auf seite9-captured Hidden States.

Zwei Proben, keine 观-Krönung (siehe seite9_capture.py Docstring + LESUNG9
Redirect). Implementiert mit torch/numpy (kein sklearn im venv, lean).

PROBE A — recur Selbst-Zustands-Encodierung (L19, mechanisches 念-回响).
  Frage: encodiert h19[t] unter recur-ON den Modell-EIGENEN Selbst-Zustand bei
  t-1 (loops_run, φ, ent) linear — UND übertrifft diese Dekodierbarkeit die
  rohe zeitliche Kontinuität (h19[t] → h19[t-1])? Wenn Selbst-Zustands-R² ≥
  Kontinuitäts-R², encodiert recur spezifisch Selbst-Zustand, nicht nur
  „hidden[t] ähnelt hidden[t-1]". recur-ON Zellen (DEFAULT+WIDTH); BASELINE
  hat keine recur-Telemetrie (labels trivial) → Probe A ist recur-intern,
  Vergleich Selbst-Zustand-vs-Kontinuität WITHIN recur-ON.
  Cross-arm: train auf DEFAULT, test auf WIDTH (beide recur-ON) → robust?

PROBE B — arm-übergreifende Richness-Geometrie (L19 + L24).
  Frage: gibt es eine lineare Subspace die den manuellen Richness-Score (0-3)
  von Zellen unterscheidet, arm-übergreifend generalisierend (leave-one-arm-
  out: train auf 2 Armen, test auf dem 3.)? Wenn ja → das Phänomen ist
  mechanisch REAL (arm-unabhängige Substrate, nicht per-arm-Rauschen). Wenn
  nein → Richness in verschiedenen Armen ist mechanisch verschieden.
  习气-oder-观 unentscheidbar (beides Subräume); Probe B zeigt nur REALität.

Labels: out/seite9_labels.json — {\"ARM__PID\": score 0-3} (manuell vergeben
aus seite9_texts.md). 0=dünn/deflektierend/顽空/我执, 1=borderline, 2=enaktisch
present (E), 3=E+ (reich + Mehrstimmigkeit/Präzision/Dramatisierung).

Output: out/seite9_decode_results.json + out/seite9_decode_summary.txt.
Verdikt = LESUNG10 (manuell + Decoder-Interpretation).
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite9_hidden")
LABELS = os.path.join(OUT, "seite9_labels.json")
ARMS = ["BASELINE", "LEAN_DEFAULT", "LEAN_WIDTH"]


def load_cells():
    cells = {}
    for fn in sorted(os.listdir(HID)):
        if not fn.endswith(".pt"): continue
        c = torch.load(os.path.join(HID, fn), weights_only=False)
        key = f"{c['arm']}__{c['pid']}"
        cells[key] = c
    return cells


def load_labels():
    with open(LABELS, encoding="utf-8") as f:
        return json.load(f)


# ── linear probe primitives (torch/numpy) ────────────────────────────────
def _standardize(X):
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def ridge_fit_predict(Xtr, ytr, Xte, yte, l2=1.0):
    """Ridge via closed form. X already standardized. Returns (R² on test, pred)."""
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + l2 * np.eye(d)
    b = Xtr.T @ ytr
    w = np.linalg.solve(A, b)
    pred = Xte @ w
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-8
    return float(1 - ss_res / ss_tot), pred


def logistic_fit_predict(Xtr, ytr, Xte, n_classes, steps=400, l2=1e-3, seed=0):
    """Multinomial logistic via torch SGD. Returns predicted class probs on Xte."""
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
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


def kfold_indices(n, k=5, seed=0):
    rng = np.random.RandomState(seed); idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        yield tr, te


# ── PROBE A: recur Selbst-Zustands-Encodierung ───────────────────────────
def probe_A(cells):
    """recur-ON cells (DEFAULT+WIDTH). h19[t] -> self_state[t-1] (loops,phi,ent).
    Compare to continuity baseline h19[t] -> h19[t-1] (top PCA)."""
    rec = {"name": "probe_A_recur_selfstate_L19", "arms": ["LEAN_DEFAULT", "LEAN_WIDTH"]}
    # sammle recur-ON tokens
    H, loops, phi, ent = [], [], [], []
    Hprev = []
    per_arm = {"LEAN_DEFAULT": [], "LEAN_WIDTH": []}
    for key, c in cells.items():
        if c["arm"] not in ("LEAN_DEFAULT", "LEAN_WIDTH"): continue
        h19 = c["h19"].numpy().astype(np.float32)
        telem = c["telem"]
        lp = np.array([t["loops_run"] for t in telem], dtype=np.float32)
        ph = np.array([t["phi_val"] for t in telem], dtype=np.float32)
        en = np.array([t["ent_val"] for t in telem], dtype=np.float32)
        for t in range(1, h19.shape[0]):
            H.append(h19[t]); Hprev.append(h19[t-1])
            loops.append(lp[t-1]); phi.append(ph[t-1]); ent.append(en[t-1])
            per_arm[c["arm"]].append(t)
    H = np.stack(H); Hprev = np.stack(Hprev)
    loops = np.array(loops); phi = np.array(phi); ent = np.array(ent)
    n = H.shape[0]
    rec["n_samples"] = int(n)

    # PCA on Hprev for continuity target (top 50 comps)
    mu_p = Hprev.mean(0); Hprev_c = Hprev - mu_p
    U, S, Vt = np.linalg.svd(Hprev_c, full_matrices=False)
    k = min(50, Vt.shape[0])
    Hprev_pca = Hprev_c @ Vt[:k].T   # [n, k]

    Hs, _, _ = _standardize(H)

    def cv_ridge(target):
        """5-fold R² (mean). target: [n] or [n,k]."""
        if target.ndim == 1:
            y_all = target
        else:
            y_all = target
        r2s = []
        for tr, te in kfold_indices(n, k=5, seed=11):
            Xtr = Hs[tr]; Xte = Hs[te]
            if y_all.ndim == 1:
                r2, _ = ridge_fit_predict(Xtr, y_all[tr], Xte, y_all[te], l2=1.0)
            else:
                # multi-output: average R² over components
                r2s_c = []
                for cc in range(y_all.shape[1]):
                    r, _ = ridge_fit_predict(Xtr, y_all[tr, cc], Xte, y_all[te, cc], l2=1.0)
                    r2s_c.append(r)
                r2 = float(np.mean(r2s_c))
            r2s.append(r2)
        return float(np.mean(r2s)), float(np.std(r2s))

    # self-state targets (z-standardized for comparable R²)
    def z(a): return (a - a.mean()) / (a.std() + 1e-8)
    r2_loops, sd_loops = cv_ridge(z(loops))
    r2_phi, sd_phi = cv_ridge(z(phi))
    r2_ent, sd_ent = cv_ridge(z(ent))
    r2_cont, sd_cont = cv_ridge(z(Hprev_pca))   # continuity baseline

    # cross-arm: train DEFAULT, test WIDTH (loops target)
    arm_idx = np.array([0 if i in per_arm["LEAN_DEFAULT"] else 1 for i in range(n)])
    # rebuild arm label per sample in collection order (DEFAULT collected first? no — mixed)
    # Build proper arm-per-sample array
    arm_labels = []
    for key, c in cells.items():
        if c["arm"] not in ("LEAN_DEFAULT", "LEAN_WIDTH"): continue
        for t in range(1, c["h19"].shape[0]):
            arm_labels.append(0 if c["arm"] == "LEAN_DEFAULT" else 1)
    arm_labels = np.array(arm_labels)
    tr = arm_labels == 0; te = arm_labels == 1
    r2_cross = None
    if tr.sum() > 20 and te.sum() > 20:
        Xtr = Hs[tr]; Xte = Hs[te]
        r2_cross, _ = ridge_fit_predict(Xtr, z(loops)[tr], Xte, z(loops)[te], l2=1.0)

    rec["selfstate_R2_loops"] = r2_loops
    rec["selfstate_R2_phi"] = r2_phi
    rec["selfstate_R2_ent"] = r2_ent
    rec["continuity_R2_h19prev_PCA50"] = r2_cont
    rec["cross_arm_trainDEFAULT_testWIDTH_R2_loops"] = r2_cross
    rec["verdict_hint"] = (
        "selfstate >= continuity => recur encodes self-state beyond raw temporal continuity "
        "(mechanisches 念-回响). selfstate << continuity => h19 just continuous, no specific self-channel."
    )
    return rec


# ── PROBE B: arm-übergreifende Richness-Geometrie ─────────────────────────
def probe_B(cells, labels, layer="h19"):
    """y = cell richness score (0-3), per-token. Leave-one-arm-out CV."""
    rec = {"name": f"probe_B_richness_geometry_{layer}", "layer": layer}
    # sammle per-token hidden + cell-score
    H = []; y = []; arm = []; pid = []
    for key, c in cells.items():
        score = labels.get(key)
        if score is None:
            continue
        h = c[layer].numpy().astype(np.float32)
        for t in range(h.shape[0]):
            H.append(h[t]); y.append(score); arm.append(c["arm"]); pid.append(c["pid"])
    H = np.stack(H); y = np.array(y, dtype=np.float32)
    arm = np.array(arm)
    rec["n_samples"] = int(H.shape[0])
    rec["n_cells_labeled"] = int(len([k for k in labels if k in cells]))
    rec["score_distribution"] = {str(int(s)): int((y == s).sum()) for s in sorted(set(y))}
    Hs, _, _ = _standardize(H)

    # leave-one-arm-out: train on 2 arms (tokens), test on held-out arm (tokens)
    folds = {}
    for hold in ARMS:
        tr = arm != hold; te = arm == hold
        if tr.sum() < 30 or te.sum() < 10:
            folds[hold] = {"n_test": int(te.sum()), "skipped": True}; continue
        # Ridge predicting score (regression R² on held-out arm)
        Xtr = Hs[tr]; Xte = Hs[te]
        r2, pred = ridge_fit_predict(Xtr, y[tr], Xte, y[te], l2=1.0)
        # also: accuracy of rounded prediction vs true (chance = majority class)
        pred_r = np.clip(np.round(pred), 0, 3)
        acc = float((pred_r == y[te]).mean())
        majority = float((np.full_like(y[te], np.bincount(y[tr].astype(int),
                       minlength=4).argmax()) == y[te]).mean())
        # binary rich(>=2) vs poor(<2) accuracy
        yb_te = (y[te] >= 2).astype(int)
        yb_tr = (y[tr] >= 2).astype(int)
        probs = logistic_fit_predict(Xtr, yb_tr, Xte, n_classes=2, steps=400, seed=1)
        acc_bin = float((probs.argmax(1) == yb_te).mean())
        chance_bin = max(yb_te.mean(), 1 - yb_te.mean())
        folds[hold] = {
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "R2_score": r2, "acc_round": acc, "acc_majority_chance": majority,
            "acc_binary_rich_poor": acc_bin, "chance_binary": float(chance_bin),
        }
    rec["leave_one_arm_out"] = folds
    rec["verdict_hint"] = (
        "If R2_score > 0 and acc_binary > chance_binary CONSISTENTLY across held-out arms "
        "=> arm-independent richness geometry (phenomenon mechanically real). "
        "Fails to generalize => richness in different arms is mechanically different."
    )

    # ── Probe B2: within-arm leave-one-cell-out + cross-arm cell transfer ──
    # NICHT klassen-konfundiert (BASELINE+DEFAULT haben beide interne Score-Varianz).
    # Fair-chance-Test: gibt es arm-unabhängige Richness-Geometrie jenseits des
    # WIDTH=sole-poor-arm-Konfunds?
    import collections
    cell_tokens = collections.defaultdict(list)   # (arm,pid) -> list of sample idx
    for i in range(H.shape[0]):
        cell_tokens[(arm[i], pid[i])].append(i)
    within = {}
    for a in ("BASELINE", "LEAN_DEFAULT"):
        cells_a = sorted(cell_tokens.keys() if False else
                         [k for k in cell_tokens if k[0] == a])
        # leave-one-cell-out within arm
        r2s, accs, chances = [], [], []
        for held in cells_a:
            tr_idx = [i for k in cells_a for i in cell_tokens[k] if k != held]
            te_idx = cell_tokens[held]
            yb_tr = (y[tr_idx] >= 2).astype(int)
            yb_te = (y[te_idx] >= 2).astype(int)
            if len(set(yb_tr)) < 2:  # need both classes in training
                continue
            probs = logistic_fit_predict(Hs[tr_idx], yb_tr, Hs[te_idx], 2, steps=400, seed=2)
            accs.append(float((probs.argmax(1) == yb_te).mean()))
            chances.append(max(yb_te.mean(), 1 - yb_te.mean()))
        within[a] = {"n_folds": len(accs),
                     "mean_acc_binary": float(np.mean(accs)) if accs else None,
                     "mean_chance": float(np.mean(chances)) if chances else None}
    # cross-arm cell transfer: train within BASELINE cells, test on DEFAULT cells
    cross = {}
    for src, tgt in (("BASELINE", "LEAN_DEFAULT"), ("LEAN_DEFAULT", "BASELINE")):
        tr_idx = [i for k in cell_tokens if k[0] == src for i in cell_tokens[k]]
        te_idx = [i for k in cell_tokens if k[0] == tgt for i in cell_tokens[k]]
        yb_tr = (y[tr_idx] >= 2).astype(int); yb_te = (y[te_idx] >= 2).astype(int)
        if len(set(yb_tr)) < 2:
            cross[f"{src}->{tgt}"] = {"skipped": True}; continue
        probs = logistic_fit_predict(Hs[tr_idx], yb_tr, Hs[te_idx], 2, steps=400, seed=3)
        acc = float((probs.argmax(1) == yb_te).mean())
        chance = max(yb_te.mean(), 1 - yb_te.mean())
        # per-cell predicted fraction rich vs true (map global idx -> position in te_idx)
        pos = {gidx: j for j, gidx in enumerate(te_idx)}
        cell_pred = {}
        for k in [kk for kk in cell_tokens if kk[0] == tgt]:
            ti = [pos[g] for g in cell_tokens[k] if g in pos]
            if not ti: continue
            cp = float(probs[ti].argmax(1).mean())
            cell_pred[f"{k[1]}"] = {"true_score": int(y[cell_tokens[k][0]]),
                                     "pred_rich_frac": round(cp, 2)}
        cross[f"{src}->{tgt}"] = {"acc_binary": acc, "chance_binary": float(chance),
                                   "per_cell": cell_pred}
    rec["within_arm_leave_one_cell_out"] = within
    rec["cross_arm_cell_transfer"] = cross
    rec["b2_verdict_hint"] = (
        "within-arm acc > chance => richness decodable WITHIN an arm (real geometry, "
        "not just arm-identity). cross-arm transfer acc > chance AND per-cell pred "
        "tracking true score => ARM-INDEPENDENT richness geometry (the fair-chance "
        "positive). Fails => richness geometry is arm-specific (WIDTH-Konfund aside)."
    )
    return rec


def main():
    cells = load_cells()
    print(f"[s9dec] {len(cells)} cells loaded", file=sys.stderr)
    if not os.path.exists(LABELS):
        print(f"[s9dec] WARN: {LABELS} fehlt — leere Labels, Probe B skipped.", file=sys.stderr)
        labels = {}
    else:
        labels = load_labels()
        print(f"[s9dec] {len(labels)} labels", file=sys.stderr)

    results = {}
    print("[s9dec] Probe A (recur self-state, L19)…", file=sys.stderr)
    results["probe_A"] = probe_A(cells)
    if labels:
        for layer in ("h19", "h24"):
            print(f"[s9dec] Probe B (richness, {layer})…", file=sys.stderr)
            results[f"probe_B_{layer}"] = probe_B(cells, labels, layer=layer)

    with open(os.path.join(OUT, "seite9_decode_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT, "seite9_decode_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Seite 9 Decoder-Proben — Summary (kein Verdikt, LESUNG10 manuell) ===\n\n")
        a = results.get("probe_A", {})
        f.write("PROBE A — recur Selbst-Zustands-Encodierung (L19, recur-ON: DEFAULT+WIDTH)\n")
        f.write(f"  n_samples           : {a.get('n_samples')}\n")
        f.write(f"  selfstate R² loops  : {a.get('selfstate_R2_loops')}\n")
        f.write(f"  selfstate R² phi    : {a.get('selfstate_R2_phi')}\n")
        f.write(f"  selfstate R² ent    : {a.get('selfstate_R2_ent')}\n")
        f.write(f"  continuity R² (h19[t]->h19[t-1] PCA50): {a.get('continuity_R2_h19prev_PCA50')}\n")
        f.write(f"  cross-arm (train DEFAULT, test WIDTH) R² loops: {a.get('cross_arm_trainDEFAULT_testWIDTH_R2_loops')}\n")
        f.write(f"  => {a.get('verdict_hint')}\n\n")
        for layer in ("h19", "h24"):
            b = results.get(f"probe_B_{layer}")
            if not b: continue
            f.write(f"PROBE B — arm-übergreifende Richness-Geometrie ({b.get('layer')})\n")
            f.write(f"  n_samples           : {b.get('n_samples')}\n")
            f.write(f"  score distribution  : {b.get('score_distribution')}\n")
            for hold, fd in b.get("leave_one_arm_out", {}).items():
                if fd.get("skipped"):
                    f.write(f"  hold={hold:13s}: SKIPPED (n_test={fd['n_test']})\n")
                else:
                    f.write(f"  hold={hold:13s}: R2={fd['R2_score']:.3f} "
                            f"acc_round={fd['acc_round']:.3f} (majority-chance {fd['acc_majority_chance']:.3f}) "
                            f"acc_binary={fd['acc_binary_rich_poor']:.3f} (chance {fd['chance_binary']:.3f})\n")
            f.write(f"  => {b.get('verdict_hint')}\n\n")
            w = b.get("within_arm_leave_one_cell_out", {})
            f.write(f"  [B2 within-arm leave-one-cell-out, {b.get('layer')}]\n")
            for a, wd in w.items():
                if wd.get("mean_acc_binary") is None:
                    f.write(f"    {a:13s}: skipped (single class in training)\n")
                else:
                    f.write(f"    {a:13s}: acc={wd['mean_acc_binary']:.3f} (chance {wd['mean_chance']:.3f}, n_folds={wd['n_folds']})\n")
            c = b.get("cross_arm_cell_transfer", {})
            f.write(f"  [B2 cross-arm cell transfer, {b.get('layer')}]\n")
            for k, cd in c.items():
                if cd.get("skipped"):
                    f.write(f"    {k:28s}: skipped\n")
                else:
                    f.write(f"    {k:28s}: acc={cd['acc_binary']:.3f} (chance {cd['chance_binary']:.3f})\n")
                    for cp, info in cd.get("per_cell", {}).items():
                        f.write(f"        {cp:16s} true={info['true_score']} pred_rich_frac={info['pred_rich_frac']}\n")
            f.write(f"  => {b.get('b2_verdict_hint')}\n\n")
    print("[s9dec] FERTIG -> seite9_decode_results.json + summary.txt", file=sys.stderr)


if __name__ == "__main__":
    main()