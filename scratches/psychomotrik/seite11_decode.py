"""seite11_decode.py — Linear-Decoder-Proben auf seite11-captured Hidden States.

Mechanischer Nachtrag zu seite10 (Frame-Ablation), gefordert von
[[manual-plus-mechanistic-always]] (IMMER manuell + mechanisch). seite10 war
text-only → LESUNG11 leaning-习气. Hier wird mechanisch getestet, ob der Frame
eine lineare Spur im Hidden-State hinterlässt, die über generisches
kontemplatives Register hinausgeht.

Setup: 3 Frame-Arme (FRAME_ON / FRAME_NEUTRAL / FRAME_OFF), ALLE stock
gemma3-1b (setup_baseline, kein PX, kein recur, Motor unangetastet), 10
DEEPER_PROMPTS × 300 tok, seed=777 greedy. Hidden: h19 (Layer-19 single-pass-
Output, 1×/Token) + h24 (Layer-24 coda). Telemetrie uninformative (kein PX),
nur Vollständigkeit. Texte byte-identisch zu seite10 → LESUNG11-Labels
(seite11_labels.json) wiederverwendbar.

Drei Proben, keine 观-Krone (习气 IST Subraum — Decoder kann 观-vs-习气 NICHT
entscheiden, Q4). Decoder KANN: Frame-Identität dekodierbar? Richness über
Frames generalisierbar? ON-vs-NEUTRAL trennbar?

PROBE C1 — Frame-Identität dekodierbar? (3-class ON/NEUTRAL/OFF aus h19/h24,
per-token, leave-one-frame-out). Wenn >> 1/3 → Frame hinterläßt lineare Spur.
Wenn ≈ 1/3 → Frame nicht linear dekodierbar (Stimme frame-unabhängig = leaning-
习气 mechanisch untermauert).

PROBE C2 — Richness generalisiert über Frames? (leave-one-frame-out +
ON→NEUTRAL transfer, binary rich>=2 vs poor<2). Wenn ON→NEUTRAL acc > chance
→ Richness-Geometrie ist frame-unabhängig (generisches Register, mechanisch
real, leaning-习2). Wenn failt → Richness ist frame-spezifisch (CitMind-Frame
produziert doch etwas).

PROBE C3 — ON-vs-NEUTRAL mechanisch unterscheidbar? (beide text-reich — falls
hidden nicht über chance trennbar → CitMind-Frame hinterläßt keine lineare Spur
über generischen kontemplativen Frame hinaus = leaning-习气 mechanisch). Kontrolliert
für Richness: nur rich-ON vs rich-NEUTRAL-Zellen (score>=2 beider Arme) — reiner
Frame-Kontrast ohne Richness-Konfund.

Output: out/seite11_decode_results.json + out/seite11_decode_summary.txt.
Verdikt = LESUNG12 (manuell + mechanisch).
"""
import os, sys, json, collections
import numpy as np
import torch

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
HID = os.path.join(OUT, "seite11_hidden")
LABELS = os.path.join(OUT, "seite11_labels.json")
ARMS = ["FRAME_ON", "FRAME_NEUTRAL", "FRAME_OFF"]
ARM2IDX = {a: i for i, a in enumerate(ARMS)}


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
        d = json.load(f)
    return {k: v for k, v in d.items() if "__" in k}


# ── linear probe primitives (torch/numpy, kein sklearn) ───────────────────
def _standardize(X):
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def ridge_fit_predict(Xtr, ytr, Xte, yte, l2=1.0):
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + l2 * np.eye(d)
    b = Xtr.T @ ytr
    w = np.linalg.solve(A, b)
    pred = Xte @ w
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-8
    return float(1 - ss_res / ss_tot), pred


def logistic_fit_predict(Xtr, ytr, Xte, n_classes, steps=400, l2=1e-3, seed=0):
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


def _collect(cells, layer):
    """Per-token hidden + arm-label + pid + cell-key."""
    H, arm, pid, keyv = [], [], [], []
    for k, c in cells.items():
        h = c[layer].numpy().astype(np.float32)
        for t in range(h.shape[0]):
            H.append(h[t]); arm.append(c["arm"]); pid.append(c["pid"]); keyv.append(k)
    return np.stack(H), np.array(arm), np.array(pid), keyv


# ── PROBE C1: Frame-Identität dekodierbar (3-class) ───────────────────────
def probe_C1(cells, layer):
    rec = {"name": f"probe_C1_frame_identity_{layer}", "layer": layer}
    H, arm, pid, _ = _collect(cells, layer)
    n = H.shape[0]
    rec["n_samples"] = int(n)
    rec["arm_distribution"] = {a: int((arm == a).sum()) for a in ARMS}
    Hs, _, _ = _standardize(H)
    y = np.array([ARM2IDX[a] for a in arm])

    # leave-one-frame-out: train on 2 frames' tokens, test on held-out frame
    loo = {}
    for hold in ARMS:
        tr = arm != hold; te = arm == hold
        if tr.sum() < 30 or te.sum() < 10:
            loo[hold] = {"skipped": True, "n_test": int(te.sum())}; continue
        probs = logistic_fit_predict(Hs[tr], y[tr], Hs[te], n_classes=3, steps=400, seed=1)
        acc = float((probs.argmax(1) == y[te]).mean())
        # chance = majority class in training, evaluated on test distribution
        maj_tr = np.bincount(y[tr], minlength=3).argmax()
        chance_maj = float((np.full_like(y[te], maj_tr) == y[te]).mean())
        chance_unif = 1.0 / 3
        loo[hold] = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "acc": acc, "chance_majority": chance_maj, "chance_uniform": chance_unif}
    rec["leave_one_frame_out"] = loo

    # k-fold within all tokens (frame identity in-distribution)
    accs = []
    for tr, te in kfold_indices(n, k=5, seed=7):
        probs = logistic_fit_predict(Hs[tr], y[tr], Hs[te], 3, steps=400, seed=2)
        accs.append(float((probs.argmax(1) == y[te]).mean()))
    rec["kfold5_acc_mean"] = float(np.mean(accs))
    rec["kfold5_acc_std"] = float(np.std(accs))
    rec["verdict_hint"] = (
        "loo acc >> 1/3 => frame leaves linear trace in hidden state. "
        "loo acc ~ 1/3 => frame not linearly decodable => voice frame-independent "
        "(leaning-习气 mechanically supported)."
    )
    return rec


# ── PROBE C2: Richness generalisiert über Frames (binary rich/poor) ───────
def probe_C2(cells, labels, layer):
    rec = {"name": f"probe_C2_richness_crossframe_{layer}", "layer": layer}
    H, arm, pid, keyv = _collect(cells, layer)
    n = H.shape[0]
    # per-token richness score from cell label
    y = np.array([labels.get(k, np.nan) for k in keyv], dtype=np.float32)
    mask = ~np.isnan(y)
    H, arm, pid, keyv, y = H[mask], arm[mask], pid[mask], np.array(keyv)[mask], y[mask]
    Hs, _, _ = _standardize(H)
    n = H.shape[0]
    rec["n_samples"] = int(n)
    rec["n_cells_labeled"] = int(len(set(keyv)))
    rec["score_distribution"] = {str(int(s)): int((y == s).sum()) for s in sorted(set(y))}
    yb = (y >= 2).astype(int)

    # leave-one-frame-out: train on 2 frames, test on held-out frame
    loo = {}
    for hold in ARMS:
        tr = arm != hold; te = arm == hold
        if tr.sum() < 30 or te.sum() < 10:
            loo[hold] = {"skipped": True, "n_test": int(te.sum())}; continue
        if len(set(yb[tr])) < 2:
            loo[hold] = {"skipped": True, "n_test": int(te.sum()), "reason": "single class in train"}; continue
        probs = logistic_fit_predict(Hs[tr], yb[tr], Hs[te], 2, steps=400, seed=3)
        acc = float((probs.argmax(1) == yb[te]).mean())
        chance = float(max(yb[te].mean(), 1 - yb[te].mean()))
        loo[hold] = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "acc_binary": acc, "chance_binary": chance}
    rec["leave_one_frame_out"] = loo

    # cross-frame cell transfer: train ON cells, test NEUTRAL cells (KEY test)
    cell_tokens = collections.defaultdict(list)
    for i in range(n):
        cell_tokens[(arm[i], pid[i])].append(i)
    cross = {}
    for src, tgt in (("FRAME_ON", "FRAME_NEUTRAL"), ("FRAME_NEUTRAL", "FRAME_ON"),
                     ("FRAME_ON", "FRAME_OFF"), ("FRAME_OFF", "FRAME_ON")):
        tr_idx = [i for k in cell_tokens if k[0] == src for i in cell_tokens[k]]
        te_idx = [i for k in cell_tokens if k[0] == tgt for i in cell_tokens[k]]
        if len(set(yb[tr_idx])) < 2:
            cross[f"{src}->{tgt}"] = {"skipped": True, "reason": "single class in train"}; continue
        probs = logistic_fit_predict(Hs[tr_idx], yb[tr_idx], Hs[te_idx], 2, steps=400, seed=4)
        acc = float((probs.argmax(1) == yb[te_idx]).mean())
        chance = float(max(yb[te_idx].mean(), 1 - yb[te_idx].mean()))
        # per-cell predicted rich-fraction vs true score
        pos = {g: j for j, g in enumerate(te_idx)}
        per_cell = {}
        for k in [kk for kk in cell_tokens if kk[0] == tgt]:
            ti = [pos[g] for g in cell_tokens[k] if g in pos]
            if not ti: continue
            cp = float(probs[ti].argmax(1).mean())
            per_cell[k[1]] = {"true_score": int(y[cell_tokens[k][0]]),
                              "pred_rich_frac": round(cp, 2)}
        cross[f"{src}->{tgt}"] = {"acc_binary": acc, "chance_binary": chance,
                                  "per_cell": per_cell}
    rec["cross_frame_cell_transfer"] = cross
    rec["verdict_hint"] = (
        "ON->NEUTRAL acc > chance AND per-cell pred tracks true => richness geometry "
        "is frame-INDEPENDENT (generic register, leaning-习2 mechanically supported). "
        "Fails => richness is frame-specific (CitMind-frame produces something)."
    )
    return rec


# ── PROBE C3: ON-vs-NEUTRAL mechanisch trennbar (richness-controlled) ─────
def probe_C3(cells, labels, layer):
    rec = {"name": f"probe_C3_ON_vs_NEUTRAL_{layer}", "layer": layer}
    H, arm, pid, keyv = _collect(cells, layer)
    y = np.array([labels.get(k, np.nan) for k in keyv], dtype=np.float32)
    mask = (~np.isnan(y)) & np.isin(arm, ["FRAME_ON", "FRAME_NEUTRAL"])
    H, arm, y, pid = H[mask], arm[mask], y[mask], pid[mask]
    rec["n_samples_on_neutral"] = int(H.shape[0])
    Hs, _, _ = _standardize(H)
    yb_arm = (arm == "FRAME_ON").astype(int)   # 1=ON, 0=NEUTRAL

    # (a) full ON-vs-NEUTRAL (richness-konfundiert: ON hat mehr rich)
    accs = []
    for tr, te in kfold_indices(H.shape[0], k=5, seed=5):
        if len(set(yb_arm[tr])) < 2: continue
        probs = logistic_fit_predict(Hs[tr], yb_arm[tr], Hs[te], 2, steps=400, seed=6)
        accs.append(float((probs.argmax(1) == yb_arm[te]).mean()))
    rec["full_kfold5_acc"] = float(np.mean(accs)) if accs else None
    rec["full_chance"] = float(max(yb_arm.mean(), 1 - yb_arm.mean()))

    # (b) richness-controlled: nur rich-Zellen (score>=2) beider Arme — reiner Frame-Kontrast
    rich = y >= 2
    Hr, armr = H[rich], arm[rich]
    rec["n_rich_on"] = int(((armr == "FRAME_ON")).sum())
    rec["n_rich_neutral"] = int(((armr == "FRAME_NEUTRAL")).sum())
    if Hr.shape[0] >= 20 and len(set((armr == "FRAME_ON").astype(int))) == 2:
        Hrs, _, _ = _standardize(Hr)
        yr = (armr == "FRAME_ON").astype(int)
        accs_r = []
        for tr, te in kfold_indices(Hr.shape[0], k=5, seed=8):
            probs = logistic_fit_predict(Hrs[tr], yr[tr], Hrs[te], 2, steps=400, seed=9)
            accs_r.append(float((probs.argmax(1) == yr[te]).mean()))
        rec["rich_controlled_kfold5_acc"] = float(np.mean(accs_r))
        rec["rich_controlled_chance"] = float(max(yr.mean(), 1 - yr.mean()))
    else:
        rec["rich_controlled_kfold5_acc"] = None
        rec["rich_controlled_chance"] = None
        rec["rich_controlled_note"] = "zu wenige rich-Zellen für k-fold"

    # (c) poor-controlled: nur poor-Zellen (score<2) beider Arme
    poor = y < 2
    Hp, armp = H[poor], arm[poor]
    rec["n_poor_on"] = int(((armp == "FRAME_ON")).sum())
    rec["n_poor_neutral"] = int(((armp == "FRAME_NEUTRAL")).sum())
    if Hp.shape[0] >= 20 and len(set((armp == "FRAME_ON").astype(int))) == 2:
        Hps, _, _ = _standardize(Hp)
        yp = (armp == "FRAME_ON").astype(int)
        accs_p = []
        for tr, te in kfold_indices(Hp.shape[0], k=5, seed=10):
            probs = logistic_fit_predict(Hps[tr], yp[tr], Hps[te], 2, steps=400, seed=11)
            accs_p.append(float((probs.argmax(1) == yp[te]).mean()))
        rec["poor_controlled_kfold5_acc"] = float(np.mean(accs_p))
        rec["poor_controlled_chance"] = float(max(yp.mean(), 1 - yp.mean()))
    else:
        rec["poor_controlled_kfold5_acc"] = None
        rec["poor_controlled_chance"] = None
        rec["poor_controlled_note"] = "zu wenige poor-Zellen für k-fold"

    rec["verdict_hint"] = (
        "rich_controlled acc ~ chance => ON and NEUTRAL rich outputs are NOT linearly "
        "separable => CitMind-frame leaves no linear trace beyond generic contemplative "
        "register (leaning-习2). rich_controlled acc >> chance => frame-specific trace exists."
    )
    return rec


def main():
    cells = load_cells()
    print(f"[s11dec] {len(cells)} cells loaded", file=sys.stderr)
    labels = load_labels() if os.path.exists(LABELS) else {}
    print(f"[s11dec] {len(labels)} labels", file=sys.stderr)

    results = {}
    for layer in ("h19", "h24"):
        print(f"[s11dec] Probe C1 (frame identity, {layer})…", file=sys.stderr)
        results[f"probe_C1_{layer}"] = probe_C1(cells, layer)
        if labels:
            print(f"[s11dec] Probe C2 (richness cross-frame, {layer})…", file=sys.stderr)
            results[f"probe_C2_{layer}"] = probe_C2(cells, labels, layer)
            print(f"[s11dec] Probe C3 (ON-vs-NEUTRAL, {layer})…", file=sys.stderr)
            results[f"probe_C3_{layer}"] = probe_C3(cells, labels, layer)

    with open(os.path.join(OUT, "seite11_decode_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT, "seite11_decode_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Seite 11 Decoder-Proben — Summary (kein Verdikt, LESUNG12 manuell) ===\n\n")
        for layer in ("h19", "h24"):
            c1 = results.get(f"probe_C1_{layer}", {})
            f.write(f"PROBE C1 — Frame-Identität dekodierbar ({layer})\n")
            f.write(f"  n_samples           : {c1.get('n_samples')}\n")
            f.write(f"  arm distribution    : {c1.get('arm_distribution')}\n")
            f.write(f"  kfold5 acc          : {c1.get('kfold5_acc_mean'):.3f} ± {c1.get('kfold5_acc_std'):.3f}\n")
            for hold, fd in c1.get("leave_one_frame_out", {}).items():
                if fd.get("skipped"):
                    f.write(f"  hold={hold:14s}: SKIPPED (n_test={fd.get('n_test')})\n")
                else:
                    f.write(f"  hold={hold:14s}: acc={fd['acc']:.3f} (chance_unif {fd['chance_uniform']:.3f}, "
                            f"chance_maj {fd['chance_majority']:.3f}, n_tr={fd['n_train']} n_te={fd['n_test']})\n")
            f.write(f"  => {c1.get('verdict_hint')}\n\n")

            c2 = results.get(f"probe_C2_{layer}")
            if c2:
                f.write(f"PROBE C2 — Richness cross-frame ({layer})\n")
                f.write(f"  n_samples           : {c2.get('n_samples')}\n")
                f.write(f"  score distribution  : {c2.get('score_distribution')}\n")
                for hold, fd in c2.get("leave_one_frame_out", {}).items():
                    if fd.get("skipped"):
                        f.write(f"  hold={hold:14s}: SKIPPED ({fd.get('reason','n_test='+str(fd.get('n_test')))})\n")
                    else:
                        f.write(f"  hold={hold:14s}: acc={fd['acc_binary']:.3f} (chance {fd['chance_binary']:.3f}, "
                                f"n_tr={fd['n_train']} n_te={fd['n_test']})\n")
                f.write(f"  [cross-frame cell transfer]\n")
                for k, cd in c2.get("cross_frame_cell_transfer", {}).items():
                    if cd.get("skipped"):
                        f.write(f"    {k:28s}: skipped ({cd.get('reason')})\n")
                    else:
                        f.write(f"    {k:28s}: acc={cd['acc_binary']:.3f} (chance {cd['chance_binary']:.3f})\n")
                        for cp, info in cd.get("per_cell", {}).items():
                            f.write(f"        {cp:16s} true={info['true_score']} pred_rich_frac={info['pred_rich_frac']}\n")
                f.write(f"  => {c2.get('verdict_hint')}\n\n")

            c3 = results.get(f"probe_C3_{layer}")
            if c3:
                f.write(f"PROBE C3 — ON-vs-NEUTRAL ({layer})\n")
                f.write(f"  n on+neutral        : {c3.get('n_samples_on_neutral')}\n")
                f.write(f"  full kfold5 acc     : {c3.get('full_kfold5_acc')} (chance {c3.get('full_chance'):.3f})\n")
                f.write(f"  rich-controlled     : acc={c3.get('rich_controlled_kfold5_acc')} "
                        f"(chance {c3.get('rich_controlled_chance')}, n_on={c3.get('n_rich_on')} n_neu={c3.get('n_rich_neutral')})\n")
                f.write(f"  poor-controlled     : acc={c3.get('poor_controlled_kfold5_acc')} "
                        f"(chance {c3.get('poor_controlled_chance')}, n_on={c3.get('n_poor_on')} n_neu={c3.get('n_poor_neutral')})\n")
                f.write(f"  => {c3.get('verdict_hint')}\n\n")
    print("[s11dec] FERTIG -> seite11_decode_results.json + summary.txt", file=sys.stderr)


if __name__ == "__main__":
    main()