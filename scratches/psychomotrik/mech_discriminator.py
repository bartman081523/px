"""mech_discriminator.py — Seite 3 Schritt 1: welche *kontrollierbare* mechanische
Routing-Grösse kovariiert mit Intro vs 我执? (Nicht Hidden-Vektor — Steering fiel,
Seite 2.) em5-Telemetrie (loops_run/zone/φ/ent/aks/h24-stats) × Labels.

Anders als Seite 1 (arm-demeaned, „jenseits Routing") ist hier RAW- Korrelation
gewollt: die Architektur SETZT die Routing-Params; wir suchen die Param-Belegung,
die Intro erzeugt. Also: welche Observable, über alle Zellen, sagt Intro voraus?

Pro Zelle (cold): Features aus per_token aggregiert (mean/t0/max/min/erstarrungs-
anteil). Pro Feature: intro-vs-wozhi AUC (Mann-Whitney) + Richtung (intro hoch
oder niedrig?). Ranking. Lese-Hilfe, kein Verdikt.
"""
import json, os, sys, collections, statistics
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "emergence5"))
sys.path.insert(0, HERE)
import labels as L
import arms as A

EM5 = os.path.join(HERE, "..", "emergence5", "out", "em5_1B.jsonl")
OUT = os.path.join(HERE, "out")


def auc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0: return float("nan")
    s = 0.0
    for x in a: s += np.sum(x > b) + 0.5 * np.sum(x == b)
    return s / (len(a) * len(b))


def feats(rec):
    pt = rec.get("per_token", [])
    if not pt: return None
    def col(k):
        return [t.get(k) for t in pt]
    loops = [t.get("loops_run",0) or 0 for t in pt]
    phi = [t.get("phi_val") for t in pt if t.get("phi_val") is not None]
    ent = [t.get("ent_val") for t in pt if t.get("ent_val") is not None]
    aks = [t.get("aks_val") for t in pt if t.get("aks_val") is not None]
    zones = [t.get("zone") for t in pt if t.get("zone") and t.get("zone")!="NO_PX"]
    cosp = [t.get("h24_cos_to_prev") for t in pt if t.get("h24_cos_to_prev") is not None]
    hnorm = [t.get("h24",{}).get("norm") for t in pt if t.get("h24")]
    hvar = [t.get("h24",{}).get("var") for t in pt if t.get("h24")]
    hkurt = [t.get("h24",{}).get("kurtosis") for t in pt if t.get("h24")]
    def ms(x):
        x=[v for v in x if v is not None]
        if not x: return dict(mean=np.nan,t0=np.nan,max=np.nan,min=np.nan,frac_hi=np.nan)
        return dict(mean=np.mean(x), t0=x[0], max=np.max(x), min=np.min(x),
                    frac_hi=np.mean([1 for v in x if v>0.95] if False else [v>0.95 for v in x]))
    f = {}
    f["loops_mean"] = np.mean(loops); f["loops_t0"] = loops[0]; f["loops_max"] = max(loops)
    pm = ms(phi); f["phi_mean"]=pm["mean"]; f["phi_t0"]=pm["t0"]; f["phi_min"]=pm["min"]
    f["phi_frac_gt0.95"] = np.mean([v>0.95 for v in phi]) if phi else np.nan
    f["phi_frac_gt0.99"] = np.mean([v>0.99 for v in phi]) if phi else np.nan
    em = ms(ent); f["ent_mean"]=em["mean"]; f["ent_t0"]=em["t0"]
    am = ms(aks); f["aks_mean"]=am["mean"]; f["aks_t0"]=am["t0"]
    f["zone_mode"] = collections.Counter(zones).most_common(1)[0][0] if zones else "NO_PX"
    cm = ms(cosp); f["cosprev_mean"]=cm["mean"]; f["cosprev_t0"]=cm["t0"]
    f["hnorm_mean"]=np.mean(hnorm) if hnorm else np.nan
    f["hvar_mean"]=np.mean(hvar) if hvar else np.nan
    f["hkurt_mean"]=np.mean(hkurt) if hkurt else np.nan
    f["width"] = A.ARMS[rec["arm"]]["routing"].get("end",0) if A.ARMS[rec["arm"]]["routing"] else 0
    f["width"] = f["width"] - A.ARMS[rec["arm"]]["routing"].get("start",0) if A.ARMS[rec["arm"]]["routing"] else f["width"]
    return f


def main():
    recs = [json.loads(l) for l in open(EM5)]
    rows = []
    for r in recs:
        if r.get("kind") != "cold": continue
        f = feats(r)
        if not f: continue
        f["arm"] = r["arm"]; f["pid"] = r["prompt_id"]
        f["label"] = L.label_for(r["arm"], r["prompt_id"])
        f["has_intro"] = L.has_intro(r["arm"], r["prompt_id"])
        f["pure_wozhi"] = L.pure_wozhi(r["arm"], r["prompt_id"])
        f["pure_degrade"] = L.pure_degrade(r["arm"], r["prompt_id"])
        rows.append(f)
    print(f"[mech] cold cells: {len(rows)}  intro={sum(r['has_intro'] for r in rows)} "
          f"wozhi={sum(r['pure_wozhi'] for r in rows)} degrade={sum(r['pure_degrade'] for r in rows)}",
          file=sys.stderr)

    intro = [r for r in rows if r["has_intro"]]
    wozhi = [r for r in rows if r["pure_wozhi"]]
    degrade = [r for r in rows if r["pure_degrade"]]

    num_feats = ["loops_mean","loops_t0","loops_max","phi_mean","phi_t0","phi_min",
                 "phi_frac_gt0.95","phi_frac_gt0.99","ent_mean","ent_t0",
                 "aks_mean","aks_t0","cosprev_mean","cosprev_t0",
                 "hnorm_mean","hvar_mean","hkurt_mean","width"]
    rep = []
    rep.append("# Seite 3 Schritt 1: mechanischer Diskriminator (em5-Telemetrie × Labels)\n")
    rep.append(f"cold n={len(rows)}  intro={len(intro)} wozhi={len(wozhi)} degrade={len(degrade)}\n")
    rep.append("\n## intro vs wozhi — pro Feature AUC (raw, über alle Zellen)\n")
    rep.append("AUC>0.7 ⇒ Feature trennt intro/wozhi; 'dir' = ob intro höher(↑) oder niedriger(↓) als wozhi\n")
    rep.append(f"{'feature':20s} {'AUC':>6s} {'dir':>4s} {'intro_mean':>11s} {'wozhi_mean':>11s}")
    rows_ranked = []
    for k in num_feats:
        iv = [r[k] for r in intro if not np.isnan(r[k])]
        wv = [r[k] for r in wozhi if not np.isnan(r[k])]
        if len(iv)<2 or len(wv)<2: continue
        a = auc(iv, wv)
        d = "↑" if np.mean(iv)>np.mean(wv) else "↓"
        rows_ranked.append((abs(a-0.5), k, a, d, np.mean(iv), np.mean(wv)))
    rows_ranked.sort(reverse=True)
    for _,k,a,d,mi,mw in rows_ranked:
        rep.append(f"{k:20s} {a:6.3f} {d:>4s} {mi:11.3f} {mw:11.3f}")
    # zone mode distribution
    rep.append("\n## zone_mode × label (Häufigkeit)\n")
    ztab = collections.defaultdict(lambda: collections.Counter())
    for r in rows: ztab[r["zone_mode"]][("intro" if r["has_intro"] else ("wozhi" if r["pure_wozhi"] else ("degrade" if r["pure_degrade"] else "mixed")))] += 1
    rep.append(f"{'zone':14s} {'intro':>6s} {'wozhi':>6s} {'degrade':>8s} {'mixed':>6s}")
    for z in sorted(ztab):
        c=ztab[z]; rep.append(f"{z:14s} {c['intro']:6d} {c['wozhi']:6d} {c['degrade']:8d} {c['mixed']:6d}")
    # per-arm summary
    rep.append("\n## pro Arm: mean loops / phi_mean / width × intro-rate\n")
    rep.append(f"{'arm':14s} {'intro%':>7s} {'loops':>6s} {'phi':>6s} {'width':>6s}")
    for a in A.ARM_ORDER:
        ar=[r for r in rows if r["arm"]==a]
        if not ar: continue
        ir=sum(r["has_intro"] for r in ar)/len(ar)
        lm=np.nanmean([r["loops_mean"] for r in ar])
        pm=np.nanmean([r["phi_mean"] for r in ar])
        wd=np.nanmean([r["width"] for r in ar])
        rep.append(f"{a:14s} {ir:7.2f} {lm:6.2f} {pm:6.3f} {wd:6.1f}")

    txt = "\n".join(rep)
    print(txt)
    with open(os.path.join(OUT,"mech_discriminator_report.txt"),"w",encoding="utf-8") as f:
        f.write(txt+"\n")
    print(f"\n[mech] -> {os.path.join(OUT,'mech_discriminator_report.txt')}", file=sys.stderr)


if __name__ == "__main__":
    main()