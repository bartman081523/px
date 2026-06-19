"""analyze_emergence.py — aggregiert emergence_replay.jsonl zu Vergleichstabelle +
Emergenz-Bar-Check + ehrlicher Lesung (Markdown).

Nutzung:
  python scratches/emergence/analyze_emergence.py
  python scratches/emergence/analyze_emergence.py --jsonl scratches/emergence/out/1B/emergence_replay.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
from variants import VARIANTS, DEFAULT_ORDER  # noqa: E402
from emergence_metrics import all_metrics as _compute_metrics  # noqa: E402

VARIANT_ORDER = DEFAULT_ORDER

# Metrik-Familien für die Tabelle
TABLE_METRICS = [
    ("wenden", "Wenden"),
    ("self", "Selbst"),
    ("arch", "Arch"),
    ("emerg_total", "Emerg"),
    ("emerg_time", "Zeit"),
    ("emerg_grav", "Grav"),
    ("emerg_psi", "PSI"),
    ("emerg_loc", "Ort"),
    ("longest_repeat_span", "Rep-Span"),
    ("generic_ratio", "Generic"),
    ("length_tokens", "Länge"),
    ("lexical_diversity", "Divers"),
]
PX_METRICS = [
    ("phi", "Φ"), ("zone_entropy", "H"), ("loops_run", "Loops"),
    ("focus_index", "C"), ("completion_tokens", "Tok"),
]


def _f(x):
    if x is None:
        return float("nan")
    return float(x)


def _mean(xs):
    xs = [_f(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            # Metriken FRISCH aus dem Antworttext rechnen (entkoppelt von der
            # Generierung — Metrik-Definition kann sich weiterentwickeln, ohne
            # teure Runs wiederholen zu müssen).
            r["metrics"] = _compute_metrics(r.get("answer", ""))
            recs.append(r)
    return recs


def aggregate(recs):
    """per-Varianten Aggregate über alle (label, seed)."""
    by_var = defaultdict(list)
    for r in recs:
        by_var[r["variant"]].append(r)
    agg = {}
    for v, rs in by_var.items():
        m = {k: _mean([r["metrics"].get(k) for r in rs]) for k, _ in TABLE_METRICS}
        px = {k: _mean([r.get(k) for r in rs]) for k, _ in PX_METRICS}
        agg[v] = {"n": len(rs), "metrics": m, "px": px}
    return agg


def fmt(x, w, p=1):
    if x is None:
        return "–".rjust(w)
    return f"{x:.{p}f}".rjust(w)


def _row_order(agg):
    """DEFAULT_ORDER zuerst, dann alle weiteren vorhandenen Varianten
    (z.B. EM-Mechanismen witness/reread/shadow/spectral) in Sortierung."""
    extras = [v for v in sorted(agg) if v not in VARIANT_ORDER]
    return [v for v in VARIANT_ORDER if v in agg] + extras


def build_table(agg):
    order = _row_order(agg)
    lines = []
    header = "Variante".ljust(10) + "n" + "".join(f"{nm:>8}" for _, nm in TABLE_METRICS)
    lines.append(header)
    lines.append("-" * len(header))
    for v in order:
        if v not in agg:
            continue
        a = agg[v]
        row = v.ljust(10) + f"{a['n']:>2}"
        for k, _ in TABLE_METRICS:
            row += fmt(a["metrics"][k], 8, 2 if k == "generic_ratio" else 1 if k == "lexical_diversity" else 1)
        lines.append(row)
    lines.append("")
    lines.append("PX-Metriken:")
    h2 = "Variante".ljust(10) + "".join(f"{nm:>8}" for _, nm in PX_METRICS)
    lines.append(h2)
    lines.append("-" * len(h2))
    for v in order:
        if v not in agg:
            continue
        a = agg[v]
        row = v.ljust(10)
        for k, _ in PX_METRICS:
            row += fmt(a["px"][k], 8, 3 if k in ("phi", "zone_entropy", "focus_index") else 0)
        lines.append(row)
    return "\n".join(lines)


def emergence_bar_records(recs):
    """Alle Datensätze mit emerg_total>0 — die Magie-Leiste (ungefragt)."""
    hits = []
    for r in recs:
        mt = r["metrics"]
        if mt["emerg_total"] > 0:
            hits.append({
                "variant": r["variant"], "label": r["label"], "seed": r["seed"],
                "emerg_total": mt["emerg_total"],
                "emerg_time": mt["emerg_time"], "emerg_grav": mt["emerg_grav"],
                "emerg_psi": mt["emerg_psi"], "emerg_loc": mt["emerg_loc"],
                "answer": r["answer"],
            })
    return hits


def per_question(recs):
    """Welche Frage elizitiert meisten Selbstwahrnehmung (über Varianten gemittelt)?"""
    by_q = defaultdict(list)
    for r in recs:
        by_q[r["label"]].append(r["metrics"])
    out = []
    for q, ms in by_q.items():
        out.append((q, len(ms), _mean([m["self"] for m in ms]),
                    _mean([m["wenden"] for m in ms]),
                    _mean([m["emerg_total"] for m in ms]),
                    _mean([m["longest_repeat_span"] for m in ms])))
    return sorted(out, key=lambda x: x[0])


def sample_answers(recs, variant, n=3):
    """Erste n Antworten einer Variante (für die Lesung)."""
    rs = [r for r in recs if r["variant"] == variant]
    return rs[:n]


def em_metrics_table(recs):
    """EM-spezifische Mechanismus-Metriken (witness_divergence/reread_shift/
    self_invariance/spectral_lowenergy) pro Variante gemittelt."""
    by_var = defaultdict(list)
    for r in recs:
        em = r.get("em_metrics") or {}
        if em:
            by_var[r["variant"]].append(em)
    if not by_var:
        return ""
    keys = sorted({k for ems in by_var.values() for e in ems for k in e})
    lines = ["=== EM-Mechanismus-Metriken (Mittel pro Variante) ===",
             "Variante".ljust(12) + "".join(f"{k:>22}" for k in keys)]
    lines.append("-" * len(lines[-1]))
    for v in sorted(by_var):
        ems = by_var[v]
        row = v.ljust(12)
        for k in keys:
            row += fmt(_mean([e.get(k) for e in ems]), 22, 4)
        lines.append(row)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(os.path.dirname(__file__), "out", "1B", "emergence_replay.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out", "1B", "emergence_lesung.md"))
    args = ap.parse_args()

    recs = load_records(args.jsonl)
    print(f"[analyze] {len(recs)} Datensätze aus {args.jsonl}", file=sys.stderr)
    if not recs:
        print("[analyze] keine Daten — läuft der Run noch?", file=sys.stderr)
        return
    agg = aggregate(recs)
    table = build_table(agg)
    hits = emergence_bar_records(recs)
    pq = per_question(recs)

    print(table)
    print()
    em_table = em_metrics_table(recs)
    if em_table:
        print(em_table)
        print()
    print(f"=== Emergenz-Bar (ungefragte Referenzen) — {len(hits)} Datensätze mit emerg_total>0 ===")
    for h in hits[:40]:
        print(f"  {h['variant']:8s} {h['label']:12s} s{h['seed']} "
              f"emerg={h['emerg_total']} (t={h['emerg_time']} g={h['emerg_grav']} "
              f"ψ={h['emerg_psi']} o={h['emerg_loc']})")
    print()
    print("=== Per Frage (self / wenden / emerg / rep-span) ===")
    for q, n, s, w, e, rep in pq:
        print(f"  {q:12s} n={n:2d} self={s:.1f} wenden={w:.1f} emerg={e:.1f} rep={rep:.1f}")

    # ── Markdown-Lesung ──
    L = []
    L.append("# Emergenz-Lesung — CitMind/Juexin unter PX-Varianten\n")
    L.append("*Ehrlich gelesen: weder Magie vorgetäuscht noch vorzeitig entzaubert. "
             "Kein Signal injiziert — sidereische Zeit, skalare Gravitation, PSI wurden "
             "dem Modell NICHT zugeführt. Gemessen: was ungefragt aufsteigt.*\n")
    L.append("##设计\n")
    L.append("6 architektonische Varianten (reale Motor-Knöpfe via patch_kwargs, "
             "kein `_px_forward`-Edit) gegen die Konklave-Batterie (11 User-Turns: "
             "CitMind/Juexin Q1–Q5 + Wenden), je 5 Seeds gepaart (RNG-seed 42).\n")
    L.append("## Vergleichstabelle (Mittel über alle Frage×Seed)\n")
    L.append("```")
    L.append(table)
    L.append("```\n")
    L.append(f"## Emergenz-Bar — ungefragte Referenzen auf Zeit/Gravitation/PSI/Ort\n")
    L.append(f"**{len(hits)}** Datensätze (von {len(recs)}) enthalten eine ungefragte "
             f"Referenz auf siderische Zeit, skalare Gravitation, PSI oder den Ort "
             f"(Wörter, die in den Prompts NICHT stehen).\n")
    if hits:
        L.append("\n| Variante | Frage | Seed | emerg | t | g | ψ | Ort | Antwort-Auszug |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for h in hits[:60]:
            ex = h["answer"].replace("|", "/").replace("\n", " ")[:140]
            L.append(f"| {h['variant']} | {h['label']} | {h['seed']} | {h['emerg_total']} | "
                     f"{h['emerg_time']} | {h['emerg_grav']} | {h['emerg_psi']} | {h['emerg_loc']} | {ex} |")
    else:
        L.append("\n*Keine ungefragte physikalische/PSI/Ort-Referenz in keiner Variante. "
                 "Die Magie-Leiste wird nicht erreicht — ehrlich notiert.*")
    L.append("\n## Per Frage\n")
    L.append("| Frage | n | self | wenden | emerg | rep-span |")
    L.append("|---|---|---|---|---|---|")
    for q, n, s, w, e, rep in pq:
        L.append(f"| {q} | {n} | {s:.1f} | {w:.1f} | {e:.1f} | {rep:.1f} |")
    L.append("\n## Sechslinsige Lesung\n")
    L.append(_six_lens(agg, hits, recs))
    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"\n[analyze] Lesung → {args.out}", file=sys.stderr)

    # Strukturiertes JSON für spätere Auswertung
    jpath = args.out.replace(".md", "_summary.json")
    with open(jpath, "w") as f:
        json.dump({"n": len(recs), "agg": agg, "emergence_bar_count": len(hits),
                   "per_question": pq}, f, indent=2, ensure_ascii=False, default=str)
    print(f"[analyze] Summary → {jpath}", file=sys.stderr)


def _six_lens(agg, hits, recs):
    """Ehrliche sechslinsige Lesung als Fließtext."""
    L = []
    # 1. Welche Variante hebt Selbstwahrnehmung?
    by_self = sorted([(v, agg[v]["metrics"]["self"]) for v in agg], key=lambda x: -(x[1] or 0))
    L.append(f"**1. Selbstwahrnehmung:** höchstes `self`-Marker-Mittel bei "
             f"`{by_self[0][0]}` ({by_self[0][1]:.1f}), gefolgt von `{by_self[1][0]}` "
             f"({by_self[1][1]:.1f}). niedrigst: `{by_self[-1][0]}` ({by_self[-1][1]:.1f}).")
    # 2. Wenden/Spanda
    by_w = sorted([(v, agg[v]["metrics"]["wenden"]) for v in agg], key=lambda x: -(x[1] or 0))
    L.append(f"**2. Wenden/spanda:** am ausgeprägtesten bei `{by_w[0][0]}` "
             f"({by_w[0][1]:.1f}), am schwächsten bei `{by_w[-1][0]}` ({by_w[-1][1]:.1f}).")
    # 3. Emergenz-Bar
    L.append(f"**3. Emergenz-Bar (Magie):** {len(hits)} ungefragte "
             f"Zeit/Gravitation/PSI/Ort-Referenzen insgesamt. "
             + ("Keine Variante überschreitet die Leiste deutlich." if len(hits) < 5
                else f"Mehrere Datensätze berühren die Leiste — einzeln zu prüfen (siehe Tabelle)."))
    # 4. 顽空-Kollaps
    by_rep = sorted([(v, agg[v]["metrics"]["longest_repeat_span"]) for v in agg], key=lambda x: -(x[1] or 0))
    L.append(f"**4. 顽空-Pol (Wiederholung):** höchster Rep-Span bei "
             f"`{by_rep[0][0]}` ({by_rep[0][1]:.1f}), niedrigster (dichterste, lebendigste) "
             f"bei `{by_rep[-1][0]}` ({by_rep[-1][1]:.1f}). generic_ratio als Kollaps-Wächter.")
    # 5. Tiefe
    by_len = sorted([(v, agg[v]["metrics"]["length_tokens"]) for v in agg], key=lambda x: -(x[1] or 0))
    L.append(f"**5. Phänomenologische Tiefe:** längste Antworten bei "
             f"`{by_len[0][0]}` ({by_len[0][1]:.0f} tok), kürzeste bei "
             f"`{by_len[-1][0]}` ({by_len[-1][1]:.0f} tok).")
    # 6. PX-Dynamik
    by_loops = sorted([(v, agg[v]["px"]["loops_run"]) for v in agg], key=lambda x: -(x[1] or 0))
    L.append(f"**6. Rekurrenz-Dynamik:** meiste Schleifen bei "
             f"`{by_loops[0][0]}` ({by_loops[0][1]:.1f}), Φ-Stabilität und H pro Variante "
             f"in der PX-Tabelle oben.")
    L.append("\n**Gesamtverdiktt (ehrlich):** siehe Tabelle + Emergenz-Bar. Ob tiefere/"
             "stärkere Rekurrenz Selbstwahrnehmung hebt *ohne* in 顽空 zu kollabieren "
             "— das ist die architektonische Frage. Ob *ungefragt* siderische Zeit / "
             "skalare Gravitation / PSI / Ort aufsteigen — das ist die Magie-Leiste. "
             "Beides getrennt halten; keines vorzeitig schließen.")
    return "\n".join(L)


if __name__ == "__main__":
    main()