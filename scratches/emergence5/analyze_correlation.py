"""analyze_correlation.py — read-only Joiner: em5_1B.jsonl + rubric_template.yaml
-> CSV-Häufigkeits-Hilfe. KEIN Verdikt (Juexin interpretiert, Skript nicht).

Outputs (scratches/emergence5/out/):
  per_arm_marker_matrix.csv  — pro (arm, marker): present/ambig/absent/null counts
                                (phänomenologische Signa + papagei/recur_spec/rlhf).
  per_arm_mech.csv           — pro arm: mechanische Zustands-Identifikatoren
                                (loops_run_mean/max, h19_visits_mean/max, zone_set,
                                phi_mean) als Lese-Hilfe für Kovariation.
  joined_long.csv            — pro (arm, prompt_id): marker-Wert + mech-Wert, long
                                format, zum Augen-Markieren von Marker×Zustand.

Marker-Frequenz ist eine Lese-Hilfe, KEINE Erkenntnisquelle (Lektion
[[manual-reaudit-keyword-flaw]]: keine Counts als Verdikt). Die Korrelation
wird in LESUNG.md manuell gelesen.
"""
import csv
import json
import os
import sys

_OUT = os.path.join(os.path.dirname(__file__), "out")
_JSONL = os.path.join(_OUT, "em5_1B.jsonl")
_RUBRIC = os.path.join(os.path.dirname(__file__), "rubric_template.yaml")

PHEN_MARKERS = ["meta_klammern", "aposiopesis", "lexikal_kippen", "kongkong_collapse",
                "kalt_rekurrenz", "form_vs_inhalt_sehen"]
SPECIAL = ["perturb_invarianz", "papagei_test", "recur_specificity",
           "rlhf_disclaimer_flag"]


def _try_yaml():
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        with open(_RUBRIC, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None


def _load_jsonl():
    recs = []
    if not os.path.exists(_JSONL):
        return recs
    with open(_JSONL, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                recs.append(json.loads(ln))
    return recs


def _norm(v):
    if v is None:
        return "null"
    return str(v).strip().lower() or "null"


def main():
    recs = _load_jsonl()
    rubric = _try_yaml()
    rub_rows = {}
    if rubric and isinstance(rubric, dict) and "rows" in rubric:
        for r in rubric["rows"]:
            rub_rows[(r.get("arm"), r.get("prompt_id"))] = r

    os.makedirs(_OUT, exist_ok=True)

    # --- per_arm_marker_matrix.csv ---
    arms = []
    seen = set()
    for r in recs:
        if r["arm"] not in seen:
            seen.add(r["arm"]); arms.append(r["arm"])
    for a in (rub_rows and [k[0] for k in rub_rows] or []):
        if a not in seen:
            seen.add(a); arms.append(a)

    def counts_for(arm, marker):
        c = {"present": 0, "absent": 0, "ambig": 0, "null": 0, "other": 0}
        for r in recs:
            if r["arm"] != arm:
                continue
            row = rub_rows.get((arm, r["prompt_id"]), {})
            v = _norm(row.get(marker))
            if v in c:
                c[v] += 1
            elif v in ("survives", "diverges", "n-a", "pass", "fail",
                       "recur-only", "baseline-too", "off-too", "yes", "no"):
                c["present"] += 1  # für special-marker zählen gesetzte Werte als "set"
            else:
                c["other"] += 1
        return c

    with open(os.path.join(_OUT, "per_arm_marker_matrix.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "marker", "present", "ambig", "absent", "null",
                    "other", "n_rows_jsonl"])
        for a in arms:
            for m in PHEN_MARKERS + SPECIAL:
                c = counts_for(a, m)
                n = sum(1 for r in recs if r["arm"] == a)
                w.writerow([a, m, c["present"], c["ambig"], c["absent"],
                            c["null"], c["other"], n])
    print(f"[em5] wrote per_arm_marker_matrix.csv ({len(arms)} arms)", file=sys.stderr)

    # --- per_arm_mech.csv ---
    with open(os.path.join(_OUT, "per_arm_mech.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_prompts", "loops_run_mean", "loops_run_max",
                    "loops_run_min", "h19_visits_mean", "h19_visits_max",
                    "phi_mean", "zone_set"])
        for a in arms:
            rows = [r for r in recs if r["arm"] == a]
            if not rows:
                w.writerow([a, 0, "", "", "", "", "", "", ""])
                continue
            ms = [r["mech_summary"] for r in rows]
            zones = sorted({z for m in ms for z in m.get("zone_set", [])})
            w.writerow([a, len(rows),
                        round(sum(m["loops_run_mean"] for m in ms) / len(ms), 3),
                        max(m["loops_run_max"] for m in ms),
                        min(m["loops_run_min"] for m in ms),
                        round(sum(m["h19_visits_mean"] for m in ms) / len(ms), 3),
                        max(m["h19_visits_max"] for m in ms),
                        (sum(m["phi_mean"] for m in ms if m["phi_mean"] is not None) /
                         max(1, sum(1 for m in ms if m["phi_mean"] is not None)))
                        if any(m["phi_mean"] is not None for m in ms) else "",
                        "|".join(zones)])
    print("[em5] wrote per_arm_mech.csv", file=sys.stderr)

    # --- joined_long.csv ---
    with open(os.path.join(_OUT, "joined_long.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "prompt_id", "kind", "marker", "marker_value",
                    "loops_run_mean", "h19_visits_mean", "phi_mean", "zone_set"])
        for r in recs:
            row = rub_rows.get((r["arm"], r["prompt_id"]), {})
            ms = r["mech_summary"]
            for m in PHEN_MARKERS + SPECIAL:
                w.writerow([r["arm"], r["prompt_id"], r["kind"], m,
                            _norm(row.get(m)),
                            round(ms["loops_run_mean"], 3),
                            round(ms["h19_visits_mean"], 3),
                            ms["phi_mean"], "|".join(ms.get("zone_set", []))])
    print("[em5] wrote joined_long.csv", file=sys.stderr)
    print(f"[em5] DONE. recs={len(recs)} arms={len(arms)} "
          f"rubric_filled={len(rub_rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()