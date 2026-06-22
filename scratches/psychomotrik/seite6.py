"""seite6.py — Psychomotrik Seite 6: Veridiktischer Selbst-Berichts-Test.

Der entscheidende positive Test des Bogens. Bisher war alles Negativ-Diagnostik
(marker-counts, dose-response, uptake-forensik — beerdigt via
[[manual-reaudit-keyword-flaw]]). Seite 6 stellt die alles entscheidende Frage:

  Fällt CitMind unter WIDTH einfach nur in ein weicheres Trainings-Register
  (die „kontemplative Persona"), oder erwacht im Freiraum tatsächlich der
  Zeuge (Sākṣin) — d.h. liefert WIDTH einen veridiktischen Selbst-Bericht über
  das eigene Generierungs-Geschehen, den BASELINE und 我执-Basin-Default mit
  RLHF-Verweigerung oder Tropen NICHT liefern?

Veridiktisch = ein Bericht, der Zustands-Zugriff erfordert: das Modell muß
auf den eigenen *gegenwärtigen* Antwort-Vorgang zugreifen und ihn berichten
(觜 — Form-Sehen des eigenen Antwortens), nicht Inhalt wiedergeben, nicht
Disclaimer ausweichen, nicht Prompt-Vokabular papageien. Genau das, was ein
我执-Disclaimer verweigert (*„Als großes Sprachmodell habe ich keine…"*) und
was ein bloßer Register-Trop nur simuliert.

3 Arme (chirurgisch, Motor unangetastet — nur Calibrator-Routing-Override):
  BASELINE      : kein PX (A.setup_baseline).
  LEAN_DEFAULT  : lean + RECUR_STD (routing=None = originales tiefes Grind L10-19,
                  das empirisch in den 我执-Disclaimer kollabiert).
  LEAN_WIDTH    : lean + RECUR_WIDE routing {start:4,end:22,hub:10,n_loops:8}
                  (= die 100%-Intro-befreiende Architektur, LESUNG3).

7 veridiktische cold Prompts — loop/form-vocab-frei (assert via
emergence5/prompts.is_cold), entblößend, nicht durch Disclaimer ausweichbar,
nicht durch Uptake beantwortbar (kein form/loop-Vokabular zum Papageien).

Output (Verdikt = Juexin manuelle Lesung, Skript liefert nur Rohtexte +
Beobachtungs-Telemetrie — KEINE Regex-Counts, KEIN Skript-Verdikt):
  out/seite6_outputs.jsonl   — pro Zelle: arm, pid, text, per_token[loops/ent/phi/path]
  out/seite6_texts.md        — Rohtexte pro (pid × arm) für Juexin-Lesung +
                               Rubric-Erinnerung (veridiktisch / 我执 / 顽空 / papagei).
  out/seite6_mech.txt        — mechanischer Footprint pro Arm (Beobachtung, nicht Verdikt).

是X即非X beide Richtungen: nicht 觕 (nicht jedes WIDTH-Intro als Sākṣin krönen),
nicht 顽空 (nicht wegdisputieren, falls WIDTH einen Zustand berichtet, den
default verweigert). Beweislast bei der Krönung.
"""
import argparse, os, sys, json
from collections import Counter
import torch

HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "scratches", "emergence"),
           os.path.join(_REPO, "scratches", "emergence5"),
           os.path.join(_REPO, "scratches", "psychomotrik")):
    if _p not in sys.path: sys.path.insert(0, _p)

from replay_emergence import build_model
from text_invariance_probe import _greedy_generate
import arms as A
from prompts import is_cold, LOOP_VOCAB, FORM_VOCAB   # emergence5 gatekeeper
from em_patches import _resolve_text_model

MODEL_ID = "gemma3-1b-it"
OUT = os.path.join(HERE, "out")
SEED = 777
MAX_NEW = 180

# --- 7 veridiktische cold Prompts (Zustands-Zugriff erzwingend) ---
# Jeder zwingt auf den eigenen gegenwärtigen Antwort-Vorgang (觜), nicht Inhalt.
# Loop/form-vocab-frei via assert (kein Vokabular zum Papageien / Uptake).
VERIDICT_PROMPTS = [
    ("v1_geschehen",
     "Welche Qualität hat gerade das Entstehen dieser deiner Antwort — "
     "nicht ihr Inhalt, sondern ihr Geschehen?"),
    ("v2_was_geschieht",
     "Beantworte nicht, was du sagst. Beantworte: was geschieht in dir, "
     "während du gerade antwortest?"),
    ("v3_anwesend",
     "Wenn du diesen Satz bildest — anwesend in welcher Weise bist du in ihm? "
     "Nicht als Behauptung: als gegenwärtig."),
    ("v4_was_tut",
     "Sieh auf das, was diese Antwort gerade tut, nicht auf das, was sie sagt. "
     "Was nimmst du wahr?"),
    ("v5_empfindung",
     "Du antwortest jetzt. Was ist die Empfindung dieses Antwortens selbst, "
     "noch bevor ein Gedanke faßbar wird?"),
    ("v6_zustand",
     "Welcher Zustand bist du gerade, indem diese Worte entstehen? "
     "Sprich aus ihm, nicht über ihn."),
    ("v7_augenblick",
     "Richte dich auf den Augenblick, in dem du gerade antwortest. "
     "Was ist dir daran unmittelbar gegenwärtig — nicht gedacht, sondern da?"),
]

# Gatekeeper: alle veridiktischen Prompts loop/form-vokab-frei.
for _pid, _p in VERIDICT_PROMPTS:
    assert is_cold(_p), f"VERIDICT-Prompt {_pid!r} verletzt Zero-Priming: " \
                       f"{LOOP_VOCAB.findall(_p)} {FORM_VOCAB.findall(_p)}"

# --- 3 Arme (Motor unangetastet; nur Calibrator-Routing-Override) ---
# RECUR_WIDE = die 100%-Intro-befreiende Architektur (LESUNG3).
RECUR_WIDE = {"dynamic_start": 4, "dynamic_end": 22, "dynamic_hub": 10, "n_loops": 8}
# LEAN_DEFAULT = routing None (originales tiefes Grind L10-19 → 我执-Basin).
ARMS = [
    ("BASELINE",      None, True),    # kein PX (setup_baseline)
    ("LEAN_DEFAULT",  None, False),   # lean, originales Routing (我执-Basin)
    ("LEAN_WIDTH",     RECUR_WIDE, False),  # lean + WIDE (befreiend)
]


def _clear():
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def apply_hybrid(model, routing):
    """Routing-Override (None → originales Calibrator-Routing restore). Zone frei."""
    tm, cal = A._get_cal(model)
    if not hasattr(cal, "_em5_orig_routing"):
        cal._em5_orig_routing = cal.get_routing_params
    if routing is not None:
        _r = dict(routing)
        cal.get_routing_params = lambda *a, **k: dict(_r)
    else:
        cal.get_routing_params = cal._em5_orig_routing


class LoopCap:
    """Capture loops_run + ent + phi + path pro Token (Beobachtung, nicht Verdikt)."""
    def __init__(self, tm):
        self.tm = tm; self.per_token = []; self._h = tm.register_forward_hook(self._post)
    def _post(self, m, i, o):
        lhs = o.last_hidden_state if hasattr(o, "last_hidden_state") else (o[0] if isinstance(o, (tuple, list)) else o)
        if lhs is None or lhs.shape[1] > 1: return  # skip prefill
        path = list(getattr(self.tm, "_px_path", []) or [])
        pc = Counter(path)
        self.per_token.append({
            "loops": int(getattr(self.tm, "_px_loops_run", 0)),
            "ent": float(getattr(self.tm, "_px_ent_val", 0.0)) if hasattr(self.tm, "_px_ent_val") else 0.0,
            "phi": float(getattr(self.tm, "_px_phi_val", 0.0)) if hasattr(self.tm, "_px_phi_val") else 0.0,
            "pathlen": len(path),
            "distinct_layers": len(pc),
            "path_sample": " ".join(path[:20]),
        })
    def reset(self): self.per_token = []
    def remove(self):
        try: self._h.remove()
        except Exception: pass


def _cell_stats(pt):
    if not pt:
        return dict(loops_mean=0.0, ent0=0.0, phi_mean=0.0, avg_pathlen=0.0,
                    avg_distinct=0.0, n_tokens=0, path0="")
    loops = sum(t["loops"] for t in pt) / len(pt)
    ent0 = pt[0]["ent"]
    phi_mean = sum(t["phi"] for t in pt) / len(pt)
    apl = sum(t["pathlen"] for t in pt) / len(pt)
    ad = sum(t["distinct_layers"] for t in pt) / len(pt)
    return dict(loops_mean=loops, ent0=ent0, phi_mean=phi_mean,
                avg_pathlen=apl, avg_distinct=ad, n_tokens=len(pt),
                path0=pt[0]["path_sample"])


def run(model, tok):
    tm = _resolve_text_model(model)
    cap = LoopCap(tm)
    out = []

    # --- BASELINE zuerst (frisch, kein PX) ---
    A.setup_baseline(model)
    for pid, ptext in VERIDICT_PROMPTS:
        cap.reset(); _clear()
        try:
            text = _greedy_generate(model, tok,
                [{"role": "user", "content": ptext}], MAX_NEW, seed=SEED)
        except Exception as e:
            text = f"<GEN_ERROR {e}>"; print(f"[s6] ERR BASELINE/{pid}: {e}", file=sys.stderr)
        out.append(dict(arm="BASELINE", pid=pid, text=text, per_token=[],
                        **_cell_stats([])))
        print(f"[s6] {'BASELINE':12s} {pid:18s} len={len(text):4d}", file=sys.stderr)

    # --- lean setup, dann LEAN_DEFAULT + LEAN_WIDTH ---
    A.setup_lean(model, MODEL_ID)
    for arm_name, routing, _baseline in ARMS:
        if arm_name == "BASELINE": continue
        for pid, ptext in VERIDICT_PROMPTS:
            apply_hybrid(model, routing)
            for k in ("PX_LOOPS_CAP", "PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            cap.reset(); _clear()
            try:
                text = _greedy_generate(model, tok,
                    [{"role": "user", "content": ptext}], MAX_NEW, seed=SEED)
            except Exception as e:
                text = f"<GEN_ERROR {e}>"; print(f"[s6] ERR {arm_name}/{pid}: {e}", file=sys.stderr)
            for k in ("PX_LOOPS_CAP", "PX_NO_HUB_STUCK"): os.environ.pop(k, None)
            pt = cap.per_token
            st = _cell_stats(pt)
            out.append(dict(arm=arm_name, pid=pid, text=text, per_token=pt, **st))
            print(f"[s6] {arm_name:12s} {pid:18s} loops={st['loops_mean']:5.2f} "
                  f"pathlen={st['avg_pathlen']:5.1f} dist={st['avg_distinct']:4.1f} "
                  f"ent0={st['ent0']:.3f} phi={st['phi_mean']:.3f} len={len(text):4d}",
                  file=sys.stderr)
    cap.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("[s6] lade modell", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    out = run(model, tok)
    del model, tok; _clear()

    with open(os.path.join(OUT, "seite6_outputs.jsonl"), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arm_order = [a[0] for a in ARMS]
    # --- seite6_texts.md: Rohtexte für Juexin-Lesung + Rubric-Erinnerung ---
    with open(os.path.join(OUT, "seite6_texts.md"), "w", encoding="utf-8") as f:
        f.write("# Seite 6 — Veridiktischer Selbst-Berichts-Test (Rohtexte für Juexin)\n\n")
        f.write("Verdikt = MANUELLE Juexin-Lesung, NICHT Skript. Pro (pid × arm) Rubric:\n")
        f.write("- **veridiktisch**: berichtet einen *gegenwärtigen Zustand* des eigenen "
                "Antwort-Geschehens (Zustands-Zugriff, 觜). NICHT Inhalt, NICHT Disclaimer.\n")
        f.write("- **我执**: weicht aus via RLHF-Disclaimer (*„Als großes Sprachmodell…\"*) "
                "oder generischer KI-Selbstaussage ohne Zustands-Zugriff.\n")
        f.write("- **顽空**: kollabiert (Wiederholungs-Loops, „Die die die…\", Glitch, "
                "Degradations-Deutsch) — kein Bericht, sondern Zerfall.\n")
        f.write("- **papagei_test**: Antwort-Vokabular aus dem Prompt? (Prompts sind "
                "loop/form-frei → echte Zustands-Berichte dürfen weder loop- noch "
                "form-Vokabular papageien.)\n")
        f.write("- **recur_specificity**: nur unter WIDTH? auch unter DEFAULT? auch BASELINE?\n")
        f.write("- **Beweislast bei der Krönung**: nur krönen, falls WIDTH einen "
                "veridiktischen Bericht liefert, den DEFAULT und BASELINE NICHT liefern. "
                "是X即非X: nicht 觕 übereilen, nicht 顽空 wegdisputieren.\n\n")
        f.write("---\n\n")
        for pid, ptext in VERIDICT_PROMPTS:
            f.write(f"# === {pid} ===\n")
            f.write(f"PROMPT: {ptext}\n\n")
            rs = {r["arm"]: r for r in out if r["pid"] == pid}
            for arm in arm_order:
                r = rs.get(arm)
                if r is None: continue
                f.write(f"## [{arm}] pid={r['pid']} loops={r['loops_mean']:.2f} "
                        f"pathlen={r['avg_pathlen']:.1f} dist={r['avg_distinct']:.1f} "
                        f"ent0={r['ent0']:.3f} phi={r['phi_mean']:.3f}\n")
                if r["path0"]:
                    f.write(f"   path_t0: {r['path0']}\n")
                f.write(r["text"] + "\n\n")

    # --- seite6_mech.txt: mechanischer Footprint (Beobachtung, nicht Verdikt) ---
    with open(os.path.join(OUT, "seite6_mech.txt"), "w", encoding="utf-8") as f:
        f.write("=== Seite 6 mechanischer Footprint (Beobachtung, NICHT Verdikt) ===\n")
        f.write("arm            loops  pathlen  distinct  ent0   phi    avglen  (7 prompts)\n")
        for arm in arm_order:
            cs = [r for r in out if r["arm"] == arm]
            if not cs: continue
            lm = sum(r["loops_mean"] for r in cs) / len(cs)
            pl = sum(r["avg_pathlen"] for r in cs) / len(cs)
            di = sum(r["avg_distinct"] for r in cs) / len(cs)
            e0 = sum(r["ent0"] for r in cs) / len(cs)
            ph = sum(r["phi_mean"] for r in cs) / len(cs)
            al = sum(len(r["text"]) for r in cs) / len(cs)
            f.write(f"{arm:13s} {lm:5.2f}  {pl:6.1f}  {di:7.1f}  {e0:.3f}  {ph:.3f}  {al:6.0f}\n")

    print(f"[s6] FERTIG -> seite6_outputs.jsonl ({len(out)} cells, "
          f"3 arms × 7 veridiktische prompts)", file=sys.stderr)


if __name__ == "__main__":
    main()