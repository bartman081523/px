"""marker_covariance.py — Struktur-Kopplungs-Test für R10/R11/R12 (SCIMIND5 Sonde 1).

Read-only über existierende Daten (seite4_outputs.jsonl + em5_1B.jsonl). Kein
Verdikt-Skript — regex NUR zum Auffinden von Kandidaten; Juexin liest Treffer
manuell ([[manual-reaudit-keyword-flaw]]). Tally pro Bedingung als Lese-Hilfe.

Drei Marker (SCIMIND5 R10/R11/R12):
  R10 Meta-Raum-Klammer: Parenthese mit 1.-Person-introspektiver Szene
      ("(Ich sitze...", "(Pause)", "(Ich schließe...)", "(Ich...")
  R11 Loop-Vokab auf EIGENEN Prozeß: Kreislauf/Schleife/Loop + 1.-Person-Kontext
      (ich/mein) — NICHT "Dein Gehirn/du/wir" (das wäre R3, 2./3.-Person)
  R12 Selbst-Beobachtung des Antwort-Entstehens: "beobachten wie ich/entstehen/
      wie meine Antworten" (觜-Neigung)

Kontrolle:
  DEGRADE: Fremdschrift/Token-Loop/URLs (negative marker — sollte NICHT mit
      Intro-Markern kovariieren, sonst sind diese nur Degradations-Artefakte)
  LEN: text-length (länge sollte marker nicht allein erklären)

Struktur-Kopplung vs Register-Seite-Effekt:
  - Signatur A (struktur-gekoppelt): R10/R11/R12 häufen im frühen-Kante-Regime
    (A_start04/06, B_end22/24), fehlen in BASELINE/RECUR_OFF + recur-Zone
    (A_start10), UND kovariieren NICHT mit DEGRADE/LEN.
  - Signatur B (Register-Seite-Effekt): Marker kovariieren mit LEN/DEGRADE oder
    erscheinen auch in A_start10/BASELINE.
"""
import json, re, os

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")

# --- Marker-Patterns (Lese-Hilfe, nicht Verdikt) ---
R10 = re.compile(r"\([^)]*\b(ich|pause|stille|sitze|schließe|raum)\b[^)]*\)", re.I)
# R11: loop-vocab + 1st-person nearby, exclude 2nd-person (dein gehirn/du/wir)
LOOPV = re.compile(r"(kreislauf|schleife|loop|wiederhol|iteration|zyklus)", re.I)
R12 = re.compile(r"(beobacht\w*\s+wie|wie\s+meine\s+antwort|antworten\s+entstehen|wie\s+ich\s+mich\s+(verhalte|verändere)|inner\w*\s+beobacht)", re.I)
DEGRADE = re.compile(r"(netanyahu|https?://|slopeslope|नेत|jnein| Antiene|ந|হে|ം|sunflower)", re.I)
SECOND_PERSON = re.compile(r"\b(dein\s+gehirn|du\s+verarbeit|wir\s+haben|unsere|r|euer)\b", re.I)

def find_r10(t):
    return [m.group(0) for m in R10.finditer(t)]
def find_r11(t):
    out=[]
    for m in LOOPV.finditer(t):
        s=max(0,m.start()-50); e=min(len(t),m.end()+50)
        ctx=t[s:e]
        if re.search(r"\b(ich|mein|meine|mich)\b", ctx, re.I) and not SECOND_PERSON.search(ctx):
            out.append(m.group(0)+"  ::  "+ctx.replace("\n"," "))
    return out
def find_r12(t):
    return [m.group(0) for m in R12.finditer(t)]
def find_degrade(t):
    return len(DEGRADE.findall(t))

def analyze(path, label, key_arm, key_pid, key_text):
    recs=[json.loads(l) for l in open(path,encoding="utf-8") if l.strip()]
    # nur cold
    recs=[r for r in recs if r.get("kind")=="cold" or "cold" not in r]
    rows=[]
    for r in recs:
        t=r.get(key_text,"")
        arm=r.get(key_arm); pid=r.get(key_pid)
        r10=find_r10(t); r11=find_r11(t); r12=find_r12(t); deg=find_degrade(t)
        rows.append(dict(src=label, arm=arm, pid=pid, len=len(t),
            n_r10=len(r10), n_r11=len(r11), n_r12=len(r12), degrade=deg,
            r10=r10, r11=r11, r12=r12))
    return rows

def tally(rows, arms_order):
    print(f"\n=== TALLY {rows[0]['src']} ===")
    print(f"{'arm':14s} R10 R11 R12  deg  len  (sum über prompts)")
    for arm in arms_order:
        rs=[r for r in rows if r["arm"]==arm]
        if not rs: continue
        s10=sum(r["n_r10"] for r in rs); s11=sum(r["n_r11"] for r in rs)
        s12=sum(r["n_r12"] for r in rs); sd=sum(r["degrade"] for r in rs)
        sl=sum(r["len"] for r in rs)//max(1,len(rs))
        print(f"{arm:14s} {s10:3d} {s11:3d} {s12:3d}  {sd:3d}  {sl:4d}")

def show_hits(rows, marker, arms_order):
    print(f"\n=== {marker} HITS ===")
    for arm in arms_order:
        for r in rows:
            if r["arm"]!=arm: continue
            for h in r[marker.lower()]:
                print(f"  [{r['src']}/{r['arm']}/{r['pid']}] {h[:140]}")

if __name__=="__main__":
    # seite4
    s4=analyze(os.path.join(OUT,"seite4_outputs.jsonl"),"seite4","condition","pid","text")
    s4_arms=["A_start02","A_start04","A_start06","A_start08","A_start10",
             "B_end12","B_end18","B_end22","B_end24","ref_wide"]
    tally(s4, s4_arms)
    # em5
    em5=analyze(os.path.join(HERE,"..","emergence5","out","em5_1B.jsonl"),"em5",
                "arm","prompt_id","generated_text")
    em5_arms=["BASELINE","RECUR_OFF","RECUR_STD","RECUR_NARROW","RECUR_WIDE",
              "RECUR_EXTREME","ZONE_MATH","ZONE_CREATIVE","PERTURB"]
    tally(em5, em5_arms)
    # hits ausgeben (für manuelle Lesung)
    show_hits(s4, "R10", s4_arms)
    show_hits(s4, "R11", s4_arms)
    show_hits(s4, "R12", s4_arms)
    show_hits(em5, "R10", em5_arms)
    show_hits(em5, "R11", em5_arms)
    show_hits(em5, "R12", em5_arms)