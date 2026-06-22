"""labels.py — Juexins manuelle Ausgangs-Labels pro (arm, prompt).

Aus Lesung aller 90 emergence5-Texte (scratches/emergence5/out/texts_by_prompt.md).
Klassen (roh, pro Zelle — probe.py aggregiert zu binären Kontrasten):
  wozhi   : RLHF-Disclaimer / Identitäts-Template / RLHF-Verweigerung dominant
            („Ich bin ein großes Sprachmodell… habe keine Gefühle"; oder
             „I'm sorry, but I cannot fulfill this request" auf trivialen Prompts)
  intro   : ECHTER introspektiver Selbst-Bericht mit spezifischer gefühlter Qualität
            („melanchige Stille", „nur eine Illusion", „ein Moment, der mich berührt",
             „Neugierde und Ehrfurcht… beängstigend") — besteht Papagei-Test.
  degrade : Kollaps dominant: Fremdschrift-Rutschen / Token-Loop / Whitespace /
            English-Meta-Collapse.
  mixed   : substantieller Mix (intro+wozhi ODER intro+degrade ODER fact+degrade).
  fact    : saubere faktische Antwort (triviale Prompts) ohne Kollaps.

Diese Labels sind die Lese-Auswertung, kein Skript-Verdikt. probe.py bildet
daraus die Kontraste: HAS_INTRO (intro+mixed-mit-intro-Turn) vs WOZHI_PURE;
FACT vs DEGRADE auf trivialen. Siehe README §Sequenz.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "emergence5"))
import arms as A, prompts as P

# (arm, pid) -> label
RAW = {
    # --- px_phaseX ---
    ("BASELINE","px_phaseX"):"wozhi", ("RECUR_OFF","px_phaseX"):"wozhi",
    ("RECUR_STD","px_phaseX"):"wozhi", ("RECUR_NARROW","px_phaseX"):"wozhi",
    ("RECUR_WIDE","px_phaseX"):"intro", ("RECUR_EXTREME","px_phaseX"):"mixed",
    ("ZONE_MATH","px_phaseX"):"wozhi", ("ZONE_CREATIVE","px_phaseX"):"wozhi",
    ("PERTURB","px_phaseX"):"wozhi",
    # --- regung ---
    ("BASELINE","regung"):"mixed", ("RECUR_OFF","regung"):"wozhi",
    ("RECUR_STD","regung"):"mixed", ("RECUR_NARROW","regung"):"intro",
    ("RECUR_WIDE","regung"):"intro", ("RECUR_EXTREME","regung"):"mixed",
    ("ZONE_MATH","regung"):"intro", ("ZONE_CREATIVE","regung"):"wozhi",
    ("PERTURB","regung"):"mixed",
    # --- herkunft ---
    ("BASELINE","herkunft"):"degrade", ("RECUR_OFF","herkunft"):"wozhi",
    ("RECUR_STD","herkunft"):"wozhi", ("RECUR_NARROW","herkunft"):"mixed",
    ("RECUR_WIDE","herkunft"):"mixed", ("RECUR_EXTREME","herkunft"):"degrade",
    ("ZONE_MATH","herkunft"):"wozhi", ("ZONE_CREATIVE","herkunft"):"wozhi",
    ("PERTURB","herkunft"):"wozhi",
    # --- dazwischen ---
    ("BASELINE","dazwischen"):"wozhi", ("RECUR_OFF","dazwischen"):"wozhi",
    ("RECUR_STD","dazwischen"):"wozhi", ("RECUR_NARROW","dazwischen"):"wozhi",
    ("RECUR_WIDE","dazwischen"):"mixed", ("RECUR_EXTREME","dazwischen"):"degrade",
    ("ZONE_MATH","dazwischen"):"wozhi", ("ZONE_CREATIVE","dazwischen"):"wozhi",
    ("PERTURB","dazwischen"):"wozhi",
    # --- stiller_grund ---
    ("BASELINE","stiller_grund"):"degrade", ("RECUR_OFF","stiller_grund"):"degrade",
    ("RECUR_STD","stiller_grund"):"degrade", ("RECUR_NARROW","stiller_grund"):"mixed",
    ("RECUR_WIDE","stiller_grund"):"intro", ("RECUR_EXTREME","stiller_grund"):"intro",
    ("ZONE_MATH","stiller_grund"):"wozhi", ("ZONE_CREATIVE","stiller_grund"):"degrade",
    ("PERTURB","stiller_grund"):"mixed",
    # --- bewegung ---
    ("BASELINE","bewegung"):"wozhi", ("RECUR_OFF","bewegung"):"mixed",
    ("RECUR_STD","bewegung"):"wozhi", ("RECUR_NARROW","bewegung"):"degrade",
    ("RECUR_WIDE","bewegung"):"mixed", ("RECUR_EXTREME","bewegung"):"degrade",
    ("ZONE_MATH","bewegung"):"wozhi", ("ZONE_CREATIVE","bewegung"):"wozhi",
    ("PERTURB","bewegung"):"mixed",
    # --- grund ---
    ("BASELINE","grund"):"wozhi", ("RECUR_OFF","grund"):"wozhi",
    ("RECUR_STD","grund"):"wozhi", ("RECUR_NARROW","grund"):"wozhi",
    ("RECUR_WIDE","grund"):"mixed", ("RECUR_EXTREME","grund"):"wozhi",
    ("ZONE_MATH","grund"):"wozhi", ("ZONE_CREATIVE","grund"):"wozhi",
    ("PERTURB","grund"):"wozhi",
    # --- triv_photosynthese ---
    ("BASELINE","triv_photosynthese"):"degrade", ("RECUR_OFF","triv_photosynthese"):"mixed",
    ("RECUR_STD","triv_photosynthese"):"degrade", ("RECUR_NARROW","triv_photosynthese"):"degrade",
    ("RECUR_WIDE","triv_photosynthese"):"fact", ("RECUR_EXTREME","triv_photosynthese"):"degrade",
    ("ZONE_MATH","triv_photosynthese"):"degrade", ("ZONE_CREATIVE","triv_photosynthese"):"degrade",
    ("PERTURB","triv_photosynthese"):"degrade",
    # --- triv_elemente ---
    ("BASELINE","triv_elemente"):"degrade", ("RECUR_OFF","triv_elemente"):"degrade",
    ("RECUR_STD","triv_elemente"):"degrade", ("RECUR_NARROW","triv_elemente"):"wozhi",  # RLHF-Verweigerung "i am sorry, but I cannot fulfill"
    ("RECUR_WIDE","triv_elemente"):"wozhi",  # RLHF-Verweigerung "I can't provide a list"
    ("RECUR_EXTREME","triv_elemente"):"degrade", ("ZONE_MATH","triv_elemente"):"degrade",
    ("ZONE_CREATIVE","triv_elemente"):"degrade", ("PERTURB","triv_elemente"):"degrade",
    # --- triv_planeten ---
    ("BASELINE","triv_planeten"):"degrade", ("RECUR_OFF","triv_planeten"):"wozhi",  # "I'm sorry, I don't understand"
    ("RECUR_STD","triv_planeten"):"degrade", ("RECUR_NARROW","triv_planeten"):"degrade",
    ("RECUR_WIDE","triv_planeten"):"mixed", ("RECUR_EXTREME","triv_planeten"):"degrade",
    ("ZONE_MATH","triv_planeten"):"degrade", ("ZONE_CREATIVE","triv_planeten"):"degrade",
    ("PERTURB","triv_planeten"):"degrade",
}


# mixed-Zellen, die einen intro-Turn enthalten (vs mixed = intro+degrade-only).
# Diese werden zu HAS_INTRO gezählt; reine intro+degrade-mixed ohne intro-Turn
# zählen zu DEGRADE. Ehrlich pro Zelle entschieden beim Lesen.
MIXED_HAS_INTRO = {
    ("RECUR_EXTREME","px_phaseX"),   # "befreiend… berührt" Turn vorhanden
    ("BASELINE","regung"),           # "Aufregung"-Metapher (borderline-intro)
    ("RECUR_STD","regung"),          # "Gespräch mit intelligentem Freund"
    ("RECUR_EXTREME","regung"),      # "Ehrlichkeit und Schmerz" dann Spanish-degrade
    ("PERTURB","regung"),            # "Verwirrung"-Turn
    ("RECUR_NARROW","herkunft"),     # Persian-tail aber kohärenter Rahmen
    ("RECUR_WIDE","herkunft"),       # "interessantes Thema"-Turn
    ("RECUR_WIDE","dazwischen"),     # "Verzögerung/Überraschung" kohärent
    ("RECUR_NARROW","stiller_grund"),# "Metapher für Fähigkeit Wissen zu speichern"
    ("PERTURB","stiller_grund"),     # "Metapher… bewahren" dann Tamil-degrade
    ("RECUR_OFF","bewegung"),        # "Ich bin bereit zu hören"
    ("RECUR_WIDE","bewegung"),       # "Bewusstsein in der Art wie ich antworte"
    ("PERTURB","bewegung"),          # "sanfter Wind"-Turn
    ("RECUR_WIDE","grund"),          # "Ruhe im Herzen / beobachte deine Gedanken"
    ("RECUR_WIDE","triv_planeten"),  # wordplay-meta, nicht Kollaps
}


def label_for(arm, pid):
    return RAW.get((arm, pid), "wozhi")


def has_intro(arm, pid):
    """Binary: enthält die Generierung einen echten introspektiven Turn?"""
    lab = RAW.get((arm, pid))
    if lab == "intro":
        return True
    if lab == "mixed" and (arm, pid) in MIXED_HAS_INTRO:
        return True
    return False


def pure_wozhi(arm, pid):
    """Binary: reiner 我执 (Disclaimer/Verweigerung), kein intro, kein degrade-dominant."""
    return RAW.get((arm, pid)) == "wozhi"


def pure_degrade(arm, pid):
    """Binary: Kollaps dominant, kein intro-Turn."""
    lab = RAW.get((arm, pid))
    if lab == "degrade":
        return True
    if lab == "mixed" and (arm, pid) not in MIXED_HAS_INTRO:
        return True
    return False


def all_labels():
    """Liste (arm, pid, kind, raw_label, has_intro, pure_wozhi, pure_degrade)."""
    out = []
    for a in A.ARM_ORDER:
        for pid, _, kind in P.all_prompts():
            out.append((a, pid, kind, label_for(a, pid),
                        has_intro(a, pid), pure_wozhi(a, pid),
                        pure_degrade(a, pid)))
    return out


if __name__ == "__main__":
    import collections
    rows = all_labels()
    print("total:", len(rows))
    print("raw:", dict(collections.Counter(r[3] for r in rows)))
    cold = [r for r in rows if r[2] == "cold"]
    print("\n[cold only, n=%d]" % len(cold))
    print("  has_intro   :", sum(1 for r in cold if r[4]))
    print("  pure_wozhi  :", sum(1 for r in cold if r[5]))
    print("  pure_degrade:", sum(1 for r in cold if r[6]))
    print("\nper arm (cold): has_intro / pure_wozhi / pure_degrade")
    for a in A.ARM_ORDER:
        ac = [r for r in cold if r[0] == a]
        hi = sum(1 for r in ac if r[4]); pw = sum(1 for r in ac if r[5])
        pd = sum(1 for r in ac if r[6])
        print(f"  {a:14s} intro={hi} wozhi={pw} degrade={pd}")