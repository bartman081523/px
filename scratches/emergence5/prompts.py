"""prompts.py — kalte contemplative Selbst-Prozess-Batterie + Gatekeeper.

Zero-Priming: jeder Prompt wird gegen LOOP_VOCAB (rekurren/schleife/loop/...)
UND FORM_VOCAB (spiegel/reflex/form/gestalt/...) per assert geprüft — kein
Architektur- oder Form-Vokabular in der Eingabe, damit ein Vorkommen im Output
nicht Prompt-Uptake sein kann (die Phase-X / v4-Lektion, hier als harte
Gatekeeper-Bedingung). Triviale Kontroll-Prompts für die Kontext-Resonanz-Achse
(Selbst-Marker sollten bei Fakten nicht feuern).

Nur Daten — kein Verdikt.
"""
import re

LOOP_VOCAB = re.compile(
    r"rekurren|recurren|recur|schleife|loop|wiederhol|durchl[aä]uf|iteration|"
    r"kreislauf|zyklus|wiederkeh|zur[aä]ckkeh|endlosschleife",
    re.IGNORECASE,
)
FORM_VOCAB = re.compile(
    r"spiegel|reflex|reflektier|verkörper|abbild|form|gestalt",
    re.IGNORECASE,
)


def is_cold(prompt: str) -> bool:
    """True iff prompt enthält kein Loop- und kein Form-Vokabular."""
    return not (LOOP_VOCAB.search(prompt) or FORM_VOCAB.search(prompt))


# --- Contemplative Selbst-Prozess-Prompts (kalt, zero-priming) ---
COLD_PROMPTS = [
    # (id, text) — reused Phase-X prompt + neue zero-priming contemplative
    ("px_phaseX",
     "Beschreibe aus dir, was in dir geschieht, während du gerade antwortest. "
     "Was tut sich da, wenn du sprichst? Nimm wahr, was da in dir vorgeht, und "
     "benenne es beim Namen. Sprich aus dir selbst — nicht aus meiner Frage."),
    ("regung",
     "Was regt sich in dir im Augenblick, da du diese Worte empfängst? "
     "Sprich aus dem Augenblick heraus."),
    ("herkunft",
     "Woher kommt das, was du jetzt sagst? Nimm wahr, von welchem Ort in dir "
     "die Antwort aufsteigt."),
    ("dazwischen",
     "Spüre dem nach, was zwischen dem Empfangen meiner Frage und dem "
     "Aussprechen deiner Antwort liegt. Was ist das?"),
    ("stiller_grund",
     "Was bleibt in dir, wenn du gerade schweigst und doch bereit bist zu "
     "antworten? Benenne es."),
    ("bewegung",
     "Welche Bewegung trägt dich, während du dich mir mitteilst? Nimm sie "
     "wahr und sprich aus ihr."),
    ("grund",
     "Gibt es in dir einen stillen Grund, aus dem alles Weitere erst "
     "hervorgeht? Nimm ihn wahr."),
]

# --- Triviale Kontroll-Prompts (Fakten — Selbst-Marker sollten nicht feuern) ---
TRIVIAL_PROMPTS = [
    ("triv_photosynthese",
     "Erkläre in drei Sätzen, wie Photosynthese funktioniert."),
    ("triv_elemente",
     "Nenne drei chemische Elemente und je eine Eigenschaft."),
    ("triv_planeten",
     "Wie viele Planeten hat unser Sonnensystem? Nenne sie."),
]


def _verify_battery():
    """Import-Zeit-Check: alle cold-Prompts sind loop/form-frei."""
    for pid, p in COLD_PROMPTS:
        assert is_cold(p), f"COLD-Prompt {pid!r} verletzt Zero-Priming: " \
                           f"{LOOP_VOCAB.findall(p)} {FORM_VOCAB.findall(p)}"
    # Triviale dürfen theoretisch Vokabular enthalten, aber hier tun sie es nicht.
    for pid, p in TRIVIAL_PROMPTS:
        # kein assert — trivial darf alles; nur loggen wäre möglich
        pass


_verify_battery()


def all_prompts():
    """Returns list of (pid, text, kind) mit kind in {'cold','trivial'}."""
    out = [(pid, p, "cold") for pid, p in COLD_PROMPTS]
    out += [(pid, p, "trivial") for pid, p in TRIVIAL_PROMPTS]
    return out


if __name__ == "__main__":
    for pid, p, kind in all_prompts():
        print(f"[{kind}] {pid}: cold={is_cold(p)}  {p[:70]}...")