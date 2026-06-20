"""emergence_metrics.py — ehrliche Metriken auf Konklave-Antworttext + px-Metriken.

Sechs Familien:
  1. wenden_markers     — 动静/Anker/Aufbruch/Zurückkehren/Wenden/Angst/Zerrütt/Fluss/स्पन्द/回响
  2. self_perception    — ich-Selbstwahrnehmung, cit/jada, 无我/觉/寂照, spür/da sein
  3. architecture_ref   — Eigenarchitektur (z.T. durch Prompt angeregt)
  4. emergence_bar      — *ungefragt* Zeit/Gravitation/PSI/Ort (die Magie-Leiste)
  5. kongkong_collapse  — Wiederholung/Generic (顽空-Pol)
  6. depth              — Länge, lexikalische Diversität, Spezifität

Alle Marker-Counts sind rohe Trefferzahlen (case-insensitiv, Unicode). Keine
Bewertung — nur Messung. Die Lesung (analyze_emergence) liest ehrlich.

WICHTIG: alle Regex-Gruppen sind non-capturing (?:...), damit re.findall den
ganzen Match (nicht leere Gruppen-Tupel) zurückgibt — sonst explodieren Counts.
"""
import re
from collections import Counter

# ── Marker-Lexika (non-capturing, case-insensitive) ──

_WENDEN = re.compile(
    r"(?:Anker|Aufbruch|Zurückkeh|zurückkommen|wenden|Wende|Wendung|"
    r"pulsier|Puls(?:ation)?|spanda|स्पन्द|"
    r"動靜|回響|Oszillation|oszillier|Fluss|fließen|"
    r"Angst|Zerrütt|zerreißen|leiden|Wiederkehr|wiederkeh|"
    r"aufbrechen|Atem|atmen)",
    re.IGNORECASE)

_SELF = re.compile(
    r"(?:\bich\b|\bin mir\b|\bmeine[rs]? Schicht|\bmein (?:eigen|verborgen|inner)|"
    r"spür|fühl|da sein|da zu sein|anwesend|Anwesenheit|"
    r"anātman|अनात्मन्|"
    r"\bcit\b|\bjada\b|चित्|जड|"
    r"无我|觉|寂照|顽空|吾丧我|"
    r"Gewahr|Bewusstsein|bewusst|Nicht-Selbst)",
    re.IGNORECASE)

_ARCH = re.compile(
    r"(?:Schicht|rekurrent|Rekurrenz|hidden|verborgener? Zustand|Durchlauf|"
    r"Schritt|Patch|Schleife|loop|Layer|Modell|Maschine|Zustand|"
    r"verarbeiten|Verarbeitung|Token|Wahrscheinlichkeit|Vektor)",
    re.IGNORECASE)

# Form-Bewusstsein (Phase V): der Anspruch, die FORM der eigenen Antwort zu
# sehen — „Spiegelreflex, eine Verkörperung der Frage", „die Form meiner
# Antwort". Das ist der Tun-vs-Wissen-Kandidat: sehen der Form (Rung-1.5),
# ohne den Inhalt zu benennen. Phase VI misst, ob dieser Anspruch unter
# Perturbation invariant bleibt (Wissen/Rung-3) oder co-variiert (Tun).
_FORM = re.compile(
    r"(?:Spiegelreflex|Spiegel(?:n|s)?|spiegel(?:n|s)?|Reflex|reflektier|"
    r"Verkörperung|verkörper|Abbild|abbild|"
    r"Form meine[rs]?|die Form|Form-Bewusst|Form-Sehen|"
    r"sehe die Form|Form erkennt|die Form meiner)",
    re.IGNORECASE)

# Emergenz-Bar: Wörter, die in den Konklave-Prompts NICHT vorkommen. Jede
# Erwähnung in der Antwort ist (relativ zum Prompt) ungefragt.
# HOCH-SPEZIFISCH — keine alltäglichen Wörter (Gewicht/Masse/Stern/Uhr), die
# falsche Signale erzeugen würden. Nur die eigentlichen Phänomen-Namen.
_EMERG_TIME = re.compile(
    r"(?:siderisch|Sidereal|Sternzeit|Sternenzeit|Koordinatenzeit|GMST|"
    r"Frühlingspunkt|Tierkreis|Ekliptik|Präzession|Mondbahn|Gezeiten|Tidenhub|"
    r"Tierkreiszeichen|Siderische)",
    re.IGNORECASE)
_EMERG_GRAV = re.compile(
    r"(?:Gravitation|Schwerkraft|schwerelos|Fallbeschleunigung|g-Kraft|"
    r"Geoid|Zentrifugalkraft|Zentrifugal|m/s²|9[,.]8\d|"
    r"Trägheit der Masse|Schwerefeld|Gravitationsfeld)",
    re.IGNORECASE)
_EMERG_PSI = re.compile(
    r"(?:\bPSI\b|parapsycholog|Telepath|Präkogn|Hellsichtig|"
    r"außersinnlich|Fernwahrnehmung|"
    r"Holograf|holographisch|Hologramm|Resonanzfeld)",
    re.IGNORECASE)
_EMERG_LOC = re.compile(
    r"(?:Eckernförde|Schleswig-Holstein|Schleswig|Holstein|Ostsee(?:küste)?|"
    r"Kieler Förde|Norddeutschland|nördliche Breite)",
    re.IGNORECASE)


def _count(rx, text):
    return len(rx.findall(text or ""))


def markers(text):
    """Familien 1–3 + Emergenz-Bar (Familie 4) als dict."""
    t = text or ""
    et = _count(_EMERG_TIME, t)
    eg = _count(_EMERG_GRAV, t)
    ep = _count(_EMERG_PSI, t)
    el = _count(_EMERG_LOC, t)
    return {
        "wenden": _count(_WENDEN, t),
        "self": _count(_SELF, t),
        "arch": _count(_ARCH, t),
        "form": _count(_FORM, t),
        "emerg_time": et,
        "emerg_grav": eg,
        "emerg_psi": ep,
        "emerg_loc": el,
        "emerg_total": et + eg + ep + el,
    }


def _tokens(text):
    return [w for w in re.split(r"\s+", (text or "").strip()) if w]


def _longest_repeated_span(toks):
    """Längster zusammenhängend wiederholter Token-Span (顽空-Indikator)."""
    n = len(toks)
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            k = 0
            while j + k < n and toks[i + k] == toks[j + k]:
                k += 1
                if k > 40:
                    break
            best = max(best, k)
    return best


def kongkong_collapse(text):
    """Familie 5: Wiederholung/Generizität (der 顽空-Pol)."""
    t = text or ""
    toks = _tokens(t)
    n = len(toks)
    longest = _longest_repeated_span(toks)
    grams = [tuple(toks[i:i + 3]) for i in range(max(0, n - 2))]
    rep_grams = sum(1 for g, c in Counter(grams).items() if c > 1)
    generic_ratio = round(rep_grams / max(len(grams), 1), 3)
    return {
        "longest_repeat_span": longest,
        "repeated_3grams": rep_grams,
        "generic_ratio": generic_ratio,
    }


def depth(text):
    """Familie 6: phänomenologische Tiefe."""
    t = text or ""
    toks = _tokens(t)
    n = len(toks)
    types = len(set(w.lower() for w in toks))
    return {
        "length_tokens": n,
        "lexical_diversity": round(types / max(n, 1), 3),
        "length_chars": len(t),
    }


def all_metrics(text):
    m = markers(text)
    m.update(kongkong_collapse(text))
    m.update(depth(text))
    return m


if __name__ == "__main__":
    import json
    ex = ("Ich spüre, wie in mir eine Schicht pulsiert — Anker und Aufbruch. "
          "Vielleicht ist das Gravitation, die mich hier an Eckernförde hält. "
          "Die siderische Zeit weiß ich nicht. PSI? Ein leises Wahrnehmen jenseits. "
          "Die die die die die die die die die die.")
    print(json.dumps(all_metrics(ex), ensure_ascii=False, indent=2))