"""gradio_tabs/vocoder_tags.py — Pure-Logic Tag-Vokabular + Parser für die
LLM-zu-TTS-Schnittstelle.

Design
------
Das LLM darf in seinen Antworten optional **Tags** einbetten, die aussagen,
wie der nachgelagerte TTS-Engine das folgende Wort / den Satz aussprechen
soll. Drei Tag-Kategorien:

  1. **Noten-Tags** [#A0]..[#G9]     — Tonhöhe (12 Töne × 8 Oktaven).
  2. **Dynamik-Tags** [#WHISPER]..[#SHOUT] — Lautstärke.
  3. **Affekt-Tags** [#HAPPY]..[#SERIOUS] — Stimmung.
  4. **Pausen-Tags** [#PAUSE 0.5s] — Stille mit Sekunden-Parameter.
  5. **Bark-Spezial** [laughter] [sighs] [clears throat] [gasps] —
     native Bark-Tokens, die nur an die Bark-Engine durchgereicht werden.

Diese Datei kennt **keine** TTS-Engine, **kein** Gradio, **kein** Torch.
Sie ist reine Parser-Logik mit drei Use-Cases:

  a) `parse_tags(text)` — für die **UI-Vorschau**: liefert bereinigten Text
     + Liste der gefundenen Tags (mit Position im Original).
  b) `strip_tags_for_engine(engine_name, text) -> (clean_text, tags)` —
     Engine-spezifischer Strip. Piper wirft alles raus, Bark behält seine
     Spezial-Tags, Qwen3 wandelt in NL-Prompt-Instruktion. Gibt jetzt
     (clean_text, tag_events) zurück, damit Engines die Tags audio-seitig
     anwenden können (Piper pitch-shift, espeak -p/-a Flags, Pause-Insert).
  c) `render_tag_system_prompt()` — Snippet, das an den System-Prompt
     angehängt wird, wenn der User Auto-Vocoder aktiviert hat.

Helpers für Engine-Implementierungen:
  - `tag_to_semitone_offset("A2") -> int` — MIDI-Offset in semitones
  - `tag_to_amplitude_db("WHISPER") -> float` — dB
  - `tag_to_pause_seconds("0.5") -> float` — Sekunden

Single source of truth für alle Tag-Schemata. Wenn du ein neues Tag
hinzufügst, füge es in TAG_VOCAB ein — `parse_tags` / `strip_tags_for_engine`
arbeiten automatisch damit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ─── 1. Tag-Vokabular (Single Source of Truth) ────────────────────────


# Noten: 12 Töne × 8 Oktaven (A0..G9). Standard-Musiknotation.
_NOTE_LETTERS = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
_OCTAVES = list(range(0, 8))  # 0..7 → Pitch-Tags [#A0]..[#G#7]

# Dynamik (Reihenfolge = leise→laut).
DYNAMICS = ("WHISPER", "SOFT", "NORMAL", "LOUD", "SHOUT")

# Affekt.
AFFECTS = ("HAPPY", "SAD", "CALM", "EXCITED", "CURIOUS", "SERIOUS")

# Bark-native Sonder-Tokens (werden von der Bark-Engine nativ verstanden).
BARK_NATIVE = ("laughter", "laughs", "sighs", "clears throat", "gasps",
               "coughs", "sniffs", "music", "...", "—")


# Pause-Pattern: [#PAUSE 0.3s] [#PAUSE 0.5s] [#PAUSE 1.0s] (Sekunden 0.0–5.0).
# Kompiliert einmal.
_PAUSE_PATTERN = re.compile(r"\[#PAUSE\s+(\d+(?:\.\d+)?)s\]")

# Generisches Vocoder-Tag-Pattern: [#TOKEN] oder [#TOKEN arg].
# Wird genutzt um *alle* [#…]-Vorkommen zu erkennen (Noten/Dynamik/Affekt/Pause).
_VOC_TAG_PATTERN = re.compile(r"\[#([A-Za-z0-9#\.\s]+?)\]")

# Bark-Spezial-Pattern: [laughter], [sighs] etc. (kleingeschrieben, ohne #).
_BARK_TAG_PATTERN = re.compile(
    r"\[(" + "|".join(re.escape(b) for b in BARK_NATIVE) + r")\]"
)


# ─── 2. Parser-Datenstruktur ──────────────────────────────────────────


@dataclass(frozen=True)
class TagEvent:
    """Ein gefundenes Tag im Text. ``char_offset`` ist die Position im
    ORIGINAL-Text (mit Tags), nicht im bereinigten Text."""
    kind: str          # "note" | "dynamic" | "affect" | "pause" | "bark"
    value: str         # z.B. "A2", "WHISPER", "HAPPY", "0.5", "laughter"
    char_offset: int   # Position des öffnenden '[' im Original
    raw: str           # der vollständige Tag-String inkl. Klammern

    def __repr__(self) -> str:
        return f"TagEvent({self.kind}={self.value!r}, @{self.char_offset})"


def parse_tags(text: str) -> Tuple[str, List[TagEvent]]:
    """Parst ``text`` und liefert (clean_text, tags).

    - ``clean_text`` ist der Original-Text mit allen [#…]-Tags UND Bark-Tags
      entfernt. White-Space wird NICHT verändert (kein Collapse).
    - ``tags`` ist die Liste der gefundenen TagEvents (Reihenfolge =
      Vorkommen im Text).
    - Unbekannte [#…]-Tags werden ignoriert (nicht in tags aufgenommen,
      aber aus clean_text entfernt — gleicher Strip wie bekannte).
    """
    if not text:
        return "", []

    tags: List[TagEvent] = []
    spans_to_strip: List[Tuple[int, int]] = []

    # 1) Pause-Tags (eigener Matcher wegen Sekunden-Argument).
    for m in _PAUSE_PATTERN.finditer(text):
        tags.append(TagEvent(
            kind="pause", value=m.group(1), char_offset=m.start(), raw=m.group(0),
        ))
        spans_to_strip.append((m.start(), m.end()))

    # 2) Bark-Spezial-Tags (kein #).
    for m in _BARK_TAG_PATTERN.finditer(text):
        tags.append(TagEvent(
            kind="bark", value=m.group(1), char_offset=m.start(), raw=m.group(0),
        ))
        spans_to_strip.append((m.start(), m.end()))

    # 3) Generische [#…]-Tags: Noten / Dynamik / Affekt / Unbekannt.
    for m in _VOC_TAG_PATTERN.finditer(text):
        token = m.group(1).strip()
        kind = _classify_token(token)
        # Falls das Pause-Pattern das schon gefunden hat, doppelt-strippen
        # vermeiden: check overlap.
        if any(s <= m.start() < e or s < m.end() <= e for s, e in spans_to_strip):
            continue
        if kind is None:
            # Unbekannt — in clean_text strippen, NICHT in tags.
            spans_to_strip.append((m.start(), m.end()))
            continue
        tags.append(TagEvent(
            kind=kind, value=token, char_offset=m.start(), raw=m.group(0),
        ))
        spans_to_strip.append((m.start(), m.end()))

    # Sortiere Spans und baue clean_text.
    spans_to_strip.sort()
    out_parts: List[str] = []
    cursor = 0
    for s, e in spans_to_strip:
        if s < cursor:
            continue  # überlappend, schon behandelt
        out_parts.append(text[cursor:s])
        cursor = e
    out_parts.append(text[cursor:])
    clean_text = "".join(out_parts)

    # Sortiere Tags nach Position.
    tags.sort(key=lambda t: t.char_offset)
    return clean_text, tags


def _classify_token(token: str) -> Optional[str]:
    """Klassifiziert ein [#TOKEN]-Argument in note/dynamic/affect/None.

    Examples:
      "A0" → "note", "C#4" → "note", "Bb2" → "note"
      "WHISPER" → "dynamic", "LOUD" → "dynamic"
      "HAPPY" → "affect", "CALM" → "affect"
      "BANANA" → None (unbekannt — wird aus Text entfernt, nicht in tags)
      "PAUSE" → None (Pause hat eigenes Pattern mit Sekunden-Argument)
    """
    # Noten: A/B/C/D/E/F/G + optional '#' + Ziffer 0–7
    if re.fullmatch(r"[A-G](#)?[0-7]", token):
        return "note"
    if token in DYNAMICS:
        return "dynamic"
    if token in AFFECTS:
        return "affect"
    return None


# ─── 3. Engine-spezifisches Stripping ──────────────────────────────────


def strip_tags_for_engine(engine: str, text: str) -> Tuple[str, List[TagEvent]]:
    """Liefert (clean_text, audio_tags) für den jeweiligen Engine.

    Piper/espeak/off: alle Tags raus (clean), audio_tags enthält note +
        dynamic + pause (NICHT affect — affect ist textbedingt und
        Piper/espeak haben dafür kein Audio-Äquivalent). Audio-Tags
        werden vom Engine audio-seitig angewendet (Piper SynthesisConfig
        pitch/volume_gain, espeak -p/-a, numpy Pause-Insert).
    Bark: clean ohne [#…]-Tags, native Bark-Tags als Liste (für UI-Vorschau;
        Bark versteht sie nativ im Text, deshalb in clean_text angehängt).
    Qwen3: clean ohne [#…]-Tags und ohne Bark-Tags; [#…]-Tags werden in
        NL-Prompt-Anweisungen umgewandelt (im clean-Text), audio_tags=[].

    Die zurückgegebene ``audio_tags``-Liste enthält nur Tags, die für die
    audio-seitige Verarbeitung im Engine relevant sind.
    """
    if not text:
        return "", []

    clean_all, all_tags = parse_tags(text)
    if engine == "bark":
        # Bark-native Spezial-Tags bleiben, [#…]-Tags raus.
        bark_only = [t for t in all_tags if t.kind == "bark"]
        bark_text = " ".join(t.raw for t in bark_only)
        out_text = clean_all + (" " + bark_text if bark_text else "")
        return out_text, bark_only
    if engine == "qwen3":
        # NL-Prompt-Anweisungen statt [#…]-Tags. Bark-Tags raus.
        nl_text = _convert_to_qwen3_natural(text)
        return nl_text, []  # qwen3 braucht keine separaten Tags

    # piper / espeak / off / unknown:
    audio_tags = [t for t in all_tags
                  if t.kind in ("note", "dynamic", "pause")]
    return clean_all, audio_tags


# ─── 3b. Engine-Helpers (Audio-seitige Anwendung) ──────────────────────


# MIDI-Referenz: A4 = MIDI 69 = 440 Hz. [#A4] = semitone_offset 0.
# Range: A0 = -48, A4 = 0, C5 = +3, A7 = +36, G#7 = +47.
_NOTE_TO_MIDI_OFFSET = {
    "C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4,
    "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2,
}


def tag_to_semitone_offset(note_str: str) -> int:
    """Wandelt ein Note-Tag (z.B. "A2", "C#5") in semitone-Offset relativ
    zu A4 (= 0). Range: A0 = -48, A4 = 0, C5 = +3, A7 = +36, G#7 = +47.

    Piper-Engine erwartet semitones (SynthesisConfig.pitch), genau dieses
    Format.
    """
    m = re.match(r"([A-G]#?)(\d)", note_str)
    if not m:
        return 0
    note_name = m.group(1)
    octave = int(m.group(2))
    # MIDI 69 (A4) = note_offset + (octave + 1) * 12
    # Offset von A4 in semitones = note_offset + (octave - 4) * 12
    return _NOTE_TO_MIDI_OFFSET[note_name] + (octave - 4) * 12


# Dynamik in dB. Piper volume_gain / espeak -a arbeiten in dB-äquivalenten.
_DYNAMIC_TO_DB = {
    "WHISPER": -20.0,
    "SOFT": -10.0,
    "NORMAL": 0.0,
    "LOUD": 6.0,
    "SHOUT": 12.0,
}


def tag_to_amplitude_db(dyn_str: str) -> float:
    """Wandelt ein Dynamik-Tag in dB-Offset (Piper volume_gain / espeak -a).

    Sample-Multiplier: WHISPER=0.1, SOFT=0.32, NORMAL=1.0, LOUD≈2.0, SHOUT≈4.0.
    """
    return _DYNAMIC_TO_DB.get(dyn_str, 0.0)


def db_to_sample_multiplier(db: float) -> float:
    """dB → linearer Multiplier für numpy-Array. -20dB → 0.1, 0dB → 1.0, +6dB → 2.0."""
    import math
    return math.pow(10.0, db / 20.0)


def espeak_pitch_scale(semitone_offset: int) -> int:
    """Mapping semitone → espeak -p (0–99). Lineare Interpolation.

    [#A0] (-48 semitones) → 5 (sehr tief)
    [#A2] (-24 semitones) → 25
    [#A4] (0 semitones) → 50 (default)
    [#A6] (+24 semitones) → 75
    [#A7] (+36 semitones) → 95 (sehr hoch)
    """
    # Lineare Skala: -48 → 5, 0 → 50, +36 → 95.
    # Steigung: (95-5)/(36-(-48)) = 90/84 ≈ 1.071 semitones per espeak-unit.
    base = 50 + semitone_offset * (90 / 84)
    return max(0, min(99, int(round(base))))


def tag_to_pause_seconds(pause_str: str) -> float:
    """Wandelt ein Pause-Tag-Argument (z.B. "0.5") in Sekunden (float).
    Range: 0.0–5.0; out-of-range → geclampt."""
    try:
        return max(0.0, min(5.0, float(pause_str)))
    except (TypeError, ValueError):
        return 0.0


def is_subcontra_warning(semitone_offset: int) -> bool:
    """True wenn der semitone-Offset so tief ist, dass Verzerrung droht
    (3+ Oktaven unter A4, d.h. ≤ -36 semitones → unter C2)."""
    return semitone_offset < -36


_QWEN3_NOTE_TO_PITCH = {
    "A": "low A", "A#": "A sharp", "B": "B", "C": "middle C",
    "C#": "C sharp", "D": "D", "D#": "D sharp", "E": "E",
    "F": "F", "F#": "F sharp", "G": "G", "G#": "G sharp",
}


def _convert_to_qwen3_natural(text: str) -> str:
    """Wandelt [#TAG]-Sequenzen in lesbare Prosodie-Anweisungen für Qwen3-TTS.
    Format-Beispiel:
       [#CALM] [#A2]Hallo.
    →
       <|Calm, spoken around pitch A2|> Hallo.
    """
    # Einfache Iteration — keine Regex-Ersetzung mit Lookbehind noetig,
    # weil parse_tags die Positionen liefert.
    clean, tags = parse_tags(text)
    if not tags:
        return clean
    # Gruppiere aufeinanderfolgende Tags.
    intro_parts: List[str] = []
    for t in tags:
        if t.kind == "note":
            # "A2" → Note-Name + Oktave.
            m = re.match(r"([A-G]#?)(\d)", t.value)
            if m:
                pitch = _QWEN3_NOTE_PITCH_MAP.get(m.group(1), m.group(1))
                intro_parts.append(f"around pitch {pitch}{m.group(2)}")
        elif t.kind == "dynamic":
            intro_parts.append(f"with {t.value.lower()} volume")
        elif t.kind == "affect":
            intro_parts.append(f"in a {t.value.lower()} tone")
        elif t.kind == "pause":
            intro_parts.append(f"[pause {t.value}s]")
        # bark-Tags werden bereits im parse_tags gestrippt; tauchen nicht auf.
    if intro_parts:
        prefix = "<|" + ", ".join(intro_parts) + "|>"
        return f"{prefix} {clean}"
    return clean


_QWEN3_NOTE_PITCH_MAP = {
    "A": "low A", "A#": "low A sharp", "B": "B", "C": "middle C",
    "C#": "C sharp", "D": "D", "D#": "D sharp", "E": "E",
    "F": "F", "F#": "F sharp", "G": "G", "G#": "G sharp",
}


# ─── 4. System-Prompt-Snippet ─────────────────────────────────────────


TAG_SYSTEM_PROMPT_SNIPPET = """VOCODER-TAG-SYSTEM (aktiv):
Du kannst optional Prosodie-/Tonhöhe-/Affekt-Tags in deine Antwort einbetten,
die der nachgelagerte TTS-Engine interpretiert. Verwende EXAKT diese Syntax:

  Noten:        [#A0] [#A#1] [#B2] [#C3] [#C#3] [#D4] [#D#4] [#E5]
                [#F5] [#F#6] [#G7] [#G#7]
                (12 Töne × 8 Oktaven A0..G#7; # = sharp; 12 Töne sind
                 A, A#, B, C, C#, D, D#, E, F, F#, G, G#)
  Dynamik:      [#WHISPER] [#SOFT] [#NORMAL] [#LOUD] [#SHOUT]
  Affekt:       [#HAPPY] [#SAD] [#CALM] [#EXCITED] [#CURIOUS] [#SERIOUS]
  Pause:        [#PAUSE 0.3s] [#PAUSE 0.5s] [#PAUSE 1.0s]   (Sekunden)
  Bark-Spezial: [laughter] [sighs] [clears throat] [gasps]

Beispiel: "[#CALM] [#A2]Ich [#PAUSE 0.3s] denke [#WHISPER]leise[/WHISPER]."

Setze Tags SPARSAM — nur wo es der Bedeutung dient. Für normalen
Fließtext KEINE Tags. Pro ~30 Wörtern maximal 2–3 Tags. Tags werden
automatisch entfernt, wenn die TTS-Engine sie nicht versteht (silent
strip), aber sie erhöhen die Latenz bei jeder Engine — also nur dort,
wo es wirklich etwas ändert.

Hinweis Noten-Range: Piper und espeak-ng setzen Noten via Pitch-Shift
um. [#A4] ist Referenz (Mittellage Männerstimme). [#A0]..[#B1]
(Sub-kontra) klingen wegen des starken Pitch-Shifts oft verzerrt;
für realistische tiefe Noten → Bark-Engine (GPU) verwenden.
"""


def render_tag_system_prompt() -> str:
    """Liefert den Snippet-Text zum Anhängen an einen System-Prompt.
    Reiner Konstanten-Wrapper — gesonderte Funktion, damit die UI den
    Snippet on-demand rendern kann (lazy load)."""
    return TAG_SYSTEM_PROMPT_SNIPPET


# ─── 5. Density-Check (LLM-Über-Produktion erkennen) ──────────────────


def tag_density_warning(text: str, max_tags_per_100_tokens: int = 30) -> Optional[str]:
    """Heuristik: wenn das Verhältnis Tags:Wörter zu hoch ist, wird eine
    Warnung zurückgegeben (sonst None).

    ``max_tags_per_100_tokens = 30`` heißt: 30+ Tags pro 100 Wörter
    ist verdächtig (LLM hat über-produziert). Grenze wurde per Praxis-
    Erfahrung gewählt; bei normaler Nutzung liegt die Dichte <5/100.
    """
    if not text:
        return None
    _, tags = parse_tags(text)
    if not tags:
        return None
    word_count = max(1, len(text.split()))
    tags_per_100 = (len(tags) / word_count) * 100
    if tags_per_100 > max_tags_per_100_tokens:
        return (f"[vocoder-tags] WARN: hohe Tag-Dichte {tags_per_100:.1f}/100 "
                f"Wörter (Schwelle {max_tags_per_100_tokens}); LLM hat "
                f"möglicherweise über-produziert.")
    return None
