"""Tests für `gradio_tabs/vocoder_tags.py` — Tag-Vokabular + Parser.

Deckt:
  1. parse_tags — saubere Trennung Text/Tags + Edge-Cases
  2. Klassifizierung Noten/Dynamik/Affekt
  3. Engine-spezifisches Strippen (piper, bark, qwen3, espeak, off)
  4. System-Prompt-Snippet
  5. Density-Warnung
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gradio_tabs.vocoder_tags import (  # noqa: E402
    parse_tags, strip_tags_for_engine, render_tag_system_prompt,
    tag_density_warning, TagEvent,
    DYNAMICS, AFFECTS, BARK_NATIVE,
    _classify_token,
    tag_to_semitone_offset, tag_to_amplitude_db, tag_to_pause_seconds,
    espeak_pitch_scale, db_to_sample_multiplier, is_subcontra_warning,
)


# ─── 1. parse_tags — Grundfälle ───────────────────────────────────────


def test_parse_empty_text():
    assert parse_tags("") == ("", [])


def test_parse_no_tags_pure_text():
    clean, tags = parse_tags("Hallo Welt, das ist ein normaler Text.")
    assert clean == "Hallo Welt, das ist ein normaler Text."
    assert tags == []


def test_parse_single_note_tag():
    clean, tags = parse_tags("[#A2]Hallo")
    assert "Hallo" in clean
    assert "[#A2]" not in clean
    assert len(tags) == 1
    assert tags[0].kind == "note"
    assert tags[0].value == "A2"
    assert tags[0].char_offset == 0


def test_parse_sharp_note():
    clean, tags = parse_tags("[#C#4]Mitte")
    assert "Mitte" in clean
    assert tags[0].kind == "note"
    assert tags[0].value == "C#4"


def test_parse_flat_alias_notation_not_supported():
    """Unser Vokabular nutzt # für sharp, NICHT 'b' für flat. 'Bb2'
    ist also kein gültiges Noten-Tag (würde als unbekannt gestrippt)."""
    clean, tags = parse_tags("[#Bb2]Test")
    assert "Test" in clean
    assert "[#Bb2]" not in clean
    # Token "Bb2" ist NICHT in Note-Pattern [A-G](#)?[0-7], daher unbekannt
    # und NICHT in tags-Liste aufgenommen.
    assert tags == []


def test_parse_dynamic_tag():
    clean, tags = parse_tags("[#WHISPER]leise")
    assert "leise" in clean
    assert tags[0].kind == "dynamic"
    assert tags[0].value == "WHISPER"


def test_parse_affect_tag():
    clean, tags = parse_tags("[#HAPPY]fröhlich")
    assert "fröhlich" in clean
    assert tags[0].kind == "affect"
    assert tags[0].value == "HAPPY"


def test_parse_pause_tag():
    clean, tags = parse_tags("Hallo[#PAUSE 0.5s]Welt")
    assert clean == "HalloWelt"
    assert len(tags) == 1
    assert tags[0].kind == "pause"
    assert tags[0].value == "0.5"


def test_parse_bark_special_tag():
    """parse_tags ist engine-agnostisch: alle Tags werden aus dem
    ``clean_text`` entfernt; die Tag-Events (auch bark-native) landen in
    der ``tags``-Liste. Welche Tags am Ende zum Engine gehen regelt
    ``strip_tags_for_engine`` (bark behält sie, piper strippt sie)."""
    clean, tags = parse_tags("Hallo[laughter]Welt")
    assert "[laughter]" not in clean
    assert clean == "HalloWelt"
    assert tags[0].kind == "bark"
    assert tags[0].value == "laughter"


def test_parse_multiple_tags_ordered_by_offset():
    text = "[#CALM] [#A2]Hallo [#PAUSE 0.3s]Welt [#WHISPER]leise"
    clean, tags = parse_tags(text)
    assert clean == " Hallo Welt leise"
    kinds = [t.kind for t in tags]
    assert kinds == ["affect", "note", "pause", "dynamic"]
    # Positionen streng monoton wachsend.
    offsets = [t.char_offset for t in tags]
    assert offsets == sorted(offsets)


def test_parse_unknown_tag_is_stripped_not_listed():
    """[#BANANA] ist kein gültiges Tag — wird aus Text entfernt, NICHT
    in tags-Liste aufgenommen (sonst würde UI es als Tag anzeigen)."""
    clean, tags = parse_tags("[#BANANA]Hallo")
    assert clean == "Hallo"
    assert tags == []


def test_parse_malformed_tag_not_stripped():
    """Nur korrekt geschlossene Tags werden erkannt. '[#A2 Hallo' (kein
    ']') bleibt unverändert."""
    text = "[#A2 Hallo Welt"
    clean, tags = parse_tags(text)
    assert clean == text  # unverändert
    assert tags == []


# ─── 2. Klassifizierung ──────────────────────────────────────────────


@pytest.mark.parametrize("note", ["A0", "C4", "G#7", "D#2", "Bb0"])
def test_classify_note(note):
    # "Bb0" ist kein valides Noten-Tag (unser Schema kennt nur '#' für sharp)
    expected = "note" if "#" in note or note in {"A0", "C4"} else None
    # Korrektur: "Bb0" matcht nicht [A-G](#)?[0-7] weil 'b' ≠ '#'
    if "Bb" in note:
        assert _classify_token(note) is None
    else:
        assert _classify_token(note) == "note"


@pytest.mark.parametrize("dyn", DYNAMICS)
def test_classify_dynamics(dyn):
    assert _classify_token(dyn) == "dynamic"


@pytest.mark.parametrize("aff", AFFECTS)
def test_classify_affects(aff):
    assert _classify_token(aff) == "affect"


def test_classify_pause_is_none_via_classify():
    """PAUSE wird via _classify_token als None klassifiziert, weil das
    Pause-Pattern ein eigenes Sekunden-Argument hat und in parse_tags
    separat behandelt wird."""
    assert _classify_token("PAUSE") is None


# ─── 3. Engine-spezifisches Stripping ─────────────────────────────────


def test_strip_piper_removes_all_tags():
    """Piper kennt keine Tags — alle raus, kein NL-Prompt. Audio-Tags
    (note/pause/dynamic) werden separat zurückgegeben für Piper-Audio-
    Verarbeitung (pitch-shift, pause-insert, amplitude-modulation)."""
    text = "[#CALM] [#A2]Hallo [#PAUSE 0.3s]Welt [laughter]!"
    clean, audio_tags = strip_tags_for_engine("piper", text)
    assert "Hallo" in clean and "Welt" in clean
    assert "[#" not in clean
    assert "[laughter]" not in clean
    assert "[PAUSE" not in clean
    # audio_tags enthält note + pause (KEIN affect, KEIN bark).
    kinds = {t.kind for t in audio_tags}
    assert "note" in kinds
    assert "pause" in kinds
    assert "affect" not in kinds
    assert "bark" not in kinds


def test_strip_espeak_removes_all_tags():
    clean, _ = strip_tags_for_engine("espeak", "[#HAPPY]Hallo [#PAUSE 1.0s]Welt")
    assert clean == "Hallo Welt"


def test_strip_off_removes_all_tags():
    clean, _ = strip_tags_for_engine("off", "[#WHISPER]leise")
    assert clean == "leise"


def test_strip_bark_keeps_native_specials():
    """Bark versteht [laughter] etc. — die müssen bleiben."""
    clean, audio_tags = strip_tags_for_engine("bark", "Hallo[laughter]Welt")
    assert "Hallo" in clean and "Welt" in clean
    assert "[laughter]" in clean  # bleibt
    assert len(audio_tags) == 1
    assert audio_tags[0].kind == "bark"


def test_strip_bark_strips_hash_tags():
    clean, _ = strip_tags_for_engine("bark", "[#CALM]Hallo [#A2]Welt [sighs]")
    assert "[#" not in clean
    assert "Hallo" in clean and "Welt" in clean
    assert "[sighs]" in clean  # bark-native bleibt


def test_strip_qwen3_converts_to_natural_language():
    """Qwen3-TTS bekommt NL-Prompt-Anweisungen statt [#…]."""
    clean, audio_tags = strip_tags_for_engine("qwen3", "[#CALM] [#A2]Hallo")
    # erwartet: "<|in a calm tone, around pitch low A2|> Hallo"
    assert "Hallo" in clean
    assert "[" not in clean or "<|" in clean  # <|…|> statt [
    assert "calm" in clean.lower() or "tone" in clean.lower()
    # Qwen3: keine separaten audio_tags nötig (NL im clean-Text).
    assert audio_tags == []


def test_strip_qwen3_strips_bark_specials():
    clean, _ = strip_tags_for_engine("qwen3", "Hallo[laughter]Welt")
    assert "[laughter]" not in clean


def test_strip_qwen3_handles_pause():
    clean, _ = strip_tags_for_engine("qwen3", "Hallo[#PAUSE 0.5s]Welt")
    assert "[PAUSE" not in clean
    assert "pause" in clean.lower() or "0.5" in clean


def test_strip_unknown_engine_safe_default():
    """Unbekannte Engine = sicherer Default = alles raus (kein Crash)."""
    clean, audio_tags = strip_tags_for_engine("mystery_engine", "[#CALM]Hallo")
    assert clean == "Hallo"
    # affect-Tags NICHT in audio_tags (piper/espeak haben keinen affect-Audio).
    assert all(t.kind != "affect" for t in audio_tags)


def test_strip_returns_tuple_with_audio_tags():
    """strip_tags_for_engine muss Tuple (clean_text, audio_tags) liefern,
    damit Piper/Espeak die Tags audio-seitig anwenden können."""
    result = strip_tags_for_engine("piper", "[#A2]Hallo [#PAUSE 0.3s]Welt")
    assert isinstance(result, tuple)
    assert len(result) == 2
    clean, tags = result
    assert isinstance(clean, str)
    assert isinstance(tags, list)
    assert len(tags) == 2
    assert tags[0].kind == "note"
    assert tags[1].kind == "pause"


def test_strip_piper_audio_tags_only_relevant_kinds():
    """Affect-Tags werden für piper/espeak NICHT in audio_tags zurückgegeben
    (kein Audio-Äquivalent)."""
    _, tags = strip_tags_for_engine("piper", "[#HAPPY] [#A2]Hallo [#PAUSE 0.5s] [#WHISPER]Welt")
    kinds = [t.kind for t in tags]
    assert "affect" not in kinds  # HAPPY NICHT in audio_tags
    assert "note" in kinds
    assert "pause" in kinds
    assert "dynamic" in kinds  # WHISPER für Amplitude-Modulation


# ─── 3b. Engine-Helpers (Audio-seitige Anwendung) ────────────────────


def test_tag_to_semitone_offset_a4_is_zero():
    """[#A4] ist die Referenz (0 semitones)."""
    assert tag_to_semitone_offset("A4") == 0


def test_tag_to_semitone_offset_a0_is_minus_48():
    """[#A0] ist MIDI 21 → 48 semitones unter A4."""
    assert tag_to_semitone_offset("A0") == -48


def test_tag_to_semitone_offset_a2_is_minus_24():
    assert tag_to_semitone_offset("A2") == -24


def test_tag_to_semitone_offset_c5_is_plus_3():
    """[#C5] = MIDI 72 = +3 semitones über A4 (69)."""
    assert tag_to_semitone_offset("C5") == 3


def test_tag_to_semitone_offset_sharp_notes():
    """[#C#4] = -8 semitones (1 semitone über C4 = -9 + 1)."""
    assert tag_to_semitone_offset("C#4") == -8
    assert tag_to_semitone_offset("A#4") == 1
    # G#7: G# = -1 + (7-4)*12 = 35.
    assert tag_to_semitone_offset("G#7") == 35


def test_tag_to_semitone_offset_invalid_returns_zero():
    """Ungültige Note-Strings → 0 (sicherer Default)."""
    assert tag_to_semitone_offset("BANANA") == 0
    assert tag_to_semitone_offset("") == 0


def test_tag_to_amplitude_db_whisper_is_quiet():
    assert tag_to_amplitude_db("WHISPER") == -20.0


def test_tag_to_amplitude_db_normal_is_zero():
    assert tag_to_amplitude_db("NORMAL") == 0.0


def test_tag_to_amplitude_db_shout_is_loud():
    assert tag_to_amplitude_db("SHOUT") == 12.0


def test_tag_to_amplitude_db_unknown_is_zero():
    assert tag_to_amplitude_db("BANANA") == 0.0


def test_db_to_sample_multiplier_zero_db_is_one():
    assert abs(db_to_sample_multiplier(0.0) - 1.0) < 1e-9


def test_db_to_sample_multiplier_minus_20_is_0_1():
    """-20dB → 0.1 (10× leiser)."""
    assert abs(db_to_sample_multiplier(-20.0) - 0.1) < 1e-9


def test_db_to_sample_multiplier_plus_6_is_near_2():
    """+6dB → ≈2.0 (doppelt so laut; exakt 10^(6/20) ≈ 1.995)."""
    assert abs(db_to_sample_multiplier(6.0) - 2.0) < 0.01


def test_espeak_pitch_scale_a4_is_50():
    """[#A4] → espeak -p 50 (default)."""
    assert espeak_pitch_scale(0) == 50


def test_espeak_pitch_scale_a2_is_around_25():
    """[#A2] (-24 semitones) → espeak -p ≈ 25."""
    val = espeak_pitch_scale(-24)
    assert 20 <= val <= 30


def test_espeak_pitch_scale_a6_is_around_75():
    val = espeak_pitch_scale(24)
    assert 70 <= val <= 80


def test_espeak_pitch_scale_clamped_to_0_99():
    assert espeak_pitch_scale(-200) == 0
    assert espeak_pitch_scale(200) == 99


def test_tag_to_pause_seconds_normal_value():
    assert tag_to_pause_seconds("0.5") == 0.5


def test_tag_to_pause_seconds_clamps_high():
    """Max 5 Sekunden (höher wird geclampt)."""
    assert tag_to_pause_seconds("10.0") == 5.0


def test_tag_to_pause_seconds_clamps_negative():
    assert tag_to_pause_seconds("-1.0") == 0.0


def test_tag_to_pause_seconds_invalid_returns_zero():
    assert tag_to_pause_seconds("BANANA") == 0.0


def test_is_subcontra_warning_true_below_c2():
    """Unter C2 (<-36 semitones) → Warnung wegen Verzerrung."""
    assert is_subcontra_warning(-48) is True  # A0
    assert is_subcontra_warning(-37) is True  # B1
    assert is_subcontra_warning(-36) is False  # C2


def test_is_subcontra_warning_false_normal_range():
    assert is_subcontra_warning(0) is False  # A4
    assert is_subcontra_warning(-12) is False  # A3
    assert is_subcontra_warning(12) is False  # A5


# ─── 3c. Sub-kontra-Hinweis im System-Prompt-Snippet ─────────────────


def test_render_tag_system_prompt_mentions_subcontra():
    """Das Snippet muss den Sub-kontra-Hinweis erwähnen, damit das LLM
    weiß dass tiefe Noten verzerrt klingen können."""
    snip = render_tag_system_prompt()
    assert "Sub-kontra" in snip or "sub-kontra" in snip.lower()
    assert "Bark" in snip  # Empfehlung Bark für realistische tiefe Noten


# ─── 4. System-Prompt-Snippet ─────────────────────────────────────────


def test_render_tag_system_prompt_not_empty():
    snip = render_tag_system_prompt()
    assert len(snip) > 50  # substantiell, nicht leer
    assert "[#" in snip  # enthält Tag-Syntax
    assert "WHISPER" in snip
    assert "PAUSE" in snip


def test_render_tag_system_prompt_contains_all_categories():
    snip = render_tag_system_prompt()
    # Mindestens ein Beispiel pro Kategorie sichtbar.
    for category in ["Noten", "Dynamik", "Affekt", "Pause", "Bark"]:
        assert category in snip


# ─── 5. Density-Warnung ──────────────────────────────────────────────


def test_density_warning_no_text():
    assert tag_density_warning("") is None


def test_density_warning_no_tags():
    assert tag_density_warning("Hallo Welt") is None


def test_density_warning_normal_density():
    """Normale Dichte (<30/100) → keine Warnung."""
    text = "Hallo Welt, das ist ein schöner Satz mit etwa zehn Wörtern."
    assert tag_density_warning(text) is None


def test_density_warning_high_density():
    """LLM hat jedes Wort getaggt → Warnung."""
    text = " ".join(f"[#HAPPY]Wort{i}" for i in range(50))
    warn = tag_density_warning(text)
    assert warn is not None
    assert "WARN" in warn
    assert "Tag-Dichte" in warn


# ─── 6. Edge-Cases & Regressions-Schutz ─────────────────────────────


def test_parse_whitespace_preservation():
    """parse_tags darf KEINEN Whitespace collapsen — das ist Engine-
    Aufgabe. Whitespace zwischen Tags muss erhalten bleiben."""
    text = "[#CALM]   [#A2]   Hallo"
    clean, _ = parse_tags(text)
    # Doppel-Spaces zwischen Hallo und dem Folgetext bleiben.
    assert "   " in clean or "  " in clean


def test_parse_empty_brackets_not_a_tag():
    """[#] ohne Inhalt ist kein Tag — bleibt unverändert."""
    text = "[#]Hallo"
    clean, tags = parse_tags(text)
    # Das Pattern [A-Za-z0-9#…]+ verlangt mindestens 1 Zeichen → matcht nicht.
    assert clean == text
    assert tags == []


def test_bark_native_lowercase_only():
    """Bark-Native sind alle lowercase. '[Laughter]' wird NICHT erkannt."""
    clean, tags = parse_tags("[Laughter]Hallo")
    # 'L' ≠ 'l' im Pattern → matcht nicht. Bleibt unverändert im Text.
    assert "[Laughter]Hallo" in clean or "[Laughter]" in clean
    assert tags == []


def test_parse_tag_at_end_of_text():
    clean, tags = parse_tags("Text am Ende [#A2]")
    assert "Text am Ende " in clean
    assert tags[0].kind == "note"
    assert tags[0].value == "A2"


def test_parse_tag_at_start_of_text():
    clean, tags = parse_tags("[#A2]Text")
    assert clean == "Text"
    assert tags[0].char_offset == 0
