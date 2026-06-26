"""Tests für `gradio_tabs/tts_engine.py` — Audio-seitige Vocoder-Anwendung.

Diese Tests prüfen:
  1. _split_text_by_tags — korrekte Segmentierung an Tag-Boundaries
  2. EspeakEngine — `-p`/`-a` Flags durchstellen + Stille einfügen
  3. PiperEngine — SynthesisConfig mit pitch/volume_gain durchstellen
     (gemockt, weil piper-tts nicht in Sandbox installiert)
  4. Sub-kontra-Warnung wird geloggt
  5. Pause-Insert via numpy

Echte Engine-Inferenz wird NICHT getestet (würde Modell-Downloads
auslösen). Wir mocken piper/espeak an den subprocess/PiperVoice-Levels.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gradio_tabs.tts_engine import (  # noqa: E402
    EspeakEngine, PiperEngine, _split_text_by_tags,
)
from gradio_tabs.vocoder_tags import (  # noqa: E402
    parse_tags, tag_to_semitone_offset, TagEvent,
)


# ─── 1. _split_text_by_tags — Segmentierung ──────────────────────


def test_split_empty_text_no_tags():
    """Leere Text + keine Tags → leere Liste."""
    assert _split_text_by_tags("", []) == []


def test_split_text_without_tags_returns_single_segment():
    """Text ohne Tags → ein Segment mit (text, 0, 0.0, 0.0)."""
    segs = _split_text_by_tags("Hallo Welt", [])
    assert segs == [("Hallo Welt", 0, 0.0, 0.0)]


def test_split_text_with_note_tag_only():
    """[#A2] vor "Hallo Welt" → ein Segment mit pitch=-24."""
    _, tags = parse_tags("[#A2]Hallo Welt")
    segs = _split_text_by_tags("[#A2]Hallo Welt", tags)
    assert segs == [("Hallo Welt", -24, 0.0, 0.0)]


def test_split_text_with_pause_inserts_silence_before_next():
    """[#PAUSE 0.5s] zwischen zwei Phrasen → 0.5s Pause vor der zweiten."""
    text = "Erste Phrase[#PAUSE 0.5s]Zweite Phrase"
    _, tags = parse_tags(text)
    segs = _split_text_by_tags(text, tags)
    # 2 Segmente: erste mit pause=0, zweite mit pause=0.5.
    assert len(segs) == 2
    assert segs[0] == ("Erste Phrase", 0, 0.0, 0.0)
    assert segs[1] == ("Zweite Phrase", 0, 0.0, 0.5)


def test_split_text_with_dynamic_tag_changes_amplitude():
    """[#WHISPER] vor Phrase → amplitude_db=-20."""
    text = "[#WHISPER]leise"
    _, tags = parse_tags(text)
    segs = _split_text_by_tags(text, tags)
    assert segs == [("leise", 0, -20.0, 0.0)]


def test_split_text_note_then_text_changes_pitch():
    """[#A2] ändert Pitch der FOLGENDEN Phrase."""
    text = "[#A2]Erste [#A4]Zweite"
    _, tags = parse_tags(text)
    segs = _split_text_by_tags(text, tags)
    assert segs[0] == ("Erste ", -24, 0.0, 0.0)
    assert segs[1] == ("Zweite", 0, 0.0, 0.0)


def test_split_text_combined_note_dynamic_pause():
    """[#CALM][#A2] Phrase [#PAUSE 0.3s][#LOUD] nächste."""
    text = "[#CALM] [#A2]Hallo [#PAUSE 0.3s][#LOUD]Welt"
    _, tags = parse_tags(text)
    segs = _split_text_by_tags(text, tags)
    # Segment 1: "Hallo " mit pitch=-24, amp=0.0 (CALM ist affect, ignoriert).
    # Segment 2: "Welt" mit pitch=-24 (von A2 geerbt), amp=6 (LOUD neu),
    #            pause=0.3.
    assert segs[0] == ("Hallo ", -24, 0.0, 0.0)
    assert segs[1] == ("Welt", -24, 6.0, 0.3)


def test_split_text_bark_tags_ignored():
    """Bark-Native-Tags werden in piper/espeak-Splits ignoriert.
    Da strip_tags_for_engine bereits bark-tags rausfiltert, bekommen wir
    hier nur die audio-relevanten Tags (note/dynamic/pause)."""
    text = "Hallo[laughter]Welt"
    _, all_tags = parse_tags(text)
    audio_tags = [t for t in all_tags
                  if t.kind in ("note", "dynamic", "pause")]
    segs = _split_text_by_tags(text, audio_tags)
    # Keine note/dynamic/pause → ein Segment.
    assert segs == [("Hallo[laughter]Welt", 0, 0.0, 0.0)]


def test_split_text_affect_tags_ignored():
    """Affect-Tags (CALM, HAPPY, ...) werden ignoriert — piper/espeak
    sehen nur EIN Sub-Segment (kein Split an Affect-Boundaries)."""
    text = "[#CALM][#A2]Hallo [#HAPPY]Welt"
    _, tags = parse_tags(text)
    segs = _split_text_by_tags(text, tags)
    # CALM + HAPPY ignoriert → 1 Segment mit pitch=-24 für den ganzen
    # bereinigten Text "Hallo Welt".
    assert segs == [("Hallo Welt", -24, 0.0, 0.0)] or len(segs) == 1
    assert segs[0][1] == -24  # A2 sets pitch for the whole phrase
    assert "Hallo" in segs[0][0]
    assert "Welt" in segs[0][0]  # HAPPY stripped from middle


def test_split_text_pause_at_end_no_segment_after():
    """[#PAUSE 0.5s] am Text-Ende → 0.5s Stille wird an das letzte
    Segment angehängt (pending_pause_s). Piper/Espeak fügen diese
    Stille dann nach dem letzten synthetisierten Chunk ein."""
    text = "Hallo[#PAUSE 0.5s]"
    _, all_tags = parse_tags(text)
    audio_tags = [t for t in all_tags
                  if t.kind in ("note", "dynamic", "pause")]
    segs = _split_text_by_tags(text, audio_tags)
    # 1 Segment mit pause=0.5.
    assert len(segs) == 1
    assert segs[0] == ("Hallo", 0, 0.0, 0.5)


# ─── 2. EspeakEngine: -p/-a Flags durchstellen ─────────────────


def test_espeak_synth_no_tags_uses_default_flags(tmp_path):
    """Ohne Tags: espeak-Call hat -p 50, -a 100 (defaults)."""
    with patch("shutil.which", return_value="/usr/bin/espeak-ng"):
        with patch("subprocess.run") as mock_run:
            eng = EspeakEngine()
            # Erstelle leere WAV die espeak "produzieren würde".
            seg_wav = tmp_path / "seg.wav"
            seg_wav.write_bytes(b"")  # espeak produziert echte WAV, hier gemockt
            mock_run.return_value = MagicMock(returncode=0)
            # Wir mocken espeak so dass er eine gültige WAV schreibt.
            def fake_run(cmd, **kwargs):
                out_path = cmd[cmd.index("-w") + 1]
                # Schreibe 1 Frame Stille in WAV.
                import wave
                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(b"\x00\x00")
                return MagicMock(returncode=0)
            mock_run.side_effect = fake_run

            res = eng.synthesize("Hallo Welt", output_dir=str(tmp_path))
            assert res.engine_name == "espeak"
            # Default-Flags sollten da sein.
            cmd = mock_run.call_args[0][0]
            assert "-p" in cmd
            assert "50" in cmd  # default pitch
            assert "-a" in cmd
            assert "100" in cmd  # default amplitude


def test_espeak_synth_with_note_tag_uses_pitch_flag(tmp_path):
    """[#A2] im Text → espeak -p ≈ 25."""
    with patch("shutil.which", return_value="/usr/bin/espeak-ng"):
        eng = EspeakEngine()
        text = "[#A2]Hallo"
        _, tags = parse_tags(text)
        # Wir mocken espeak + schreiben fake-WAV.
        with patch("subprocess.run") as mock_run:
            def fake_run(cmd, **kwargs):
                out_path = cmd[cmd.index("-w") + 1]
                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(b"\x00\x00")
                return MagicMock(returncode=0)
            mock_run.side_effect = fake_run
            eng.synthesize(text, tags=tags, output_dir=str(tmp_path))
            # Der eine espeak-Call (für die ganze Phrase "Hallo") muss -p ≈ 25 haben.
            cmd = mock_run.call_args[0][0]
            assert "-p" in cmd
            pitch_idx = cmd.index("-p")
            pitch_val = int(cmd[pitch_idx + 1])
            # A2 = -24 semitones → espeak_pitch_scale ≈ 25.
            assert 20 <= pitch_val <= 30


def test_espeak_synth_with_pause_inserts_silence(tmp_path):
    """[#PAUSE 0.5s] im Text → WAV hat zusätzliche Stille."""
    import numpy as np
    with patch("shutil.which", return_value="/usr/bin/espeak-ng"):
        eng = EspeakEngine(sample_rate=22050)
        text = "Hi[#PAUSE 0.5s]da"
        _, tags = parse_tags(text)
        with patch("subprocess.run") as mock_run:
            def fake_run(cmd, **kwargs):
                out_path = cmd[cmd.index("-w") + 1]
                # Sehr kurze WAV produzieren (10 ms Audio).
                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(b"\x00\x00" * 220)  # 10ms Stille
                return MagicMock(returncode=0)
            mock_run.side_effect = fake_run
            res = eng.synthesize(text, tags=tags, output_dir=str(tmp_path))
            # Audio-Dauer sollte >= 0.5s sein (durch Pause-Insert).
            assert res.audio_duration_s >= 0.4


def test_espeak_subcontra_warning_logged(capsys, tmp_path):
    """Bei pitch < -36 semitones wird eine Warnung geloggt."""
    with patch("shutil.which", return_value="/usr/bin/espeak-ng"):
        eng = EspeakEngine()
        text = "[#A0]Hallo"  # A0 = -48 semitones
        _, tags = parse_tags(text)
        with patch("subprocess.run") as mock_run:
            def fake_run(cmd, **kwargs):
                out_path = cmd[cmd.index("-w") + 1]
                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(b"\x00\x00" * 220)
                return MagicMock(returncode=0)
            mock_run.side_effect = fake_run
            eng.synthesize(text, tags=tags, output_dir=str(tmp_path))
            captured = capsys.readouterr()
            assert "sub-kontra" in captured.out.lower() or \
                   "verzerrt" in captured.out.lower()


# ─── 3. PiperEngine: SynthesisConfig pitch/volume_gain ───────────


def test_piper_synth_to_wav_uses_synthesis_config_when_available(tmp_path):
    """Wenn piper.config.SynthesisConfig vorhanden ist, wird es mit
    pitch/volume_gain an PiperVoice.synthesize durchgereicht."""
    out_path = tmp_path / "test.wav"

    # Mock piper Module + SynthesisConfig.
    mock_synth_config_cls = MagicMock()
    mock_synth_config_instance = MagicMock()
    mock_synth_config_cls.return_value = mock_synth_config_instance

    mock_chunk = MagicMock()
    mock_chunk.audio_int16_bytes = b"\x00\x00" * 220  # 10ms
    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = iter([mock_chunk])

    piper_module = MagicMock()
    piper_module.PiperVoice.load.return_value = mock_voice

    piper_config_module = MagicMock()
    piper_config_module.SynthesisConfig = mock_synth_config_cls

    eng = PiperEngine()
    # Inject mocks.
    eng._voice = mock_voice
    eng._synthesis_config_cls = mock_synth_config_cls
    eng._has_synthesis_config = True
    eng._loaded = True  # skip real load

    eng._piper_synth_to_wav("Hallo", pitch_semi=-24.0, amp_db=-10.0, out_path=str(out_path))

    # SynthesisConfig wurde mit den richtigen Werten aufgerufen.
    mock_synth_config_cls.assert_called_once_with(pitch=-24.0, volume_gain=-10.0)
    # PiperVoice.synthesize bekam text + synth_config.
    mock_voice.synthesize.assert_called_once()
    args, kwargs = mock_voice.synthesize.call_args
    assert args[0] == "Hallo"
    assert kwargs.get("synth_config") is mock_synth_config_instance


def test_piper_synth_to_wav_without_synthesis_config(tmp_path):
    """piper-tts < 1.2.0: einfacher Aufruf ohne synth_config."""
    out_path = tmp_path / "test.wav"

    mock_chunk = MagicMock()
    mock_chunk.audio_int16_bytes = b"\x00\x00" * 220
    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = iter([mock_chunk])

    eng = PiperEngine()
    eng._voice = mock_voice
    eng._has_synthesis_config = False  # < 1.2.0
    eng._loaded = True

    eng._piper_synth_to_wav("Hallo", pitch_semi=0.0, amp_db=0.0, out_path=str(out_path))
    # Kein synth_config im Aufruf.
    mock_voice.synthesize.assert_called_once_with("Hallo")
    # WAV existiert.
    assert out_path.exists()


def test_piper_synth_with_pause_inserts_silence(tmp_path):
    """piper-Synthese mit Pause-Tag → WAV hat zusätzliche Stille."""
    eng = PiperEngine(sample_rate=22050)

    # Mock voice + chunk.
    mock_chunk = MagicMock()
    mock_chunk.audio_int16_bytes = b"\x00\x00" * 220  # 10ms
    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = iter([mock_chunk])
    eng._voice = mock_voice
    eng._synthesis_config_cls = MagicMock()
    eng._has_synthesis_config = True
    eng._loaded = True

    text = "Hi[#PAUSE 0.5s]da"
    _, tags = parse_tags(text)

    # Wir können piper nicht echte sub-files schreiben lassen, weil
    # _piper_synth_to_wav.mock_chunk audio_int16_bytes liefert, was
    # 22050Hz-konform ist. Das reicht für Duration-Check.
    res = eng.synthesize(text, tags=tags, output_dir=str(tmp_path))
    # Audio-Dauer >= 0.5s durch Pause-Insert.
    assert res.audio_duration_s >= 0.4


# ─── 4. Tag-System-Integration ─────────────────────────────


def test_full_pipe_text_to_audio_segments():
    """End-to-end: Text mit Tags → Liste von Audio-Segmenten mit
    korrekten Parametern (piper-Semantik)."""
    text = "[#CALM] [#A2]Hallo [#PAUSE 0.3s][#LOUD]Welt [#A4]!"
    _, tags = parse_tags(text)
    # audio_tags sind note/dynamic/pause (KEIN affect).
    audio_tags = [t for t in tags
                  if t.kind in ("note", "dynamic", "pause")]
    segs = _split_text_by_tags(text, audio_tags)

    # Erwartung: 3 Segmente
    #   1. "Hallo " pitch=-24 amp=0.0 pause=0.0
    #   2. "Welt " pitch=-24 amp=6.0 pause=0.3
    #   3. "!" pitch=0 amp=6.0 pause=0.0
    assert len(segs) == 3
    assert segs[0][0] == "Hallo "
    assert segs[0][1] == -24
    assert segs[0][2] == 0.0
    assert segs[0][3] == 0.0
    assert segs[1][0] == "Welt "
    assert segs[1][1] == -24  # A2 inherited
    assert segs[1][2] == 6.0  # LOUD
    assert segs[1][3] == 0.3
    assert segs[2][0] == "!"
    assert segs[2][1] == 0  # A4 explicit
    assert segs[2][2] == 6.0  # LOUD inherited
    assert segs[2][3] == 0.0


def test_pipe_with_only_note_tags():
    """Nur Note-Tags, kein Pause/Dynamic → alle Segmente gleiche Amplitude."""
    text = "[#A2]Hallo [#A4]Welt"
    _, tags = parse_tags(text)
    audio_tags = [t for t in tags if t.kind in ("note", "dynamic", "pause")]
    segs = _split_text_by_tags(text, audio_tags)
    assert segs[0] == ("Hallo ", -24, 0.0, 0.0)
    assert segs[1] == ("Welt", 0, 0.0, 0.0)