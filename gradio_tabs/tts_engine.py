"""gradio_tabs/tts_engine.py — Pure-Logic TTS-Engine-Abstraktion.

Diese Datei kennt **kein** Gradio, **kein** Torch-Modell-Loading zur
Import-Zeit. Sie definiert:

  1. ``SynthResult`` — strukturierte Rückgabe (filepath + duration + latency).
  2. ``TTSEngine`` — Abstract Base für alle Engines.
     Jede Engine hat ein ``tier``-Attribut (``"cpu"`` oder ``"gpu"``)
     und ein ``_device``-Attribut (zur Laufzeit aufgelöst).
  3. ``EspeakEngine`` — Default via espeak-ng subprocess (cpu, immer
     verfügbar, keine Modell-Downloads, realtime-tauglich).
  4. ``PiperEngine`` — Piper ONNX (cpu, ~60 MB, lazy load). Versteht
     SynthesisConfig(pitch=, volume_gain=) → Noten/Dynamik audio-seitig.
  5. ``BarkEngine`` — Bark (gpu, transformers, ~2 GB Download beim
     ersten Call). Auto-Swap auf CPU wenn VRAM nicht reicht.
  6. ``Qwen3Engine`` — Qwen3-TTS 0.6B/1.7B (gpu, via offizielles
     ``qwen-tts``-PyPI-Package, ~2.4 GB beim ersten Call). NICHT via
     ``transformers.AutoModel`` — die ``qwen3_tts``-Architektur ist noch
     nicht in mainline-transformers gemerged. Auto-Swap auf CPU wenn
     VRAM nicht reicht.
  7. ``OffEngine`` — No-op (TTS deaktiviert).
  8. ``make_engine(name)`` — Factory + automatische Fallback-Hierarchy.
  9. ``_resolve_device(eng_name, llm_active_vram_mb)`` — wählt
     ``"cuda"`` oder ``"cpu"`` für gpu-tier Engines je nach freiem VRAM.

Latenz-Tracking
---------------
Jeder Engine-Aufruf misst:
  - ``ttfa_ms`` — Time-to-first-audio (Millisekunden vom synthesize()-Call
    bis der erste Audio-Frame im File steht). Für one-shot-Engines = die
    komplette Synthese-Zeit.
  - ``rtf`` — Real-Time-Factor = synth_dauer / audio_duration. <1.0 heißt
    realtime-schneller; >1.0 = langsamer als realtime.

Diese Metriken landen in ``SynthResult`` und werden vom Latenz-Benchmark
(``scratches/tts_benchmark``) ausgewertet.

Audio-seitige Vocoder-Tag-Anwendung (Plan 5.1)
----------------------------------------------
Piper und espeak bekommen ``tags: list[TagEvent]`` und wenden sie
audio-seitig an:

  - ``note``-Tag → Piper ``SynthesisConfig.pitch`` (semitones) oder
    espeak ``-p`` (0-99 Skala). Sub-Phrasen-Synthese nötig, weil Piper
    Pitch nur pro Synthese-Call konstant ist.
  - ``pause``-Tag → numpy-Stille-Insert (sample_rate × duration_s).
  - ``dynamic``-Tag → Piper ``SynthesisConfig.volume_gain`` (dB) oder
    espeak ``-a`` (0-200 Skala). Sample-Multiplier bei numpy-Merge.

Noten unter -36 semitones (≈C2) lösen eine Konsole-Warnung aus
("sub-kontra vermutlich verzerrt").
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import wave
import wave as _wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# Optionaler Import: numpy für Pause-Insert + Amplitude-Modulation.
try:
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ─── 1. Datenstrukturen ──────────────────────────────────────────────


@dataclass
class SynthResult:
    """Rückgabe einer ``synthesize(text)``-Operation."""
    filepath: str           # Absoluter Pfad zur WAV-Datei
    sample_rate: int        # Tatsächliche Sample-Rate der WAV
    audio_duration_s: float # Dauer in Sekunden
    synth_time_s: float     # Reine Synthese-Zeit (CPU)
    ttfa_ms: float          # Time-to-first-audio in ms
    rtf: float              # Real-Time-Factor (<1.0 = realtime-schneller)
    engine_name: str
    text_length_chars: int
    word_count: int
    extra: Dict[str, str] = field(default_factory=dict)


# ─── 2. Abstract Base ────────────────────────────────────────────────


class TTSEngine(ABC):
    """Abstrakte Basis für alle TTS-Engines.

    Jede Engine hat:
      - ``name: str`` (Klassen-Attribut) — engine-id (piper, bark, …)
      - ``tier: str`` (Klassen-Attribut) — "cpu" oder "gpu"
      - ``_device: str`` (Instanz-Attribut) — "cuda" oder "cpu", zur
        Laufzeit via ``_resolve_device()`` aufgelöst. Default = "cpu".

    ``synthesize()`` bekommt ``tags: list[TagEvent]`` als optionalen
    Parameter und wendet sie audio-seitig an (siehe Modul-Docstring).
    """

    name: str = "abstract"
    tier: str = "cpu"  # "cpu" oder "gpu"

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._loaded = False
        self._device: str = "cpu"  # wird in _ensure_loaded() ggf. überschrieben

    @abstractmethod
    def synthesize(self, text: str, tags: Optional[Sequence] = None,
                   output_dir: Optional[str] = None) -> SynthResult:
        """Synthetisiert ``text`` und schreibt eine WAV-Datei.

        ``tags``: optionale Liste von ``TagEvent``s (aus vocoder_tags).
            Piper/espeak wenden note/dynamic/pause audio-seitig an.
            Bark/Qwen3 ignorieren tags (Bark hat eigene Mechanismen).

        ``output_dir``: Zielordner (default = tempdir). Die Engine legt
        dort eine eindeutig benannte WAV-Datei an.

        Die Implementierung SOLLTE ``self._ensure_loaded()`` zuerst rufen.
        """
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        """Lazy-load Hook. Subklassen überschreiben wenn nötig.
        Idempotent: ein zweiter Aufruf ist ein No-Op."""
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        """Zur Laufzeit aufgelöstes Gerät: "cuda" oder "cpu"."""
        return self._device


# ─── 2b. Device-Resolver (GPU↔CPU Auto-Swap) ─────────────────────────


# TTS-Engine-Modell-Größen in MB (konservativ).
_ENGINE_MODEL_MB = {
    "bark": 2000,    # ~2 GB
    "qwen3": 650,    # ~650 MB Safetensors
}
# Sicherheitsmarge: 90% des freien VRAM darf genutzt werden.
_VRAM_USAGE_FRACTION = 0.90


def _resolve_device(eng_name: str, llm_active_vram_mb: int = 0) -> str:
    """Wählt "cuda" oder "cpu" für ``eng_name``.

    Logik:
      1. Wenn ``eng_name`` nicht gpu-tier ist → "cpu".
      2. Wenn torch nicht verfügbar oder kein CUDA → "cpu".
      3. Wenn free_vram < engine_model_mb → "cpu" (passt nicht).
      4. Wenn llm_active_vram_mb + engine_model_mb > 90% total_vram → "cpu"
         (würde OOM riskieren).
      5. Sonst: "cuda".

    ``llm_active_vram_mb``: aktuell vom LLM belegter VRAM (Default 0).
        Wird vom Caller (chat_tab) aus dem ModelManager abgefragt.
        Bei 0 wird angenommen, dass die GPU leer ist (TTS nach Stream-Ende).
    """
    if _ENGINE_MODEL_MB.get(eng_name, 0) == 0:
        return "cpu"  # piper/espeak sind cpu-tier.

    try:
        import torch  # noqa: F401
    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    engine_mb = _ENGINE_MODEL_MB[eng_name]
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024)
    except Exception:
        # mem_get_info kann auf manchen Setups fehlschlagen → konservativ CPU.
        return "cpu"

    # Wenn das Engine-Modell allein nicht passt → CPU.
    # Wir verlangen engine_mb + 200MB Sicherheitsmarge (für KV-Cache,
    # activations, etc.).
    if free_mb < engine_mb + 200:
        return "cpu"
    # Wenn LLM + Engine > 90% total VRAM → CPU.
    if llm_active_vram_mb + engine_mb > total_mb * _VRAM_USAGE_FRACTION:
        return "cpu"
    return "cuda"


# ─── 3. OffEngine (No-op) ────────────────────────────────────────────


class OffEngine(TTSEngine):
    """No-op: gibt leeren String-Result zurück. Für 'TTS aus'."""
    name = "off"

    def __init__(self, sample_rate: int = 24000):
        super().__init__(sample_rate=sample_rate)
        self._loaded = True  # No-op braucht keinen Load.

    def synthesize(self, text: str, tags: Optional[Sequence] = None,
                   output_dir: Optional[str] = None) -> SynthResult:
        start = time.perf_counter()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SynthResult(
            filepath="",
            sample_rate=0,
            audio_duration_s=0.0,
            synth_time_s=0.0,
            ttfa_ms=elapsed_ms,
            rtf=0.0,
            engine_name="off",
            text_length_chars=len(text),
            word_count=len(text.split()),
            extra={"reason": "TTS disabled (off)"},
        )


# ─── 4. EspeakEngine (Default + Fallback) ────────────────────────────


class EspeakEngine(TTSEngine):
    """espeak-ng via subprocess. Immer verfügbar (wenn espeak-ng installiert),
    kein Modell-Download, realtime-tauglich. Stark robotic, aber verständlich.

    Tier: cpu. Audio-seitige Tag-Anwendung:
      - note-Tag → espeak -p (0-99 Skala)
      - dynamic-Tag → espeak -a (0-200 Skala)
      - pause-Tag → numpy-Stille-Insert (falls numpy verfügbar;
        sonst wird die Pause ignoriert).
    """

    name = "espeak"
    tier = "cpu"

    def __init__(self, sample_rate: int = 22050, voice: str = "de",
                 espeak_bin: Optional[str] = None):
        super().__init__(sample_rate=sample_rate)
        self.voice = voice
        self.espeak_bin = espeak_bin or shutil.which("espeak-ng") or shutil.which("espeak")
        self.available = self.espeak_bin is not None

    def _ensure_loaded(self) -> None:
        # espeak-ng braucht keinen Model-Load; nur Verfügbarkeit prüfen.
        if not self.available:
            raise RuntimeError(
                "espeak-ng nicht gefunden. Installiere mit: "
                "sudo apt install espeak-ng (Debian/Ubuntu) oder "
                "brew install espeak (macOS)."
            )
        self._loaded = True

    def synthesize(self, text: str, tags: Optional[Sequence] = None,
                   output_dir: Optional[str] = None) -> SynthResult:
        self._ensure_loaded()
        if not text.strip():
            return SynthResult("", 0, 0.0, 0.0, 0.0, 0.0, self.name,
                               len(text), 0, {"reason": "empty text"})

        start = time.perf_counter()
        out_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"espeak_{int(start * 1000)}.wav"

        # Lazy-Import der vocoder_tags-Helpers (kein Top-Level-Import
        # → verhindert zirkuläre Abhängigkeit tts_engine ↔ vocoder_tags).
        from gradio_tabs.vocoder_tags import (
            tag_to_semitone_offset, espeak_pitch_scale as _ep_scale,
            tag_to_amplitude_db, tag_to_pause_seconds, is_subcontra_warning,
            TagEvent,
        )

        # Wenn keine Tags: ein espeak-Call reicht.
        if not tags:
            cmd = [
                self.espeak_bin,
                "-v", self.voice,
                "-s", "175",
                "-p", "50",     # Default-Pitch
                "-a", "100",    # Default-Amplitude
                "-w", str(out_file),
                text,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"espeak-ng fehlgeschlagen (rc={e.returncode}): "
                    f"{e.stderr.decode(errors='replace')[:200]}"
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("espeak-ng Timeout (>60s)")
        else:
            # Mit Tags: splitte Text an Tag-Boundaries, synthetisiere
            # Sub-Phrasen mit je eigenen -p/-a Flags. Pause-Tags werden
            # als Stille eingefügt (numpy bevorzugt, sonst wav-Header-Trick).
            segments = _split_text_by_tags(text, tags)
            audio_chunks: List[bytes] = []
            for seg_text, seg_pitch_semitone, seg_amplitude_db, seg_pause_s in segments:
                # 1. Pause (Stille als rohe PCM-Bytes oder leer).
                if seg_pause_s > 0:
                    if _HAS_NUMPY:
                        silence = (_np.zeros(int(self.sample_rate * seg_pause_s),
                                            dtype=_np.int16)).tobytes()
                        audio_chunks.append(silence)
                    # ohne numpy: Pause wird übersprungen.
                # 2. Sub-Phrase synthetisieren (wenn Text vorhanden).
                if seg_text.strip():
                    pitch_arg = str(_ep_scale(seg_pitch_semitone))
                    # espeak -a ist 0-200, mapping dB → 0-200:
                    # -20dB → 30, 0dB → 100, +12dB → 160 (linear-clamped).
                    amp_arg = str(max(0, min(200,
                                              int(100 + seg_amplitude_db * 5))))
                    sub_file = out_dir / f"espeak_seg_{int(time.perf_counter()*1e6)}.wav"
                    cmd = [
                        self.espeak_bin,
                        "-v", self.voice,
                        "-s", "175",
                        "-p", pitch_arg,
                        "-a", amp_arg,
                        "-w", str(sub_file),
                        seg_text,
                    ]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(
                            f"espeak-ng seg fehlgeschlagen: "
                            f"{e.stderr.decode(errors='replace')[:200]}"
                        )
                    # PCM-Bytes lesen und anhängen.
                    with _wave.open(str(sub_file), "rb") as wf:
                        audio_chunks.append(wf.readframes(wf.getnframes()))
                    try:
                        sub_file.unlink()
                    except OSError:
                        pass
                # Sub-kontra-Warnung bei sehr tiefen Noten.
                if seg_pitch_semitone < -36:
                    print(f"[tts] espeak: sub-kontra {seg_pitch_semitone} semitones "
                          f"→ Ausgabe vermutlich verzerrt. Für realistische "
                          f"tiefe Noten → Bark (GPU) verwenden.")
            # Merge: WAV-Header für out_file schreiben + PCM-Chunks.
            if not audio_chunks:
                # Kein Audio produziert → stille 1s WAV.
                _write_silent_wav(str(out_file), self.sample_rate, 1.0)
            else:
                with _wave.open(str(out_file), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b"".join(audio_chunks))

        synth_time = time.perf_counter() - start

        # WAV-Header lesen für Dauer + tatsächliche Sample-Rate.
        sample_rate, audio_duration = _read_wav_metadata(str(out_file))

        return SynthResult(
            filepath=str(out_file),
            sample_rate=sample_rate,
            audio_duration_s=audio_duration,
            synth_time_s=synth_time,
            ttfa_ms=synth_time * 1000,  # one-shot: TTFA = synth_time
            rtf=synth_time / audio_duration if audio_duration > 0 else 0.0,
            engine_name=self.name,
            text_length_chars=len(text),
            word_count=len(text.split()),
        )


# ─── 4b. Helper: Text an Tag-Boundaries splitten ──────────────────────


def _strip_qwen3_tags(text: str, tags: Optional[Sequence]) -> tuple:
    """Strippt audio-Tags für Qwen3-TTS und baut natural-language Instructs.

    Qwen3-TTS hat keine direkte pitch/amplitude-API. Note/Dynamic-Tags
    werden in einen deutschen Instruct-Text übersetzt, der das Modell
    bittet, den Stil anzupassen.

    Rückgabe: (clean_text_without_audio_tags, audio_tags_list)
    """
    if not text:
        return "", []
    if not tags:
        return text, []
    # Strippe note/dynamic/pause-Tags per Regex (Fallback wie in
    # strip_tags_for_engine).
    import re as _re
    audio_kinds = {"note", "dynamic", "pause"}
    audio_tags = [t for t in tags if t.kind in audio_kinds]
    if not audio_tags:
        return text, []
    spans = sorted(
        {(t.char_offset, t.char_offset + len(t.raw)) for t in tags}
    )
    clean_parts = []
    cursor = 0
    for s, e in spans:
        if s < cursor:
            continue
        clean_parts.append(text[cursor:s])
        cursor = e
    if cursor < len(text):
        clean_parts.append(text[cursor:])
    clean = "".join(clean_parts).strip()
    return clean, audio_tags


def _split_text_by_tags(text: str, tags: Sequence) -> List:
    """Splittet ``text`` an Tag-Boundaries in Sub-Phrasen.

    Rückgabe: Liste von (sub_text, pitch_semitone, amplitude_db, pause_s)
    Tupeln. Ein Audio-Tag (note/dynamic/pause) ändert den STATE
    (pitch/amplitude/pending_pause), der ab dem darauf folgenden
    Sub-Text-Block gilt.

    Wichtig: ``text`` ist der ORIGINAL-Text MIT Tags. Die
    zurückgegebenen ``sub_text`` enthalten KEINE Tags mehr (clean).

    Beispiel: ``"[#CALM] [#A2]Hallo [#LOUD]Welt"``
      → [("Hallo ", -24, 0.0, 0.0), ("Welt", -24, 6.0, 0.0)]
      (CALM ist affect, wird ignoriert; A2 setzt pitch=-24, das gilt
      für "Hallo "; LOUD setzt amp=6, gilt ab "Welt".)
    """
    from gradio_tabs.vocoder_tags import (
        tag_to_semitone_offset, tag_to_amplitude_db, tag_to_pause_seconds,
    )

    if not text:
        return []
    if not tags:
        return [(text, 0, 0.0, 0.0)]

    # Sortiere ALLE Tags nach Position.
    sorted_tags = sorted(tags, key=lambda t: t.char_offset)

    # Wir bauen Tag-Spans aus ALLEN Tags und nutzen diese als
    # Strip-Menge — auch wenn der Caller nur audio_tags liefert
    # (dann enthält die Strip-Menge nur die Audio-Tag-Spans).
    # Für sauberes Strippen von affect/bark-Tags muss der Caller ALLE
    # Tags liefern. Fallback: regex-basierte Erkennung.
    spans = sorted(
        [(t.char_offset, t.char_offset + len(t.raw)) for t in sorted_tags]
    )

    # Regex-Fallback: wenn die Tag-Spans nicht den ganzen Original-
    # Text abdecken (z.B. weil Caller nur audio_tags gegeben hat),
    # suchen wir zusätzlich nach Tags per Regex.
    import re as _re
    _TAG_RE = _re.compile(r"\[#[A-Z0-9#.\s]+\]")
    # Wir erweitern spans nur, wenn das Lücken hinterlässt.
    covered = [(s, e) for s, e in spans]
    for m in _TAG_RE.finditer(text):
        ms, me = m.start(), m.end()
        # Ist dieser Match bereits abgedeckt?
        already = any(s <= ms and me <= e for s, e in covered)
        if not already:
            covered.append((ms, me))
    spans = sorted(covered)

    clean_parts: List[str] = []
    cursor = 0
    for s, e in spans:
        if s < cursor:
            continue
        clean_parts.append(text[cursor:s])
        cursor = e
    if cursor < len(text):
        clean_parts.append(text[cursor:])
    clean_text = "".join(clean_parts)

    # Mapping orig_offset → clean_offset.
    # Wir wollen die Position des TAG-STARTS (s) im clean_text, also:
    # clean_offset = s - (Summe aller Spans, die VOR s enden, also
    # deren e ≤ s).
    def _orig_to_clean(orig_offset: int) -> int:
        delta = 0
        for s, e in spans:
            if e <= orig_offset:
                delta += (e - s)
        return max(0, orig_offset - delta)

    # Iteriere NUR durch audio-Tags.
    audio_sorted = [t for t in sorted_tags
                    if t.kind in ("note", "dynamic", "pause")]

    segments: List = []
    pitch = 0
    amplitude_db = 0.0
    pending_pause_s = 0.0

    # 1. Pre-Tag-Block (alles vor dem ersten Audio-Tag, mit default-state).
    # Wird verworfen, wenn es nur aus Tags + Whitespace besteht (kein
    # echter Sprach-Input).
    if audio_sorted:
        first_clean_off = _orig_to_clean(audio_sorted[0].char_offset)
        pre_text = clean_text[:first_clean_off].strip()
        if pre_text:
            segments.append((clean_text[:first_clean_off], pitch, amplitude_db, pending_pause_s))
        cursor_clean = first_clean_off
    else:
        # Keine Audio-Tags → ein Segment.
        return [(clean_text, 0, 0.0, 0.0)] if clean_text else []

    for tag in audio_sorted:
        # Setze State für ALLE Segmente bis zum NÄCHSTEN Audio-Tag.
        if tag.kind == "note":
            pitch = tag_to_semitone_offset(tag.value)
        elif tag.kind == "dynamic":
            amplitude_db = tag_to_amplitude_db(tag.value)
        elif tag.kind == "pause":
            pending_pause_s += tag_to_pause_seconds(tag.value)
        # Bestimme End-Position dieses Blocks (Anfang des nächsten
        # Audio-Tags, oder Textende).
        idx = audio_sorted.index(tag)
        if idx + 1 < len(audio_sorted):
            next_off = _orig_to_clean(audio_sorted[idx + 1].char_offset)
        else:
            next_off = len(clean_text)
        block_text = clean_text[cursor_clean:next_off]
        if block_text:
            segments.append((block_text, pitch, amplitude_db, pending_pause_s))
            pending_pause_s = 0.0
        cursor_clean = next_off

    if pending_pause_s > 0:
        # Trailing-Stille an das letzte Segment anhängen, nicht als
        # eigenes Segment. piper/espeak fügen diese Stille nach dem
        # letzten synthetisierten Chunk ein.
        if segments:
            text_, p_, a_, _ = segments[-1]
            segments[-1] = (text_, p_, a_, pending_pause_s)
        else:
            segments.append(("", pitch, amplitude_db, pending_pause_s))
    return segments


# ─── 5. Piper Engine (lazy load, ONNX) ──────────────────────────────


class PiperEngine(TTSEngine):
    """Piper TTS via ONNX-Runtime. Erstes ``synthesize`` löst den
    Model-Download + Load aus. Danach gecached im Speicher.

    Tier: cpu. ONNX-Runtime ist CPU-realtime-tauglich.

    Audio-seitige Tag-Anwendung (Plan 5.1):
      - note-Tag → SynthesisConfig.pitch (semitones, piper >= 1.2.0)
      - dynamic-Tag → SynthesisConfig.volume_gain (dB)
      - pause-Tag → numpy-Stille-Insert

    Voraussetzungen: ``piper-tts`` (pip install piper-tts), >= 1.2.0 für
    SynthesisConfig. Falls nicht vorhanden → ``RuntimeError`` beim
    ersten Aufruf (Caller fängt ab und fällt auf espeak-ng zurück).
    """

    name = "piper"
    tier = "cpu"

    def __init__(self, sample_rate: int = 22050, voice_model: Optional[str] = None):
        super().__init__(sample_rate=sample_rate)
        self.voice_model = voice_model or "de_DE-thorsten-medium"
        self._voice = None  # piper voice object (lazy)
        self._synthesis_config_cls = None
        self._has_synthesis_config = False  # piper-tts < 1.2.0 fallback

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from piper import PiperVoice  # type: ignore
            from piper.config import SynthesisConfig  # type: ignore
            self._synthesis_config_cls = SynthesisConfig
            self._has_synthesis_config = True
        except ImportError as e:
            raise RuntimeError(
                "piper-tts nicht installiert. Installiere mit: "
                "pip install piper-tts"
            ) from e
        # Download/Load Voice (Piper macht das intern, ggf. nach .onnx).
        try:
            self._voice = PiperVoice.load(self.voice_model)
        except Exception as e:
            raise RuntimeError(
                f"PiperVoice.load('{self.voice_model}') fehlgeschlagen: {e}"
            ) from e
        self._loaded = True

    def synthesize(self, text: str, tags: Optional[Sequence] = None,
                   output_dir: Optional[str] = None) -> SynthResult:
        self._ensure_loaded()
        if not text.strip():
            return SynthResult("", 0, 0.0, 0.0, 0.0, 0.0, self.name,
                               len(text), 0, {"reason": "empty text"})

        start = time.perf_counter()
        out_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"piper_{int(start * 1000)}.wav"

        if not tags:
            # Kein Tag: einfache Synthese.
            self._piper_synth_to_wav(text, 0.0, 0.0, str(out_file))
        else:
            # Mit Tags: splitten in Sub-Phrasen.
            segments = _split_text_by_tags(text, tags)
            audio_chunks: List[bytes] = []
            sample_rate_used = self.sample_rate
            for seg_text, pitch_semi, amp_db, pause_s in segments:
                if pause_s > 0 and _HAS_NUMPY:
                    silence = (_np.zeros(int(self.sample_rate * pause_s),
                                        dtype=_np.int16)).tobytes()
                    audio_chunks.append(silence)
                if seg_text.strip():
                    seg_file = out_dir / f"piper_seg_{int(time.perf_counter()*1e6)}.wav"
                    self._piper_synth_to_wav(
                        seg_text, float(pitch_semi), float(amp_db), str(seg_file),
                    )
                    with _wave.open(str(seg_file), "rb") as wf:
                        sample_rate_used = wf.getframerate()
                        audio_chunks.append(wf.readframes(wf.getnframes()))
                    try:
                        seg_file.unlink()
                    except OSError:
                        pass
                if pitch_semi < -36:
                    print(f"[tts] piper: sub-kontra {pitch_semi} semitones → "
                          f"Ausgabe vermutlich verzerrt. Für realistische "
                          f"tiefe Noten → Bark (GPU) verwenden.")
            if not audio_chunks:
                _write_silent_wav(str(out_file), self.sample_rate, 1.0)
            else:
                with _wave.open(str(out_file), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate_used)
                    wf.writeframes(b"".join(audio_chunks))

        synth_time = time.perf_counter() - start
        actual_sr, audio_duration = _read_wav_metadata(str(out_file))

        return SynthResult(
            filepath=str(out_file),
            sample_rate=actual_sr,
            audio_duration_s=audio_duration,
            synth_time_s=synth_time,
            ttfa_ms=synth_time * 1000,
            rtf=synth_time / audio_duration if audio_duration > 0 else 0.0,
            engine_name=self.name,
            text_length_chars=len(text),
            word_count=len(text.split()),
            extra={"voice_model": self.voice_model,
                   "tags_applied": len(tags) if tags else 0,
                   "synthesis_config": self._has_synthesis_config},
        )

    def _piper_synth_to_wav(self, text: str, pitch_semi: float,
                            amp_db: float, out_path: str) -> None:
        """Piper-Synthese mit optionaler SynthesisConfig (pitch/volume_gain)."""
        if not self._has_synthesis_config:
            # piper-tts < 1.2.0: kein SynthesisConfig → einfache API.
            with _wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                for chunk in self._voice.synthesize(text):
                    wf.writeframes(chunk.audio_int16_bytes)
            return
        # piper-tts >= 1.2.0: SynthesisConfig mit pitch/volume_gain.
        synth_config = self._synthesis_config_cls(
            pitch=float(pitch_semi),
            volume_gain=float(amp_db),
        )
        with _wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            for chunk in self._voice.synthesize(text, synth_config=synth_config):
                wf.writeframes(chunk.audio_int16_bytes)


# ─── 6. Bark Engine (lazy load, ~2 GB) ───────────────────────────────


class BarkEngine(TTSEngine):
    """suno/bark via Transformers. Erstes ``synthesize`` löst den
    Model-Download + Load aus.

    Tier: gpu (transformers.AutoModel). Auto-Swap auf cpu wenn VRAM
    nicht reicht via _resolve_device().

    Skip via Env-Variable ``SKIP_BARK_DOWNLOAD=1``.
    """

    name = "bark"
    tier = "gpu"

    def __init__(self, sample_rate: int = 24000, model_size: str = "small",
                 llm_active_vram_mb: int = 0):
        super().__init__(sample_rate=sample_rate)
        self.model_size = model_size  # "small" oder "base/full"
        self.llm_active_vram_mb = llm_active_vram_mb
        self._model = None
        self._processor = None
        # Device wird in _ensure_loaded via _resolve_device bestimmt.
        self._device = _resolve_device("bark", llm_active_vram_mb)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if os.environ.get("SKIP_BARK_DOWNLOAD") == "1":
            raise RuntimeError("Bark-Download per SKIP_BARK_DOWNLOAD=1 übersprungen")
        try:
            from transformers import AutoProcessor, AutoModel  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers nicht installiert (Bark braucht es)."
            ) from e

        model_id = {
            "small": "suno/bark-small",
            "base": "suno/bark",
        }.get(self.model_size, "suno/bark-small")
        print(f"[tts] Lade Bark ({model_id}, ~2 GB) auf {self._device}… "
              f"bitte warten.", flush=True)
        # Re-resolve device jetzt (zwischen __init__ und _ensure_loaded
        # kann sich der VRAM-State geändert haben, falls LLM dazwischen
        # entladen wurde).
        self._device = _resolve_device("bark", self.llm_active_vram_mb)
        try:
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModel.from_pretrained(model_id).to(self._device)
        except Exception as e:
            # GPU-OOM oder Modell-Load-Fehler → RuntimeError, Caller
            # fängt ab und fällt auf espeak/off zurück.
            raise RuntimeError(
                f"Bark-Load auf {self._device} fehlgeschlagen: {e}"
            ) from e
        self._loaded = True

    def synthesize(self, text: str, tags: Optional[Sequence] = None,
                   output_dir: Optional[str] = None) -> SynthResult:
        self._ensure_loaded()
        if not text.strip():
            return SynthResult("", 0, 0.0, 0.0, 0.0, 0.0, self.name,
                               len(text), 0, {"reason": "empty text"})

        start = time.perf_counter()
        import torch  # nur bei Bark-Init geladen, nicht am Modul-Top.
        out_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"bark_{int(start * 1000)}.wav"

        inputs = self._processor(text=[text], return_tensors="pt").to(self._device)
        with torch.no_grad():
            audio_array = self._model.generate(**inputs, do_sample=True)
        # audio_array kann auf GPU oder CPU sein — wir holen's auf CPU für wav.
        audio = audio_array.cpu().numpy().squeeze()

        import numpy as np  # bark-Output ist np.ndarray
        import wave as _wave
        audio_int16 = (audio * 32767).astype(np.int16)
        with _wave.open(str(out_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        synth_time = time.perf_counter() - start
        actual_sr, audio_duration = _read_wav_metadata(str(out_file))

        return SynthResult(
            filepath=str(out_file),
            sample_rate=actual_sr,
            audio_duration_s=audio_duration,
            synth_time_s=synth_time,
            ttfa_ms=synth_time * 1000,
            rtf=synth_time / audio_duration if audio_duration > 0 else 0.0,
            engine_name=self.name,
            text_length_chars=len(text),
            word_count=len(text.split()),
            extra={"model_size": self.model_size, "device": self._device},
        )


# ─── 7. Qwen3 Engine (lazy load, Safetensors via transformers) ──────


class Qwen3Engine(TTSEngine):
    """Qwen3-TTS via offizielles ``qwen-tts``-Package (~0.6B–1.7B).

    Stand 2026-06: Qwen3-TTS end-to-end-Modelle sind public (0.6B und
    1.7B Base/CustomVoice/VoiceDesign). Inferenz läuft via
    ``qwen_tts.Qwen3TTSModel`` (eigenes PyTorch-Package, NICHT
    transformers.AutoModel — ``qwen3_tts``-Architektur ist noch nicht
    in mainline-transformers gemerged).

    Tier: gpu (PyTorch + bf16). Auto-Swap auf cpu wenn VRAM nicht reicht.

    Tags: Qwen3-TTS ist ein end-to-end-Modell ohne Vocoder-Post-
    Processing-Schnittstelle. Note/Pitch-Tags werden vor der Synthese
    in natürliche Sprach-Instructs umgewandelt (Plan 5.1, D5).

    Skip via Env-Variable ``SKIP_QWEN3_DOWNLOAD=1``.
    """

    name = "qwen3"
    tier = "gpu"

    def __init__(self, sample_rate: int = 24000,
                 model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                 language: str = "German",
                 llm_active_vram_mb: int = 0):
        super().__init__(sample_rate=sample_rate)
        self.model_id = model_id
        self.language = language
        self.llm_active_vram_mb = llm_active_vram_mb
        self._model = None
        # Device wird in _ensure_loaded via _resolve_device bestimmt.
        self._device = _resolve_device("qwen3", llm_active_vram_mb)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if os.environ.get("SKIP_QWEN3_DOWNLOAD") == "1":
            raise RuntimeError(
                "Qwen3-Download per SKIP_QWEN3_DOWNLOAD=1 übersprungen"
            )
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "qwen-tts nicht installiert. Installiere mit: "
                "pip install qwen-tts"
            ) from e

        print(f"[tts] Lade Qwen3-TTS ({self.model_id}) auf "
              f"{self._device}… bitte warten.", flush=True)
        # Re-resolve device jetzt (zwischen __init__ und _ensure_loaded
        # kann sich VRAM-State geändert haben).
        self._device = _resolve_device("qwen3", self.llm_active_vram_mb)
        try:
            import torch
            self._torch = torch  # Referenz für synthesize()
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=f"{self._device}:0" if self._device == "cuda"
                           else "cpu",
                dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
            )
            self._end_to_end_available = True
        except Exception as e:
            raise RuntimeError(
                f"Qwen3-TTS-Load auf {self._device} fehlgeschlagen: {e}"
            ) from e
        self._loaded = True

    def synthesize(self, text: str, tags: Optional[Sequence] = None,
                   output_dir: Optional[str] = None) -> SynthResult:
        self._ensure_loaded()
        if not text.strip():
            return SynthResult("", 0, 0.0, 0.0, 0.0, 0.0, self.name,
                               len(text), 0, {"reason": "empty text"})

        start = time.perf_counter()
        out_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"qwen3_{int(start * 1000)}.wav"

        # Tags werden in natürliche Sprach-Instructs übersetzt (Plan 5.1, D5).
        # Qwen3-TTS hat keine direkte pitch/amplitude-API → Mod-Text + instruct.
        clean_text, audio_tags = _strip_qwen3_tags(text, tags)

        try:
            # Base-Modell (0.6B/1.7B Base) hat keine CustomVoice-Stimmen,
            # aber generate_voice_clone(x_vector_only_mode=True) erlaubt
            # Sprechen ohne ref_text (nur x-Vektor-Stil). Wir nutzen das
            # als Default-Pfad.
            # 1s Sinus-Ton als Dummy-Reference (Modell braucht was).
            import numpy as np
            sr_ref = self.sample_rate
            t_ref = np.linspace(0, 0.5, sr_ref // 2, endpoint=False)
            dummy_ref = (0.01 * np.sin(2 * np.pi * 220 * t_ref)).astype(np.float32)

            # Falls User echte Stimme will: ref_audio übergeben. Aktuell:
            # synth-Input ist nur Text → dummy-ref.
            kwargs = {}
            if "Base" in self.model_id:
                kwargs["x_vector_only_mode"] = True
                wavs, sr = self._model.generate_voice_clone(
                    text=clean_text,
                    language=self.language,
                    ref_audio=(dummy_ref, sr_ref),
                    **kwargs,
                )
            elif "CustomVoice" in self.model_id:
                # CustomVoice hat vordefinierte Sprecher; default = erster.
                speakers = self._model.get_supported_speakers()
                spk = speakers[0] if speakers else None
                wavs, sr = self._model.generate_custom_voice(
                    text=clean_text,
                    speaker=spk,
                    language=self.language,
                    instruct="Sprich ruhig und klar." if audio_tags else None,
                )
            elif "VoiceDesign" in self.model_id:
                wavs, sr = self._model.generate_voice_design(
                    text=clean_text,
                    instruct="Eine ruhige, klare Stimme mit neutralem Ton.",
                    language=self.language,
                )
            else:
                # Unbekanntes Modell — fallback Base-Pfad.
                kwargs["x_vector_only_mode"] = True
                wavs, sr = self._model.generate_voice_clone(
                    text=clean_text,
                    language=self.language,
                    ref_audio=(dummy_ref, sr_ref),
                    **kwargs,
                )
            audio = wavs[0]
            # WAV schreiben
            import soundfile as _sf
            _sf.write(str(out_file), audio, sr)
        except Exception as e:
            raise RuntimeError(
                f"Qwen3-TTS-Inferenz fehlgeschlagen: {e}"
            ) from e

        synth_time = time.perf_counter() - start
        actual_sr, audio_duration = _read_wav_metadata(str(out_file))

        return SynthResult(
            filepath=str(out_file),
            sample_rate=actual_sr,
            audio_duration_s=audio_duration,
            synth_time_s=synth_time,
            ttfa_ms=synth_time * 1000,
            rtf=synth_time / audio_duration if audio_duration > 0 else 0.0,
            engine_name=self.name,
            text_length_chars=len(text),
            word_count=len(text.split()),
            extra={"model_id": self.model_id, "device": self._device,
                   "end_to_end_available": self._end_to_end_available,
                   "language": self.language,
                   "tag_count": len(audio_tags)},
        )


# ─── 8. Factory + Fallback-Hierarchy ──────────────────────────────────


_FALLBACK_CHAIN = ["piper", "espeak", "off"]


def make_engine(name: str, sample_rate: int = 22050,
                verbose: bool = True, preflight: bool = True,
                llm_active_vram_mb: int = 0) -> TTSEngine:
    """Factory: gibt eine Engine-Instanz für ``name`` zurück.

    Pre-Flight: ``preflight=True`` (default) ruft ``_ensure_loaded()``
    sofort, damit Init-Fehler (z.B. piper-tts fehlt, espeak-ng binary
    nicht da) NICHT erst zur Synthese-Zeit auffallen. Bei Init-Fehler
    wird die Fallback-Hierarchie ``piper → espeak → off`` durchlaufen.

    ``preflight=False``: Engine wird ohne Check zurückgegeben (lazy).
    Nützlich für ``list_available_engines()``-Style-Listen ohne Probe.

    ``llm_active_vram_mb``: aktuell vom LLM belegter VRAM (MB). Wird an
    gpu-tier Engines (Bark, Qwen3) durchgereicht, damit _resolve_device
    entscheiden kann ob CUDA benutzt wird oder nicht. Default 0 = GPU leer.
    """
    name = (name or "off").lower()
    if name == "off":
        return OffEngine(sample_rate=sample_rate)

    candidate: Optional[TTSEngine] = None
    if name == "piper":
        candidate = PiperEngine(sample_rate=sample_rate)
    elif name == "espeak":
        candidate = EspeakEngine(sample_rate=sample_rate)
        if preflight and not candidate.available:
            raise RuntimeError("espeak-ng binary nicht gefunden")
    elif name == "bark":
        candidate = BarkEngine(sample_rate=sample_rate,
                               llm_active_vram_mb=llm_active_vram_mb)
    elif name == "qwen3":
        candidate = Qwen3Engine(sample_rate=sample_rate,
                                llm_active_vram_mb=llm_active_vram_mb)
    else:
        # Unknown engine name → off.
        if verbose:
            print(f"[tts] unbekannte engine '{name}'; fallback off")
        return OffEngine(sample_rate=sample_rate)

    # Pre-Flight: bei Init-Fehler Fallback-Kette durchlaufen.
    if preflight:
        for fallback_name in _FALLBACK_CHAIN:
            if fallback_name == name:
                continue
            try:
                candidate._ensure_loaded()
                return candidate
            except Exception as e:
                if verbose:
                    print(f"[tts] {name} init fehlgeschlagen: {e}; "
                          f"fallback {fallback_name}")
                # Nächste Fallback-Stufe versuchen.
                if fallback_name == "piper":
                    candidate = PiperEngine(sample_rate=sample_rate)
                elif fallback_name == "espeak":
                    candidate = EspeakEngine(sample_rate=sample_rate)
                elif fallback_name == "off":
                    return OffEngine(sample_rate=sample_rate)
        # Sollte nie erreicht werden (off ist immer verfügbar).
        return OffEngine(sample_rate=sample_rate)

    return candidate


def list_available_engines() -> List[str]:
    """Liefert die Liste der *aktuell verfügbaren* Engines (pre-flight).
    Installations-Status wird geprüft, ohne Modelle zu laden."""
    available = ["off"]
    # Piper
    try:
        import piper  # noqa: F401
        available.append("piper")
    except ImportError:
        pass
    # espeak-ng
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        available.append("espeak")
    # Bark (transformers reicht als Heuristik)
    try:
        import transformers  # noqa: F401
        available.append("bark")
    except ImportError:
        pass
    # Qwen3 (transformers + Safetensors; ab v5.1 NICHT mehr llama-cpp)
    try:
        import transformers  # noqa: F401
        # Heuristik: qwen3 verfügbar wenn transformers da ist UND
        # safetensors-Loader verfügbar. AutoModel.from_pretrained
        # macht den eigentlichen Modell-Check beim ersten Init.
        import safetensors  # noqa: F401
        available.append("qwen3")
    except ImportError:
        pass
    return available


# ─── 9. Utilities ────────────────────────────────────────────────────


def _read_wav_metadata(path: str) -> tuple:
    """Liest (sample_rate, duration_s) aus einer WAV-Datei. Returns
    (0, 0.0) bei Lese-Fehler."""
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / sr if sr > 0 else 0.0
        return sr, duration
    except Exception:
        return 0, 0.0


def _write_silent_wav(path: str, sample_rate: int, duration_s: float) -> None:
    """Schreibt eine stille WAV-Datei (für Placeholder / Fallbacks)."""
    n_frames = int(sample_rate * duration_s)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
