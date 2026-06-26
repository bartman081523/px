"""Tests für `gradio_tabs/tts_engine.py` — Engine-Protokoll + Factory.

Diese Tests laufen in **jeder** Umgebung — sie hängen nicht davon ab,
ob piper/bark/llama-cpp/espeak-ng installiert sind. Wenn eine Lib fehlt,
muss die Factory sauber auf den Fallback zurückfallen (kein Crash).
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gradio_tabs.tts_engine import (  # noqa: E402
    SynthResult, TTSEngine, OffEngine, EspeakEngine,
    PiperEngine, BarkEngine, Qwen3Engine,
    make_engine, list_available_engines,
    _read_wav_metadata, _write_silent_wav, _FALLBACK_CHAIN,
    _resolve_device,
)


# ─── 1. SynthResult + OffEngine (keine externen Deps) ──────────────


def test_synth_result_dataclass_fields():
    r = SynthResult(
        filepath="/tmp/x.wav", sample_rate=22050, audio_duration_s=1.5,
        synth_time_s=0.3, ttfa_ms=300.0, rtf=0.2, engine_name="off",
        text_length_chars=10, word_count=2,
    )
    assert r.filepath == "/tmp/x.wav"
    assert r.sample_rate == 22050
    assert r.rtf == 0.2


def test_off_engine_synthesize_returns_empty():
    eng = OffEngine()
    res = eng.synthesize("Hallo Welt")
    assert res.filepath == ""
    assert res.engine_name == "off"
    assert res.audio_duration_s == 0.0
    assert res.word_count == 2


def test_off_engine_is_loaded_true():
    """OffEngine braucht keinen Load → is_loaded ist True nach Init."""
    eng = OffEngine()
    assert eng.is_loaded is True


# ─── 2. Factory + Fallback-Hierarchie ─────────────────────────────


def test_make_engine_off_returns_off_engine():
    eng = make_engine("off", verbose=False)
    assert isinstance(eng, OffEngine)
    assert eng.name == "off"


def test_make_engine_unknown_falls_back_to_off():
    """Unbekannter Engine-Name → off (kein Crash)."""
    eng = make_engine("nonexistent_engine", verbose=False)
    assert isinstance(eng, OffEngine)


def test_make_engine_piper_falls_back_when_piper_missing():
    """Wenn piper-tts nicht installiert ist, fällt die Factory auf
    espeak-ng oder off zurück (nie Crash)."""
    with patch("gradio_tabs.tts_engine.PiperEngine._ensure_loaded",
               side_effect=RuntimeError("mock piper unavailable")):
        eng = make_engine("piper", verbose=False)
        # Akzeptiere espeak oder off (je nachdem was verfügbar).
        assert isinstance(eng, (EspeakEngine, OffEngine))


def test_make_engine_bark_falls_back_when_bark_missing():
    """Wenn transformers/llama_cpp für bark nicht da ist → fallback."""
    # Erzwungener Init-Fehler durch Mocking der _ensure_loaded.
    with patch("gradio_tabs.tts_engine.BarkEngine._ensure_loaded",
               side_effect=RuntimeError("mock bark unavailable")):
        eng = make_engine("bark", verbose=False)
        assert isinstance(eng, (EspeakEngine, OffEngine))


def test_make_engine_qwen3_falls_back_when_llama_cpp_missing():
    with patch("gradio_tabs.tts_engine.Qwen3Engine._ensure_loaded",
               side_effect=RuntimeError("mock llama_cpp unavailable")):
        eng = make_engine("qwen3", verbose=False)
        assert isinstance(eng, (EspeakEngine, OffEngine))


def test_make_engine_preflight_false_returns_unloaded_piper():
    """Mit preflight=False bekommt man die Engine ohne Init-Check
    (lazy; nützlich für UI-Listen ohne Probe)."""
    with patch("gradio_tabs.tts_engine.PiperEngine._ensure_loaded",
               side_effect=RuntimeError("would fail if called")):
        eng = make_engine("piper", verbose=False, preflight=False)
        assert isinstance(eng, PiperEngine)
        assert eng.is_loaded is False


def test_make_engine_espeak_when_binary_missing_raises_preflight():
    """Wenn espeak-ng/espeak binary fehlt UND preflight=True, wird
    klar geworfen (User kann das fangen und auf off umschalten).
    preflight=False würde ohne Check OffEngine-Instanz-Logik durchlaufen."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="espeak-ng binary"):
            make_engine("espeak", verbose=False, preflight=True)


def test_make_engine_espeak_preflight_false_creates_engine():
    """Mit preflight=False bekommt man auch ohne binary eine
    EspeakEngine-Instanz (Lazy; würde beim synthesize fehlschlagen)."""
    with patch("shutil.which", return_value=None):
        eng = make_engine("espeak", verbose=False, preflight=False)
        assert isinstance(eng, EspeakEngine)
        assert eng.available is False


# ─── 3. Fallback-Kette ────────────────────────────────────────────


def test_fallback_chain_is_stable():
    """Die Fallback-Hierarchie (piper → espeak → off) ist dokumentiert
    und sollte sich nicht ungewollt ändern."""
    assert _FALLBACK_CHAIN == ["piper", "espeak", "off"]


def test_fallback_chain_documented_in_module():
    """Sicherheits-Check: piper hat Vorrang vor espeak (Default)."""
    assert _FALLBACK_CHAIN.index("piper") < _FALLBACK_CHAIN.index("espeak")
    assert _FALLBACK_CHAIN.index("espeak") < _FALLBACK_CHAIN.index("off")


# ─── 4. list_available_engines (pre-flight) ───────────────────────


def test_list_available_engines_returns_list():
    res = list_available_engines()
    assert isinstance(res, list)
    assert "off" in res  # off ist IMMER verfügbar


def test_list_available_engines_off_always_first():
    res = list_available_engines()
    assert res[0] == "off"


# ─── 5. EspeakEngine Init (kein subprocess-Aufruf nötig) ──────────


def test_espeak_engine_available_flag():
    """Verfügbarkeit wird korrekt aus PATH abgeleitet."""
    eng = EspeakEngine()
    # In Sandbox: kein espeak-ng → available=False.
    # In User-Venv: kann True sein. Test prüft nur die Konsistenz.
    if eng.available:
        assert eng.espeak_bin is not None
    else:
        assert eng.espeak_bin is None


def test_espeak_engine_ensure_loaded_raises_when_missing():
    """Wenn espeak-ng nicht da, wirft _ensure_loaded RuntimeError."""
    with patch("shutil.which", return_value=None):
        eng = EspeakEngine()
        with pytest.raises(RuntimeError, match="espeak-ng nicht gefunden"):
            eng._ensure_loaded()


def test_espeak_engine_synthesize_empty_text():
    """Leerer Text → leere SynthResult ohne subprocess-Call."""
    eng = EspeakEngine()
    if not eng.available:
        pytest.skip("espeak-ng nicht installiert (Sandbox)")
    res = eng.synthesize("")
    assert res.filepath == ""
    assert res.audio_duration_s == 0.0


# ─── 6. WAV-Utilities ────────────────────────────────────────────


def test_write_silent_wav_creates_valid_file(tmp_path):
    """Eine stille WAV wird geschrieben, ist abspielbar."""
    out = tmp_path / "silent.wav"
    _write_silent_wav(str(out), 22050, 0.5)
    assert out.exists()
    sr, duration = _read_wav_metadata(str(out))
    assert sr == 22050
    assert abs(duration - 0.5) < 0.01


def test_read_wav_metadata_missing_file():
    """Fehlende Datei → (0, 0.0), kein Crash."""
    sr, duration = _read_wav_metadata("/nonexistent/path/x.wav")
    assert sr == 0
    assert duration == 0.0


def test_read_wav_metadata_corrupt_file(tmp_path):
    """Kaputte WAV-Datei → (0, 0.0), kein Crash."""
    out = tmp_path / "corrupt.wav"
    out.write_bytes(b"NOT A WAV FILE")
    sr, duration = _read_wav_metadata(str(out))
    assert sr == 0
    assert duration == 0.0


# ─── 7. Engine-Struktur (Lazy-Load-Vertrag) ───────────────────────


def test_piper_engine_starts_unloaded():
    """PiperEngine lädt NICHT beim Init — erst bei synthesize()."""
    eng = PiperEngine()
    assert eng.is_loaded is False
    assert eng._voice is None


def test_bark_engine_starts_unloaded():
    eng = BarkEngine()
    assert eng.is_loaded is False
    assert eng._model is None


def test_qwen3_engine_starts_unloaded():
    eng = Qwen3Engine()
    assert eng.is_loaded is False
    assert eng._model is None


def test_skip_bark_download_env_var():
    """SKIP_BARK_DOWNLOAD=1 verhindert das Bark-Modell-Loading."""
    import os
    os.environ["SKIP_BARK_DOWNLOAD"] = "1"
    try:
        eng = BarkEngine()
        with pytest.raises(RuntimeError, match="SKIP_BARK_DOWNLOAD"):
            eng._ensure_loaded()
    finally:
        del os.environ["SKIP_BARK_DOWNLOAD"]


# ─── 8. RTF-Berechnung ───────────────────────────────────────────


def test_rtf_calculation_in_synth_result():
    """RTF = synth_time / audio_duration. <1.0 = realtime-schneller."""
    # Konstanter Test ohne Engine: nutze OffEngine mit künstlichen Daten.
    # (OffEngine liefert audio_duration=0 → rtf=0.0 als Default.)
    eng = OffEngine()
    res = eng.synthesize("Hallo")
    assert res.rtf == 0.0


def test_synth_result_extra_dict_default_empty():
    """SynthResult.extra ist ein Dict (für Engine-spezifische Metadaten)."""
    r = SynthResult(
        filepath="", sample_rate=0, audio_duration_s=0.0,
        synth_time_s=0.0, ttfa_ms=0.0, rtf=0.0, engine_name="x",
        text_length_chars=0, word_count=0,
    )
    assert r.extra == {}
    assert isinstance(r.extra, dict)


# ─── 9. Tier-Konzept (Plan 5.1: CPU/GPU-Aufteilung) ───────────────


def test_piper_engine_tier_is_cpu():
    """Piper ist CPU-tier (ONNX-Runtime ist CPU-realtime-tauglich)."""
    assert PiperEngine.tier == "cpu"


def test_espeak_engine_tier_is_cpu():
    """espeak-ng ist CPU-tier (binary, kein Modell)."""
    assert EspeakEngine.tier == "cpu"


def test_bark_engine_tier_is_gpu():
    """Bark ist GPU-tier (transformers, ~2GB Modell)."""
    assert BarkEngine.tier == "gpu"


def test_qwen3_engine_tier_is_gpu():
    """Qwen3 ist GPU-tier (transformers, ~650MB)."""
    assert Qwen3Engine.tier == "gpu"


def test_off_engine_tier_is_cpu():
    """OffEngine ist trivial CPU-tier."""
    assert OffEngine.tier == "cpu"


def test_bark_engine_resolves_device_on_init():
    """BarkEngine.__init__ ruft _resolve_device auf, _device ist gesetzt."""
    eng = BarkEngine()
    # Default: ohne CUDA → "cpu".
    assert eng._device in ("cpu", "cuda")
    assert eng.device == eng._device


def test_qwen3_engine_resolves_device_on_init():
    eng = Qwen3Engine()
    assert eng._device in ("cpu", "cuda")


def test_cpu_tier_engines_have_cpu_device():
    """Piper/espeak sind immer CPU, _resolve_device gibt 'cpu' zurück."""
    assert _resolve_device("piper") == "cpu"
    assert _resolve_device("espeak") == "cpu"
    assert _resolve_device("unknown_engine") == "cpu"


# ─── 10. Device-Resolver mit/ohne CUDA ──────────────────────────────


def test_resolve_device_bark_no_cuda_returns_cpu(monkeypatch):
    """Wenn torch.cuda.is_available() False → 'cpu', auch wenn free_vram
    groß wäre."""
    import sys
    # Simuliere "torch nicht verfügbar" → "cpu".
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _resolve_device("bark", llm_active_vram_mb=0) == "cpu"


def test_resolve_device_no_torch_returns_cpu():
    """Wenn torch gar nicht importierbar → 'cpu'."""
    # Wir löschen torch temporär aus sys.modules, falls da.
    import sys
    saved = sys.modules.pop("torch", None)
    # Block torch-Import temporär.
    blocked = {"torch": None}
    import importlib
    orig_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None
    # Pragmatisch: monkeypatched torch via sys.modules.
    sys.modules["torch"] = None  # None blockiert Import.
    try:
        result = _resolve_device("bark", llm_active_vram_mb=0)
        assert result == "cpu"
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)


def test_resolve_device_with_cuda_low_vram_returns_cpu(monkeypatch):
    """Wenn free_vram < engine_model_mb + 200MB Margin → 'cpu' (passt nicht)."""
    import sys
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (500 * 1024 * 1024,
                                                   12000 * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # Bark braucht 2000MB; free=500MB → cpu.
    assert _resolve_device("bark", llm_active_vram_mb=0) == "cpu"


def test_resolve_device_with_cuda_enough_vram_returns_cuda(monkeypatch):
    """Wenn free_vram ≥ engine_model_mb + 200MB UND LLM nicht zu viel → 'cuda'."""
    import sys
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (8000 * 1024 * 1024,
                                                   12000 * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # Bark braucht 2000MB + 200MB Margin = 2200MB; free=8000MB, LLM=0MB → cuda.
    assert _resolve_device("bark", llm_active_vram_mb=0) == "cuda"


def test_resolve_device_llm_too_big_returns_cpu(monkeypatch):
    """Wenn LLM + Engine > 90% total → 'cpu'."""
    import sys
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (8000 * 1024 * 1024,
                                                   12000 * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # LLM=10000MB + Bark=2000MB = 12000MB = 100% > 90% → cpu.
    assert _resolve_device("bark", llm_active_vram_mb=10000) == "cpu"


def test_resolve_device_qwen3_smaller_model(monkeypatch):
    """Qwen3 braucht nur 650MB → bekommt cuda auch bei knapperem VRAM
    als Bark (2GB)."""
    import sys
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (1500 * 1024 * 1024,
                                                   12000 * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # Bark (2000MB) passt nicht in 1500MB free → cpu.
    assert _resolve_device("bark", llm_active_vram_mb=0) == "cpu"
    # Qwen3 (650MB) passt mit Margin → cuda.
    assert _resolve_device("qwen3", llm_active_vram_mb=0) == "cuda"


# ─── 11. SKIP_QWEN3_DOWNLOAD env-var ──────────────────────────────


def test_skip_qwen3_download_env_var():
    """SKIP_QWEN3_DOWNLOAD=1 verhindert das Qwen3-Tokenizer-Loading."""
    import os
    os.environ["SKIP_QWEN3_DOWNLOAD"] = "1"
    try:
        eng = Qwen3Engine()
        with pytest.raises(RuntimeError, match="SKIP_QWEN3_DOWNLOAD"):
            eng._ensure_loaded()
    finally:
        del os.environ["SKIP_QWEN3_DOWNLOAD"]


# ─── 12. make_engine mit llm_active_vram_mb ──────────────────────


def test_make_engine_bark_passes_vram_param():
    """make_engine('bark', llm_active_vram_mb=5000) reicht vram-Wert
    an BarkEngine weiter."""
    eng = make_engine("bark", verbose=False, preflight=False,
                      llm_active_vram_mb=5000)
    assert isinstance(eng, BarkEngine)
    assert eng.llm_active_vram_mb == 5000


def test_make_engine_qwen3_passes_vram_param():
    eng = make_engine("qwen3", verbose=False, preflight=False,
                      llm_active_vram_mb=3000)
    assert isinstance(eng, Qwen3Engine)
    assert eng.llm_active_vram_mb == 3000
