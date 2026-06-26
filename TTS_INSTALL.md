# TTS Engines — Installations-Anleitung

Plan 5.1 bringt fünf TTS-Engines. Drei davon brauchen externe Software
(Piper, espeak-ng, Bark, Qwen3-TTS). Diese Datei erklärt Installation,
Fallback-Verhalten, und Skip-Optionen für jede Engine.

## Übersicht

| Engine | Tier | Modell-Größe | Fallback bei Init-Fehler |
|---|---|---|---|
| `off`   | —    | 0 (kein Audio) | letzte Station der Kette |
| `espeak` | CPU (binary) | 0 | `off` |
| `piper` | CPU (ONNX) | ~60 MB | `espeak` → `off` |
| `bark`  | GPU (transformers) | ~2 GB | `piper` → `espeak` → `off` |
| `qwen3` | GPU (transformers) | ~650 MB | `piper` → `espeak` → `off` |

**Fallback-Hierarchie**: `make_engine(name)` versucht zuerst den
gewünschten Namen, dann `piper`, dann `espeak`, dann `off`.
Damit bekommt man auch ohne Installation mindestens eine
funktionierende Engine — entweder Echtzeit-Sprache (Piper/espeak)
oder gar nichts (`off`).

**Audio-Output**: Browser (Web-UI: `gr.Audio`) bzw. Datei-Pfad
(CLI: `--tts-output-file`). **Kein** System-Audio via
`aplay`/`afplay` (siehe Plan 5.1, D6).

**GPU-Auto-Swap**: ML-Engines (Bark, Qwen3) gehen automatisch auf
GPU, wenn nach dem LLM-Stream genug freies VRAM da ist
(`_resolve_device` prüft via `torch.cuda.mem_get_info()`,
200 MB Margin + 90 %-Schwelle). Wenn nicht → CPU bzw. Fallback
auf Piper/espeak.

---

## 1. espeak-ng (CPU, binary) — Minimal-Setup

Espeak-NG ist die Default-CPU-Engine. Binary-only, kein Python-Paket.

### Debian / Ubuntu

```bash
sudo apt-get update
sudo apt-get install espeak-ng
# Smoke-Test:
espeak-ng -v de "Hallo Welt"
```

### Fedora / RHEL

```bash
sudo dnf install espeak-ng
espeak-ng -v de "Hallo Welt"
```

### macOS

```bash
brew install espeak
espeak -v de "Hallo Welt"
```

### Windows

Lade espeak-ng von <https://github.com/espeak-ng/espeak-ng/releases>
herunter, entpacke, füge das `bin/`-Verzeichnis zum `PATH` hinzu.

### Was espeak kann

- Sprache: jede via `-v <lang>` (Standard: englisch)
- Tags, die espeak versteht: `[#A0]` → `-p 5`, `[#A2]` → `-p 25`,
  `[#A4]` → `-p 50`, `[#WHISPER]` → `-a 30`,
  `[#LOUD]` → `-a 175`. Pause-Tags via Stille-Insert.
- **Sub-kontra (A0..B1)**: espeak pitch-shift ist limitiert. Bei
  extrem tiefen Noten (>3 Oktaven unter A4) wird im Log eine
  Warnung ausgegeben.

### Verifikation

```bash
python -c "
from gradio_tabs.tts_engine import make_engine
eng = make_engine('espeak', preflight=True)
print(eng.name, 'verfügbar:', eng.available)
"
```

---

## 2. Piper (CPU, ONNX) — Mittlere Qualität, Echtzeit

Piper ist die Default-CPU-Engine, wenn nichts anderes gewählt wird.
ONNX-Runtime, ~60 MB pro Voice, deutsche Stimmen verfügbar.

### Voraussetzungen

```bash
pip install piper-tts>=1.2.0
```

`>= 1.2.0` ist Pflicht für `SynthesisConfig(pitch=…, volume_gain=…)`.
Ältere Versionen können die Audio-Tags (Noten, Dynamik) nicht
audio-seitig anwenden — sie fallen auf "Strip + Ignore" zurück.

### Voice-Modell herunterladen

```bash
# Methode A: piper --download-voice
piper --download-voice de_DE-thorsten-medium --download-dir ~/.local/share/piper

# Methode B: HuggingFace direkt
huggingface-cli download rhasspy/piper-voices \
    --local-dir ~/.local/share/piper \
    --include "de/de_DE/thorsten/medium/*"
```

### Empfohlene deutsche Stimmen

- `de_DE-thorsten-medium` (männlich, ~60 MB)
- `de_DE-eva_k-medium` (weiblich)
- `de_DE-kerstin-low` (weiblich, schnell, ~30 MB)

### Konfiguration

`PiperEngine(voice_model=…)` mit absolutem Pfad zur `.onnx`-Datei.
Default-Suche: `~/.local/share/piper/<voice>/<voice>.onnx`.

### Verifikation

```bash
python -c "
from gradio_tabs.tts_engine import make_engine
eng = make_engine('piper', preflight=True)
print(eng.name, 'verfügbar:', eng.available)
print('Voice:', getattr(eng, 'voice_model', '(default)'))
"
```

### Was Piper kann

- Tags werden via `SynthesisConfig(pitch, volume_gain)` audio-seitig
  angewendet (Plan 5.1, D2).
- Pause via numpy-Stille-Insert.
- Sub-phrase splitting: pro Audio-Tag wird eine separate Piper-Synthese
  gemacht → `[#A2]Hallo [#A4]Welt` = zwei Piper-Calls mit
  unterschiedlichen Pitch-Configs.

---

## 3. Bark (GPU, transformers) — Höchste Qualität

Bark ist die hochwertigste Engine, aber auch die schwerste (~2 GB
Modell-Download, GPU erforderlich für Echtzeit).

### Voraussetzungen

```bash
pip install transformers torch accelerate
# torch mit CUDA:
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

### Modell-Download

Beim ersten `synthesize()`-Call lädt `transformers.AutoModel.from_pretrained("suno/bark-small")` automatisch herunter (~2 GB nach `~/.cache/huggingface/`).

### Skip-Option für Tests / Sandbox

```bash
export SKIP_BARK_DOWNLOAD=1
# Jetzt wirft BarkEngine._ensure_loaded() RuntimeError("Bark-Download per SKIP_BARK_DOWNLOAD=1 übersprungen")
# Fallback-Kette: bark → piper → espeak → off
```

### GPU-Modus

Auto-Swap passiert via `_resolve_device("bark", llm_active_vram_mb)`:

- Wenn `torch.cuda.is_available()` UND `free_vram > 2200 MB` UND
  `LLM-VRAM + 2000 MB < 90 % total` → `cuda`.
- Sonst: `cpu` (langsam, ~10–30× Echtzeit).

### Was Bark kann

- Bark-native Tags wie `[laughter]`, `[sighs]`, `♪` werden via
  `strip_tags_for_engine("bark", text)` direkt durchgereicht.
- Noten/Whisper/LOUD/Pause werden **ignoriert** (Bark kennt das
  nicht; das LLM soll die Tags weglassen, oder der User hört
  einfach eine moderate Stimme).

### Verifikation

```bash
SKIP_BARK_DOWNLOAD=1 python -c "
from gradio_tabs.tts_engine import make_engine
eng = make_engine('bark', preflight=True)
print('Fällt zurück auf:', eng.name)
"
```

---

## 4. Qwen3-TTS (GPU, qwen-tts) — Mittel-klein

Qwen3-TTS via das offizielle `qwen-tts`-PyPI-Package (NICHT
`transformers.AutoModel` — die `qwen3_tts`-Architektur ist noch nicht
in mainline-transformers gemerged, das Package stellt das eigene
`Qwen3TTSModel.from_pretrained(...)` bereit). Tier `gpu` via
`_resolve_device`, auto-swap auf cpu wenn VRAM nicht reicht.

### Voraussetzungen

```bash
pip install qwen-tts torch safetensors
# torch mit CUDA:
pip install torch --extra-index-url https://download.pytorch.org/whl/cu130
```

### Modell-Download

Default HF-Cache verwenden (`~/.cache/huggingface/`), **KEIN**
`--local-dir` setzen und **KEIN** `HF_HOME` überschreiben — sonst
landet das Modell auf der falschen Partition (z.B. wenn die ML4-Partition
voll ist, bricht xet mit `No space left on device` ab).

```bash
# Variante A: via qwen-tts CLI (empfohlen für Test/Verify)
python -c "
from qwen_tts import Qwen3TTSModel
m = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base')
print('geladen:', m)
"

# Variante B: via huggingface-cli
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base
# landet automatisch in ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/
```

### Skip-Option

```bash
export SKIP_QWEN3_DOWNLOAD=1
# wirft RuntimeError; Fallback auf piper → espeak → off
```

### Stand 2026-06 (verifiziert)

Drei Modell-Varianten sind auf HuggingFace verfügbar und werden von
der Engine automatisch erkannt:

- **0.6B-Base** (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`, ~2.4 GB) — Default
  für `--tts-engine qwen3`. Voice-Clone-Pfad mit
  `generate_voice_clone(x_vector_only_mode=True)` (kein
  Referenz-Audio nötig, Modell synthetisiert Stil-Vektor selbst).
- **1.7B-Base** (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`) — gleiche API, mehr
  Qualität, mehr VRAM.
- **0.6B/1.7B-CustomVoice** — vordefinierte Sprecher via
  `generate_custom_voice(speaker=...)`. Liste der Speaker via
  `model.get_supported_speakers()`.
- **1.7B-VoiceDesign** — freie Stimm-Beschreibung via
  `generate_voice_design(instruct="...")`.

Die Engine-Default-ID ist `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (Base +
x-vector-only); kann via `Qwen3Engine(model_id=...)` überschrieben
werden.

### Was Qwen3 kann

- **Noten/Note-Tags**: Qwen3 hat keine direkte pitch-API. Tags werden
  in natürliche deutsche Sprache übersetzt (Plan 5.1, D5).
- **Dynamik/Dynamic-Tags**: gleichermaßen in Instruct-Text
  übersetzt.
- **Pause-Tags**: native Audio-Stille-Insert.
- **Affect-Tags** (`[#HAPPY]` etc.): werden gestrippt — Qwen3 hat
  keine Affekt-API, der User-Hinweis an das Modell ist Aufgabe des
  System-Prompts.

### Performance (RTX 2060 12 GB, 0.6B-Base, cuda, ohne flash-attn)

RTF ≈ 1.86 (5.6 s für 4.64 s Audio). Echtzeit mit `flash-attn`-
Installation; ohne lebt das Modell auf PyTorch-MHA (langsamer).

```bash
pip install flash-attn --no-build-isolation
```

### Warnung

```
flash-attn is not installed. Will only run the manual PyTorch version.
```

Kann ignoriert werden — ist nur eine Performance-Warnung.

---

## 5. Skip-Optionen (für CI / Sandbox)

```bash
# Verhindert Modell-Download (Tests fallen auf Piper/espeak/off zurück):
export SKIP_BARK_DOWNLOAD=1
export SKIP_QWEN3_DOWNLOAD=1
```

Diese env-vars werden in `tests/test_tts_engine.py` benutzt, um
die Tests ohne 2 GB Download durchzuhalten.

---

## 6. Diagnose-Befehle

```bash
# Welche Engines sind verfügbar?
python -c "from gradio_tabs.tts_engine import list_available_engines; print(list_available_engines())"

# Welches Tier pro Engine?
python -c "
from gradio_tabs.tts_engine import PiperEngine, EspeakEngine, BarkEngine, Qwen3Engine
for e in [PiperEngine, EspeakEngine, BarkEngine, Qwen3Engine]:
    print(f'{e.__name__}: tier={e.tier}, device={getattr(e(), \"_device\", \"?\")}')"

# Welcher Device-Resolver entscheidet für "bark" bei 0 MB LLM-VRAM?
python -c "
from gradio_tabs.tts_engine import _resolve_device
print('bark (LLM idle):', _resolve_device('bark', llm_active_vram_mb=0))
print('bark (LLM busy 8GB):', _resolve_device('bark', llm_active_vram_mb=8000))
print('qwen3 (LLM idle):', _resolve_device('qwen3', llm_active_vram_mb=0))
"

# Einen kurzen Satz synthetisieren (CLI):
python streaming_bridge.py \
    --message "Hallo Welt, ich bin da." \
    --tts-engine piper \
    --tts-output-file /tmp/hallo.wav
aplay /tmp/hallo.wav  # oder einfach Browser-Audio via Web-UI
```

---

## 7. Bekannte Limitierungen

- **Sub-kontra (A0..B1)** klingt in espeak/piper stark verzerrt
  (Vocoder für Männerstimme ~120 Hz, Faktor 4 runter = am Limit
  der Trainingsdaten). Für realistische tiefe Töne → Bark.
- **Qwen3 end-to-end** ist Stand 2026-06 nicht public.
- **ABC-Notation** ist Backlog (Plan 5.1, Abschnitt "Was NICHT in
  diesem Plan ist"). User möchte das später als stabiles Format.
- **Streaming-TTS** während LLM-Stream ist Backlog. Aktuell
  one-shot nach Stream-Ende.
- **Voice-Cloning** (User-Audio hochladen) ist Backlog.

---

## 8. Plan-5.1-Verweise

- D1 (Tier-Architektur): `gradio_tabs/tts_engine.py: _resolve_device()`
- D2 (Piper pitch/speed/amplitude): `_piper_synth_to_wav()` mit
  `SynthesisConfig`
- D3 (espeak pitch/amplitude): `EspeakEngine.synthesize()` mit
  `-p`/`-a`-Flags
- D4 (Sub-kontra-Warnung): `EspeakEngine._emit_warning()` bei
  pitch_offset < -36
- D5 (Qwen3 via Safetensors): `Qwen3Engine._ensure_loaded()`
- D6 (Web-Audio): `chat_tab.py` — `gr.Audio(type="filepath", autoplay=True)`
- D7 (Tests ohne echten Bark): `tests/test_tts_engine.py` mit
  `SKIP_BARK_DOWNLOAD=1`
- D8 (Install-Commands): dieses Dokument hier