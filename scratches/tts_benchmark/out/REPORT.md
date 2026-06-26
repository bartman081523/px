# TTS Latenz-Benchmark Report

Gestartet: 2026-06-26T12:51:13.696281+00:00
Beendet: 2026-06-26T12:51:13.706880+00:00
Python: `3.14.5`

## Verfügbare Engines

- off

## Init-Zeit pro Engine (einmaliger Warmup)

| Engine | Init-Zeit (s) |
|---|---|
| piper | n/a (nicht verfügbar) |
| bark | n/a (nicht verfügbar) |
| qwen3 | n/a (nicht verfügbar) |
| espeak | n/a (nicht verfügbar) |

## Latenz pro Antwortlänge

RTF < 1.0 = realtime-schneller; < 0.5 = gefühlt realtime.

### Engine: `piper`

⚠ **Engine nicht verfügbar**: `init_error: factory fell back from piper to off`

### Engine: `bark`

⚠ **Engine nicht verfügbar**: `init_error: factory fell back from bark to off`

### Engine: `qwen3`

⚠ **Engine nicht verfügbar**: `init_error: factory fell back from qwen3 to off`

### Engine: `espeak`

⚠ **Engine nicht verfügbar**: `init_error: espeak-ng binary nicht gefunden`

## Empfehlung

- **Nicht verfügbar / Init-Fehler**: piper, bark, qwen3, espeak — vor dem Lauf installieren oder Skip-Grund prüfen.

→ Diese Empfehlung wird im nächsten Schritt (Integration) als Default-Auswahl in chat_tab.py / streaming_bridge.py verwendet.