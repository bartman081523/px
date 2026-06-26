# TTS Latenz-Benchmark

Misst für jede verfügbare TTS-Engine die echte CPU-Latenz.

## Was gemessen wird

Pro Engine (`piper`, `bark`, `qwen3`, `espeak`):
- **Init-Zeit** (Sekunden) — wie lange der erste `synthesize()`-Call braucht, um Modell/Binary zu laden.
- **TTFA** (Time-to-first-audio, ms) — vom `synthesize()`-Call bis WAV-Datei geschrieben.
- **RTF** (Real-Time-Factor) — `synth_time / audio_duration`. **<1.0 = realtime-schneller**, >1.0 = langsamer als realtime.
- **Latenz für 5 Antwortlängen** (10 / 30 / 80 / 150 / 300 Wörter) — entspricht typischen LLM-Antworten.

## Output

`out/benchmark_results.json` (rohe Daten) + `out/REPORT.md` (lesbarer Report mit Empfehlung).

## Methode

Pro Antwortlänge:
- 3 Beispiel-Antworten (verschiedene Sätze, ähnlicher Wort-Anzahl)
- Mittelwert + Std über die 3 Runs
- Engines, die Init fehlschlagen → `available: false`, kein Crash

## Ausführen

```bash
python scratches/tts_benchmark/run.py
```

Output landet in `scratches/tts_benchmark/out/`. Per Scratch-Regel: committed.

## Realtime-Schwelle

RTF < 2.0 gilt als "akzeptabel für Chat-UX" (User wartet höchstens 2× so
lange wie der gesprochene Text dauert). RTF < 0.5 = "gefühlt realtime".
RTF > 5.0 = "deutlich merkbar verzögert" — als Default für den User
nicht empfohlen, aber als explizite Option weiterhin verfügbar.