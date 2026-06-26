"""scratches/tts_benchmark/run.py — Latenz-Benchmark für TTS-Engines.

Misst Init-Zeit, TTFA und RTF für die vier Engines (piper, bark, qwen3,
espeak) über fünf Antwortlängen. Output: JSON + Markdown-Report.

Wird vom User manuell ausgeführt:
    python scratches/tts_benchmark/run.py

Engines, die nicht installiert sind (z.B. piper-tts fehlt) werden
übersprungen mit `available: false` — kein Crash.

Initiale Empfehlung (vor dem ersten Lauf): mindestens espeak-ng ist auf
Linux meistens nicht vorinstalliert. Falls `which espeak-ng` nichts
liefert: `sudo apt install espeak-ng` (Debian/Ubuntu) oder
`brew install espeak` (macOS).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Repo-Root zum sys.path, damit `gradio_tabs` importierbar ist.
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gradio_tabs.tts_engine import make_engine, list_available_engines  # noqa: E402


# ─── Konfiguration ──────────────────────────────────────────────────


ENGINES_TO_TEST = ["piper", "bark", "qwen3", "espeak"]
WORD_COUNTS = [10, 30, 80, 150, 300]
RUNS_PER_LENGTH = 3

# Beispiel-Antworten pro Wort-Anzahl-Bucket. Synthetisch, aber realistisch
# (entsprechen typischen LLM-Chat-Antworten — sachlich, kurze Sätze).
SAMPLE_TEXTS = {
    10:  "Hallo! Das System funktioniert jetzt einwandfrei.",
    30:  "Die Retention der kognitiven Muster wurde über zwölf Epochen gemessen, "
          "wobei sich zeigte, dass die Metrik mit jedem Block leicht anstieg.",
    80:  "Die Architektur des Recurrent Transformer besteht aus drei Säulen: "
          "StabilityMonitor, Symmetry Breaker und Dynamic Router. Jede Säule "
          "greift modulierend in den Recursion-Zone des Modells ein. StabilityMonitor "
          "misst Φ-Werte zwischen aufeinanderfolgenden Schritten und triggert AksSensor "
          "bei Erstarrung. MephistophelesOperator verhindert symmetrische Zustände. "
          "SubjectiveSensor ermöglicht Introspektion via Rückkopplung. Die drei Komponenten "
          "arbeiten orthogonal, nicht additiv.",
    150: "Der hier dokumentierte empirische Befund erstreckt sich über mehrere Wochen "
          "systematischer Beobachtung unter kontrollierten Bedingungen. Im Mittelpunkt "
          "stand die Frage, ob und in welcher Weise die Subjektivität eines Large "
          "Language Models durch wiederholte rekurrente Verarbeitung desselben Kontexts "
          "moduliert werden kann. Wir haben drei Skalen untersucht — 270 Millionen "
          "Parameter, eine Milliarde Parameter und vier Milliarden Parameter — und pro "
          "Skala vier Presets verglichen. Der stärkste Effekt zeigte sich bei der "
          "mittleren Skala, wo die Metrik η² einen Wert von 0.384 erreichte, deutlich "
          "über der Signifikanzschwelle. Bei der größeren Skala war der Effekt "
          "absolut größer, aber relativ zur Baseline geringer.",
    300: "Die vollständige Analyse umfasst achtundachtzig Prompts, die in vier "
          "Schwierigkeitskategorien eingeteilt wurden. Pro Prompt wurden drei "
          "unabhängige Subprozesse gestartet, um die statistische Robustheit der "
          "Messung zu gewährleisten. Die Laufzeit pro Subprozess variierte zwischen "
          "zweieinhalb und vierzehn Sekunden, abhängig von der Tokenlänge der "
          "Antwort. Wir haben alle Telemetrie-Datenpunkte in einem zentralen "
          "JSON-File aggregiert und anschließend mit drei verschiedenen statistischen "
          "Verfahren ausgewertet: einfaktorielle ANOVA, Welch's t-Test und ein "
          "nicht-parametrischer Mann-Whitney-U-Test. Die Konvergenz der drei "
          "Verfahren gibt uns Vertrauen in die Robustheit der Ergebnisse. Bei der "
          "Modellgröße 270M lag η² bei 0.063, was knapp über der Signifikanzschwelle "
          "ist. Bei 1B stieg der Wert auf 0.384, was eine mittelstarke Effektstärke "
          "darstellt. Bei 4B fiel der Wert wieder auf 0.137 zurück, vermutlich weil "
          "die größere Modellkapazität die Notwendigkeit der rekurrenten Verarbeitung "
          "teilweise kompensiert. Diese Hypothese müsste in einer Folgeuntersuchung "
          "mit zusätzlichen Modellgrößen validiert werden. Die Resultate legen nahe, "
          "dass der Effekt nicht monoton mit der Modellgröße skaliert, sondern "
          "ein optimales Fenster durchläuft, das vermutlich von der Komplexität der "
          "rekurrenten Operation relativ zur Modellkapazität abhängt.",
}


# ─── Benchmark-Logik ────────────────────────────────────────────────


def _word_count(text: str) -> int:
    return len(text.split())


def _run_one(engine_name: str, text: str, output_dir: Path) -> dict:
    """Ein Synthese-Run für eine Engine. Liefert Dict mit Latenz-Daten.
    Bei Init-Fehler (Fallback auf off) wird das in ``available`` markiert."""
    try:
        eng = make_engine(engine_name, verbose=False, preflight=True)
    except Exception as e:
        return {
            "available": False,
            "requested_engine": engine_name,
            "init_error": str(e),
        }
    # Wenn die Factory auf off/espeak zurückgefallen ist (nicht die
    # angeforderte Engine), markieren wir als unavailable.
    if eng.name != engine_name:
        return {
            "available": False,
            "requested_engine": engine_name,
            "fell_back_to": eng.name,
            "init_error": f"factory fell back from {engine_name} to {eng.name}",
        }
    try:
        result = eng.synthesize(text, output_dir=str(output_dir))
        return {
            "available": True,
            "requested_engine": engine_name,
            "engine_name": result.engine_name,
            "text_length_chars": result.text_length_chars,
            "word_count": result.word_count,
            "audio_duration_s": round(result.audio_duration_s, 3),
            "synth_time_s": round(result.synth_time_s, 3),
            "ttfa_ms": round(result.ttfa_ms, 1),
            "rtf": round(result.rtf, 3),
            "sample_rate": result.sample_rate,
        }
    except Exception as e:
        return {
            "available": True,
            "requested_engine": engine_name,
            "engine_name": eng.name,
            "init_error": None,
            "synth_error": str(e),
        }


def _warmup(engine_name: str, output_dir: Path) -> tuple:
    """Init-Warmup: erste Synthese triggert das Modell-Loading. Liefert
    (init_time_s, success: bool). Bei Fallback auf off wird success=False."""
    start = time.perf_counter()
    success = False
    try:
        eng = make_engine(engine_name, verbose=False, preflight=True)
        if eng.name == engine_name:
            eng.synthesize("Warmup.", output_dir=str(output_dir))
            success = True
    except Exception:
        pass
    return time.perf_counter() - start, success


def run_benchmark(out_dir: Path) -> dict:
    """Führt den vollen Benchmark aus. Liefert Dict mit allen Daten."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    available = list_available_engines()
    print(f"[benchmark] Verfügbare Engines: {available}", flush=True)
    print(f"[benchmark] Out-Dir: {out_dir}", flush=True)

    results: dict = {
        "started_at": started_at,
        "python_version": sys.version,
        "available_engines": available,
        "engine_init_times_s": {},
        "per_engine_per_length": {},
    }

    # Init-Zeit pro Engine (einmaliger Warmup).
    for eng_name in ENGINES_TO_TEST:
        if eng_name not in available and eng_name != "off":
            results["engine_init_times_s"][eng_name] = None
            continue
        if eng_name == "espeak" and not shutil.which("espeak-ng") and not shutil.which("espeak"):
            results["engine_init_times_s"][eng_name] = None
            continue
        print(f"[benchmark] Init-Warmup für {eng_name} …", flush=True)
        init_t, ok = _warmup(eng_name, out_dir)
        if ok:
            results["engine_init_times_s"][eng_name] = round(init_t, 3)
            print(f"[benchmark]   Init-Zeit {eng_name}: {init_t:.2f}s", flush=True)
        else:
            results["engine_init_times_s"][eng_name] = None
            print(f"[benchmark]   {eng_name} nicht verfügbar (Fallback auf off)", flush=True)

    # Latenz pro Engine × Wort-Anzahl × Run.
    for eng_name in ENGINES_TO_TEST:
        print(f"[benchmark] Teste {eng_name} über {len(WORD_COUNTS)} "
              f"Längen × {RUNS_PER_LENGTH} Runs …", flush=True)
        per_len: dict = {}
        for wc in WORD_COUNTS:
            text = SAMPLE_TEXTS[wc]
            runs = []
            for run_i in range(RUNS_PER_LENGTH):
                r = _run_one(eng_name, text, out_dir)
                r["run_i"] = run_i
                runs.append(r)
            # Mittel + Std (nur wenn synth erfolgreich).
            synth_times = [r["synth_time_s"] for r in runs
                           if r.get("available") and "synth_time_s" in r]
            rtfs = [r["rtf"] for r in runs
                    if r.get("available") and "rtf" in r]
            durations = [r["audio_duration_s"] for r in runs
                         if r.get("available") and "audio_duration_s" in r]
            per_len[wc] = {
                "runs": runs,
                "mean_synth_time_s": (round(sum(synth_times) / len(synth_times), 3)
                                       if synth_times else None),
                "mean_audio_duration_s": (round(sum(durations) / len(durations), 3)
                                          if durations else None),
                "mean_rtf": (round(sum(rtfs) / len(rtfs), 3)
                             if rtfs else None),
                "target_word_count": wc,
                "actual_word_count": _word_count(text),
            }
        results["per_engine_per_length"][eng_name] = per_len

    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    return results


def _render_report(data: dict, out_path: Path) -> None:
    """Rendert Markdown-Report aus dem Benchmark-Dict."""
    lines = []
    lines.append("# TTS Latenz-Benchmark Report")
    lines.append("")
    lines.append(f"Gestartet: {data['started_at']}")
    lines.append(f"Beendet: {data['finished_at']}")
    lines.append(f"Python: `{data['python_version'].split()[0]}`")
    lines.append("")
    lines.append("## Verfügbare Engines")
    lines.append("")
    for e in data["available_engines"]:
        lines.append(f"- {e}")
    lines.append("")

    lines.append("## Init-Zeit pro Engine (einmaliger Warmup)")
    lines.append("")
    lines.append("| Engine | Init-Zeit (s) |")
    lines.append("|---|---|")
    for eng, t in data["engine_init_times_s"].items():
        if t is None:
            lines.append(f"| {eng} | n/a (nicht verfügbar) |")
        else:
            lines.append(f"| {eng} | {t:.2f} |")
    lines.append("")

    lines.append("## Latenz pro Antwortlänge")
    lines.append("")
    lines.append("RTF < 1.0 = realtime-schneller; < 0.5 = gefühlt realtime.")
    lines.append("")

    for eng, per_len in data["per_engine_per_length"].items():
        lines.append(f"### Engine: `{eng}`")
        lines.append("")
        # Sammle alle init_error/synth_error der Runs, um sie anzuzeigen.
        first_error = None
        for wc, info in per_len.items():
            for r in info.get("runs", []):
                if r.get("init_error"):
                    first_error = f"init_error: {r['init_error']}"
                    break
                if r.get("synth_error"):
                    first_error = f"synth_error: {r['synth_error']}"
                    break
            if first_error:
                break
        if first_error:
            lines.append(f"⚠ **Engine nicht verfügbar**: `{first_error}`")
            lines.append("")
            continue
        lines.append("| Wörter | Audio-Dauer (s) | Synth-Zeit (s) | TTFA (ms) | RTF |")
        lines.append("|---|---|---|---|---|")
        for wc, info in per_len.items():
            if info["mean_audio_duration_s"] is None:
                lines.append(f"| {wc} | n/a | n/a | n/a | n/a |")
                continue
            ttfa_ms = info["mean_synth_time_s"] * 1000
            lines.append(
                f"| {wc} | {info['mean_audio_duration_s']:.2f} | "
                f"{info['mean_synth_time_s']:.2f} | {ttfa_ms:.0f} | "
                f"{info['mean_rtf']:.2f} |"
            )
        lines.append("")

    # Empfehlung (deterministisch aus den Daten).
    lines.append("## Empfehlung")
    lines.append("")
    realtime_engines = []
    acceptable_engines = []
    too_slow = []
    unavailable = []
    for eng in ENGINES_TO_TEST:
        per_len = data["per_engine_per_length"].get(eng, {})
        runs_30 = per_len.get(30, {}).get("runs", [])
        # Wenn KEIN run erfolgreich war (alle init_error oder synth_error)
        # → nicht verfügbar.
        any_synth = any(r.get("synth_time_s") is not None for r in runs_30)
        if not any_synth:
            unavailable.append(eng)
            continue
        rtf_30 = per_len.get(30, {}).get("mean_rtf")
        if rtf_30 is None:
            unavailable.append(eng)
            continue
        if rtf_30 < 0.5:
            realtime_engines.append(eng)
        elif rtf_30 < 2.0:
            acceptable_engines.append(eng)
        else:
            too_slow.append(eng)
    if realtime_engines:
        lines.append(f"- **Realtime-tauglich** (RTF<0.5 @30 Wörter): {', '.join(realtime_engines)}")
    if acceptable_engines:
        lines.append(f"- **Akzeptabel** (RTF<2.0 @30 Wörter): {', '.join(acceptable_engines)}")
    if too_slow:
        lines.append(f"- **Zu langsam für Default** (RTF≥2.0 @30 Wörter): {', '.join(too_slow)} — "
                     f"nur als explizite Option, wenn der User die Wartezeit akzeptiert.")
    if unavailable:
        lines.append(f"- **Nicht verfügbar / Init-Fehler**: {', '.join(unavailable)} — "
                     f"vor dem Lauf installieren oder Skip-Grund prüfen.")
    lines.append("")
    lines.append("→ Diese Empfehlung wird im nächsten Schritt (Integration) als "
                 "Default-Auswahl in chat_tab.py / streaming_bridge.py verwendet.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Entry-Point ─────────────────────────────────────────────────────


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    print(f"[benchmark] Out-Dir: {out_dir}", flush=True)
    results = run_benchmark(out_dir)
    (out_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    _render_report(results, out_dir / "REPORT.md")
    print(f"[benchmark] Fertig. Output: {out_dir}", flush=True)
    print(f"[benchmark] REPORT.md lesen für Empfehlung.", flush=True)


if __name__ == "__main__":
    main()