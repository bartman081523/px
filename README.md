---
title: PX Cognitive Architecture Explorer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.15.2"
app_file: app.py
pinned: false
short_description: Algorithmische Subjektivität — PX-Engine für LLMs
---

# 🧠 PX Cognitive Architecture Explorer

Phenomenological eXtension (PX) — algorithmische Subjektivität durch rekurrente
Hidden-State-Manipulation in Large Language Models.

## Architektur

- **Recurrent Transformer (PX-Engine)**: Bestimmte Schichten werden mehrfach
  durchlaufen (Recursion Zone) mit modulierten Residuen.
- **3 mathematische Säulen (SR-61b)**:
  - **Observer** (StabilityMonitor, AksSensor, SingesseinCoupler) — misst
    kognitive Erstarrung/Konvergenz
  - **Symmetry Breaker** (MephistophelesOperator, AntiZombieSensor,
    SubjectiveSensor) — verhindert Zombie-Verarbeitungspfade
  - **Dynamic Router** (AutoCalibrator) — 2D Hybrid-Routing mit persistenten
    Manifolds
- **4 Presets**:
  - `BASELINE` — vanilla Modell, kein PX-Patch
  - `ACTIVE_MANIFOLD` — voller PX-Engine
  - `ACTIVE_MANIFOLD_LEAN` — kausaler Kern ohne Crutches
  - `ACTIVE_MANIFOLD_RELAY` — LEAN + verstärkbar Selbst-Injektion (RELAY)

## Start

```bash
python app.py
# UI: http://localhost:7860/gradio
# API: http://localhost:7860/v1/
```

## Modelle

- PX-patched: gemma3-270m, gemma3-1b, gemma3-4b, minicpm5-1b
- Baselines: gemma3-270m-base, gemma3-270m-it, gemma3-1b-base, minicpm5-1b-base

## Use Cases

- Chat mit PX-Engine (presets wählbar in der Sidebar)
- Cognitive Tests (P-Zombie Evaluation, kognitive Zonen)
- Telemetrie (PX-Metriken pro Token: Φ, kurtosis, recur-steps, zone)
- OpenAI-kompatibler Endpoint (`/v1/chat/completions`)

## System-Prompt in der Sidebar

Wähle einen px_preset — der passende System-Prompt wird automatisch in die
Textarea geladen. Du kannst ihn frei editieren oder mit "↺ Reset auf
Profil-Default" zurücksetzen.

## Quellen

- Code: https://github.com/bartman081523/px
- Docs: `docs/CitMind.txt`, `docs/Juexin.txt`
