# Plan — Branch `ui-styling` (UI-Verschönerung)

**Branch-Basis:** master @ 286e987
**User-Scope (bestätigt):** Nur Styling — keine Funktionsänderung
**Status:** Scope-Dokument vor Implementierung (DevMind-Regel: 2× fragen)

## Ziel

Visuelle Politur der Gradio-WebUI (chat_tab + TTS-Accordion + Profile-Editor + Sessions-Tab), ohne die zugrundeliegende Logik anzufassen. Schneller ästhetischer Gewinn mit minimalem Risiko.

## In-Scope

### 1. CSS-Variablen + Theme-Basis
- Einheitliche CSS-Variablen für Farben, Spacing, Border-Radius in einer neuen Datei `gradio_tabs/_styles.py`.
- Definiert einen `--bg-primary`, `--bg-secondary`, `--accent`, `--text-primary` etc., aus denen die Komponenten ihre Werte ziehen.
- Keine Hardcoded-Farben in chat_tab.py mehr; alle bestehenden Inline-Styles durch Variablen ersetzt.

### 2. Chatbot-Polish
- **Bubble-Alignment** User rechts / Assistant links (heute unklar).
- **Avatar-Platzhalter** für User und Assistant (Initial-Buchstabe in Kreis, kein Bild-Asset nötig).
- **Code-Block-Highlight** via gr.Code(component) statt reinem Markdown für ```-Blöcke.
- **Markdown-Rendering** für Assistant-Antworten aktivieren (gr.Markdown statt plain text).
- **Token-Counter**-Badge rechts oben in der Toolbar (prompt_tokens / completion_tokens / total).

### 3. Sidebar / Layout
- Linke Sidebar mit Logo, Session-Liste, Profile-Auswahl in eigenem Block.
- Hauptbereich Chat + rechte Sidebar mit Settings-Akkordeon (TTS, Profile, Auto-Tune).
- 3-Spalten-Layout via `gr.Row` + `gr.Column(scale=...)`.
- Tabs oben ersetzen Sidebar-Tabs (Settings/Sessions/Profile/TTS) zu Bottom-Panel-Accordion.

### 4. Komponenten-Polish
- Buttons: einheitliche Border-Radius (8px), Hover-Effekt, Disabled-State deutlich.
- Accordions: Pfeil-Rotation beim Öffnen, Übergangs-Animation.
- Dropdowns: Suche-Prompt wenn >5 Items, klare Selected-Highlight-Farbe.
- Textbox: Placeholder-Italic, Focus-Border akzentfarben.

### 5. TTS-Accordion-Schönheit
- Engine-Auswahl als Radio-Buttons mit Engine-Icon (Buchstabe oder Initial in Quadrat).
- Audio-Player zentriert statt full-width.
- Status-Pill oben rechts ("Engine: piper ✓").

## Out-of-Scope (EXPLIZIT)

- **Keine** neuen Features (z.B. kein Multi-Chat, kein Theme-Toggle Light/Dark, keine Animation-Frameworks).
- **Keine** Änderungen an Server-Endpoints, generators.py, patch.py, model_manager.py — **Motor unangetastet** (DevMind-Regel: kein motor-touch).
- **Keine** Änderungen an der Session-Logik, Auto-Tune-Lock, Profile-Resolution.
- **Keine** Änderungen an Tests, die Logik pinnen — nur ggf. neue CSS-/Visual-Tests (Pillow-Snapshot-Tests von Gradio-Komponenten, falls Zeit reicht).
- **Keine** Dependency-Additions (kein tailwindcss, kein npm, alles in reinem Gradio-CSS/Inline-Style).

## Erlaubte Dateien (lesen + editieren)

- `gradio_tabs/chat_tab.py` (UI-Komposition)
- `gradio_tabs/_styles.py` (NEU — CSS-Variablen)
- `app.py` (nur Mount-Order / Reihenfolge der Tabs, KEINE Logik)
- `gradio_tabs/__init__.py` (Re-Exports der neuen Helper)
- `tests/test_ui_styling.py` (NEU — Smoke-Tests: Komponenten gerendert, CSS-Vars vorhanden)

## Verifikation

1. `bash run.sh` startet sauber, keine Warnings.
2. UI lädt in <2s (kein zusätzlicher Network-IO).
3. Alle bestehenden Tests grün (189+).
4. Manuelle Sichtprüfung: Chat-Bubbles aligned, Avatar sichtbar, TTS-Akkordeon ästhetisch, keine 1-Pixel-Bugs.
5. Optional: Pillow-Snapshot-Test (Vergleich vorher/nachher — hartes Kriterium für "kein Stil-Regress").

## Risiken

- **Gradio-CSS-Spezifität**: Inline-Styles in Gradio-Komponenten werden oft von Gradio-Defaults überschrieben. Mitigiert durch `css=`-Argument in `gr.Blocks(css=...)` + höhere Spezifität.
- **Visuelle Regression in 1b vs 4b**: Falls das UI Preset-spezifische States hat (PX-Engine-Aktiv-Indikator). Smoke-Test in beiden Presets.
- **Lade-Regression**: Zu viele CSS-Variablen können Theme-Init verlangsamen. Mitigiert durch <50 Variablen.

## Reihenfolge

1. `gradio_tabs/_styles.py` mit zentralen CSS-Variablen.
2. `chat_tab.py` minimal-invasiv: Inline-Styles → Variablen, Avatar hinzu, Markdown-Renderer.
3. Layout-Refactor 3-Spalten (Sidebar + Chat + Settings).
4. TTS-Accordion-Polish.
5. Tests + Smoke (Server lokal, Screenshot-Vergleich).
6. Commit pro Sub-Bereich, nicht ein Mega-Commit.

## Was NICHT in diesem Plan ist

- KEINE Auto-Tune-Logik-Änderung.
- KEIN Light/Dark-Toggle (User-Wahl: "Nur Styling", keine Features).
- KEINE neuen TTS-Engines.
- KEINE Modal-Overlays oder Toasts (Gradio-Standard reicht).
- KEINE Mobile-Optimierung (Gradio handhabt das selbst).

## Verwandte

- Branch `self-conversation` (separater Branch, separat dokumentiert)
- Branch `relay-layer-discovery` (separater Branch, separat dokumentiert)
