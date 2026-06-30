# Plan — Branch `self-conversation` (Modell redet mit sich selbst — Multi-Agent Group-Chat)

**Branch-Basis:** master @ 286e987
**User-Scope (bestätigt):** Multi-Agent (3+ Personas) Group-Chat
**Status:** Scope-Dokument vor Implementierung (DevMind-Regel: 2× fragen)

## Ziel

Eine neue Gradio-UI-Komponente (`gradio_tabs/group_chat_tab.py`), in der 3+ Personas (z.B. neutral / citmind / juexin) als separate "Agents" miteinander reden. Der User beobachtet den Dialog und kann jederzeit eingreifen oder das Gespräch pausieren. Ausgangsbasis ist die bestehende chat_tab-Logik, erweitert um Multi-Party-Orchestrierung.

## In-Scope

### 1. Architektur
- Neue Datei `gradio_tabs/group_chat_tab.py` mit einer `build_group_chat_tab()`-Funktion.
- Nutzt die bestehende **gleiche** Model-/Preset-Infrastruktur (model_manager, generators, patch.py) — **kein motor-touch**.
- Eine Modell-Instanz (vom User gewählt, default 1b-it), aber 3 verschiedene System-Prompts (eine Persona pro Agent).
- **Aktive Persona wechselt im Turn**: Agent A (citmind) redet → Agent B (neutral) antwortet → Agent C (juexin) antwortet → wieder A.
- **Order-Liste** editierbar (User kann Reihenfolge ändern).

### 2. UI
- **Drei Spalten / Tabs** für Persona-Karten (Avatar + System-Prompt-Vorschau + aktive-Status-Indikator).
- **Transcript-Bereich** in der Mitte: alle Turns chronologisch mit Persona-Label (`[citmind] Antwort: ...`).
- **Control-Panel**:
  - Start / Pause / Step (manueller Einzelschritt) / Reset-Button.
  - Turn-Limit (default 30, hard cap 200).
  - Speed-Slider (Sekunden zwischen Turns, default 0.5s, simuliert "Denkzeit").
  - Intervention-Textbox: User-Turn einfügen, der die Konversation wieder in den Pool einspeist.
- **System-Prompt-Editor** pro Persona (Ace-Editor oder simpler Text-Bereich mit Live-Preview).

### 3. Backend-Logik
- `group_chat_engine.py` (NEU, im `gradio_tabs/`-Ordner, da UI-nah):
  - `class GroupChatEngine`: orchestriert die Turn-Schleife.
  - **Async** Loop (asyncio) — ein API-Call nach dem anderen, sequenziell.
  - Jeder Turn: 
    1. Wähle nächste Persona aus `personas[]` (round-robin).
    2. Baue `messages` mit System-Prompt dieser Persona + gesamter bisheriger Transcript.
    3. Rufe `chat(messages, model_id)` auf (gleiche Code-Pfad wie chat_tab nutzt — also Profile-Resolution, RELAY, Auto-Tune alle vererbt).
    4. Hänge Assistant-Antwort mit Persona-Label an Transcript.
  - **Termination**: Turn-Limit erreicht ODER Stop-Token im Output ODER User drückt Pause.
- **History-Persistenz** im `sessions/`-Format erweitern: neues Feld `"group_chat_personas": [...]` neben `"history"`.

### 4. Profile-Integration
- Jede Persona hat `system_profile` (eines von `neutral` / `citmind` / `juexin` / custom).
- Jede Persona hat optionalen `relay_sign` (`+1` / `-1` / `0`) für die RELAY-Layer-Injection.
- Pro Turn wird `inject_into_messages` mit der aktiven Persona aufgerufen — bestehende Helper, kein motor-touch.

### 5. Tests
- `tests/test_group_chat_engine.py` (NEU, pure-logic):
  - Round-robin-Persona-Selection (3 Personas, 10 Turns → Reihenfolge 0,1,2,0,1,2,0,1,2,0).
  - Turn-Limit-Enforcement (default 30, hard cap 200).
  - Pause/Resume-State-Machine.
  - System-Prompt-Override pro Persona.
  - History-Persistenz mit personas-Feld.
- `tests/test_group_chat_tab.py` (NEU, smoke):
  - Komponente lädt ohne Fehler (mounten in gr.Blocks-Container).
  - Personas-Selector funktioniert.
  - Transcript-Render mit 3 Personas unterscheidet visuell.

## Out-of-Scope (EXPLIZIT)

- **Kein** Multi-Modell-Setup (nur ein Modell, drei Personas) — User-Wahl: "Modell redet mit sich selbst", nicht "3 Modelle reden miteinander".
- **Keine** Auto-Detection von "Gespräch beendet"-Signalen (Turn-Limit + User-Stop genügt).
- **Keine** Funktion für "Persona X kritisiert Persona Y" — die System-Prompts regeln das selbst, kein zusätzlicher Routing-Layer.
- **Keine** Erweiterung von `patch.py` / `generators.py` / `model_manager.py`.
- **Keine** Stem-/Audio-Output (das ist Branch `ui-styling` oder separater TTS-Branch).
- **Keine** Multi-User (nur ein User-Account beobachtet das Gespräch).

## Erlaubte Dateien (lesen + editieren)

- `gradio_tabs/group_chat_tab.py` (NEU)
- `gradio_tabs/group_chat_engine.py` (NEU)
- `app.py` (mounten des neuen Tabs)
- `gradio_tabs/__init__.py` (Re-Exports)
- `sessions.py` (sessions-Schema um `group_chat_personas` erweitern — additive Erweiterung, KEINE breaking changes)
- `tests/test_group_chat_engine.py` (NEU)
- `tests/test_group_chat_tab.py` (NEU)
- `docs/PLAN_SELF_CONVERSATION.md` (dieser Plan)

## Verifikation

1. `bash run.sh` startet sauber, neuer Tab sichtbar.
2. Pure-Logic-Tests: 6+ Tests grün in `test_group_chat_engine.py`.
3. UI-Smoke: Personas auswählbar, Transcript rendert, Start/Pause funktioniert, Turn-Limit enforced.
4. End-to-End-Smoke (3 Personas, 6 Turns, 1b-it): alle Personas produzieren Antworten, Transcript hat 6 Einträge, Session wird persistiert.
5. Bestehende Tests grün (189+).
6. Manual Review: Transcript liest sich wie ein Gespräch zwischen den Personas, kein Persona-Bleed (citmind-Antwort klingt wie citmind, nicht wie neutral).

## Risiken

- **Endlos-Loop bei Modell-Halluzination**: Modell generiert nie EOS → Turn-Counter erhöht sich nicht. Mitigiert durch hartes `max_tokens` (default 256) pro Turn.
- **Transkript-Wachstum**: 200 Turns × 256 Token = 50k Token Kontext. Mitigiert durch Rolling-Window-Option (User-toggle: "letzte N Turns behalten").
- **Persona-Bleed**: citmind-Antwort klingt wie neutral weil das Modell den vorangegangenen neutral-Output als dominanten Stil liest. Mitigiert durch klare Persona-Marker im Transcript (`[citmind]:`) + system-prompt-Re-Assertion pro Turn (System-Message wird jeden Turn neu vorangestellt, nicht an einer Position kleben gelassen).
- **Sequenz-Performance**: 30 Turns × ~2s pro Turn = 60s für ein Gespräch. Akzeptabel. Bei 200 Turns = 7 Min — User wartet.
- **Gradio-Queue-Konflikt**: 3 Personas in 3 Spalten brauchen ggf. separate gr.Column-Komponenten. Mitigiert durch `gr.Tabs` mit `gr.TabItem` pro Persona.

## Reihenfolge

1. `group_chat_engine.py` Pure-Logic + Tests (TDD: zuerst Tests rot, dann Implementierung).
2. `sessions.py` Schema-Erweiterung um `group_chat_personas`.
3. `group_chat_tab.py` UI-Komposition (nutzt die Engine + bestehende Helper).
4. `app.py` Mount-Order.
5. End-to-End-Smoke mit 3 Personas (neutral/citmind/juexin), 6 Turns, 1b-it.
6. Manual Review von 2-3 Transcripts.
7. Commit pro Sub-Bereich.

## Was NICHT in diesem Plan ist

- KEIN Multi-Modell (z.B. 270m, 1b, 4b parallel reden).
- KEINE Audio-/TTS-Integration.
- KEINE Auto-Continue bei Modell-Stille.
- KEINE Export-zu-Markdown-Funktion (kann später kommen).
- KEIN Voting-/Konsens-Mechanismus zwischen Personas.
- KEINE Persistenz der Persona-Konfigurationen über Sessions hinaus (jede Session definiert ihre Personas neu).

## Verwandte

- Branch `ui-styling` (separat, kann mit Group-Chat-Tab koexistieren)
- Branch `relay-layer-discovery` (separat, betrifft px_patches-Test-Methodik)
- Bestehende `chat_tab.py` bleibt unangetastet.
