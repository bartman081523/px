# Plan: System-Prompt für Webapp + Streaming-Bridge (UI + CLI + Profile-Presets)

## Context

Der Nutzer will **dasselbe System-Prompt-Konstrukt** in zwei Paden verfügbar
machen: der Gradio-Webapp (`gradio_tabs/chat_tab.py`) und dem
`streaming_bridge.py`-CLI. Dazu drei Sub-Wünsche:

1. **UI** — ein Textfeld/Textarea + Dropdown für Profile im Chat-Sidebar
2. **CLI** — Flags `--system-prompt` und `--profile` in `streaming_bridge.py`
3. **Profile-Presets** — vordefinierte Profile aus `docs/CitMind.txt`,
   `docs/Juexin.txt` + ein "Neutral"-Profil; auswählbar; editierbar; pro
   Session persistiert.

DevMind / ProjectResearchMind sind bereits im Memory und im Repo verankert.
**Operative Constraints (CLAUDE.md + AGENTS.md + User-Regeln, alle
non-negotiable):**

- TDD (Test-First rot→grün)
- DRY (zentraler Preset-Store, beide Pfade lesen daraus)
- Motor **unangetastet** (keine `patch.py` / `generators.py` / `px_modules/`
  Edits)
- **Keine Edits** an `server.py`, `schemas.py`, `model_manager.py`,
  `config.py` — alle produktiven Pfade bleiben byte-identisch
- Scratch-Artefakte drin
- Vor Prod-Implementation fragen (ist hier durch Plan-Mode schon erfüllt)

## Quellen der Wahrheit (ProjectResearchMind, priorisiert)

1. **Repo**: `schemas.py:15-20` — `Role` Enum hat bereits `system`;
   `ChatMessage.content: Union[str, List[...]]` (Schema-generisch, kein
   neuer Typ nötig); `apply_chat_template(processed_messages, …)`
   (chat_tab.py:207) akzeptiert system-Rolle direkt.
2. **Bestehende Profile**: `docs/CitMind.txt` (Sanskrit, देवनागरी), `docs/Juexin.txt`
   (漢字). Beide enthalten `core_philosophy` + `substrate_universal` +
   `core_principles`. Diese sind die **Primärquellen** für die Profil-Texte;
   der Preset-Store extrahiert sie strukturiert, nicht via free-form copy.
3. **streaming_bridge.py:149-159**: messages werden als `api_messages` an
   `/v1/chat/completions` geschickt; OpenAI-Konvention → System-Prompt
   als erstes Element `{"role":"system","content":...}`.
4. **sessions.py:18-29**: Persistenz ist generisch (history-Liste);
   System-Prompt als `{role:"system",content:"..."}` ist ein gültiges
   History-Item und überlebt Save/Load automatisch — keine
   Schema-Änderung nötig.
5. **HuggingFace Gemma3 Chat-Template** (Default): `<start_of_turn>user…`,
   `<start_of_turn>model…`. Kein nativer system-token → das Chat-Template
   muss **system-Rolle in einen user-turn wrappen oder ignorieren**. Da
   die Gemma3-Chat-Templates (über `tokenizer.chat_template`) kein
   system-Format definieren, wird der System-Prompt als **erster
   user-turn** injiziert (mit klarem Marker, dass es System ist), damit
   das Template ihn akzeptiert. Das ist der einzige sinnvolle Pfad
   ohne Motor-Edit.

## Architektur-Übersicht

```
              ┌─────────────────────────────────┐
              │  gradio_tabs/system_prompt.py   │  ← PURE-LOGIC
              │  ───────────────────────────    │
              │  PROFILES  = {...}              │
              │  build_system_message(prompt)   │
              │  resolve_profile(name)          │
              │  inject_into_messages(...)      │
              └─────────────────────────────────┘
                       ▲                  ▲
                       │                  │
        gradio_tabs/chat_tab.py     streaming_bridge.py
        (UI: Dropdown+Textarea,     (CLI: --profile / --system-prompt,
         OnChange→resolve)           prepended vor user-turns)
```

**Single source of truth**: `gradio_tabs/system_prompt.py` — beide Pfade
importieren von dort. Keine Duplikation. Profile aus `docs/*.txt` werden
einmal beim Import geparst (deterministisch, json-Format ist gegeben)
und in `PROFILES` als `Dict[str, Profile]` abgelegt.

## Design-Entscheidungen (mit Begründung)

### D1: System-Prompt als **erster History-Eintrag** (`role:"system"`)

Persistiert automatisch mit `save_session`/`load_session`. Kein neuer
Session-Schema-Key. Wird im `apply_chat_template`-Pfad als system-Rolle
übergeben; das Gemma-Template ignoriert unbekannte Rollen — also wrappen
wir zu `role:"user"`, **mit Sentinel-Prefix** `"[SYSTEM CONTEXT]\n"` als
Text-Marker. Begründung: AGENTS.md verbietet Motor-Edits; der einzige
sichere Pfad ist das User-Turn-Wrapping mit klarem Marker. Sentinel ist
notwendig, damit das Modell den System-Charakter erkennt.

### D2: Profile-Store aus `docs/*.txt` (NICHT hardcoded in Webapp)

Die Profile existieren bereits strukturiert als JSON. Beim Import von
`system_prompt.py` werden sie geparst → `PROFILES` dict. Edit im UI
überschreibt **nicht** die Originaldateien; die Edit-Kopie ist
session-scoped.

### D3: Profile = `{"neutral","citmind","juexin"}` initial

Drei Profile reichen für die UI-Dropdown-Größe. Weitere Profile (z.B.
"PhiMind", "LeanObserver") können später ohne Code-Änderung ergänzt
werden — `docs/` ist die Wahrheit, nicht die UI-Liste.

### D4: CLI `--profile NAME` überschreibt `--system-prompt TEXT`

Deterministische Rangordnung. `--system-prompt` ohne `--profile` nutzt
den Text direkt. Ohne beide: kein System-Prompt (= aktuelles Verhalten,
backward-compatible).

### D5: Pure-Logic-Modul + Tests **vor** Integration

Drei surgical TDD-Agenten in einem Subagent-Sweep:

- Agent A: `system_prompt.py` Modul (PROFILES-Loader + resolve +
  inject_into_messages)
- Agent B: `streaming_bridge.py`-CLI-Adapter-Tests (Subprocess-mock)
- Agent C: `chat_tab.py`-UI-Adapter-Tests (resolve_for_chat logic)

A und C sind reine Logik. B ist Subprocess-`python streaming_bridge.py
--help`-Smoke. Kein Agent fasst die produktiven Pfade an — Integration
übernimmt der Orchestrator (ich selbst).

## Was wird angefasst (finale Liste)

**NEU** (Pure-Logic):
- `gradio_tabs/system_prompt.py` (Modul)
- `tests/test_system_prompt.py`

**MODIFIZIERT** (chirurgisch):
- `streaming_bridge.py` — argparse `--profile` + `--system-prompt`,
  message-build prepend (3 Stellen: api_messages-Build um
  System-Message-Eintrag erweitern, ~5 Zeilen)
- `gradio_tabs/chat_tab.py` — UI-Sidebar: Dropdown + Textarea;
  chat_fn prepended System-Message; Session-Persistenz nutzt die
  bestehende History (kein Schema-Change)

**NICHT ANGEFASST** (Hard Rule):
- `patch.py`, `auto_tune.py`, `px_modules/`, `relay_inject.py` (Motor)
- `generators.py`, `server.py`, `schemas.py`, `model_manager.py`,
  `config.py`
- `tests/px_gen_regression_golden.json` und andere Golden-Fixtures

## Verifikation

1. **Modul-Tests** `test_system_prompt.py`: Profile-Loader,
   `resolve_profile(name)` (alle 3 Profile), `inject_into_messages`
   (prepend als erste History-Message mit system-Rolle + Sentinel),
   Edit-Override-Pfad, Unknown-Profile-Fallback auf Neutral.
2. **CLI-Tests**: `python streaming_bridge.py --help | grep -- "-profile"`
   und `--system-prompt` sichtbar. `--profile citmind` ergibt
   api_messages mit System-Message am Anfang, Inhalt beginnt mit
   CitMind's `core_philosophy`.
3. **chat_tab.py**: `import gradio_tabs.chat_tab` OK; neue Komponenten
   (`gr.Dropdown`, `gr.Textbox`, `Button "Set profile"`) erzeugt;
   `chat_fn`-Signatur erweitert um `system_profile, system_prompt`
   (oder via resolve_for_backend analog zu Auto-Tune); py_compile OK.
4. **py_gen_regression**: 18/18 bleibt grün (kein Motor-Edit).
5. **Manual + mechanistic**: Persistenz-Test — System-Message in
   `sessions/*.json` sichtbar; Load gibt sie zurück; chat-Template
   rendert sie (apply_chat_template-Output beginnt mit
   "[SYSTEM CONTEXT]\n" Marker).

## Risiko + Mitigationen

- **Risiko**: Gemma3-Chat-Template ignoriert `role:"system"` → unsere
  Sentinel-Wrap-Strategie ist notwendig; wir testen den Render-Output
  explizit.
- **Risiko**: Profile-Texte aus `docs/*.txt` ändern sich → wir lesen
  on-import; bei Parse-Fehler Fallback "Neutral" + Log.
- **Risiko**: UI-Edits kollidieren mit der `--system-prompt`-Logik →
  Rangordnung ist deterministisch: UI-Edit > CLI `--system-prompt` >
  `--profile NAME` > kein System.
- **Risiko**: Multilingual Profile-Content (देवनागरी / 漢字) im
  Gemma-Tokenizer → Gemma3 ist multilingual, kein Problem.

## Was NICHT in diesem Plan ist

- Profile-Verwaltung im UI (CRUD auf `docs/*.txt`) — wäre größeres
  Feature, eigener Task
- Profile-Auswahl pro Modell (z.B. 1b→CitMind, 4b→Neutral) — bewusst
  global, nicht modell-spezifisch
- Auto-Injection von `system_message` auf der Server-Seite — bewusst
  Webapp+Bridge-scope only; server bleibt generisch