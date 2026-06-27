# Plan 6.2c — Few-Shot-Snip + Density-Auto-Strip (Kurzfristig)

## Context

Plan 6.2 (commit 14e86d0) Befund: Variante C (Few-Shot-Turns) gewinnt klar
gegenüber A/B/D/E:

| Metrik | A | **C** |
|--------|---|---|
| tag_rate | 0.6 | **1.0** |
| note_tag_rate | 0.5 | **0.8** |
| dynamic_tag_rate | 0.2 | **0.4** |
| pause_tag_rate | 0.3 | **0.6** |

Highlight p08: A→0 Tags, C→5 mit semantisch intendierten HAPPY/EXCITED/
WHISPER/C#5. **Caveats**: Density-Warnings bleiben (2× wie A), Phantasie-
Tag [#ALARM] taucht auf.

**User-Entscheidung**: "Nur Kurzfristig" + "Density-Warning-Filter (Auto-Strip)".

**Ziel**: 
1. `append_tag_snippet` mit `few_shot=True`-Flag erweitern — injiziert
   die existierenden 3 C-SHOTS als user/assistant-Turns vor der echten
   Frage. **Kein Library-Aufbau** (4 Stile), nur der eine Stil aus C.
2. Density-Auto-Strip im TTS-Engine-Post-Processing: bei Überschreitung
   der Tag-Dichte (>30/100w) werden überzählige Tags aus dem Output
   entfernt, damit TTS nicht überflutet wird.

**Hard Rule**: KEIN motor-touch an `gradio_tabs/vocoder_tags.py` für die
Vocoder-Library (`TAG_VOCAB`, `parse_tags` Signatur). ABER: neue
**Post-Processing-Funktion** `strip_tags_for_engine_with_density_cap` ist
erlaubt — sie nutzt die existierende `parse_tags` und ist orthogonal zur
Library.

KEINE Änderungen an: `config.py`, `model_manager.py`, `patch.py`,
`generators.py`, `server.py`, `schemas.py`, `px_modules/`, `relay_inject.py`.

## Approach

### Schritt 1: `append_tag_snippet` mit `few_shot`-Flag

**Aktueller Vertrag** (`gradio_tabs/system_prompt.py:302-371`):
```python
def append_tag_snippet(messages, snippet) -> list:
    """Hängt snippet an den ersten system-Eintrag an."""
```

**Erweiterung**:
```python
def append_tag_snippet(messages, snippet, *, few_shot: bool = False) -> list:
    """Hängt snippet an system-Eintrag.
    
    few_shot=True: zusätzlich werden die CITMIND_TAG_FEWSHOT_TURNS
    (3 user/assistant-Paare aus c_few_shot.SHOTS) zwischen system-
    Eintrag und ersten user-Turn eingefügt.
    """
```

**Wo kommen die SHOTS her?** Aus `scratches/tag_production/variants/c_few_shot.py`
importieren — aber Cross-Package-Dependency von `gradio_tabs/` nach
`scratches/` ist eine schlechte Richtung. Pragmatischer: SHOTS als
Konstante in `gradio_tabs/system_prompt.py` duplizieren (mit Hinweis auf
Source-of-Truth in c_few_shot.py). ODER: `CITMIND_TAG_FEWSHOT_TURNS` als
neue exportierte Konstante in c_few_shot.py + import. Ich wähle
**Import** (kein Drift zwischen empirischer Quelle und Motor).

**Was passiert mit `merge_system_into_first_user`?** Die Few-Shot-Turns
überleben den Merge, weil `merge_system_into_first_user` nur den
system-Eintrag strippt und in den ersten user-Turn packt. Die
user/assistant-Turns zwischen system und echtem user bleiben als
Multi-Turn-Liste erhalten. Gemma3-Chat-Template iteriert einfach drüber.

### Schritt 2: Density-Auto-Strip im TTS-Engine-Pfad

**Aktueller Vertrag** (`gradio_tabs/vocoder_tags.py:182-216`):
```python
def strip_tags_for_engine(engine, text) -> (clean_text, audio_tags):
    """Engine-spezifisches Strippen."""
```

**NEUE Funktion** (additiv, ändert `strip_tags_for_engine` nicht):
```python
def strip_tags_with_density_cap(engine, text, *, max_tags_per_100_tokens=30) -> tuple:
    """strip_tags_for_engine + Auto-Strip wenn Dichte überschritten.
    
    Bei Überschreitung: behält nur die ersten N Tags (proportional zu
    word_count × max_tags_per_100_tokens / 100), strippt den Rest aus
    dem Output. audio_tags wird ebenfalls gekappt.
    
    Returns: (clean_text, audio_tags, density_warning_str_or_None)
    """
```

Diese Funktion ist eine **additive Helper-Funktion**, die der TTS-Pfad
in `gradio_tabs/chat_actions.py` (oder wo der Aufruf sitzt) optional
nutzen kann. Sie bricht KEINE existierenden Tests, weil sie eine
neue Funktion ist.

### Schritt 3: Tests

Pure-logic, keine GPU. Mindestens:

**`tests/test_system_prompt.py`** (existiert vermutlich, erweitern):
- `test_append_tag_snippet_no_few_shot_default` — few_shot=False
  default-Verhalten, kein Turn zwischen system und user.
- `test_append_tag_snippet_with_few_shot` — few_shot=True fügt 6 Turns
  (3 user + 3 assistant) zwischen system und user ein.
- `test_append_tag_snippet_with_few_shot_empty_snippet` — wenn Snip
  leer, werden trotzdem Few-Shot-Turns eingefügt (Snip ist optional).

**`tests/test_vocoder_tags.py`** (existiert, erweitern):
- `test_strip_with_density_cap_below_threshold_unchanged` — normale
  Dichte (<30) → verhält sich wie `strip_tags_for_engine`.
- `test_strip_with_density_cap_above_threshold_caps_tags` — Dichte >30,
  Tags werden auf Schwelle × word_count/100 gekappt.
- `test_strip_with_density_cap_returns_warning_string` — bei Cap wird
  warning-String zurückgegeben.
- `test_strip_with_density_cap_preserves_first_tags` — die ersten N
  Tags (nach Position) bleiben, der Rest wird gestrippt.

### Schritt 4: Smoke-Run Few-Shot im empirischen Harness

`scratches/tag_production/variants/c_few_shot.py` importiert jetzt aus
`gradio_tabs.system_prompt.CITMIND_TAG_FEWSHOT_TURNS` (oder umgekehrt —
SHOTS werden nach `gradio_tabs` exportiert und C-Variante importiert
sie). Damit ist die Single Source of Truth der Motor, der empirische
Runner konsumiert sie nur.

Ein kleiner Smoke: `python -c "from gradio_tabs.system_prompt import
append_tag_snippet; ...; msgs = append_tag_snippet([{'role':'user','content':'Hallo'}], 'VOCODER', few_shot=True); print(len(msgs))"`
erwartet 1 system + 3 user + 3 assistant + 1 user = 8 turns.

### Schritt 5: Vergleichs-Run (kurz)

Variante A (ohne few_shot) vs C-NEU (mit few_shot via neuem Flag) —
nur 5 Prompts × 256 Token, ~2 Min, validiert dass `append_tag_snippet(
few_shot=True)` in der Praxis die gleiche Compliance bringt wie der
empirische C-Variante-Pfad. **Empirischer Sanity-Check, kein voller
Re-Run.**

## Critical Files

| Datei | Was |
|-------|-----|
| `gradio_tabs/system_prompt.py` | `append_tag_snippet(messages, snippet, *, few_shot=False)` Signatur erweitern, `CITMIND_TAG_FEWSHOT_TURNS` Konstante aus `c_few_shot.py` importieren |
| `scratches/tag_production/variants/c_few_shot.py` | `SHOTS` exportieren als `CITMIND_TAG_FEWSHOT_TURNS` (oder `system_prompt` exportiert die Konstante und C importiert) |
| `gradio_tabs/vocoder_tags.py` | NEUE additive Funktion `strip_tags_with_density_cap` |
| `tests/test_system_prompt.py` | 3 neue Few-Shot-Tests |
| `tests/test_vocoder_tags.py` | 4 neue Density-Cap-Tests |
| `gradio_tabs/chat_actions.py` (oder TTS-Aufrufer) | Optional: `strip_tags_with_density_cap` in TTS-Pfad einklinken — minimal, ein Aufruf-Site |

## Implementation

### Schritt 1.1: SHOTS-Konstante in `c_few_shot.py` exportieren

```python
# variants/c_few_shot.py
CITMIND_TAG_FEWSHOT_TURNS = SHOTS = [
    {"role": "user", "content": "Sage etwas Trauriges in einem Satz."},
    {"role": "assistant", "content": "[#SAD] [#CALM] [#A3]Ich atme [#PAUSE 0.5s]leise.[#PAUSE 0.3s]"},
    {"role": "user", "content": "Jetzt etwas Fröhliches."},
    {"role": "assistant", "content": "[#HAPPY] [#EXCITED] [#C#5]Heute scheint die Sonne![#PAUSE 0.2s]"},
    {"role": "user", "content": "Und flüsternd?"},
    {"role": "assistant", "content": "[#WHISPER] [#A2]Hörst du mich?[#PAUSE 1.0s]"},
]
```

### Schritt 1.2: `append_tag_snippet` Few-Shot-Flag

```python
def append_tag_snippet(messages, snippet, *, few_shot: bool = False) -> list:
    # ... existing code: append snippet to system entry ...
    
    if few_shot:
        # Find first user-turn after system. Insert CITMIND_TAG_FEWSHOT_TURNS
        # between system and user (or at end if no user).
        from scratches.tag_production.variants.c_few_shot import (
            CITMIND_TAG_FEWSHOT_TURNS,
        )
        sys_idx = next((i for i, m in enumerate(out) if m["role"] == "system"), None)
        insert_pos = (sys_idx + 1) if sys_idx is not None else 0
        out = out[:insert_pos] + list(CITMIND_TAG_FEWSHOT_TURNS) + out[insert_pos:]
    
    return out
```

**Lazy-Import OK**: `from scratches...` ist hässlich in `gradio_tabs/`.
Alternative: `CITMIND_TAG_FEWSHOT_TURNS` in `gradio_tabs/system_prompt.py`
als Modul-Konstante definieren, `c_few_shot.py` importiert sie. Single
Source of Truth = `gradio_tabs/system_prompt.py`, c_few_shot ist nur
Wrapper.

### Schritt 2: `strip_tags_with_density_cap`

```python
def strip_tags_with_density_cap(engine, text, *, max_tags_per_100_tokens=30):
    clean_text, audio_tags = strip_tags_for_engine(engine, text)
    word_count = max(1, len(text.split()))
    max_tags = int((word_count * max_tags_per_100_tokens) / 100)
    if len(audio_tags) <= max_tags:
        return clean_text, audio_tags, None
    # Behalte die ersten N audio_tags (nach Position im Original).
    # Überzählige Tags müssen aus clean_text entfernt werden.
    kept = audio_tags[:max_tags]
    dropped = audio_tags[max_tags:]
    # Baue clean_text neu: parse_tags mit cap.
    full_clean, full_tags = parse_tags(text)
    kept_set = set(id(t) for t in kept)
    # ... (Rekonstruktion ohne die dropped tags)
    warning = (f"[vocoder-tags] WARN: Dichte-Cap aktiv — {len(dropped)} "
               f"überzählige Tags entfernt (>{max_tags_per_100_tokens}/100w)")
    return capped_clean, kept, warning
```

Vereinfachung: Statt Rekonstruktion kann ich auf `parse_tags` mit
einem eigenen Stripper setzen, der nur die ersten N Tags pro Kind
behält. Aber das ist motor-touch der Library. Pragmatischer: einfach
die überzähglichen `raw`-Strings aus dem Original-Text strippen.

## Reihenfolge

1. `variants/c_few_shot.py`: `SHOTS` exportieren als `CITMIND_TAG_FEWSHOT_TURNS`
2. `gradio_tabs/system_prompt.py`: `append_tag_snippet(..., few_shot=False)` 
   erweitern + `CITMIND_TAG_FEWSHOT_TURNS` importieren (oder in
   `system_prompt.py` definieren und c_few_shot importieren — ich
   entscheide mich für Definition in system_prompt)
3. `gradio_tabs/vocoder_tags.py`: neue Funktion `strip_tags_with_density_cap`
4. Tests in `tests/test_system_prompt.py` + `tests/test_vocoder_tags.py`
5. Regression: alle existierenden Tests grün
6. Smoke: empirischer Vergleich A vs C-neu (5 Prompts, ~2 Min)
7. Commit

## Verifikation

```bash
# Pure-Logic Tests
/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
  -m pytest tests/test_system_prompt.py tests/test_vocoder_tags.py -v
```

Erwartung: alle existierenden + 7 neue Tests grün.

```bash
# Smoke Few-Shot-Pipeline (kein Modell-Aufruf, nur messages-Bau)
/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -c "
import sys; sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand')
from gradio_tabs.system_prompt import append_tag_snippet, inject_into_messages
from gradio_tabs.vocoder_tags import render_tag_system_prompt
msgs = inject_into_messages([{'role':'user','content':'Hallo'}], 'citmind')
msgs = append_tag_snippet(msgs, render_tag_system_prompt(), few_shot=True)
print(f'turns: {len(msgs)}')  # erwartet: 1 system + 3 user + 3 assistant + 1 user = 8
for m in msgs: print(m['role'], m['content'][:50])
"
```

```bash
# Empirischer Vergleich (5 Prompts × 256 Token)
# A (ohne few_shot) vs C-neu (mit few_shot via neuem Flag)
cd /run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/tag_production
python -c "
import sys; sys.path.insert(0, '..')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand')
# Bau A-Pfad mit few_shot=False
# Bau C-neu-Pfad mit few_shot=True (aber via append_tag_snippet, nicht via
# c_few_shot.SHOTS direkt — Sanity-Check dass Motor den gleichen Effekt hat)
"
```

Oder einfacher: ein neuer Variant F = `append_tag_snippet(..., few_shot=True)`
mit dem CitMind-Pfad aus A. Smoke-Run F vs A.

## Risiken & Edge-Cases

1. **Lazy-Import-Reihenfolge**: `from scratches.tag_production...` in
   `gradio_tabs/system_prompt.py` ist ungewöhnlich. Alternative: 
   `CITMIND_TAG_FEWSHOT_TURNS` in `system_prompt.py` definieren als
   single source of truth, `c_few_shot.py` importiert von dort.
   Vermeidet Cross-Direction-Dependency. **Entscheidung**: Definition
   in `gradio_tabs/system_prompt.py`.

2. **merge_system_into_first_user + few_shot**: Gemma3 lehnt system-Rolle
   ab → system-Inhalt wird in ersten user-Turn gepackt. Few-Shot-Turns
   sind zwischen system und echtem user. Nach merge: user-turn hat
   CitMind-Snippet als Prefix, dann kommt SHOTS[0].user → SHOTS[0].assistant
   → SHOTS[1].user → ... → SHOTS[2].assistant → echter user. Gemma3
   iteriert Multi-Turn-Liste — funktioniert.

3. **Density-Cap-Reihenfolge**: Wenn das Modell 20 Note-Tags in 5
   Wörtern produziert (P10 in C), cappt der Filter auf 1.5 Tags (≤30/100w).
   Das ist sehr aggressiv. Pragmatisch: bei P10 sind die Tags in zwei
   Zeilen, also vielleicht eher 2 Tags pro Zeile = 4 Tags. Cap ist OK.
   Der User will das so.

4. **Phantasie-Tag-Behandlung**: [#ALARM] (aus p07 in C) ist
   unbekannt in der Library, wird aber im parse_tags aus dem Text
   gestrippt. Density-Cap wirkt auf `audio_tags` (nur note/dynamic/pause)
   — Phantasie-Tags werden vorher schon rausgefiltert. Plan 6.2
   Empfehlung #4 (Phantasie-Tag-Metrik) ist ausgeschlossen in dieser
   Kurzfristig-Version.

## Was NICHT in diesem Plan ist

- KEINE 4-Stile-Library (bedrohlich/fragend/poetisch/neutral) — nur die
  existierenden 3 C-SHOTS.
- KEINE Phantasie-Tag-Metrik (parse_tags kennt sie, aber metrics
  zählt sie noch nicht).
- KEIN Degradation-Loop-Fix (EOS-Handling-Bug) — eigener Plan.
- KEINE Änderungen am Webapp-Pfad (`chat_tab.py`, `streaming_bridge.py`)
  für die few_shot=True Default-Setzung. UI-Toggle wäre ein
  Folge-Plan. Aktuell: API ist verfügbar, aber kein UI-Caller.
- KEINE motor-touchs an `parse_tags` Signatur, `TAG_VOCAB`, etc.

## Verwandte Pläne

- Plan 6.1 (16c5750) — Variante A Baseline
- Plan 6.2 (14e86d0) — 5 Varianten Vergleich, C gewinnt
- **Plan 6.2c (dieser Plan)** — Few-Shot als opt-in Flag + Density-Auto-Strip
- Plan 6.2d (postponed) — ABC-Notation alternative API (D zeigte Ausfall)
- Plan 6.2e (cancelled) — CitMind-Default überdenken (E zeigte keine Notwendigkeit)

## Scratch-Konvention

Code-Edits in `gradio_tabs/`, `scratches/tag_production/variants/`,
`tests/`. Edits in `scratches/tag_production/out/` für Smoke-Outputs.
