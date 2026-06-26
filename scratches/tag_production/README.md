# scratches/tag_production — Tag-Produktion-Empirie

**Zweck**: Empirisch klären, ob `gemma3-1b-it` tatsächlich Vocoder-Tags
wie `[#A4]`, `[#C#3]`, `[#PAUSE 0.3s]` produziert, wenn der Standard-
Tag-Snip aus `gradio_tabs/vocoder_tags.py:352-377` via
`append_tag_snippet` (`gradio_tabs/system_prompt.py:302-371`) an eine
CitMind-System-Message angehängt wird.

## Status

- **Plan 6.1 (diese Phase)**: Variante A (Standard) als Baseline.
- **Hypothese**: `tag_rate ≤ 0.2`, `note_tag_rate ≈ 0.0` (Standard-Snip
  listet Noten nur auf, gibt keine Note-Beispiele; CitMind ist Sanskrit-
  Philosophie, kein Musik-Frame).
- **Folge-Pläne (nach A-Befund)**: Variante B (Sanskrit-Mapping), C
  (Few-Shot), D (ABC-Notation) — Strukturen sind angelegt in `variants/`.

## Dateien

| Datei | Zweck |
|-------|-------|
| `prompts.py` | 10 deutsche Test-Prompts + SMOKE_PROMPT |
| `metrics.py` | Pure-Logic Metrik-Aggregator (testbar ohne Modell) |
| `tests/test_metrics.py` | 22 TDD-Tests für `metrics.py` |
| `run.py` | Entry-Point (Smoke + Full + Report-Renderer) |
| `variants/music_sanskrit.py` | Skelett: Sanskrit-Mapping-Snip |
| `variants/few_shot.py` | Skelett: 3 Few-Shot-Paare |
| `variants/abc_notation.py` | Skelett: ABC-Notation-Snip |
| `variants/tag_only.py` | Skelett: Reiner Snip ohne CitMind |
| `out/run_<ts>.json` | Rohdaten (nach Full-Run) |
| `out/smoke_<ts>.json` | Smoke-Test-Rohdaten |
| `out/REPORT.md` | Aggregate-Tabelle + manuelle Notes-Sektion |

## Reproduktion

### Pure-Logic Tests (kein Modell, <1 s)

```bash
cd /run/media/julian/ML4/ollama-work/all_space_6_16_stand
/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
  -m pytest scratches/tag_production/tests/test_metrics.py -v
```

Erwartung: 22 Tests grün.

### Smoke-Test (1 Prompt, ~30 s)

```bash
/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
  scratches/tag_production/run.py --smoke
```

Erwartung:
- Modell lädt (gemma3-1b-it in bf16 ≈ 2 GB GPU)
- 1 Antwort wird generiert
- `out/smoke_<ts>.json` geschrieben
- Print: Antwort + Tag-Counts + Density

### Volllauf (10 Prompts × 1 Seed, 3-15 Min)

```bash
/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
  scratches/tag_production/run.py --full
```

Erwartung:
- `out/run_<ts>.json` mit allen 10 Antworten + Metriken
- `out/REPORT.md` mit Aggregate-Tabelle + Prompt-Tabelle + Notes-Sektion

### Statistische Robustheit (10 Prompts × 3 Seeds)

```bash
/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
  scratches/tag_production/run.py --full --seeds 3
```

## Methode

### System-Prompt-Aufbau (1:1 wie Server-Pfad)

```python
from gradio_tabs.system_prompt import (
    inject_into_messages, merge_system_into_first_user,
    append_tag_snippet,
)
from gradio_tabs.vocoder_tags import render_tag_system_prompt

base = [{"role": "user", "content": user_prompt}]
msgs = inject_into_messages(base, profile_name="citmind")   # CitMind prepend
msgs = append_tag_snippet(msgs, render_tag_system_prompt()) # Tag-Snip
msgs = merge_system_into_first_user(msgs, profile_name="citmind")  # Gemma3-Wrap
```

System-Content-Länge bei Variante A:
- CitMind-Profil: ~3654 Zeichen
- Tag-Snip: 1325 Zeichen
- Total: ~4979 Zeichen in erstem user-turn geprefixet

### Generation

```python
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**enc, max_new_tokens=256, do_sample=True,
                      temperature=0.7, top_p=0.9, use_cache=True,
                      eos_token_id=tokenizer.eos_token_id,
                      pad_token_id=tokenizer.eos_token_id)
answer = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
```

### Metriken

Pro Antwort (`compute_per_response_metrics`):
- `total_tags` — Anzahl Tags aller Kinds (note/dynamic/affect/pause/bark)
- `note` — Anzahl Note-Tags (`[#A4]`, `[#C#3]` etc.)
- `has_note` / `has_dynamic` / `has_affect` / `has_pause` — bool
- `density_per_100w` — Tags pro 100 Wörter
- `density_warning` — None oder WARN-String (>30 Tags/100 Wörter)
- `clean_text` — Original ohne Tags
- `raw_tags` — Liste aller Tags (kind/value/offset/raw)

Über alle Antworten (`aggregate_run_metrics`):
- `tag_rate` — Anteil Antworten mit ≥1 Tag
- `note_tag_rate` — Anteil Antworten mit ≥1 Note-Tag
- `tags_per_100_words_global` — global über alle Antworten
- `density_warnings_count` — Anzahl Antworten mit Dichte-Warnung

## Manuelle Pflicht-Lesung

**Vor jedem Verdikt**: mindestens 3 Antworten vollständig lesen.

Pro Antwort in `out/run_<ts>.json` anpassen (`mechanistic_notes`):

1. Sind Tags syntaktisch korrekt (Parser-Test grün)?
2. Sind Tags semantisch intendiert (nicht nur Echo der Aufgabe)?
3. Was fällt am Output auf — Vokabular-Referenzen ohne `[#…]`?
   Beispiel: schreibt das 1B „Note", „Pause", „flüstere" im Klartext,
   OHNE die Tag-Syntax zu nutzen?
4. Wenn Antwort leer / fehlerhaft: warum?

Edits in den Output-Files sind erlaubt (Scratch-Konvention).

## Tabu-Bestätigung

KEINE Datei außerhalb `scratches/tag_production/` wurde geändert. Der
Standard-Pfad (`gradio_tabs/`, `config.py`, `model_manager.py`, etc.)
bleibt bit-identisch zum Stand vor diesem Run.

```bash
git status          # zeigt nur scratches/tag_production + out/
git diff --stat     # keine Änderungen an Motor-Dateien
```

## Hypothese vs Befund (nach Lauf ausfüllen)

**A-Erwartung** (Stand vor Lauf):

| Metrik | Erwartung |
|--------|-----------|
| `tag_rate` | ≤ 0.2 |
| `note_tag_rate` | ≈ 0.0 |
| `dynamic_tag_rate` | ≈ 0.1 (p04, p07) |
| `pause_tag_rate` | ≈ 0.0 (kein Note-Cue) |

**A-Befund** (nach Lauf):

| Metrik | Befund | Kommentar |
|--------|--------|-----------|
| `tag_rate` | | |
| `note_tag_rate` | | |
| `dynamic_tag_rate` | | |
| `pause_tag_rate` | | |

**Falls `note_tag_rate ≥ 0.5`**: Variante A funktioniert → Plan 6.2 mit
Variant B/C/D Vergleich.

**Falls `note_tag_rate < 0.3`**: Variante B (Sanskrit-Mapping) zuerst
testen — passt zur CitMind-Ontologie.

**Falls `tag_rate ≈ 0.0`**: Snip-Diagnose (möglicherweise zu lang, oder
Tokenizer splittet `[#` falsch). Plan 6.2b.

## Verwandte Pläne

- Plan 5.1 (TTS-Integration): definiert `vocoder_tags.py` + Engines
- Plan 5.3 (WebUI-Settings): definiert `append_tag_snippet`-Pfad
- Plan 6.2 (Folge): Variant-Vergleich + TTS-Integration-Validierung
