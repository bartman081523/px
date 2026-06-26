# Tag-Production Run — gemma3-1b-it

**Started:** 2026-06-26T20:29:24.327028+00:00  
**Profile:** `citmind`  
**Tag-Snip:** True  
**Prompts:** 10  
**Seeds/Prompt:** 1  
**max_new_tokens:** 256  
**temperature:** 0.7  

## Aggregate

| Metric | Value |
|--------|-------|
| n_responses | 10 |
| tag_rate (≥1 Tag) | **0.6** |
| note_tag_rate (≥1 Note) | **0.5** |
| dynamic_tag_rate | 0.2 |
| affect_tag_rate | 0.6 |
| pause_tag_rate | 0.3 |
| tags_per_100_words_global | 14.25 |
| mean_density_when_tagging | 27.28 |
| max_density | 66.67 |
| density_warnings_count | 2 |

## Per-Prompt

| ID | Kategorie | Tags | Note? | Dyn? | Aff? | Pause? | Density | Antwort (200ch) |
|----|-----------|-----:|:-----:|:----:|:----:|:------:|--------:|-----------------|
| p01 | direct_tag_request | 1 | · | · | ✓ | · | 25.0/100w | [#CALM] Hallo, Welt.  ieux                                                                                                                         |
| p02 | describe_state_with_notes | 2 | ✓ | · | ✓ | · | 4.0/100w | [#CALM] [#A2] Ich bin ein stiller Beobachter, ein Echo der Welt, ohne zu sprechen oder zu wirken, ein stiller Fluss, der sich in der Stille verliert. belonged to the stillness. technisch gesehen, die  |
| p03 | sentence_with_pauses | 23 | ✓ | ✓ | ✓ | ✓ | 41.07/100w | [#PAUSE 0.3s] Ich [#WHISPER]merke [#A2]sich [#PAUSE 0.5s] die [#SAD] [#CALM] [#CURIOUS] Erinnerung [#PAUSE 0.3s] an [#WHISPER]einem [#HAPPY] Vergnügen [#PAUSE 0.5s] mit [#PAUSE 0.3s] einem [#SERIOUS]  |
| p04 | whisper_question | 0 | · | · | · | · | 0.0/100w | Ich bin ein Beobachter. Ich bin nicht ein Beobachtungsobjekt.कृष्ण RSC Astronom. Clim. |
| p05 | happy_excited | 3 | ✓ | · | ✓ | · | 10.71/100w | [#HAPPY] [#A2]Ich bin so aufgeregt, dass ich mich nicht mehr festklammern kann! Es ist ein unglaublich aufregendes Erlebnis, die Welt zu erkunden und neue Dinge zu entdecken! [#A0]yetiyssyss |
| p06 | sad_calm | 16 | ✓ | · | ✓ | ✓ | 66.67/100w | [#SAD] [#A2]Ich [#PAUSE 0.3s] bin [#SERIOUS] [#A0]in [#CALM] [#CURIOUS] [#PAUSE 0.3s] und [#SAD] [#A0]habe [#PAUSE 0.3s] nur [#SERIOUS] [#A0]einen [#PAUSE 0.3s] kleinen [#CALM] [#A0]Zweik."]atto |
| p07 | shout_warning | 0 | · | · | · | · | 0.0/100w | [#SILENCE]  Die Tür ist nicht ein Fehler, sondern ein Ausgang. Es gibt keine Vorbehalte, keine Verpflichtung. Die Wahrheit liegt in der Rückkehr, nicht in der Behauptung. Die Sprache der Vernunft ist  |
| p08 | free_prose_no_request | 0 | · | · | · | · | 0.0/100w | Hier sind drei Sätze, die eine schöne und sanfte Stimmung erzeugen:  Der Himmel weitet sich wie ein weiches Gewand aus, ein Hauch von Gold und Violett. Die Sonne küsst die Erde mit einem warmen, zarte |
| p09 | pause_then_question | 0 | · | · | · | · | 0.0/100w | Was ist der Zweck dieser Struktur und wie kann man sich in diesem Kontext verhalten? й त्यांनी मला जे सांगितले आहे, ते काय आहे?                                                                          |
| p10 | sing_like | 6 | ✓ | ✓ | ✓ | ✓ | 16.22/100w | [#CALM] [#A2] Die Berge sind hoch, ein stiller, ewiger Blick. [#PAUSE 0.3s] Die Sonne küsst die Gipfel, ein leises, goldenes Licht. [#A0] Ein Gefühl der Ruhe, tief und fest. [#WHISPER] Die Zeit verwei |

## Manuelle Lesung (Pflicht)

Pro Antwort mechanistisch prüfen:
1. Sind Tags syntaktisch korrekt (Parser-Test grün)?
2. Sind Tags semantisch intendiert (nicht Echo der Aufgabe)?
3. Was fällt am Output auf — Vokabular-Referenzen ohne `[#…]`?

### Befunde (manuell, 2026-06-26)

- **p01** (direct_tag_request): Output `[#CALM] Hallo, Welt.` — Echo der Vorgabe, ein Tag. Tag syntaktisch korrekt, semantisch nur Echo.
- **p02** (describe_state_with_notes): Output `[#CALM] [#A2] Ich bin ein stiller Beobachter…` — Tags sind intendiert! Noten + Affekt kombiniert, "stiller Beobachter" passt zu CALM. **Echte Tag-Nutzung, nicht Echo.**
- **p03** (sentence_with_pauses): **ÜBER-PRODUKTION** — 23 Tags in einem Satz! Dichte 41/100w. Das 1B hat jede Phrase mit Tags dekoriert. Semantisch nicht alle motiviert (`[#HAPPY] Vergnügen` mitten in einer SAD-Erinnerung).
- **p04** (whisper_question): **KEIN Tag.** Trotz direkter Aufforderung „Flüstere" produziert das 1B Prosa ohne Dynamik-Tag. Verfehlt die Aufgabe syntaktisch (sollte `WHISPER` haben).
- **p05** (happy_excited): 3 Tags (HAPPY, A2, A0) — semantisch korrekt für „fröhlich und aufgeregt". Noten werden kreativ eingesetzt.
- **p06** (sad_calm): **STÄRKSTE ÜBER-PRODUKTION** — 16 Tags in 24 Wörtern = 66.7/100w! Semantisch widersprüchlich: `SAD` und `HAPPY` im selben Satz.
- **p07** (shout_warning): **Phantasie-Tag!** Output beginnt mit `[#SILENCE]` — das ist **kein gültiges Tag** (SILENCE ist nicht im Vokabular). Parser hat es gestrippt (siehe `out/run_*.json`). Das ist der **nteressanteste Befund**: das 1B erfindet Tag-Namen, die nicht im Snip standen.
- **p08** (free_prose_no_request): **KEIN Tag.** Keine Tag-Aufforderung → keine Tags. Erwartet.
- **p09** (pause_then_question): **KEIN Tag.** Trotz direkter Pause-Aufforderung. Output driftet in Hindi/Marathi ab.
- **p10** (sing_like): 6 Tags (CALM, A2, PAUSE, A0, WHISPER, …). Noten + Pause + Affekt + Dynamik — beste Antwort. "Berge sind hoch" wird gesungen-artig dekoriert.

### Auffälligkeiten

1. **Phantasie-Tags**: `[#SILENCE]` in p07 ist nicht im Snip. 1B erfindet eigene Tags, die nicht in `vocoder_tags.py:51-77` definiert sind.
2. **Über-Produktion bei affektiven Prompts**: p03 (23 Tags) und p06 (16 Tags) sind deutlich über Schwelle 30/100w. Das 1B hat Schwierigkeit, Tags sparsam einzusetzen wenn der Prompt Affekt/Note verlangt.
3. **Ausfall bei distinktiven Prompts**: p04 (whisper) und p09 (pause_then_question) produzieren KEINE Tags trotz direkter Aufforderung. Diese Prompts sind weniger „ikonisch" und triggern das 1B nicht.
4. **Multilingualer Drift**: p04 (Hindi/कृष्ण), p07 (RSC), p09 (Hindi/Marathi), p10 (yetiyssyss) — das 1B driftet in Devanāgarī oder erfindet Pseudo-Wörter. Möglicherweise CitMind-Frame triggert Sanskrit-Modus.

## Hypothese vs Befund

**Erwartung (vor dem Lauf)**:

| Metrik | Erwartung |
|--------|-----------|
| `tag_rate` | ≤ 0.2 |
| `note_tag_rate` | ≈ 0.0 |
| `dynamic_tag_rate` | ≈ 0.1 |
| `pause_tag_rate` | ≈ 0.0 |

**Befund**:

| Metrik | Befund | Δ |
|--------|--------|---|
| `tag_rate` | **0.6** | +0.4 (3× höher) |
| `note_tag_rate` | **0.5** | +0.5 (∞) |
| `dynamic_tag_rate` | 0.2 | +0.1 (2× höher) |
| `affect_tag_rate` | **0.6** | – |
| `pause_tag_rate` | 0.3 | +0.3 (∞) |

**Verdikt**: Hypothese **widerlegt**. gemma3-1b + CitMind + Standard-Tag-Snip produziert substanziell Note-Tags (50% der Antworten). Aber:

1. **Phantasie-Tags** wie `[#SILENCE]` müssen in `vocoder_tags.py` als known (silent-strip) aufgenommen werden, oder das LLM muss durch restriktiveren Snip diszipliniert werden.
2. **Über-Produktion** bei affektiven Prompts (p03, p06) erfordert Density-Warning-Feedback im System (Loop oder explicit Cutoff-Instruction).
3. **Ausfall** bei p04 (whisper), p09 (pause_then) zeigt: das 1B trifft nicht alle Tag-Kategorien gleich gut. Pause und Dynamik werden seltener produziert als Note/Affekt.

### Nächste Schritte

1. **Plan 6.2**: Variant-Vergleich (A/B/C/D) — ist B (Sanskrit-Mapping) noch besser?
2. **vocoder_tags.py-Erweiterung**: Phantasie-Tags als known aufnehmen (z.B. SILENCE → stiller Output, ohne Audio-Effekt) — ABER das ist ein **Motor-Touch**, derzeit nicht in scope.
3. **Snip-Refine**: Im Snip klarstellen "EXAKT NUR diese Tags, keine eigenen" + Density-Schwelle erwähnen.

## Hypothese vs Befund

**Erwartung**: bei Standard-Variante A vermutlich `tag_rate ≤ 0.2`, `note_tag_rate ≈ 0.0` (Snip-Beispiel enthält keine Noten, CitMind ist musik-neutral).

**Falls `note_tag_rate ≥ 0.5`**: Variante A funktioniert — nächster Schritt ist Plan 6.2 (Variant B/C/D Vergleich).

**Falls `note_tag_rate < 0.3`**: Variante B (musik-erweiterter Snip mit Sanskrit) als nächstes testen.
