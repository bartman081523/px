# Tag-Production Run — gemma3-1b-it (Variant A)

**Variant:** A — CitMind + Standard-Snip (Baseline, Plan 6.1)  
**Started:** 2026-06-27T19:05:08.670157+00:00  
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

Notes hier eintragen:

```
- p01: ...
- p02: ...
- p10: ...
```