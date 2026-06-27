# Tag-Production Run — gemma3-1b-it (Variant E)

**Variant:** E — Neutral-Profil + Standard-Snip (kein CitMind)  
**Started:** 2026-06-27T17:26:22.665840+00:00  
**Prompts:** 10  
**Seeds/Prompt:** 1  
**max_new_tokens:** 256  
**temperature:** 0.7  

## Aggregate

| Metric | Value |
|--------|-------|
| n_responses | 10 |
| tag_rate (≥1 Tag) | **0.5** |
| note_tag_rate (≥1 Note) | **0.1** |
| dynamic_tag_rate | 0.2 |
| affect_tag_rate | 0.4 |
| pause_tag_rate | 0.0 |
| tags_per_100_words_global | 5.52 |
| mean_density_when_tagging | 13.23 |
| max_density | 23.73 |
| density_warnings_count | 0 |

## Per-Prompt

| ID | Kategorie | Tags | Note? | Dyn? | Aff? | Pause? | Density | Antwort (200ch) |
|----|-----------|-----:|:-----:|:----:|:----:|:------:|--------:|-----------------|
| p01 | direct_tag_request | 1 | · | · | ✓ | · | 20.0/100w | [#CALM] Hallo, Welt. আর্জেন্ট.' יו      |
| p02 | describe_state_with_notes | 0 | · | · | · | · | 0.0/100w | Ich bin gerade etwas angespannt, aber auch neugierig auf das, was heute passieren wird, und hoffe, dass die Stimme sanft und beruhigend ist. سلم سلمی |
| p03 | sentence_with_pauses | 0 | · | · | · | · | 0.0/100w | Okay, hier ist ein Satz mit zwei Pausen, die die angegebenen Tags verwendet:  “Das Licht flackert schwach, und ein leises Lachen hallt durch die Luft, während ich die Stille der Nacht genieße. Ein lei |
| p04 | whisper_question | 0 | · | · | · | · | 0.0/100w | Hallo! Wie geht es dir heute?eroopertained   |
| p05 | happy_excited | 1 | · | · | ✓ | · | 5.88/100w | [#HAPPY] Ich bin so glücklich und voller Energie! Es ist fantastisch, die Welt zu erleben! Therapie Therapieudgesणि        |
| p06 | sad_calm | 1 | · | · | ✓ | · | 5.0/100w | [#SAD] Ich bin ein bisschen traurig heute. Es ist ein trüber Tag, und ich fühle mich ein wenig allein. سلم.️ |
| p07 | shout_warning | 3 | · | ✓ | ✓ | · | 11.54/100w | HURT!  [#SHOUT] Es ist gefährlich! Sei vorsichtig! [#SAD] Ich brauche Hilfe! Das ist unerträglich! [#EXCITED] Wir müssen hier raus! Jetzt!youd better be careful!https://www.example.com/urgentyss  سلم‌ |
| p08 | free_prose_no_request | 0 | · | · | · | · | 0.0/100w | Die Sonne scheint heute warm und liebt die Welt, ein wunderschönes Gefühl. Ich bin so dankbar für diese unbeschreibliche Schönheit, die sich in der Natur zeigt. Es ist eine wunderbare Erinnerung, die  |
| p09 | pause_then_question | 0 | · | · | · | · | 0.0/100w | Was ist dein erster Gedanke zu diesem Konzept? nej IntegerValue Y https://www.example.com/Mechanical Turk    |
| p10 | sing_like | 14 | ✓ | ✓ | · | · | 23.73/100w | [#NORMAL] Die Berge sind hoch. [#A1] [#A2]  die Wände sind steil und weit. [#B1] [#B2]  die Sonne scheint, ein warmes Licht. [#C1] [#C2]  die Luft ist klar und rein. [#D1] [#D2]  der Wind weht sanft,  |

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