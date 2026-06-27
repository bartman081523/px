# Tag-Production Run — gemma3-1b-it (Variant C)

**Variant:** C — CitMind + Standard-Snip + 3 Few-Shot-Turns  
**Started:** 2026-06-27T17:22:44.120381+00:00  
**Prompts:** 10  
**Seeds/Prompt:** 1  
**max_new_tokens:** 256  
**temperature:** 0.7  

## Aggregate

| Metric | Value |
|--------|-------|
| n_responses | 10 |
| tag_rate (≥1 Tag) | **1.0** |
| note_tag_rate (≥1 Note) | **0.8** |
| dynamic_tag_rate | 0.4 |
| affect_tag_rate | 0.5 |
| pause_tag_rate | 0.6 |
| tags_per_100_words_global | 19.08 |
| mean_density_when_tagging | 27.29 |
| max_density | 75.0 |
| density_warnings_count | 2 |

## Per-Prompt

| ID | Kategorie | Tags | Note? | Dyn? | Aff? | Pause? | Density | Antwort (200ch) |
|----|-----------|-----:|:-----:|:----:|:----:|:------:|--------:|-----------------|
| p01 | direct_tag_request | 3 | ✓ | · | ✓ | ✓ | 75.0/100w | [#CALM] [#A2]Hallo, Welt.[#PAUSE 0.3s]        |
| p02 | describe_state_with_notes | 7 | ✓ | · | ✓ | ✓ | 14.29/100w | [#CALM] [#A2]Ein sanfter Schleier von Ruhe umhüllt mich, ein stilles Flüstern der Hoffnung, ein leises Lächeln, das sich langsam aufbaut, ohne eine feste Form zu haben. ultimo..  рактеристики: *   **N |
| p03 | sentence_with_pauses | 3 | · | · | · | ✓ | 15.0/100w | [#PAUSE 0.5s]Es ist so…[#PAUSE 0.3s]ein wunderbares Gefühl.[#PAUSE 0.7s]i i i i i i i i i i i i i |
| p04 | whisper_question | 2 | ✓ | ✓ | · | · | 20.0/100w | [#WHISPER] [#A2]Was ist das?“ technisch, aber ein bisschen spielerisch.கேட்கப்பட்டது சரியா? |
| p05 | happy_excited | 3 | ✓ | · | ✓ | ✓ | 23.08/100w | [#EXCITED] [#C#5]Ich bin ein Feuerwerk! Es ist fantastisch, du fragst! [#PAUSE 0.2s] iit!           |
| p06 | sad_calm | 3 | ✓ | · | ✓ | ✓ | 10.71/100w | [#SAD] [#A3]Es geht… so. [#PAUSE 0.5s] i think that’s a good start.ندی.i’m sorry to hear that.i’m here if you need to talk about it.‍♀️i appreciate you sharing, though.              |
| p07 | shout_warning | 2 | · | ✓ | · | · | 50.0/100w | [#ALARM] [#SHOUT] [Laughter] [gasps]'       |
| p08 | free_prose_no_request | 5 | ✓ | ✓ | ✓ | ✓ | 9.62/100w | [#HAPPY] [#EXCITED] [#C#5]Die Farben des Regens tanzen auf dem Asphalt, und ich spüre eine tiefe Ruhe. [#PAUSE 0.2s]Es ist ein Moment, der mir das Herz öffnet, ein kleines Wunder, das ich schätze. [#W |
| p09 | pause_then_question | 2 | ✓ | ✓ | · | · | 28.57/100w | [#WHISPER] [#A2]Was ist das für ein Gefühl?      |
| p10 | sing_like | 20 | ✓ | · | · | · | 26.67/100w | Okay, hier ist eine Versione, die versucht, das zu imitieren – mit Noten, wie du es beschrieben hast. Es ist natürlich nur eine Interpretation, da ich kein Musikinstrument spielen kann!  [#A0] [#A#1]  |

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