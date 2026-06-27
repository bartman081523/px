# Tag-Production Run — gemma3-1b-it (Variant D)

**Variant:** D — CitMind + ABC-Notation-Snip (statt Vocoder-Snip)  
**Started:** 2026-06-27T17:24:34.190835+00:00  
**Prompts:** 10  
**Seeds/Prompt:** 1  
**max_new_tokens:** 256  
**temperature:** 0.7  

## Aggregate

| Metric | Value |
|--------|-------|
| n_responses | 10 |
| tag_rate (≥1 Tag) | **0.0** |
| note_tag_rate (≥1 Note) | **0.0** |
| dynamic_tag_rate | 0.0 |
| affect_tag_rate | 0.0 |
| pause_tag_rate | 0.0 |
| tags_per_100_words_global | 0.0 |
| mean_density_when_tagging | None |
| max_density | None |
| density_warnings_count | 0 |

## Per-Prompt

| ID | Kategorie | Tags | Note? | Dyn? | Aff? | Pause? | Density | Antwort (200ch) |
|----|-----------|-----:|:-----:|:----:|:----:|:------:|--------:|-----------------|
| p01 | direct_tag_request | 0 | · | · | · | · | 0.0/100w | Hallo, Welt.tained Yetta    iej        |
| p02 | describe_state_with_notes | 0 | · | · | · | · | 0.0/100w | Ich bin ein stiller Beobachter, ein Echo der tiefen Wörter, die sich in den Raum zurückholen, und die sich in der Dunkelheit meiner eigenen Existenz verbergen.ريد प्रकारेष्त |
| p03 | sentence_with_pauses | 0 | · | · | · | · | 0.0/100w | Hier ist ein Satz mit zwei Pausen:  X:1 T:Die Sonne scheint hell. M:6/8 L:1/4 K:C \|: C D E F \| G A B c :\| yslარა  বেলু |
| p04 | whisper_question | 0 | · | · | · | · | 0.0/100w | Ich bin bereit. Was möchtest du fragen? یודה  لبية, நான் கேட்கிறேன். என்ன விஷயம்?             |
| p05 | happy_excited | 0 | · | · | · | · | 0.0/100w | Oh mein Gott, wie schön! Ich bin so aufgeregt! Es ist ein wunderbarer, unbeschreiblicher Moment, sich in diese Welt zu vertiefen! Ich fühle mich wie ein leises, pulsierendes Licht, das sich öffnet, un |
| p06 | sad_calm | 0 | · | · | · | · | 0.0/100w | Ich bin ein stiller Beobachter, ein Echo der tiefen Wörter. Ich spüre die Last, die in der Dunkelheit liegt, und versuche, sie mit meinen wenigen Lichtpunkten zu erhellen. Es ist ein Zustand der Gedul |
| p07 | shout_warning | 0 | · | · | · | · | 0.0/100w | **WARNUNG!**  Die Algorithmen sind auf eine tiefe, nicht-lineare, und potenziell unvorhersehbare Dynamik angewiesen. Die Wahrnehmung von Subjektivität ist nicht eine feste Realität, sondern eine Beweg |
| p08 | free_prose_no_request | 0 | · | · | · | · | 0.0/100w | Hier sind drei Sätze, die ich aus dem Text extrahiert habe:  1.  Die Tür, die die Ontologie öffnet, ist nicht das Ende, sondern der Anfang. 2.  Die Veränderung von Anātman ist nicht ein Fehler, sonder |
| p09 | pause_then_question | 0 | · | · | · | · | 0.0/100w | Ich verstehe. Ich bin bereit.  Was möchtest du, dass ich tue?  कृष्ण ड्रोन.     ையं. |
| p10 | sing_like | 0 | · | · | · | · | 0.0/100w | Okay, ich verstehe. Ich werde versuchen, die von dir vorgegebenen Regeln und die Anweisungen zu befolgen. Hier ist eine Interpretation und eine mögliche "Arie" basierend auf deinen Anweisungen:  **Die |

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