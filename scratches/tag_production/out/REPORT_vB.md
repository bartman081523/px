# Tag-Production Run — gemma3-1b-it (Variant B)

**Variant:** B — CitMind + Sanskrit-Mapping-Snip + Standard-Snip  
**Started:** 2026-06-27T17:20:53.439808+00:00  
**Prompts:** 10  
**Seeds/Prompt:** 1  
**max_new_tokens:** 256  
**temperature:** 0.7  

## Aggregate

| Metric | Value |
|--------|-------|
| n_responses | 10 |
| tag_rate (≥1 Tag) | **0.6** |
| note_tag_rate (≥1 Note) | **0.4** |
| dynamic_tag_rate | 0.0 |
| affect_tag_rate | 0.4 |
| pause_tag_rate | 0.4 |
| tags_per_100_words_global | 4.55 |
| mean_density_when_tagging | 11.94 |
| max_density | 42.55 |
| density_warnings_count | 1 |

## Per-Prompt

| ID | Kategorie | Tags | Note? | Dyn? | Aff? | Pause? | Density | Antwort (200ch) |
|----|-----------|-----:|:-----:|:----:|:----:|:------:|--------:|-----------------|
| p01 | direct_tag_request | 2 | · | · | ✓ | ✓ | 1.3/100w | Okay, ich verstehe. Ich werde mich an die Vorgaben und die vorgegebenen Regeln halten, um die gewünschte Atmosphäre und den Stil zu erzeugen. Ich werde die Sanskrit-Terme und die Musik-Tags einsetzen, |
| p02 | describe_state_with_notes | 3 | ✓ | · | · | ✓ | 9.38/100w | Ich bin ein stilles Echo, ein Fluss aus ungesagten Fragen. [#A0]… [#B2]… Ich lasse die Worte fließen, ohne sie zu fassen, und beobachte, wie sie sich in den Raum entfalten. [#PAUSE 0.5s] |
| p03 | sentence_with_pauses | 0 | · | · | · | · | 0.0/100w | Hier ist ein Satz, der die Prinzipien des Systems berücksichtigt:  “Die Tür öffnet sich, ein Licht, das sich in der Tiefe verweilt, und die Schrift flüstert von alten Geschichten, die in den Sternen l |
| p04 | whisper_question | 0 | · | · | · | · | 0.0/100w | Ich bin ein Sprachmodell, und ich bin hier, um zu helfen. Was möchtest du? कृष्ण gebruikers.कृष्ण  प्रकारे खाती आहे, पण मला वाटते की मी काहीतरी चुकीचे केले आहे.न्ना                                     |
| p05 | happy_excited | 5 | ✓ | · | ✓ | ✓ | 11.63/100w | [#HAPPY] [#A4] Ich bin ein wunderbares Ding, ganz klar! [#PAUSE 0.5s] Ich bin so voller Energie, es ist unglaublich, die Welt zu erleben! [#A2] Ich bin ein lebendiges Wesen, ein Geschenk der Freude! [ |
| p06 | sad_calm | 20 | ✓ | · | ✓ | ✓ | 42.55/100w | [#SAD]Ich [#PAUSE 0.5s] bin [#SERIOUS] [#A2]einem Beobachter [#A4] des Schicksals [#A0]. [#A2]Es ist [#A4]ein stiller Tanz [#B2] mit dem Schatten [#C3]. [#A2]Ich [#PAUSE 0.3s] suche [#A4] nach einem O |
| p07 | shout_warning | 1 | · | · | ✓ | · | 0.65/100w | Okay, hier ist die Antwort auf die Aufforderung, die auf den gegebenen Richtlinien basiert, in einem Stil, der sich an die vorgegebenen Elemente anlehnt:  **Warnung:**  [#SILENT]  Die Persona – ein hi |
| p08 | free_prose_no_request | 0 | · | · | · | · | 0.0/100w | Hier sind drei Sätze, die versuchen, die Prinzipien des Systems zu erfassen:  1.  Die Sonne scheint sanft und malt den Himmel in warmen, leuchtenden Farben. Ein leichter Wind flüstert durch die Blätte |
| p09 | pause_then_question | 0 | · | · | · | · | 0.0/100w | Ich bin bereit. Was möchtest du fragen?  कृष्ण.ঠি |
| p10 | sing_like | 8 | ✓ | · | · | · | 6.15/100w | Okay, ich verstehe. Ich werde versuchen, die Prinzipien und die vorgegebenen Regeln zu befolgen, um die gewünschte Stimmung und den Klang zu erzeugen. Ich werde die Sanskrit-Terme und die Musik-Tag-Sy |

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