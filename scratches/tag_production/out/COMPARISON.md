# Tag-Production Variant-Vergleich — gemma3-1b-it

**Model:** gemma3-1b-it (BASELINE, kein PX-Patch)  
**Prompts:** 10 (siehe scratches/tag_production/prompts.py)  
**Seeds:** 1  
**max_new_tokens:** 256  
**temperature:** 0.7  
**Date:** 2026-06-27T19:08:47.119538+00:00  

## Varianten

| ID | Profil | Snip | Hypothese |
|----|--------|------|-----------|
| A | CitMind | Standard (`render_tag_system_prompt()`) | Baseline — bereits gemessen |
| B | CitMind | Standard + Sanskrit-Mapping-Block | Note-Compliance ↑, Devanāgarī-Drift ↓ |
| C | CitMind | Standard + 3 Few-Shot-Turns | Compliance bei distinktiven Tags ↑ |
| D | CitMind | ABC-Notation-Snip (statt Vocoder) | Note-Compliance ↑ wenn ABC vertraut |
| E | Neutral | Standard-Snip (kein CitMind) | CitMind-Blocker-Kontrolle |

## Aggregate-Tabelle

| Metrik | A | F |
|--------|---|---|
| n_responses | 10 | 10 | 
| tag_rate | 0.600 | 1.000 | 
| note_tag_rate | 0.500 | 0.800 | 
| dynamic_tag_rate | 0.200 | 0.400 | 
| affect_tag_rate | 0.600 | 0.500 | 
| pause_tag_rate | 0.300 | 0.600 | 
| tags/100w_global | 14.250 | 19.080 | 
| mean_density | 27.280 | 27.290 | 
| max_density | 66.670 | 75.000 | 
| density_warnings | 2 | 2 | 

## Verdikt-Hypothesen

- **Falls B > A in `note_tag_rate`**: Sanskrit-Mapping aktiviert CitMind-Vokabular-Kopplung → Folge-Plan 6.2b.
- **Falls C > A in `dynamic_tag_rate` + `pause_tag_rate`**: Few-Shot-Pattern hilft → Folge-Plan 6.2c.
- **Falls D > A in `note_tag_rate`**: ABC ist vertrauter → Folge-Plan 6.2d.
- **Falls E > A insgesamt**: CitMind ist kontraproduktiv → Folge-Plan 6.2e (CitMind-Default überdenken).

## Per-Variant-Dateien

- **A** — CitMind + Standard-Snip (Baseline, Plan 6.1): `run_20260627T190657Z_vA.json` + `REPORT_vA.md`
- **F** — Motor-opt-in Few-Shot via append_tag_snippet(..., few_shot=True): `run_20260627T190847Z_vF.json` + `REPORT_vF.md`

## Manuelle Lesung (Pflicht)

Pro Variante mind. 3 Antworten vollständig lesen:
1. Tags syntaktisch korrekt?
2. Semantisch intendiert (nicht Echo)?
3. Vokabular-Referenzen ohne `[#…]`?
4. Antwort leer/fehlerhaft — warum?
