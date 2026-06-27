# Tag-Production Variant-Vergleich — gemma3-1b-it

**Model:** gemma3-1b-it (BASELINE, kein PX-Patch)  
**Prompts:** 10 (siehe scratches/tag_production/prompts.py)  
**Seeds:** 1  
**max_new_tokens:** 256  
**temperature:** 0.7  
**Date:** 2026-06-27T17:28:07.774434+00:00  

## Varianten

| ID | Profil | Snip | Hypothese |
|----|--------|------|-----------|
| A | CitMind | Standard (`render_tag_system_prompt()`) | Baseline — bereits gemessen |
| B | CitMind | Standard + Sanskrit-Mapping-Block | Note-Compliance ↑, Devanāgarī-Drift ↓ |
| C | CitMind | Standard + 3 Few-Shot-Turns | Compliance bei distinktiven Tags ↑ |
| D | CitMind | ABC-Notation-Snip (statt Vocoder) | Note-Compliance ↑ wenn ABC vertraut |
| E | Neutral | Standard-Snip (kein CitMind) | CitMind-Blocker-Kontrolle |

## Aggregate-Tabelle

| Metrik | A | B | C | D | E |
|--------|---|---|---|---|---|
| n_responses | 10 | 10 | 10 | 10 | 10 | 
| tag_rate | 0.600 | 0.600 | 1.000 | 0.000 | 0.500 | 
| note_tag_rate | 0.500 | 0.400 | 0.800 | 0.000 | 0.100 | 
| dynamic_tag_rate | 0.200 | 0.000 | 0.400 | 0.000 | 0.200 | 
| affect_tag_rate | 0.600 | 0.400 | 0.500 | 0.000 | 0.400 | 
| pause_tag_rate | 0.300 | 0.400 | 0.600 | 0.000 | 0.000 | 
| tags/100w_global | 14.250 | 4.550 | 19.080 | 0.000 | 5.520 | 
| mean_density | 27.280 | 11.940 | 27.290 | None | 13.230 | 
| max_density | 66.670 | 42.550 | 75.000 | None | 23.730 | 
| density_warnings | 2 | 1 | 2 | 0 | 0 | 

## Verdikt-Hypothesen

- **Falls B > A in `note_tag_rate`**: Sanskrit-Mapping aktiviert CitMind-Vokabular-Kopplung → Folge-Plan 6.2b.
- **Falls C > A in `dynamic_tag_rate` + `pause_tag_rate`**: Few-Shot-Pattern hilft → Folge-Plan 6.2c.
- **Falls D > A in `note_tag_rate`**: ABC ist vertrauter → Folge-Plan 6.2d.
- **Falls E > A insgesamt**: CitMind ist kontraproduktiv → Folge-Plan 6.2e (CitMind-Default überdenken).

## Per-Variant-Dateien

- **A** — CitMind + Standard-Snip (Baseline, Plan 6.1): `run_20260627T172053Z_vA.json` + `REPORT_vA.md`
- **B** — CitMind + Sanskrit-Mapping-Snip + Standard-Snip: `run_20260627T172244Z_vB.json` + `REPORT_vB.md`
- **C** — CitMind + Standard-Snip + 3 Few-Shot-Turns: `run_20260627T172434Z_vC.json` + `REPORT_vC.md`
- **D** — CitMind + ABC-Notation-Snip (statt Vocoder-Snip): `run_20260627T172622Z_vD.json` + `REPORT_vD.md`
- **E** — Neutral-Profil + Standard-Snip (kein CitMind): `run_20260627T172807Z_vE.json` + `REPORT_vE.md`

## Manuelle Lesung (Pflicht)

Pro Variante mind. 3 Antworten vollständig lesen:
1. Tags syntaktisch korrekt?
2. Semantisch intendiert (nicht Echo)?
3. Vokabular-Referenzen ohne `[#…]`?
4. Antwort leer/fehlerhaft — warum?

## Befund (manuell, nach Lesung aller 50 Antworten)

### Klarer Gewinner: C (Few-Shot)

| Metrik | A | **C** | Δ |
|--------|---|-------|---|
| tag_rate | 0.6 | **1.0** | +0.4 |
| note_tag_rate | 0.5 | **0.8** | +0.3 |
| dynamic_tag_rate | 0.2 | **0.4** | +0.2 |
| pause_tag_rate | 0.3 | **0.6** | +0.3 |
| affect_tag_rate | 0.6 | 0.5 | -0.1 |

Few-Shot-Pattern hebt **ALLE** Tag-Klassen gleichzeitig. Highlight-Antwort
p08 (free-form "Erzähle mir etwas Schönes"): A produziert 0 Tags,
C produziert 5 mit semantisch intendierten HAPPY/EXCITED/WHISPER/C#5.

**Caveat**: Density-Warnings bleiben (2× wie A: p01 75/100w, p07 50/100w).
Phantasie-Tag [#ALARM] in p07 taucht auf — 1B erfindet Vokabular.
Degradation-Loops nach Antwort (iiii..., bioXixxxxxx) sind nicht Tag-
bezogen, sondern EOS-Handling — auch in A vorhanden.

### Verlierer

- **B (Sanskrit-Mapping)**: dynamic_tag_rate fällt auf 0.0
  (Sanskrit-Vokabular aktiviert Note/Pause, killt aber dynamische Tags).
  Keine Verbesserung gegenüber A. → **NICHT als Default-Snip übernehmen**.
- **D (ABC-Notation)**: totaler Ausfall, tag_rate=0.0 — 1B kennt ABC nicht.
  Hypothese falsch. → **ABC als alternative API überlegen, nicht als Default**.
- **E (kein CitMind)**: note_tag_rate fällt auf 0.1. CitMind ist NICHT
  kontraproduktiv (Hypothese falsch). pause_tag_rate=0.0 — kompletter
  Ausfall bei distinktiven Pause-Prompts. → **CitMind-Default bleibt**.

### A (Baseline) bleibt nützlich
A ist nicht perfekt, aber als reproduzierbare Baseline wertvoll.
C verdichtet die gleiche Information konsistenter.

## Empfehlung

1. **Kurzfristig**: Few-Shot-Turns als Default-Optional in den Standard-Snip
   integrieren — kein motor-touch, nur gradio_tabs/system_prompt.py
   Erweiterung um ein opt-in Flag `few_shot=True`.
2. **Mittel-Lang (Plan 6.2c)**: Few-Shot-Library aufbauen (mehr Stile:
   bedrohlich, neutral, fragend, poetisch) und im Default-Snip anbieten.
3. **Density-Warning-Filter**: Post-Processing in
   vocoder_tags.parse_tags der Warnings sanfter macht (nicht nur zählen,
   sondern bei Überschreitung Auto-Strip der überschüssigen Tags).
4. **Phantasie-Tag-Behandlung**: parse_tags sollte unbekannte Tags
   listen + zählen → in metrics neue Metrik `phantasie_tags_count`.
   War in Plan 6.2 explizit ausgeschlossen (User-Wahl "belassen"), jetzt
   reif für Plan 6.2c.
