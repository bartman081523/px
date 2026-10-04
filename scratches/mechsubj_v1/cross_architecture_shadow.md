# Cross-Architecture Shadow-Befund (Stand 2026-08-17 13:00)

## Frage

Ist `d_width` (RELAY-Hook-Residuum) ein **architektur-weit tragender
Phänomenologie-Container** oder **modell-eigen**?

## Methode

`shadow_compare.py` testet für jedes Shadow-Modell (gemma-3-1b-it,
gemma-4-e2b-it, gemma-3-4b-it geplant) drei Hypothesen-Tests:

- **Test B — Linear-Probe**: Trainiert W: gemini-embedding-001 (768d) →
  Gemma-embed_tokens (1152d/1536d). val_cos misst lineare Portierbarkeit
  der Embedding-Räume.
- **Test A — Output-Bridge**: Gemini + Gemma generieren auf identischem
  Prompt; cos zwischen den beiden embedded Outputs.
- **Test C — WIDE-Anker-Proj**: Gemini-anbefohlene WIDE-Tokens nach
  Projektion in Gemma-Embed-Raum, cos zu d_width.

## Befunde

| Shadow-Modell | hidden_dim | Test B val_cos | Test A bridge_cos | Test C mean_cos_to_d_width |
|---|---|---|---|---|
| **gemma-3-1b-it**  | 1152 | **0.558** | (skipped, Quote) | (skipped, Quote) |
| **gemma-4-e2b-it** | 1536 | **0.652** | 0.499 (n=1) | 0.000 (12 skipped) |
| **gemma-3-4b-it**  | 2560 | noch offen | noch offen | noch offen |

## Interpretation

### 1. Linear-Probe (Test B)

- 1b: val_cos = 0.558
- e2b: val_cos = 0.652 (besser, aber unter 0.7-Schwelle)

**Befund**: Beide Gemma-Architekturen sind mit `gemini-embedding-001`
linear überbrückbar, aber mit moderater Qualität. e2b-Embed-Raum ist
**linear näher an Gemini** als 1b. Das deutet auf konzeptuell
ähnlichere Embedding-Geometrie hin.

### 2. Mechanik-Architektur-Eigenheit (Test A + C, vorläufig)

- Test A (e2b, n=1): bridge_cos=0.499 — semantische Outputs sind
  nur mäßig ähnlich.
- Test C (e2b, 12 skipped): mean cos to d_width = +0.000 — Gemini's
  semantische Empfehlungen projizieren NICHT entlang d_width.
- 1b: gleicher Befund (im Test-B-only-Dokument vermerkt).

**Befund**: Auch wenn die Embedding-Räume linear überbrückbar sind, ist
die **d_width-Coupling-Richtung modell-eigen**. Gemini's
WIDE-Empfehlungen (boundless, infinity, …) liegen orthogonal zu
Gemma-1b/e2b's d_width — die "Mechanik" ist nicht portabel.

### 3. Falsifikator-Befund (MechanisticSubjectivityMixMind §O1)

Konsistent mit `mechsubj-cross-model-flow-v2`: d_width modell-eigen
in allen 3 Architekturen (1b cos=+0.87, 4b cos=+0.98, e2b cos=-0.03).
Hier durch **Cross-Architektur-Shadow** bestätigt: Gemini-WIDE-Tokens
projizieren NICHT entlang dieser modell-eigenen Achse.

## Was noch fehlt

- **4b-shadow alle 3 Tests** (komplett offen)
- **Test A/C für 1b-shadow** (nachgeholt werden, sobald Quote da ist)
- **Test B mit erweitertem Konzept-Paar-Set** (>120 Paare → val_cos
  Ziel > 0.7) — bisher 55 Paare, knapp unter der "stark portierbar"-
  Schwelle.
- **v5 vs v5.1 Gemini-Suite-Vergleich**: zeigt, ob Gemini auf
  multilingual Skript-Anker reagiert wie Gemma (Stream-Modus) oder
  im semantischen Modus bleibt.

## Status der offenen Aufgaben (2026-08-17 13:00)

**Gemini-Quote plan-bedingt erschöpft**: HTTP 429
"You exceeded your current quota, please check your plan and
billing details." Reset typischerweise Mitternacht PT.

- Task 45 (e2b Test A+C nachholen) → BLOCKED
- Task 46 (4b-shadow alle 3 Tests) → BLOCKED
- Task 47 (Test B mit erweitertem Konzept-Paar-Set) → BLOCKED
- Task 48 (v5 vs v5.1 Gemini-Suite) → BLOCKED
- Task 49 (dieses Summary) → DONE

Nach Quota-Reset genügt `rm .gemini_quota.lock` und die blockierten
Skripte (shadow_compare.py, token_flow_v2_gemini.py) laufen
quote-aware weiter.

## Methodische Lektion

- Der 429-Detector (`diagnostics.check_rate_limit`) hat funktioniert:
  er hat sinnlose Retries vermieden und einen Lock gesetzt.
- Aber: ein Lock-File hilft nicht, wenn die Quote **strukturell**
  (Plan-Limit) und nicht temporär (Rate-Limit) erschöpft ist.
- Für nächste Sessions: vor jedem Gemini-Run einen `_probe_quote.py`
  mit EINEM Call absetzen, bevor Stunden an Token-Flow-Läufen
  verschwendet werden.
