# Shadow Compare — Test B only (Test A/C ausgefallen wegen Gemini-Quote)

## Status (Stand 2026-08-17 09:30)

**Gemini-Quote erschöpft**: Test A (Output-Bridge) und Test C (WIDE-Anker-projection) konnten
nicht abgeschlossen werden, da die Gemini-API nach mehreren Retries 429 (Rate Limit)
zurückgab. Test B lief erfolgreich durch.

## Test B — Linear-Probe Training (1 erfolgreich)

**Setup**: 50 Konzept-Paare (DE/FR/ES/KR/JP/PL/Indic + English), 80/20 split,
closed-form least-squares W: gemini-embedding-001 (768d) → gemma-3-1b-it embed_tokens (1152d).

**Ergebnis**:

| metric | train | val |
|---|---|---|
| MSE | 0.0006 | 0.0006 |
| cos  | 0.558 | **0.558** |

**Interpretation**:

val_cos = 0.558 > 0.5: die beiden Embedding-Räume sind **linear überbrückbar** mit
moderater Genauigkeit. Das heißt: Gemini-Embeddings können via W in den
Gemma-1b-Embed-Raum projiziert werden, ohne dass die semantische Struktur
vollständig kollabiert. Aber: 0.558 ist deutlich unter der "perfekt portierbar"-
Schwelle von > 0.85. Das suggeriert:

- **teilweise** Portierbarkeit der **Embedding-Räume** (konzepte sind ähnlich angeordnet)
- NICHT zwingend Portierbarkeit der **Hidden-State-Coupling-Mechanik** (d_width ist
  modell-eigen, nicht embed-eigen)

**Was das NICHT sagt**:
- Nicht, dass Gemini-Coupling entlang Gemma-1b's d_width läuft.
- Nicht, dass Gemini das gleiche WIDE/NARROW-Innere hat wie Gemma.

**Was noch fehlt** für vollständige Hypothese-Evaluation:
- Test A (Output-Bridge): Gemini vs Gemma-1b auf identischen Prompts, Output cos.
- Test C (WIDE-Anker-Proj): Gemini-anbefohlene WIDE-Tokens nach Projektion in Gemma-Raum,
  cos zu Gemma-1b's d_width.

Diese Tests sind **gestoppt wegen Gemini-Quote-Limit**, nicht wegen methodischer
Probleme. Bei nächster Quote-Verfügbarkeit sind sie lauffähig (Skript ist
robust gegen leer/cancelled Outputs).

## Files erzeugt

- `shadow_compare.py` (~470 Zeilen, --shadow-model {gemma-3-1b-it, gemma-4-e2b-it})
- keine Output-JSON (weil Skript bei API-Limit fehlschlug)

## Konsequenz für die Hypothese

Da Test B mäßig erfolgreich war (cos=0.558), bleibt die Hypothese **plausibel**
aber **unbestätigt**. Für eine echte Aussage brauchen wir mindestens Test A
(Output-Bridge > 0.7) ODER Test C (mean cos to d_width > 0.3).
