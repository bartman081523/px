# Mechanik-Test: Klasse-C Prompt-Hypothese — Verdict

Anzahl Cells: 20
Bedingungen: ['BASELINE', 'LEAN']

## Pro-Bedingung Aggregate

| Bedingung | n | mean WIDE% | mean NARROW% | mean cos(h_L16, d_width) |
|---|---|---|---|---|
| BASELINE | 10 | 5.9% | 7.4% | +0.8885 |
| LEAN | 10 | 8.6% | 1.5% | +0.8956 |

## Per-Prompt WIDE-Klassen-Anteil (nach Bedingung)

| Prompt | BASELINE | LEAN |
|---|---|---|
| p01 | 31.7% | 17.6% |
| p02 | 2.5% | 11.1% |
| p03 | 8.5% | 6.0% |
| p04 | 1.0% | 6.0% |
| p05 | 8.0% | 19.1% |
| p06 | 0.0% | 0.5% |
| p07 | 3.0% | 7.5% |
| p08 | 1.5% | 8.0% |
| p09 | 2.5% | 5.3% |
| p10 | 0.5% | 5.0% |

## Cosinus(h_L, d_width) Trajektorie (Mittelwert pro Bedingung)

| Bedingung | L13 | L16 | L19 | L21 |
|---|---|---|---|---|
| BASELINE | +0.8975 | +0.8885 | +0.8611 | +0.8580 |
| LEAN | +0.9029 | +0.8956 | +0.8718 | +0.8680 |

## Falsifikator-Prüfung (MechanisticSubjectivityMixMind v1.0)

### O2: d_width rein recur-induziert (BASELINE WIDE% < 0.55)
- Gemessen: BASELINE mean WIDE% = 5.9%
- Status: **BESTÄTIGT (Hypothese gefährdet)**
- → Prompt-only reicht NICHT; d_width ist recur-induziert

### O3: Papagei-Test (cosinus zu d_width ≠ 0)
- Gemessen: BASELINE mean cos(h_L16, d_width) = +0.8885
- Status: **WIDERLEGT (echte Kopplung)**
- → Kopplung an d_width-Richtung ist messbar (positiv oder negativ)

## Synthese

**HYPOTHESE WIDERLEGT** (O2): BASELINE WIDE% = 5.9% < 55%.
d_width ist recur-induziert. Prompt-only reicht nicht. RELAY-Hook bleibt notwendig für die L21-Injection.

**Manuelle Lesung** der Texte (siehe Roh-Outputs in HIDDEN_OUT/*.pt oder Index in `mechtest_index.jsonl`) ergänzt die mechanische Aussage. 是X即非X-Wache: weder Krönung (auch bei hohem WIDE%) noch Weglesen (auch bei niedrigem).
