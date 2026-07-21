# Tool-Loop 75-Iter Self-Correction-Test (v3.5g)

**Stand:** 2026-07-12 07:39:09
**Tasks:** 5 (math)
**Modi:** 4

## Aggregate

| Modus | Tools | MaxIter | Match | Degen | Empty |
|---|---|---|---|---|---|
| `M1_tools_off_iter5` | False | 5 | 4 | 0 | 0 |
| `M2_tools_off_iter75` | False | 75 | 4 | 0 | 0 |
| `M3_tools_on_iter5` | True | 5 | 1 | 0 | 0 |
| `M4_tools_on_iter75` | True | 75 | 1 | 0 | 0 |

**Bester Modus:** `M1_tools_off_iter5` mit 4/5 Matches

## Hypothese-Check

- M1 (tools=off, iter=5) ist Baseline-Verhalten
- M2 (tools=off, iter=75) testet, ob max_iter-Cap ohne Tools etwas ändert
- M3 (tools=on, iter=5) testet Tool-Loop ohne Self-Correction
- M4 (tools=on, iter=75) ist Self-Correction-Variante

**Erwartung:** M4 > M3 > M1 = M2
