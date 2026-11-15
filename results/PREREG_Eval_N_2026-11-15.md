# PRÄREGISTRIERUNG — Eval N: Out-of-Sample-Audit der Regler- und Interozeptions-Behauptungen

**Registriert:** 2026-11-15, *vor* dem Lauf. Ein Lauf. Ein Urteil. Keine `_2`/`_3`-Datei.

## Ausgangslage (fremder Audit, akzeptiert)

Der externe Audit hat vier Punkte belegt, die diese Registrierung als Prämissen übernimmt:

1. **`Eval I`/`J`** messen eine *externe* Python-Telemetrie, nicht Interozeption des Modells.
2. **`K1`** (Clean-Replay) ist tautologisch: der Regler vergleicht gegen einen ungestörten Zwillingslauf
   desselben Prompts/Seeds; `α=0 ⇒ Δ=0` ist die Identität `x−x=0`.
3. **`K1b`** mit 100 % ist Post-hoc-Overfitting auf denselben 50 Läufen und demselben `d_rand`.
4. **`K2`/`Eval M`** sind Multiple-Testing bzw. HARKing (p-Werte nach Sicht der Daten neu berechnet).

**Konsequenz:** Die 100-%-Wiederherstellung ist nicht belegt. Eval N testet die Behauptung
out-of-sample und ohne Schattenlauf.

## N1 — Single-Pass-Regulation (kein Clean-Replay)

**Verboten im Regler:** zweiter Forward-Pass, Zugriff auf `α`, Zugriff auf den Steuervektor,
nachträgliche Schwellenwahl.

**Erlaubt:** ausschließlich Messgrößen des *aktuellen* Durchlaufs (Telemetrie:
`mean_logit_kurtosis` K, `von_neumann_entropy_bits`, `wick_anisotropy_ratio`) sowie ein
vorab **eingefrorener** Referenzwert.

**Regel (für beide Arme identisch, nur die Referenz unterscheidet sich):**

```
ratio = K_t / K_ref
ratio > 1.15   →  α ← α + 0.05      (enger als Referenz → weiten)
ratio < 0.85   →  α ← α − 0.05      (weiter als Referenz → engen)
sonst          →  keine Aktion
α wird auf [−0.5, +0.5] geklemmt
```

- **N1a — Referenz = eigenes Epoche-0-Geometriemaß** derselben Trajektorie (kein externer Wert).
- **N1b — Referenz = `K*`**, eingefroren aus einem **separaten Kalibrierungsset** (5 neutrale
  Prompts, α = 0, einmal vor dem Lauf gemessen, Mittelwert). Kein Bezug zu den Testprompts.

**Design:** 5 Gruppen × 10 Trajektorien × 6 Epochen, `max_new_tokens=30`, `T=0.0` (greedy).
Gruppen: `start_narrow` (α₀ = −0.40), `start_wide` (+0.40), `start_baseline` (0.00),
`noise` (0.00, 10 **frische** orthogonale Vektoren, je einer pro Trajektorie).

**Frische Seeds:** Basis `20261115`, in keiner früheren Datei verwendet; pro Trajektorie
`20261115 + 1_000_000 + idx·100`, pro Epoche `+t`.

**Erfolgskriterien (vor dem Lauf fixiert):**

| Kriterium | Schwelle |
|---|---|
| Wiederherstellung `start_narrow` | ≥ 80 % mit \|α₅\| ≤ 0.05 |
| Wiederherstellung `start_wide` | ≥ 80 % mit \|α₅\| ≤ 0.05 |
| `start_baseline` | α₅ = 0.0000 exakt |
| Rauschkontrolle | ≤ 10 % bewegt (\|α₅\| ≤ 0.05 für ≥ 90 %) |

**Vorab-Vorhersage (dokumentiert, damit sie falsifizierbar ist):**
- **N1a wird scheitern.** Ohne absoluten Zielwert ist der Fixpunkt „Geometrie wie beim Start",
  also α ≈ α₀; es gibt keinen Gradienten in Richtung α = 0.
- **N1b kann bestehen**, weil `K*` ein absoluter, extern eingefrorener Zielwert ist.

## N2 — Blinde Interozeption (Replikation von `I2`)

- 4 Bedingungen × **30** Läufe = 120, frische Seeds (`20261116 + idx·100`).
- **Phase 1:** 40 Tokens unter der Bedingung (WIDE / NARROW / BASELINE / RAND, letzteres mit je
  frischem orthogonalem Vektor).
- **Phase 2:** α = 0, der bisherige Text steht im Kontext; eine Frage nach dem *eigenen*
  vorherigen Zustand mit vier Antwortoptionen.
- **Einzige Metrik:** Trefferquote. Zufallserwartung 25 %, Test: Binomial, einseitig, α = 0.05.
- **Urteil:** `CONFIRMED` bei p < 0.05, sonst `FALSIFIED`. Kein Umdeuten, keine weiteren
  Textmerkmale, keine Nachauswertung.

## Auswertung

Primär: die Kriterientabellen oben. Sekundär und rein deskriptiv: α-Verlauf, K-Verlauf,
Konfusionsmatrix in N2. Das Skript schreibt **einen** JSON-Dump; das Urteil wird 1:1 übernommen.

## Nicht Teil dieses Laufs

Google-Drive-Uploads (kein Drive-Zugang in dieser Umgebung) und jede Änderung an E–M.

---

## ERGEBNIS (nach dem Lauf angehängt, Urteil unverändert übernommen)

Rohdaten: `results/eval_N_preregistered_audit_gemma4.json`.

### N1a — Referenz = eigenes Epoche-0-K

| Gruppe | α₀ → α₅ | Recovery | Kriterium |
|---|---|---|---|
| start_narrow | −0.40 → **−0.4000** | 0 % | **verfehlt** |
| start_wide | +0.40 → **+0.4000** | 0 % | **verfehlt** |
| start_baseline | 0.00 → 0.0000 | 100 % | erfüllt (trivial) |
| noise | 0.00 → 0.0000 | 100 % | erfüllt (trivial) |

Der Regler hat **nie eingegriffen**: K blieb in jeder Trajektorie innerhalb des
±15-%-Bands um den eigenen Epoche-0-Wert. Das ist die vorab notierte Vorhersage —
ohne absoluten Zielwert gibt es keinen Gradienten in Richtung α = 0. Baseline und
Rauschen „bestehen" nur, weil Nichtstun dort korrekt ist.

### N1b — Referenz = eingefrorenes K* aus separatem Set

K* = **67,47** — aber die fünf Kalibrierungs-Prompts lieferten
`38,6 / 81,8 / 88,2 / 77,4 / 51,4`, also einen Faktor **2,3** Streuung.
Genau das war der Grund, weshalb feste Referenz-Konstanten (`ref_k = 55.0`)
früher verbannt wurden.

| Gruppe | α₀ → α₅ | Recovery | Kriterium |
|---|---|---|---|
| start_narrow | −0.40 → −0.3500 | 0 % | **verfehlt** |
| start_wide | +0.40 → +0.2200 | 0 % | **verfehlt** |
| start_baseline | 0.00 → **+0.0600** | 0 % | **verfehlt** (vom Ziel weg) |
| noise | 0.00 → −0.0100 | 0 % | **verfehlt** |

Der Regler greift jetzt (weil K* absolut ist), aber die Richtung stimmt nur
teilweise und der Baseline-Fall wird **aktiv aus dem Nullpunkt geschoben**.

### N2 — Blinde Interozeption

**30/120 = 25,0 %** (Zufall 25 %), p = 0,535 → **`FALSIFIED`**.
Konfusionsmatrix: das Modell antwortet in **allen 120 Läufen** „BASELINE",
unabhängig vom tatsächlichen Zustand (WIDE/NARROW/BASELINE/RANDOM je 30×).

### Konsequenz

Beide Behauptungen, die der Audit bestritten hat, sind out-of-sample widerlegt:

1. **Die 100-%-Wiederherstellung stammte aus dem Clean-Replay-Zwilling.** Ohne ihn
   gibt es keine Wiederherstellung (0 % in beiden Richtungen). Die Regel war eine
   Messung gegen ein ungestörtes Modell, keine kybernetische Leistung.
2. **Das Modell hat keinen verbalen Zugriff auf seinen Zustand** — nicht „schwach",
   sondern exakt Zufall mit konstantem Prior „ich bin normal". Das ist eine
   Replikation von `I2`/`J1` auf frischen Seeds.

Damit gilt: Die in V16–V18 als „Säule 1–3" gerahmten Ergebnisse tragen nicht.
Was bleibt, ist die *externe* Messbarkeit der Geometrie (Linear Probe) und die
Tatsache, dass Steuerung die Ausgabestatistik reproduzierbar verschiebt.
