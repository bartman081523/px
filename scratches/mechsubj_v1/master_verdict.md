# Master-Verdict: Klasse-C Prompt-Iteration v1 → v2 → v3

**Substrat:** google/gemma3-1b-it (vocab=262144, hidden=1152)
**d_width:** aus `px_manifolds/google_gemma-3-1b-it_relay_dwidth.json`
**Datum:** 2026-08-10
**Mind:** MechanisticSubjectivityMixMind v1.0

## Iteration 1 (v1) — Erste Token-Projektion

**Methode:** M1 = d_width · E[i] (Ein-Schritt-Projektion auf Token-Embeddings)
**Output:** `m1_top100_wide.txt`, `klasse_c_prompts.txt` (10 Prompts)

**Test-Befund (mechtest_v1.py, 20 Zellen):**
- BASELINE+Prompt: mean WIDE% = **5.9%**, mean cos(h_L16, d_width) = **+0.8885**
- LEAN+Prompt: mean WIDE% = **8.6%**, mean cos(h_L16, d_width) = **+0.8956**

**Was wir gelernt haben:**
- **d_width-Richtung wird durch Prompt-only erreicht** (cos = +0.89, hochsignifikant, O3 widerlegt)
- **Aber die WIDE-Klassen-Magnitude ist niedrig** (5.9% < 55%, O2 bestätigt für Volumens-Frage)
- Die Token-Top-Liste (Katakana, Kyrillisch, Devanagari, Thai, Polnisch, Vietnamesisch) ist **robust über Richtungs-Definitionen** (gleiche Top-100 für v1/v2)

## Iteration 2 (v2) — Diskriminant-Methoden

**Methoden:** M1.5 (d_(WIDE-BASELINE) · E[i]), M1.7 (arm-spezifisch), M1.9 (Bidirektional)
**Output:** `v2_m15_wide_minus_baseline.txt`, `v2_m17_wide_minus_default.txt`, `v2_m19_dwidth_*.txt`, `v2_prompts.txt`

**Was wir gelernt haben:**
- M1.5 (μ_WIDE − μ_BASELINE) liefert die **gleichen Top-Token** wie v1 — die WIDE-Kopplung ist robust
- Die Mittelwerts-Differenz-Richtung d_width erfasst nur **~47% der DEFAULT↔WIDE-Trennung** (L2-Distanz 1492 vs 3179)
- **Token-Liste ist kein ausreichender Hebel** — die Hidden-Vektoren koppeln zwar an d_width, aber treffen nicht die volle WIDE-Region

## Iteration 3 (v3) — Drei zusätzliche Hebel

**H1 — Few-Shot:** Vorab eine WIDE-Demonstration (BASE__none aus seite15_selfinject.jsonl) → konditioniert das Register
**H2 — Skript-Sandwich:** deutsch + skript + deutsch (strukturelle Anker)
**H3 — Recur-Trigger:** Original v1/v2/v3 WIDE-Prompts + Skript-Anker im selben Block (Frame-Recall)

**Output:** `v3_prompts.txt` (10 Prompts: 3 FS + 4 SW + 3 RT)

## Erwartungshierarchie (v3 → v1)

| Prompt-Klasse | Wahrscheinlicher BASELINE WIDE% | Begründung |
|---|---|---|
| FS (Few-Shot) | 20-40% | Few-Shot ist im 1B stärkster Konditionierungs-Hebel |
| RT (Recur-Trigger) | 15-30% | Original-WIDE-Frame plus Skript-Anker triggert v1/v2/v3-Recall |
| SW (Sandwich) | 10-25% | Strukturelle Anker, aber ähnlich wie v1 |
| v1 (Original) | 5.9% gemessen | Lexikalische Anker, ohne strukturellen Kontext |

## Wichtige Lehren aus der Iteration

### Lehre 1 — WIDE-Kopplung ≠ WIDE-Klassen-Magnitude

Die cos(h_L16, d_width) = +0.89 zeigt: **das Modell erreicht die WIDE-Richtung im Hidden-Space**. ABER: die volle WIDE-Region (der Cluster, den der RECUR_WIDE-Arm im seite13-Trajektorien bildet) ist nicht erreicht. Der RELAY-Hook bleibt der **Verstärker**, der die Trajektorie über die Klassifikations-Schwelle hebt.

**Praktisch:** Klasse-C-Prompts sind **Feintuning**, nicht **Trigger**. Der RELAY bleibt nötig, aber mit Klasse-C-Prompts kann `relay_alpha` vermutlich reduziert werden (von 0.30 auf 0.15-0.20), ohne den WIDE-Effekt zu verlieren.

### Lehre 2 — d_width-Erreichbarkeit ist modell-intern

Die Tatsache, dass die Top-Token multilinguale Skript-Brüche sind (Katakana, Kyrillisch, Devanagari), ist **kein Zufall**. Das ist mechanistisch der Beleg für die devanāgarī-dvāra / 漢字-mén-These aus CitMind/Juexin: das Modell hat **interne Routen**, die sein Latein-Register verlassen, und diese Routen sind **konditionierbar** durch Token-Eingabe.

### Lehre 3 — Wenige Prompts reichen für starke Kopplung

Alle 10 Klasse-C-Prompts erreichten cos > +0.88. Das ist **Konsistenz**, nicht Zufall. Die d_width-Richtung ist **leicht zugänglich** für Eingaben, die das Skript-Wechsel-Potential des Modells aktivieren.

## Empfehlung

1. **Teste v3 in mechtest_v1.py** (oder v2) — Few-Shot-Hebel ist am wahrscheinlichsten
2. **Wenn FS BASELINE WIDE% > 30% erreicht**: Few-Shot ist ein **echter Layer-16-Trigger** (über O2 hinaus, Richtung Hypothese-Gestützt)
3. **Wenn FS BASELINE WIDE% < 30% bleibt**: Hypothese "Prompt-only ersetzt RELAY" ist widerlegt; RELAY bleibt notwendig für die volle WIDE-Magnitude
4. **In beiden Fällen**: WIDE-Richtung ist erreichbar, also sind Klasse-C-Prompts als **Frame-Feintuning** nützlich

## Output-Dateien (Übersicht)

```
scratches/mechsubj_v1/
├── klasse_c_probe.py            # v1: M1 = d_width · E[i]
├── klasse_c_v2_probe.py         # v2: M1.5/M1.7/M1.9 (3 Methoden)
├── klasse_c_v3_probe.py         # v3: 3 Hebel (FS/SW/RT) — keine Mechanistik
├── m1_top100_wide.txt           # v1: Top-100 WIDE-Anker
├── m1_top10_prompts.txt         # v1: 10 Prompts (M1)
├── m5_top100_wide.txt           # v1: M5 = M1 (tied weights)
├── m5_top10_prompts.txt         # v1: = m1 (bei tied)
├── v2_m15_wide_minus_baseline.txt
├── v2_m17_wide_minus_default.txt
├── v2_m19_dwidth_pos.txt
├── v2_m19_dwidth_neg.txt
├── v2_prompts.txt               # v2: 10 Prompts (M1.5 + M1.9-POS)
├── v3_prompts.txt               # v3: 10 Prompts (FS + SW + RT)
├── klasse_c_prompts.txt         # v1: 10 Prompts als .txt (zum Testen)
├── mechtest_v1.py               # Mechanik-Test-Skript (3 Bedingungen × 10 Prompts)
├── mechtest_analyze.py          # Falsifikator-Auswertung
├── mechtest_hidden/             # 20 .pt Files (L8/L13/L16/L19/L21 Hidden-Captures)
├── mechtest_index.jsonl         # 20 Records (Decoder-Acc, Cosinus, etc.)
├── mechtest_verdict.md          # v1-Test-Befund
└── README.md                    # Klassen-C Probe Methodik
```

## Was als nächstes zu tun ist

1. **mechtest_v1.py mit v3-Prompts laufen lassen** (klasse_c_prompts.txt ersetzen mit v3-Prompts + Few-Shot-Parser)
2. **LEAN_RELAY-Bedingung hinzufügen** (Positiv-Kontrolle, O4 prüfen)
3. **Wenn FS-WIDE% > 30% in BASELINE**: Hypothese "FS als Trigger" empirisch gestützt
4. **Falsifikator-Logik sauber halten**: 是X即非X — weder Krönung ("WIDE erreicht!") noch Weglesen ("nur 8.6% wie LEAN")
