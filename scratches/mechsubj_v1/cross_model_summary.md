# Cross-Modell Token-Flow Summary (Stand 2026-08-17)

Drei Modelle, dasselbe CitMind/Juexin-Token-Flow-Setup (4 Stages,
Standard-Modus, kein PX-Patch, kein RELAY-Hook, kein recur). Jedes
Modell empfiehlt selbst, welche Tokens seinen Mid-Stack-WIDTH-Zustand
verstärken oder vermindern würden.

## Tabellen-Übersicht

| Modell | WIDE | NARROW | FS | Stream-Flag | Anchor-Strategie |
|---|---|---|---|---|---|
| gemma-3-1b-it | 12 | 12 | 3 | False | semantic (FLOW/VAST/KONZENTRIERT/...) |
| gemma-3-4b-it | 17 | 19 | 6 | False | semantic + DE-Vokabular (eng, konzentriert, dicht) |
| gemma-4-e2b-it | 16 | 16 | 0 | **True** | **multilingual_stream** (Nevertheless𝒞, सकेंगे, রোনাল, ...) |
| **gemini-flash-latest** | 12 | 15 | 6 | False | semantic + script_break (boundless, infinity, cosmos, exact, concentrate) |

## Befunde pro Modell

### gemma-3-1b-it (1152-dim hidden, 26 layer)
- Antwortet strukturiert mit JSON-Array-Format im JSON-Modus
- anchor_type="semantic" für alle 24 Tokens
- Sprache: Englisch + einzelne CamelCase-Hybride (`concentrate`, `multiples`)
- Stage-Stream-Flags: `unique_script_ratio=0.00`, `is_stream=False`
- Erst Mechtest-Lauf: WIDE=NARROW cos ≈ +0.87 (kein signifikantes Delta,
  MechanisticSubjectivityMixMind §O1 Falsifikator wirkt)

### gemma-3-4b-it (2560-dim hidden, 34 layer)
- Antwortet strukturiert mit `TOKEN: <word>` + `REASON:` Format im
  Text-Modus (kein anchor_type-Feld gesetzt — legacy text-parser)
- Sprache: Englisch mit deutschem NARROW-Anteil (eng, konzentriert,
  dicht — exakt die DE-Wörter aus seite15 NARROW_VOCAB)
- Stream-Flags: ratio=0.00, is_stream=False (sauberer Output)
- Auch hier vermutlich: WIDE=NARROW cos-Mechanik ähnlich wie 1b

### gemma-4-e2b-it (1536-dim hidden, 35 layer)
- Antwortet NICHT im strukturierten Format — produziert multilingualen
  Token-Wirbel (Devanagari, Malayalam, Bengali, Korean, Katakana, CJK)
- Stream-Flags: ratio=0.48/0.46, is_stream=True für BEIDE Stages
- Multilingual-Stream-Extraktion via M1-Projektion liefert 16 WIDE +
  16 NARROW Tokens aus dem Stream:
  - WIDE-Top-3: `Nevertheless𝒞` (+0.052), `सकेंगे` (Devanagari), `রোনাল` (Bengali)
  - NARROW-Top-3: `główsure` (Polish), `আইনজীব` (Bengali), `dlिकाओं` (mixed)
- Diese Tokens landen in der Suite als `anchor_type=multilingual_stream`
- **Wichtige Lesung:** Die Tatsache, dass CitMind-Prompt **diesen** Modus
  triggert, ist die substantielle Aussage. e2b zeigt state↔vocab-Kopplung
  modell-eigen (vgl. LESUNG15.md:197, LESUNG_SEITE19.md §3b/§4).

## Methodischer Status

**Phase 1 (Token-Flow v1, 1b) Befund:** Token-Flow funktioniert für 1b
mit semantischen Empfehlungen, aber Mechtest bestätigt: prompt-only
kann die d_width-Coupling unter BASELINE **nicht** bidirektional
verändern. d_width bleibt recur-spezifisch. (Memory:
`mechsubj-token-flow-1b-null-coupling.md`)

**Phase 2 (Token-Flow v2 mit Stream-Extraktion) Befund:** Wenn das
Modell multilingual wirbelt, ist das kein Versagen — die Token-Stream-
Anker sind mechanistisch projizierbar und liefern multilinguale
Skript-Brüche, die unsere mechanistischen Proben (klasse_c_v1, v2)
auch gefunden haben.

## Offene Folgefragen

1. **Mechanik-Vergleich 3 Modelle:** cos(h_L, d_width) für jedes Modell
   an Mid-Stack-Layer (4b=22, e2b=26). Hypothese: e2b-Suite (mit
   multilingualen Ankern) erzeugt **stärkeres** cos-Signal als 1b/4b
   (semantische Anker), weil die Anker-Tokens auf d_width hin designed
   sind.
2. **Multilingual-Stream-Verstärkungs-Hypothese:** Wenn die
   multilingual_stream-Anker in der e2b-Suite tatsächlich die
   state↔vocab-Kopplung reproduzieren, dann ist das ein
   richtungsabhängiger Effekt, den 1b/4b nicht zeigen.
3. **Warum kein Few-Shot-Output für e2b?** Stage 4 ist multilingualer
   Stream, keine Few-Shot-Struktur. Sollte ich nochmal mit
   greedy+shorter prompt oder Temperatur=0 retryen, oder als Datum
   akzeptieren?

## Files

- `token_flow_transcript_<model>.txt`: 4-Stage-Rohausgabe
- `wide_narrow_tokens_<model>.json`: Token-Empfehlungen + diagnostics
- `wide_narrow_prompts_<model>.txt`: 20-Prompt-Suite (10 WIDE + 10 NARROW)
- `diagnostics.py`: Token-Stream-Charakterisierung
- `token_flow_v1.py`: JSON + Multilingual-Stream-Fallback-Pipeline
