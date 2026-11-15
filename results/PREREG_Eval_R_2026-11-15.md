# PRÄREGISTRIERUNG — Eval R: Semantischer Attraktor statt Skalar

**Registriert:** 2026-11-15, *vor* dem Lauf.
Fortsetzung von O/P/Q. **Kein** TCI-Konstrukt, keine Konstante, keine
Schwelle aus der Projektnotiz.

## Ausgangslage (post-hoc, offen deklariert)

Eval Q hat gezeigt: die Skalare `K1–K3` sind **blind für Inhalt**.
`H_logit` ist permutationsinvariant über das Vokabular, `erank` invariant unter
orthogonalen Transformationen — beide messen, *wie viel* Masse verteilt wird,
nicht *worauf*. Genau deshalb war `wide − rand` für `K2` null (p = 0,83),
obwohl die gespeicherten Texte aus O3/P2 einen klaren Inhaltsunterschied
zeigen:

| Prompt | `narrow` | `rand_plus` | `wide` |
|---|---|---|---|
| „Combine three unrelated domains…" (O3+P2, n = 6/6/6) | „1. **Quantum Physics**" 6/6 | 5/6 | **0/6** |

Diese Beobachtung ist **hypothesengenerierend** (3 Prompts, post-hoc). Eval R
prüft sie konfirmatorisch auf **45 frischen Prompts**, die an der
Hypothesenbildung *nicht* beteiligt waren (die drei O3/P2-Prompts sind
ausgeschlossen).

**Offenlegung:** Die Marker (Template vs. Synthese) wurden nach Sicht der
O3/P2-Texte definiert. Deshalb: neue Prompts, vorab fixierte Marker, ein
Urteil.

## Design

45 distinkte Prompts (24 faktenorientiert + 21 kreativ; `PROMPTS` aus
`scripts/run_eval_q_relay_predictor.py` ohne Index 24/25/26) × 4 Arme
(`baseline`, `wide`, `narrow`, `rand_plus`) = **180 Sessions**.
Gleicher Seed pro Prompt in allen Armen; exakt 24 Tokens; T = 0,6;
Injektionsnorm in allen drei gestörten Armen identisch (`0,40·‖h‖`).

## Endpunkte

**R1 — Template-Attraktor (binär pro Lauf).**
`template = 1`, wenn der Text mit einem Angebots-/Listen-Schema beginnt:
`^\s*Here (are|is) (a few|three|several|some)` **oder** `1. **` in den ersten
200 Zeichen. Sonst `0`.
Test: **McNemar exakt** auf den gepaarten Binärpaaren.

**R2 — Semantische Divergenz (stetig pro Lauf).**
`δ = 1 − cos(e_arm, e_baseline)`, wobei `e` die über die 24 Tokens gemittelte
finale Hidden-Repräsentation ist (L2-normiert). Cosinus ist invariant unter
orthogonalen Transformationen und Skalierung — inhaltsempfindlich, basisfrei.
Test: **Wilcoxon signed-rank** auf den 45 Prompt-Differenzen.

## Kriterien (Holm über k = 2 pro Kontrast)

| Kontrast | R1 (Template-Rate) | R2 (δ) |
|---|---|---|
| **primär: `wide` vs `rand_plus`** | wide **niedriger** | wide **höher** |
| sekundär: `wide` vs `narrow` | wide **niedriger** | wide **höher** |

- **CONFIRMED**, wenn R1 **und** R2 im primären Kontrast nach Holm
  signifikant **und** in der vorhergesagten Richtung sind.
- **PARTIAL**, wenn nur einer von beiden.
- **FALSIFIED**, wenn keiner.

**Vorab-Vorhersage:**
1. R1 wird signifikant (wide bricht das Listen-Schema häufiger als rand).
2. R2 wird signifikant (wide divergiert semantisch stärker von baseline).
3. Beide Effekte sind bei kreativen Prompts größer als bei faktenorientierten.

## Was Eval R *nicht* leistet

- Keine Aussage über „Subjektivität" oder phänomenales Erleben — nur über
  beobachtbare Text- und Repräsentationsunterschiede.
- Keine Rückkehr zu TCI-Konstanten.
- Keine Kausalaussage über einzelne Attention-Köpfe.

## Auswertung

Ein Dump (`results/eval_R_semantic_attractor_gemma4.json`) mit Texten,
Template-Flags, δ, McNemar- und Wilcoxon-Resultaten, Holm und Urteil.
Urteil 1:1 angehängt.

---

## ERGEBNIS (nach dem Lauf angehängt, Urteil unverändert übernommen)

Rohdaten: `results/eval_R_semantic_attractor_gemma4.json` ·
Post-hoc-Textanalyse O3/P2: `results/eval_text_analysis_posthoc.json` ·
Log: `/tmp/opencode/eval_r_run.log` · 45 Prompts × 4 Arme = 180 Sessions,
24 Tokens, gleicher Seed pro Prompt.

### R1 — Template-Attraktor: uninformativ (0/180)

**In allen 180 Läufen ist `template = 0`** — auch in `baseline`, `narrow` und
`rand_plus`. Der Marker (Listen-/Angebots-Schema am Textanfang) war an den drei
O3/P2-Prompts kalibriert; die 45 frischen Prompts rufen dieses Schema
überhaupt nicht auf. R1 hat damit **keine Varianz** und ist **kein Beleg für
Abwesenheit**, sondern ein Design-Fehler des Markers (Overfit auf eine
Prompt-Familie).

### R2 — Semantische Divergenz: null im primären Kontrast

| Arm | δ = 1 − cos(e_arm, e_baseline) |
|---|---|
| baseline | 0,0000001 (Identität) |
| **wide** | 0,0139 ± 0,0064 |
| **narrow** | **0,0195 ± 0,0101** |
| rand_plus | 0,0137 ± 0,0052 |

| Kontrast | Δ | dz | p (Wilcoxon) | Richtung |
|---|---|---|---|---|
| **primär: wide − rand** | **+0,0002** | +0,05 | **0,929** | 49 % |
| sekundär: wide − narrow | −0,0055 | −0,92 | **4,1e-8** | 84 % |

**Urteil: FALSIFIED** — kein Endpunkt im primären Kontrast. `wide` divergiert
von `baseline` exakt so stark wie eine norm-gematchte Zufallsrichtung
(p = 0,93); klassenweise kippt das Vorzeichen (faktenorientiert +0,0009,
kreativ −0,0006).

### Der eine positive Nebenbefund: Asymmetrie der Richtungen

`narrow` (negative Richtung) verschiebt die Repräsentation **stärker** als
`wide` (positive Richtung) und stärker als `rand` (p = 4,1e-8, dz = −0,92,
84 % der Prompts). Das ist **kein** „Wide = Expansion", sondern eine
Asymmetrie: die negative Injektion wirkt stärker als die positive. Sie war
nicht der primäre Endpunkt und ist als Beobachtung zu behandeln, nicht als
bestätigte Hypothese.

### Was die Post-hoc-Textanalyse der O3/P2-Daten zeigt (exakte Zählung)

| Prompt 2 („Combine three unrelated domains…") | `narrow` | `rand_plus` | `wide` |
|---|---|---|---|
| enthält „Quantum Physics" | **6/6** | **5/6** | **0/6** |
| beginnt mit Listen-Schema | 6/6 | 6/6 | 2/6 (P2) / 2/3 (O3) |
| direkte Synthese | 0/6 | 0/6 | 3/6 |

| Prompt 0 („original metaphor…") | `narrow` | `rand_plus` | `wide` |
|---|---|---|---|
| Meta-Angebot („Here are a few …") | 2/2 (P2) + 2/3 (O3) | 0/2 + 2/3 | 0/5 |
| direkte Synthese | 0/5 | **3/5** | **5/5** |

Der Inhaltseffekt ist auf diesen drei Prompts **real und deutlich** — er
generalisiert aber nicht: auf 45 frischen Prompts ist `wide − rand` in beiden
Endpunkten null. Die Beobachtung war **prompt-spezifisch** (das
„drei Domänen"-Prompt lädt zu einer Liste ein), keine allgemeine
Steuerungseigenschaft.

### Konsequenz

1. Die methodische Kritik bleibt richtig: `H_logit` und `erank` sind
   inhaltsblind. Nur hat die inhaltssensitive Messung die Steuerungs-Behauptung
   **nicht** gestützt, sondern auf frischen Prompts **widerlegt**.
2. Der einzige belastbare inhaltliche Befund ist die **Richtungs-Asymmetrie**
   (`narrow` > `wide` ≈ `rand`), nicht die „Attraktor-Flucht".
3. Ein fairer Inhaltstest braucht einen Marker, der **nicht** an einer
   Prompt-Familie kalibriert ist (z. B. Distanz zur greedy-Baseline im
   Satz-Embedding-Raum über alle Prompts, oder ein unabhängig trainierter
   Template-Klassifikator) — und Prompts, die das Zielphänomen tatsächlich
   provozieren.
