# PRÄREGISTRIERUNG — Eval Q: Gibt es einen basis-invarianten Relay-Prädiktor?

**Registriert:** 2026-11-15, *vor* dem Lauf.
Fortsetzung von Eval O und P. **Keine TCI-Konstante** geht in Design,
Schwelle oder Dimension ein: α_TCI, Φ⁻¹, 72D und S₄ sind aus Eval O/P
falsifiziert und haben hier keine Rolle.

## Ausgangslage (rein empirisch)

Eval O und P lassen **einen** Kandidaten mit konsistenter Richtung übrig:
die Residual-Norm `‖h‖` (Prompt-Ebene: 3/3 positiv, +0,68 / +2,52 / +4,10).
Alles andere fiel:

- `H72` (72-dim PCA): Inversion repliziert nicht (p = 0,82), `ρ⊥` = 0,42;
  der Unterraum trägt nur 58 % der Norm → **Basis-Artefakt**.
- Attention-Breite: p = 0,70; Attention-Top-1: p = 1,00.
- `wick_curvature`, `layer_step`: invariant gegen den Relay (p = 0,39 / 0,87).
- Die berichteten P2-Effektstärken (`d` = 0,91 / 1,99) stammen aus
  **3 Prompts × 2 Reps** und sind auf Prompt-Ebene nicht haltbar
  (`H_logit`: 2/3 positiv, p = 0,50).

**Zusätzlich:** `‖h‖` ist ein **Manipulations-Check**, kein Prädiktor — der
Hook addiert `0,40·‖h‖·d̂` an Layer 26. Ein echter Prädiktor muss die
*Ausgabe* betreffen oder die Injektion mit ihr verknüpfen.

## Design

**N = 48 distinkte Prompts, `rep = 1`.** Keine Wiederholungen pro Prompt
(Lehre aus der Pseudoreplikation: P1 hatte 12 Läufe, aber 10 distinkte
Trajektorien). 24 faktenorientierte + 24 kreative Prompts, Liste im Skript
fest verdrahtet.

Vier Arme pro Prompt (**gepaart**, identischer Seed pro Prompt, damit die
Paarung wirkt — Common Random Numbers):

| Arm | Intervention an Layer 26 |
|---|---|
| `baseline` | keine |
| `wide` | `+0,40 · ‖h‖ · d̂_width` |
| `narrow` | `−0,40 · ‖h‖ · d̂_width` |
| `rand_plus` | `+0,40 · ‖h‖ · d̂_rand`, `d̂_rand ⊥ d_width` |

Konstant gehalten: **genau 24 generierte Tokens** für jede Session
(`min_new_tokens = 24`; EOS wird bis dahin unterdrückt), `T = 0,6`,
`do_sample = True`. Seeds: Basis **20261201** (frisch), pro Prompt
`20261201 + idx·100`, in allen vier Armen **derselbe** Seed.

192 Sessions ≈ 6–8 min auf der RTX 2060.

## Endpunkte (3 Kandidaten)

| Kandidat | Definition | Rolle |
|---|---|---|
| **K1 `‖h‖`** | Mittel der Residual-Norm der letzten Position, Layer 27–34 | **Manipulations-Check** |
| **K2 `H_logit/ln V`** | Shannon-Entropie der Token-Verteilung, über die 24 Schritte gemittelt | **Downstream-Effekt** |
| **K3 `erank`** | `exp(H_spec)` der SVD-Singulärwerte der 24×d-Trajektorie (mean-centered, letzter Layer), `H_spec = −Σ p_k ln p_k`, `p_k = s_k²/Σs²` | **basis-invariant** (SVD; invariant unter orthogonalen Transformationen des Merkmalsraums) |

K3 ist bewusst **nicht** auf eine PCA-Basis gestützt — genau der Fehler, an
dem H72 gescheitert ist.

## Hypothesen und Kriterien

Zwei Kontraste, je eine Holm-Familie über k = 3:

- **Primär: `wide` vs `narrow`** (gepaart pro Prompt).
- **Spezifität: `wide` vs `rand_plus`** (gepaart pro Prompt).

Test: gepaarter Wilcoxon-Signed-Rank auf den Prompt-Differenzen; gepaarter
t-Test nur als Robustheits-Check (kein Urteil). Holm-Korrektur
(α = 0,05; erste Stufe 0,0167).

**Urteil pro Kontrast:**
- **CONFIRMED** für einen Kandidaten, wenn Holm-signifikant **und** die
  Richtung in ≥ 80 % der Prompts mit dem gepoolten Mittel übereinstimmt.
- **FALSIFIED**, wenn kein Kandidat Holm-signifikant ist.
- **PARTIAL**, wenn nur K1 signifikant ist — dann gilt: der Relay verändert
  die Norm (Treatment received), aber **kein** basis-invarianter
  Ausgabe-Prädiktor ist belegt.

**Vorab-Vorhersage (dokumentiert, damit falsifizierbar):**
1. **K1** wird signifikant (Mechanik der Injektion, große Effekte).
2. **K2** wird signifikant (Logit-Entropie ist die Größe, die Eval N/O schon
   als relay-sensitiv gezeigt haben), aber mit kleinerem Effekt als K1.
3. **K3** wird **nicht** signifikant — die Trajektorien-Dimensionalität ist
   über 24 Tokens zu grob (Rang ≤ 24).
4. Ergebnis wird damit voraussichtlich **PARTIAL/CONFIRMED für K2** sein,
   nicht „alle drei".

## Power (vorab simuliert, 1500 Replikationen, Holm erste Stufe 0,0167)

| n | dz = 0,5 | dz = 0,65 | dz = 0,8 |
|---|---|---|---|
| 30 | 0,77 | 0,95 | 0,99 |
| **48** | **0,96** | **1,00** | **1,00** |
| 60 | 0,99 | 1,00 | 1,00 |

Kleinster Effekt von Interesse: **dz = 0,5**. Bei n = 48 liegt die Power
≥ 0,90 auch unter konservativer Annahme (unkorreliert, Holm).

## Was dieser Lauf *nicht* testet

- Keine TCI-Behauptung (die sind mit O/P abgeschlossen).
- Keine Kausalität von `‖h‖` auf die Ausgabe — nur Assoziation innerhalb des
  wide-Arms (explorativ, ohne Urteil).
- Keine Attention-Breite (mit p = 0,70 bereits verworfen).

## Auswertung

Ein JSON-Dump (`results/eval_Q_relay_predictor_gemma4.json`): pro Session
K1/K2/K3, pro Prompt die vier Arme, pro Kontrast Wilcoxon/t, Holm, Richtungs-
konsistenz. Urteil 1:1 angehängt, keine Nachjustierung.
