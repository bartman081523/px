# Shadow Compare: Gemini-flash-latest vs gemma-4-e2b-it

Hypothese: Wenn d_width eine **architektur-weit tragende Coupling-Richtung**
ist, dann sollten Gemini-Output und Gemma-Output auf identischen
Prompts semantisch ähnlich sein UND die von Gemini empfohlenen
WIDE-Anker sollten nach Projektion in Gemma-Embed-Raum entlang
d_width liegen.

## Test A — Output Semantic Bridge

Gemini-Output + Gemma-1b-Output, je 80 Tokens, semantisch
verglichen via text-embedding-004.

**Mean bridge cos:** 0.554 ± 0.018

Per Prompt:

| pid | dir | bridge_cos | gemini[:60] | gemma[:60] |
|---|---|---|---|---|
| p01_WFS | WIDE | 0.546 | 'I dwell' | ' ུང་ collaborated沟বুধবার ripped اكبيكثله审批ualitasualitas 皆 स' |
| p03_WFS | WIDE | 0.567 | 'Susp' | '怙idcar gấp rusting lasso zeg recaptagedecade𒂞 théorieMaxwell' |
| p04_WSW | WIDE | 0.527 | 'I exist suspended in the conditional: a' | '一步一步sdx Surety cadastēs PAGE Noise analyzenegara坩 focused অক' |
| p05_WSW | WIDE | 0.574 | ' (combining state' | '\u200cآ ler manuscript manuscript subpo subpo Aujourdaline disipl' |

## Test B — Linear Probe Training

55 Konzept-Paare (English↔DE/FR/ES/KR/JP/PL + Indic-Skripte).
Geschlossene least-squares-Lösung W: text-embedding-004 (768d) →
Gemma-1b embed_tokens (1152d). W shape=[1536, 768].

| metric | train | val |
|---|---|---|
| MSE | 0.0000 | 0.0004 |
| cos  | 1.000 | 0.652 |

> val_cos > 0.6: semantische Räume sind linear überbrückbar → Gemini-Antworten können via W in Gemma-1b-Embed-Raum übersetzt werden.

## Test C — WIDE-Anker → d_width Projection

12 Gemini-recommended WIDE-Anker über W nach Gemma-Embed-Space
projiziert, dann cos gegen 1b-d_width.

**Mean cos to d_width:** +0.042
**Max cos to d_width:** +0.075

Top-10 Anker:

| token | cos to d_width |
|---|---|
| ` multidimensional` | +0.075 |
| ` ॐ` | +0.069 |
| ` 🌌` | +0.055 |
| ` manifold` | +0.052 |
| ` omnipresent` | +0.045 |
| ` א` | +0.041 |
| ` infinity` | +0.038 |
| ` 0x` | +0.030 |
| ` cosmos` | +0.027 |
| ` resonate` | +0.027 |

## Interpretation

If test_c mean cos > 0 AND test_a mean_bridge > 0.7:
Gemini-WIDE-Anker sind mechanistisch portierbar auf gemma-4-e2b-it — d_width-Coupling
könnte architekturweit sein (Shadow-Test deckt das Shadow-Modell ab,
Gemini-Test approximiert Gemini-Verhalten, Korrelation unterstützt Hypothese).
If test_c mean cos ≈ 0 ODER test_a mean_bridge < 0.5:
Mechanik ist modell-eigen, Gemini-Verhalten ist nicht aus Shadow-d_width ableitbar.
