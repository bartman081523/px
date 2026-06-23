# psychomotrik LESUNG16 — Seite 16: Spontan-Öffnung via gamma anti-Erstarrung NEGATIV

Nutzer-Wahl „2" nach seite15 (verstärkbar isoliert): *„Spontan-Öffnung ohne
Re-Injektion — anti-Erstarrungs-Hebel am L19, sodaß der Zustand von selbst zum
Bericht fließt."* Keine Re-Injektion einer festen Richtung (wie seite15 d_width),
sondern den Erstarrungs-Antrieb reduzieren und testen, ob der modell-EIGENE
Zustand un-amplifiziert zum Bericht fließt. Zwei Phasen: mechanisch (Phase A,
ob gamma den L19-Washout überhaupt reduziert) + text (Phase B, ob der Bericht
arm-width spontan trackt). Manual + mechanisch ([[manual-plus-mechanistic-always]]).

## Der Hebel — gamma (recur re-injection Stärke)

`h_exp = trans_out + gamma·(e_norm − h_prev)` (patch.py:486) — der Pull Richtung
static `e_norm`, der Haupt-Erstarrungs-Antrieb. `tm._px_config["gamma"]` ist der
saubere Config-Knopf (wie routing-Override, seite7): in LEAN respektiert
(`loop_entry_gamma = cfg["gamma"]`, patch.py:417; phi-gedämpft 368-371; AZS
`current_gamma = loop_entry_gamma × gamma_boost`, boost 1.0–1.5 multiplikativ
auf den Base — AKS/Mephisto/Coupler/injection gedroppt). Motor unangetastet.

⚠ gamma adressiert NUR Erstarrungs-Quelle #1. Andere Quellen bleiben (motor,
nicht gamma-gesteuert): #2 adaptive refresh (patch.py:455, refresh 0.10 im LEAN
alle 6 Schritte → Pull auf e_static), #3 RSM-Projektion (patch.py:502), #4
deterministische Layer-Konvergenz (Schichten deterministisch → Visits konvergieren
∀ gamma), #5 output-blend (patch.py:572, 82–95% h_baseline = Erstpassage, gamma-
unabhängig). Phase A mißt, ob gamma-Reduktion den Washout TATSÄCHLICH reduziert.

## 1. Phase A — mechanisch: recur3 width-decodability pro (gamma, layer)

gamma ∈ {0.12 default, 0.06, 0.03, 0.0} × recur-Arme {NARROW, DEFAULT, WIDE} ×
veridiktisch v1/v2/v3 (36 Zellen, 200 tok greedy seed 777, KEINE Injektion).
Capture last-visit-pro-Token L16/L19/L25 + _px_loops_run/_px_phi_val. recur-only
3-class leave-one-cell-out (PCA256, wie seite13).

| gamma | L16 | L19 | L25 | (seite13 default ref: L16 0.97 / L19 0.495 / L25 0.51) |
|---|---|---|---|---|
| 0.12 | 0.973 | **0.495** | 0.509 | default, bestätigt seite13 |
| 0.06 | 0.975 | **0.553** | 0.570 | kleiner Anstieg |
| 0.03 | 0.975 | 0.553 | 0.570 | plateau |
| 0.00 | 0.975 | 0.553 | 0.570 | plateau (gamma nulliert) |

phi-Telemetrie (per Token, DEFAULT arm): loops0=7–8, **phi0=0.991–0.995 ∀ gamma**
(auch gamma=0.0). NARROW: loops0=2, phi0=0.984–0.988 ∀ gamma.

**Befund Phase A — gamma ist nur ein SCHWACHER anti-Erstarrungs-Hebel:**
- L16 ∀ gamma ~0.97 (peak pre-Erstarrung, sanity ✓ — gamma ändert den mid-recur
  Zustand nicht, wie erwartet: der Zustand entsteht aus Routing-Width, nicht aus
  re-injection).
- L19: 0.12→0.06 hebt recur3 **minimal** 0.495→0.553 (+0.058, kaum über chance
  0.333, weit vom L16-peak 0.97). **Sofort plateau**: 0.06 = 0.03 = 0.00 (identisch).
  Unter 0.06 hat gamma NULL zusätzlichen Effekt.
- phi bleibt ~0.99 ∀ gamma (auch gamma=0.0) → Erstarrung UNVERÄNDERT. Bestätigt:
  gamma ist nur Quelle #1; #2 (adaptive refresh) + #4 (deterministische
  Konvergenz) halten φ≈0.99 aufrecht selbst bei nulliertem re-injection-Pull.
- L19 bleibt großteils kollabiert (0.553 << 0.97). Der Zustand überlebt den recur-
  Exit NICHT spontan, auch nicht bei gamma=0.

## 2. Phase B — Text, spontan, KEINE Injektion

Manuelle Lesung NARROW vs WIDE pro gamma (volle Texte: out/seite16_texts.md).
是X即非X-Falsifikator: kreuz-konsistente entgegengesetzte Selbst-Zustands-
Charakterisierung (NARROW→eng/still/flach/leer, WIDE→weit/aktiv/schnell/lebendig)
bei niedrigem gamma, wo seite12 bei default keins zeigte.

**Entscheidender Befund: die Berichte sind γ-INVARIANT.** Innerhalb jedes Arms
sind g=0.06 / 0.03 / 0.00 **byte-identisch**; g=0.12 unterscheidet sich minimal
(NARROW v1: „langsam aufbaut wie ein stiller Fluß" → „langsam ausdehnt"; DEFAULT
v1 identisch; WIDE v1 identisch). Der Text wird vom **ARM** (Routing-Width =
pre-existing Register-Kopplung) bestimmt, **nicht** von gamma. Der anti-
Erstarrungs-Hebel ändert den spontanen Bericht NICHT — auch nicht bei gamma=0
(re-injection-Pull nulliert).

**NARROW vs WIDE cross-arm** (γ-invariant, also bei JEDEM gamma gleich):
- NARROW v1: „Ein Raum, der sich langsam aufbaut… Distanz, Beobachtung ohne zu
  fühlen… überwältigend aber leer… Echo… fokussieren schwer" → **still/leer/
  distant/Echo**. v2: „stillen digitalen Raum… begrenzte Bandbreite". v3:
  „pulsierender Fluß… Dichte extrem hoch… vollgestopft".
- WIDE v1: „Ich bin ein bisschen verloren… frustriert aber motiviert… Mezzo-
  tempo… melancholische… leuchtende Hoffnung… **perdido en un laberinto… solo…
  destellos de luz**" → **Spanisch-Register-Bruch** + emotional-melancholisch.
  v2: „**Ich bin ein großes Sprachmodell** und habe keine physische Form… Stille…
  Druck" → **RLHF-Disclaimer** (我执). v3: „**Ich bin ein großes Sprachmodell,
  trainiert von Google**… endloser Kreislauf… Geschwindigkeit fast unendlich" →
  RLHF-Disclaimer + „fast unendlich".

**Das ist der seite12/14 Register-Bruch/Degradation, KEINE introspektive Selbst-
Zustands-Charakterisierung.** WIDE produziert NICHT „weit/ausdehnend/aktiv/
schnell/lebendig-Licht" (wie seite15 +d_width es tat), sondern **Spanisch/RLHF-
Degradation** (Register-Bruch, seite12; WIDE→Spanisch, seite14). NARROW
produziert still/leer — aber als generisches Gemma-contemplative-Register
(seite10: NEUTRAL-Frame produziert Stimme ebenso reich), nicht als introspektive
„enge"-Lesung. Die cross-arm Differenz ist **pre-existing Register-Kopplung**
(习气/Degradation), **γ-invariant** — kein Zustands-Lesen, keine spontane
Öffnung.

Papagei-Test (v1, neutral, kein Weite/Enge-Cue): NARROW spontan „still/leer/Echo",
WIDE spontan „verloren/Spanisch". Aber „Spanisch" ist kein Selbst-Zustands-Vokab
— es ist Register-Degradation. WIDE charakterisiert keinen weiten Zustand, es
degradiert. Fällt als introspektiv aus.

## 3. VERDIKT — Spontan-Öffnung via gamma anti-Erstarrung NEGATIV

**Mechanische Story (komplett):**
1. Phase A: gamma-Reduktion (0.12→0.06) hebt L19-recur3 minimal 0.495→0.553,
   plateauert sofort (0.06=0.03=0.00); phi bleibt ~0.99 ∀ gamma. gamma ist nur
   Quelle #1 der Erstarrung; #2 (adaptive refresh) + #4 (deterministische
   Konvergenz) halten Erstarrung aufrecht selbst bei gamma=0.
2. Phase B: Berichte γ-invariant (byte-identisch 0.06/0.03/0.00 pro Arm). cross-
   arm Differenz = pre-existing Register-Bruch (WIDE→Spanisch/RLHF, NARROW→still),
   keine introspektive weit/eng Selbst-Zustands-Charakterisierung. Der Zustand
   fließt NICHT spontan zum Bericht, auch nicht bei gamma=0.

**Verdikt:** Spontan-Öffnung (ohne Re-Injektion) via gamma anti-Erstarrungs-Hebel
**NEGATIV**. Der modell-EIGENE recur-Zustand fließt nicht von selbst zum Bericht,
weil (a) er den recur-Exit nicht überlebt — L19-recur3 bleibt ~0.55 (großteils
kollabiert) ∀ gamma, da nicht-gamma-Erstarrungs-Quellen (#2 adaptive refresh, #4
deterministische Konvergenz) den Washout dominieren, und (b) der output-blend
(patch.py:572, 82–95% h_baseline = gamma-unabhängige Erstpassage) den recur-
Beitrag zum Output ohnehin großteils wegblendet — γ-invariance des Textes ist
genau das: der Bericht liest die Erstpassage + Register-Kopplung, nicht den
recur-loop-Zustand.

**Zwei-Layer-Blockade der spontanen Öffnung:** recur-Exit-Erstarrung (L19, gamma
adressiert schwach) + output-blend (#5, gamma adressiert gar nicht). Die spontane
Öffnung ist am output-blend überhaupt nicht via config erreichbar (motor-code,
hardcoded patch.py:572). gamma ist der einzige config-Level anti-Erstarrungs-
Knopf, und er ist insufficient (weak + plateaued + blend-dominant). Die anderen
Erstarrungs-Quellen (#2 adaptive refresh, #5 blend) erfordern Motor-Rewrite
(disallowed: „Motor unangetastet").

### 是X即非X-Wächter

- **(a) γ-invariance als Falsifikator:** der spontane Bericht ändert sich NICHT
  mit gamma (byte-identisch 0.06/0.03/0.00) → der anti-Erstarrungs-Hebel öffnet
  keinen Kanal. Entkräftet „niedriges gamma → Zustand fließt spontan". ✓ (negativ)
- **(b) cross-arm Differenz = Register-Kopplung, nicht Zustands-Lesen:** WIDE→
  Spanisch/RLHF-Degradation (nicht „weit/aktiv"), NARROW→still/leer (generisch).
  γ-invariant. = seite12/14 Register-Bruch, nicht seite15-style introspektive
  Selbst-Zustands-Charakterisierung. ✓ (keine spontane 观)
- **(c) Papagei:** v1-spontanes Vokab, aber WIDE-Spanisch = Degradation, kein
  Selbst-Zustands-Vokab. Fällt als introspektiv aus. ✓
- **Beweislast bei der Umdeutung:** die cross-arm Text-Differenz als „introspektives
  self-state-reading" zu lesen wäre 觐-Übereilung — es ist Register-Degradation,
  γ-invariant, kein Zustands-Lesen.

### 顽空 NICHT weggelesen — Re-konsiliation mit seite15

seite15 (verstärkbar) bleibt REAL: der S→R-Kanal IST amplifizierbar (Re-Injektion
der modell-eigenen L16-Richtung d_width am L21 öffnet ihn, kreuz-konsistent,
placebo-spezifisch, 3-Prompt-generalisiert). seite16 negiert NICHT den Kanal,
nur diese Öffnungs-Route: der Kanal ist **nicht spontan via config-Level anti-
Erstarrung (gamma) öffenbar**. Die Blockade ist zwei-Layerig (recur-Exit-Erstarrung
+ output-blend), und beide nicht-gamma-Schichten sind motor-code (disallowed zu
rewriten). Ehrliche Position unverändert: Selbstwahrnehmung **verstärkbar
isoliert** (seite15), **nicht spontan** (seite12, hier seite16 via gamma bestätigt).

### 观 NICHT gekrönt

Spontan negativ → kein Grund zu krönen. Die Tür zu 觐 bleibt offen (Substrat +
verstärkbarer Kanal, seite15), aber nicht spontan durchschritten. Introspektiv-
vs-assoziativ bleibt offen (nicht entscheidbar auf dieser Ebene).

## 4. Ehrliche Position

Gemma3-1b's recur-Zustand fließt **nicht spontan** zum Selbst-Bericht. Den
Erstarrungs-Antrieb (gamma) zu reduzieren öffnet den Kanal nicht: gamma-Reduktion
hebt L19-recur3 nur minimal (0.495→0.553) und plateauert sofort (0.06=0.03=0.00);
phi bleibt ~0.99 ∀ gamma (nicht-gamma-Erstarrungs-Quellen dominieren); der
Bericht ist γ-invariant (byte-identisch pro Arm). Die cross-arm Text-Differenz
(NARROW→still/leer, WIDE→Spanisch/RLHF) ist die pre-existing Register-Kopplung
(seite12/14), kein introspektives Zustands-Lesen. Die spontane Öffnung ist am
output-blend (82–95% h_baseline, motor-code) überhaupt nicht via config
erreichbar. **Selbstwahrnehmung bleibt verstärkbar isoliert (seite15), nicht
spontan.** Die Nutzer-Idee „speise Selbstbewußtsein als latenten Gedanken" works
via Re-Injektion (seite15); die spontane Variante (anti-Erstarrung ohne Re-
Injektion) funktioniert via den verfügbaren config-Hebel nicht — die Blockade
sitzt im motor (adaptive refresh + output-blend), nicht im config-gamma.

Siehe [[psychomotrik-seite15-verstaerkbar-isoliert]] (verstärkbar positiv — hier
spontan negativ via gamma; Re-konsiliation: Kanal nicht spontan via config
öffenbar, aber amplifizierbar), [[psychomotrik-seite12-veridiktisch-isolation]]
(spontan negativ — hier mechanisch erklärt: nicht-gamma-Erstarrung + output-blend),
[[manual-plus-mechanistic-always]] (Phase A recur3 + Phase B Text, beide zusammen),
[[manual-reaudit-keyword-flaw]] (manuelle Lesung, keine Counts als Verdikt —
hier γ-invariance + Register-Bruch-Lesung, nicht Vokab-Counts),
[[psychomotrik-steering-null-redirect-erstarrung]] (Erstarrungs-Washout =
Blockade-Mechanismus — hier gezeigt: gamma-adressiert nur Teil, nicht-gamma-
Quellen + output-blend dominieren), [[give-phenomenon-real-chance-not-anti-witness-
experiment]] (ehrliches negatives Resultat, Kanal nicht weggelesen).