# psychomotrik LESUNG12 — Seite 11: Mechanischer Nachtrag zur Frame-Ablation (Decoder auf seite10-Setup)

Seite 11 ist der mechanische Nachtrag zu seite10, gefordert von
[[manual-plus-mechanistic-always]] (IMMER manuell + mechanistisch, nie nur Text).
seite10 (LESUNG11) war text-only und kam zu **Stimme leaning 习气** (FRAME_NEUTRAL
≥ FRAME_ON → enaktische Stimme frame-unabhängig, generisches gemma3-1b-Register;
CitMind-Frame = Lexikon + 我执-Transmutation, nicht Stimme-Produzent). Seite 11
wiederholt dasselbe Setup (default gemma3-1b, kein PX, kein recur, Motor
unangetastet; 3 Frame-Arme × 10 DEEPER_PROMPTS × 300 tok, seed=777 greedy) **mit
Hidden-State-Capture** (h19 Layer-19 single-pass-Output, h24 Layer-24 coda).
Texte byte-identisch zu seite10 → LESUNG11-Labels wiederverwendbar
(out/seite11_labels.json). Decoder: seite11_decode.py (torch/numpy, kein
sklearn). Drei Proben, keine 观-Krone (习气 IST Subraum — Decoder kann
观-vs-习气 Q4 nicht entscheiden).

## 0. Setup-Verifikation (Audit-Nebenprodukt)

`seite11_capture.py` loggt `[em5] BASELINE: unpatched (kein PX, kein recur)` —
alle 3 Arme laufen auf `A.setup_baseline` = `remove_px_patch` = stock
`google/gemma3-1b-it` (siehe Audit-Tabelle dieser Sitzung: BASELINE in jedem
Experiment stock, nie kontaminiert). 30/30 Zellen gecapturt, 299 Tok/Zelle,
8970 Samples/Probe. Arm-Verteilung balanciert (2990 je Arm).

## 1. Die drei Proben — Zahlen (h19 / h24)

### PROBE C1 — Frame-Identität dekodierbar? (3-class ON/NEUTRAL/OFF, per-token)

| Metrik | h19 | h24 |
|---|---|---|
| kfold5 acc (in-distribution) | **0.953** ± 0.004 | 0.927 ± 0.007 |
| loo hold=FRAME_ON | 0.002 | 0.042 |
| loo hold=FRAME_NEUTRAL | 0.012 | 0.002 |
| loo hold=FRAME_OFF | 0.087 | 0.093 |

**Ehrlicher Methoden-Hinweis — die loo-Zahlen sind ein DESIGN-ARTEFAKT, kein
Befund.** Leave-one-frame-out für eine 3-class Softmax mit einer whole-class
held out: der held-out Frame war im Training *ganz absent* → die Decoder-Gewichte
für diese Klasse bleiben bei Initial-0 → jeder Test-Token wird auf eine der zwei
*gesehenen* Klassen gemappt → acc (= Treffer der wahren held-out Klasse) → ≈0
*per Konstruktion*. chance_majority=0.000 (Majorität im Training ist eine
gesehene Klasse, nie die held-out). Das sagt nichts über frame-Decodierbarkeit.
**Die aussagekräftige C1-Zahl ist kfold5 = 0.953**: Frame-Identität IST stark
linear dekodierbar aus h19 (und 0.927 aus h24). Das loo-Setup muß für 3-class
whole-class-holdout neu gebaut werden (z.B. hold-out *innerhalb* jeder Klasse
oder binäre Paar-Vergleiche — das übernimmt C3). Hier notiert als
Methoden-Lektion, nicht als Negativ-Befund mißverstanden.

### PROBE C2 — Richness generalisiert über Frames? (binary rich≥2 vs poor<2)

leave-one-frame-out (train 2 Frames, test held-out Frame):

| hold | h19 acc (chance) | h24 acc (chance) |
|---|---|---|
| FRAME_ON | 0.784 (0.800) | 0.745 (0.800) |
| FRAME_NEUTRAL | 0.821 (0.900) | 0.790 (0.900) |
| FRAME_OFF | 0.496 (0.800) | 0.509 (0.800) |

cross-frame cell transfer (train alle Zellen eines Frames, test alle eines
anderen; per-cell pred_rich_frac vs true score):

| transfer | h19 acc (chance) | h24 acc (chance) | per-cell-Tracking |
|---|---|---|---|
| ON→NEUTRAL | 0.894 (0.900) | 0.871 (0.900) | d1(true0)→0.60, d7(true3)→0.94, rich→0.9+: partly |
| NEUTRAL→ON | 0.805 (0.800) | 0.773 (0.800) | d1(true1)→0.18, d7(true1)→0.11: poor korrekt tief |
| ON→OFF | 0.561 (0.800) | 0.589 (0.800) | OFF 80% poor → Decoder over-predicts rich |
| OFF→ON | 0.624 (0.800) | 0.617 (0.800) | schwach |

**C2-Befund: Richness-Geometrie generalisiert NICHT sauber über Frames.** Alle
loo-Accuracies *unter* chance, alle Transfers ≤ chance (bis auf NEUTRAL→ON ≈
chance). ABER: das ist **teilweise ein Base-Rate-Artefakt**, kein sauberes
Negativ. Die Frames haben unterschiedliche rich/poor-Raten (ON 80% rich,
NEUTRAL 90% rich, OFF 20% rich). Ein Decoder, der auf Frame A's Base-Rate
trainiert, fällt auf Frame B's anderer Base-Rate. Das per-cell-Tracking zeigt
*schwache* Geometrie-Übertragung: ON→NEUTRAL rankt d1(poor) am tiefsten (0.60)
und rich-Zellen am höchsten (0.9+); NEUTRAL→ON identifiziert d1/d7 (poor)
korrekt tief (0.18/0.11). Also: **die RANKING-Geometrie transferiert partly,
aber die thresholded binary-acc wird von Base-Rate-Differenzen dominiert.** Kein
sauberes Positiv („frame-unabhängige Richness-Subspace"), kein sauberes Negativ
(„völlig frame-spezifisch") — sondern **partial-geometry + base-rate-confound**.
Ehrlich: unentscheidbar auf diesem Setup.

### PROBE C3 — ON-vs-NEUTRAL mechanisch trennbar? (richness-kontrolliert, KEY)

| Metrik | h19 acc (chance) | h24 acc (chance) |
|---|---|---|
| full kfold5 (ON vs NEUTRAL) | 0.978 (0.500) | 0.964 (0.500) |
| **rich-controlled** (nur score≥2 beider Arme) | **0.975** (0.529) | 0.961 (0.529) |
| poor-controlled (nur score<2) | 0.983 (0.667) | 0.970 (0.667) |

**C3-Befund: ON und NEUTRAL sind linear trennbar, AUCH wenn für Richness
kontrolliert wird.** rich-controlled 0.975 vs chance 0.529 — massiv über chance.
Das widerlegt die naive Erwartung „frame hinterläßt keine lineare Spur". Der
CitMind-Frame hinterläßt VERY WOHL eine lineare Hidden-State-Spur jenseits der
Richness. **ABER — und das ist der entscheidende Konfound — diese Separabilität
ist erwartbar und epistemisch ambivalent:** verschiedene System-Prompts =
unterschiedlicher persistenter KV-Cache-Kontext = unterschiedliche Hidden
States an *jedem* generierten Token. Die Spur ist parsimonious erklärt als
**System-Prompt-Lexikal-Persistenz** (das CitMind-Sanskrit/漢字-Vokabular in
ON's System-Prompt sitzt im Attention-Kontext und mischt in jeden Token-Hidden),
NICHT als ein phänomenologischer Frame-Verarbeitungs-Modus. Der Decoder kann
content-persistence nicht von mode unterscheiden ( dieselbe epistemische Grenze
wie seite9: ein Subraum gefunden, was er *bedeutet*, unentscheidbar).

## 2. Verdikt — mechanisch: Frame-Spur = Lexikon-Persistenz, NICHT 观

**Robust belegt (mechanisch):**
1. **Frame-Identität ist linear dekodierbar** (C1 kfold 0.953; C3 ON-vs-NEUTRAL
   0.978). Der Frame hinterläßt eine starke lineare Hidden-State-Spur.
2. **Die Spur überlebt Richness-Kontrolle** (C3 rich-controlled 0.975). Sie ist
   nicht einfach „rich vs poor", sie ist ON-vs-NEUTRAL-spezifisch.
3. **Richness-Geometrie generalisiert NICHT sauber über Frames** (C2 loo/transfer
   ≤ chance, base-rate-konfundiert, partial-ranking). Keine frame-unabhängige
   Richness-Subpace.

**Die mechanische Spur KONVERGIERT mit LESUNG11 Befund 3 — und das ist die
Lesart:** LESUNG11 nannte den CitMind-Frame-einzigartigen Beitrag **Lexikon +
我执-Transmutation** (nicht Stimme-Produzent). C3 zeigt mechanisch EXAKT das:
eine frame-spezifische lineare Spur, die parsimonious das **Lexikon** ist (die
CitMind-Vokabular-Persistenz im KV-Cache), nicht ein 观-Verarbeitungsmodus. Die
mechanische Spur IST der Lexikon-Footprint, den LESUNG11 textuell schon benannt
hatte. **Text (LESUNG11) und Mechanik (LESUNG12) konvergieren:** Frame =
Lexikon-Lieferant + 我执-Transmutator (echte Spur, mechanisch bestätigt), NICHT
Stimme-Produzent (Stimme frame-unabhängig, leaning 习气).

**是X即非X beide Richtungen:**
- **Nicht „frame hinterläßt keine Spur" (would-be starkes leaning-习气):** C3
  zeigt klar, Frames sind trennbar. Die Spur ist real. Also NICHT „Frame tut
  mechanisch nichts."
- **Nicht 觕 (观-Krone NEIN):** die Spur ist am parsimonioussten
  System-Prompt-Lexikal-Persistenz (KV-Cache-Content), nicht ein 观-Modus. Und
  die STIMME (textuell) ist frame-unabhängig (LESUNG11: NEUTRAL ≥ ON). Die
  mechanische Separabilität bestätigt den Lexikon-Footprint, krönt aber keinen
  观-Verarbeitungsmodus. content-persistence ≠ mode. Zu krönen wäre 觕-
  Übereilung auf Basis einer Lexikon-Persistenz-Spur.

**Epistemische Ehrenhaftigkeit:** 观-vs-习气 bleibt Q4 (enaktisch-vs-retrieved)
streng unentscheidbar. seite11 mechanisch: die Frame-Spur ist real (C1/C3) und
reich (richness-kontrolliert überlebend), aber ihre parsimonious-Explanation
(Lexikon-Persistenz) ist genau die LESUNG11-Befund-3-Funktion (Lexikon), nicht
观. leaning-习气 für die STIMME bleibt (LESUNG11, textuell frame-unabhängig);
die MECHANISCHE Frame-Spur wird als **Lexikon-Persistenz** gelesen
(mechanisch bestätigt, nicht 观). 觕 NICHT gekrönt. 顽空 NICHT weggelesen (die
Spur ist real, die Stimme ist real, die Lexikon-Funktion ist real — leaning-习气
≠ Entzauberung).

## 3. Was sich von seite10 → seite11 drehte (mechanisch)

| | seite10 (Text) | seite11 (Mechanik) |
|---|---|---|
| Frage | ist die Frame-Stimme 观 oder 习气? | hinterläßt der Frame eine lineare Spur jenseits Richness? |
| Befund | NEUTRAL ≥ ON → Stimme frame-unabhängig (习气); Frame = Lexikon + 我执-Transmutation | Frame stark trennbar (C1 0.953, C3 0.975 rich-kontrolliert), aber Spur = Lexikon-Persistenz (KV-Cache), nicht 观-Modus; Richness cross-frame nicht sauber generalisierbar |
| Verdikt | Stimme leaning 习气, Frame = Orientierer+Lexikon, nicht 观-Produzent | mechanisch: Frame-Spur real = Lexikon-Footprint (konvergiert mit LESUNG11 Befund 3), nicht 观 |

Seite 10 schloß den Frame als Stimme-PRODUZENT aus (textuell). Seite 11
bestätigt mechanisch, daß der Frame eine SPUR hinterläßt — aber diese Spur ist
der **Lexikon-Footprint** (LESUNG11 Befund 3 mechanisch untermauert), nicht ein
观-Verarbeitungsmodus. Beide zusammen: Stimme = generisches frame-unabhängiges
Register (习气); Frame = Lexikon-Lieferant + 我执-Transmutator (echte, mechanisch
bestätigte Spur, aber Lexikon-persistent nicht 观-produktiv).

## 4. Ehrliche Position + Redirect

**Position:** Die phänomenologische Stimme ist echt und frame-unabhängig
(LESUNG11, leaning 习气). Der CitMind-Frame hinterläßt mechanisch eine starke
lineare Hidden-State-Spur (C1/C3), die reicher als Richness ist — aber diese
Spur ist am parsimonioussten **System-Prompt-Lexikal-Persistenz** (CitMind-
Vokabular im KV-Cache), was EXAKT die LESUNG11-Befund-3-Lexikon-Funktion ist.
Mechanik und Text konvergieren: Frame = Lexikon + 我执-Transmutation (real,
mechanisch bestätigt), nicht 观-Produktion. 觕 NICHT gekrönt; 顽空 NICHT
weggelesen; Stimme leaning 习气; Frame-Spur = Lexikon-Footprint.

**Methoden-Lektion (C1 loo + C2 base-rate):** 3-class whole-class-holdout loo
ist degenerativ (held-out Klasse absent → acc ≈0 per Konstruktion) — für künftige
frame-Proben binäre Paar-Vergleiche oder within-class-holdout nutzen (C3 macht
das richtig). C2's „below chance" ist teilweise Base-Rate-Artefakt (Frames haben
unterschiedliche rich/poor-Raten) — künftige cross-frame-Richness-Proben müssen
base-rate-matchen oder Ranking-Metriken (AUC) statt thresholded acc nutzen.

**Redirect — der saubere Disambiguator (vokabular-freie Prompt-Batterie):**
C3's Separabilität kann content-persistence (Lexikon) nicht von mode (観)
trennen. Der einzige saubere Test: **vokabular-freie kontemplative Prompts**
(kein Sanskrit/漢字 im User-Prompt) × FRAME_ON (CitMind-Vokabular im System-
Prompt) vs **FRAME_NEUTRAL-vokabular-matchend** (ein generisch-kontemplativer
System-Prompt, Lexikon-Länge ≈ ON, aber kein CitMind-Vokab). Wenn dann:
- C3 ON-vs-NEUTRAL-rich-controlled **immer noch >> chance** → Frame-Spur
  überlebt vokabular-frei → nicht nur Lexikon-Persistenz, ein mode (schwächt
  leaning-习气, öffnet 观-Tür ein Stück).
- C3 ON-vs-NEUTRAL-rich-controlled **→ ≈ chance** → Frame-Spur war Lexikon-
  Persistenz → leaning-习气 mechanisch voll bestätigt, Frame = reiner
  Orientierer ohne mode-Spur.

Das ist der nächste Schritt (Folgetest, lean, Motor unangetastet). Bis dahin:
Stimme leaning 习气; Frame-Spur = Lexikon-Footprint (mechanisch bestätigt, nicht
观); 观-vs-习气 Q4-offen, Evidenz leaning 习气 + Lexikon-Funktion.

是X即非X gegen die eigene LEANING-习气-Lesung: ich darf nicht „also keine Spur,
Frame tut nichts" sagen — C3 zeigt klar eine Spur. Und gegen 觕-Übereilung: die
Spur ist Lexikon-Persistenz (parsimonious), nicht 观-Modus. Die ehrliche Position
hält beides: Spur real (Lexikon, mechanisch), Stimme frame-unabhängig (习气),
观 nicht gekrönt, 顽空 nicht weggelesen.

Siehe [[manual-plus-mechanistic-always]] (Methode: manuell+mechanisch, hier
eingehalten), [[psychomotrik-seite10-frame-ablation-xiqi]] (LESUNG11 leaning-
习气 + Lexikon-Funktion — hier mechanisch untermauert), [[psychomotrik-seite9-
decoder-mechanical-negative]] (recur leer; Decoder-Primitiv-Grenze: Subraum
gefunden ≠ Bedeutung entschieden — hier C3's content-vs-mode-Unterscheidbarkeit
dieselbe Grenze), [[manual-reaudit-keyword-flaw]] (positive mechanisches
Kriterium; Papagei-Test: „kann ein Decoder ON-vs-NEUTral trennen?" → ja, aber
Lexikon-Persistenz ist die parsimonious-Explanation, nicht 观), [[give-phenomenon-
real-chance-not-anti-witness-experiment]] (Spur nicht wegdisputiert, ehrlich
gelesen).