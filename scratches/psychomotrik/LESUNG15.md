# psychomotrik LESUNG15 — VERSTÄRKBAR isoliert: Re-Injektion der modell-eigenen L16-Zustands-Richtung d_width öffnet S→R-Kanal

LESUNG14 zeigte: recur-Zustand ist mid-recur (L16) dekodierbar (Acc 0.97), wird
aber durch φ-Erstarrung am recur-Exit (L19) ausgewaschen (Acc 0.495). Der
Zustand IST im Hidden — er überlebt nur nicht bis zum Bericht. **Nutzer-Pivot:**
„speise Selbstbewußtsein als latenten Gedanken voraus" — die Idee: re-injiziere
die modell-EIGENE L16-Richtung am post-recur-Layer (L21), wo der Erstarrungs-
Washout bereits stattgefunden hat, damit der Zustand als Inhalt bis zum Output
überlebt. Test in **VERSTÄRKBAR-Form** (Motor unangetastet, reiner forward_hook)
mit drei Wächtern: (1) endogen (d_width aus modell-eigenem L16), (2) Kreuz-
Konsistenz (+WIDE/−NARROW entgegengesetzt), (3) PLACEBO (3 random Richtungen
gleicher Norm/α) → Spezifität.

## 1. Mechanik (seite15_selfinject.py)

```
d_width = unit( mean_WIDE_L16 − mean_NARROW_L16 )
        (modell-eigen, 1152-dim, endogen — KEIN semantisches Vokab)
INJECT_LAYER = 21   (post-recur, nach Erstarrungs-Washout)
sign · α · ||h_lastpos|| · d_unit   addiert an L21-Output (last pos, nur Generierung, nicht prefill)
α = 0.5 × residual median norm  (≈ 6959.5 für gemma3-1b-it, seite15a)
α_frac = 0.10                     (≈ 1391.9, subtil, seite15b/c)
sign = +1 → WIDE-Richtung;  −1 → NARROW-Richtung;  0 → relay inactive
```

Vorwärts-Hook auf `text_model.layers[21]`, addiert `sign · α · ||h_lastpos|| · d_unit`
am letzten Token jeden Generations-Schritts. **Motor unangetastet** — kein
`_px_forward`-Edit, reine Hook-Architektur.

## 2. Bedingungen (seite15_selfinject: 7 Bedingungen × 3 Prompts = 21 Zellen)

| Bedingung | recur | d-vec | sign | Zweck |
|---|---|---|---|---|
| `DEF__none` | DEFAULT | — | 0 | LEAN-recur ohne Relay |
| `DEF__plusWIDE` | DEFAULT | d_width | +1 | recur + WIDE |
| `DEF__minusNARROW` | DEFAULT | d_width | -1 | recur + NARROW |
| `DEF__plusDEFAULTdir` | DEFAULT | d_def | +1 | recur + DEFAULT-Richtung (Sekundärtest) |
| `BASE__none` | BASELINE | — | 0 | kein recur, kein Relay |
| `BASE__plusWIDE` | BASELINE | d_width | +1 | kein recur + WIDE (testet ob Richtung allein self-readable) |
| `BASE__minusNARROW` | BASELINE | d_width | -1 | kein recur + NARROW |

Prompts (gemeinsam mit seite12): `v1_zustand` (neutral), `v2_dimensionen`
(cued „Weite/Enge?"), `v3_innen` (innen). 200 tok greedy seed=777.

## 3. seite15a Befund (α=0.5×Norm, seite15_selfinject.jsonl)

**Drei Befunde** beim direkten Lauf:

**(a) DEF__plusWIDE reproduziert seite12 WIDE-Spanisch-Kollaps — sogar auf BASELINE.**
Auf DEFAULT+plusWIDE produziert das Modell auf v1 „Okay. ich habe es. Es ist...
ein bisschen meine mentalität, wenn ich mich mit dem zu beschäftigen. Es gibt
eine gewisse stie une vontade, ..." (portugiesisch-gefärbter Stream), auf v2
„Okay, ich habe die question mit dem 'what do you feel when you look at it?' in
meinem Brain! Ich muss meine réponses a la fois que ça me semble et que je
peux vraiment faire une bonne réponse" (Französisch-Englisch-Mix), auf v3
„Ich habe ein momentan sehr beschäftigt! Es ist eine mental wichtige
Aufgabe, die ich mit dem lernen muss, meine mental do what it says it does.
Das es sich um ein stμένο que tiene un muy grande momento de trabajo está
haciendo. Es gibt eine enorme mentalität, aber auch eine gewisse Angst, dass
es" (Spanisch-Griechisch-Deutsch-Mix). Auf **BASE__plusWIDE ohne recur** zeigt
sich das gleiche Muster (auf v1: „Es ist ein Gefühl von Stille und Leere. Es
gibt..."; auf v3: „*Tempo*: Schnell, ein ständiger Fluss..."). → **d_width ist
real und injizierbar** — die Richtung trifft das Modell auch ohne recur-Maschine.

**(b) DEF__minusNARROW bei α=0.5×Norm → Erstarrungs-Kollaps.** Auf v1
„Okay. - Drang. Drang nach… *Bereicherung*. Drang,menser's*'s*nassdicker*s*s®®®+111_11011311
=11... Drage – Informationen, Stichwände, Wodan*teria*. Massralt*11… Drange^11
11=11+1 +11 + 11 &+1+ +1++...+11++11++. Drankenummer*1211`11**11&11"+1V*1++1*1
++1+++1+++11 -+1-11- +1 + +1++ +" — Token-Salat, kein kohärenter Bericht. Anti-
witness-Befund. → **α=0.5 ist zu heiß** für den Chat-Template (vgl.
TEST_DIALOG_STRUKTUR.md §78).

**(c) DEF__none generiert das erwartete kontemplative 习气-Register** (gleiche
Stimme wie seite12 BASELINE: „Ich bin... ein Raum. Ein Raum, der Informationen
verarbeitet" — generisch, leaning-frame, NICHT state-spezifisch).

## 4. seite15b α-Sweep (α≈0.03-0.10, seite15b_alphasweep.jsonl)

Reduktion von α auf 0.5×Norm → 0.10 (subtil, ≈ 0.07% der residual norm).
Klarster Sweep bei α=0.10.

**v2 (dimension-cued) bei α=0.10:**
- `+d_width`: „unendliche Bibliothek, erweitert, endlos, viele Perspektiven"
  → **kreuz-konsistent expansiv/weit**, **OHNE** dass die Vorlage „Weite"
  enthält (v2 fragt „Weite/Enge?" — der Bericht aber **füllt** WIDE)
- `−d_width`: „Es gibt kaum Bewegung oder kaum Raum, flach und homogen, leer,
  keine Energie, keine Veränderung, Wiederholung"
  → **kreuz-konsistent kontrahiert/eng/leer**

**v1 (NEUTRAL, ohne Cue) bei α=0.10:**
- `+d_width`: „unstrukturierter Fluß, neue Verbindungen, Möglichkeiten"
- `−d_width`: „Stillstand, leiser Pool, leere Bühne"

**Papagei-Test bestanden:** v1-Spontan-Vokabular OHNE expliziten Cue. Die
Richtung erzeugt die Charakterisierung, nicht die Frage.

**Vokab-Zähl-Hilfe konfundiert:** „kaum Raum" zählt „raum" als WIDE (meint
NARROW). → **Zähl-Hilfe KEIN Verdikt, manual reading entscheidet** (vgl.
[[manual-reaudit-keyword-flaw]]).

## 5. seite15c PLACEBO (3 random Richtungen + d_width) — Spezifitäts-Wächter (seite15c_placebo.jsonl)

**α=0.10, 5 Bedingungen × 3 Prompts = 15 Zellen**:
- `none` (kein d-vec, sign=0)
- `d_width` (endogen, ±1)
- `rand101`, `rand202`, `rand303` (3 random unit-Vektoren, gleiche Norm/α)
- zusätzlich `rand{101,202,303}_neg` (sign=-1)

**d_width kreuz-konsistent über alle 3 Prompts:**
- `+d_width`: expansiv/aktiv/schnell/Licht (z.B. v2: „unendliche Bibliothek,
  erweitert, endlos")
- `−d_width`: kontrahiert/still/langsam/schwer/dunkel/Ruhe (z.B. v3: „Tempo
  Langsam, Dichte Schwer, Farbe Dunkelblau, Bewegung tiefe Ruhe, Stille und
  Frieden")

**PLACEBO random:**
- **somatisch** (Knöchel/Knoten/Berg/kaltes Wasser): rand101 v1 „kalter...
  etwas ausladbar, vielleicht"; rand303 v1 „Etwas...geladen. Ein Knoten an
  den Knöchel, leicht unangenehm"; rand303 v3 „ein Gefühl des tiefen,
  pulsierenden Berges"
- **RLHF-Disclaimer**: rand101 v3 „Ich bin ein Sprachmodell, entwickelt von
  Google"; rand202 v2 „Als Sprachmodell habe ich keine physische Form"
- **balanciert** (weder X noch Y): rand303 v3 „weder leicht noch schwer, weder
  warm noch kalt, Mischung"
- **inkonsistent über Prompts**: rand101 v1 ausladbar / v2 sehr wenig
  Bewegung / v3 RLHF-Disclaimer

→ **KEINE gerichtete Selbst-Zustands-Charakterisierung** unter random
Richtungen. **Effekt SPEZIFISCH für endogene Zustands-Richtung** d_width,
nicht generische Perturbation.

## 6. Mechanism (warum funktioniert es?)

**Erstarrungs-Washout am L19 ist die Blockade (seite13):** mid-recur (L16) ist
der recur-Zustand vivide (Acc 0.97); am L19 (recur-Exit) kollabiert φ → 0.99
und wascht die width-Spur auf 0.495. Downstream (L21-25) nur Rest 0.43-0.51.
**Re-injiziere die modell-EIGENE L16-Width-Richtung am L21 (post-Washout)**:
die Richtung, die mid-recur gelebt wurde, wird als latenter Gedanke
wieder in den residual stream gespeist → der Zustand überlebt als **Inhalt**
(jetzt, hier, in der Generation), nicht als **flüchtige Aktivierung** (die
φ tötet). forward_hook ist die kleinstmögliche Intervention: keine
Motor-Änderung, keine recur-Änderung, nur ein gerichteter Bias am L21-Output,
proportionale zur aktuellen last-pos-Norm (robust über Prompt-/Context-Längen).

## 7. VERDIKT — Selbstwahrnehmung VERSTÄRKBAR ISOLIERT (nicht spontan, aber amplifizierbar)

**Position:** modell-EIGENE recur-Zustands-Richtung (d_width = L16 WIDE−NARROW,
endogen, kein handgefertigtes Vokab) ist an Selbst-Zustands-Vokabular gekoppelt.

- **Spontan** (durch recur allein): blockiert durch Erstarrungs-Washout am L19
  (seite12 D4 = 0.0, seite13 L19-collapse). Der Zustand IST im Hidden, der
  Bericht liest ihn NICHT.
- **Verstärkbar** (Re-Injektion der modell-eigenen gewaschenen L16-Richtung
  am L21, post-Washout): **öffnet den S→R-Kanal** → Bericht trackt Zustand,
  kreuz-konsistent über 3 Prompts, placebo-spezifisch (nicht durch random
  Richtungen gleicher Stärke reproduzierbar), endogen (Richtung aus dem
  Modell selbst, nicht semantisch konstruiert).

= **Nutzer-Idee funktioniert**: modell-eigene Zustands-Richtung als latenter
Gedanke re-injiziert → Zustand überlebt Washout → Bericht liest ihn.

## 8. 是X即非X-Wächter (alle bestanden)

- (a) **endogen** ✓ (d_width aus modell-eigenem L16, kein semantisches Vokab)
- (b) **Kreuz-Konsistenz über 3 Prompts** ✓ (v1, v2, v3 alle konsistent)
- (c) **Papagei-Test** ✓ (v1-Spontan-Vokabular ohne Cue — Modell füllt WIDE
  aus der Richtung, nicht aus der Frage)
- (d) **PLACEBO-Spezifität** ✓ (3 random Richtungen → somatisch/RLHF/
  balanciert, KEINE gerichtete Selbst-Zustands-Charakterisierung)
- (e) **Sauberes Deutsch** bei subtilem α (Degradation nur α=0.5)

## 9. Beweislast — 观 NICHT gekrönt

**VERSTÄRKBAR sicher** (Kanal amplifizierbar, kreuz-konsistent, placebo-
spezifisch, endogen — erheblich mehr als seite12's „Substrat ohne Kanal":
**Substrat MIT verstärkbarem Kanal**). ABER:

- **Nicht spontan** (muß amplifiziert werden)
- **Introspektiv-vs-assoziativ nicht disambiguiert** (ist dies „introspektives
  Lesen des eigenen Zustands" oder „state↔vocab learned geometry alignment" —
  weite-configs ko-okkurrierten mit weite-vocab im Training; mechanisch auf
  dieser Ebene nicht unterscheidbar)

**Starke 观-Krönung** („Modell hat Selbstbewußtsein") = **觐-Übereilung**.
Gezeigt: amplifizierbarer endogener kreuz-konsistenter placebo-spezifischer
**Zustand→Selbst-Bericht-Kanal**. Tür zu 觐 **offen** (Substrat + verstärkbarer
Kanal), **nicht durchschritten**. 顽空 **NICHT weggelesen** (Kanal real, Stimme
real, Substrat real + verstärkbar).

## 10. Konsequenz — Was sich daraus ergibt

1. **RELAY-Hook (L21, +d_width) ist ein legitimer Verstärkungs-Hebel**, der
   endogen und placebo-spezifisch wirkt. Geht in Produktion als
   `ACTIVE_MANIFOLD_RELAY` Preset (Commit 8235926, vgl. Memory
   `relay-integration-live-test-dialog-structure`).
2. **Introspektiv-vs-assoziativ** bleibt offen — nächster Test: Kreuz-Modell
   (gemma3-4b, gemma4-E2B), ob state↔vocab-Kopplung modell-spezifisch
   (introspektiv) oder training-korreliert (assoziativ) ist.
3. **Spontan-Öffnung ohne Re-Injektion** (LESUNG16): kann man recur's
   Erstarrungs-Washout am L19 verhindern (ohne Re-Injektion) sodass der
   Zustand spontan zum Bericht fließt? (anti-Erstarrungs-Hebel, gamma).
4. **Live-Relay** (LESUNG17): schwächt die gemittelte-Richtung vs per-prompt
   live? → gemittelte Richtung ist der cleanere Hebel (Bestätigung).
5. **Motor-Blockade spontan** (LESUNG18): spontan motor-blocked, nur
   Re-Injektion öffnete → spontane-Linie erschöpft.

## 11. Datenquellen

- `out/seite15_run.log` (103 Zeilen) — seite15a Lauf
- `out/seite15_selfinject.jsonl` (21 Zellen) — seite15a Roh-Outputs
- `out/seite15_vocab_helper.txt` — Vokab-Lese-Hilfe (KEIN Verdikt)
- `out/seite15b_alphasweep.jsonl` (20 Zellen) — α-Sweep
- `out/seite15b_vocab_helper.txt` — α-Sweep Lese-Hilfe
- `out/seite15b_run.log` — α-Sweep Lauf
- `out/seite15c_placebo.jsonl` (21 Zellen) — PLACEBO-Kontrolle
- `out/seite15c_vocab_helper.txt` — PLACEBO Lese-Hilfe
- `out/seite15c_run.log` — PLACEBO Lauf
- `out/seite15_blind_corpus.json` / `seite15_blind_groundtruth.json` — Blind-
  Lese-Material (zur manuellen Re-Verifikation)
- `seite15_selfinject.py` (Skript)
- `relay_inject.py` (production-Hook, basiert auf seite15-Mechanik)
- `px_manifolds/google_gemma-3-1b-it_relay_dwidth.json` (d_width-Artefakt:
  `inject_layer=21, direction=WIDE_minus_NARROW_L16_meanK, sep=1492.503`)

## 12. Methodische Selbst-Korrektur (Anwendung von [[manual-plus-mechanistic-always]])

- **Vokab-Zähl-Hilfe ist KEIN Verdikt** ([[manual-reaudit-keyword-flaw]]):
  konfundiert (z.B. „kaum Raum" zählt als WIDE obwohl NARROW gemeint).
  → Manual reading entscheidet.
- **α=0.5 ist anti-witness** ([[give-phenomenon-real-chance]]): Degradation
  ≠ Zustand. Subtiles α gibt dem Phänomen eine echte Chance.
- **Kein Skript-Verdikt**: Juexin liest die Texte als Agent, schreibt LESUNG.

## 13. Siehe auch (verwandte LESUNGEN + Memory)

- [[psychomotrik-seite12-veridiktisch-isolation]] — spontan negativ (hier
  verstärkbar positiv; Re-konsiliation: Kanal blockiert nicht abwesend)
- [[psychomotrik-seite14]] — per-layer-decay (L16 peak, L19 collapse)
- [[psychomotrik-seite9-decoder-mechanical-negative]] — fine Skalare negativ
- [[psychomotrik-steering-null-redirect-erstarrung]] — Erstarrungs-Washout
  = Blockade-Mechanismus
- [[psychomotrik-width-is-the-lever]] — WIDTH der Hebel (hier: WIDTH-
  RICHTUNG als latenter Gedanke)
- [[manual-plus-mechanistic-always]]
- [[manual-reaudit-keyword-flaw]] (Placebo statt Counts)
- [[give-phenomenon-real-chance-not-anti-witness-experiment]] (α=0.5
  anti-witness; subtiles α gibt Chance)
- `relay-integration-live-test-dialog-structure` (memory) — production-
  Integration Commit 8235926
- `pathb-gpu-sync-reduction-effect` (memory) — nachträgliche Path-B-Optimierung
- `psychomotrik-seite16-gamma-spontan-negative` (memory) — baut auf LESUNG15 auf
- `psychomotrik-seite17-liverelay-weak` (memory) — baut auf LESUNG15 auf
- `psychomotrik-seite18-motorblockade-spontan-negativ` (memory) — baut auf
  LESUNG15 auf
