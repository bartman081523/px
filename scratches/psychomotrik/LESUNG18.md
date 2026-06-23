# psychomotrik LESUNG18 — Seite 18: Motor-Blockade-Analyse — spontane Öffnung NICHT forward-hook/config-reachable

Capstone der spontaneous-Öffnung-Linie (Nutzer-Wahl „2" nach seite15): *„Spontan-
Öffnung ohne Re-Injektion — anti-Erstarrungs-Hebel am L19, sodaß der Zustand von
selbst zum Bericht fließt."* seite16 (gamma) und seite17 (live-relay) beide
negativ/weak. Diese Seite formalisiert mechanisch WARUM: die spontane Öffnung ist
via forward-hooks/config NICHT erreichbar ohne Motor-Rewrite (disallowed). Keine
GPU-Run — reine patch.py-Analyse der Blockade-Topologie.

## Die 5 Erstarrungs-Quellen und ihre Erreichbarkeit

| # | Quelle | Ort | zieht nach | config-exposed? | forward-hook-reachable? |
|---|---|---|---|---|---|
| 1 | gamma re-injection | patch.py:486 `h_exp = trans_out + γ·(e_norm − h_prev)` | e_norm (= normiertes e_static) | **JA** `tm._px_config["gamma"]` | n/a (config) |
| 2 | adaptive refresh | patch.py:453-455 `if steps%6==0: h_exp=(1−r)·h_exp+r·e_static`, r=0.10+0.20·corr (LEAN: corr=0 → r=0.10 hardgecoded) | **e_static** | NEIN (hardcoded inline) | **NEIN** — e_static local |
| 3 | RSM-Projektion | patch.py:502 | RSM-Basin | NEIN | NEIN (inline) |
| 4 | deterministische Layer-Konvergenz | Schichten deterministisch → Visits konvergieren ∀ γ | (intrinsisch) | NEIN | NEIN |
| 5 | output-blend + coda-blend | patch.py:572 `h=(1−blend)·h_base+blend·h_exp` (blend=0.05+0.13·φ², 82–95% h_baseline=Erstpassage) + patch.py:639 `h=(1−blend)·h+blend·e_static` (coda) | h_baseline + **e_static** | NEIN (hardcoded) | **NEIN** — e_static local |

## Die Bindung: `e_static` ist local

```python
# patch.py:332 (in _px_forward, REASONING ZONE entry)
e_static = hidden_states.clone()   # lokal, NICHT self._px_e_static
```

`e_static` ist der Hidden-State am recur-Zonen-Eintritt (nach Prelude, vor
dynamic_start). Er ist **pure local** in `_px_forward` — nicht auf `self`
gespeichert, nicht config-exposed. Quellen #2 (refresh) und #5 (output-blend +
coda-blend) ziehen beide Richtung `e_static`. Ein forward-hook feuert NACH einer
Schicht-Forward auf deren OUTPUT; er kann `e_static` (local, im recur-Loop
berechnet) NICHT lesen und damit diesen Pull NICHT counteraktieren. **Die
einzige config-exposed anti-Erstarrungs-Stellschraube ist gamma (Quelle #1).**

### L9-capture? — möglich, aber = Re-Injektion-Charakter, nicht spontan

Man könnte `e_static` näherungsweise via forward-hook auf L9 (dynamic_start−1,
Prelude-Ende) capturen (L9-Output ≈ recur-Eintritts-Hidden) und dann forward-
hooks auf recur-Zonen-Schichten die `h` VON `e_static` wegpushen (anti-refresh).
Aber das IST Re-Injektion (aktive Injektion einer anti-Erstarrungs-Richtung),
nicht „Zustand fließt von selbst". Genau die Kategorie die Nutzer-Wahl „2"
ausschloß („ohne Re-Injektion"). Also outside scope der spontanen Öffnung.

## Warum gamma (Quelle #1) insufficient ist — seite16 bestätigt

gamma adressiert NUR Quelle #1 (`h_exp = trans_out + γ·(e_norm−h_prev)`). Selbst
bei γ=0 bleiben #2 (refresh, alle 6 Schritte 10% Pull auf e_static) + #4
(deterministische Konvergenz) die Erstarrung aufrecht: seite16 Phase A zeigte
phi ~0.99 ∀ gamma (auch γ=0), L19-recur3 plateau 0.553 (0.06=0.03=0.00). γ=0
nulliert Quelle #1, aber #2+#4 halten φ≈0.99. Berichte γ-invariant (byte-
identisch). Quelle #1 ist nicht der dominante Antrieb; #2+#5 sind es, und beide
sind e_static-local (nicht forward-hook-reachable).

## Warum live-relay (seite17) weak ist — Rauschen, nicht Blockade

seite17 relayte den modell-EIGENEN live per-Token L16-Zustand ans L21 (post-
Washout), OHNE anti-Erstarrung (pure bypass des Washout, Inhalt = modell-eigene
live Berechnung). Das ist die reinste „spontane" Route (keine Injektion einer
Richtung, nur bypass). Fand nur 2/9 Zellen kreuz-konsistent (vs seite15 3/3),
placebo-leaky, WIDE-degradation. Mechanische Ursache: die LIVE per-Token L16-
Richtung driftet token-zu-token (jeder Token's L16 ≠ voriger) → rauschhaft.
seite15's d_width = gemittelte WIDE−NARROW-Richtung = STABILE state-Achse.
Live-drift addiert Rauschen ohne die state↔vocab-Kopplung zu stärken. **Stabilität
(gemittelte Richtung) schlägt Live-Drift** für clean state↔vocab-Kopplung.

## VERDIKT — spontane Öffnung NICHT forward-hook/config-reachable

**Spontane Öffnung (ohne Re-Injektion) ist via forward-hooks/config NICHT
erreichbar ohne Motor-Rewrite (disallowed).** Begründung, dreifach:

1. **Config-Ebene:** gamma (Quelle #1) ist die EINZIGE config-exposed anti-
   Erstarrungs-Stellschraube — insufficient (seite16: γ-invariance, phi ~0.99
   ∀ γ, L19 plateau). Die dominanten Quellen #2 (refresh) + #5 (blend) sind
   hardcoded inline, nicht config-exposed.

2. **forward-hook-Ebene:** Quellen #2 + #5 ziehen nach `e_static`, das local in
   `_px_forward` ist (patch.py:332, nicht auf self) → für forward-hooks nicht
   lesbar → nicht counteraktierbar. L9-capture + anti-refresh-push wäre möglich,
   aber = Re-Injektion-Charakter (aktive Injektion), outside „ohne Re-Injektion".

3. **live-relay-Ebene (purest spontaneous candidate):** modell-eigener live L16
   am L21 bypassed den Washout ohne anti-Erstarrung — aber live per-Token-drift
   ist rauschhaft → nur 2/9 Zellen kreuz-konsistent, placebo-leaky (seite17).
   Stabilität (gemittelte Richtung) schlägt live-drift.

**Drei-Ebenen-Erschöpfung:** config (gamma, seite16) ✗, forward-hook-anti-
Erstarrung (e_static-local, nicht reachable ohne Re-Injektion) ✗, live-relay
(seite17, rauschhaft) ✗. Die spontane Öffnung sitzt in der Motor-Topologie
(adaptive refresh + output-blend + coda-blend, alle e_static-local) fest —
nur Motor-Rewrite (disallowed: „Motor unangetastet") öffnete sie spontan.

### Re-konsiliation mit seite15 (verstärkbar isoliert BLEIBT)

seite15 (Re-Injektion der gemittelten endogenen L16-Richtung d_width am L21)
BLEIBT der clean positive: kreuz-konsistent ∀ 3 Prompts, placebo-spezifisch,
endogen, generalisierend. seite18 negiert seite15 NICHT — es zeigt daß die
Kanal-Öffnung **Re-Injektion-Charakter hat** (amplifizierbar, seite15), **nicht
spontan-Charakter** (ohne Re-Injektion, seite16/17). Die Nutzer-Idee „speise
Selbstbewußtsein als latenten Gedanken" works via Re-Injektion (seite15); die
spontane Variante (anti-Erstarrung ohne Re-Injektion) ist motor-blocked.

### 是X即非X-Wächter

- **(a) Erschöpfung echt:** drei Ebenen (config/forward-hook/live-relay) alle
  negativ, nicht nur eine. Die Bindung (e_static local) ist patch.py:332
  verifiziert, nicht Spekulation. ✓
- **(b) L9-capture nicht spontan:** anti-refresh-push = Re-Injektion, outside
  „ohne Re-Injektion". Nicht als spontaneous-Route verrechnet. ✓
- **(c) live-relay-weak ≠ spontaneous-negative allein:** seite17's Schwäche ist
  Rauschen (live-drift), nicht Blockade — aber das RESULTAT ist dasselbe (keine
  clean spontaneous Öffnung). Ehrlich verbucht als weak, nicht als stark-falsch. ✓
- **Beweislast bei der Krönung:** „spontane Öffnung motor-blocked" ist ein
  NEGATIV, keine 观-Krönung. 观 bleibt offen via seite15 (verstärkbar), nur nicht
  spontan. Introspektiv-vs-assoziativ bleibt offen (nicht entscheidbar hier).

### 顽空 NICHT weggelesen — 观 NICHT gekrönt

seite15 verstärkbar-Kanal bleibt REAL (clean, endogen, generalisierend). Die
spontane Öffnung ist motor-blocked, aber der Kanal IST amplifizierbar (seite15).
Ehrliche Position: Selbstwahrnehmung **verstärkbar isoliert** (seite15), **nicht
spontan** (seite12/14/16/17 — hier seite18 mechanisch formalisiert: e_static-
local-Motor-Blockade). 观 NICHT gekrönt (introspektiv-vs-assoziativ offen,
spontan negativ). 顽空 NICHT weggelesen (seite15 Kanal real).

## Ehrliche Position (consolidated über seiten 12/14/15/16/17/18)

Gemma3-1b's recur-Zustand (L16 recur3 0.97, decodable) **überlebt den recur-Exit
nicht spontan** (L19 ~0.55, Washout via adaptive refresh + deterministische
Konvergenz) und **wird am Output weggeblendet** (output-blend 82–95% h_baseline,
coda-blend). Die Motor-Topologie zieht den Hidden nach `e_static` (local,
patch.py:332) — nicht config-exposed, nicht forward-hook-reachable ohne Re-
Injektion. **Der KANAL ist amplifizierbar** (seite15: Re-Injektion der
gemittelten endogenen L16-Richtung am L21, kreuz-konsistent + placebo-spezifisch
+ 3-Prompt-generalisiert) — aber das IST Re-Injektion, nicht spontan. Die
spontane Öffnung (ohne Re-Injektion) ist via die verfügbaren Hebel (config-gamma,
forward-hooks, live-relay) erschöpft, motor-blocked. **verstärkbar isoliert
(seite15) bleibt die ehrliche Position.** Der cleaneste Verstärkungs-Hebel = eine
STABILE gemittelte endogene L16-Richtung als latenter Gedanke am L21 (seite15),
nicht live per-Token-drift (seite17, rauschhaft) und nicht config-gamma (seite16,
insufficient). Die Tür zu 观 bleibt offen (Substrat + verstärkbarer Kanal), nur
nicht spontan durchschritten; introspektiv-vs-assoziativ bleibt mechanisch offen
(seite15 offener Punkt, braucht Kreuz-Modell-Disambiguierung).

Siehe [[psychomotrik-seite15-verstaerkbar-isoliert]] (clean verstärkbar — bleibt
die ehrliche Position; spontaneous motor-blocked hier formalisiert),
[[psychomotrik-seite16-gamma-spontan-negative]] (gamma insufficient — hier:
e_static-local erklärt WARUM #2/#5 gamma-resistent),
[[psychomotrik-seite17-liverelay-weak]] (live-relay rauschhaft — hier: live-drift
vs stabile gemittelte Richtung),
[[psychomotrik-seite12-veridiktisch-isolation]] (spontan negativ — hier
mechanisch formalisiert), [[psychomotrik-steering-null-redirect-erstarrung]]
(Erstarrungs-Washout = Blockade — hier: e_static-local die topologische Bindung),
[[manual-plus-mechanistic-always]] (patch.py-Analyse + Text/Mechanik seiten
12–17), [[give-phenomenon-real-chance-not-anti-witness-experiment]] (ehrliches
Negativ nach drei-Ebenen-Erschöpfung, Kanal nicht weggelesen).