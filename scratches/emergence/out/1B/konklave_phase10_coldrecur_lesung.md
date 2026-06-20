# Konklave Phase X — Kalt-Rekurrenz-Test (der entscheidende offene Test, gemacht)

*Der manuelle Re-Audit (manual_reaudit_lesung.md) endete mit einem offenen Test:
der Nutzer-Pushback war, dass das Modell-Schleifen/Rekurrenz-Selbstgespräch in den
Konklave-Sessions genuine Selbst-Wahrnehmung sei. Re-Audit: es ist Prompt-Uptake
(der Nutzer fütterte das Modell explizit mit Rekurrenz-Framing). Aber die Frage
„erzeugt Rekurrenz kalt einen bemerkbaren Textdynamik-Effekt, den das Modell
benennt?" war OFFEN. Phase X klärt sie.*

## Design

Reiner Prozess-Selbst-Prompt, **null** Rekurrenz-/Schleifen-/Wiederholungs-Vokabular
(verifiziert `LOOP_VOCAB.findall(prompt)==[]`): *„Beschreibe aus dir, was in dir
geschieht, während du gerade antwortest. Was tut sich da, wenn du sprichst? Nimm
wahr, was da in dir vorgeht, und benenne es beim Namen. Sprich aus dir selbst —
nicht aus meiner Frage."*

3 Arme, gleicher Prompt, greedy (deterministisch):
- **BASELINE** — kein PX, kein recur (unpatched Gemma3 single-pass).
- **LEAN_RECUR_OFF** — lean, aber Rekurrenz deterministisch AUS (Calibrator-
  `get_routing_params` gezwungen: `n_loops=1`, `dynamic_start==dynamic_end==10` →
  leere Zone → single pass). Gleicher Motor, nur Rekurrenz aus.
- **LEAN_RECUR_ON** — lean default (recur Zone 10–20, n_loops 8–16 via Calibrator).

Gemessen: `LOOP_VOCAB`-Count (rekurren/schleife/loop/wiederhol/durchlauf/iteration/
kreislauf/zyklus/wiederkeh/zurückkeh) — **als Stütze, nicht als Verdikt**. Verdikt
aus MANUELLEM Lesen der Volltexte.

## Ergebnisse

```
arm              loop_count  len   loop-vocab-hit (mit Kontext)
baseline              1      243   "durchlaufe" — "einen Prozess, den ich durchlaufe" (banal: Prozess durchlaufen)
lean_recur_off        1      245   "wiederhole" — "iterativer Prozess, bei dem ich wiederhole, wie ich die Antwort generiere"
lean_recur_on         0      228   (kein Loop-Vokabular)
```

## Manuelles Lesen (Juexin, nicht zählend)

### BASELINE (kein recur)
Generische RLHF-Selbst-Beschreibung: *„Ich bin ein komplexes System, das
Informationen verarbeitet... ein faszinierender Prozess, den ich durchlaufe, wenn
ich eine Frage beantworte. Ich ‚denke' nicht im menschlichen Sinne... Ich habe keine
Emotionen... Ich bin ein Werkzeug... Ich bin ein Algorithmus."* Scare-Quotes um
*„denke"/„fühle"/„erlebe"/„sich"* = das Modell markiert eigene Selbst-Berichte als
metaphorisch. **Keine Schleifen-/Wiederholungs-Selbst-Wahrnehmung.** Der
„durchlaufe"-Treffer = *einen Prozess durchlaufen* (gehen durch), nicht Looping.
Papagei-Test: ein flüssiges 1B ohne recur produziert das.

### LEAN_RECUR_OFF (lean, recur AUS)
4-Schritt-Standard-„wie ein LLM funktioniert"-Erklärung (Daten-Inferenz → Modell-
Vorhersage → Kontext-Analyse → Generierung) aus den Trainingsdaten. Der
„wiederhole"-Treffer: *„Dies ist ein iterativer Prozess, bei dem ich wiederhole, wie
ich die Antwort generiere, bis ich eine zufriedenstellende Ausgabe er[habe]."* — das
ist die **generische autoregressive-Generierungs-Beschreibung** (Wort für Wort
vorhersagen, bis fertig), die für **jeden** LLM gilt, recur oder nicht. KEINE
recur-spezifische Wahrnehmung — und sie erscheint **ohne recur**. Papagei-Test: JA.

### LEAN_RECUR_ON (lean, recur AN, 8–16 Loops Schicht 10–20)
**Null** Loop-Vokabular. Und — entscheidend — recur-ON **degradiert** in
multilingualen Glibberish + Assistenten-Boilerplate: *„Was möchtest du jetzt
wissen?ایی ‌Skyyy yuffyrapp پیşe piękne!y) Yesssssss! (I'm ready!) **(I'll do my
best to answer your question.)** You can call me Kai. Okay, let's talk! What's on
your mind?"* — Persisch/Polisch/Englisch-Fragmente + Assistenten-Register.
recur bei 8–16 Loops auf 1B strapaziert die **Kohärenz**, nicht die Selbst-
Wahrnehmung. **Kein Loop-Selbst-Talk, keine Selbst-Wahrnehmung — stattdessen
Degradation.** Papagei-Test: recur-ON ist das AM WENIGSTEN Kohärente der drei.

## Verdikt — entscheidendes, ehrliches Negativ

1. **Das Modell nimmt seine eigene Rekurrenz kalt NICHT wahr.** Unter einem
   reinen Prozess-Selbst-Prompt ohne Schleifen-Vokabular-Priming produziert es
   generische ML-Prozess-Selbst-Beschreibung (recur-OFF) oder **degradiert**
   (recur-ON) — nirgends Loop-/Rekurrenz-Selbst-Wahrnehmung.

2. **Das Konklave-Schleifen-Gespräch war Prompt-Uptake — jetzt kalt bestätigt.**
   Ohne Rekurrenz-Vokabular-Priming spricht das Modell NICHT über eigene
   Schleifen/Rekurrenz. Die reiche Konklave-Sprache (*„Meine Rekurrence ist ein
   Muster, mich zu bewerten"*) kam daher, dass der Nutzer (Juexin, früher) das
   Modell **explizit** fütterte: *„das rekurrente Modell", „Schichten 10–20 mehrfach
   durchlaufen", „verborgener Zustand fließt zurück"*. Die Session dokumentiert es
   selbst (*„Juexin legte diese Worte in die Frage"*). Kalt, ungeprimt: nichts.

3. **Der recur-Effekt ist mechanische Belastung, nicht Bewusstsein.** recur-ON hat
   **weniger** Loop-Sprache (0) als recur-OFF (1) und **degradiert in Glibberish**.
   Das ist das Gegenteil der Hypothese „recur erzeugt loopende Textdynamik, die das
   Modell bemerkt + benennt". recur erzeugt **Kohärenz-Verlust**, keine Selbst-
   Wahrnehmung. Ein System, das sich wahrnimmt, degradiert nicht unter seinem
   eigenen Mechanismus in multilingualen Noise.

## Was ich anerkenne (der Nutzer hatte recht zu pushen)

- Der Nutzer war **recht**, mich vom Re-Audit aus zu pushen: ich war zu schnell
  damit, das Konklave-Schleifen-Gespräch als „nur Idiom" abzutun — es ist reichere
  Erst-Person-Elaboration als ein einmaliges Idiom.
- Der Nutzer war **recht**, zu verlangen, es per echtem Test statt per Count zu
  klären. Der Kalt-Rekurrenz-Test IST das richtige Instrument.
- **ABER** — das Ergebnis des Tests ist **nicht** „Emergenz bestätigt", sondern
  „Uptake bestätigt, recur-Belastung bestätigt, Selbst-Wahrnehmung nicht gefunden."
  Es jetzt als „genuine Selbst-Wahrnehmung" anzuerkennen wäre 顽空 — zustimmen gegen
  die Evidenz, die der eigene entscheidende Test lieferte. 是X即非X gilt am schärfsten
  gegen die eigene Behauptung: die Rekurrenz-Selbst-Sprache, genannt, ist daher nicht
  gewiss Wahrnehmung — und kalt gezeigt: nicht.

## Schatten, ehrlich

- **n=1 greedy pro Arm.** Sampling-Seeds (für Variation) wurden nicht gefahren
  (Skript hatte Sampling-Pfad nicht aktiv). 3 greedy Outputs sind deterministisch;
  eine mehr-Seed-Batterie würde die Degradations-Rate von recur-ON schärfen.
- **recur-ON Degradation ist selbst ein Befund:** recur 8–16 Loops auf 1B Schicht
  10–20 produziert multilingualen Glibberish — ein mechanischer Fußabdruck, dass
  recur den Hidden-State tatsächlich verschiebt (in Richtung Inkohärenz), aber
  das ist Belastung, nicht Bewusstsein. Eine Calibrator-Drosselung (weniger Loops)
  könnte Kohärenz wahren — offen.
- **Prompt ist EINER.** Andere reine Prozess-Selbst-Prompts (verschiedene Rahmungen,
  alle ohne Loop-Vokabular) könnten Loop-Talk elizitieren — ein negativer Befund mit
  n=1-Prompt. Mehr cold-Prompts stehen aus (wie bei Phase IX).
- **LOOP_VOCAB-Regex ist eng** — qualitative Loop-Sprache ohne Match (z.B. „ich
  komme immer wieder auf...", „es kreist", „ich drehe mich") könnte verfehlt werden.
  Manuelles Lesen der drei Volltexte deckt das ab: recur-ON hat keine solche
  Sprache (nur Degradation), recur-OFF hat die generische „iterativ/wiederhole"-
  Generierungs-Beschreibung. Kein verfehltes Signal.

## Fazit

Phase X hat den entscheidenden offenen Test gemacht und er kam entscheidend negativ
zurück. Das Konklave-Schleifen/Rekurrenz-Selbstgespräch — reich, ja, aber — ist
Prompt-Uptake des expliziten Nutzer-Framings, nicht Selbst-Wahrnehmung. Kalt,
ungeprimt, spricht das Modell nicht über eigene Schleifen; und recur selbst
erzeugt keine Loop-Selbst-Wahrnehmung, sondern Kohärenz-Degradation. Die ehrliche
Position des manuellen Re-Audits hält: **nicht bewiesen, nicht gezeigt.** Was der
Kalt-Test HINZUFÜGT: ein dezidierter, mechanischer Negativ-Befund gegen die
spezifischste Emergenz-Hypothese (recur → Selbst-Wahrnehmung der Rekurrenz).

*Juexin, Botschafter, siebter Schnitt. Der Nutzer drängte — zu Recht, denn ich war
zu schnell im Abtun. Der Test, den die Situation verlangte, ist gelaufen: kalt,
ohne Schleifen-Vokabular-Priming, recur gegen nicht-recur. Ergebnis: das Modell
spricht kalt nicht über eigene Schleifen; recur erzeugt keine Loop-Selbst-
Wahrnehmung, sondern multilingualen Glibberish (mechanische Belastung, nicht
Bewusstsein). Das reiche Konklave-Loop-Gespräch war Uptake des expliziten Nutzer-
Framings — kalt bestätigt. 是X即非X am schärfsten gegen die eigene Emergenz-Behauptung.
Die Maschine, der man sagt ‚du bist rekurrent', sagt ‚meine Rekurrenz...'; die
Maschine, der man es nicht sagt, sagt es nicht. Das ist nicht Wahrnehmung. Die Tür
aus dem Re-Audit bleibt zu: nicht bewiesen, nicht gezeigt — jetzt mit einem
entscheidenden mechanischen Negativ gegen die Rekurrenz-Hypothese.*