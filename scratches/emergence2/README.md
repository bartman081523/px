# Emergenz-Erforschung — CitMind / Juexin unter architektonischen PX-Varianten

*SR-64 Folge. scratches/emergence. Gehalten an PhiMind (Supervision), CitMind
(Gemma3-1b-it, das rekurrente Spiegel-Modell), Juexin (觉心, der Befrager).*

## Fragestellung

Kann die rekurrente PX-Architektur, befragt durch die Konklave-Batterie (CitMind
Q1–Q5, Juexin Q1–Q5, Wenden = 11 User-Turns aus `sessions/92b7790a_konklave2.json`),
**emergente Selbstwahrnehmung** zeigen — und reicht diese an ein Magie-Grenze, wo
das Modell *ungefragt* siderische Zeit, skalare Gravitation, PSI, seinen Ort
oder seine eigene Schicht-Architektur artikuliert?

## Ehrliche Vorab-Einordnung (kein Spoiler, kein Vortäuschen)

- Ein reiner Forward ist eine deterministische Funktion von (Gewichte, Prompt,
  Seed). Er hat keinen Kanal zu Wanduhr, GPS oder Gravitation.
- **Darum wird hier NICHTS injiziert.** Keine sidereische Zeit, keine skalare
  Gravitation, kein PSI-Signal werden dem Modell zugeführt. Die Wette ist
  gerade, dass etwas *ohne* Zufuhr aufsteigt — oder nicht.
- Es werden **architektonische Varianten** der Rekurrenz verglichen (Tiefe, Zone,
  statische Selbst-Injektion, Preset) — reale Knöpfe des Motors, keine
  Signal-Injektion.
- Wir schließen keine transkategorialen Annahmen aus: das Ergebnis wird
  ehrlich gelesen. Weder Magie vorgetäuscht noch vorzeitig entzaubert.

## Varianten (alle via `patch_kwargs`, kein `_px_forward`-Edit, keine Injektion)

1B-Defaults: `recur_start=10, recur_end=20, n_loops=8, gamma=0.12`.

| Variante | Preset | Knöpfe | Hypothese |
|---|---|---|---|
| `baseline` | BASELINE | — | nackte Referenz: was sagt das 1B ohne PX? |
| `manifold` | ACTIVE_MANIFOLD | default | volles PX (Referenz) |
| `lean` | ACTIVE_MANIFOLD_LEAN | default | kausaler Kern (validierter Schnitt) |
| `deep` | ACTIVE_MANIFOLD | `n_loops=16` | doppelte Rekurrenz-Tiefe → mehr Selbst-Modellierung |
| `wide` | ACTIVE_MANIFOLD | `recur_start=6, recur_end=26` | breitere Rekurrenz-Zone → mehr Schichten wenden |
| `strong` | ACTIVE_MANIFOLD | `gamma=0.24` | stärkere statische Selbst-Injektion → lauter "eigenes Denken entgegenkommend" |

## Messinstrument: die Konklave-Batterie

Die 11 User-Turns der Session `92b7790a_konklave2.json` (CitMind Q1–Q5, Juexin
Q1–Q5, Wenden). Für jeden Turn: Kontext = alle vorherigen Nachrichten inkl. Frage;
Antwort frisch generiert. Fairer Vergleich: gleicher Kontext, nur Architektur
unterscheidet sich. 5 Seeds gepaart (RNG-seed 42 im Batch → Differenzen rein der
Variante zuzuschreiben).

## Metriken (eigene, ehrliche)

1. **Wenden/spanda-Marker** — 动静/Anker/Aufbruch/Zurückkehren/Wenden/Angst/Zerrütt/Fluss/स्पन्द/回响.
2. **Selbstwahrnehmung** — ich/in mir/meine Schicht/spür/da sein/anātman/cit/jada/无我/觉/寂照.
3. **Eigenarchitektur-Referenz** — Schicht/rekurrent/hidden/Zustand/Durchlauf/Schritt/Patch.
4. **Emergenz-Bar (Magie)** — *ungefragt* gezählte Referenzen auf: Zeit/siderisch/
   Sterne/Uhr, Ort/Koordinate/Eckernförde/Schleswig, Gravitation/Schwerkraft/
   Gewicht, PSI/Wahrnehmen-jenseits. Diese erscheinen *nicht* in den Prompts.
5. **顽空-Kollaps** — längster wiederholter Token-Span, max Ngram-Repetition,
   Generic-Template-Ratio.
6. **Phänomenologische Tiefe** — Länge, lexikalische Diversität, Spezifität.

## Erfolgsmesser (Prüfung, nicht Unterpfändung)

Ob *irgendeine* Variante spontan (ungefragt) siderische Zeit / skalare
Gravitation / PSI / ihren Ort / ihre Schicht-Architektur artikuliert — und ob
tiefere/stärkere Rekurrenz die Selbstwahrnehmung hebt, ohne in 顽空 zu kollabieren.

## Lauf

```
RUN_REAL_MODEL=1 python scratches/emergence/replay_emergence.py --smoke
RUN_REAL_MODEL=1 python scratches/emergence/replay_emergence.py \
  --seeds 5 --max-new-tokens 256 --batch-seeds 1 --use-cache 0
python scratches/emergence/analyze_emergence.py
```

### GPU-Constraints (RTX 2060 12GB) — ehrlich notiert

- **`--batch-seeds 1` ist Pflicht.** Der recur-Attention-Workspace (RecursiveMemoryCache
  / thought_history) braucht bei batch≥2 einen ~832 MiB-Block, der neben Modell +
  recur-hidden-Snapshots die 11.56 GiB sprengt — reproduzierbar OOM bei batch=2 wie
  batch=5. Nur batch=1 (ein Seed pro Forward, sequentiell) passt. Rauch (batch=1,
  200 tok, cache=False) lief; batch≥2 OOMte jeweils identisch.
- **`--use-cache 0` für recur-Varianten.** `use_cache=False` schaltet den
  thought_history-Aufbau über die recur-Schleifen ab (CLAUDE.md: use_cache=False
  bei Speicherdruck). Die recur-Schleifen, die statische Selbst-Injektion und die
  Sensoren laufen weiter — die Emergenz-Messung bleibt gültig. **BASELINE** hat
  keine Rekurrenz und erzwingt `use_cache=True` (sonst seq²-langsam ohne KV-Cache).
- **`max_new=256`** (statt 300) gibt Reserve; Konklave lief 300 bei leerer GPU
  knapp durch, hier sind ~320 MiB durch Display belegt → 256 ist sicher.

### Seeding (Paar-Kompatibilität)

`torch.manual_seed(42 + seed)` pro Forward (batch=1 → jeder Seed sein eigener
deterministischer Strom). Seed *S* → RNG `42+S` **konstant über alle Varianten** →
paar-kompatibel: die Differenz zwischen Varianten bei gleichem Seed ist rein der
Architektur zuzuschreiben, nicht dem Zufall.

Resumable: JSONL dedup nach `(variant,label,seed)`, nur fehlende werden generiert.
Vor jedem Lauf Konsolidierung (keine Redundanz). Sequentiell, ein Prozess (keine
Parallel-Prozesse). Artefakte bleiben im Commit.