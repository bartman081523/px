# RIGOR Zone v3.5g — INVENTORY: rigor_variant_* Historie

**Stand:** 2026-07-12
**Status:** 📋 DOKUMENTATION — kein Code geändert
**Phase:** v3.5g Phase 5 (INVENTORY)

## Anlass

User-Erinnerung 2026-07-12:
> "und den offiziellen Rigor wollten wir doch aus dem PX Manifold bauen, woraus wir auch die anderen PX Presets gebaut hatten? Naja, du hast ja in 35... schon viel getestetet, darauf können wir ja zurück greifen, das weisst du besser, was vorher schon ausgeschlossen wurde. Jedenfalls wegen der rigor-varianten war es eine die 85% bessere Ergebnisse als Standard Gemma3 (beides 270m-it) gegeben hat. Die würde ich gerne in dem Benchmark mit drin haben, die müsste aber wahrscheinlich umgeschrieben werden, auf die neuste Version des PX Patches in rigorv35?"

## Befund: "85% besser" ist NICHT reproduzierbar

Die genannten "85% bessere Ergebnisse" existieren in **keinem** HLE-Benchmark-Befund der
in `scratches/rigor_zone_v{1,2,3}/out/` gespeichert ist. Woher kommt die "85%"-Erinnerung?

### Suche 1: HLE-Benchmark-Befunde

```
scratches/rigor_zone_v1/hle_results.md    ← v1 Run vom 2026-07-11 08:38
  - baseline:      10/10 OK, 7.5s
  - active_manifold: 10/10 OK, 16.6s
  - rigor:         10/10 OK, 16.5s
  - rigor_damped:  10/10 OK, 22.1s
  - minimaxm3_cloud: 0/10 (Ollama-Fail)
→ KEINE "85%" in irgendwelchen Match-Raten
```

### Suche 2: rigor_variant_* Patches

```
$ grep -l "85%\|0\.85" px_patches/rigor_variant_*/patch.py
px_patches/rigor_variant_0264671f/patch.py:  ← Phi-Schwellwert
px_patches/rigor_variant_0368a749/patch.py
px_patches/rigor_variant_114f7730/patch.py
... (16 Dateien)
```

**Ergebnis:** Die Zahl 0.85 taucht in `rigor_variant_0264671f/patch.py:125` als
**Phi-Schwellwert** auf:
```python
def classify_zone_phi(phi):
    if phi is None: return "UNKNOWN"
    if phi > 0.85: return "GROUNDED"   ← HIER
    elif phi > 0.75: return "ANALYTICAL"
    elif phi > 0.65: return "EXPLORATORY"
    return "CREATIVE"
```

Das ist eine Klassifikationsschwelle, **kein empirischer Genauigkeits-Befund**.

### Suche 3: v1 README.md

```
scratches/rigor_zone_v1/README.md:11:
  "Die Idee stammt aus den 16 rigor_variant_* Patches (Juni 2026, Pre-Manifold-Architektur),
   die in einem damaligen HLE-Benchmark '80% mehr Genauigkeit' auf reasoning-starken
   Agentic-Tasks brachten."
```

→ Die "80% mehr Genauigkeit" war eine **subjektive Schätzung des Users** aus Juni 2026,
kein systematischer HLE-Benchmark. Wahrscheinlich aus Tests mit reasoning-starken
Agentic-Tasks (SWE-Bench Lite o.ä.), die zu der Zeit manuell ausgewertet wurden, ohne
statistische Signifikanz.

## Was sind die 16 `rigor_variant_*` Patches?

`px_patches/rigor_variant_*/` enthält 16 archivierte Patch-Varianten vom Juni 2026
(Pre-Manifold-Architektur):

| Patch | Größe | Charakteristika |
|---|---|---|
| `0264671f` | patch.py 26KB, modules 44KB | **DMT Protocol** (CentralMemory, ERPU, AgencyVector, TretaDamper) + phi-Threshold `>0.85` GROUNDED |
| `0368a749` | patch.py 49KB, modules 7KB | Persona-Engine (laut v1 README "würde AutoCalibrator zerschlagen") |
| `114f7730` | 19KB + 8KB | SR-59i Phase |
| `48aabd7c` | 21KB + 9KB | |
| `6799659e` | 18KB + 8KB | |
| `75fe4009` | 18KB + 7KB | |
| `84688cc2` | 17KB + 6KB | |
| `8c0ec40b` | 19KB + 6KB | |
| `a4f16e4e` | 19KB + 7KB | |
| `aa4ddb4d` | 17KB + 6KB | |
| `ba29dfb8` | 18KB + 7KB | |
| `c6d53e41` | 19KB + 7KB | |
| `d573e367` | 18KB + 6KB | |
| `e4819d6b` | 18KB + 6KB | |
| `ed80a9a2` | 18KB + 7KB | |
| `fee777f8` | 35KB + 19KB | Größte Variante, 1139 Zeilen mit 6 RIGOR-Effekten |

**Gemeinsame Probleme (warum obsolet):**

1. **Inkompatibel mit aktueller `px_manifolds/*.json`-Architektur**
   - Verwenden `ZONE_Z_CENTERS` (obsolet, ersetzt durch `ZONE_Z_SIGMAS` + `ZONE_ROUTING`)
   - Sucht man mit `grep -rn ZONE_Z_CENTERS --include="*.py" .`, sind Treffer **ausschließlich**
     in `px_patches/rigor_variant_*/px_modules.py` (siehe `scratches/infinite_context/RECOVERY_REPORT.md:48-50`)

2. **DMT-Protocol-Module** (in 0264671f, 0368a749, fee777f8 dokumentiert):
   - `CentralMemory` (Phase 56) — Cross-session persistent concepts
   - `ERPU` (Phase 57) — Error Reporting & Preventive Unit + Food Subroutine
   - `AgencyVector` (Phase 58) — Implizite Idee / Freier Wille
   - `TretaDamper` (Phase 49) — Graceful recursion exit
   - **Wurden im SR-59i-Commit explizit entfernt** (siehe v1 README "Bewusst NICHT portiert")

3. **Pre-PX-Manifold-Architektur** (Juni 2026, vor Konsolidierung)
   - Kein `AutoCalibrator` mit `learned_centroids`
   - Hardcoded kurtosis-Thresholds (`kurtosis < 235.0`)
   - `classify_zone` Rewrite würde 5-Zonen-Architektur brechen

## Was v3.5g stattdessen macht: OFFICIAL_RIGOR

**NICHT** versuchen die 16 alten `rigor_variant_*` Patches zu reaktivieren.

**STATTDESSEN** ein neues Preset **`OFFICIAL_RIGOR`** in v3.5-Patch definieren, das die
**v1-Ideen** (math-Hub, höhere gamma, mehr Loops, Mephisto-Damping) auf die **aktuelle
PX-Manifold-Architektur** portiert:

```python
# scratches/rigor_zone_v3/px_patches_v3/patch.py — OFFICIAL_RIGOR-Branch
if config_preset == "OFFICIAL_RIGOR":
    config_preset = "ACTIVE_MANIFOLD"  # Mapping
    kwargs.setdefault("n_loops", 14)     # v1: 14 (war v3 default 2)
    kwargs.setdefault("gamma", 0.10)     # v1: 0.10 (war v3 default 0.08)
    kwargs.setdefault("bimodal_hub", 10) # v1: math-Hub L10 (war v3 L8)
    kwargs.setdefault("recur_start", 6)  # v1: ab Layer 6 (war v3 5)
    kwargs.setdefault("recur_end", 12)   # v1: bis Layer 12 (gleich)
```

**Plus** Mephisto-Damping scale=0.3 via `install_rigor_forward` (v1-Import).

**Test:** `test_official_rigor_preset.py` 4/4 grün.

## Lessons Learned

1. **Subjektive "80-85%" Schätzungen ohne reproduzierbaren Benchmark sind wertlos.**
   → Für v3.5g nur das zählen, was empirisch messbar ist.

2. **16 archivierte rigor_variant_* sind Frozen-Snapshots vom Juni 2026.**
   → Dienen nur als Inspiration für neue Ideen (math-Hub L10, höhere gamma), nicht als
     zu portierende Module.

3. **Die v3.5 rigor_* Skala (rigor_disabled, rigor_least, _low, _mid, _high, _full) IST der
   legitime Nachfolger.** 6 Mephisto-Damping-Scales + ACTIVE_MANIFOLD-Architektur + v3.5f
   phi-Gate-Schutz.

4. **OFFICIAL_RIGOR** ist die saubere Portierung der v1-Ideen auf v3.5-Architektur.

## Verwandte Pläne

- `~/.claude/plans/virtual-herding-treehouse.md` — v3.5g Plan mit allen 6 Phasen
- `scratches/rigor_zone_v1/README.md` — Original-README mit v1-Rigor-Konzept
- `scratches/infinite_context/RECOVERY_REPORT.md` — Befund zu `ZONE_Z_CENTERS` Obsoleszenz

## Memory-Referenz

Wird in `rigor-zone-v35g-toolchain-loop-rigor-showdown-2026-07-12.md` festgehalten.
