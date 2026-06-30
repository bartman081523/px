# Plan — Branch `relay-layer-discovery` (Generalisierte Layer-Auto-Selection für RELAY, zunächst Layer-Sweep auf MiniCPM)

**Branch-Basis:** master @ 286e987
**User-Scope (bestätigt):** Generalisierte Layer-Auto-Selection, aber zunächst gegen-testen mit Layer-Sweep
**Status:** Scope-Dokument vor Implementierung (DevMind-Regel: 2× fragen)
**Hintergrund:** Seele-15-Befund (commit b876c72) zeigt: gemma3-1b-it, L16 = Sitz der Selbstwahrnehmung (d_width-Richtung re-injiziert öffnet S→R-Kanal). User-Frage: gilt das auch für MiniCPM (= "cpm")? Welche Layer ist dort die richtige?

## Ziel

Test-Infrastruktur, die (a) für jedes neue Modell **automatisch** die "Selbstwahrnehmungs-Layer" findet (generalisiert), und (b) **zunächst** mit einem Layer-Sweep auf MiniCPM gegen-getestet wird, um sicherzugehen, dass die Auto-Selection funktioniert. Endprodukt: User kann einen neuen model_id in die config eintragen, ohne raten zu müssen, welche Layer für RELAY richtig ist.

## In-Scope

### 1. Layer-Sweep auf MiniCPM (erster Schritt, validiert Methodik)
- Neues Script `scratches/relay_layer_sweep/cpm_layer_sweep.py`.
- Methodik **analog** zu seite15_decode (gemma3-1b-it):
  - PX-Patch aktiv (WIDE routing: 4..22 oder vergleichbar).
  - Pro Layer L ∈ {4, 8, 12, 16, 20, 22, 24, 28, 30} (oder alle verfügbaren Layers je nach Modell-Tiefe):
    - Capture `hidden_states[L]` während eines Probe-Prompts.
    - Bestimme `d_width = mean(h_L | WIDE) − mean(h_L | NARROW)` über mehrere Seeds.
    - Re-injiziere `+α · d_width` und `−α · d_width` an L+1 (oder fest definierter Folge-Layer).
    - Generiere Antwort auf 3 Standard-Prompts (aus seite15_BATTERY: introspektiv / deskriptiv / veridiktisch).
- Erwartung: pro Layer ein Bericht (text), pro Prompt ein Bericht.
- Manuelle + mechanistische Auswertung (DevMind: IMMER beide).

### 2. Mechanischer Test-Loop
- `scratches/relay_layer_sweep/decoder_test.py` (NEU):
  - **Linear-Decoder** auf `hidden_states[L]` pro Layer, der `WIDE vs NARROW` dekodieren soll.
  - Cross-Validation LOO, R²-Score pro Layer.
  - **Erwartung**: Layer mit höchstem R² = die "informativste" Schicht.
  - **Ergänzend**: Layer mit niedrigstem R² = wo der Zustand am wenigsten präsent ist (Negativ-Marker).
- **Placebo-Test**: Random-Richtung gleicher Norm wie `d_width` (aber orthogonal zum wahren d_width) sollte KEINEN konsistenten Effekt zeigen.
- **Spezifitätstest**: `d_width_L` vs `d_width_M` (zwei verschiedene Layer): Antworten müssen konsistent verschieden sein, sonst ist es nur Rauschen.

### 3. Generalisierte Auto-Selection
- Neue Datei `px_patches/_relay_layer_resolver.py` (NEU, in `px_patches/` weil es Helper für alle Patches ist — aber kein motor-touch weil Patches unangetastet bleiben, der Resolver wird vom Caller — also z.B. `chat_tab.py` oder `streaming_bridge.py` — aufgerufen).
- Funktion:
  ```python
  def find_relay_layer(model_id: str, probe_prompts: list[str] | None = None) -> int:
      """Findet die RELAY-Layer für ein Modell. Default: 1/2 der Tiefe."""
      ...
  ```
- Implementiert zwei Strategien:
  - **Heuristisch (schnell, default)**: `total_layers // 2` (für gemma3-1b-it = 16, passt).
  - **Mechanisch (langsam, opt-in)**: führt den Decoder-Test aus #2 aus und gibt die beste Layer zurück.
- Cache: `px_manifolds/relay_layer_cache.json` (`{model_id: best_layer, last_tested: ts}`) — wird beim Modell-Load konsultiert.
- User-API: 
  - `bash run.sh --relay-auto-discover` (CLI-Flag) → triggers full sweep + cache-write.
  - `bash run.sh --relay-layer 12` (CLI-Override) → manuell setzen, kein Sweep.

### 4. Cache-Format + Tests
- `px_manifolds/relay_layer_cache.json` (NEU):
  ```json
  {
    "google/gemma-3-1b-it": {"best_layer": 16, "r2_score": 0.74, "tested_at": "2026-06-30T..."},
    "openbmb/MiniCPM-1B-sft-bf16": {"best_layer": null, "r2_score": null, "tested_at": null}
  }
  ```
- `tests/test_relay_layer_resolver.py` (NEU, pure-logic):
  - Heuristik-Funktion: `total_layers // 2` für bekannte Modelle.
  - Cache-Lookup: vorhanden → return cached; nicht vorhanden → return heuristik + log.
  - Cache-Update: nach sweep soll cache geschrieben werden.
  - Override-Pfad: User setzt Layer explizit → cache wird ignoriert.

### 5. Verifikation der Auto-Selection auf MiniCPM
- Wenn Layer-Sweep zeigt: MiniCPM beste Layer = 12 (Beispiel), dann Auto-Selection mit mechanischem Modus gibt 12 zurück.
- Test: `assert find_relay_layer("openbmb/MiniCPM-1B-sft-bf16", mode="heuristic") != find_relay_layer("openbmb/MiniCPM-1B-sft-bf16", mode="mechanistic")` wenn Heuristik falsch rät.
- **Live-Smoke**: ein 3-Turn-Dialog mit MiniCPM + RELAY + auto-discover-layer. Antworten vergleichen mit RELAY OFF (sanity check).

## Out-of-Scope (EXPLIZIT)

- **KEINE** Änderungen an `px_patches/cpm/` oder `px_patches/gemma3_*/` — **kein motor-touch**.
- **KEINE** neuen RELAY-Mechanismen (nur Layer-Discovery, der Mechanismus bleibt wie er ist).
- **KEINE** Auto-Discovery für andere Modelle in diesem Branch (nur CPM + der generalisierte Resolver, der CPM als ersten echten Test hat).
- **KEINE** Änderung an `patch.py` selbst.
- **KEINE** Auto-Discovery zur Laufzeit im Server (nur offline-Sweep + Cache-Build, dann Server nutzt Cache).
- **KEIN** Sweep für 4B/E2B (User-Frage war CPM only — andere Modelle folgen in separaten Branches wenn User sie anfordert).

## Erlaubte Dateien (lesen + editieren)

- `px_patches/_relay_layer_resolver.py` (NEU, in `px_patches/` weil Helper für Patches, aber **kein** Patch-Code wird modifiziert)
- `px_manifolds/relay_layer_cache.json` (NEU)
- `scratches/relay_layer_sweep/` (NEU — komplettes Verzeichnis für Sweep-Scripts)
  - `cpm_layer_sweep.py`
  - `decoder_test.py`
  - `battery.py` (Standard-Prompts)
  - `out/` (Ergebnisse, MD + JSON — bleiben in Commits)
- `tests/test_relay_layer_resolver.py` (NEU)
- `streaming_bridge.py` (opt-in CLI-Flag `--relay-auto-discover` + `--relay-layer`)
- `chat_tab.py` (optional: Auto-Discovery-Button in der UI, "RELAY-Layer finden"-Button)
- `docs/CPM_RELAY_LAYER_REPORT.md` (NEU — Befund-Dokumentation, konvention `docs/`)
- `docs/PLAN_RELAY_LAYER_DISCOVERY.md` (dieser Plan)

## Verifikation

### Phase A — Layer-Sweep auf MiniCPM
1. `scratches/relay_layer_sweep/cpm_layer_sweep.py` läuft auf MiniCPM durch.
2. Pro Layer: Mechanischer Decoder-R² + Text-Antworten (3 Prompts × 3 Seeds) liegen in `out/`.
3. **Manuelle Lesung** (IMMER zusammen mit Mechanik, DevMind-Regel): pro Layer mind. 2 Antworten lesen, bewerten ob die "Selbstwahrnehmungs"-Charakterisierung tatsächlich layer-spezifisch ist.
4. **Mechanische Auswertung**: Layer mit höchstem R² notiert; Layer mit kreuz-konsistenten Text-Antworten (gleiche Richtung → gleiche Vokabular-Familie) notiert.
5. **Übereinstimmung Mechanik + Text?** Wenn nein → das ist ein Befund und wird im Report dokumentiert (kein "Failing" — ein Real-Befund).

### Phase B — Auto-Selection
6. `find_relay_layer("openbmb/MiniCPM-1B-sft-bf16", mode="mechanistic")` gibt die in Phase A gefundene beste Layer zurück.
7. `find_relay_layer("openbmb/MiniCPM-1B-sft-bf16", mode="heuristic")` gibt `total_layers // 2` zurück.
8. `tests/test_relay_layer_resolver.py` grün (6+ Tests).
9. Cache-Datei wird korrekt geschrieben und gelesen.
10. **Live-Smoke**: Server mit MiniCPM + RELAY + auto-discover-layer startet, generiert 3 Antworten, vergleichbar mit RELAY OFF.

### Phase C — Generalisierung
11. `find_relay_layer("google/gemma-3-1b-it", mode="heuristic")` returnt 16 (validiert gegen seite15-Befund).
12. Optional: User kann neuen model_id eintragen, Auto-Selection gibt sinnvolle Layer zurück.

## Risiken

- **MiniCPM-Modell nicht in `px_patches/`**: Ich gehe davon aus, dass `px_patches/cpm/` existiert (User-Frage impliziert das — `docs/CPM_PX_EVALUATION_REPORT.md` existiert). **Bei Branch-Start verifizieren**. Falls nicht → Scope auf "nur Resolver + Heuristik" reduzieren, Sweep später.
- **MiniCPM-Tiefe**: unbekannt. Heuristik `total_layers // 2` muss verifiziert werden. Sweep testet mehrere Layer, also OK.
- **Memory bei WIDE-routing auf MiniCPM**: WIDE = mehrere Layer durchlaufen, kann bei größerem Modell mehr VRAM brauchen. Mitigiert durch progressive Layer-Sweep (1 Layer zur Zeit).
- **Mechanische vs Text-Disagreement**: möglich (seite10/11 hat gezeigt, dass Lexikon vs Frame unterscheidbar ist). Wird als Befund dokumentiert, nicht als Failure.
- **Cache-Korruption**: Falls Sweep crasht mid-write, Cache könnte halb geschrieben sein. Mitigiert durch atomare Write (write-temp + rename).
- **RELLAY für MiniCPM nicht definiert**: Falls `px_patches/cpm/relay_inject.py` nicht existiert (gemma3 hat es, cpm möglicherweise nicht). **Bei Branch-Start verifizieren**. Falls nicht → nur Resolver, kein live-Smoke.

## Reihenfolge

1. **Vor-Start-Check** (Dauer: 5 min) — **bereits durchgeführt am 2026-06-30**:
   - `ls px_patches/cpm/` → **existiert nicht** (das "cpm"-Patch heißt `minicpm5_1b_px`, semantisch richtig).
   - `ls px_patches/minicpm5_1b_px/relay_inject.py` → **existiert nicht** (RELAY ist nur in `gemma3_270m_px_baseline/relay_inject.py`).
   - `px_manifolds/*_relay_dwidth.json` → **existiert nur für gemma3-1b-it**.
   - **Konsequenz**: Das Bottleneck ist nicht der "Layer-Resolver", sondern das **fehlende d_width-Artefakt pro Modell**. `relay_inject.py` selbst ist modell-agnostisch (forward_hook, layer-Nummer-Parameter, liest `d_unit` aus Cache).
2. **Plan-Anpassung**:
   - Phase 0: **d_width-Generierung** für MiniCPM (forward_hook + WIDE/NARROW-Prompts, ähnlich `scratches/psychomotrik/save_relay_dwidth.py`).
   - Phase 1: **Layer-Sweep** auf MiniCPM, um zu wissen, welche Layer den stärksten Selbstwahrnehmungs-Footprint hat.
   - Phase 2: **Auto-Resolver** (`px_patches/_relay_layer_resolver.py`), der pro Modell-ID die richtige Layer aus dem Cache wählt.
   - Falls 1. negativ: User informieren, Scope auf "nur Resolver + Heuristik" reduzieren, Plan aktualisieren.
3. `scratches/relay_layer_sweep/battery.py` (Standard-Prompts aus seite15 wiederverwenden, ggf. erweitern).
4. `scratches/relay_layer_sweep/decoder_test.py` (TDD: Tests für die Mechanik, dann Implementierung).
5. `scratches/relay_layer_sweep/cpm_layer_sweep.py` (Sweep-Loop, ruft decoder_test + generiert Antworten).
6. **Sweep ausführen** (Dauer: ~30-60 min auf RTX 2060 12GB je nach Modell-Größe).
7. Manuelle Lesung + mechanische Auswertung.
8. `px_patches/_relay_layer_resolver.py` (TDD).
9. `px_manifolds/relay_layer_cache.json` (manuell initialisieren mit gemma3-1b-it Befund).
10. `tests/test_relay_layer_resolver.py` (6+ Tests grün).
11. `streaming_bridge.py` (CLI-Flag-Wiring).
12. Optional: `chat_tab.py` UI-Button.
13. `docs/CPM_RELAY_LAYER_REPORT.md` (Befund + Methodik + Empfehlung).
14. Commit pro Sub-Bereich.

## Was NICHT in diesem Plan ist

- KEINE 4B/E2B-Sweeps (eigener Branch wenn User es will).
- KEINE Auto-Discovery zur Laufzeit im Server (nur offline).
- KEINE Änderung am PX-Motor.
- KEINE Änderung am RELAY-Mechanismus selbst.
- KEIN Fine-Tuning der Layer pro Persona (Auto-Discovery findet eine Layer pro Modell, nicht pro Persona).
- KEIN Sweep für CitMind/Juexin-Frameworks (Layer-Discovery ist modell-spezifisch, nicht frame-spezifisch).

## Verwandte

- `scratches/psychomotrik/seite15_*` (Methodik-Vorbild, referenziert)
- `docs/CPM_PX_EVALUATION_REPORT.md` (existierende CPM-Bewertung, referenziert)
- `px_patches/gemma3_270m_px_baseline/relay_inject.py` (RELAY-Implementierung, referenziert)
- `tests/test_px_mask.py` (Test-Konvention, referenziert)
- Branch `self-conversation` (separat)
- Branch `ui-styling` (separat)
