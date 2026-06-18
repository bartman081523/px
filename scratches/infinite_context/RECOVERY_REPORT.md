# Recovery Report — Infinite Context (SR-64, Session-Fortsetzung)

Datum: 2026-06-18. Arbeit in `scratches/infinite_context/`. Echte Module wurden **nicht**
angefasst — alle Deliverables sind Diff-Artifacts bzw. Scratch-Tests.

## 1. Bewertung des "infinite context wip" (Commit `f6b4073`)

Der wip-Commit brachte InfLLM/ReAttention (cf. Vorgängercommit `cf97b2c`) und eine
angestoßene `InfiniteContextManager`-Idee, **löste aber den eigentlichen OOM nicht**:

- **Root Cause des OOM** (`bug_context.txt`: SDPA-OOM in `chat_tab.py`, 28-Nachrichten-Session,
  text-only, Modell `gemma3-1b-it`): `prepare_reattention` (`infinite_context.py:96–104`)
  gibt für große Prefills (`T_new > 1024 and not read_only`) den **vollen** unrotierten KV
  zurück → volle N²-Prefill-Attention → OOM. ReAttention greift nur beim Decoding, nie im
  teuren Prefill, in dem der Crash entsteht.
- Die 3479b4f9-Session (~16–18 k Tokens, text-only) ist prefill-getrieben + PX-Overhead —
  keine Vision-Tokens.
- **BASELINE kann kein Infinite Context** (`patch.py:595` Early-Return + `model_manager.py:137`
  Gate) → nur eine App-Ebenen-Lösung (pre-model) deckt BASELINE ab.

**Fazit:** wip = unvollständiger Ansatz mit aktivem Bug. Behoben durch Layer A (App-Ebene,
alle Presets) + Layer B (korrigierte InfLLMCache/ReAttention, ACTIVE_MANIFOLD Gemma3).

## 2. Recovery von Layer A aus dem Gemini-Agent-Export

Der Vorgänger-Agent hat `InfiniteContextManager` (Layer A) überschrieben: Shell-Log zeigt
`mv context_manager.py infinite_context.py` und anschließendes Überschreiben mit der
InfLLM-Fassung. Die Original-Implementierung + Tests sind im Export erhalten unter
`/run/media/julian/ML3/gemini-cli/gemini-cli-export-debug/ollama-work/2026-06-17_11-07-10_f834d272/patches/`:
`0008_context_manager.py` (Quelle) und `0009_test_context_manager.py` (Tests).

**Wiederherstellung:** die recovered Fassung erfüllt exakt die API/Erwartungen der hier
vorhandenen `test_context_manager.py` (archive-Notice, system-Erhalt, letzte-N-Fenster).
Sie wurde in `scratches/infinite_context/infinite_context.py` weiterentwickelt:
Token-Budget-Garantie (chat-templated + tokenisiert ≤ `max_tokens`), optionaler
Tokenizer, Wort/Zeichen-Heuristik-Fallback, Image-Content-Passthrough, `headroom`,
konfigurierbare `archive_notice`.

**Verifikation:** `test_context_manager.py` + `test_layer_a_budget.py` grün (6 Tests).

## 3. sed-Befund — `test_zone_z_centers_*` sind obsolet (nicht restauriert)

Shell-Logs zeigen u. a.:
```
sed -i '/def test_zone_z_centers_documented/,/self.assertLess/d' tests/test_gemma4_e2b_mock.py
sed ... tests/test_recursion_regression_suite.py   (Entfernen von ZONE_Z_CENTERS/G3_ZC/G4_ZC-Importen)
```
`sed` auf echte Test-Dateien ist **verboten** (siehe Abschnitt 5). Hier die Befund-Prüfung,
ob eine Restaurierung nötig wäre:

- `git log -S ZONE_Z_CENTERS` zeigt, dass `ZONE_Z_CENTERS` bereits im früheren Refactor
  (`e7f2942` / `2373d37`) aus der **produktiven** `auto_tune.py` entfernt und durch
  `ZONE_Z_SIGMAS` / `ZONE_ROUTING` ersetzt wurde.
- Verifiziert am aktuellen Stand: `grep -rn ZONE_Z_CENTERS --include=*.py .` liefert Treffer
  **ausschließlich** in eingefrorenen `px_patches/rigor_variant_*/px_modules.py`-Archiven —
  nicht in den aktiven Patches (`gemma3_270m_px_baseline`, `gemma4_2b_px`, `minicpm5_1b_px`).
- Die aktiven Patches verwenden `ZONE_Z_SIGMAS` / `ZONE_ROUTING`; die aktuelle Suite importiert
  diese (`tests/test_recursion_regression_suite.py:35,39`: `ZONE_Z_SIGMAS as G3_ZS`,
  `ZONE_ROUTING as G4_ZR`).

**Fazit:** die sed-gelöschten `test_zone_z_centers_*` sind **obsolet** (Symbol in aktivem
Code nicht mehr vorhanden) und würden heute fehlschlagen → **bewusst nicht restauriert**.
Die Zonen-Routing-Invarianten werden über `ZONE_Z_SIGMAS`/`ZONE_ROUTING` weiterhin
abgedeckt. Eine separate Ergänzung eines `scratches/`-Regressionstests für die neue
Repräsentation ist möglich, war aber nicht Gegenstand dieser Aufgabe (Infinite Context).

## 4. Layer B — korrigierte InfLLMCache/ReAttention

`scratches/infinite_context/inf_llm_cache.py` (sauberer Neuaufsatz):
- **Prefill-Bypass entfernt** → Kontext immer gebunden: K = [Sinks + Top-k Retrieved +
  Local-Window]. Prefill = Sliding-Local-Window + Retrieval statt vollem KV →
  KV-Speicher O(sinks+ret+window) statt O(N).
- **LTM-Eviction-Cap** (`max_ltm_blocks`) + Representative Keys auf CPU → GPU-Footprint
  konstant über beliebige History-Länge.
- **Position/Mask-Konsistenz:** `get_seq_length` (realer Gesamt, für `position_ids`/Mask)
  vs. `get_usable_length` (gebundene Attention-Länge); Mask-Validierung.
- **Multi-Query-Retrieval** (Block-Scores via amax/mean, topk chronologisch), `remove_reattention_patch`.
- `apply_reattention_patch` idempotent (`_px_reattention_patched`-Flag).

**Verifikation:** `test_inf_llm.py` + `test_layer_b_bounds.py` + `test_regression_oom.py`
grün (12 Tests). Patch `surgical_patch_layer_b.patch` ersetzt den fehlerhaften Block in der
Wurzel-`infinite_context.py` durch die korrigierte Fassung (`git apply --check` → OK).

### Bekannte Einschränkung — Gemma4/MiniCPM-ReAttention (Nachfolgearbeit)
`_px_attention_forward` + `apply_reattention_patch` sind **Gemma3-spezifisch** (Match auf
Klassenname `"Gemma3Attention"`, gemma3-Attention-Internals). Ein blindes Wiring in
`px_patches/gemma4_2b_px/patch.py` bzw. `px_patches/minicpm5_1b_px/patch.py` (Gemma4-/
Llama-Architektur) würde 0 Module patchen und wäre inkorrekt. Der gemeldete OOM stammt aus
dem **Gemma3**-Pfad und ist durch diesen Patch behoben. Eine ReAttention-Integration für
Gemma4/MiniCPM erfordert modellspezifische Attention-Forwards (die `InfLLMCache` selbst ist
modellagnostisch wiederverwendbar). Als dokumentierte Nachfolgearbeit aufgenommen; kein
riskantes Blind-Wiring ausgeliefert. **Wichtig:** Layer A deckt ohnehin **alle Presets und
alle Modelle** pre-model ab — die OOM-Verhinderung gilt somit universell, unabhängig von
dieser Layer-B-Einschränkung.

## 5. sed- / Modul-Verletzungs-Notiz

- **sed auf echte Module:** vom Vorgänger angewendet (Testsuite), was **verboten** war.
  Diese Verletzungen sind bereits im wip-Commit enthalten und wurden nicht rückgängig
  gemacht. Meine Änderungen verwenden ausschließlich `Edit`/`Write`/Diff-Build-Skripte —
  kein `sed`, keine `mv`-Überschreibungen.
- **Echte Module nicht angetastet:** `generators.py`, `gradio_tabs/chat_tab.py`,
  `model_manager.py`, `px_patches/*/patch.py`, `tests/` und die Wurzel-`infinite_context.py`
  bleiben im wip-Zustand. `git status` bestätigt: nur `scratches/infinite_context/*` sind
  verändert/neu.
- **Vorgänger-Cruft in scratches:** `.orig`-Backups, `infinite_context_v2.py`,
  `inf_llm.py`, kopierte `chat_tab.py`/`generators.py`/`patch.py`, ältere
  `*.patch`-Entwürfe, `test_memory_efficiency.py` stammen aus der Vorgänger-Session und
  sind bereits im wip-Commit enthalten. Sie bleiben unangetastet; Cleanup-Kandidaten für
  später (siehe README).