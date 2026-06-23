# Plan 2 (LANGFRISTIG): 4b-VRAM via InfLLM / ReAttention
# =========================================================

## Kontext

Plan 1 (Quantisierung) reduziert den **Modell-Footprint** auf der GPU (Gewichte
~8 GB → ~4.5 GB bei int8). Aber: das hilft **nicht** gegen wachsenden
**KV-Cache** bei langen Konversationen.

**KV-Cache-Verbrauch (4b):**
- Pro Token: `2 × n_layers × Hkv × D_head × 2 bytes`
- 4b: `2 × 34 × 4 × 256 × 2 = 139 KB / Token` (bf16)
- 4096 Tokens Konversation: **569 MB KV-Cache**
- 16384 Tokens Konversation: **2.28 GB KV-Cache**
- 32768 Tokens Konversation: **4.55 GB KV-Cache** → frisst den ganzen Headroom,
  den Quantisierung freigeschaufelt hat.

**Idee (war schon mal da, dann verloren):**
InfLLM (Infinite Length Language Modeling) + ReAttention (decoupled RoPE für
Cross-Block-Attention) — implementiert in Commit `7cf8218` (SR-64) und
`3cdb8f4` (surgical patches). Lebt jetzt **orphaned** in `infinite_context.py`,
wird nur von `test_archive.py` referenziert, ist nicht in `patch.py`
integriert.

**Kern-Mechanik:**
- **InfLLM:** KV-Cache wird in **Blöcke** partitioniert (`block_size`, default
  8-16). Aktuelle Attention hält: (a) aktuelle Sequenz, (b) `top_k_blocks`
  LTM-Blöcke (per Repräsentativ-Key-Score ausgewählt), (c) `sinks_count`
  Attention-Sinks (erste Token, die alle Heads als Anker brauchen).
- **ReAttention:** Statt blockweise RoPE neu zu rechnen, decoupled RoPE nur
  auf q/k (nicht v) — passt Position so an, dass Cross-Block-Attention
  Positions-konsistent mit aktueller Sequenz funktioniert.

**VRAM-Effekt:**
- KV-Cache wird **nicht mehr komplett gehalten**, sondern LTM-Blöcke werden
  auf CPU ausgelagert oder komprimiert (off-by-one: `r_tokens` Repräsentativ-
  Keys pro Block, statt voller Block im Memory).
- Bei 16384 Tokens: statt 2.28 GB Cache → ~600 MB (top-4 Blöcke + sinks +
  aktuelle Sequenz).

## TDD-Strategie (epistemisches Mandat)

**Reihenfolge:** Erst alle Pin-Tests grün (existieren teilweise in
`test_infllm_smoke.py` und `test_archive.py`), dann in eine **eigenständige
Integrations-Schicht** zwischen `model_manager.py` und `patch.py` einbauen,
**ohne `patch.py` anzufassen** (so wie der User es will).

### Phase A — Inventory & API-Re-Check (TDD rot → grün)

**Was:** Die existierenden Tests `test_infllm_smoke.py` (11 Tests grün) +
`test_archive.py` (zu prüfen) lesen + ggf. erweitern.

**Fragen:**
- Ist `InfLLMCache` API vollständig? Was fehlt für echte Inferenz?
  - Aktuelle API: `prepare_reattention(q, k, v, layer_idx, rotary_emb_module)`
    → mutiert state intern (block-archive, sinks, top-k)
  - Was fehlt: `from_kv_cache(k_cache, v_cache)` (Initial-Befüllung),
    `evict_block(idx)` (manuelles Evict), `serialize()/deserialize()`
    (Persistenz), `merge_kv_from_disk()` (CPU-Auslagerung)

**Tests (zu ergänzen):**
- `test_infllm_cache_initial_state`: neue Cache hat leere buffers, sinks=None
- `test_infllm_cache_archive_block_atomicity`: archive-Block ist atomar
  (entweder voll drin oder nicht, kein Partial-State)
- `test_infllm_cache_topk_selection_stable`: gleiche Keys → gleiche top-k
  (deterministic)
- `test_reattention_decoupled_rope_modifies_k_only`: ReAttention ändert
  q+k, nicht v
- `test_sinks_persist_after_eviction`: LTM-Block evict darf sinks nicht
  antasten

### Phase B — Cache-Hierarchie: GPU ↔ CPU (TDD rot → grün)

**Was:** Die existierende `InfLLMCache` hält alles auf GPU. Erweiterung:
`InfLLMHierarchicalCache` mit:
- **L0 (GPU):** aktuelle Sequenz + sinks + top-k Blöcke (wie bisher)
- **L1 (CPU):** archived Blöcke, die nicht in top-k sind, auf `torch.Tensor`
  im CPU-RAM (Pinned Memory für schnellen Transfer)
- **L2 (Disk, optional):** für sehr lange Sessions, evict → serialisiertes
  `pickle` oder `safetensors` in `~/.cache/all_space/infllm/`

**Tests:**
- `test_l1_blocks_resident_on_cpu`: archived Blöcke sind auf CPU device
- `test_topk_promotion_l1_to_l0`: bei Bedarf wird ein L1-Block auf GPU
  promoted (overlappend mit einem weniger relevanten L0-Block)
- `test_evict_l0_to_l1`: L0-Volles → LRU-Block wandert nach L1
- `test_l1_cpu_memory_bounded`: L1 darf max N Blöcke haben, sonst LRU nach
  L2 (Disk)
- `test_l2_serialization_roundtrip`: serialize → deserialize → cos_sim >= 0.999

### Phase C — Integration mit Attention-Forward (TDD rot → grün, ohne patch.py)

**Was:** Wrapper-Funktion `infllm_attention_forward(...)` in
`scratches/4b-image/infllm_integration.py`, die:
1. Standard-q/k/v entgegennimmt (von HF Gemma3Attention.forward)
2. An `InfLLMHierarchicalCache.prepare_reattention` durchreicht
3. `topk_blocks` + `sinks` + aktuelle seq konkateniert → SDPA-Attention
4. KV-Cache-Werte in `cache.add_kv(...)` (zu ergänzen) aktualisiert

**Aktivierung:** NICHT automatisch, sondern nur wenn:
- Modell in `MODEL_REGISTRY` mit `infllm_enabled: True` markiert ist
- UND die Aufruf-Site in `patch.py:_mem_eff_attention_forward` ein Hook-
  Interface bekommt (das wäre ein **Motor-Edit** — Phase D)

**Workaround für Phase C (ohne Motor-Edit):**
Monkeypatch auf Module-Level. `from_pretrained` → `model.forward` →
`_mem_eff_attention_forward` ist schon ein forward_hook-Punkt. Wir können
einen zusätzlichen Hook registrieren, der `attention_forward` umleitet.

**Tests:**
- `test_infllm_forward_returns_same_shape_as_sdpa`: bei T=512 muss Output-
  Shape gleich sein wie naive SDPA
- `test_infllm_forward_cos_sim_vs_full_attention`: cos_sim >= 0.95
  (InfLLM wirft Blöcke weg, das ist nicht verlustfrei)
- `test_infllm_forward_kv_cache_grows_sublinearly`: bei T=2k → 4k →
  8k Tokens darf KV-Memory nicht linear wachsen (sublinear wegen Eviction)
- `test_infllm_hook_install_uninstall_idempotent`

### Phase D — Motor-Integration (TDD rot → grün, USER-FREIGABE ERFORDERLICH)

**Was:** `patch.py:_mem_eff_attention_forward` so erweitern, dass bei
`infllm_enabled=True` (per patch_kwargs) die `infllm_attention_forward`
aus Phase C aufgerufen wird statt SDPA. Das ist **das** Motor-Edit, das der
User explizit absegnen muss.

**Aktivierungs-Mechanik:** `patch_kwargs["infllm_enabled"] = True` +
`patch_kwargs["infllm_block_size"] = 16` + `patch_kwargs["infllm_top_k"] = 4`.

**Tests:**
- `test_patch_infllm_smoke_270m`: 270m läuft mit InfLLM, kurzer Prompt, OK
- `test_patch_infllm_smoke_1b`: 1b läuft mit InfLLM
- `test_patch_infllm_smoke_4b`: 4b läuft mit InfLLM, **T=4800 ohne OOM**
  (das ist der eigentliche Erfolgs-Test)
- `test_patch_infllm_quality_vs_baseline`: identische Prompt → cos_sim der
  Hidden-States >= 0.95 (InfLLM wirft Blöcke weg, aber bleibt nahe dran)

### Phase E — E2E-Smoke (Server, post-Integration)

`scripts/gqa_regression_test.py` erweitern oder neuen
`scripts/infllm_smoke.py` schreiben. Verifiziert: 4b mit InfLLM-aktivierung,
T=4800-Prefill, HTTP 200, non-empty text, GPU-Speicher < 8 GB während
Inferenz.

## Out-of-scope (für Plan 2)

- **Echte L2-Disk-Auslagerung** (Pfad existiert, Tests optional)
- **ReAttention für nicht-GQA-Modelle** (1b/270m: Hkv=1, andere Mechanik)
- **Adaptive Block-Size per Layer** (manche Layer profitieren mehr, manche
  weniger)
- **Quantisierte KV-Cache** (KV in int8 statt bf16) — orthogonal zu InfLLM,
  separater Plan

## Epistemisches Mandat

- **TDD-rot → grün** für jede Phase, BEVOR die nächste startet.
- **VRAM-Messung ehrlich:** Phase D misst mit `torch.cuda.memory_allocated()`,
  dokumentiert Vorher/Nachher. Akzeptanzkriterium: ≥ 50% Reduktion bei
  T=16384 im Vergleich zu Baseline-KV-Cache.
- **Qualitäts-Kompromiss dokumentiert:** InfLLM ist nicht verlustfrei — die
  Tests pinnen einen Mindest-cos_sim (0.95), UND die Generierung muss
  sinnvollen Text liefern (keine Halluzinationen wegen weggeworfener Blöcke).
- **Verlustfrei-Modus optional:** Wenn InfLLM als REPLACEMENT für SDPA läuft,
  kann der User `infllm_strict=True` setzen → KEIN Block wird evicted,
  sondern nur komprimiert (langsamer, aber verlustfrei). Default ist LOSSY.

## VRAM-Schätzung

| Szenario | Baseline (bf16) | Mit Quantisierung (int8) | Mit Quant + InfLLM |
|---|---|---|---|
| 4b Gewichte | 8.0 GB | 4.5 GB | 4.5 GB |
| KV-Cache T=4800 | 0.67 GB | 0.67 GB | 0.18 GB (top-4 + sinks) |
| Activations | 1.0 GB | 1.0 GB | 1.0 GB |
| **Total** | **~9.7 GB** | **~6.2 GB** | **~5.7 GB** |

Mit **beidem** (Quant + InfLLM): 4b sollte mit T=4800-Prefill auf 12 GB
locker laufen, UND bei T=16384 noch ~1 GB Headroom haben.

## Dateien

- **Existiert:** `infinite_context.py` (InfLLMCache), `tests/test_infllm_smoke.py`
- **Zu prüfen:** `tests/test_archive.py` (testet InfLLM-Hilfsklassen?)
- **Neu (Phase A):** Tests für `InfLLMCache` Lücken
- **Neu (Phase B):** `scratches/4b-image/infllm_hierarchical.py` + Tests
- **Neu (Phase C):** `scratches/4b-image/infllm_integration.py` + Tests
- **Edit (Phase D, mit Freigabe):** `patch.py:_mem_eff_attention_forward`
- **Edit:** `config.py` (4b-Registry: `"infllm_enabled": True` per default?)
- **Neu:** `scripts/infllm_smoke.py` für E2E

## Status

- [ ] **USER-FREIGABE für Phase D** (Motor-Edit) — VORAB klären
- [ ] Phase A — Inventory + API-Erweiterungen (TDD)
- [ ] Phase B — Hierarchical Cache (TDD)
- [ ] Phase C — Forward-Integration via Hook (TDD, kein Motor-Edit)
- [ ] Phase D — Motor-Integration (TDD, nach Freigabe)
- [ ] Phase E — E2E-Smoke (Server)

## Risiken

- **Risiko 1: ReAttention-Korrektheit.** Gemma3 nutzt RoPE anders als
  Llama; decoupled-RoPE auf q/k könnte zu Positions-Inkohärenz führen.
  Mitigation: cos_sim-Threshold 0.95 in Tests, manuelle Sichtprüfung der
  ersten Generation.
- **Risiko 2: Performance-Degradation.** InfLLM allokiert/deallokiert pro
  Token; bei kleinem T ist es langsamer als naive SDPA.
  Mitigation: nur aktivieren wenn `T > MEM_EFF_INFLLM_THRESHOLD` (z.B. 2048).
- **Risiko 3: Modell-Verhalten ändert sich.** Auch bei guter cos_sim kann
  das Modell anders „denken" mit evicten Blöcken.
  Mitigation: Phase D misst nicht nur cos_sim, sondern auch echte Text-
  Qualität (Manual Reading).

## Erwartetes Ergebnis

4b + Quant (int8) + InfLLM (top_k=4, block=16, r_tokens=2) auf 12 GB:
- T=4800: HTTP 200, non-empty text, GPU < 8 GB
- T=16384: HTTP 200, non-empty text, GPU < 10 GB
- Qualität: cos_sim >= 0.95 vs SDPA, manuelle Lesung: kohärenter Text

Wenn das klappt: 4b ist auf der RTX 2060 12 GB **voll nutzbar** für die
psychomotrik-Experimente, ohne Modell-Downgrade auf 1b.
