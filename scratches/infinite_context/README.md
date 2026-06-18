# Infinite Context — Surgical Patches (SR-64)

Lösung für den OOM bei langen Sessions (Reported: `sessions/3479b4f9.json`,
text-only, ~16–18 k Tokens, `gemma3-1b-it`, RTX 2060 12 GB). Siehe
`RECOVERY_REPORT.md` für Bewertung des wip + Recovery-Details.

## Architektur — zwei Layer

### Layer A — `InfiniteContextManager` (App-Ebene, garantiert OOM-frei, ALLE Presets)
`scratches/infinite_context/infinite_context.py`. Token-Budget Sliding-Window **vor** dem
Modell (pre-model, vor `apply_chat_template`). Wirkt unter **BASELINE und ACTIVE_MANIFOLD**
für **alle Modelle** (Gemma3/Gemma4/MiniCPM), weil es das Modell gar nicht erreicht — es
begrenzt den Prompt, der tokenisiert ans Modell geht. **Das ist der universelle OOM-Schutz.**
- Garantie: chat-templated + tokenisiert ≤ `max_tokens`.
- System-Nachrichten bleiben erhalten; jüngste Nachrichten innerhalb des Budgets;
  Archive-Notice bei Verwurf; Image-Content-Passthrough; `headroom`; Tokenizer optional
  (Wort/Zeichen-Heuristik-Fallback).

### Layer B — korrigierte `InfLLMCache` + ReAttention (architektonisch, ACTIVE_MANIFOLD)
`scratches/infinite_context/inf_llm_cache.py`. Behebt den wip-Bug (Prefill-Bypass, der den
OOM verursachte): KV-Kontext immer gebunden `[Sinks | Retrieved | Local-Window]`,
LTM-Eviction-Cap, Representative Keys auf CPU → GPU-Footprint konstant über beliebige
History-Länge. ACTIVE_MANIFOLD, **Gemma3** (der bereits gewiredete Pfad). Gemma4/MiniCPM:
dokumentierte Nachfolgearbeit (modellspezifische Attention-Forwards nötig; siehe
RECOVERY_REPORT §4). Da Layer A universell ist, ist der OOM-Schutz davon unberührt.

## Deliverables (Diff-Artifacts — nicht angewendet)

| Datei | Wirkung |
|---|---|
| `surgical_patch_layer_a.patch` | `generators.py` (Stream + Non-Stream, je nach `processed_messages` vor `apply_chat_template`) + `gradio_tabs/chat_tab.py:chat_fn`: `InfiniteContextManager(max_tokens=2048,…).process_history(processed_messages, tokenizer)` (try/except-gesichert). Preset-agnostisch. |
| `surgical_patch_layer_b.patch` | Wurzel-`infinite_context.py`: ersetzt fehlerhaften `InfLLMCache`/`apply_reattention_patch`-Block durch korrigierte Fassung. Behebt den Prefill-Bypass, der den OOM verursacht. |

Beide Patches verifiziert: `git apply --check` → Exit 0 (s.u.).

## Tests

CPU-Suite (keine GPU nötig, CI-fähig) — **18/18 grün**:
```
cd scratches/infinite_context
../../open-mythos_p2/venv_openmythos/bin/python -m pytest \
  test_context_manager.py test_inf_llm.py test_layer_a_budget.py \
  test_layer_b_bounds.py test_regression_oom.py -q
```
- `test_layer_a_budget.py`: Budget-Garantie, System-Erhalt, Image-Passthrough, headroom.
- `test_layer_b_bounds.py`: `get_usable_length` bleibt gebunden über History-Wachstum;
  LTM-Eviction-Cap; seq-length vs. usable-length; CPU-Offload; Patch/Unpatch.
- `test_regression_oom.py`: simuliert 3479b4f9-Wachstum (14/30/60/120 Paare) — naive N²
  überschreitet Budget-Proxy, Layer A Prompt ≤ BUDGET, Layer B KV ≤ KV_BOUND.

E2E-Smoke (optional, GPU-guarded, `ALL_SPACE_E2E=1`): `test_e2e_session_smoke.py` lädt
`gemma3-1b-it` + 3479b4f9 über `ModelManager`, injiziert Layer A, generiert → kein OOM,
nicht-leerer Text.

## Patches anwenden (Trockenprüfung + Anwendung)

```bash
# 1) Trockenprüfung (keine echte Moduleänderung)
git apply --check scratches/infinite_context/surgical_patch_layer_a.patch
git apply --check scratches/infinite_context/surgical_patch_layer_b.patch
# 2) Anwenden (nur wenn gewünscht — ändert dann echte Module)
git apply scratches/infinite_context/surgical_patch_layer_a.patch
git apply scratches/infinite_context/surgical_patch_layer_b.patch
```
Hinweis: Die Patches sind absichtlich **nicht** angewendet — die echten Module bleiben im
wip-Zustand. Die Tests in `scratches/` laufen gegen die dortigen Scratch-Kopien der Layer.

## GPU-Validierungs-Notiz
- Layer B (`InfLLMCache`/ReAttention) ist CPU-unit-getestet (Bounds/Eviction/Shapes), aber
  die Integration in den echten Gemma3-Forward sollte vor Produktivnahme auf der RTX 2060
  einmal gegen 3479b4f9 validiert werden (`test_e2e_session_smoke.py` mit `ALL_SPACE_E2E=1`).
- Layer A ist GPU-frei und wirkt pre-model → ohne GPU-Validierung produktiv sicher
  (reine Prompt-Begrenzung).

## Cleanup-Kandidaten (Vorgänger-Cruft, nicht Teil dieser Änderung)
In `scratches/infinite_context/` vom Vorgänger committed und unangetastet gelassen:
`*.orig` (chat_tab/generators/patch), `infinite_context_v2.py`, `inf_llm.py`, kopierte
`chat_tab.py`/`generators.py`/`patch.py`, ältere `infinite_context.patch`/`infllm_reattn.patch`,
`test_memory_efficiency.py`, `.pytest_cache`/`__pycache__`. Später entfernbar.