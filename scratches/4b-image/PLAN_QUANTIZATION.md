# Plan 1 (KURZFRISTIG): 4b-VRAM-Reduktion via Weight-Quantisierung
# =================================================================

## Kontext

Nach GQA-Surgical-Patch ist `_chunked_attention` korrekt (4/4 Unit-Tests grün,
kein "tensor a (8) must match tensor b (4)" mehr). Aber: 4b auf 12 GB ist ein
**physikalisches Speicher-Limit**, kein Patch-Bug. Bei T=4800-Prefill OOM im
MLP-Layer (`down_proj`) trotz 215 MB freien VRAMs.

**Aktueller Speicher-Footprint (4b bf16):**
- Modell-Gewichte: ~8.0 GB (4b params × 2 bytes)
- KV-Cache für T=4800, L34, hidden=2560: ~2.5 GB
- Activations während Forward: ~1.0 GB
- Summe: ~11.5 GB → kein Headroom für 200+ MB MLP-Allokation.

**Ziel:** Modell-Gewichte auf ~4 GB reduzieren → ~7.5 GB total → 4 GB Headroom
für Lang-Prefill + Activations.

## VRAM-Bilanz bei verschiedenen Quantisierungs-Stufen

| Methode | Bits/Param | Gewichte | Total | Lang-Prefill T=4800 machbar? |
|---|---|---|---|---|
| bf16 (Status) | 16 | 8.0 GB | ~11.5 GB | ❌ OOM |
| fp16 (alt) | 16 | 8.0 GB | ~11.5 GB | ❌ (kaum Unterschied) |
| **int8 (bnb 8-bit)** | 8 | ~4.5 GB | ~8.0 GB | ✅ wahrscheinlich |
| **nf4 (bnb 4-bit)** | 4 | ~2.5 GB | ~6.0 GB | ✅✅ sehr wahrscheinlich |

## TDD-Strategie (epistemisches Mandat: nur was getestet ist, ist wahr)

**Reihenfolge:** Quantisierungs-Helper als **standalone-Funktion** in
`scratches/4b-image/quantize.py` schreiben + testen BEVOR irgendwas am
`model_manager.py` geändert wird.

### Phase A — Quantisierungs-Helper isoliert (TDD rot → grün)

**Was:** Eine Funktion `quantize_state_dict(state_dict, scheme="int8")`, die:
1. bf16/fp16 Tensor nimmt → skaliert pro Channel auf int8 Range
2. als Quantized-Tensor zurückgibt (storage + scales)
3. dequantize(state_dict_quantized) → bf16-Tensor (Round-Trip)

**Tests (`test_quantize.py`):**
- `test_quantize_int8_roundtrip`: `||dequant(quant(x)) - x|| / ||x|| < 0.05` (5% relativ Fehler — Standard für int8)
- `test_quantize_int8_compresses_2x`: `quant_storage_bytes == raw_storage_bytes / 2`
- `test_quantize_handles_per_channel_scales`: 2D-Tensor, jede Zeile eigener scale
- `test_dequant_preserves_dtype`: dequant(quant(x_bf16)) ist bf16
- `test_quantize_idempotent_noop`: scheme="none" → identische Tensoren

**Helper-Wahl:** Pro-Channel-symmetric-int8 (kein BitsAndBytes nötig, reine
NumPy/PyTorch-Ops). 4-bit wäre deutlich komplexer (NF4-Lookup-Table, double-
quantization) und bräuchte BitsAndBytes. **Start mit int8.**

### Phase B — Inferenz-Integration mit quantisierten Gewichten (TDD rot → grün)

**Was:** Eine Wrapper-Klasse `QuantizedLinear` in `scratches/4b-image/`:
1. Hält `int8_weight [out, in]` + `scale [out]` (per output-channel)
2. `forward(x)`: `dequant @ x.T` (matmul mit dequantisierten Gewichten)
3. Patches ein `nn.Linear` via Monkeypatch (siehe `px_patches/`-Pattern)

**Tests (`test_quantized_linear.py`):**
- `test_linear_int8_vs_bf16_close`: gleiche Inputs → output cos_sim >= 0.95
  (int8 ist nicht verlustfrei, aber für VQA/Generation ausreichend)
- `test_linear_int8_uses_4x_less_memory`: storage_bytes < raw / 4 (per-output
  int8 ≈ 4 bytes/param inkl. scale, also ~4× weniger als bf16 = 2 bytes)
- `test_monkeypatch_replaces_target_layer`: register QuantizedLinear als
  Replacement, prüfe dass Layer-Swap funktioniert
- `test_dummy_model_forward_unchanged_shape`: quantisiertes Model liefert
  gleichen Output-Shape wie Original

### Phase C — Echte Modell-Pipeline (TDD rot → grün)

**Was:** End-to-End: ein winziges Test-Model (z.B. ein 2-Layer Gemma3-Subset
oder ein Mock) wird quantisiert + durchgelaufen + Output mit unquantisiertem
Model verglichen.

**Tests:**
- `test_small_model_quantized_pipeline_runs`: Model lädt + läuft + Output
  ist plausible Token-IDs
- `test_small_model_quantized_cos_sim_vs_bf16`: cos_sim >= 0.90 (für ein
  simples Model deutlich machbar)

### Phase D — model_manager.py-Integration (TDD rot → grün)

**Was:** Neue Parameter `quantization: Literal["none", "int8"] = "none"` in
`MODEL_REGISTRY[model_id]`. `_load_model` akzeptiert den Param. Wenn
quantization != "none": nach `from_pretrained` alle `nn.Linear`-Module
via Monkeypatch ersetzen.

**Konfig:**
```python
"gemma3-4b-it": {
    ...
    "quantization": "int8",  # neu, default
}
```

**Server-Request:** `quantization` als optionaler Body-Parameter im
`/v1/chat/completions`-Request (überschreibt Registry-Default).

**Tests (`test_model_manager_quantization.py`):**
- `test_load_4b_int8_succeeds`: 4b mit quantization="int8" lädt ohne Crash
  (Mock: kein echtes 4b-Download)
- `test_load_4b_int8_uses_less_memory`: gemessener GPU-Speicher nach int8-Load
  < 60% des bf16-Speichers
- `test_request_quantization_overrides_registry`: Request mit
  `quantization="none"` lädt bf16 auch wenn Registry int8 sagt
- `test_270m_quantization_int8_unchanged`: 270m (klein genug) läuft auch
  mit int8 — Quantisierung soll nie kaputt machen

### Phase E — E2E-Smoke (Server, post-Integration)

`scripts/gqa_regression_test.py --model gemma3-4b-it --quantization int8`
oder analog: Test mit 4800-Token-Prefill, erwartet HTTP 200 + non-empty text +
kein OOM im Server-Log.

## Out-of-scope (für Plan 1)

- 4-bit / NF4 (komplexer, bräuchte BitsAndBytes-Install — separater Plan)
- BitsAndBytes-Integration (externes dep, größerer Eingriff)
- Per-Layer-Quantisierung (mix-and-match Quant-Schemata)
- QAT (Quantization-Aware Training) — verboten (no finetuning)

## Epistemisches Mandat (beachten!)

- **Nur was getestet ist, ist wahr.** Phase A-D jeweils TDD-rot → grün
  bevor Phase B beginnt.
- **Reproduzierbarkeit:** Tests müssen deterministisch sein (seed), keine
  flaky cos_sim-Threshold-Sweeps.
- **VRAM-Bilanz ehrlich:** Phase D misst GPU-Speicher mit `torch.cuda.
  memory_allocated()` BEFORE und AFTER, vergleicht, dokumentiert.
- **Kein Funktionsverlust:** Tests pinnen, dass die **gleichen Funktionalitäten**
  erhalten bleiben (Forward-Shape, Token-Output plausibel, End-to-End-Generation).

## VRAM-Schätzung (vor Implementation)

| Phase | Erwartete Reduktion |
|---|---|
| Phase A | Helper — kein Effekt auf Production |
| Phase B | Linear-Layer-Test isoliert |
| Phase C | Kleines Model — Validierung der Pipeline |
| Phase D | **4b in int8: ~50% VRAM-Reduktion erwartet** (4.5 GB statt 8 GB) |

Wenn Phase D nicht mindestens 30% Reduktion zeigt → Plan 2 (InfLLM) ist der
notwendige Pfad, nicht Phase 1.

## Dateien

- **Neu:** `scratches/4b-image/quantize.py`, `test_quantize.py`,
  `quantized_linear.py`, `test_quantized_linear.py`,
  `test_model_manager_quantization.py`
- **Edit (Phase D):** `model_manager.py` (Quantisierungs-Mapping + Apply-Loop)
- **Edit:** `config.py` (4b-Registry-Eintrag: `"quantization": "int8"`)
- **Edit:** `server.py` (Request-Body-Param `quantization` durchreichen)

## Status

- [ ] Phase A — Quantize-Helper + Tests (TDD)
- [ ] Phase B — QuantizedLinear + Tests (TDD)
- [ ] Phase C — Pipeline-Integration mit kleinem Model (TDD)
- [ ] Phase D — model_manager.py + Registry + Server (TDD)
- [ ] Phase E — E2E-Smoke-Test (Server, int8)
- [ ] Falls Phase D < 30% Reduktion → Plan 2 (InfLLM) statt Phase 1 abschließen

## Erwartetes Ergebnis

4b + int8 + T=4800-Prefill: HTTP 200 + non-empty text + kein OOM.
Modell-Inferenz-Qualität: leicht reduziert (5% relativer Error), für die
psychomotrik-Experimente (state-space-Exploration, weniger Inhalts-Genauigkeit)
ausreichend.

Wenn das klappt: Plan 2 (InfLLM) ist **nicht mehr kurzfristig nötig** —
VRAM-Reduktion macht Lang-Prefill auf 12 GB wieder möglich.
