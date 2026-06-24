# Plan 3 Phase A/C Befunde — ehrlich

## Phase A: int8 KV-Cache (forward_hook) — TDD grün, prod kontraproduktiv

### Test-Befund (mini-stack, 5/5 grün)
- install+uninstall idempotent ✓
- cos_sim round-trip = 1.0 (per-channel int8 für normalverteilte Werte verlustfrei)
- VRAM-Reduktion 100% (wenn int8 KV auf CPU verschoben)
- attention output cos_sim = 1.0 (vs SDPA)

### Produktiv-Befund (4b real, T=4602)
- **OHNE int8 KV hook**: 11.69 GB peak, knapp OK
- **MIT int8 KV hook**: 8.75 GB in use, **OOM** weil Quant+DeQuant 2x Speicher brauchen

→ int8 KV hook **verschlechtert** Prefill-Situation. Hook ist kontraproduktiv.

### Warum?
Beim Prefill liegt die ganze Sequenz auf einmal im Forward. Hook quantisiert
+ dequantisiert IN-PLACE, was während des forwards 2x Tensoren im Memory
hält. Der Storage-Vorteil (int8 auf CPU) greift erst NACH dem Forward
wenn past_key_values zurückgegeben wird — aber dann ist generate schon
fertig.

### Lösung: hook conditional machen
Hook nur anwenden wenn `T_new <= 64` (incremental decode), NICHT beim
Prefill. Das spart dann echt KV-Cache-Speicher bei langen Generierungen.

## Phase C: E2E-Messung mit 4b + int8

| Setup | T | Peak VRAM | Status |
|---|---|---|---|
| int8 weights only, eager attn | 4502 | 11.69 GB | ✓ OK |
| int8 weights + int8 KV hook | 4602 | 8.75+2.50 → OOM | ✗ FAIL |
| int8 weights, SDPA attn | 8002 | OOM (10.80+0.49) | ✗ FAIL |
| int8 weights, use_cache=False | 8002 | 11.14 GB | ✓ OK (langsam!) |

### T=4500 ist die effektive Grenze für 4b + int8 weights auf 12GB
- eager attention: peak 11.69 GB
- SDPA hilft nicht genug für T=8000
- use_cache=False löst T=8000 aber ist unbenutzbar langsam

## Realer Status — UPDATED 2026-06-24 nach Phase D

- **4b + int8 weights funktioniert bis T≈4500** (Peak 11.69 GB, 0.3 GB Reserve)
- **T>4500 hatte OOM** mit full generate, jetzt **gelöst durch chunked_prefill**
- **chunked_prefill bei T=8002: peak 6.4 GB, 32s für 16 tokens** (vs OOM, vs 196s mit use_cache=False)
- **Bit-Äquivalenz: chunked == full** bei T=4502 (byte-identisch)
- **Server-Integration: auto-detect in generators.py**: T>4500 → chunked_generate, sonst model.generate

## Phase D: Chunked-Prefill (gelöst!)

### Was es ist
`chunked_generate` iteriert `text_model.forward()` in Chunks (default 512 tokens),
jeder Chunk baut den KV-Cache inkrementell auf. score-Matrix pro Chunk = [chunk, T_so_far]
statt [T_total, T_total]. Bei T=8000: 8x kleinere attention-matrix → 6.4 GB statt 12+.

### Critical fix #1: Gemma3 sliding-window Bug
HF `DynamicSlidingWindowLayer.get_mask_sizes(query_length)` berechnet bei
inkrementellem Prefill `kv_length = sliding_window - 1 + query_length = 1535`
für chunk=512 nach cumulative=1024. ABER `full_key_states` returned hat nur
1024 Tokens. SDPA-Error: T_q=512 vs T_k=1024 vs mask=1535.

**Fix**: `chunked_generate` baut einen **full-only DynamicCache** (alle Layer
als `full_attention`, `sliding_window=None`). Sliding-window-Pattern geht
für den Prefill verloren, decode danach funktioniert normal. Bei T=4502 ist
chunked-Prefill byte-identisch zu full generate (verifiziert).

### Critical fix #2: lm_head nicht quantisieren
`quantize_all_linears` quantisiert auch `lm_head` (262208×2560 = 1.34 GB
bf16). Aber `QuantizedLinear.forward` dequantisiert int8 → fp32 → bf16
und materialisiert 2.7 GB temporär. Bei chunked_prefill mit T=8002: OOM.

**Fix**: `quantize_all_linears(skip_names=("lm_head",))` lässt lm_head in
bf16 (1.34 GB Storage, 0 temporärer overhead).

### Critical fix #3: HF inference_mode + autograd-Konflikt
`QuantizedLinear.forward` braucht autograd-tracking, aber `inference_mode`
macht output-Tensors inference-only. Wenn `lm_head` aus `inference_mode`
heraus aufgerufen wird, crashed es.

**Fix**: `chunked_generate` macht nur den `text_model.forward()` in
`inference_mode`. Nach dem forward: `with torch.no_grad(): hidden.clone()`,
dann `model.lm_head(hidden)`.

### Resultat
| Setup | T | Peak VRAM | dt für 16 tokens | Status |
|---|---|---|---|---|
| int8 weights only, full generate | 4502 | 11.69 GB | ~13s | ✓ (knapp) |
| int8 weights + chunked_prefill | 4502 | 5.93 GB | 26s | ✓ |
| int8 weights + chunked_prefill | 8002 | 6.41 GB | 32s | ✓ |
| int8 weights + use_cache=False | 8002 | 11.14 GB | 196s | ✓ aber lahm |

### Bit-Äquivalenz
`test_chunked_prefill_vs_full.py`: chunked == full (byte-identisch) bei T=4502.
cos_sim ≈ 1.0 zwischen den hidden-states. Plan 3 Ziel: T>4500 mit PX-Patches
+ 12 GB VRAM. Erreicht.

### Server-Integration
`generators.py:_px_gen_kwargs` setzt `_px_use_chunked_prefill=True` wenn:
- Model ist 4b/E2B (nicht 270m/1b)
- Input > 4500 tokens
- User hat use_cache nicht explizit gesetzt

`generate_chat_completion` (non-streaming) ruft `chunked_generate` statt
`model.generate` wenn Marker gesetzt.

**Streaming-Pfad umgestellt** (Plan 3 Phase D final): `chunked_generate`
akzeptiert jetzt einen HF `TextIteratorStreamer` als `streamer=` parameter.
`generate_chat_completion_stream` startet einen worker-thread der
chunked_generate mit dem SSE-Streamer aufruft. Vorteil: 10x schneller
als use_cache=False Fallback.

| Setup | Pfad | T | dt für 16 tok |
|---|---|---|---|
| chunked_generate | non-streaming | 6010 | 24.7s |
| chunked_generate + streamer | streaming | 6010 | 24.7s (dt_first=18.7s) |
| use_cache=False Fallback | streaming (old) | 6010 | 269s |

### Streaming chunked_generate Implementation
- `chunked_generate(..., streamer=None)`: optional HF TextIteratorStreamer-
  kompatibler Parameter. Token-IDs werden an `streamer.put()` gepusht;
  der Streamer dekodiert selbst.
- `generate_chat_completion_stream`: bei chunked-Marker wird
  `_chunked_worker` in eigenem Thread gestartet, der chunked_generate
  mit dem bereits erstellten TextIteratorStreamer aufruft. SSE-Schleife
  nutzt denselben Streamer.
- `eos_token_id` muss in `chunked_generate` ein **einzelner int** sein,
  nicht die Liste aus `_inject_eot_eos` — Pop + Index-Extraktion in
  generators.py.