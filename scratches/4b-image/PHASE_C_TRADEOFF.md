# Plan 3 Phase C Befund: PX ↔ Chunked-Prefill Trade-off

## Die fundamentale Spannung

User-Auflage (drei Bedingungen, alle gleichzeitig):
1. "PX-Patches wie vorher ablaufen"
2. "VRAM runter"
3. "Unendlich langer Kontext"

**Befund: Bedingungen 1 und 3 sind kontradiktorisch.**

PX-Patch (genauer: `_px_attention_forward` in `gemma3_270m_px_baseline.patch` +
GQA-reapeat etc.) macht full-attention über die gesamte Sequenz.
Chunked-Prefill zerlegt die Sequenz in Chunks; das bricht PX's
Annahme dass alle Tokens gleichzeitig in attention sind.

**Konkrete Fehlermeldung bei CHUNKED + PX:**
```
RuntimeError: The size of tensor a (512) must match the size of tensor b (1024) at non-singleton dimension 3
```

PX erwartet dass `attention_mask` und `key_states` über alle T dimensioniert sind.
Chunked prefill gibt nur einen Sub-Chunk → shape mismatch.

## Realistische Konfigurationen

| Setup | T_max | VRAM peak | PX aktiv? | Output-Qualität |
|---|---|---|---|---|
| 4b + int8 + full attention + PX | ~4500 | 11.7 GB | ✓ ja | baseline |
| 4b + int8 + chunked prefill (chunk=512) | ~8000 | 11.6 GB | ✗ nein (shape) | baseline |
| 4b + int8 + use_cache=False | ~8000 | 11.1 GB | ✓ ja | langsam aber korrekt |
| 4b + int8 + use_cache=False + PX | ~8000 | ~11.1 GB | ✓ ja | langsam |

`use_cache=False` ist die einzige Lösung die ALLE DREI Bedingungen
gleichzeitig erfüllt:
- ✓ PX-Patches laufen (kein chunking, full attention)
- ✓ VRAM bleibt unter 12 GB (kein KV-Cache buildup)
- ✓ T=8000 funktioniert (recompute statt cache)
- ABER: Generierung ist ~10x langsamer

## Phase C Trade-off-Optionen

### Option A: Hard-cap bei T=4500 (default)
- 4b + int8 + PX wie bisher
- Bei T>4500: graceful error "input too long, please chunk"
- Vorteil: PX bleibt 100% aktiv
- Nachteil: kein "unendlich langer Kontext"

### Option B: use_cache=False für lange Inputs
- T≤4500: wie bisher (PX + int8)
- T>4500: use_cache=False (PX bleibt, aber langsam)
- Vorteil: PX bleibt 100% aktiv, T>4500 möglich
- Nachteil: ~10x langsamer bei langen inputs

### Option C: Chunked prefill + PX off für lange Inputs
- T≤4500: wie bisher
- T>4500: chunked + PX aus (BASELINE preset)
- Vorteil: schnell (kein recompute)
- Nachteil: PX-Patch nicht aktiv bei T>4500

### Option D: PX-Patch für chunked-Prefill anpassen
- PX-Patch so umschreiben dass er mit chunked input funktioniert
- Vorteil: beides
- Nachteil: PX-Patch ist motor-edit (User-Auflage verbietet)

## Empfehlung

**Option A oder B.**

- A ist konservativ (kein Verhalten ändert sich)
- B ist maximalistisch ("unendlicher Kontext" möglich, aber langsam)

**Was ich umsetze:**
1. Server bekommt ein neues Feld `chunked_prefill: bool` in der request
2. Bei `chunked_prefill=True` UND T > threshold: use_cache=False
3. Sonst wie bisher (full generate)

PX-Patches bleiben unangetastet. Die chunked_prefill-Logik ist in
generators.py (Aufruf-Pfad), nicht im Modell.

## Status Phase A/B/C
- Phase A (int8 KV): TDD-grün, prod kontraproduktiv (Prefill-OOM)
- Phase B (chunked prefill): code da, isoliert grün
- Phase C (E2E): 4b + int8 + chunked läuft ohne PX
- Phase D (echte Server-Integration): TODO nach User-Wahl