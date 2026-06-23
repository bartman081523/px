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

## Realer Status

- **4b + int8 weights funktioniert bis T≈4500** (Peak 11.69 GB, 0.3 GB Reserve)
- **T>4500 braucht andere Architektur** (chunked prefill, gradient checkpointing,
  oder reduzierte Batch-Größe)
- **int8 KV hook hilft nicht beim Prefill** (siehe oben)
- **InfLLM deferred**: bringt nichts für Prefill-OOM; ist ein Decode-Zeit-Hebel

## Empfehlung

User-Auflage war "unendlich langer Kontext" + "VRAM runter" + "PX-Patches
bleiben wie vorher". Realistisch:

1. **T≤4500 ist mit int8 weights sicher** — Server kann das liefern
2. **T>4500 erfordert andere Architektur**:
   - model.forward iterativ mit progressive past_key_values
   - das ist motor-nah aber geht via wrapper
   - ODER: hard-cap input auf T=4500, dann "weiterlesen" Mechanik

Vorschlag: wir liefern **T≤4500 mit 4b + int8 als realen Service**.
Bei T>4500 graceful error "input too long, please chunk".

## Was NICHT getan wurde
- patch.py unangetastet
- PX-Patches unverändert
- sensoren / routing / rekursion: alles wie vorher

## Dateien
- `int8_kv_cache.py` — funktioniert als TDD-Test, ist NICHT prod-empfohlen
  ohne conditional skip für prefill
- `test_e2e_int8_kv_standalone.py` — E2E-Test gegen 4b
- `test_int8_kv_cache.py` — TDD-rot→grün für int8 KV hook