# Phase D Gate-Befund: VRAM-Reduktion FÄLLT

**Status:** Phase D (Motor-Integration) ist gemäß User-Gate **nicht freigegeben**.

## Akzeptanzkriterium (User-Auflage)

> "ok, D ist freigegeben, wenn dadurch der VRAM-Verbrauch runter geht"

## Mess-Ergebnisse (Mini-Stack-Test)

Test: `scratches/4b-image/test_infllm_vram_reduction.py`

### VRAM-Reduktion
```
    T    no-infllm       infllm  reduction
  256     0.0125GB     0.0152GB    -22.0%
  512     0.0164GB     0.0210GB    -28.1%
 1024     0.0305GB     0.0325GB     -6.6%
 2048     0.0840GB     0.1225GB    -45.8%
```
Akzeptanz: ≥ +30% Reduktion. Gemessen: **-45.8%** (das ist Erhöhung!). Gate **FAIL**.

### Output-Qualität
- Akzeptanz: cos_sim ≥ 0.85 vs SDPA
- Gemessen: cos_sim = 0.7418
- Gate **FAIL**.

## Architektonische Analyse

Das Meine-Annahme war falsch. InfLLM komprimiert die **archivierten LTM-Blöcke**
(top-k + sinks), aber der **aktuelle Forward-Pass rechnet weiter mit voller
Sequenz** (Buffer k/v = T_local + retrieved + sinks). Das kostet mehr
Speicher als naive SDPA (wo nur der KV-Cache linear wächst).

InfLLM ist ein **Decode-Zeit-Hebel** (attention-Matrix skaliert linear im
T_hist beim 1-token-decode), kein Prefill-Hebel (wo attention-Matrix
quadratisch in T_prefill ist).

## Realer Anwendungsfall: T=4800 Prefill OOM

Das war der ursprüngliche Schmerz-Punkt. InfLLM löst ihn **nicht** —
es hilft beim **dekodieren** einer sehr langen Sequenz. Für **Prefill-OOM**
sind andere Hebel nötig:

- **MEM_EFF_CHUNK** (bereits in patch.py: chunked-attention für Prefill)
- **Quantized KV-Cache** (int8 KV statt bf16; separater Plan)
- **Größere PYTORCH_CUDA_ALLOC_CONF-Tuning**

## Phase D: NICHT ausgeführt

- `patch.py` ist unangetastet
- Motor-Edit (User-Vorgabe: "vor prod imolementierung von 2 vorher fragen")
  wäre Conditional gewesen, ist aber durch Gate-Fail hinfällig
- Phase A-C bleibt im `scratches/` als Forschungsartefakt, geht nicht in Prod

## Was bleibt

- Plan 1 (int8 Quantization) ist **fertig + grün** für T ≤ 4200
- 4b Server läuft auf Port 7860 (PID 1129713) als **4b + int8 Quantization**
- InfLLM-Code bleibt im scratches als Forschungs-Skelett, ist nicht
  motor-koppelt, kann jederzeit wieder aktiviert werden mit anderen
  Parametern