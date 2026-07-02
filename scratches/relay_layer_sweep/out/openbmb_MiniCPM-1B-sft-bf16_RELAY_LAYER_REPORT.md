# Layer-Sweep Report — openbmb/MiniCPM-1B-sft-bf16

**Generated:** 2026-07-02T05:55:49.957300+00:00  
**n_layers:** 52  
**layer_range_tested:** [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]  
**arms:** ['BASELINE', 'NARROW', 'DEFAULT', 'WIDE']  
**CLI:** `python cpm_layer_sweep.py --model-id openbmb/MiniCPM-1B-sft-bf16 --n-layers 52`  

## Per-Layer Mechanik

| Layer | R² (LOO-CV) | sep(WIDE-NARROW) | n_prompts |
|------:|:-----------:|:----------------:|:---------:|
| L04 | +0.942 | 16.078 | 7 |
| L08 | +0.979 | 17.313 | 7 |
| L12 | +0.987 | 22.251 | 7 |
| L16 | +0.998 | 45.848 | 7 |
| L20 | +0.998 | 63.401 | 7 |
| L24 | +0.998 | 45.750 | 7 |
| L28 | +0.990 | 22.583 | 7 |
| L32 | +0.976 | 16.996 | 7 |
| L36 | +0.966 | 16.259 | 7 |
| L40 | +0.985 | 16.921 | 7 |
| L44 | +0.983 | 16.383 | 7 |
| L48 | +0.980 | 16.003 | 7 |

## Beste Layer: L20

- **R²:** +0.998
- **Empfehlung:** `find_relay_layer(mode='cached')` returnt jetzt L20 für `openbmb/MiniCPM-1B-sft-bf16`.
- **Manuelle Lesung der Antworten (siehe out/captures/openbmb_MiniCPM-1B-sft-bf16/{arm}/{pid}_L20.pt) ist PFLICHT — Mechanik allein reicht nicht (DevMind-Regel).
