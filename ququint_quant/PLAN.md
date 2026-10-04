# PLAN — GF(5)/Ququint-Quantisierung für LLM-Gewichte

Quelle: `gf5_quant.txt` (Entwurf aus den riemann-Experimenten).
Zweig: `gf5-ququint-quant`. Arbeitet CPU-seitig neben dem laufenden Server
(keine VRAM-Konkurrenz).

---

## Stufe 1 — Quantizer-Prototyp (dieses Paket, v0.1)

`gf5_llm_quantizer.py`:

1. **Quantisierung**: 5 Zustände {−2,−1,0,1,2}. Per-Block-Skalare
   (Blockgröße 64, fp16-Ziel): S₀ = amax(|W_b|)/2, Q = clip(round(W/S), −2, 2).
   Globale K-Ratio-Feinsuche (k ∈ 0.50..1.30, Schritt 0.05) minimiert die
   Blocksumme ‖W − Q·S‖² — das ist die 1D-Ausprägung des L2-Scale-Fit, den
   `pt_ququint_simulator.py`/`pt_ququint_vqe.py` (COBYLA, 5×5-Raum) im
   riemann-Repo bereits etabliert haben.
2. **Packing**: 13 Ququints pro uint32, Basis 5, LSB-Rang zuerst
   (U32 = Σ (q_k+2)·5^k, 5^13 = 1 220 703 125 ≤ 2^32 → 2,4615 bit/Gewicht).
   Verlustfreier Roundtrip inkl. Grenzfälle (alles −2 → U=0, alles +2 →
   U = 5^13−1). Digit-Encoding-Vorbild: `pt_shor_ququint.py`
   (EXPERIMENT 037, Ququint-Digit-Encoding, little-endian).
3. **Fehlvermessung** (rel-L2, am selben Material, gleiche Blockgröße):
   - GF5-amax / GF5-fit (qmax 2)
   - ternär-amax / ternär-fit (BitNet-1.58-Stil, qmax 1)
   - INT4-symmetrisch (±7, amax) als 4-bit-Referenz
   - Material: synthetisch N(0,1) / Laplace / Outlier-Mix (5 % Spalten ×8)
     und echte F32-Gewichte des Ternary-Bonsai-2-27B
     (`model.layers.N.linear_attn.in_proj_{a,b}.weight`, 48×5120, F32 —
     die MLps der Bonsai-Konvertierung sind bereits ternär gepackt).
4. **Superset-Nachweis**: GF5 enthält die ternären Zustände {−1,0,+1}·d
   exakt → Dequant eines echten Bonsai-Tensors
   (`dequant_pack`, `runtime_qwen35_ptq.py:48`) und GF5-Requantisierung
   muss rel-L2 = 0 liefern. Damit ist die Verlustfrei-Konvertierung
   ternäres Modell → GF5-Pack beweisbar und numerisch demonstriert.

Ergebnisse: `results_v0_1.json` + Konsolen-Tabelle.

## Stufe 2 — GF(5)-GEMV-Kernel (Triton, shift-add)

Bei BS=1-Dekodierung ist die GPU Bandbreiten-limitiert (GEMV). Kernideen
aus `gf5_quant.txt`, operationalisiert:

- **Digit-Extraktion**: pro Record u (uint32) laufen LSB zuerst:
  d_k = u mod 5; u ← (u − d_k)/5 — in konstanter Zeit über
  Barrett-Reduktion (M = floor(2^32/5) = 0x33333333) bzw. Triton-Integer-
  Division. Arithmetik-Tabellen-Vorbild: `pt_ququint_gf5.py` (GF5_ADD/
  GF5_MUL, EXPERIMENT 026); die mod-5-Logik desselben Modells.
- **Mul-freie Akkumulation**: q = d−2 ∈ {−2..2}: q=0 → skip; q=±1 → ±x;
  q=±2 → ±(x+x) (predizierte Adds/Selects statt FMA). Sign via Maske.
- **Layout**: Records (28 B wäre PTQ1_0; hier 13×(2,46) ≈ 4,05 B/Record
  u32) — ein Warp pro Ausgabe-Zeile, Records sequential + x-Segment
  im Shared/L1.
- Referenz-Gegenstück im eigenen Repo: `ptq10_matvec` (Triton, PTQ1_0-
  Layout) — dieselbe Triton-Struktur mit GF5-decoden statt trit-table.

**Ergebnis (2026-10-04, `gf5_gemv.py` → `results_kernel_v0_2.json`):**
Feld-parametrischer Kernel `_gfp_mv` (P ∈ {5,3}, ND ∈ {13,20}, block-
aligniert auf Bonsai-128er-Blöcke mit fp16-Skalar je Block; uint32-Records
als int32-View auf der GPU — High-Bit-Records (3^20−1 > 2^31) über Maske
`& 0xFFFFFFFF` nach int64-Sign-Extend; mul-freie Digit-Akkumulation).
Material: down_proj 5120×17408 (89,1 M Gewichte, real dequant):

| Variante | Bytes/MV | ms | GB/s | max\|Δ\| vs f32 |
|---|---|---|---|---|
| dense bf16 | 178,3 | 0,609 | 292,5 | 0,0529 |
| ptq10_u8 (Produktion) | 19,5 | 0,972 | 20,1 | 0,0529 |
| **gf3_u32** | 20,9 | **0,240** | 87,2 | **0,0000** |
| **gf5_u32** | 29,2 | **0,258** | 113,5 | **0,0000** |

GF3 ist 4,1× / GF5 3,8× schneller als der produktive PTQ1_0-Matvec bei
max|Δ| = 0,0000 — Verlustfreiheit end-to-end DURCH den Kernel bewiesen
(f32-Partials statt bf16-Rundung; dense/ptq10 teilen sich den x-bf16-
Rundungsfehler 0,0529). VRAM-Peak 0,57 GiB neben dem laufenden Server.

## Stufe 2b — GF(2)-Kernel (XNOR/popcount, offen)

GF(2) = BNN-Sign, 32 Digits/u32 (1,125 bit block-aligniert). Quantisierer
und Pack sind in `gf5_llm_quantizer.py` (v0.2) geliefert inkl.
GF2-Digit-Encoding-Fix: d = (q+1)//2 ∈ {0,1}, decode q = 2d−1 — direkt
q+1 wäre {0,2} und KEIN gültiges Basis-2-Digit (Überlauf-Beweis in
`scratches/ququint/kernel_debug.py`). Der mod-2-Decode ist kernel-seitig
popcount/XOR-spezialisiert → eigenes Stufen-Thema, nicht Teil dieses
Durchlaufs.

## Stufe 3 — Runtime-Integration

Geplant:
- `gq5_matvec` analog `ptq10_matvec` als Ersatzlinear hinter dem
  ternary-Patch (Config-Flag), Fallback auf dense Predequant.
- Converter: GGUF/HF-Packed → GF5-Repack (Superset-Pfad ist verlustfrei,
  s. o.); Skalen pro Block fp16 anhängen.
- Server/Bridge-Flag (`px_quant: "gq5"`) um A/B gegen PTQ1_0 zu fahren
  (Perp/η²-Routinen aus `eval/runner_ternary.py` wiederverwenden).

**Ergebnis (2026-10-04, GELIEFERT, Commits 2a058696 + a41dccda):** Umsetzung
als GF(3)-Pfad (nicht GF5 — GF3 ist bit-treu zum ternären Bonsai-Material,
max|Δ| = 0, und der schnellste Kernel):

- `ququint_quant/repack_gf3.py`: Offline-Converter PTQ1_0-Artefakt →
  GF(3)-Artefakt (Repack, kein Requant — Verlustfreiheit by construction).
- `px_patches/ternary_bonsai_27b_px/gf3_quant.py`: Kernel-Runtime
  (`build_and_load_gf3`, gepacktes lm_head) — ersetzt den PTQ1_0-Lauf als
  Default (`config.py: weight_format: "gf3"`; Env-Override
  `PX_WEIGHT_FORMAT=ptq10`).
- `px_patches/ternary_bonsai_27b_px/long_context.py`: der eigentliche
  Decode-Nutzen ausgespielt — KV-4bit-Cache + Chunked-Prefill
  (`generate_long`): kv4 1,1 GiB @131k (bf16 wäre 8,0 GiB), Logits nur je
  Chunk-Ende/Decode-Schritt, Streamer-Schnittstelle für SSE.
- Server-Wiring (generators.py): non-stream `_generate_long_completion`
  + Stream-Langpfad; Routing-Schwelle `_PX_LONGCTX_THRESHOLD = 3000`
  (HF-Kurzpfad materialisiert vocab-breite Logits: T × 262144 × bf16 ≈
  T/2 MiB — OOM ab ~4k Token auf 12 GiB); `_px_pre_generation_cache_flush`
  vor jeder Generierung (CLAUDE.md-Konvention).

Beweise: Preset-Matrix-Smoke grün; 9181-Tok-Streaming-Langpfad (SSE nach
kv4-Chunked-Prefill, TTFB 208 s); CitMind-Dialog 5 Turns vollständig
(relay L34, phi ≈ 0,99, px-Routing adaptiv je Turn, 0 OOM). A/B-Eval
(Perp/η² GF3-vs-PTQ10 + η²-Tuning) bleibt offen (Stufe 4 / Task #9).

## Stufe 4 — Evaluation

- VRAM-Effektivvergleich an 27B (ehrlich: PTQ1_0 liegt bei 1,75 bit —
  GF5 ist dort KEIN Speichergewinn; Nutzenfall für GF5 sind
  ternär-fehlerlimitierte Modelle zwischen 2 und 4 bit).
- Rekonstruktions-/Perplexitätsdreieck: PTQ1_0 (ternär) vs GF5 vs INT4.
- Token-Durchsatz (GEMV-Bench vs `ptq_decode_bench.py`).

## Ehrliche Positionierung (aus dem Speicher-Layout abgeleitet)

Bits/Gewicht: flat (global aligniert) vs block-aligniert auf Bonsai-128er-
Blöcke (Pad + fp16-Skalar einkalkuliert); GB bei 25,05e9 ternär-gepackten
U8-Gewichten (bits/8, Referenz-Basis der Konvertierung):

| Format | flat | 128-block-aligniert | ≈ Gewichte-GB (aligniert) |
|---|---|---|---|
| Bonsai PTQ1_0 (heute) | 1,75 | 1,75 | 5,93 GB (Bestand) |
| GF5/Ququint | 2,4615 | 2,625 | ≈ 8,2 GB |
| GF3/Triquint | 1,60 | 1,875 | ≈ 5,9 GB |
| GF2/BNN | 1,00 | 1,125 | ≈ 3,5 GB |
| INT4-sym | 4,0 (+Skalen) | 4,125 | ≈ 12,9 GB |

**Korrektur (2026-10-04):** block-aligniert auf 128er-Blöcke ist GF3
(1,875 bit) KEIN VRAM-Gewinn gegen PTQ1_0 (1,75 bit) — das Align-Pad
(12/140 Digits) frisst den flat-Vorteil (1,60). GF3s Nutzen ist der
DECODE: gemessen 4,1× schneller als der produktive Kern bei exakter
f32-Ausgabe. Ein echter VRAM-Gewinn entsteht erst bei größeren
Skalar-Blöcken (256/512 → GF3 ~1,72/1,66 bit) oder bei GF2/BNN
(1,125 bit, hoher Qualitätspreis, Kernel Stufe 2b). GF5 bleibt das
Präzisions-Upgrade (±2-Outlier-Ebene, freie Sparsity).

## Risiken

- Block-Skalare dominieren bei kleiner Blockgröße die Fehlerbilanz
  (fit-K-Ratio vs Nullen-Anteil → Sparsity-Verlust).
- Digit-Loop (13 iteriert) im Kernel kann Latenz addieren —
  gegen unroll-Layout messen.
- Nicht-skalierte Norm/einprojekt-Gewichte brauchen parallelen
  fp16/fp32-Pfad (haben sie in der Runtime schon).

## Abhängigkeiten

- `safetensors`, `numpy`, `torch` (venv `open-mythos`).
- Modell: `/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf/model.safetensors`.
- riemann-Quellen (Referenz, Kopien bleiben in ML4/riemann):
  `pt_ququint_gf5.py`, `pt_ququint_simulator.py`, `pt_ququint_vqe.py`,
  `pt_shor_ququint.py`.