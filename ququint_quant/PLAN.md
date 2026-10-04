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

## Stufe 3 — Runtime-Integration

- `gq5_matvec` analog `ptq10_matvec` als Ersatzlinear hinter dem
  ternary-Patch (Config-Flag), Fallback auf dense Predequant.
- Converter: GGUF/HF-Packed → GF5-Repack (Superset-Pfad ist verlustfrei,
  s. o.); Skalen pro Block fp16 anhängen.
- Server/Bridge-Flag (`px_quant: "gq5"`) um A/B gegen PTQ1_0 zu fahren
  (Perp/η²-Routinen aus `eval/runner_ternary.py` wiederverwenden).

## Stufe 4 — Evaluation

- VRAM-Effektivvergleich an 27B (ehrlich: PTQ1_0 liegt bei 1,75 bit —
  GF5 ist dort KEIN Speichergewinn; Nutzenfall für GF5 sind
  ternär-fehlerlimitierte Modelle zwischen 2 und 4 bit).
- Rekonstruktions-/Perplexitätsdreieck: PTQ1_0 (ternär) vs GF5 vs INT4.
- Token-Durchsatz (GEMV-Bench vs `ptq_decode_bench.py`).

## Ehrliche Positionierung (aus dem Speicher-Layout abgeleitet)

| Format | bit/Gewicht | 27B-Gewichte |
|---|---|---|
| Bonsai PTQ1_0 (heute) | 1,75 | 5,93 GB |
| GF5/Ququint | 2,4615 | 8,32 GB |
| INT4-sym | 4,0 (+Skalen) | ~13,5 GB |

GF5 gewinnt gegen INT4 (~40 %) und gegen BitNet-1.58-nominal in
Präzision; gegenüber dem vorliegenden Bonsai-Format ist es ein
Präzisions-Upgrade (±2-Outlier-Ebene, freie Sparsity), kein
Speicher-Upgrade.

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