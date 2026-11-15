# Capability-Kampagne 2026-11-14 — gemessen, nicht behauptet

Alle Zahlen kommen aus der eigenen OpenAI-kompatiblen API (`server.py` auf `:7860`
bzw. `llama-server` auf `:9931`). Harness: `bench/capability_bench.py`.
Rohdaten: `bench/results/*.json`.

## 0. Zwei echte Bugs, die die Messung zuerst zerstört haben

### Bug A — Preset-Rücksstände (behoben, `px_patches/gemma4_2b_px/patch.py`)
`apply_px_patch` setzte `_px_repetition_penalty=1.15` und `_px_no_repeat_ngram_size=3`,
`remove_px_patch` entfernte sie **nicht**. Nach `ACTIVE_MANIFOLD → BASELINE` blieb die
Token-Loop-Bremse aktiv, `generators._px_gen_kwargs` las sie weiter und zerlegte Code:
`def add( a, b):` / `result. append(...)` / `result.Append(...)`.
Derselbe Prompt über den Direktpfad war fehlerfrei.
Fix: `_strip_px_attrs()` löscht **jedes** `_px_*`-Attribut. Tests: `tests/test_px_preset_residue.py`.

### Bug B — n-Gramm-Sperre im Code (behoben, `generators.py`)
`no_repeat_ngram_size=3` ist Teil der Presets. Auf Coding-Tasks kollabierte das Modell:

| Tier (8 harte Tasks) | vor Bug-B-Fix | nach Fix |
|---|---|---|
| BASELINE | 7/8 | 7/8 |
| ACTIVE_MANIFOLD | **0/8** | **7/8** |
| ACTIVE_MANIFOLD_LEAN | **0/8** | **7/8** |
| ACTIVE_MANIFOLD_RELAY:+1 | **0/8** | 1/8 |
| ACTIVE_MANIFOLD_RELAY:-1 | **0/8** | 2/8 |

Fix: `NoRepeatNgramWhenProse` (LogitsProcessor) greift nur, solange kein Code-Fence
(```) geöffnet wurde — Prosa-Schutz bleibt, Code wird nicht mehr zerstört.
Tests: `tests/test_code_aware_ngram.py`. Abschaltbar via `PX_CODE_AWARE_NGRAM=0`.

**Das ist der größte Einzelhebel der Kampagne:** von 0 % auf 88 % Coding unter denselben
Presets, ohne eine einzige Gewichtsänderung.

## 1. Coding (Pass@1, greedy, max_tokens 300–700)

| Modell | Kontext | easy (12) | hard (8) | frontier (7) | frontier + self-repair ×2 |
|---|---|---|---|---|---|
| gemma4-E2B, bf16, px-server | 4 k | 12/12 (100 %) | 7/8 (88 %) | **3/7 (43 %)** | **5/7 (71 %)** |
| Ternary-Bonsai-27B, PTQ1_0/GF3, llama.cpp | 131 k | 12/12 (100 %) | 8/8 (100 %) | **6/7 (86 %)** | nicht gemessen |

Frontier = `parse_ini`, `eval_expr` (rekursiver Parser ohne `eval`), `topological_sort`,
`min_window_substring`, `count_inversions`, `tree_codec`, `roman_roundtrip`.
Fehlerbilder: gemma4 scheitert an Rekursion/Validierung (`AssertionError`), das 27B
nur an einer überstrengen eigenen Römer-Validierung.

Self-repair (Testausgabe zurückfüttern, 2 Runden) hebt gemma4 von 43 % auf 71 % — Faktor 1,65
bei 3,5× Zeit. Das ist reines Scaffolding, keine Modelländerung.

## 2. Kontext (Needle-in-a-Haystack, Tiefe 50 %)

| Modell | belegte Tokens | Treffer | Zeit |
|---|---|---|---|
| gemma4-E2B | 607 / 1 169 / 2 291 | 3/3 | 1,4–3,4 s |
| gemma4-E2B | ~4 600 | **HTTP 500** (Kontextfenster) | — |
| Bonsai-27B | 4 537 / 17 995 / 35 937 | 3/3 | 11 / 47 / 107 s |
| Bonsai-27B | **123 415** | **Treffer** | 565 s (Prefill 220 tok/s) |

Damit ist die Harness-Anforderung **≥ 128 k** erfüllt und end-to-end belegt
(`llama-server --ctx-size 131072`, KV q8_0, 4 Slots unified).

## 3. Geschwindigkeit

| Stack | Decode | Prefill |
|---|---|---|
| gemma4-E2B (12 GB, bf16, Telemetrie-Hooks) | 18–19 tok/s | n. g. |
| Bonsai-27B (PTQ1_0, 131 k KV q8_0) | 16–17 tok/s | 220–400 tok/s |

Erwartung bestätigt: 27B bei 1,58 bit decodiert etwa so schnell wie ein 5B-Modell in bf16
(beide bandbreitenbegrenzt), aber Prefill ist der Engpass langer Kontexte.

## 4. Größe / Quantisierung (gemessen am echten Modell)

Byte-Inventar `google/gemma-4-E2B-it` (567 Lineartensoren, bf16 = 16 bit):
**10,21 GB**. GF(5)-Pack (13 Ququints + fp16-Block-Skalar / 128): **1,68 GB (6,1×)**,
GF(3): **1,20 GB (8,5×)**.

Kernel (realer Bonsai-Tensor 5120×17408, RTX 2060, aus `ququint_quant/results_kernel_v0_2.json`):

| Format | Bytes | ms | max_abs_err |
|---|---|---|---|
| dense bf16 | 178,3 MB | 0,6095 | 0,0529 |
| ptq10_u8 | 19,5 MB | 0,9723 | 0,0529 |
| **gf5_u32** | 29,2 MB | **0,2577** | **0,0** |
| **gf3_u32** | 20,9 MB | **0,2395** | **0,0** |

→ GF5: **2,37× schneller, 6,1× kleiner**; GF3: **2,55× schneller, 8,5× kleiner** — verlustfrei
auf ternärem Material (Superset-Eigenschaft).

## 5. Harness-Nachweis (opencode gegen die eigene API)

`opencode run --standalone -m llamacpp-gf3/ternary-bonsai-27b-gf3 --auto "Create fizzbuzz.py … then run it"`
→ Datei geschrieben (`Write`-Tool), `python3` ausgeführt (`bash`-Tool), Ergebnis korrekt
zurückgemeldet. Provider-Config: `~/.config/opencode/opencode.json`, `baseURL :9931`,
`limit.context 131072`.

## 6. Verdichtete Bilanz (Capability-Dichte)

`Dichte = Pass@1_frontier × Kontext_Tokens × tok/s / Gewichts-Bytes`

| Stack | Dichte | relativ |
|---|---|---|
| gemma4-E2B bf16 (4 k, 10,2 GB) | 1,79 × 10⁻⁶ | 1× |
| gemma4-E2B hypothetisch GF3 (8,5× kleiner) | 1,52 × 10⁻⁵ | 8,5× |
| Bonsai-27B GF3 @131 k (5,6 GB) | 3,13 × 10⁻⁴ | **≈ 175×** |

**Ehrliches Urteil:** 10 000× ist mit diesen Hebeln nicht erreichbar — und die Zahl wäre
auch falsch gerahmt. Erreichbar und gemessen sind: 100 % Coding-Rettung durch einen
Serving-Fix (0 → 88 %), 2× Coding-Frontier durch den 27B-Stack, 32–54× Kontext,
6,1–8,5× kleiner durch GF(5)/GF(3), 2,4–2,6× schnellerer GEMV-Kernel, 1,65× durch
Self-Repair. In Kombination: **~175× Capability-Dichte**, nicht 10 000×.

## 7. Nicht getestet (bewusst)

- **Argon/KenosisLogos** (`/run/media/julian/ML6/KenosisLogos`): 1B Hybrid-MLA-MoE mit
  Latent-KV (320 Floats/Token global, O(1) lokal). Genau der Mechanismus, der 128 k+
  billig macht — aber in diesem Lauf nicht geladen/benchmarkiert (GPU belegt, Konvertierung
  nicht verifiziert). Kandidat für die nächste Runde.
- **GF(5)-End-to-End-Modell**: nur der GEMV-Kernel und die Packgröße sind gemessen; ein
  vollständig GF5-quantisiertes Gemma-4 mit lauffähigem Forward gibt es noch nicht.
- **128 k-Prefill im Harness-Alltag**: 565 s für 123 k Tokens sind gemessen, aber für
 agentische Schleifen noch zu langsam (KV-Cache-Wiederverwendung nötig).

---

## 8. Runde 2 (2026-11-14, nach „Proceed"): Argon/KenosisLogos + Betriebsgrenzen

### 8.1 Argon-1B (`/run/media/julian/ML6/KenosisLogos`) — getestet

| Prüfung | Ergebnis |
|---|---|
| `verify_argon.py` | 6/6 Stufen bestanden (0 Missing Keys, Latent-KV bitgenau: 1,35e-4 Logit-Delta, Generierung stabil) |
| `compare_gemma3_vs_argon.py` | 6/6 Benchmarks bestanden |
| Parameter | **2,246 B** (MoE: 8 Experten × 1728 inter, 2 aktiv; Vocab 262 144) |
| KV/Token (global) | **320 Floats** (256 c_kv + 64 k_rope) vs. gemma-3-1B **512** ⇒ **−37,5 %** |
| KV @ 32 k (global) | **80 MB** vs. 128 MB |
| Lokale Schichten | Ringpuffer bei 512 (sliding_window) ⇒ **O(1)** statt linear |
| Multimodal | Early Fusion (Vision 2D-RoPE + Audio) in 1B — gemma-3-1B kann das nicht |
| Coding easy (12) | **5/12 (42 %)** |
| Coding hard (8) | **0/8 (0 %)** |
| Decode | 16,4 tok/s |

**Lesart:** Argon ist ein *Architektur-Demonstrator* für billigen KV-Speicher, lokale O(1)-Caches
und Early-Fusion-Multimodalität — **kein** Fähigkeits-Upgrade. Auf Coding liegt es auf
gemma-3-1B-Niveau, also deutlich unter gemma4-E2B (12/12, 7/8) und weit unter dem 27B.
Der Wert liegt darin, dass die KV-Ersparnis mit *beliebigen* Gewichten kombinierbar wäre
(gemma4-Klasse + MLA-Cache), nicht in den aktuellen Argon-Gewichten.

Nebenbefund: im Direktpfad generiert Argon ohne `ArgonEntropyFloorLogitsProcessor` immer
exakt 400 Tokens (kein EOS) — die Anti-Collapse-Komponente ist für stabile Längen Pflicht.

### 8.2 Betriebsgrenzen, die die Messung zuerst blockierten

1. **Modell-Disk war zu 100 % voll** (194 MB frei, `/run/media/julian/home_julian`). `llama-server`
   meldete daraufhin irreführend „No such file or directory" statt ENOSPC, obwohl die GGUF-Datei
   intakt war. Modell nach `/var/tmp/bonsai/` verschoben (Root-FS, 18 GB frei) — läuft.
2. **Ubatch-Tuning am 131k-Kontext nicht möglich:** `-ub 1024` braucht 968 MB Compute-Buffer,
   `-ub 2048` 1,42 GB → beide OOM bei 12 GB. Der Default (`-ub 512`) ist auf dieser Karte das Maximum.
   Prefill bleibt damit 220–400 tok/s; 128k ≈ 6–9 min pro kaltem Prompt.

### 8.3 Stand des Harness (verifiziert nach dem Umzug)

- `llama-server` auf `127.0.0.1:9931`, `--ctx-size 131072`, KV q8_0, 11,5 GB VRAM.
- Needle bei 17 995 Tokens erneut getroffen (48 s) auf dem neuen Pfad.
- `opencode run --standalone -m llamacpp-gf3/ternary-bonsai-27b-gf3 --auto` schreibt und
  führt Dateien aus (`hello.py` → `[2, 3, 5, 7, 11]`).
- Der zuvor belegte 123 415-Token-Treffer bleibt gültig (gleiches GGUF, gleiche Flags);
  er wurde vor dem Disk-Umzug gemessen.

---

## 9. GF(5)-Gemma-4 (Runde 3) — gebaut, gemessen, ehrlich eingeordnet

Artefakt: `/home/julian/.cache/huggingface/gf5-gemma-4-E2B-it/`
Code: `ququint_quant/gf5_gemma4.py` (`pack` / `eval`).

### 9.1 Was gebaut wurde
- Jeder 2-D-Gewichtstensor der Sprachpfade → blockweise GF(5) {−2,−1,0,1,2},
  per-Block-fp16-Skalar, 13 Ququints/u32 (LSB zuerst), k = 0,6
  (auf echten Gewichten gemessen optimal: rel-L2 0,316 vs 0,451 bei k = 1,0).
- `embed_tokens_per_layer` (262 144 × 8960, 4,7 GB) als **zeilen-aligned**
  Lookup gepackt: nur angefragte Zeilen werden entpackt.
- `embed_tokens` (geteilt mit lm_head) bleibt bf16, Vision-/Audio-Tower bleiben bf16.
- Zwei Varianten: **GF5** (1 Stufe) und **GF5²** (Residual-Stufe).

### 9.2 Größe

| Variante | Quelle | gepackt | Rest (bf16) | gesamt | Faktor |
|---|---|---|---|---|---|
| GF5 (1 Stufe) | 10,25 GB | 1,37 GB | 1,75 GB | **3,12 GB** | 3,28× |
| GF5² (2 Stufen) | 10,25 GB | ~2,75 GB | 1,75 GB | **4,50 GB** | 2,28× |

### 9.3 Qualität (Logit-Parität gegen bf16, 3 Prompts, letzte Position)

| Variante | rel-L2 (Gewichte) | Cosine | Top-1 | Top-50-Overlap | KL (nats) |
|---|---|---|---|---|---|
| GF5 | 0,315 | 0,72–0,96 | **0/3** | **0,02** | 8,1–14,6 |
| GF5² | 0,114 | **0,990–0,998** | **3/3** | 0,18–0,38 | 0,000–0,42 |

**GF5 mit fünf Stufen ist für dieses Modell zu grob.** Bei 2,59 bit bricht die
Next-Token-Verteilung zusammen (Top-50 praktisch disjunkt, Text unbrauchbar).
Die Residual-Stufe rettet die Qualität auf INT4-Niveau (rel-L2 0,114 vs.
0,117 bei INT4-sym) — kostet aber 5,17 bit, also **mehr** als INT4 (4,0 bit).

### 9.4 Geschwindigkeit — mit integriertem GF(5)-Kernel im Forward

Der Kernel (`ququint_quant/gf5_gemv.py::gfp_matvec`) ist jetzt im Decode-Pfad
von `GF5Linear` eingebunden: `u` liegt im Kernel-Layout (RB, m), `s` als
(nb, m), Decode (Batch 1, ein Token) ruft `gfp_matvec` direkt auf, Prefill
nutzt weiter den Dequant-Pfad.

| Pfad | Decode | VRAM (Peak) | Bemerkung |
|---|---|---|---|
| Dequant pro Token (vorher) | 0,05 tok/s (21 s/Token) | 2,4 GB | reines Speicherformat |
| **Kernel im Forward** | **8,5 tok/s** (0,118 s/Token) | **3,7 GB** | 2 Stufen = 2 Kernel-Calls je Schicht |
| bf16-Referenz | ~18 tok/s | ~10 GB | volle Gewichte |

**170× schneller als der Dequant-Pfad**, 2,7× weniger VRAM als bf16 — aber
noch ~2× langsamer als bf16, weil pro Schicht zwei Matvecs laufen.

### 9.4b Batched GEMM-Kernel für Prefill (`ququint_quant/gf5_gemm.py`)

Der Prefill dequantisiert nicht mehr in numpy, sondern rechnet direkt aus den
Records: pro 128er-Block 10 `tl.dot`-Aufrufe mit K = 16 (13 Digits + 3 Pads,
weil `tl.dot` K ≥ 16 verlangt; 5^16 > 2^32, daher ist ein 16-Digit-Record
nicht möglich).

| T (Prompt-Tokens) | Prefill | Rate |
|---|---|---|
| 87 | 1,26 s | 69 tok/s |
| 396 | 3,87 s | 102 tok/s |
| 1558 | 13,0 s | 120 tok/s |

Vorher (numpy-Dequant, 278 Matrizen) ≈ **20 s pro Prompt, unabhängig von T**.
Der Kernel ist damit **~16× schneller** im Prefill und skaliert mit der
Prompt-Länge.

Ende-zu-Ende (32 Tokens generieren, inkl. Prefill): **5,6 s = 5,7 tok/s**
(vorher 24,8 s = 1,1 tok/s), VRAM-Peak **3,76 GB**, **0 Kernel-Fallbacks**.

Korrektheit: Kernel-GEMM vs. Dequant-Matmul max. rel. Fehler **2,3e-3**
(bf16-Ausgaberundung), Logit-Parität unverändert (Cosine 0,990–0,998,
Top-1 3/3).

### 9.4c Tile-Tuning (statt K = 32) — 1,3× mehr Prefill

Die naheliegende Idee „zwei Records pro Dot (K = 26 → 32)" wurde geprüft und
**verworfen**: die 26 Digits zweier Records liegen im `u`-Vektor nicht
zusammenhängend, der Kernel bräuchte einen `tl.gather` über eine
lane-abhängige Record-Achse — den gibt es in dieser Triton-Version nicht.
Stattdessen wurde die Kachelgröße optimiert (gemessen auf 1536×6144):

| T | BM=64, BT=64 | BM=32, BT=256 |
|---|---|---|
| 512 | 14,0 ms | **10,8 ms** |
| 1558 | 34,6 ms | **30,1 ms** |
| 3000 | 62,6 ms | **40,4 ms** |

Heuristik jetzt: `T < 256 → (64, 64)`, sonst `(32, 256)`. Ergebnis:

| T | Prefill vorher | Prefill jetzt |
|---|---|---|
| 87 | 1,26 s | 1,26 s |
| 396 | 3,87 s | **3,10 s** (128 tok/s) |
| 1558 | 13,0 s | **9,92 s** (157 tok/s) |

### 9.4d Dritte Residualstufe (Qualitätsvariante)

`gf5-gemma-4-E2B-it-s3/` (4,38 GB, 2,34× kleiner), rel-L2 **0,047**
(2 Stufen: 0,114; 1 Stufe: 0,316):

| Variante | rel-L2 | Cosine | Top-1 | Top-50 | KL | Decode | VRAM |
|---|---|---|---|---|---|---|---|
| 2 Stufen | 0,114 | 0,990–0,998 | 3/3 | 0,18–0,38 | 0,00–0,44 | 5,5 tok/s | 3,75 GB |
| 3 Stufen | **0,047** | 0,996–0,999 | 2/3 | **0,24–0,52** | 0,00–0,36 | 4,6 tok/s | 4,37 GB |

Die dritte Stufe verbessert die Verteilungsüberlappung messbar (Top-50
0,18→0,24 und 0,38→0,52), kostet aber 0,6 GB und ~20 % Decode. **Default
bleibt 2 Stufen** (schneller), 3 Stufen liegen als Qualitätsvariante daneben.
Eine vierte Stufe (rel-L2 0,018) bringt nur noch 1,55× Kompression und wurde
nicht gepackt.

Zwei echte Kernel-Bugs wurden dabei gefunden und behoben:
1. **Illegal Memory Access** bei `nb·rpb % rc != 0`: der Kernel maskiert die
   `x`-Loads nicht, las also hinter `x_pad` (z. B. nb=12, rpb=10, rc=16).
   Fix: Tail-Pad von `rc·ND` Elementen in `gfp_matvec`.
2. **Dequant-Cache hielt 8,5 GB**: alle 278 Matrizen gleichzeitig dequantisiert
   (Peak 10,7 GB). Fix: Cache wird nach jedem Forward freigegeben (3,7 GB).

### 9.5 Fazit
GF(5) gewinnt auf der Größen-/Qualitätsfront **nicht** gegen INT4: einstufig
zu verlustbehaftet, zweistufig (5,17 bit) größer als INT4 (4,0 bit) bei
gleichem Fehler. Der Kernel läuft jetzt im Forward und bringt 8,5 tok/s bei
3,7 GB — funktional, aber weder schneller noch kleiner als eine
INT4-GEMM-Referenz. Der verbleibende Vorteil ist strukturell (mul-freie
Akkumulation, exakte {−2,−1,0,1,2}-Symmetrie, kein Nullteiler-Problem),
nicht informativ. Für gemma-4-E2B bleibt es ein Negativ-Ergebnis — jetzt
aber mit lauffähigem Kernel, nicht nur mit einem Packformat.

## 10. KV-Wiederverwendung im Harness (Runde 3)

`llama-server --parallel 1 --cache-reuse 256` (131k ctx, KV q8_0):

| Turn | prompt_n | cache_n | wall |
|---|---|---|---|
| 1 (kalt) | 16 838 | 0 | 44,5 s |
| 2 (identisch) | 4 | 16 834 | **0,4 s (111× schneller)** |
| 3 (+9 Tokens) | 525 | 16 322 | **2,4 s** |

Inkrementelle Harness-Turns kosten damit Sekunden statt Minuten — der
128k-Kontext ist praktisch nutzbar, solange der Prefix stabil bleibt.

---

## 11. Qualitätsrunde (selective bf16 / mehrstufig) und HF-Upload

### 11.1 Was verbessert wurde
- **`--keep-bf16`**: Key-Substrings bleiben bitgenau (Attention, Gates).
- **Mehrstufige Lookup-Quantisierung**: `embed_tokens_per_layer` (4,7 GB)
  wird zeilen-aligned mit Residualstufen gepackt und nur zeilenweise
  dequantisiert. Vorher lag dort rel-L2 0,30 (eine Stufe) — die Per-Layer-
  Inputs jedes Layers waren verrauscht.
- **`_DtypeCastLinear` für alle bf16-Linears**: der Gemma-4-Forward hält
  float32-Buffer (RoPE), sonst prallen fp32-Aktivierungen auf bf16-Gewichte.
- **Embedding-Output-Dtype** wird erzwungen (sonst hebt sich der ganze
  Residualstrom auf fp32).

### 11.2 Grenzkurve (gemessen, greedy-Generierung)

| Variante | Größe | rel-L2 | Cosine | Top-1 | Top-50 | Text |
|---|---|---|---|---|---|---|
| 1 Stufe, alles | 3,1 GB | 0,316 | 0,72–0,96 | 0/3 | 0,02 | unbrauchbar |
| 2 Stufen, alles | 4,5 GB | 0,114 | 0,990–0,998 | 3/3 | 0,18–0,38 | Artefakte |
| 2 Stufen, Attn exakt | 4,2 GB | 0,114 | 0,994–0,999 | 3/3 | 0,20–0,42 | Artefakte |
| 3 Stufen, Attn exakt | 4,7 GB | 0,050 | 0,996–0,999 | 2/3 | 0,24–0,52 | Artefakte |
| **3 Stufen, Attn+Gates exakt, Embedding 3-stufig** | **6,2 GB** | **0,045** | **0,997–0,999** | **3/3** | **0,28–0,58** | **Artefakte (weniger)** |
| 4 Stufen, Attn+Gates exakt | 5,2 GB | 0,020 | – | – | – | Artefakte |

**Befund:** Selbst bei rel-L2 0,018 (≈10 bit) bleiben Token-Artefakte
(verschmolzene Wörter, gelegentliche Wiederholungen). Die Empfindlichkeit
liegt am Modell (Per-Layer-Embedding-Injektion in jeden Layer), nicht an der
Stufenzahl. Mit dem Dequant-Pfad (`PX_GF5_NO_KERNEL=1`) ist der Text
identisch — die Kernel sind also nicht die Ursache.

### 11.3 Upload
`neuralworm/gemma-4-E2B-it-gf5` (HF, public): 6,2 GB, Runtime
(`gf5_runtime.py`, `gf5_gemv.py`, `gf5_gemm.py`), `generate.py`, Manifest,
README mit Messwerten und Grenzen. Kein `AutoModel.from_pretrained`-Drop-in.

---

## 12. Kalibrierte Quantisierung (AWQ-Stil) — versucht, gemessen, ohne Gewinn

Vorschlag war: blockweise amax-Quantisierung durch aktivierungsgewichtete
Skalierung ersetzen (AWQ-Stil), weil die Kanäle starke Ausreißer haben.

**Umsetzung:** `ququint_quant/gf5_calibrate.py` sammelt für alle 277 linearen
Schichten das mittlere |x| je Eingangskanal über 10 diverse Texte (EN/DE/Code).
Der Packer skaliert die Gewichte spaltenweise mit `s = (a/median(a))^0.5`
(geclippt auf [0,01, 20]) und legt den Faktor im Manifest ab; die Runtime
dividiert die Aktivierung vor dem Kernel durch `s` — kein Kernel-Eingriff.

**Ausreißer sind real** (Rechtfertigung für AWQ):
`layers.20.mlp.down_proj` max/median = 51, `per_layer_input_gate` = 304,
p99/median bis 7,4.

| Variante | Größe | rel-L2 | PPL (4 Texte) | Text |
|---|---|---|---|---|
| bf16 | 10,25 GB | – | **2778** | sauber |
| GF5 2 Stufen, ohne AWQ | 4,5 GB | 0,114 | – | Artefakte |
| GF5 2 Stufen, **mit AWQ** | 4,95 GB | 0,130 | **3016 (+8,5 %)** | Artefakte |

**Befund:** AWQ senkt den *aktivierungsgewichteten* Fehler, aber die
Token-Artefakte bleiben; die PPL liegt 8,5 % über bf16. Bei greedy-Decoding
reicht diese Verschiebung, um nach wenigen Tokens in andere Pfade zu laufen.
Der Flaschenhals ist damit nicht die Skalierung, sondern die **Auflösung**:
5 Zustände pro Gewicht (2,59 bit/Stufe) sind für dieses Modell zu grob, und
mehr Stufen kosten linear Größe (7,76 bit für rel-L2 0,045).

**Konsequenz:** Das hochgeladene Modell bleibt die 3-Stufen-Variante
(6,2 GB, rel-L2 0,045, Top-50 0,28–0,58). AWQ wurde nicht hochgeladen, weil
es bei gleicher Textqualität größer ist. Wer echte Kompression mit
brauchbarer Qualität will, kommt an INT4/GPTQ-Kalibrierung mit
Fehlerkompensation (nicht nur Skalierung) nicht vorbei.

---

## 13. Warum GF(5) hier nicht trägt: die Fehler*struktur*, nicht die Fehlermenge

Zwei Kontrollen mit dem bf16-Modell, gleicher Prompt, gleiche Dekodierung:

| Störung | Betrag | Ausgabe |
|---|---|---|
| keine | – | ```python\ndef add(a, b): ... ``` (sauber) |
| iid-Gauss auf alle Linears | rel-L2 0,010 | sauber |
| iid-Gauss auf alle Linears | rel-L2 0,018 | sauber (`return a + b`) |
| iid-Gauss auf alle Linears | rel-L2 0,045 | sauber (Docstring) |
| iid-Gauss auf `embed_tokens_per_layer` | rel-L2 0,020 / 0,050 | sauber |
| **GF(5), 3 Stufen** | rel-L2 0,045 | **Artefakte** (`defdef`, `argsArgs`) |
| **GF(5), 2 Stufen** | rel-L2 0,111 | Artefakte (`def funcadd`, fehlende Newlines) |

Ein **isotroper** Gewichtsfehler von 4,5 % ist für dieses Modell harmlos; der
**quantisierungs-strukturierte** Fehler gleicher Größe nicht. Grund: blockweise
amax-Skalierung mit 5 Stufen erzeugt eine systematische, block-konstante
Verzerrung (Clipping der Tails bei k = 0,6, viele Nullen pro Block) — die
liegt genau auf den Richtungen, auf die das Modell reagiert, während
iid-Rauschen sich über alle Richtungen verteilt.

**Konsequenz:** Der Engpass ist nicht die Kalibrierung (AWQ getestet: PPL
+8,5 %, Artefakte bleiben) und nicht die Stufenzahl (bis 10,35 bit getestet),
sondern das 5-Zustands-Raster selbst. Ein Format mit 5 Stufen pro Gewicht ist
für dieses Modell nicht ausreichend — unabhängig davon, wie gut man die
Skalare legt.
