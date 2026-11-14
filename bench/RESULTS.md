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
