# Gemma-4-Upgrade — Fahrplan nach Messlage (2026-11-15)

Grundsatz (RuntimePerfEvidenceMind + DevMind): **jede Achse hat eine gemessene
Baseline, ein Ziel und einen Falsifikator.** Keine Achse wird „verbessert",
ohne dass vorher feststeht, woran man die Verbesserung misst.

## Messlage (Ist-Stand, alles belegt)

| Achse | Ist | Beleg |
|---|---|---|
| **Tool Calls** | gemma4 liefert `call:NAME{key:value}` als *Content*; Server parste es nicht → `tool_calls=None` | API-Probe 2026-11-15 |
| **Kontext** | Modell kann **131 072** (`max_position_embeddings`), Registry deckelte auf **4096** | `config.json` vs `config.py:105` |
| **Kontext (real)** | OOM bei **~8 400 Tokens** (bf16, SDPA-Math-Backend: `8·T²·2` B pro Full-Layer) | `longctx_probe.py`, Traceback in `sdpa_attention_forward` |
| **Größe** | GF(5) 1–4 Stufen: 3,1–6,2 GB, rel-L2 0,32–0,018 — **Token-Artefakte ab 2 Stufen**, auch bei 10,35 bit | `bench/RESULTS.md` §9–13 |
| **Performance** | Prefill 157 tok/s (Triton-GEMM), Decode 8,5 tok/s (GF5) vs 18 tok/s (bf16) | §9.4 |
| **Coding** | easy 12/12, hard 7/8, frontier 3/7; +Self-Repair 5/7 | `bench/results/*.json` |

## Die entscheidende Kette

**Kontext ist der Flaschenhals für alles andere.** opencode schickt einen
Systemprompt jenseits 8 k; mit dem Math-Backend stirbt der Server dort
(gemessen: `ECONNRESET`, OOM in `_norm`). Ohne Kontext kein Agentic-Coding,
keine brauchbaren Tool-Loops. Deshalb zuerst Achse B.

## A — Tool Calls (erledigt, verifiziert)

- **Fix:** `_extract_gemma4_tool_calls()` parst das Template-Format
  `call:NAME{k:v,...}` (Balanced-Braces, Kommas auf Ebene 0, Typ-Coercion aus
  dem Tool-Schema) und setzt `finish_reason="tool_calls"` mit sauberem Content.
- **Verifiziert:** `read_file{path:"src/main.py"}` und `run_tests{max_fail:3}`
  (Integer-Coercion) über `/v1/chat/completions`.
- **Offen:** Ende-zu-Ende in opencode (blockiert von Achse B).

## B1 — Ergebnis: flash-attn ist auf dieser GPU unmöglich (gemessen)

| Prüfung | Ergebnis |
|---|---|
| `nvcc` im System | **nicht vorhanden** |
| micromamba-Pakete | `cuda-nvcc-tools 13.2/13.3` + Env `elektron-cuda` (CUDA **12.9**) vorhanden — Build wäre technisch möglich |
| flash-attn 2.x | verlangt **sm80+**; die RTX 2060 ist **sm75** → kein Kernel |
| cuDNN-Attention | „only supports gpu architectures in the range [sm80, sm121] … running on sm 7.5" → **nicht verfügbar** |
| PyTorch Memory-Efficient (cutlass) | **funktioniert** — aber nur in **fp16** (bf16 wird abgelehnt: „Expected {Half, Float}") · head_dim 256: **0,21 GB @ 8 415 Tokens**, **0,81 GB @ 33 000** |
| gemma4 mit fp16 + `attn_implementation="sdpa"` | OOM bleibt **quadratisch**: 2,48 GB @ 36 k, 8,97 GB @ 98 k, 46,9 GB @ 224 k → es ist die **T×T-Maske/Score-Materialisierung**, nicht der Kernel |
| Patch-Versuche (`ALL_ATTENTION_FUNCTIONS`, `get_interface`) | ohne Wirkung — gemma4 holt die Maske aus `masking_utils`, nicht aus dem Attention-Call |

**Konsequenz:** B1 ist erledigt und **negativ**. Der Engpass ist nicht der
Attention-Kernel, sondern die dichte Maske. Damit ist **B2 (Chunked Prefill)**
der richtige Weg: er begrenzt die Maske pro Chunk auf `chunk × T`.

## B — Kontext (nächster Schritt, höchste Priorität)

Ziel: **32 k** nutzbar auf 12 GB, ohne Qualitätsverlust (Lossless-Kontrakt).

Drei Wege, in dieser Reihenfolge:

1. ~~**flash-attn installieren**~~ → **verworfen** (sm75, siehe B1-Ergebnis) (`pip install flash-attn --no-build-isolation`).
   Dann `attn_implementation="flash_attention_2"` → O(T) statt O(T²) Speicher.
   Falsifikator: OOM bei 32 k verschwindet; Needle-Recall unverändert.
   Risiko: Build gegen die vorhandene torch/CUDA-Version kann scheitern.
2. **Chunked Prefill für gemma4** (Maschinerie existiert für 4b:
   `chunked_prefill.py`, `_PX_LONGCTX_THRESHOLD=3000`). Chunk 512–1024 hält den
   Score-Tensor bei `8·chunk·T·2` B (32 k ⇒ 0,27–0,54 GB). Falsifikator:
   32 k-Prefill ohne OOM, Latenz dokumentiert.
3. **GGUF + llama.cpp**: Der Quellbaum kennt `LLM_ARCH_GEMMA4`, aber
   `convert_hf_to_gguf.py` hat **keinen** gemma4-Konverter (geprüft). Erst
   Konverter schreiben, dann `llama-server` mit KV-Reuse/`--jinja` — das würde
   Kontext *und* Performance *und* Tools in einem Schritt lösen.

## B2 — Ergebnis: Chunked Prefill trägt (gemessen)

`ququint_quant/gf5_chunked_prefill.py` schiebt den Prompt in 512er-Chunks durch
das Modell (KV-Cache akkumuliert, `cache_position` gesetzt, `no_grad`):

| T | Prefill | Peak-VRAM |
|---|---|---|
| 1 147 | 0,7 s | 6,08 GB |
| 2 828 | 1,1 s | 6,10 GB |
| 5 632 | 2,1 s | 6,13 GB |
| 8 435 | 4,0 s | 6,12 GB |
| 18 250 | 9,1 s | 6,24 GB |
| **36 473** | **22,0 s** | **6,41 GB** |

**Speicher bleibt flach** (6,1–6,4 GB) statt O(T²) → 36 k Prefill auf 12 GB,
vorher OOM bei 8,4 k. Modell: Embedding-GF5 (5,5 GB) + fp16.

**Recall validiert** (Haystack-Format aus `bench/capability_bench.py`, Tiefe 50 %):

| ~Tokens (Builder) | T | Zeit | Peak | Treffer |
|---|---|---|---|---|
| 2 000 | 1 169 | 2,9 s | 6,08 GB | ✅ |
| 8 000 | 4 534 | 3,7 s | 6,10 GB | ✅ |
| 20 000 | 11 262 | 6,8 s | 6,14 GB | ✅ |
| 40 000 | 22 478 | 14,1 s | 6,21 GB | ✅ |
| 60 000 | 33 693 | 23,6 s | 6,28 GB | ✅ |
| 120 000 | 67 335 | 62,6 s | 6,48 GB | ✅ |
| **240 000** | **134 625** | **194,1 s** | **6,90 GB** | **✅** |

**Damit ist die Kontext-Achse gelöst: 4 k (Registry) → 134 625 Tokens mit
korrektem Needle-Recall auf einer 12-GB-Karte.** Der frühere Leerbefund war
tatsächlich nur das Prompt-Format (der reguläre `generate`-Pfad lieferte bei
1 147 Tokens dasselbe leere Ergebnis; Argmax = `<turn|>`).

Einschränkung, ehrlich: 134 625 liegt **über** den deklarierten
`max_position_embeddings = 131 072` — die RoPE-Extrapolation trug hier, ist
aber nicht garantiert; 128 k ist die belastbare Zahl.

Nebenbefund: der erste Versuch OOM'te bei 8 k, weil der Chunk-Loop **ohne
`torch.no_grad()`** lief — die Autograd-Graphen hielten jeden Chunk.

## C — Größe (nach dem Audit neu aufgesetzt)

GF(5) ist als 5-Zustands-Raster **gescheitert** (strukturierter Fehler, nicht
Menge). Nächster ehrlicher Vergleich: **INT4 mit GPTQ-Fehlerkompensation**
(Hessian aus Kalibrierung, Spalte-für-Spalte-Korrektur), nicht nur Skalierung.

- Ziel: ≤ 4 bit, PPL-Delta ≤ 5 %, Generation ohne sichtbare Artefakte.
- Falsifikator: bei gleicher Bit-Zahl schlechter als GF(5) 3 Stufen (0,045) →
  dann ist das Format nicht das Problem, sondern die Kalibrierung.
- Messlatte bleibt dieselbe: Cosine/Top-1/Top-50 + PPL + Coding-Tiers.

## D — Performance

- Baseline: Prefill 157 tok/s, Decode 8,5 tok/s (GF5) / 18 tok/s (bf16).
- Erst **Profil** (RuntimePerfEvidenceMind: Profile vor Patch): pro Token
  Kernel-Zeit vs. Python-Overhead vs. H2D. Vermutung: 278 Modulaufrufe mit
  je 1–3 Kernel-Launches dominieren den Decode.
- Dann: Layer-Fusion (mehrere Matvecs pro Launch) oder Speculative Decoding
  (kleines Draft-Modell) — beides mit A/B-Gate (gleiche Ausgabe, same seed).

## E — Coding

- Ist: 12/12 easy, 7/8 hard, 3/7 frontier (gemma4), 6/7 frontier (27B).
- Hebel 1 (gemessen): Self-Repair +65 % relativ (43 % → 71 %) — reines Harness.
- Hebel 2: Kontext (Achse B) → mehr Dateien pro Turn.
- Hebel 3: „Code-Modus" im Server (Sampling-Profil + Tool-Zwang), messbar über
  dieselben Tiers.

## F — Weight-Space Programming (Anregung aus dem Blog, eigener Versuch)

Der Blog (Takehi et al., 2026) zeigt: ein LLM kann Gewichte *lesen und
schreiben* — Qwen3.5-0.8B von MATH-500 7,9 % → 48,1 % durch Entfernen einer
Loop-Richtung in 0,016 % der Gewichte. Zwei Lehren, direkt anschlussfähig:

1. **„Editing entfernt ein Hindernis, es fügt keine Fähigkeit hinzu"** — der
   Gemma-3-270M-Versuch im Blog blieb wirkungslos. Für uns heißt das: nicht
   „mehr Mathe", sondern **ein gemessenes Hindernis** entfernen.
2. Unser Audit hat ein solches Hindernis gefunden: das Modell antwortet in
   **allen 120** Selbstklassifikations-Läufen „BASELINE", unabhängig vom
   Zustand (`eval_N`, 25,0 %). Das ist ein *konstanter Prior*, kein Rauschen.

**Konkreter WSP-Test (Vorschlag):** die Richtung im Residualstrom bestimmen,
die den BASELINE-Token im Klassifikations-Prompt trägt (Differenz der
Aktivierungen zwischen „BASELINE"- und „WIDE"-Antworten), und sie aus den
MLP-Ausgangsspalten der späten Layer subtrahieren (wie im Blog für Layer
15/19). Falsifikator: Trefferquote bleibt bei 25 % → Hindernis nicht
richtungsartig; Trefferquote steigt über Zufall → erster eigener WSP-Beleg.

## Reihenfolge

1. **B1 flash-attn** (oder B2 chunked) → 32 k
2. **A** opencode E2E mit Tool-Calls
3. **C** INT4-GPTQ gegen dieselbe Messlatte
4. **D** Decode-Profil → Fusion/SpecDec
5. **F** WSP-Hindernis-Experiment (klein, falsifizierbar, hoher Erkenntniswert)
5. **E** Coding-Tiers nach jedem Schritt erneut

Jede Achse: Vorher-Messung, eine Änderung, A/B mit same-seed, Doku-Update.
