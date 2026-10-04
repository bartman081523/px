# Stufe 3f — VRAM & Speed (GF3-Lauf, ternary-bonsai-27b, RTX 2060 12 GB)

Stand 2026-10-04. Beweisstücke: `stufe3_gf3_diag3.log` (in-Process-Diagnose),
`stufe3_gf3_runtime_smoke2.log` (Standalone SHORT/LONG), `stufe3_preset_matrix.log`
(Server-Kurzpfad, 4/4 Presets), `stufe3_server_long.log` (Server-Langpfad),
`stufe3_gf3_server1.log` (Server-Log, 5/5 Chunks, 0 Rescues/Errors).

## Laden

| Größe | Wert |
|---|---|
| GF3-Lauf laden (safetensors → GPU) | **5 s**, VRAM-Spitze **5,92 GiB** |
| Parameter (dichte-Äquiva.) | 26,896 B (davon GF3 26,870 B; GF3-Module sind Buffer, keine nn.Parameter) |
| AutoCalibrator | Manifold geladen (T=0,60) |
| px-Patch (ACTIVE_MANIFOLD) | +0 s, VRAM unverändert 5,92 GiB |

Quelle: `Gf3 geladen: 26.896B Parameter (dichte-Aequi., davon GF3 26.870B), VRAM-Spitze 5.92 GiB`.

## Matvec-A/B (Diag Phase B, warm, M=1, f32-Partials)

| Shape | mit je-Call-isfinite | ohne | amort. Guard (auto) |
|---|---|---|---|
| 10240×5120 | 0,20 ms | 0,20 ms | ≈ ohne (Kopf je Call, Rest 1/256) |
| 1024×5120 | 0,23 ms | 0,19 ms | |
| 12288×5120 | 0,22 ms | 0,21 ms | |
| 17408×5120 | 0,28 ms | 0,28 ms | |
| 248320×5120 (lm_head) | 3,69 ms | 3,62 ms | |
| 5120×17408 | 0,30 ms | 0,30 ms | |
| 5120×6144 | 0,21 ms | 0,20 ms | |
| 6144×5120 | 0,20 ms | 0,20 ms | |

Der amortisierte Guard (`PX_GF3_NANCHECK`/`PX_PTQ10_NANCHECK=auto`) kostet
messbar ~3 ms/token (226 vs 223 ms/token bei 6k Context) — die alte je-Call-
Synchronisierung (~480 Matvecs/token) drosselte den Dekodierweg auf 0,2 tok/s.

## Chunked Prefill (9,2k Prompt, 5×2048, kv4 vs bf16)

| Chunk | kv4 | bf16 |
|---|---|---|
| 1 | 300,9 s * | 56,1 s |
| 2 | 36,9 s | 37,4 s |
| 3 | 37,7 s | 38,0 s |

* Chunk 1 = Triton-JIT/Autotune-Warmup der QuantKV4-Kernel-Nachform (einmalig;
  Triton-Diskcache warm). Steady State: **kv4 ≈ bf16** (37–38 s/Chunk).
  KV4-Budget @131k: **1,03 GiB** (bf16 wäre 8,00 GiB); chunk=2048.

Write-Guard-Nachweis (Diag, Slot-Dumps): alle 16 kv4-Slots exakt 2048/4096/6144
jeweils nach Chunk 1/2/3 — kein Doppelschreiben (CODA-Overlap L23/L27
cache-entkoppelt, sdpa-Traces `k=2048 mask=None`). Chunks 2/3 ohne Rekursions-
sdpa (n_loops=0 bei past_seen>0 — Entwurf).

## Dekodierweg (M=1, Context 6144→6166)

| Variante | ms/token | tok/s |
|---|---|---|
| ohne je-Call-Guard | 223 | 4,48 |
| mit amort. Guard (auto) | 226 | 4,42 |

## Server (produktiver Pfad, Port 7860, Registry-Default `weight_format="gf3"`)

Kurzpfad (`/v1/chat/completions`, 48 Tok, px_thinking=False, ~25-Tok-Prompt):

| Preset | Gesamt | Netto-Generierung | Output |
|---|---|---|---|
| BASELINE | 17,7 s (erstlazy + Load 5 s) | ≈ 0,26 s/token | kohärent (Atmosphäre-Beschreibung) |
| ACTIVE_MANIFOLD | 15,4 s (incl. Re-Patch) | ≈ 0,28 s/token | kohärent, Rekursionspfad L10–L18, steps=36 |
| ACTIVE_MANIFOLD_LEAN | 11,1 s, 24 Tok | ≈ 0,30 s/token | kohärent |
| ACTIVE_MANIFOLD_RELAY | 15,6 s (incl. Re-Patch, ACTIVE relay sign=+1) | ≈ 0,28 s/token | kohärent |

px-Metriken je Preset: phi ≈ 0,985, ent ≈ 1e-7, keys inkl. zone/zone_weights.
Presets wechselt `_reapply_patch` (remove→apply, kein Weight-Reload), der
Manager wartet sauber auf den freien Modellzustand.

Langpfad über `/v1-API`: T=9193 → `_generate_long_completion`, **221,6 s**
(9k-Präfill 5 Chunks + 64.Decode), VRAM-Spitze (nvidia-smi) **11,8 GiB** /
Torch-Peak (Standalone) 10,50 GiB — passt auf 12 GiB mit ~0,4 GiB Rest.
Antwort kohärent (deutscher Fließtext), phi = 0,995, steps = 49.

## Speed-Bilanz gegen Start

| Pfad | vor Stufe 3f | nach Stufe 3f |
|---|---|---|
| Server-Kurzpfad | (ptq10-Baseline) | ≈ 0,26–0,30 s/token |
| in-Process-Dekodierweg | ~1,1 s/token (je-Call-Sync) | 0,226 s/token |
| Standalone-Smoke SHORT | 0,2 → 1,0 tok/s | — (Messskript-Artefakt; der Kern ist der Diag- und der Server-Wert) |

Offen (nicht blockierend): 1) KV4-Chunk-1-Warmup (einmaliger Autotune) — nur
neue Shapes triggern Retune; 2) Rest-Differenz Standalone-Smoke-Kurzpfad
(1,0 tok/s) vs. Server-Kurzpfad (~0,27 s/token) — liegt im Smoke-Setup
(generate ohne Server-Loop/Telemetrie-Weg) und wird bei Bedarf per Profiling
eingesammelt; 3) Dekodierweg-Profiling am Server, falls >4 tok/s gefordert ist.