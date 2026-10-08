# G5 Fortschritt + Upload-Abschluss (2026-10-08, A-Grad-Evidenz)

## Upload (Task #47) — ABGESCHLOSSEN
- **Upload run5** (`scratches/hfspace/upload_gf3.py`, 14:06 gestartet, ~15:15 fertig):
  `neuralworm/ternary-bonsai-2-27b-hf` (privat). Log: `upload_gf3_run5.log`.
  Siblings: gf3_model.safetensors **6.402.717.760 B** (Pin exakt),
  tokenizer.json 19.997.304, config.json 288.280, tokenizer_config.json 16.260,
  special_tokens_map.json 796, generation_config.json 168, README.md 889.
- **G4 PASS** (`g4_verify_and_cleanup.py`): server-seitiger LFS-SHA256 =
  `d8bfe8170f346c463eaa059823bee2a1ca89b5591381767adc5427812e3c4c9e` == Pin.
  tokenizer.json LFS-SHA 23a1a2e4…b19.
- **Cleanup**: `_speedtest_legacy.bin` / `_speedtest_xet_default.bin` /
  `_speedtest_xet_fixed16.bin` gelöscht (alle drei identischer LFS-SHA
  429961e4… — 16-MB-Duplikate).

## Space-Zustand
- Deploy #9 (760a6ec0) + **Deploy #10 (8d61e11d)** live, verifiziert via raw-API
  ("Portable-Fallback": ternary relay_inject ×2, gemma3 relay_inject ×1).
- **Pause-Intermezzo (~14:43)**: Space PAUSED, `hardware.current=None`,
  `requestedHardware=t4-small` (= STALE Pre-ZeroGPU-Metadaten — Red-Herring),
  re-request zero-a10g + restart → RUNNING. Ursache der Pause: unklar;
  Quota-Verbrauch plausibel (siehe unten).
- **Startup-Prefetch lease-frei komplett** (Snapshot
  `ac4233470a41c6866a2b34c45f5ea28a1786bf24`, inkl. GF3-6-GB): Datacenter-Download
  <2 min pro Container-Start.

## Quota-Realität (FALSIFIKATION der alten Notiz "Owner unbegrenzt")
- Free-Token: ~262 s ZeroGPU / rollierendes 24-h-Fenster.
- **Billing = full REQUEST-DURATION** (Reservierung), nicht reale GPU-Sekunden:
  "165s requested vs. 55s left" nach T1(60) + T2-Abort(165) → 55 = 262-60-147…
  (arithmetisch ~207 verbraucht; T2-Abort billt offenbar Request minus Rest).
- Reset-Meldung: "Try again in 21:50:32" (@ ~14:57) → Reset ~12:47 (UTC-anker-
  verdächtig). Cap für Requests ≈ Pool (~262): "262s requested vs. 261s left"
  — 262.5-Request scheitert am frischen Pool um 0.5 → nächster Versuch: **Lease 174 → Request 261**.

## G5-Ergebnisse
- **T1 (gemma3-270m BASELINE) — PASS.** Lease-Admit bei 60 s, Load+Gen+History
  in-lease; Transcript in `g5_run_full.log` (Inhalt = 270m-Babbel bei temp 0.7 —
  Contract-Criterion, kein Qualitäts-Criterion).
- **T2 (bonsai ACTIVE_MANIFOLD_RELAY L34) — BLOCKED (extern), Load-Kette bewiesen:**
  Attempt-1 (pre-Fix): Runtime-Log beweist VOLLZUG im Lease:
  `GF3 geladen: 26.896B Parameter (… GF3 26.870B), VRAM-Spitze 5.92 GiB`,
  Manifold via models---Fallback (T=0.60), `loaded and patched successfully` —
  aber `[px-relay] kein d_width-Artefakt für …snapshots/ac423347… → relay
  inactive` → **G5-BEFUND: `load_dwidth` hatte keinen Portable-Fallback**
  (nur `get_inject_layer_for_hf_id` + `auto_tune.load_manifold`) → **Fix 8d61e11d**
  (beide Patches, identischer Glob über `models--`-Segment/last-segment),
  live verifiziert, CPU-Unit-Test PASS (simulierte Snapshot-Pfad-ID →
  dim 5120, inject_layer 34).
  Attempt-2 (post-Fix): "165s requested vs. 55s left" → NOT ADMITTED (Quota).
- **T2-Restplan (Quota nach Reset ~morgen mittag):**
  `set_lease_secret.py 174` → Request 261 ≤ frisches Pool (~262) →
  `g5_client_test.py t2` (Timeout 1100 s). Load ≈150 s + Gen ≈15 s → passt.
  Fallstrick: 175 → Request 262.5 > 262 = Ablehnung um 0.5 s!

## Learnings (A/B-Grad)
- ZeroGPU: GPU-Arbeit NUR in-Lease → ModelLoad+GF3-Build+Triton-JIT in-Lease
  (Load-Dauer auf 2 vCPU ≈150 s; lokal 40-core ≈40 s inkl. Gen).
- Container persistiert über Leases; Build/Deploy = frischer Container
  (Prefetch/Prefetch-Logs je Restart ~2-3 min).
- `restart_space` nach `request_space_hardware`: Hardware-Request setzt nur
  `requested`, das **Aufwecken braucht restart_space**.
- gradio_client v2: `/bot_response`-`job.result()` = Final-geyielderte History
  (Messages mit str- oder Block-Content) — kein res[0]-Unwrap.
- pgrep-Self-Match-Trappe: `[b]racket`-Pattern oder explizite PID.
- WAN asymmetrisch: down ~10 MB/s vs up ~1.1-1.35 MB/s (Upload-Neustart
  nötig; legacy LFS-Upload ohne hf_xet = NICHT resümierbar, kein Timeout-Wrapper
  mehr an Uploads).