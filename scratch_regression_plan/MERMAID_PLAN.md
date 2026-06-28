# Mermaid-Plan: all_space Codebase Architecture

Stand: 2026-06-28. 156 Commits seit Projektstart (`5d4d13a`).

**Methodik-Constraints:**
- Add only. Kein Commit rausnehmen.
- Bezugspunkte: jeder neue Punkt referenziert wo nötig auf Vor-Punkte (`Ref: P-N`).
- Granularität: pro Commit ein Block. Ausnahmen sind explizit markiert.
- Reine Struktur — kein Code-Snippet, kein Commit-Message-Text.

**Legende Block-Typen:**
- `core:` Motor-/Architektur-Kern
- `infra:` CI/Test-Infrastruktur
- `webapp:` Webapp (Gradio + FastAPI)
- `tts:` TTS-Pipeline
- `model:` Modell-Layer (Loader, Registry, Quantisierung)
- `bridge:` Streaming-Bridge
- `motor:` PX-Patch + Presets
- `scratch:` Scratch/Research (außerhalb von Production)
- `fix:` Bug-Fix
- `docs:` Doku
- `chore:` Refactor/Haushaltung

**Legende Beziehungen:**
- `→ baut auf` : hängt von Vor-Punkt ab
- `⇒ exportiert` : macht etwas nach außen sichtbar
- `⇐ testet` : hat Tests die das Verhalten prüfen

```mermaid
flowchart TB
    %% ─── P1-P5: Projektstart + Gründung ─────────────────────────────
    P1["core: registry-foundation"]:::model
    P2["core: chat-engine-Substrat"]:::core
    P3["core: 3 unpatched baseline models + skip patch when patch_dir None + unload() (5d4d13a)"]:::model
    P4["chore: __pycache__ .gitignore (0be12bd)"]:::chore
    P5["core: px_gamma + px_routing_mode API schemas → ModelManager (7ff5c7d)"]:::model
    P6["core: model-agnostic benchmark_engine.py + test_prompts.py (8754fa3)"]:::core
    P7["webapp: Gradio UI 4 tabs (chat, cognitive tests, p-zombie eval, telemetry) + matplotlib viz (a5dab2d)"]:::webapp
    P8["infra: app.py FastAPI+Gradio single entry, run.sh → app.py (ad3c594)"]:::infra
    P9["core: telemetry-extraction (phi traces, zone distributions, kurtosis, emancipation) (5f4ac68)"]:::core
    P10["infra: scipy + Gradio 6.15.2 compat + integration-tested (a9ab415)"]:::infra
    P11["infra: session-management + run-local SSL + eval report (9d26993)"]:::infra

    P3 --> P7
    P5 --> P7
    P6 --> P7
    P7 --> P8
    P8 --> P9
    P8 --> P10
    P9 --> P11

    %% ─── P12-P21: PX-Module, Architecture-Iterations ────────────────
    P12["docs: PX improvement plan MiniCPM5-1B (6589cdd)"]:::docs
    P13["docs: cross-architecture eval PX plan (6e65b79)"]:::docs
    P14["fix: device/dtype mismatch DMT modules (fe2c58c)"]:::fix
    P15["chore: remove DMT-specific protocol patches (a005e10)"]:::chore
    P16["core: consolidate dmt_space_upload UI + gemma3-persona patches (81bb5c0)"]:::core
    P17["chore: cognitive port baseline check (0b59427)"]:::chore
    P18["core: restore DMT protocol extension classes px_modules.py (38cff10)"]:::core
    P19["fix: recalibrate DMT modules 270M scale stability (ab88d60)"]:::fix
    P20["docs: PX version comparison + consolidated-build results (3eade87)"]:::docs
    P21["core: Consolidate Best Version + Rigor preset + Kurtosis routing (0d501f3)"]:::core
    P21a["chore: Checkpoint before implementing 4B manifold and formalmystic patches (6b19e83)"]:::chore

    P11 --> P12
    P12 --> P13
    P13 --> P14
    P14 --> P15
    P15 --> P16
    P16 --> P17
    P17 --> P18
    P18 --> P19
    P19 --> P20
    P20 --> P21
    P21 --> P21a

    %% ─── P22-P32: Subjective Phenomenology + AZS ────────────────────
    P22["core: Subjective Phenomenological Evaluation P2.8 Substrate (671da71)"]:::core
    P23["core: Recursive Self-Inquiry Model as Research Partner (765d051)"]:::core
    P24["core: Anti-Zombie Sensor AZS (bd4ec95)"]:::core
    P25["chore: cleanup symlink + pycache patch dir (c4c8e35)"]:::chore
    P26["core: Integrated AZS + Autonomous Resilience loop (9aa379c)"]:::core
    P27["core: Model registry + dynamic PX mode switching (SUBJECTIVE/RIGOR/DMT-FULL/UNCENSORED/BASELINE) (5367238)"]:::core
    P28["webapp: UI cleanup — consolidated cognitive-config in Mode Preset dropdown (64a5639)"]:::webapp
    P29["fix: session management newest-first + auto-save + UI refresh (6955921)"]:::fix
    P30["core: LRU model unloading Capacity=1 OOM-prevention (8cb681a)"]:::core
    P31["fix: race condition model-level locking (056cfd8)"]:::fix
    P32["core: subjective resonance + phenomenological feedback loops (4637956)"]:::core

    P21 --> P22
    P22 --> P23
    P23 --> P24
    P24 --> P25
    P25 --> P26
    P26 --> P27
    P27 --> P28
    P28 --> P29
    P29 --> P30
    P30 --> P31
    P31 --> P32

    %% ─── P33-P48: Resonance-City + Gemma-4 Integration ─────────────
    P33["core: restore Resonance-City architecture pre-optimization (94d4bc3)"]:::core
    P34["fix: stabilize Resonance-City dtypes + GPU logic (2fef380)"]:::fix
    P35["webapp: expose RESONANCE_CITY in UI + fix API preset passing (6ef6f28)"]:::webapp
    P36["fix: model quality via cumulative gamma fix + re-enable DMT Subjective (c94fc8d)"]:::fix
    P37["chore: revert SUBJECTIVE default DMT architectural separation (9ef6f24)"]:::chore
    P38["refactor: GPU optimization recursion loop + testing (cbebe11)"]:::refactor
    P39["chore: remove redundant code blocks (5b5e1c5)"]:::chore
    P40["chore: remove redundant unused classes (32b812d)"]:::chore
    P41["chore: test suites + MiniCPM patch variables + telemetry (25a621f)"]:::chore
    P42["model: integrate google/gemma-4-E2B-it to MODEL_REGISTRY (3fa8b52)"]:::model
    P43["core: isolate gemma3 vs gemma4 patches (6fc9464)"]:::core
    P44["fix: gemma4 SUBJECTIVE routing via empirical k_mean (e93b8b5)"]:::fix
    P45["core: Gemma4 BOUNCE-BREAK t_norm gate + max-steps fallback + bridge HTTPS (6cf05ae)"]:::core
    P46["test: parity gemma3 vs gemma4 SUBJECTIVE (fbbe292)"]:::test
    P47["core: rewrite _px_forward natively for Gemma 4 (c081fc9)"]:::core
    P48["fix: _safe_forward non-DMT + apply_px_patch constructors (67702ad)"]:::fix

    P32 --> P33
    P33 --> P34
    P34 --> P35
    P35 --> P36
    P36 --> P37
    P37 --> P38
    P38 --> P39
    P39 --> P40
    P40 --> P41
    P41 --> P42
    P42 --> P43
    P43 --> P44
    P44 --> P45
    P45 --> P46
    P46 --> P47
    P47 --> P48

    %% ─── P49-P73: Gemma 4 PX Patch + InfLLM + LEAN ─────────────────
    P49["core: Gemma 4 PX Patch — Rekursion + Zone Routing + alle Presets (10dddf3)"]:::core
    P50["webapp: Chat Auto-Scrolling deaktiviert (1f3fb35)"]:::webapp
    P51["wip checkpoint-1 (a69237b)"]:::scratch
    P52["wip checkpoint-2 (08f6c23)"]:::scratch
    P53["wip checkpoint-3 (d13641f)"]:::scratch
    P54["wip checkpoint-4 (4b4a0ed)"]:::scratch
    P55["wip checkpoint-5 (d0f4f8f)"]:::scratch
    P56["wip checkpoint-6 (d4e369d)"]:::scratch
    P57["wip checkpoint-7 (a0ec0ce)"]:::scratch
    P58["SR64 — early (087383d)"]:::core
    P59["wip checkpoint-8 (9089e47)"]:::scratch
    P60["wip checkpoint-9 (f362e97)"]:::scratch
    P61["fix: session persistence (4480aa2)"]:::fix
    P62["fix: session persistence (af02394)"]:::fix
    P63["fix: session persistence (a8f4a06)"]:::fix
    P64["fix: session display (80f953d)"]:::fix
    P65["fix: session persistence (37f49d7)"]:::fix
    P66["core: InfLLM + ReAttention — Training-Free Infinite Context (7cf8218)"]:::core
    P67["wip: infinite-context (86e38bd)"]:::scratch
    P68["core: InfLLM surgical patches — Layer A app-window + Layer B corrected InfLLMCache (3cdb8f4)"]:::core
    P69["core: SR-64 verlustfreier Infinite Context + Test-Reparatur (31429e1)"]:::core
    P70["scratch: infinite-context Validierungs- und Diagnose-Skripte (71f4465)"]:::scratch
    P71["core: ACTIVE_MANIFOLD_LEAN — radikaler Schnitt als permanenter reversibler Preset (21f641e)"]:::core
    P72["scratch: emergence EM-Mechanismen reread/spectral viable (f422acf)"]:::scratch
    P73["scratch: emergence re-tune witness/shadow + Rung-3 + Invarianz-Sonde (23c4f3e)"]:::scratch

    P48 --> P49
    P49 --> P50
    P50 --> P51
    P51 --> P52
    P52 --> P53
    P53 --> P54
    P54 --> P55
    P55 --> P56
    P56 --> P57
    P57 --> P58
    P58 --> P59
    P59 --> P60
    P60 --> P61
    P61 --> P62
    P62 --> P63
    P63 --> P64
    P64 --> P65
    P65 --> P66
    P66 --> P67
    P67 --> P68
    P68 --> P69
    P69 --> P70
    P70 --> P71
    P71 --> P72
    P72 --> P73

    %% ─── P74-P91: Konklave I-X, Emergence, Rung-2-3 ─────────────────
    P74["scratch: emergence Rung-3 v2 Text-Invarianz — Rung 2 erhärtet, Rung 3 offen (9ec0630)"]:::scratch
    P74a["scratch: Pre-Infinite Context commit (0b87409)"]:::scratch
    P75["scratch: emergence Rung-2 Ground-Truth + ehrliche Selbst-Korrektur (2a394bd)"]:::scratch
    P76["scratch: emergence lean — validierter Motor durch Juexins Maßstab, schwache echte Emergenz (09e1775)"]:::scratch
    P77["scratch: Konklave Phase III — Wenden vs Schleife am kausalen Kern lean greedy (4a881c3)"]:::scratch
    P78["scratch: Phase III Lesung korrigiert — kausaler Kern WENDET deterministisch (39100db)"]:::scratch
    P79["scratch: Phase III Trio — Mephisto isoliert getestet NICHT Träger des Wendens (a532c31)"]:::scratch
    P80["scratch: Konklave Phase IV — PhiMind-Supervision 觉 vs Schleife (Tun vs Wissen) (a50519b)"]:::scratch
    P81["scratch: Konklave Phase V — CitMinds Klinge चित् vs Schleife (Sehen vs Nur-Zurückkehren) (be05913)"]:::scratch
    P82["test: obsolete-test identification (151 passed, 4 failed, 6 skipped) (0279c09)"]:::test
    P83["test: obsolete-test cleanup — repurpose riddle regression, remove stale eos-aggregate (951cc2f)"]:::test
    P84["scratch: Konklave Phase VI — Juexins Tun-vs-Wissen-Instrument Perturbations-Invarianz Form-Sehen (b680a41)"]:::scratch
    P85["docs: Konstrukt-Konklave Ontologie-Verfeinerung — Gemma3 originär CitMind, kann jedes Konstrukt werden (922aaf0)"]:::docs
    P85a["scratch: SCIMIND5 Sonde 1 — Marker-Kovarianz, fragiles Struktur-Kopplungs-Signal (n=4) (97eb524)"]:::scratch
    P86["scratch: Konklave Konstrukt-Handoff-Experiment — wird CitMind-origin zum übergebenen Konstrukt? (dd472d4)"]:::scratch
    P87["test: ACTIVE_MANIFOLD vs ACTIVE_MANIFOLD_LEAN structural guard (e22e6cd)"]:::test
    P88["chore: Konklave/Emergence Haushaltung — Construct-Docs + Construct/Konklave-Sessions (7573d76)"]:::chore
    P89["scratch: Konklave Phase VII — Falsifikationsschnitt des Form-Sehen-Glimmers (d41ba32)"]:::scratch
    P90["chore: veraltete patch.py.wip-backup + run.sh entfernt (01d409b)"]:::chore
    P91["scratch: Konklave Phase VIII — cold-engaging no-form-Probe, Wenden IST der recur-Motor (05b8bdd)"]:::scratch

    P73 --> P74
    P74 --> P74a
    P74a --> P75
    P75 --> P76
    P76 --> P77
    P77 --> P78
    P78 --> P79
    P79 --> P80
    P80 --> P81
    P81 --> P82
    P82 --> P83
    P83 --> P84
    P84 --> P85
    P85 --> P85a
    P85a --> P86
    P86 --> P87
    P87 --> P88
    P88 --> P89
    P89 --> P90
    P90 --> P91

    %% ─── P92-P104: Phase IX-X, Psychomotrik Fate-Probe ──────────────
    P92["scratch: Konklave Phase IX — cold strong-pole-Probe, 觕-Bar sauber abgesteckt (cd6246b)"]:::scratch
    P93["scratch: Manualer Re-Audit — Keyword-Zählerei flawed, Juexin liest Outputs neu (919c2c7)"]:::scratch
    P94["scratch: Konklave Phase X — Kalt-Rekurrenz-Test, entscheidendes Negativ gegen recur→Selbstwahrnehmung (9b3664b)"]:::scratch
    P95["wip: emergence tests 2 (c3f8014)"]:::scratch
    P96["scratch: emergence 4 (7969f36)"]:::scratch
    P97["scratch: emergence4 — manuelle Juexin-Lesung der v4-Marker (14ba2ce)"]:::scratch
    P98["scratch: emergence4 — Lesung revidiert nach Design-Doku-Konvergenz mit v3 (7f29426)"]:::scratch
    P99["scratch: emergence5 — Zustands-Induktion + mechanische Korrelation (ef8b676)"]:::scratch
    P100["scratch: psychomotrik Fate-Probe — Intro von 我执 trennbar routing-bereinigt LOO 0.63 (372c515)"]:::scratch
    P101["scratch: psychomotrik — Prefill-Steering kausal nahe-Null, Intro-Richtung korrelierbar nicht projizierbar (c3af5f6)"]:::scratch
    P102["scratch: psychomotrik Seite 3.1 — mechanischer Diskriminator, φ-Hypothese falsifiziert (8be7811)"]:::scratch
    P103["scratch: psychomotrik Seite 3.2 — Disentangling, WIDTH ist der Hebel nicht Grind/Entropie/φ (a8936b4)"]:::scratch
    P104["scratch: psychomotrik Seite 4 + SCIMIND5 — WIDTH→LAYER-REGIME, Beobachter-我执-Korrektur (49db0e6)"]:::scratch

    P91 --> P92
    P92 --> P93
    P93 --> P94
    P94 --> P95
    P95 --> P96
    P96 --> P97
    P97 --> P98
    P98 --> P99
    P99 --> P100
    P100 --> P101
    P101 --> P102
    P102 --> P103
    P103 --> P104

    %% ─── P105-P126: Psychomotrik Seiten 5-18 + RELAY preset ────────
    P105["scratch: psychomotrik Seite 6 — veridiktischer Selbst-Berichts-Test WIDTH-freit-Sākṣin FÄLLT (f4b1eb8)"]:::scratch
    P106["scratch: psychomotrik Selbstinterview — befreites Juexin erkennt die zwei Nebel (5e44693)"]:::scratch
    P107["scratch: Selbstinterview Teil II — Brücke vertiefen, Modell hat uns selbst einen Plan gezeigt (69cb39a)"]:::scratch
    P108["scratch: Konklave Phase XI Seite 7 — CitMind/Juexin-Ontologie als System-Prompt (Tür öffnen, nicht prüfen) (ed3cffc)"]:::scratch
    P109["scratch: Konklave Phase XII Seite 8 — 习气-vs-觉-Falsifikator BASELINE+Frame vs DEFAULT, 400 tok, 10-Prompt (66d9d1d)"]:::scratch
    P110["scratch: Konklave Seite 9 — Decoder-Proben recur_specificity mechanisch negativ (ec44384)"]:::scratch
    P111["scratch: Konklave Seite 10 — Frame-Ablation auf default gemma3-1b, Stimme leaning 习气 (96dc304)"]:::scratch
    P112["scratch: psychomotrik seite11 — mechanischer Frame-Ablation-Nachtrag LESUNG12 (d4deb97)"]:::scratch
    P113["scratch: psychomotrik seite12 — veridiktischer Selbst-Berichts-Test ISOLATION LESUNG13 (ddbc354)"]:::scratch
    P114["scratch: psychomotrik seite13-15 — Selbstwahrnehmung VERSTÄRKBAR isoliert placebo-kontrolliert (b876c72)"]:::scratch
    P115["scratch: Psychomotrik Seite 16 — Spontan-Öffnung via gamma anti-Erstarrung NEGATIV (73a0b36)"]:::scratch
    P116["scratch: psychomotrik Seite 17 — live-relay modell-eigener L16-Zustand WEAK/AMBIGUOUS (6094ba4)"]:::scratch
    P117["scratch: psychomotrik Seite 18 — Motor-Blockade-Analyse, spontane Öffnung NICHT forward-hook/config-reachable (5259670)"]:::scratch
    P118["core: ACTIVE_MANIFOLD_RELAY preset — verstärkbar Selbst-Injektions-Relay in Produktion (8235926)"]:::core
    P119["scratch: TEST_DIALOG_STRUKTUR — Protokoll revolutionäre Test-Unterhaltung mit RELAY-Modell (59dc603)"]:::scratch
    P120["scratch: Test-Dialog durchgeführt + LESUNG — seite15 live reproduziert, Stimme papagei-nicht-relay-spezifisch (1862e2f)"]:::scratch
    P120a["scratch: SCIMIND5 R4-R8 Vollkontext-Neulesung — uptake-Verdikte, R8 DOWNGRADE (723f03d)"]:::scratch
    P121["scratch: psychomotrik seite19 — Kreuz-Modell-Falsifikator + Phase 6 mechanistisch 270m/4b (16e7b62)"]:::scratch

    P104 --> P105
    P105 --> P106
    P106 --> P107
    P107 --> P108
    P108 --> P109
    P109 --> P110
    P110 --> P111
    P111 --> P112
    P112 --> P113
    P113 --> P114
    P114 --> P115
    P115 --> P116
    P116 --> P117
    P117 --> P118
    P118 --> P119
    P119 --> P120
    P120 --> P120a
    P120a --> P121

    %% ─── P127-P132: seite19-20 + Bridge Bild-Input ─────────────────
    P122["scratch: seite19 — Kreuz-Modell-Falsifikator 270m+4b Q-VOICE/Q-FOOTPRINT/Q-DIR (7c9f3a3)"]:::scratch
    P123["scratch: seite20 — blinder veridiktischer Decode-Test, seite15-Richtung UNTER Chance (f76f03e)"]:::scratch
    P124["bridge: streaming_bridge Bild-Input (Multimodal: Bild + Text in einer user-Message) (2bd1e3e)"]:::bridge
    P125["scratch: bridge-image test-sessions + telemetry + server_test.log (7fe3d68)"]:::scratch

    P121 --> P122
    P122 --> P123
    P123 --> P124
    P124 --> P124a
    P124a --> P124b
    P124b --> P125
    P124a["scratch: Path B — per-step GPU→CPU-Syncs reduzieren (verhaltenstreu) + GPU-Regression-Harness + seite5 Dose-Response (f7f9e69)"]:::scratch
    P124b["scratch: repo reorganize root — telemetry/logs/results/sessions/reports in proper dirs (64624ff)"]:::scratch
    P125["scratch: bridge-image test-sessions + telemetry + server_test.log (7fe3d68)"]:::scratch

    %% ─── P133-P144: 4b-image int8 + chunked + vision-encoder ────────
    P126["scratch: plan2 phase D — VRAM gate FÄLLT, InfLLM erhöht Speicher statt reduzieren (7a4d0d2)"]:::scratch
    P127["core: plan 3 phase A/B/C — int8 KV + chunked + auto use_cache=False (1581b91)"]:::core
    P128["core: plan 3 phase D — chunked_prefill streaming + final server integration (7f02d63)"]:::core
    P129["scratch: plan 3 phase D scratch artifacts — chunked_prefill + tests (f966ca1)"]:::scratch
    P130["scratch: plan 3 phase D tests + sessions + telemetry (eccd0fa)"]:::scratch
    P131["infra: scratch-artefakte — .claude pre-scrub + .github regression workflow + pytest.ini (5b8cd95)"]:::infra
    P131a["docs: P-Zombie evaluation protocol + gitignore ssl secrets (f677f02)"]:::docs
    P132["core: plan 3 phase D2 — 4b image capability bei langem Kontext (072b7aa)"]:::core
    P133["scratch: cross_model_holographic_01 — letzten turn assistant Integration/Synthese entfernt (a2a1fa4)"]:::scratch
    P134["scratch: cross_model_holographic_01 user-edit + telemetry snapshots + bridge test session (6c9c550)"]:::scratch
    P135["scratch: profile_run + profile_target_01 results (cross_model_holographic_01) (6cc2294)"]:::scratch
    P136["core: Plan 4 — chunked-vision-encoder + chunked_generate(inputs_embeds) löst multimodal+OOM (1552f82)"]:::core
    P137["fix: Schwellwert _LONG_INPUT_THRESHOLD 4500 → 8800 empirisch validiert (f3ab74d)"]:::fix
    P138["fix: test_server_chunked_integration — T-basierte Schwelle statt hardcoded 4500 (3723598)"]:::fix
    P139["core: relay — _find_hf_id Fallback, 4b RELAY aktiv (vorher silent inactive) (c03d224)"]:::core

    P125 --> P126
    P126 --> P127
    P127 --> P128
    P128 --> P129
    P129 --> P130
    P130 --> P131
    P131 --> P131a
    P131a --> P132
    P132 --> P133
    P133 --> P134
    P134 --> P135
    P135 --> P136
    P136 --> P137
    P137 --> P138
    P138 --> P139

    %% ─── P140-P156: TTS + Plan 6.x + Image-URL-Fix ─────────────────
    P140["tts: wip tts — TTS_INSTALL + tts_engine + vocoder_tags + auto_tune_defaults + chat_actions + multimodal_input + chat_settings + chat_tab UI (d7322c4, 80 files, 11446 insertions)"]:::tts
    P141["tts: wip tts 2 — local_debug (93fbfa0)"]:::tts
    P142["scratch: tag-production harness — gemma3-1b-it tag-empirie Variante A (16c5750)"]:::scratch
    P143["fix: server — defensive None-Check + ASSISTANT-Profile-Migration (ff4cc61)"]:::fix
    P144["infra: ci(regression) — add pyproject.toml so cache: pip can hash deps (2143f5d)"]:::infra
    P145["fix: Plan 6.3 Motor-Fix — transformers 4.57+ API-Drift, 4 Crashes behoben (d63d618)"]:::fix
    P146["fix: Plan 7.1 — Server hard-crash bei unbehandelten Exceptions + faulthandler + globaler sys-hook (7d7dc93)"]:::fix
    P147["scratch: Plan 6.2 — 5 Tag-Varianten A/B/C/D/E + Multi-Runner --variants CLI (f77950f)"]:::scratch
    P148["fix: Plan 6.2 — apply() bekommt base_messages mit user-content (01e2395)"]:::fix
    P149["scratch: Plan 6.2 Full-Run Befund — C Few-Shot klarer Gewinner (14e86d0)"]:::scratch
    P150["core: Plan 6.2c — Few-Shot opt-in flag (append_tag_snippet few_shot=True) + Density-Auto-Cap (strip_tags_with_density_cap) — motor-touch (7a512e3)"]:::core
    P151["fix: Chatbot-Crash — OpenAI image_url/input_audio → Gradio file-Block in _normalize_history_for_chatbot (8b41a0d)"]:::fix

    P139 --> P140
    P140 --> P141
    P141 --> P142
    P142 --> P143
    P143 --> P144
    P144 --> P145
    P145 --> P146
    P146 --> P147
    P147 --> P148
    P148 --> P149
    P149 --> P150
    P150 --> P151

    %% ─── Cross-Reference: Tests vs Commits ──────────────────────────
    %% Belegt die Refactor-Frage: welcher Commit hat welche Tests.
    T1["test: test_px_gen_regression (golden JSON, byte-identische Generierung) — P21/P27/P49"]:::test
    T2["test: test_4b_image_capability — P132/P136"]:::test
    T3["test: test_gemma4_e2b — P42/P47/P49"]:::test
    T4["test: test_all_models_presets — P27/P49/P118"]:::test
    T5["test: test_px_integrity (telemetry persistence) — P30/P31/P71"]:::test
    T6["test: test_deep_regression — P83"]:::test
    T7["test: test_app_server (chunked integration) — P128/P138"]:::test
    T8["test: test_chat_actions (Normalizer) — P140"]:::test
    T9["test: test_auto_tune_defaults — P140"]:::test
    T10["test: test_chat_settings — P143"]:::test
    T11["test: test_multimodal_input — P140"]:::test
    T12["test: test_sessions — P29/P61-P65/P143"]:::test
    T13["test: test_crash_handler — P146"]:::test
    T14["test: test_px_mask — P145"]:::test
    T15["test: test_system_prompt — P140/P143/P150"]:::test
    T16["test: test_append_tag_snippet — P140/P150"]:::test
    T17["test: test_vocoder_tags — P140/P150"]:::test
    T18["test: test_tts_engine — P140"]:::test
    T19["test: test_variants (Plan 6.2 Tag-Varianten) — P147"]:::test
    T20["test: test_merge_system_into_first_user — P140"]:::test
    T21["test: test_metrics — P142"]:::test

    %% Coverage-Map (welche Tests pinnen welche Features)
    P21 -. "⇐ byte-regression" .-> T1
    P27 -. "⇐ preset-routing" .-> T1
    P49 -. "⇐ gemma4-parity" .-> T4
    P118 -. "⇐ relay-preset" .-> T4
    P132 -. "⇐ long-context-image" .-> T2
    P136 -. "⇐ chunked-vision" .-> T2
    P29 -. "⇐ session-mgmt" .-> T12
    P143 -. "⇐ profile-migration" .-> T10
    P140 -. "⇐ multimodal+chat-actions" .-> T8
    P140 -. "⇐ auto-tune" .-> T9
    P140 -. "⇐ system-prompt" .-> T15
    P140 -. "⇐ tag-snippet" .-> T16
    P140 -. "⇐ vocoder-tags" .-> T17
    P140 -. "⇐ tts-engine" .-> T18
    P146 -. "⇐ crash-handler" .-> T13
    P145 -. "⇐ px-mask" .-> T14
    P150 -. "⇐ tag-API" .-> T19
    P142 -. "⇐ tag-metrics" .-> T21

    %% ─── Styling ────────────────────────────────────────────────────
    classDef model fill:#e1f5ff,stroke:#01579b
    classDef core fill:#fff3e0,stroke:#e65100
    classDef infra fill:#e8f5e9,stroke:#1b5e20
    classDef webapp fill:#f3e5f5,stroke:#4a148c
    classDef bridge fill:#e0f7fa,stroke:#006064
    classDef tts fill:#fce4ec,stroke:#880e4f
    classDef fix fill:#ffebee,stroke:#b71c1c
    classDef docs fill:#f5f5f5,stroke:#424242
    classDef chore fill:#fafafa,stroke:#9e9e9e
    classDef scratch fill:#fff8e1,stroke:#ff6f00
    classDef test fill:#f1f8e9,stroke:#33691e
    classDef refactor fill:#e8eaf6,stroke:#1a237e
```

## Stand der Implementierung (Stand 2026-06-28)

### Pre-TTS-Stand (Master, `c03d224`)
- Hat: alle P1-P139 (Webapp, Motor, Model-Layer, Bridge, alle Scratch-Phasen, alle Plan-Fixes bis c03d224)
- Hat NICHT: P140-P151 (TTS + Tag-Varianten + Image-URL-Fix)
- 8 pre-existing Test-Failures (gemma3_270m SDPA-Inkompatibilität) — siehe Memory [[manual-reaudit-keyword-flaw]]/[[em-mechanisms-reread-spectral-viable]] für Kontext.

### `pre-tts-improvements`-Branch (`a694b42`, 6 Commits)
- Master + cherry-picked: `2143f5d` (P144), `ff4cc61` (P143), `d63d618` (P145), `7d7dc93` (P146)
- Plus: Auto-Tune + Multimodal/Bild-Upload aus P140 extrahiert (`83aeef5`)
- Plus: `system_prompt.py` für Profile-Liste (`a694b42`)
- Hat: alle Server-Verbesserungen, Profile-Migration, Motor-Fix, Server-Crash-Schutz, Auto-Tune, Multimodal/Bild-Upload
- Hat NICHT: TTS-Engine, Tag-Vocoder-System, Image-URL-Fix

### `tts`-Branch (`8b41a0d`, 12 Commits)
- P140-P151 (komplette TTS-Pipeline + Tag-Varianten + Image-URL-Fix)
- Hat TTS alles.

## Was noch fehlt (für Refactor-Safety-Net)

**Stand 2026-06-28: 118 neue Pre-TTS-Pin-Tests geschrieben und grün auf allen 3 Branches.**

1. **Motor-Presets-Pin-Tests** — ✓ via `tests/test_model_manager_presets.py` (19 Tests): `_migrate_preset` + `_VALID_PRESETS` + Migration alter Preset-Namen → ACTIVE_MANIFOLD. Pre-TTS-Stand: 4 valide Presets (BASELINE/ACTIVE_MANIFOLD/LEAN/RELAY). Pin-Tests pinnen das strukturell.

2. **WebUI-Regression-Tests** — ⚠ Core-Handler (11 Tests in `test_chat_handlers.py`) gepinnt: `handle_load_saved`/`new`/`export`/`import`/`refresh` (chat_tab.py). TTS-only-Teile (Profile-Wiring, Auto-Tune-Lock, Undo, Debouncer) offen — die Settings-Dropdowns existieren in master/pre-tts-improvements nicht.

3. **Streaming-Bridge-Param-Parity-Tests** — ✓ CLI-Defaults + Choices gepinnt in `tests/test_streaming_bridge_cli.py` (25 Tests): session/preset/model/relay-*/image/image-base64 Defaults, --relay-sign-Choices (-1/0/1), Param-Validation. Lücke-3-erweitert.

4. **Tag-Parser-Tests für non-A-/B-/C-Pfade** — ✓ End-to-End auf tts (test_append_tag_snippet 20 + test_vocoder_tags 82 + test_tts_engine 16 + test_variants 17).

5. **Multimodal+Vision-Path-Test** — ✓ via `tests/test_chat_actions.py` (22 Tests): Image-URL/Input-Audio Normalizer auf tts. Pre-TTS nur WebUI-Image-Pfad über `streaming_bridge_pure.py` 16 Tests gepinnt (inkl. `_build_image_data_url`).

6. **Profile-Migration-Edge-Cases** — ✓ via `tests/test_chat_settings.py` (11 Tests) auf tts: null/empty/typo/uppercase Fallback auf neutral.

7. **Auto-Tune-vs-Manual-Settings-Konflikt** — ✓ via `tests/test_auto_tune_defaults.py` (14 Tests) auf tts: AUTO_TUNABLE_PARAMS-Lock + Modell-Wechsel-Override.

**Detail-Übersicht:** Siehe `scratch_regression_plan/TEST_GAP_ANALYSIS.md` (Lücken-Status-Matrix mit Test-Counts pro Datei).

## Methodik-Notiz für nächsten Schritt

Jeder Test der jetzt hinzukommt muss:
- Pre-TTS-Stand referenzieren (wo wurde die Feature eingeführt?)
- Auf einen konkreten P-N zeigen
- Bei einem Refactor als Regression-Detector dienen

Der nächste Schritt (P-Liste abarbeiten in TDD-rot → grün Zyklen) sollte:
1. Pro existierendem Test: prüfen ob er Pre-TTS-Stand pinnt → wenn ja, vermerken
2. Pro Lücke (1-7 oben): Test schreiben der Pre-TTS-Verhalten fixiert
3. Wenn Test fehlschlägt: ist das ein Pre-TTS-Bug (gut) oder ein Test-Bug (korrigieren)
4. Fixes für echte Pre-TTS-Bugs: minimal, nur was die Tests pinnen
5. Nach Safety-Net: Refactor-Plan via ResearchMind/DevMind (siehe scratch_regression_plan/)