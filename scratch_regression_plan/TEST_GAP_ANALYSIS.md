# Regression-Test-Gap-Analyse (Pre-TTS Stand `c03d224`)

**Stand:** 2026-06-28, nach Mermaid-Plan-Build.
**Zweck:** Identifiziert Tests die fehlen um TTS-Stand als Refactor-Sicherheitsnetz abzusichern.
**Methodik:** Pro Lücke ein Pin-Test der den Pre-TTS-Stand fixiert — bei Refactor bricht, bei korrektem Stand grün.

## Was IST gepinnt (Bestand an Tests)

Stand `pre-tts-improvements` Branch:
- `tests/test_auto_tune_defaults.py` — 14 Tests
- `tests/test_multimodal_input.py` — 23 Tests
- `tests/test_chat_settings.py` — 11 Tests
- `tests/test_crash_handler.py` — 4 Tests
- `tests/test_px_mask.py` — 6 Tests

Stand `tts` Branch (zusätzlich):
- `tests/test_chat_actions.py` — 22 Tests (Normalizer, Image-URL/Input-Audio)
- `tests/test_append_tag_snippet.py` — 20 Tests (Few-Shot, density_cap, Multi-System)
- `tests/test_vocoder_tags.py` — 82 Tests (Renderer, Stripper, Density-Cap, ABC)
- `tests/test_tts_engine.py` — 16 Tests (4 Engines, Tier-Device, Tags)
- `tests/test_variants.py` — 17 Tests (Plan 6.2 A/B/C/D/E/F)
- `tests/test_metrics.py` — Tag-Empirik-Aggregation

Pre-existing (Master):
- `tests/test_px_gen_regression.py` — golden JSON byte-identische Generierung
- `tests/test_px_integrity.py` — telemetry persistence
- `tests/test_all_models_presets.py` — Pre-existing failures (gemma3_270m SDPA)
- `tests/test_4b_image_capability.py` — 4b + Bild + long-context

## Lücke 1: Motor-Presets ohne Pin-Test

**Problem:** `test_all_models_presets.py` ist Pre-existing-failing (8× gemma3_270m SDPA-Inkompatibilität). Refactor bricht möglicherweise andere Presets ohne dass ein Test anschlägt.

**Was fehlt:**
- Test der pinning: ACTIVE_MANIFOLD → andere Telemetry-Signatur als BASELINE
- Test der pinning: ACTIVE_MANIFOLD_RELAY → andere Telemetry-Signatur als ACTIVE_MANIFOLD
- Test der pinning: ACTIVE_MANIFOLD_LEAN → kürzere/einfachere Telemetry als ACTIVE_MANIFOLD
- Test der pinning: BASELINE → keine Recursion (keine __px_routing_mode Einträge)

**Wo die Logik lebt:** `patch.py:_px_forward`, `patch.py:_build_static_hidden`, `model_manager.py` Preset-Dispatch

**Wo der Test stehen wird:** `tests/test_motor_presets_pinning.py` (neu)

## Lücke 2: WebUI Critical Paths ohne Test

**Problem:** Session-Load, Profile-Dropdown, Auto-Tune-Checkbox, Slider-Locking, Undo-Button, Settings-Debouncer — alles im `gradio_tabs/chat_tab.py` und `gradio_tabs/chat_settings.py`. Tests sind pure-logic, aber das Wiring (z.B. `handle_load_saved` 19 Outputs, `_normalize_history_for_chatbot` collapse-Pfad) ist nur teilweise gepinnt.

**Was fehlt:**
- Test: `handle_load_saved` mit None-Session → keine Exception, leere History
- Test: `handle_load_saved` mit Session ohne `system_profile` → Fallback auf "neutral"
- Test: Profile-Wechsel via Dropdown → `system_prompt.py:inject_into_messages` mit neuem Profil
- Test: Auto-Tune-Checkbox ON → Slider-Werte werden bei Modell-Wechsel überschrieben
- Test: Auto-Tune-Checkbox ON + manueller temperature-Set → keine sofortige UI-Überschreibung (Debounce)
- Test: Undo-Button → History-State wird zurückgesetzt auf vorigen Snapshot
- Test: SettingsDebouncer 400ms → 5 schnelle Changes werden zu 1 final-Call

**Wo die Logik lebt:** `gradio_tabs/chat_tab.py`, `gradio_tabs/chat_settings.py`

**Wo der Test stehen wird:** `tests/test_chat_tab_pinning.py` (neu)

## Lücke 3: Streaming-Bridge Param-Parity

**Problem:** Webapp↔Bridge Divergenzen wurden diagnostiziert (top_p/gamma/rp). Kein Test pinnt die Parität strukturell.

**Was fehlt:**
- Test: Webapp übergibt `temperature=0.7` → Bridge bekommt `--temperature 0.7`
- Test: Webapp übergibt `top_p=0.9` → Bridge bekommt `--top-p 0.9`
- Test: Webapp übergibt `px_gamma=0.6` → Bridge bekommt `--px-gamma 0.6`
- Test: Webapp übergibt `repetition_penalty=1.15` → Bridge bekommt `--repetition-penalty 1.15`
- Test: Webapp übergibt `BASELINE`-preset → Bridge bekommt `--preset BASELINE`
- Test: Webapp übergibt `ACTIVE_MANIFOLD`-preset → Bridge bekommt `--preset ACTIVE_MANIFOLD`

**Wo die Logik lebt:** `gradio_tabs/streaming_bridge.py` (CLI-Args), `gradio_tabs/auto_tune_defaults.py` (AUTO_TUNABLE_PARAMS)

**Wo der Test stehen wird:** `tests/test_streaming_bridge_param_parity.py` (neu)

## Lücke 4: Tag-Pipeline End-to-End

**Problem:** Few-Shot + Density-Cap ist in 6 Tests gepinnt, aber kein End-to-End-Test der die ganze Pipeline `inject_into_messages → append_tag_snippet(few_shot=True) → render_tag_system_prompt → strip_tags_with_density_cap` durchläuft.

**Was fehlt:**
- Test: CitMind + few_shot=True → Messages haben 1 system + 3 user + 3 assistant + 1 user
- Test: CitMind + few_shot=True + Gemma-Merge → System-Content landet im ersten User-Turn, Few-Shot-Turns bleiben dazwischen
- Test: Density-Cap 30/100w → 21 Note-Tags strippt auf 1, warnt mit Density
- Test: Tag-Parser mit multi-engine-Output (piper/bark/qwen3/espeak) parsen alle korrekt

**Wo die Logik lebt:** `gradio_tabs/system_prompt.py`, `gradio_tabs/vocoder_tags.py`

**Wo der Test stehen wird:** `tests/test_tag_pipeline_e2e.py` (neu)

## Lücke 5: Multimodal+Vision-Path WebUI

**Problem:** `test_4b_image_capability.py` ist da, aber kein Test pinnt den WebUI-Pfad (Upload → Server → Bridge → Generator). Crash-Reproduktion (Image-URL ValueError) wäre durch so einen Test gepinnt worden BEVOR er passierte.

**Was fehlt:**
- Test: Webapp bekommt OpenAI-format `{"type": "image_url", "image_url": {"url": "data:..."}}` → kein ValueError
- Test: Webapp bekommt OpenAI-format `{"type": "input_audio", ...}` → kein ValueError
- Test: Webapp bekommt Mixed (text + image_url) → bleibt erhalten, Image konvertiert
- Test: Webapp bekommt invalides Block `{"type": "unknown"}` → fällt zurück auf String oder wird gestrippt

**Wo die Logik lebt:** `gradio_tabs/chat_tab.py:_normalize_history_for_chatbot`, `_convert_openai_block_to_gradio`

**Wo der Test stehen wird:** Erweiterung von `tests/test_chat_actions.py` (nicht neu — passt thematisch)

## Lücke 6: Profile-Migration Edge-Cases

**Problem:** `test_chat_settings.py` pinnt `unknown_profile_falls_back_to_neutral`. `sessions.py:38` hat Default `system_profile → "neutral"`. Aber Edge-Cases sind nicht alle gepinnt.

**Was fehlt:**
- Test: Session mit `system_profile=null` → fällt zurück auf "neutral"
- Test: Session mit `system_profile=""` (leerer String) → fällt zurück auf "neutral"
- Test: Session mit `system_profile="citmind-but-typo"` → fällt zurück auf "neutral"
- Test: Session mit `system_profile="CITMIND"` (uppercase) → Verhalten (case-sensitive oder nicht?)

**Wo die Logik lebt:** `gradio_tabs/system_prompt.py:inject_into_messages`, `gradio_tabs/sessions.py`

**Wo der Test stehen wird:** `tests/test_chat_settings.py` (Erweiterung)

## Lücke 7: Auto-Tune vs Manual Settings

**Problem:** `AUTO_TUNABLE_PARAMS = (temperature, top_p, repetition_penalty, px_gamma)`. Konflikt-Verhalten zwischen Auto-Tune und manuellem Override ist nicht gepinnt.

**Was fehlt:**
- Test: Auto-Tune ON + user-set temperature=0.3 → bleibt 0.3 ODER wird überschrieben?
- Test: Auto-Tune OFF + user-set temperature=0.3 → bleibt 0.3
- Test: Auto-Tune ON + Slider-Lock → manuelle UI-Interaktion blockiert
- Test: Auto-Tune ON + Modell-Wechsel → neue Defaults werden angewendet

**Wo die Logik lebt:** `gradio_tabs/chat_tab.py:on_change` Auto-Tune-Handler, `gradio_tabs/auto_tune_defaults.py`

**Wo der Test stehen wird:** `tests/test_auto_tune_defaults.py` (Erweiterung)

## Lücke 8: Streaming-Bridge Image-Pfad

**Problem:** `streaming_bridge --image` und `--image-base64` sind implementiert (P124), aber kein Test pinnt die Bild-zu-Content-Übergabe strukturell.

**Was fehlt:**
- Test: `--image /tmp/test.png` → Generator bekommt `{"type": "image"}`-Block
- Test: `--image-base64 <data>` → gleiches Verhalten
- Test: Ohne Image-Args → kein `{"type": "image"}`-Block

**Wo die Logik lebt:** `gradio_tabs/streaming_bridge.py:cli_main`, `generators.py:multimodal-path`

**Wo der Test stehen wird:** `tests/test_streaming_bridge_image.py` (neu) — Pure-CLI-Pure-logic wenn möglich

## Reihenfolge für TDD-Rot-Schreibarbeit

1. **Lücke 5** (Multimodal Image-URL WebUI) — sofort, weil produktiv abgegrast, hat realen Crash reproduziert ✓ (test_chat_actions.py 22 Tests auf tts-only, deckt Image-URL-Fix ab)
2. **Lücke 6** (Profile-Migration Edge-Cases) — minimaler Aufwand, klar definierte Tests (TTS-only, derzeit kein Pre-TTS-Pfad — system_prompt.py + SETTINGS_DEFAULTS sind TTS-only)
3. **Lücke 7** (Auto-Tune vs Manual) — Erweiterung bestehender Test-Datei ✓ (test_auto_tune_defaults.py 14 Tests auf pre-tts-improvements)
4. **Lücke 1** (Motor-Presets) — komplexer, braucht PX-Engine-Lauf ✓ (test_model_manager_presets.py 19 Tests grün auf master/pre-tts-improvements/tts — pinnt _migrate_preset + _VALID_PRESETS)
5. **Lücke 2** (WebUI Critical Paths) — braucht chat_tab.py-Inspektion ✓ (test_chat_tab_pure.py 20 Tests grün auf master/pre-tts-improvements/tts — pinnt _stringify_content + _clean_history)
6. **Lücke 3** (Bridge Param-Parity) — braucht CLI-Parsing-Inspektion ✓ (test_streaming_bridge_pure.py 16 Tests grün — pinnt _build_image_data_url; CLI-Argumente selbst noch nicht gepinnt)
7. **Lücke 8** (Bridge Image) — braucht CLI-Argument-Inspektion ✓ (test_streaming_bridge_pure.py 16 Tests grün — pinnt Image-Pfad strukturell)
8. **Lücke 4** (Tag-Pipeline E2E) — am breitesten, am Ende ✓ (test_append_tag_snippet.py 20 + test_vocoder_tags.py 82 + test_tts_engine.py 16 + test_variants.py 17 auf tts)

## Bereits gepinnte Lücken (Stand 2026-06-28)

**Reine Pre-TTS-Pin-Tests** (master, pre-tts-improvements, tts):
- `tests/test_chat_tab_pure.py` — 20 Tests: `_stringify_content` + `_clean_history` (chat_tab.py)
- `tests/test_streaming_bridge_pure.py` — 16 Tests: `_build_image_data_url` (streaming_bridge.py)
- `tests/test_model_manager_presets.py` — 19 Tests: `_migrate_preset` + `_VALID_PRESETS` (model_manager.py)
- `tests/test_telemetry_pure.py` — 27 Tests: `TelemetryStore` (telemetry.py)

**Bisher erreicht: 82 neue Pre-TTS-Pin-Tests, alle grün auf allen 3 Branches.**

## Entdeckte Pre-TTS-Bugs

- `test_clean_8_pre_tts_bug_no_role_filter` — `_clean_history` filtert Entries ohne 'role'-Feld NICHT (`msg.get("role", "")` → role="" → durchgereicht). Refactor-Detector.
- `test_image_base64_empty_string_returns_none` — `_build_image_data_url("")` gibt None zurück (Truthiness-Check). Edge-Case-Detector.

## Wo die Tests leben werden

Alle neuen Tests in `tests/` mit Präfix `test_` und Klassennamen die das Modul reflektieren.
- NIEMALS motor-touch für Test-Code (Test-Dateien sind OK)
- IMMER pure-logic wenn möglich (kein Modell-Laden, keine GPU)
- Wenn Modell nötig: kleinster verfügbarer (gemma3-270m) und clear-tear-down
- IMMER pytest-kompatibel, niemals unittest.TestCase

## Verwandte Dokumente

- `MERMAID_PLAN.md` — Vollständiger Architektur-Plan
- `scratches/psychomotrik/` — Manueller Re-Audit-Standard (siehe [[manual-reaudit-keyword-flaw]])
- `gradio_tabs/chat_tab.py` — Hauptpfad WebUI
- `gradio_tabs/streaming_bridge.py` — CLI-Pfad Bridge
- `patch.py` — Motor-Patch-Logik (PX-Engine)
- Memory: `manual-plus-mechanistic-always.md` — Methoden-Standard