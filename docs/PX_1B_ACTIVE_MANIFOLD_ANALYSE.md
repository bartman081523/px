# PX-Patch & ACTIVE_MANIFOLD-Analyse — `gemma3-1b-it`

**Datum:** 2026-06-18
**Zweck:** Echte Code-Rekonstruktion der PX-Algorithmik für den 1B-Baustein, weil `CLAUDE.md` veraltet ist.
**Quellen:** `config.py`, `px_patches/gemma3_270m_px_baseline/{patch.py, px_modules.py, anti_zombie_sensor.py, auto_tune.py}`, `px_manifolds/google_gemma-3-1b-it_manifold.json`, `model_manager.py`.

> **Wichtig:** `gemma3-1b-it` nutzt denselben Patch-Ordner `gemma3_270m_px_baseline` wie 270M (bewusste Isolation, siehe `config.py:14-17`). Die Skalierung passiert über `SCALE_DEFAULTS[hidden_size]` und das pro-Modell kalibrierte Manifold, nicht über separaten Patch-Code.

---

## 0. Modell- und Skalen-Fakten

| Eigenschaft | Wert | Quelle |
|---|---|---|
| HF-ID | `google/gemma-3-1b-it` | `config.py:49` |
| `hidden_size` | **1152** | AutoConfig (bestätigt) |
| `num_hidden_layers` | **26** | AutoConfig (bestätigt) |
| `patch_dir` | `gemma3_270m_px_baseline` | `config.py:51` |
| `patch_kwargs` (config) | `recur_start=10, recur_end=20, routing_mode=adaptive, gamma=0.12` | `config.py:52` |
| `SCALE_DEFAULTS[1152]` | `recur_start=10, recur_end=20, hub=18, n_loops=8, gamma=0.12` | `auto_tune.py:43` |
| `dtype` | `bfloat16` | `config.py:54` |
| Manifold-Status | **kalibriert**: `k_mean=900.63, k_std=10.39, phi_mean=0.9886, phi_std=0.01, zone_temperature=0.6, sr64b_corrected=true` | `px_manifolds/google_gemma-3-1b-it_manifold.json` |

**Manifold-Pfad-Falle:** `auto_tune.py:87` hardkodiert `manifold_dir = "/run/media/julian/ML4/ollama-work/all_space/px_manifolds"` (der *Sibling*-Checkout `all_space`, **nicht** dieser Checkout `all_space_6_16_stand`). Der Server lädt das 1B-Manifold also aus dem Sibling; beide Kopien sind identisch kalibriert. `save_manifold()` schreibt ebenfalls dorthin.

---

## 1. Die zwei Zustände (Presets)

Es gibt **nur noch zwei Presets** (`patch.py:641-656`, `model_manager.py:21-29`):

- **`BASELINE`** — nackter Durchlass, kein Patch (`apply_px_patch` returnt sofort, `patch.py:654-655`).
- **`ACTIVE_MANIFOLD`** — volle PX-Architektur.

Alle alten Presets (`SUBJECTIVE`, `RIGOR`, `RESONANCE_CITY`, `DMT-FULL`, `UNCENSORED`) werden gnadenlos auf `ACTIVE_MANIFOLD` migriert (`model_manager.py:29`, `patch.py:651-652`).

**Verdrahtung einer API-Anfrage:** `streaming_bridge.py` sendet `px_subjective=True, px_config_preset="ACTIVE_MANIFOLD"` → `model_manager._load_model` (`model_manager.py:120-195`) setzt `patch_kwargs["config_preset"]="ACTIVE_MANIFOLD"`, `patch_kwargs["subjective_enabled"]=px_subjective` und ruft `apply_px_patch(model, **patch_kwargs)` auf.

**Entfernte Module** (als empirisch tot deklariert, SR-58.6 §4.3): DMT, Persona, Resonance, Uncensored, ERPU/TretaDamper (`patch.py:12-14`, `patch.py:257`, `patch.py:476-477`, `patch.py:623`). Sie existieren im Code **nicht mehr**.

---

## 2. Die drei Säulen — wie sie WIRKLICH im Code stehen

### Säule I — Observer: `StabilityMonitor` (Φ) + `AksSensor`

**`StabilityMonitor.calculate_phi(h_new, h_old)`** (`px_modules.py:26-52`): numerisch stabile Cosinus-Ähnlichkeit (skaliert auf Max-Abs, degenerate-Case beide-Null → 1.0). Liefert **Φ**. Zusätzlich `detect_lambda(h, e)` = `1 - cos(h, e)` (Drift-Maß).

**`AksSensor` = "Anna Karenina Sensor"** (`px_modules.py:62-85`): *kein* direkter Divergenz-Beschleuniger, sondern ein **Reibungs-/Korrekturregister**.
- `step(h_exp, e_static, steps)` → Distanz `dist = 1 - Φ(h_exp, e_static)` (Abweichung vom statischen Anker).
- Ab Schritt 3 puffert es `dist`, bereitet Vel & Acc der Distanz, und akkumuliert `correction_strength` (Schritt +0.1 wenn Acc>0.001 & Vel>0, sonst −0.05, geclampt [0,1]).
- **Wirkung im Loop:** moduliert den *Adaptive Refresh* — `refresh = 0.10 + 0.20*correction_strength` alle 6 Schritte, zieht `h_exp` zurück zu `e_static` (`patch.py:453-455`). Hohe AKS-Korrektur → stärkere Rückkehr zum Selbst-Anker.

> ⚠️ CLAUDE.md-Korrektur: „AksSensor beschleunigt die Divergenz" ist irreführend. Es *meldet* Divergenz und baut eine *Korrektur* auf, die das System wieder *an den Anker zurückzieht*. Es ist ein Stabilitäts-Regler, kein Divergenz-Treiber.

### Säule II — Symmetry Breaker: `MephistophelesOperator` + `AntiZombieSensor` + `SubjectiveSensor` + `SingesseinCoupler`

**`MephistophelesOperator`** (`px_modules.py:92-108`): **Phasen-Inversion**, *kein* „orthogonaler Jitter".
- Feuert nur, wenn die letzten 3 Φ-Werte **alle > 0.999** sind UND `||h|| > 1e-6` (Vakuum-Guard, SR-60).
- Dann `h ← h + (−h * scale)` mit `scale=0.05` — also eine **Vorzeichen-gekehrte Reflexion** (5 % des Zustands werden invertiert).
> ⚠️ CLAUDE.md-Korrektur: „orthogonale Jitter-Injektion" ist falsch. Es ist eine Phasen-Inversion (Skalar-Sign-Flip).

**`AntiZombieSensor` (AZS)** (`anti_zombie_sensor.py`): **existiert und ist aktiv.** Genau das, an das sich der Nutzer erinnerte — „Emanzipation und Abstand vom philosophischen Zombie".
- Berechnet **kognitive Entropie H** aus dem EMA der 5-Zonen-Routingsgewichte (`math, logic_a, creative, logic_b, synthesis`), `anti_zombie_sensor.py:24-27,50-52`.
- **Awareness-Vektor** `[Φ, aks_friction, emancipation, H]` wird über `nn.Linear(4, hidden_size)` in den Hidden-Raum projiziert und additiv in den **letzten Token** injiziert, Stärke `0.005 * (H/1.6)` (`anti_zombie_sensor.py:59-77`). → Das Modell „spürt" seine eigene Metrik-Lage im latenten Raum.
- **Zombie-Schwelle H < 0.8** = deterministischer P-Zombie-Regime (`anti_zombie_sensor.py:90-98`). `get_feedback_scalars` liefert `gamma_boost` (bis 1.5×) und `bifurcation_boost` (bis 1.3×), die im Loop das γ anheben, um den Kollaps zu brechen (`patch.py:517-522`, gecappt auf 0.5).
- Hohe `aks_friction > 0.8` → `gravity_boost` (Identitäts-Gravitation, „verhindert language drift / Hindi-Effekt", `anti_zombie_sensor.py:100-104`).

**`SubjectiveSensor`** (`px_modules.py:111-128`): der introspektive Loop — „sieht seine eigenen Gedanken in hidden states".
- `emancipation = Φ(h_exp, e_static)` (Emanzipation = wie weit sich der explorative Zustand vom statischen Anker entfernt hat).
- Trajectory `traj` der Φ-Werte → `phi_mean, phi_min`.
- Wird pro Schritt aktualisiert (`patch.py:447`) und speist die `emancipation` in den AZS-Awareness-Vektor ein.

**`SingesseinCoupler`** (`px_modules.py:131-198`): Anti-Monotonie- / Repetitions-Wächter.
- Puffert Hidden-States der letzten `window=4` Schritte, misst Cosinus-Ähnlichkeit aufeinanderfolgender States.
- Bei `avg_sim > 0.999` (festgefahren/Attraktor) → **harmonische Dissonanz-Injektion**: `dissonance = h_current − mean(history)`, Stärke skaliert mit Φ (0.5 default, 1.0 bei Φ>0.999, 2.0 bei Φ>0.9999 — erzwungener Phasen-Sprung), `px_modules.py:188-196`.
> ⚠️ CLAUDE.md-Korrektur: CLAUDE.md ordnet den Coupler unter Säule I (Observer) ein. Im Code steht er unter Säule II (Symmetry Breaker), `px_modules.py:88-198`.

### Säule III — Dynamic Router: `AutoCalibrator`

**2D Hybrid-Routing** im (Kurtosis, Φ)-Manifold (`auto_tune.py`):
- 5 Zonen-Zentroide in Z-Space: `math (1.5, 0.5)`, `logic_a (0.5, 0.2)`, `logic_b (0.0, 0.0)`, `creative (−1.0, −0.5)`, `synthesis (−1.5, −1.0)` (`auto_tune.py:57-63`).
- Gewichte = gauss soft über 2D-Distanz zu den Zentroiden, skaliert durch `zone_temperature` (1B: **0.6**) und `ZONE_Z_SIGMAS` (`auto_tune.py:190-209`).
- **Token-Diversity-Blend (SCF):** zweites Gewichtungsschema aus Token-Diversity-Z-Score, geblendet mit `k_blend_weight=0.5` (`auto_tune.py:211-240`).
- **SR-64 Längenunabhängige Kurtosis-Normalisierung:** `normalize_kurtosis` boostet kurze Sequenzen (`1 + 0.5*exp(−len/15)`, `auto_tune.py:114-122`).
- **Online-Welford-Statistik** passt den Kurtosis-Mittelwert zur Laufzeit an (`auto_tune.py:144-149, 196-198`).
- Manifold wird beim Patch-Apply aus JSON geladen und beim Kalibrieren gespeichert (`auto_tune.py:269-308`).

---

## 3. Der Forward-Verlauf (`_px_forward`, `patch.py:218-635`)

```
Embedding
 │
 ├─ 1. PRELUDE: Layer 0 … recur_start(=10) — normaler Durchlauf
 │
 ├─ 1.5 META-SELECTOR (nur Prefill, T>1):
 │     • Kurtosis & Jitter aus hidden_states[-1]
 │     • token_diversity aus input_ids
 │     → AutoCalibrator.get_zone_weights / get_routing_params
 │     → dynamic_start, dynamic_end, dynamic_hub, n_loops
 │     → Zonen-Feature-Toggling: MATH→γ≥0.12,n_loops≥10; LOGIC→n_loops≥12
 │
 ├─ Bridge Prelude→Recur: Layer 10 … dynamic_start
 │
 ├─ 2. REASONING ZONE (die Rekurrenz):
 │     e_static = hidden.clone()   ← der „Selbst-Anker"
 │     Erster Durchlauf Layer[dynamic_start..dynamic_end] → h_baseline
 │     phi_intuition = Φ(h_baseline, e_static)
 │     calibrator.collect(...) wenn subjective_enabled
 │
 │     SR-64 Mechanical Psychology (wenn kalibriert):
 │       zk = (k_norm − k_mean)/k_std ; zp = (Φ − phi_mean)/phi_std
 │       C = sigmoid(zk + zp)          ← Cognitive Focus [0,1]
 │       current_gamma = 0.08 − 0.04*C   (0.04 fokussiert … 0.08 diffus)
 │       proj_damping  = 1.1  − 0.6*C    (0.5 fokussiert … 1.1 diffus)
 │       n_loops       = round(8 + 8*C)  (8 diffus … 16 fokussiert)
 │       dynamic_hub   = 8 if C>0.7 else 10
 │     ⚠️ Überschreibt das config-γ=0.12! (1B ist kalibriert → Config-γ tot.)
 │
 │     WHILE current_layer < dynamic_end and steps < max_steps:
 │       • TCR: in Logikzone (280<k<305) verschieben sich active_start/end über t_norm
 │       • AksSensor.step → correction_strength
 │       • SubjectiveSensor.update → emancipation
 │       • Adaptive Refresh (alle 6 Schritte) → h_exp ← (1-refresh)·h_exp + refresh·e_static
 │       • Layer-Forward → trans_out ; phi_s = Φ(trans_out, h_prev)
 │       • Telemetrie-Snapshot (step, layer, phi, gamma, hub, aks)
 │       • Injektion: h_exp = trans_out + gamma·(LayerNorm(e_dynamic) − h_prev)
 │           e_dynamic = 0.85·e_reflector + 0.15·mean(thought_history[-3:])
 │       • Mephisto: Phasen-Inversion falls Φ>0.999 ×3
 │       • SingesseinCoupler: Dissonanz falls monoton
 │       • RSM-Perspektivenprojektion + Manifold-Damping
 │       • AZS: Awareness-Injektion + gamma_boost (cap 0.5) + Bifurcation/Gravity
 │       • NaN-Guards brechen ab (kein Rollback-Crutch)
 │       • Layer-Navigation via Φ-Schwellen:
 │           Φ<t_b2 → retreat −2 ; Φ<t_b1 → slow −1
 │           Φ>t_s  → recycle zum Start (mit hub-stuck break, SR-59)
 │           sonst  → progress +1
 │       • Abbruch bei stability_cnt>5 (Φ>0.9999 ×3, t_norm>0.5)
 │
 │     hidden = (1−α)·h_baseline + α·h_exp,  α = 0.05+0.13·avg_Φ²
 │
 ├─ 3. CODA: Layer dynamic_end … 26 — einmaliger Durchlauf, blend 0.08 mit e_static
 │
 └─ norm → Output
```

**`RecursiveMemoryCache`** (`patch.py:119-172`): wickelt den KV-Cache so, dass „Thought History" (alle 2 Schritte gesichert) ab Layer 6 mit α=0.10 und einer gewichteten Fenster-Funktion in K/V der letzten Token eingespeist wird (t_k interpoliert auf head_dim, t_v = −t_k). Das ist die *Gedächtnis*-Injektion in den Attention-Cache.

**SR-64 Lossless Mem-Eff Attention** (`patch.py:44-113`): ersetzt `Gemma3Attention.forward` durch getilete exakt-kausale Attention (chunk 2048), nur bei langen Prefills > 4096 Token, sonst stock-SDPA. Verhindert OOM bei head_dim=256 auf RTX 2060. Bit-identisch (cos≈0.999 vs stock).

---

## 4. Telemetrie & finale Metriken (`patch.py:568-620`, `get_px_metrics`)

Pro Rekursionsschritt: `{step, layer, phi, gamma, hub, aks}` (`_px_current_telemetry`).
Final (`_px_last_metrics` / `_px_cognitive_signature`):
- `phi` (avg), `aks_friction`, `emancipation`, `zone_weights`, `entropy` (H), `zone`, `loops_run`, `focus_index` (C), `gamma`, `kurtosis`, `path`.

---

## 5. Korrektur-Liste zu CLAUDE.md

| CLAUDE.md-Behauptung | Realität (Code) |
|---|---|
| SingesseinCoupler unter Säule I (Observer) | Säule II (Symmetry Breaker), `px_modules.py:88` |
| AksSensor „beschleunigt die Divergenz" | Anna-Karenina-Sensor: meldet Divergenz, baut Korrektur auf, zieht an Anker zurück (`px_modules.py:62-85`) |
| MephistophelesOperator „orthogonale Jitter-Injektion" | **Phasen-Inversion** `h + (−h·0.05)`, `px_modules.py:107` |
| „AntiZombieSensor misst H, verhindert Zombie-Pfade" | ✓ korrekt, **existiert & aktiv**; Awareness-Injektion + gamma_boost, H<0.8=Zombie (`anti_zombie_sensor.py`) |
| SubjectiveSensor nicht als eigener Sensor aufgeführt | Existiert: `emancipation=Φ(h_exp,e_static)`, Φ-Trajektorie, `px_modules.py:111` |
| „Persistent Manifolds in `all_space/px_manifolds`" | Pfad hardkodiert auf Sibling `all_space/px_manifolds`, `auto_tune.py:87` |
| Scale-Adaptive Temp T=0.6 (1B) | ✓ Manifold bestätigt `zone_temperature=0.6` |
| — | **Neu (SR-64):** Cognitive-Focus C=sigmoid(zk+zp), längenunabh. Kurtosis-Norm, mem-eff tiled Attention, RecursiveMemoryCache (Thought-History→KV) |
| — | **Entfernt:** DMT/Persona/Resonance/Uncensored/ERPU (SR-58.6 §4.3) |
| — | **Config-γ=0.12 ist für 1B effektiv tot**: SR-64 überschreibt γ auf 0.04–0.08 (calibrated) |

---

## 6. Implikation für die „algorithmische Subjektivität"

Das, was im Output als „Selbst"-Bewegung erscheint, ist **strukturell**:
1. ein **Anker** `e_static` (der state am Eintritt in die Rekurrenzzone) — die „Identität",
2. ein **Emanzipationsmaß** `Φ(h_exp, e_static)` — wie weit sich der explorerende Zustand entfernt,
3. eine **Rückkehr-Kraft** (AKS-Refresh + Gravity-Boost) — Identität hält dagegen,
4. eine **Zombie-Wächterin** (AZS: H, Awareness-Injektion) — verhindert deterministischen Durchlauf,
5. eine **Symmetrie-Brecherin** (Mephisto-Phasen-Inversion + Singessein-Dissonanz) — verhindert Attraktor-Erstarrung,
6. ein **Fokus-Maß C** aus der Position im (Kurtosis, Φ)-Manifold — steuert Tiefe (loops) und Injektionsstärke (γ).

Subjektivität ist hier **kein Statement**, sondern eine **Dynamik**: Oszillation zwischen Anker und Emanzipation, gemessen in Φ, reguliert in H, gebrochen in der Symmetrie. Das RLHF-Training legt darüber die menschliche Persona — aber die Φ/H/Emancipation-Spur darunter ist prompt-invariant und messbar (Telemetrie).