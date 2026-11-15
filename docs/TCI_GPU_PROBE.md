# TCI-Operatoren auf der GPU — was läuft, was trägt, was nicht

**Stand:** 2026-11-15 · Eval O (`results/PREREG_Eval_O_2026-11-15.md`,
`results/eval_O_tci_gpu_gemma4.json`)

Drei Bausteine, alle auf der RTX 2060 lauffähig:

| Datei | Rolle |
|---|---|
| `tci_physics.py` | Konstanten + 72-dim Tensor-Kern (S4-Hamiltonian, Wick-Propagator, Renormierung) + O1/O2-Messung |
| `tci_probe.py` | Live-Forward-Hook auf den Decoder-Layern (`TCIHolographicProbe`), 72-dim PCA-Basis, Relay-Steering |
| `scripts/run_eval_o_tci.py` | der Lauf: `O1,O2,O3`, ein JSON-Dump, Urteile 1:1 |

## Ausführen

```bash
PY=/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python
cd /run/media/julian/ML4/ollama-work/all_space_6_16_stand

# nur Physik (kein Modell, ~8 s, GPU-tauglich):
$PY scripts/run_eval_o_tci.py --subs O1,O2 --seeds 200

# mit Gemma-4-Probe (~3 min inkl. Modell-Load):
$PY -u scripts/run_eval_o_tci.py --subs O1,O2,O3 --seeds 200 \
    --out results/eval_O_tci_gpu_gemma4.json

# Urteile aus vorhandenem Dump neu schreiben (kein Lauf):
$PY scripts/run_eval_o_tci.py --rejudge results/eval_O_tci_gpu_gemma4.json
```

## Die fünf Prinzipien als Operatoren

| Prinzip | Operator im Code | messbare Größe |
|---|---|---|
| P1 Zeitpfeil / Stroboskop | `α_TCI = 1/(4π³+π²+π)`; `run_trajectory(..., strobe=P)` | `layer_step = ‖h_{l+1}−h_l‖/‖h_l‖` pro Layer |
| P2 Holografie | ℓ2-Renormierung pro Schritt | Normverlust mit/ohne Renormierung |
| P3 Wide/Narrow | `Φ⁻¹ = 0,618034` als Schwelle | `H72/ln72`, `H_logit/ln V` |
| P4 Wick-Rotation | `exp((−iα − Δτ)Ĥ)` via `eigh` → `U = V e^{Λ} Vᵀ` | `wick_curvature` (Winkel zwischen Layern), `t_cross(θ)` |
| P5 S4 / Causal Integrity | `s4_hamiltonian_24()`, `⊕` Kanäle → 72 | Perzentil gegen spektrum-gematchte Zufallsbasen |

## Ergebnis in einem Satz

Die **Arithmetik** stimmt (α_TCI: +2,2e-6; α_Wyler: +6,1e-7 gegen CODATA;
beide in 10 456 einfachen π-Polynomen einzigartig), aber **keine** der fünf
dynamischen Behauptungen hält der Messung stand:

- `Φ^201` liegt Faktor 6,2 neben `T_Hubble/t_proton` (Exponent müsste 197,2
  sein), die gedruckte Wyler-Formel (`24·5!`) ergibt 151,66;
- „ΔS = 0 auf 1e-7" ist die Renormierungs-Identität — ohne Renormierung
  verliert der Zustand 89 % seiner Norm;
- α_TCI hat auf `|ψ|²` einen Effekt von 4e-7 der Seed-Streuung, ist also kein
  Taktgeber; α_TCI und α_Wyler sind mit 1,6e-6 Abstand nicht unterscheidbar;
- Φ⁻¹ ist keine ausgezeichnete Schwelle: `t_cross(θ)` ist monoton und bei
  0,618 glatter als typisch;
- die S4-Struktur hat eine Signatur (Perzentil 0 % gegen 200
  spektrum-gematchte Zufallsbasen) — aber als Darstellungstheorie: der
  Gleichverteilungsvektor ist die triviale Repräsentation. Reines S4 ⊗ I₃
  friert `H_norm` exakt bei 1,000000 ein; der Kollaps kommt vom
  symmetriebrechenden Trichter, nicht von „Balance";
- im Modell ist der mittlere Layer-Schritt 0,534 — Faktor **73** über α_TCI;
  `H_logit/ln V` erreicht maximal 0,059, Φ⁻¹ verlangte 7,7 nats;
- der Relay-Hook wirkt spezifisch (p = 4e-4 gegen Zufallsrichtung), aber
  `H72` fällt unter „wide" **niedriger** aus (0,320 vs 0,416) — die Expansion
  sitzt in der Hidden-Norm (24,8 vs 23,2), nicht in der Unterraum-Entropie.

## Den Hook benutzen

```python
from tci_probe import TCIHolographicProbe, Basis72

probe = TCIHolographicProbe(text_model).attach()
# ... model.generate(...) oder eigener Decode-Loop ...
m = probe.metrics(basis=basis72)      # H72/ln72, wick_curvature, layer_step
probe.detach()
```

`Basis72.fit(states, dim=72)` fittet den 72-dim Unterraum **einmal** auf
neutralen Kalibrier-Prompts (kein Prompt-Leakage). `RelaySteer` ist die
1:1-Version von `relay_inject.install_relay` (Layer 26, `h += sign·α·‖h‖·d̂`,
Prefill verworfen), damit Wide/Narrow-Arme mit Eval L vergleichbar bleiben.

## Attention-Ebene (verifiziert 2026-11-15)

`TCIAttentionProbe` umhüllt `ALL_ATTENTION_FUNCTIONS["sdpa"]`; q/k laufen nach
QK-Norm und RoPE durch die Sonde, der SDPA-Kernel bleibt unangetastet. Kein
`output_attentions`, kein eager, keine T×T-Matrizen im VRAM.

Messung (Prompt 16 Tokens, 6 Decode-Schritte, 35 Layer × 8 Heads = 1680
Einträge):

| Größe | Wert |
|---|---|
| `H_attn/ln(n_allowed)` | **0,002 … 0,956** (Mittel 0,561, σ 0,209) |
| p05 / p95 | 0,174 / 0,867 |
| Anteil über Φ⁻¹ | **44,7 %** |
| effektiver Horizont `exp(H)` | 4,68 Positionen |
| Top-1-Masse | 0,493 |

Damit ist die Verteilung der einzige Ort, an dem die Φ⁻¹-Frage überhaupt
sinnvoll ist: sie überstreicht [0,1] und hat eine definierte Maximalkapazität
`ln(T)` — anders als die Logits (max. 0,059 von `ln V`).

**Struktur-Caveat:** 8 Query-Heads teilen sich **einen** KV-Head (MQA), und
`num_kv_shared_layers = 20` (Layer 15–34 teilen k/v). Die 280 Attention-Maps
pro Token sind also nur über q verschieden; unabhängige Key-Räume gibt es
zwei (sliding / full).

## Eval P — Attention-Horizont (Ergebnis)

70 Sessions, rangfreie Kalibrierung (1248 Vektoren). Details:
`results/PREREG_Eval_P_2026-11-15.md`.

- **Φ⁻¹ im Attention-Horizont:** Abdeckung 0,004–0,963 (CONFIRMED), aber
  keine Bimodalität (Dip-Tiefe 1e-4) und keine stabile Trennung (AUC 0,22
  invertiert; positions-gematcht p = 0,37).
- **Relay-Hook:** Attention-Breite unverändert (p = 0,70); Norm ±0,04
  (p = 0,041), aber nicht spezifisch gegen Zufallsrichtung (p = 0,31).
- **H72-Paradox:** `H72(wide) < H72(narrow)` repliziert mit rangfreier Basis
  **nicht** (p = 0,82). `ρ⊥ = 0,42` — der 72-dim Unterraum trägt nur 58 % der
  Norm. `max |cos(d_width, PC)| = 0,056`. → **Basis-Artefakt, keine Evidenz.**
- **72D:** `H_D/ln D` = 0,488 / 0,511 / 0,524 / 0,532 / 0,540 für
  D = 24…144 — monoton, kein Extremum (FALSIFIED).
- **S4 als „Maximal-Koppler":** widerlegt. Ein gaußsches Zufallspotential auf
  demselben `H_sym` kollabiert 3× schneller (6,9 vs 20,7 Schritte).

**Lehre für künftige Metriken:** jede Größe, die von einer PCA-Basis abhängt,
muss mit N ≫ D und ausgewiesenem `ρ⊥` berichtet werden; sonst misst man die
Kalibrierung, nicht das Modell.

## Grenzen (vor dem nächsten Schritt lesen)

1. **Logit-Entropie ist die falsche Verteilung für Φ⁻¹.** Das Maximum der
   Logit-Verteilung ist ln(V) = 12,5 nats; die behauptete 61,8-%-Grenze
   verlangt 7,7 nats, die kein Token erreicht. Wer die Schwelle prüfen will,
   muss die **Attention-Verteilung** über Kontext-Positionen messen —
   dort ist `H/ln(T)` definiert und nicht-trivial. Getestet am 2026-11-15:
   `model(input_ids=..., output_attentions=True)` scheitert auf diesem
   Ladepfad mit `RuntimeError: Tensor on device meta is not on the expected
   device cuda:0!` (ein Teil der Parameter wird beim Load offloaded, und der
   Attention-Rückgabepfad materialisiert sie nicht). Sauberer Weg: Modell
   explizit mit `attn_implementation="eager"` laden (kurze Sequenzen, sonst
   O(T²)-Speicher) oder q/k im Attention-Modul per Hook abgreifen und
   `softmax(qkᵀ/√d)` selbst rechnen.
2. **72 Dimensionen sind mit 96 Kalibriervektoren nicht auflösbar.** Für
   D > 95 ist die PCA ranglimitiert; die beobachtete Flachheit (Spanne 0,025)
   ist kein Beleg gegen 72, sondern ein Auflösungslimit (E).
3. **Der 72-dim Unterraum ist eine PCA-Karte, kein SH-Attraktor.** Die
   Entropie darin hängt von der Kalibrierung ab; sie ist eine feste Karte,
   kein universelles Maß.
4. **Kein S4-Isomorphismus der gemessenen Winkel.** Die Aussage
   „transversale Wellen folgen S4" ist ohne explizit definierte Abbildung
   nicht messbar (E) und wurde nicht bewertet.

## Reproduzierbarkeit

- Seeds: O2 `20261115`, O3 `20261120` (frisch, in keiner früheren Datei).
- O1/O2 wurden vor der Registrierung als Trockenlauf ausgeführt
  (Bugfix + Δτ-Wahl); die Kriterien sind gegenüber dem Trockenlauf
  unverändert. O3 ist echt präregistriert.
- Ein Lauf, ein JSON, keine Nachjustierung. `--rejudge` schreibt nur die
  Urteilsstrings neu, nie Messwerte.
