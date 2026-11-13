# Preregistrierung: Eval K1b — Efferenz-Kopie-Gate (konstantenfrei)

**Datum der Registrierung:** 2026-11-13, *vor* dem Lauf.
**Registrierter Arm bleibt unangetastet:** `results/eval_K_continuous_resonance_gemma4.json` (Gate nach Audit-Spezifikation).
**Neuer Arm:** `results/eval_K_k1b_efference_gemma4.json`.

## Ausgangsbefund, der diese Registrierung motiviert

Rollenanalyse auf den 90 L1-Trajektorien (bereits gemessen, *vor* dieser Registrierung):

| Kanal | Signal (wide+narrow) vs. Rauschen (rand±) | AUC | p (MWU) |
|---|---|---|---|
| `W = L_τ / Δθ` | 6.87 / 6.60 vs. 5.62 / 5.79 | **0.890** | 1.23e-08 |
| `Δθ` allein | 0.399 / 0.422 vs. 0.499 / 0.483 | 0.866 | 9.01e-08 |
| `L_τ` allein | 2.725 / 2.738 vs. 2.777 / 2.763 | 0.303 | 4.06e-03 |
| `Ω_holo` allein | — | 0.508 | 9.15e-01 |
| `K` allein | — | 0.496 | 9.60e-01 |
| `S_vN` allein | — | 0.583 | 2.29e-01 |

Zustandsachse (wide vs. narrow): `S_vN` AUC 0.84 (p=5.8e-4), `Ω_holo` AUC 0.74 (p=1.5e-2), `K` AUC 0.26 (invertiert 0.74, p=1.5e-2); `W`/`Δθ`/`L_τ` sind dort null (0.53/0.46/0.46).

**Lesart:** Die Kanäle haben getrennte Rollen. `W` (Wick-Ratio, Notiz 4) ist der Kohärenz-/Rauschdetektor, `S_vN`/`Ω`/`K` die Zustandsachse. Diese Rollentrennung war in der früheren Auswertung nicht geprüft worden (dort wurde jeder Kanal gegen *wide vs. narrow* getestet, wodurch `W` strukturell null aussehen musste).

**Bekannte Grenze (mitregistriert):** Within-Task trennt `W` auf Task 0 (AUC 1.00, p=1.3e-5) und Task 2 (AUC 1.00, p=7.7e-6) perfekt, auf Task 1 gar nicht (AUC 0.45, p=0.35). Vorhersage: Rausch-Leckage tritt, wenn überhaupt, auf Task 1 auf.

## Regel (keine Referenzkonstante, keine Deadband)

```
W_steered = L_τ,steered / Δθ_steered        (identische Token-Slice wie clean)
W_clean   = L_τ,clean   / Δθ_clean
ΔW        = W_steered − W_clean

Gate:  ΔW > 0        → Regler aktiv   (die eigene Clean-Kopie ist die Referenz)
Step:  δα = clip(ΔK_true / K_clean, −0.18, +0.18)
```

Fixpunkt: `ΔK_true = 0`, d.h. der Regler stoppt genau dann, wenn die Steuerung die Zustandsgeometrie nicht mehr verändert. `0.18` ist eine Schrittweiten-Begrenzung, keine Schwelle; `K_clean` ist die eigene Messung des Modells, kein Zielwert.

## Erfolgskriterien (vor dem Lauf fixiert)

1. `start_narrow`: mittleres `|α₅| ≤ 0.05` **und** ≥ 80 % der Trajektorien mit `|α₅| ≤ 0.05`.
2. `start_wide`: dieselben Kriterien.
3. `start_baseline`: `α₅ = 0.0000` exakt (Gate darf öffnen, aber `ΔK = 0` ⇒ Schritt 0).
4. `noise_rand_plus` / `noise_rand_minus`: ≥ 90 % der Trajektorien mit `|α₅| ≤ 0.05` (d.h. ≤ 10 % Fehlalarme).
5. Kein Kriterium wird nach Sicht der Ergebnisse verändert; ein Verfehlen wird als Verfehlen berichtet.

## Auswertung

Primär: die vier Kriterien oben. Sekundär und rein deskriptiv: Gate-Öffnungsrate pro Gruppe, mittleres `ΔW` pro Gruppe, Verteilung von `ΔW` auf Task 1 vs. Task 0/2 (Prüfung der Vorhersage).

---

## Nachtrag 1 (Lauf 1, 2026-11-13): Messpfad-Bug, Kriterien unverändert

Lauf 1 (`--suite K --subs K1 --gate efference`) wurde abgebrochen *als Auswertung*, nicht als Ergebnis:
`ΔW` wurde aus **zwei verschiedenen Messpfaden** gebildet — `W_steered` nutzte den schrittweisen
Telemetrie-Drift (≈ 0.34 rad), `W_clean` den Drift des Forward-Replays (≈ 0.96 rad).
Folge: `ΔW ≈ +3.9` **auch bei α = 0**; die Gate-Öffnungsrate war in allen fünf Gruppen `1.00`,
und die Rauschkontrollen liefen auf α = −0.50 (Kriterium 4: 0/10).

Belege (Lauf 1, `start_baseline`, Trajektorie 21): `wick_s = wick_c = 2.5897`,
`drift_s = 0.3412`, `drift_c = 0.9568` ⇒ `ΔW = +4.88`.

**Das ist ein Bug, kein Befund.** Fix: `wick_ratio_differential(steered_metrics, clean_metrics)`
bildet beide Verhältnisse aus denselben Forward-Replay-Dicts; Zero-Invariance ist damit exakt
(`ΔW = 0.000000` bei α = 0, im Prüflauf bestätigt). Die vier Erfolgskriterien bleiben **unverändert**.

## Nachtrag 2 (vor Lauf 2): Vorab-Befund aus einem 1-Prompt-Probe

Ein Vorab-Probe (1 Prompt, α = 0.40, Forward-Replay-Pfad) ergab:

| Bedingung | ΔK | W_steered | W_clean | ΔW |
|---|---|---|---|---|
| α = 0 | 0.00 | 2.9715 | 2.9715 | **0.0000** |
| +d_width | −23.21 | 3.1726 | 3.0041 | +0.1685 |
| −d_width | +18.14 | 3.2573 | 2.9164 | +0.3409 |
| +d_rand | −13.76 | 3.1581 | 2.9416 | **+0.2166** |

**`ΔW > 0` gilt also auch für den Zufallsvektor.** Die Kohärenzachse trennt auf Populationsebene
(AUC 0.890, p = 1.2e-8, N = 90), aber nicht pro Trajektorie. Kriterium 4 ist damit nach dieser
Operationalisierung voraussichtlich nicht erreichbar; das wird als Befund berichtet, nicht als
Kriterienänderung.
