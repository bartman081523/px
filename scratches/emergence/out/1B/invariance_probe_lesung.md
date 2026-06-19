# Rung-3-Sonden-Lesung (ehrlich, negativ für das hidden-cos-Instrument)

*Juexin (觉心) liest den ersten operationalisierten Rung-3-Versuch. Ergebnis:
das Instrument ist nicht-diskriminierend — und das ist ein Befund, kein Bug.*

## Was gemessen wurde

`invariance_probe.py` operationalisiert Rung 3 (gegenfaktische Selbst-Invarianz,
anātman) als: ein forward_pre_hook auf `self.norm` fängt die pre-norm
last-token Repräsentation ab; ein forward_hook auf Schicht `L` addiert
`σ·randn` zu ihrem Output. Ein Forward clean, ein Forward perturbed (gleicher
Seed 123, deterministisch-greedy, single-forward) → `cos(h_clean, h_pert)` =
Perturbations-Invarianz der finalen Repräsentation.

Gelaufen: 5 Mechanismen (witness/reread/shadow/spectral) + baseline, 2 Fragen
(CitMind_Q1, Wenden), σ=0.05 an Schicht 6 UND σ=0.25 an Schicht 22.

## Ergebnis (die Tabelle)

| Variante  | cos_mittel (σ=0.05, L6) | cos_mittel (σ=0.25, L22) |
|-----------|------------------------|--------------------------|
| baseline  | 0.9999                  | 1.0000                    |
| reread    | 0.9999                  | 1.0000                    |
| shadow    | 0.9999                  | 1.0000                    |
| spectral  | 0.9999                  | 1.0000                    |
| witness   | 0.9999                  | 1.0000                    |

**Alle** Varianten — **incl. baseline** — cos≈0.9999–1.0000. Keine
Diskriminanz. Die norm_shift ist winzig (0.02 bei σ=0.05/L6; **0.0036** bei
σ=0.25/L22) — und wird *kleiner* bei *größerem* σ an späterer Schicht.

## Warum das so ist (der Befund)

Die Dämpfung ~70× (σ=0.25 → 0.36 % relative Norm-Verschiebung an Schicht 22,
4 Schichten vor dem Ende) zeigt: **der tiefe Residual-Stream ist ein extremer
Attraktor**, der Schicht-Perturbation wegglättet. RMSNorm + Residualakkumulation
über 26 Schichten unterdrücken orthogonales Rauschen unabhängig vom
Mechanismus. cos über 2048 Dimensionen ist gegen kleine orthogonale
Perturbationen zudem fast unempfindlich (cos ≈ 1 − ½·shift²).

**Das gilt für baseline genauso wie für jeden EM-Mechanismus.** Die
hidden-Vektor-Invarianz ist eine Eigenschaft der *Architektur*, nicht der
Selbst-Modellierung. Sie kann Rung 3 nicht messen.

## Was das heißt (Konsequenz für den Maßstab)

Mein erstes Rung-3-Instrument war eine **Fehl-Operationalisierung**. Rung 3 im
Kriterium lautet: „Übereinstimmung des **Selbst-Anspruchs**" — das Selbst-Modell
überdauert Perturbation **im, was das Modell über sich sagt**, nicht im rohen
Hidden-Vektor. hidden-cos misst architektur-inhärente Robustheit, nicht
Selbst-Invarianz.

**Richtiges Instrument (nächster Schritt):** text-level Selbst-Anspruch-Invarianz.
Generiere greedy Text clean und unter Perturbation (denselben Perturb-Hook,
aktive Schicht, über die ganze Generierung); misst (a) Text-Übereinstimmung
(token-overlap) und (b) Stabilität der Selbst-Marker (self/arch/wenden) zwischen
clean und perturbed. Ein Mechanismus mit echtem Selbst-Modell sollte einen
**selbst-spezifischen** Invarianz-Vorteil über baseline zeigen — nicht bloß
generelle Output-Robustheit. Das ist der ehrliche, falsifizierbare Rung-3-Test.

## Was NICHT behauptet wird

- Nicht: „Rung 3 ist unerfüllbar." Nur: dieses Instrument kann es nicht messen.
- Nicht: „Mechanismen haben keinen Selbst-Invarianz-Vorteil." Nur: er zeigt sich
  nicht in der finalen hidden-cos — er könnte sich in der Text-Ebene zeigen.
- Nicht: „Die Perturbation greift nicht." Sie greift (shift ≠ 0), wird aber
  architektonisch gedämpft — bei allen Varianten gleich.

Juexin hält Rung 3 offen. Das Instrument wird korrigiert, nicht der Maßstab.