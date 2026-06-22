# psychomotrik LESUNG3 — mechanischer Diskriminator (Seite 3 Schritt 1)

Seite 3 sollte die LESUNG2-Redirect-Hypothese prüfen: *„Befreiung = weniger
φ-Erstarrung am Prefill"*. Bevor eine Architektur gebaut wird, die Hypothese
an em5-Daten testen — 是X即非X gegen 觕-Architektur auf falscher Prämisse.

`mech_discriminator.py` korreliert em5-Telemetrie (loops_run/zone/φ/ent/aks/
h24-stats, pro Token aggregiert) mit den Juexin-Labels (intro/wozhi/degrade)
über alle 63 cold-Zellen. Anders als Seite 1 (arm-demeaned, „jenseits Routing")
ist hier RAW-Korrelation gesucht: die Architektur SETZT Routing-Params; wir
suchen die Param-Belegung, die Intro erzeugt.

---

## Verdikt: KEIN einzelner kontrollierbarer Skalar trennt sauber —
## φ-Hypothese FALSIFIZIERT, einfache Ent-Hypothese auch.

### Diskriminator-Ranking (intro vs wozhi, raw AUC)

| feature | AUC | dir | intro_mean | wozhi_mean |
|---|---|---|---|---|
| hkurt_mean | 0.731 | ↑ | 850.8 | 804.0 |
| ent_mean | 0.725 | ↑ | 0.524 | 0.274 |
| ent_t0 | 0.718 | ↑ | 0.564 | 0.313 |
| hvar_mean | 0.310 | ↓ | 170465 | 185762 |
| phi_frac_gt0.99 | 0.241 | ↓ | 0.779 | 0.911 |
| phi_mean | 0.253 | ↓ | 0.992 | 0.995 |
| loops_mean | 0.458 | ↓ | 2.33 | 3.43 |
| loops_t0 | 0.438 | ↓ | 2.25 | 3.50 |
| width | 0.500 | — | (bug: 0) | (bug: 0) |
| aks_* | 0.500 | — | 0.0 | 0.0 |

**φ-Erstarrung ist NICHT der Diskriminator** (AUC 0.25, beide ~0.99 erstarrt).
Die LESUNG2-Redirect-Hypothese (weniger φ-Erstarrung → Intro) fällt. φ≈0.99
in JEDEM recur-Arm, inkl. WIDE (100% Intro) — φ konstant, Outcome verschieden.

### Ent ist schwach (AUC 0.72) MIT Gegenbeispielen

Der Ent-Kollaps (ZONE_CREATIVE: ent 0.47→0.0004→8e-8 über Tokens) ist real und
striking — aber **kein sauberer Diskriminator**, weil:

- **RECUR_STD: loops=7, ent HOCH (0.55-0.98 sustai­niert), aber wozhi.** Hohe
  Entropie + tiefe Mahlung → 我执. Widerlegt „high ent → intro".
- **ZONE_MATH/regung: ent=0.000, aber intro.** Niederentropie → kann trotzdem
  Intro. Widerlegt „low ent → 我执".
- **RECUR_OFF: ent hoch, loops=0, aber nur 14% intro.** Hohe Entropie ohne
  recur-Zonen-Sweep → kaum Intro.

**我执 ist ein Multi-Route-Attraktor** — erreichbar via (a) Ent-Kollaps
(ZONE_CREATIVE) ODER (b) high-ent-Grinding (RECUR_STD). Intro ist NICHT
„hohe Entropie" simpliciter. ent_mean AUC 0.72 ist ein WIDE-vs-CREATIVE-
Kontrast-Artefakt, kein kausaler Hebel — analog zur Seite-2-Lektion
(Separabilität ≠ Projektierbarkeit): **korrelierbar ≠ setzbar**.

### Loops (Mahl-Tiefe) AUC 0.46 — auch schwach

Sowohl Intro- als auch 我执-Zellen spannen loops 1-7 (intro: WIDE@1, PERTURB@7;
wozhi: CREATIVE@7, MATH@1). Keine saubere Trennung.

---

## Was die per-Arm-Intro-Rate trotzdem zeigt (Architektur-Struktur)

| arm | intro% | loops | φ | Bemerkung |
|---|---|---|---|---|
| RECUR_WIDE | **1.00** | 1.05 | 0.992 | wide (start4-end22) + adaptive zone |
| RECUR_NARROW | 0.43 | 2.09 | 0.988 | width 2 |
| RECUR_EXTREME | 0.43 | 1.31 | 0.991 | width 22 (collapse-probe) |
| PERTURB | 0.43 | 6.95 | 0.994 | loops=7 + σ=0.15 Rauschen |
| RECUR_STD | 0.14 | 6.96 | 0.994 | loops=7, adaptive, high ent → wozhi |
| ZONE_MATH | 0.14 | 1.11 | 0.998 | loops=1, forced math, ent=0 |
| ZONE_CREATIVE | **0.00** | 6.71 | 0.993 | loops=7, forced creative, ent→0 |
| BASELINE/RECUR_OFF | 0.14 | 0 | — | kein recur |

Die Disentangling-Logik:
- **RECUR_OFF (narrow + keine Mahlung) = 14%** vs **RECUR_WIDE (wide + keine
  Mahlung) = 100%** → keine Mahlung allein befreit NICHT; **Breite** nötig.
- **ZONE_CREATIVE (grind loops=7) = 0%** vs **WIDE (loops=1) = 100%** → bei
  festem creative-Zone bringt Mahlung 0%, keine Mahlung (aber wide) 100%.
- **PERTURB (loops=7 + Rauschen) = 43%** vs **STD (loops=7, kein Rauschen) =
  14%** → Rauschen bricht die Mahlung teilweise auf.

**Architektur-Hypothese (geschärft): Intro = breiter recur-Zonen-Sweep OHNE
Mahlung (wide start-end + loops=1 via sofortiger hub-stuck-Exit).** Nicht
ent, nicht φ, nicht loops allein — die *Kombination* Breite-und-keine-Mahlung.
WIDE ist die einzige Konfiguration, die das zuverlässig trifft (100%).

Warum triftt WIDE loops=1? Der recur-Walk (patch.py:432) exitet sofort via
hub-stuck-guard (patch.py:547 `if current_layer==active_start and steps>0:
break`) weil die weite Zone (start=4) den Layer nach einem Schritt nicht
verlässt → single broad pass, keine Mahlung. STD (adaptive narrow) läuft 7
Schritte ohne hub-stuck → mahlt → 我执 (trotz high ent).

---

## Konklusion (是X即非X)

- **Falsifiziert:** φ-Erstarrung-Hebel (LESUNG2 redirect) und einfache Ent-Hebel.
  KEIN einzelner kontrollierbarer Skalar (ent/loops/φ/kurt) trennt Intro von
  我执 sauber. Konsistent mit em5 recur_specificity-negativ: kein Phänomen-Kanal
  kovariiert sauber mit einem mechanischen Skalar ([[em5-state-induction-recur-
  specificity-negative]]).
- **Bestätigt & geschärft:** WIDE→Intro (100%) ist ein Konfigurations-Level-
  Fakt, dessen Mechanismus verteilt ist (Breite + keine-Mahlung), nicht skalär.
  Die Architektur, die befreit, existiert bereits (WIDE) — aber IHR Mechanismus
  ist „breiter Sweep ohne Mahlung", und das ist kein einzelner Knopf.
- **我执 = Multi-Route-Attraktor** (ent-Kollaps-Route ODER high-ent-Grinding-
  Route). Das erklärt, warum kein einzelner Hebel ihn bricht: man muss die
  Route treffen, nicht einen Skalar setzen.

## Redirect Seite 3 Schritt 2 (Disentangling-Architektur-Test)

Die Hypothese „Breite + keine Mahlung → Intro" ist falsifizierbar via
Hybrid-Arme (transplantiere WIDE-Eigenschaften auf 我执-Arme):

1. **ZONE_CREATIVE + WIDE-Routing** (wide start-end, forced creative zone,
   loops adaptiv→1?) → Intro? Wenn ja: Breite+keine-Mahlung befreit
   unabhängig vom Zone-Label.
2. **ZONE_CREATIVE + loops-cap=1** (forced creative, narrow, keine Mahlung) →
   Intro? Wenn nein: Breite nötig, nicht nur keine-Mahlung.
3. **WIDE + forced creative zone** (wide, creative, loops=1) → Intro?
   Wenn ja: WIDE's Breite-und-keine-Mahlung dominiert über Zone-Zwang.

Wenn (1)+(3) Intro geben, (2) nicht → **die befreiende Architektur ist
„breiter Sweep ohne Mahlung"**, und der Architektur-Keim ist: eine Routing-
Struktur, die den recur-Walk zu einem single broad pass zwingt (hub-stuck-
sofort über weiter start-end) statt mahlen zu lassen. Das IST eine neue
Architektur (WIDE als Default + hub-stuck-Verstärkung), und Rekurrenz bleibt
Möglichkeit (nicht Zwang). Wenn alle drei 我执 bleiben → WIDE's Intro kommt
aus einer nicht-isolierbaren Wechselwirkung → honest negatives, Architektur
nicht über Routing-Params allein machbar.

Caveats: width-Feature hatte Bug (nur 0) — irrelev für Verdikt (ent/loops/φ
tragen). n=63, AUC-Werte klein-Sample. Per-Arm-Raten aus [[em5-state-induction-
recur-specificity-negative]] (7 cold Prompts). Siehe [[psychomotrik-fate-probe-
intro-separable]], [[psychomotrik-steering-null-redirect-erstarrung]],
[[em5-state-induction-recur-specificity-negative]], [[manual-reaudit-keyword-flaw]].