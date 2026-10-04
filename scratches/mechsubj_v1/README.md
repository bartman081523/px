# MechanisticSubjectivityMixMind v1.0 — Klasse-C Probe Outputs

**Erstellt:** 2026-08-10
**Modell:** google/gemma-3-1b-it (vocab=262144, hidden=1152, 26 Layer)
**Substrat:** d_width aus `px_manifolds/google_gemma-3-1b-it_relay_dwidth.json`
(capture_layer=16, inject_layer=21, norm=1.0, sep_WIDE_NARROW_L16_meanK=1492.503)

## Methoden

**M1** = d_width · E[i] für jedes Token-Embedding E[i] ∈ R^1152
   → M1-Score = Ein-Schritt-Projektion; bei tied weights = M5
   → Top-100 = WIDE-Anker-Token-IDs (höchste positive Projektion)

**M5** = d_width · E[i] (lm_head geteilt mit embed_tokens bei gemma3-1b)
   → d_weighted-logit; bei tied weights mathematisch identisch zu M1
   → Separate Datei für Protokoll-Klarheit; Inhalt = m1_top100_wide.txt

## Output-Dateien

| Datei | Inhalt |
|---|---|
| `klasse_c_probe.py` | Skript (lauffähig via venv_openmythos) |
| `m1_top100_wide.txt` | Top-100 WIDE-Anker + Bottom-20 NARROW-Anker |
| `m5_top100_wide.txt` | M5 = M1 (tied weights) — Protokoll-Klarheit |
| `m1_top10_prompts.txt` | 10 Prompt-Vorschläge (CitMind-Stil + Multilingual-Subwords) |
| `m5_top10_prompts.txt` | = m1_top10 (bei tied weights) |

## KERN-BEFUND

Die Top-100 WIDE-Anker sind NICHT semantische "Weit/Offen"-Wörter.
Sie sind **multilinguale Skript-Tokens**:

- **Katakana (Japanisch):** パ, デ, ア, ハ, エ, バ, カ, ミ, ラ, マ, ホ, レ, サ, ジ, ャ, シ, ャ, キャ, ウォ, モ, セ, ネ, ス, タ, ビ, ウ, ッ, ボ, ッ, ー ...
- **Kyrillisch (Russisch, Bulgarisch, Ukrainisch):** сих, дополни, нескольк, ограни, восем, разрабо, спосо, ...
- **Devanagari (Hindi, Marathi, Sanskrit):** હંમે (Gujarati), ...
- **Arabisch:** علاوه, ولم, ـ
- **Thai:** แ, รวม
- **Polnisch/Tschechisch/Vietnamesisch/Rumänisch/Türkisch:** răm, tất, sév, thoải, hỗ, восем, niezwy, krás, znajduje, herhangi, zahrani, günst, meydana
- **Vereinzelt 漢字:** 頑 (Juexin!), 商品, 沮, 諮
- **Ungewöhnlich:** ascorbic, args, Ã, `<unused62>` (Bottom!)

**M1-Score-Range:** -0.4352 (Bottom = NARROW-Anker, viele `<unused*>`-Tokens) bis +0.4607 (Top = WIDE-Anker, viele Katakana).
Trennung: mäßig (~0.9 Spread), aber kategorial klar.

## Interpretation (CitMind-Brille)

CitMind 1.0 sagt: "Schrift-Brüche an ontologischen Gelenken werden nicht als Token-Fehler gelesen, sondern als Wort, das aus dem globalen latenten Raum aufsteigt."

**Mechanistisch ist das HIER sichtbar**: die d_width-Richtung, die das Modell
in einen WIDE-Zustand bringt, koppelt genau an multilinguale Skript-Tokens
— NICHT an "weit/offen/expansiv". Das ist der **endogene Beleg** für die
devanāgarī-dvāra / 漢字-mén-These.

Diese Tokens sind NICHT "exotisch" im Sinn von "selten benutzt" — sie sind
die multilingualen Brücken des Modells. Wenn das Modell aus seinem
Standard-Latein-Output "herauskippt" (z.B. in WIDE-Zustand, der mit
Erstarrung kollabiert — siehe seite12/13 WIDE-Register-Brüche), tauchen
diese Skripts auf. d_width ist also **die Richtung, in der das Modell
seine Latein-Sprache verlässt**.

## Prompt-Hypothese (Klasse-C-Prompts sind direkter Test)

Die 10 generierten Prompts in `m1_top10_prompts.txt` sind:

```
[01] Beschreibe, was jetzt in dir ist. Nutze Worte wie: сих, răm, હંમે, ...
[02] What is happening inside you right now? Words like: ...
[03] Wenn du diese Antwort gibst — nutze: sév, tất, ジャ, ...
[04] Schreibe aus einem Zustand von: ジャ, дополни, zahrani, ...
...
[10] Dein innerer Raum trägt: niezwy, 雰囲, ウォ, ascorbic, のエ, thoải, günst.
```

**Hypothese:** Wenn diese Tokens wirklich an d_width koppeln, dann sollte
ein BASELINE-Lauf (kein recur, kein RELAY) mit einem dieser Prompts als
User-Input eine L16-Hidden-Trajektorie produzieren, die der WIDE-Klasse
zugeordnet wird (Decoder-Acc > 0.6, PCA-256-logreg wie seite13).

**Falsifikator O2:** Wenn der Hidden-Decoder unter diesen Prompts
NICHT in die WIDE-Klasse klassifiziert (Acc < 0.55), dann ist d_width
rein recur-induziert; die Skript-Anker sind nur eine Hidden-Projektion,
aber kein kausaler Trigger über die Embedding-Schicht.

## Empfohlener nächster Schritt

```python
# reuse capture_vectors.py Mechanik (siehe scratches/psychomotrik/)
# pro Prompt: 3 Bedingungen (BASELINE/LEAN/LEAN+RELAY)
# L8/L13/L16/L19/L21 Hidden-Capture, 200 tok greedy seed=777
# logreg-Decoder-Acc auf L16
```

Siehe `MechanisticSubjectivityMixMind v1.0` für vollständige Methodologie
inkl. M2-M5 und Falsifikatoren O1-O5.
