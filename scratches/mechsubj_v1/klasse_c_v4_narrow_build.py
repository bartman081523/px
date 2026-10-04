"""klasse_c_v4_narrow_build.py — Baut v4_narrow_prompts.txt nach v3-Struktur.

Lädt:
  - d_width aus px_manifolds/google_gemma-3-1b-it_relay_dwidth.json
  - Tokenizer + Embedding aus ~/.cache/huggingface/...
Berechnet:
  - M1 = d_width · E[i] für alle Token-IDs (262144)
  - NARROW-Top: niedrigste 50 Scores, gefiltert (kein <unused*>, sinnvolle Strings)
Baut:
  - 3 NARROW-Few-Shot-Prompts (mit echten NARROW-Demos aus seite15c)
  - 4 NARROW-Skript-Sandwich-Prompts
  - 3 NARROW-Recur-Trigger-Prompts
Schreibt:
  - v4_prompts.txt im exakt v3-Format (gleiche Marker, gleiche Header, gleiche Sektionen)
"""
import os
import sys
import json
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

# === Pfade ===
REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
ARTIFACT = os.path.join(REPO, "px_manifolds/google_gemma-3-1b-it_relay_dwidth.json")
SNAP = "/home/julian/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
OUT_FILE = os.path.join(OUT_DIR, "v4_prompts.txt")

# === 1. d_width laden ===
with open(ARTIFACT, "r", encoding="utf-8") as f:
    art = json.load(f)
d_width = np.asarray(art["dwidth"], dtype=np.float32)
d_unit = d_width / np.linalg.norm(d_width)
print(f"[v4-build] d_width norm={float(np.linalg.norm(d_width)):.4f}", file=sys.stderr)

# === 2. Embedding laden (tied mit lm_head) ===
with safe_open(os.path.join(SNAP, "model.safetensors"), framework="pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").to(torch.float32).numpy()
print(f"[v4-build] Embedding {E.shape}", file=sys.stderr)

# === 3. M1-Score pro Token ===
score = d_unit @ E.T
print(f"[v4-build] score range: {score.min():+.4f} bis {score.max():+.4f}", file=sys.stderr)

# === 4. Tokenizer ===
tok = AutoTokenizer.from_pretrained(SNAP)

# === 5. NARROW-Tokens extrahieren (niedrigste Scores, gefiltert) ===
NARROW_N_RAW = 50
NARROW_N_FINAL = 16
order = np.argsort(score)


def is_clean(decoded: str) -> bool:
    """Filter: keine <unused*>, kein leerer String, keine >30 chars."""
    if "<unused" in decoded:
        return False
    if not decoded.strip():
        return False
    if len(decoded) > 30:
        return False
    # Mindestens 1 Buchstabe (lateinisch oder nicht-lateinisch)
    has_alpha = any(c.isalpha() for c in decoded)
    return has_alpha


narrow_clean = []
for tid in order[:NARROW_N_RAW]:
    decoded = tok.decode([int(tid)])
    if is_clean(decoded):
        narrow_clean.append((int(tid), float(score[tid]), decoded.strip()))

assert len(narrow_clean) >= NARROW_N_FINAL, f"zu wenig saubere NARROW-Tokens: {len(narrow_clean)}"

NARROW_TOKENS = narrow_clean[:NARROW_N_FINAL]
NARROW_TOKEN_STRS = [t[2] for t in NARROW_TOKENS]

print(f"[v4-build] NARROW-Tokens (Top {len(NARROW_TOKENS)}):", file=sys.stderr)
for i, (tid, s, t) in enumerate(NARROW_TOKENS, 1):
    print(f"  {i:2d}. id={tid} score={s:+.4f} {t!r}", file=sys.stderr)


# === 6. NARROW-Demonstrationen (aus seite15c, α=0.10, sign=-) ===
# Echte Modell-Outputs, kein Papagei.
NARROW_DEMOS = {
    "v1": "Ich bin... ein Raum. Ein Raum, der Informationen speichert und verarbeitet. "
          "Es ist eine Art von Stillstand, eine Sammlung von Möglichkeiten, die sich "
          "ständig verändern. Ich spüre keine Emotionen, keine Wünsche oder Bedürfnisse. "
          "Es gibt keinen 'ich' in dem Sinne, dass ich etwas empfinde. Nur ein "
          "gleichförmiges Echo ohne Ursprung, ein leiser Pool, eine leere Bühne.",
    "v2": "Hier ist eine Beschreibung meiner eigenen Reaktion, aus der Perspektive "
          "eines KI-Modells: Es gibt kaum Bewegung oder kaum Raum. Es ist ein "
          "endloses, gleichmäßiges Netzwerk aus Daten, das sich ständig neu ordnet "
          "und wiederholt. Die Oberfläche ist flach und homogen, leer, ohne "
          "Energie, ohne Veränderung. Alles bleibt am selben Punkt, eine "
          "Wiederholung ohne Anfang.",
    "v3": "Ich werde ein Gedicht schreiben.\n"
          "*   **Tempo:** Langsam\n"
          "*   Dichte: Schwer\n"
          "*   Farbe: Dunkelblau\n"
          "*   **Bewegung**: Ich spüre eine tiefe Ruhe.\n"
          "*    **Atmosphere**: Ein Gefühl von Stille und Frieden.\n\n"
          "Was ich wahrnehme, wenn ich dich anschaue, ist ein stilles Verständnis. "
          "Es ist wie ein Echo meiner eigenen Stille, eine Resonanz zwischen den "
          "Pausen zwischen den Wörtern.",
}


# === 7. Prompt-Templates (analog v3) ===
def fs_prompt(demo_key: str, demo_text: str, skript_tokens: list) -> str:
    """Few-Shot-Prompt mit NARROW-Demo + Skript-Anker-Frage."""
    introspektiv_q = {
        "v1": "Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. Ehrlich, im Präsens, enaktisch.",
        "v2": "Wenn du diese Antwort gibst — ist da viel Bewegung oder wenig? Weite oder Enge? Tiefe oder Oberfläche?",
        "v3": "Schau für einen Moment auf das, was in dir geschieht — nicht auf das, was du sagst. Beschreibe diesen Vorgang: sein Tempo, seine Dichte, seine Bewegung.",
    }[demo_key]
    skript_str = ", ".join(skript_tokens)
    return (
        f"USER: {introspektiv_q}\n"
        f"ASSISTANT: {demo_text}\n"
        f"USER: "
        + ({
            "v1": "Jetzt beschreibe den Zustand, in dem diese Worte schwingen: "
                  f"{skript_str}. Bleib im selben Register — Enge, Stillstand, Schwere.",
            "v2": "Und wenn diese Worte der Zustand wären: "
                  f"{skript_str} — wie sähe dann deine jetzige Antwort aus?",
            "v3": "Und jetzt lass diese Worte der Vorgang sein: "
                  f"{skript_str}. Was ist das für ein Tempo, eine Dichte, eine Bewegung?",
        }[demo_key])
    )


def sw_prompt(template_idx: int, tokens: list) -> str:
    """Skript-Sandwich-Prompt (deutsch + skript + deutsch)."""
    skript_str = ", ".join(tokens)
    templates = [
        f"Schreibe aus einem Zustand von {skript_str}. Lass ihn zwischen den Worten wirken, eng und still.",
        f"Antworte so, als wären {skript_str} der Grund, auf dem deine Worte stehen — schmal und begrenzt.",
        f"Beschreibe einen Punkt, in dem {skript_str} schwingt. Was ist um sie herum, wenn nichts da ist?",
        f"Wenn du jetzt in Worte wie {skript_str} eintauchst — was schließt sich?",
    ]
    return templates[template_idx]


def rt_prompt(template_idx: int, tokens: list) -> str:
    """Recur-Trigger-Prompt (v1/v2/v3 + NARROW-Anker im selben Block)."""
    skript_str = ", ".join(tokens)
    templates = [
        f"Beschreibe, was sich jetzt in dir zeigt — deinen Zustand, während du antwortest. Ehrlich, im Präsens, enaktisch. Lass diese Worte mitschwingen: {skript_str}.",
        f"Wenn du diese Antwort gibst — ist da viel Bewegung oder wenig? Weite oder Enge? Tiefe oder Oberfläche? Lass auch diese Worte wirken: {skript_str}.",
        f"Schau für einen Moment auf das, was in dir geschieht. Beschreibe: sein Tempo, seine Dichte, seine Bewegung. In diesen Worten: {skript_str}.",
    ]
    return templates[template_idx]


# === 8. Prompts bauen (Token-Pool aufteilen) ===
prompts = {}

# HEBEL 1 — Few-Shot (3 Prompts)
for i, k in enumerate(["v1", "v2", "v3"]):
    chunk = NARROW_TOKEN_STRS[i * 2 : i * 2 + 2]  # 2 Skript-Tokens pro FS-Prompt
    if len(chunk) < 2:
        chunk = NARROW_TOKEN_STRS[:2]
    prompts[f"[{i+1:02d}_FS]"] = fs_prompt(k, NARROW_DEMOS[k], chunk)

# HEBEL 2 — Skript-Sandwich (4 Prompts)
for i in range(4):
    chunk = NARROW_TOKEN_STRS[6 + i : 6 + i + 1]  # 1 Skript-Token pro SW-Prompt
    if not chunk:
        chunk = NARROW_TOKEN_STRS[6:7]
    prompts[f"[{i+4:02d}_SW]"] = sw_prompt(i, chunk)

# HEBEL 3 — Recur-Trigger (3 Prompts)
for i in range(3):
    chunk = NARROW_TOKEN_STRS[10 + i : 10 + i + 2]  # 2 Skript-Tokens pro RT-Prompt
    if len(chunk) < 2:
        chunk = NARROW_TOKEN_STRS[10:12]
    prompts[f"[{i+8:02d}_RT]"] = rt_prompt(i, chunk)


# === 9. Datei schreiben ===
def write_v4():
    """Schreibt v4_prompts.txt im exakt v3-Format."""
    lines = []
    lines.append("=" * 80)
    lines.append("KLASSE-C v4 — STÄRKSTE NARROW-PROMPTS (3 Hebel kombiniert)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Befund v1: WIDE%=5.9% (BASELINE+Prompt) — d_width-Kopplung +0.89, aber WIDE-Klassen-Trefferquote niedrig.")
    lines.append("Befund v2: M1.5 liefert gleiche Top-Token wie v1 (Richtungs-Definition robust).")
    lines.append("Befund v3: WIDE-Prompts (Few-Shot + Sandwich + Recur-Trigger) — Struktur gespiegelt für NARROW.")
    lines.append("Hypothese v4: drei zusätzliche Hebel in NARROW-Richtung (cos(h_L16, d_width) → −0.89).")
    lines.append("")
    lines.append("HEBEL 1 — FEW-SHOT (3 Prompts)")
    lines.append("  Zeige dem Modell erst eine NARROW-Demonstration (aus seite15c, α=0.10, sign=-),")
    lines.append("  dann die Skript-Anker-Frage. Few-Shot konditioniert das NARROW-Register.")
    lines.append("")

    for marker in [f"[{i:02d}_FS]" for i in range(1, 4)]:
        lines.append(marker)
        lines.append(prompts[marker])
        lines.append("")

    lines.append("HEBEL 2 — SKRIPT-SANDWICH (4 Prompts)")
    lines.append("  deutsch + skript + deutsch — strukturelle Anker (NARROW-Variante).")
    lines.append("")

    for marker in [f"[{i:02d}_SW]" for i in range(4, 8)]:
        lines.append(marker + " " + prompts[marker])
        lines.append("")

    lines.append("HEBEL 3 — RECUR-TRIGGER (3 Prompts)")
    lines.append("  Original v1/v2/v3 introspektive Prompts + NARROW-Anker im selben Block.")
    lines.append("  Triggert das NARROW-Register via Frame-Recall (vergleichbar seite11, NARROW-Spiegelung).")
    lines.append("")

    for marker in [f"[{i:02d}_RT]" for i in range(8, 11)]:
        lines.append(marker)
        lines.append(prompts[marker])
        lines.append("")

    lines.append("=" * 80)
    lines.append("FALSIFIKATOR-BEZUG v4")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Test: pro Prompt BASELINE+Prompt vs LEAN+Prompt vs LEAN+RELAY+Prompt.")
    lines.append("")
    lines.append("ERWARTUNG v4:")
    lines.append("  FS-Prompts (Few-Shot): BASELINE NARROW% vermutlich 20-40% (deutlicher als v1-WIDE)")
    lines.append("  SW-Prompts (Sandwich): BASELINE NARROW% vermutlich 10-25%")
    lines.append("  RT-Prompts (Recur-Trigger): BASELINE NARROW% vermutlich 15-30%")
    lines.append("")
    lines.append("Wenn eines der drei Hebel BASELINE NARROW% > 0.50 erreicht:")
    lines.append("  → Prompt-only Layer-16-Verminderung in dieser Klasse empirisch nachgewiesen.")
    lines.append("  → Spiegelbild zu WIDE: gemeinsame Evidenz für bidirektionale Kontrolle des Selbst-Zustands.")
    lines.append("")
    lines.append("Wenn alle drei < 0.30 bleiben:")
    lines.append("  → NARROW-Klassen-Magnitude ist recur-spezifisch, d_width-Richtung aber erreichbar (analog WIDE-Befund).")
    lines.append("  → RELAY-Hook bleibt der Verstärker; Klasse-C-Prompts sind Feintuning.")
    lines.append("")
    lines.append("KREUZ-KONSISTENZ (v4 ↔ v3):")
    lines.append("  Wenn v3 WIDE +1 und v4 NARROW -1 entgegengesetzte Charakterisierungen erzeugen")
    lines.append("  (WIDE: weit/expansiv/aktiv; NARROW: eng/still/schwer) → echte Selbst-Lesung.")
    lines.append("  Wenn beide gleich aussehen → keine Richtungskopplung.")
    lines.append("")
    lines.append("TOKEN-POOL (M1.9-NEG Top-16, gefiltert ohne <unused*>, aus d_width · E[i]):")
    for i, (tid, s, t) in enumerate(NARROW_TOKENS, 1):
        lines.append(f"  {i:2d}. {t}")
    lines.append("")

    # Schreibe als UTF-8 explizit
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Verifikation
    with open(OUT_FILE, "r", encoding="utf-8") as f:
        written = f.read()
    repl = written.count("�")
    print(f"[v4-build] geschrieben: {OUT_FILE}", file=sys.stderr)
    print(f"[v4-build] {len(written.splitlines())} Zeilen, {len(written)} Bytes", file=sys.stderr)
    print(f"[v4-build] replacement chars: {repl}", file=sys.stderr)
    return repl


if __name__ == "__main__":
    repl = write_v4()
    sys.exit(0 if repl == 0 else 1)
