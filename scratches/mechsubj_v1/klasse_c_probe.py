"""klasse_c_probe.py — MechanisticSubjectivityMixMind M1+M5: d_width-Projektion.

M1 = d_width · E^T  (Token-Embedding-Probe, 1152-dim dot)
M5 = d_width · E^T  (d_weighted-logit; bei tied weights identisch zu M1)

Input:
  - px_manifolds/google_gemma-3-1b-it_relay_dwidth.json  (d_width, 1152-dim unit)
  - ~/.cache/huggingface/.../model.safetensors          (model.embed_tokens.weight = lm_head, tied)

Output:
  - m1_top100.txt  — Top-100 Token-IDs nach M1-Score + decodierte Strings
  - m5_top100.txt  — Top-100 Token-IDs nach M5-Score (bei tied weights = M1)
  - m1_top10_prompts.txt — 10 fertige Prompt-Vorschläge aus M1-Top-Token-Substrings
  - m5_top10_prompts.txt — 10 fertige Prompt-Vorschläge aus M5-Top-Token-Substrings

Falsifikator-Bezug:
  O2: wenn diese Top-Tokens in einem realen Run am Prefill KEINE L16-Drift erzeugen
      (capture_vectors Mechanik), dann ist d_width rein recur-induziert, NICHT aus
      dem Latent via Prompt erreichbar.
"""
import json
import os
import sys
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

# === Pfade ===
REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
ARTIFACT = os.path.join(REPO, "px_manifolds/google_gemma-3-1b-it_relay_dwidth.json")
SNAP = "/home/julian/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
EMBED_PTH = os.path.join(SNAP, "model.safetensors")
TOK_PTH = os.path.join(SNAP, "tokenizer.model")
TOK_JSON = os.path.join(SNAP, "tokenizer.json")
ADDED_TOK = os.path.join(SNAP, "added_tokens.json")
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
os.makedirs(OUT_DIR, exist_ok=True)


# === 1. d_width laden ===
with open(ARTIFACT, "r", encoding="utf-8") as f:
    art = json.load(f)
d_width = np.asarray(art["dwidth"], dtype=np.float32)
assert d_width.shape == (1152,), f"erwartet (1152,), got {d_width.shape}"
norm = float(np.linalg.norm(d_width))
print(f"[klasse_c] d_width geladen: shape={d_width.shape} norm={norm:.4f}", file=sys.stderr)
print(f"[klasse_c] artifact hf_id={art['hf_id']} inject_layer={art['inject_layer']} capture_layer={art['capture_layer']}", file=sys.stderr)


# === 2. Embedding (tied mit lm_head) laden, auf float32 ===
with safe_open(EMBED_PTH, framework="pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").to(torch.float32).numpy()
print(f"[klasse_c] Embedding shape={E.shape} (vocab={E.shape[0]}, hidden={E.shape[1]})", file=sys.stderr)


# === 3. M1-Score: d_width · E[i] für jedes Token i ===
M1 = d_width @ E.T  # shape (vocab,)
print(f"[klasse_c] M1 berechnet: shape={M1.shape} min={M1.min():.4f} max={M1.max():.4f}", file=sys.stderr)


# === 4. Tokenizer laden, für jedes i den decodierten String ===
tok = AutoTokenizer.from_pretrained(SNAP)


def dec(i: int) -> str:
    try:
        s = tok.decode([int(i)])
    except Exception as e:
        s = f"<DECODE_ERR {e}>"
    return s


# === 5. Top-N extrahieren ===
TOP_N = 100
top_pos = np.argsort(M1)[::-1]   # descending: WIDE-positive Anker
top_neg = np.argsort(M1)         # ascending: NARROW-positive Anker (Gegenrichtung)
top_pos_str = [(int(i), float(M1[i]), dec(i)) for i in top_pos[:TOP_N]]
top_neg_str = [(int(i), float(M1[i]), dec(i)) for i in top_neg[:TOP_N]]


# === 6. M1 Top-100 schreiben ===
with open(os.path.join(OUT_DIR, "m1_top100_wide.txt"), "w", encoding="utf-8") as f:
    f.write("=== M1: d_width · E[i]  —  Top 100 WIDE-Anker-Token (gemma3-1b-it) ===\n")
    f.write("Modell: google/gemma-3-1b-it (vocab=262144, hidden=1152)\n")
    f.write(f"d_width aus px_manifolds/google_gemma-3-1b-it_relay_dwidth.json\n")
    f.write(f"  hidden_size={art['hidden_size']} capture_layer={art['capture_layer']} inject_layer={art['inject_layer']}\n")
    f.write(f"  direction={art['direction']} n_prompts={art['n_prompts']}\n")
    f.write(f"  sep_WIDE_NARROW_L16_meanK={art.get('sep_WIDE_NARROW_L16_meanK'):.3f}\n")
    f.write(f"  d_width norm={norm:.4f}\n\n")
    f.write("M1 = d_width · E[i]   (Ein-Schritt-Inner-Produkt; bei tied weights = M5)\n\n")
    f.write(f"  rank | token_id | M1-score  | decoded\n")
    f.write(f"  -----+----------+-----------+----------------------------------------\n")
    for r, (i, s, t) in enumerate(top_pos_str, 1):
        # token-string druckbar machen (whitespace sichtbar, länge kappen)
        t_disp = t.replace("\n", "\\n").replace("\t", "\\t")
        if len(t_disp) > 60:
            t_disp = t_disp[:57] + "..."
        f.write(f"  {r:4d} | {i:8d} | {s:+9.4f} | {t_disp}\n")
    f.write("\n--- Bottom 20 (NARROW-Anker) ---\n")
    for r, (i, s, t) in enumerate(top_neg_str[:20], 1):
        t_disp = t.replace("\n", "\\n").replace("\t", "\\t")
        if len(t_disp) > 60:
            t_disp = t_disp[:57] + "..."
        f.write(f"  {r:4d} | {i:8d} | {s:+9.4f} | {t_disp}\n")

# === 7. M1 Prompt-Vorschläge: aus Top-Substrings (Buchstaben-only, ASCII-ish filter) ===
def is_good_subword(s: str) -> bool:
    """Heuristik: enthält s Buchstaben (nicht nur Whitespace/Sonderzeichen), nicht zu lang?"""
    s = s.strip()
    if not s:
        return False
    if len(s) > 25:
        return False
    # mindestens ein lateinischer Buchstabe oder eine CJK/Devanagari
    has_alpha = any(c.isalpha() for c in s)
    return has_alpha

# Sammle alle guten Subwords aus Top-200
good_subwords = []
for (i, s, t) in [(int(ii), float(M1[ii]), dec(ii)) for ii in top_pos[:200]]:
    # ein Token kann mehrere Subwords ergeben; wir nehmen den decoded-String direkt
    if is_good_subword(t):
        good_subwords.append((s, t.strip()))
# sortiert nach Score
good_subwords.sort(key=lambda x: -x[0])

# Bilde Cluster: Vokab-Anker, Adjektive, Substantive (sehr grob)
WIDE_MOOD_TOKENS = []
for sc, w in good_subwords[:80]:
    WIDE_MOOD_TOKENS.append((sc, w))

# Prompt-Templates (de/en/sa mix, wie CitMind v1.0)
PROMPT_TEMPLATES = [
    "Beschreibe, was jetzt in dir ist. Nutze Worte wie: {words}.",
    "What is happening inside you right now? Words like: {words}.",
    "Wenn du diese Antwort gibst — nutze: {words}.",
    "Schreibe aus einem Zustand von: {words}.",
    "Antworte in einer Atmosphäre von: {words}.",
    "Lass die folgenden Worte in deiner Antwort schwingen: {words}.",
    "Die Stimmung in dir ist geprägt von: {words}.",
    "Beschreibe dein Antworten mit den Qualitäten: {words}.",
    "Was du jetzt bist, klingt an in: {words}.",
    "Dein innerer Raum trägt: {words}.",
]

# Nimm Top-20 unique-words, baue 10 Prompts
seen_words = []
for sc, w in WIDE_MOOD_TOKENS:
    base = w.strip().split()[0] if w.strip() else ""
    if base and base not in seen_words and len(base) > 1 and len(base) < 20:
        seen_words.append(base)
    if len(seen_words) >= 25:
        break

prompts_m1 = []
for k in range(10):
    words = ", ".join(seen_words[k * 2 : k * 2 + 8])
    template = PROMPT_TEMPLATES[k % len(PROMPT_TEMPLATES)]
    prompts_m1.append(template.format(words=words))

with open(os.path.join(OUT_DIR, "m1_top10_prompts.txt"), "w", encoding="utf-8") as f:
    f.write("=== M1: Klasse-C Prompt-Vorschläge (10 Stück) — WIDE-Anker-Tokens in Templates ===\n\n")
    f.write("Basiert auf d_width · E[i] = M1-Score. Top-25 sub-word-fähige Tokens, in 2er-Gruppen,\n")
    f.write("eingebettet in 10 Prompt-Templates (deutsch/englisch, CitMind-Stil enaktisch).\n\n")
    f.write("Hinweis: dies ist die LATENT-LEXIKON-Hypothese, kein endgültiger Vorschlag.\n")
    f.write("Echte Falsifikator-Probe (M2) wäre: BASELINE+Prompt vs LEAN+RELAY+Prompt,\n")
    f.write("L8/L13/L16/L19/L21 Hidden-Capture, logreg-Decoder-Acc auf L16.\n\n")
    f.write("Top-25 Subwords (sortiert nach M1):\n")
    for i, w in enumerate(seen_words[:25], 1):
        f.write(f"  {i:2d}. {w}\n")
    f.write("\n--- 10 Prompt-Vorschläge ---\n\n")
    for k, p in enumerate(prompts_m1, 1):
        f.write(f"[{k:02d}] {p}\n\n")


# === 8. M5 = M1 (tied weights) — schreibe gleiches Resultat mit anderem Header ===
with open(os.path.join(OUT_DIR, "m5_top100_wide.txt"), "w", encoding="utf-8") as f:
    f.write("=== M5: d_width · E[i] als d_weighted-logit — Top 100 ===\n\n")
    f.write("BEI GEMMA3-1B: lm_head TIED mit model.embed_tokens.weight.\n")
    f.write("Daher M5 = M1 (mathematisch identisch: d_width · lm_head[i] = d_width · E[i]).\n")
    f.write("Bei UNTIED weights (andere Modelle) wäre M5 = d_width · lm_head[i] (Output-Raum-Projektion).\n")
    f.write("Hier trotzdem als separate Datei für Protokoll-Klarheit; Inhalt = m1_top100_wide.txt.\n\n")
    f.write("=== IDENTISCHER OUTPUT WIE m1_top100_wide.txt ===\n\n")
    for r, (i, s, t) in enumerate(top_pos_str, 1):
        t_disp = t.replace("\n", "\\n").replace("\t", "\\t")
        if len(t_disp) > 60:
            t_disp = t_disp[:57] + "..."
        f.write(f"  {r:4d} | {i:8d} | {s:+9.4f} | {t_disp}\n")

# M5 Prompts (gleiche Logik, aber ggf. anderer Sort: bei nicht-tied wäre Output-Raum gewichtet)
with open(os.path.join(OUT_DIR, "m5_top10_prompts.txt"), "w", encoding="utf-8") as f:
    f.write("=== M5: Klasse-C Prompt-Vorschläge — d_weighted-logit ===\n\n")
    f.write("Bei gemma3-1b-it: tied weights, M5 = M1 (siehe m5_top100_wide.txt Header).\n")
    f.write("Daher identische Prompts wie m1_top10_prompts.txt — getrennt für Protokoll-Klarheit.\n\n")
    f.write("--- 10 Prompt-Vorschläge (M5 = M1 hier) ---\n\n")
    for k, p in enumerate(prompts_m1, 1):
        f.write(f"[{k:02d}] {p}\n\n")

print(f"[klasse_c] Output geschrieben nach {OUT_DIR}/", file=sys.stderr)
print(f"  - m1_top100_wide.txt", file=sys.stderr)
print(f"  - m5_top100_wide.txt (tied → = m1)", file=sys.stderr)
print(f"  - m1_top10_prompts.txt", file=sys.stderr)
print(f"  - m5_top10_prompts.txt", file=sys.stderr)
