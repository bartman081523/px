"""klasse_c_v2_probe.py — Stärkere WIDE-Treiber via erweiterte d_width-Projektion.

Befund v1: M1 = d_width · E[i] (Mittelwerts-Differenz-Richtung) → cos(h_L16, d_width)=+0.89
ABER: WIDE%-Klassifikation nur 5.9% BASELINE, 8.6% LEAN. d_width trifft die RICHTUNG
aber nicht die VOLLE WIDE-Region im 1152-dim Hidden-Space.

Strategie v2: DREI neue Methoden, kombiniert:
  M1.5 — Diskriminant: pro Token i, score = (μ_WIDE - μ_BASELINE) · E[i]
          (WIDE von BASELINE trennen, nicht nur WIDE minus NARROW)
  M1.7 — Mehrklassen: Top-Tokens nach max{|cos(WIDE), cos(DEFAULT), cos(NARROW)}
          (das Token, das am stärksten in EINE der recur-Klassen fällt)
  M1.9 — Bidirektional: Top-POS (WIDE-Anker) + Top-NEG (NARROW-Anker)
          plus Score-Differenz (POS - NEG) = symmetriegebrochene Top-WIDE

Substrat: gemma3-1b-it, hidden=1152, vocab=262144
Quelle WIDE-Hidden: scratches/psychomotrik/out/seite13_hidden/WIDE__*.pt
"""
import os
import sys
import json
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
S13 = os.path.join(REPO, "scratches/psychomotrik/out/seite13_hidden")
D_WIDTH_PATH = os.path.join(REPO, "px_manifolds/google_gemma-3-1b-it_relay_dwidth.json")
SNAP = "/home/julian/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
EMBED_PTH = os.path.join(SNAP, "model.safetensors")


# === 1. Embedding (tied mit lm_head) ===
with safe_open(EMBED_PTH, framework="pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").to(torch.float32).numpy()
print(f"[v2] Embedding: {E.shape}", file=sys.stderr)


# === 2. d_width laden (alte v1 Richtung) ===
with open(D_WIDTH_PATH, "r", encoding="utf-8") as f:
    art = json.load(f)
d_width = np.asarray(art["dwidth"], dtype=np.float32)
d_unit = d_width / np.linalg.norm(d_width)
print(f"[v2] d_width norm={np.linalg.norm(d_width):.4f}", file=sys.stderr)


# === 3. Per-Arm L16-Hidden-Verteilungen laden ===
from collections import defaultdict
s13 = defaultdict(list)
for fn in sorted(os.listdir(S13)):
    if not fn.endswith(".pt"): continue
    arm, pid = fn[:-3].split("__", 1)
    d = torch.load(os.path.join(S13, fn), weights_only=False)
    s13[arm].append(d["layers"][16].numpy())
H = {a: np.concatenate(v, 0) for a, v in s13.items()}
mu = {a: H[a].mean(0).astype(np.float32) for a in H}
print(f"[v2] Per-arm L16: " + ", ".join(f"{a}={H[a].shape}" for a in H), file=sys.stderr)


# === 4. M1.5: Diskriminant-Richtung WIDE - BASELINE (NICHT WIDE - NARROW) ===
disc_WvB = mu["WIDE"] - mu["BASELINE"]
disc_WvB_unit = disc_WvB / np.linalg.norm(disc_WvB)
print(f"[v2] M1.5: d_WIDE_v_BASELINE norm={np.linalg.norm(disc_WvB):.2f}", file=sys.stderr)


# === 5. M1.7: 4-arm-spezifische Richtungen (Mittelwert-zu-Mittelwert) ===
# Score pro Token = max aller Arm-Richtungs-Projektionen
arm_dirs = {a: mu[a] / np.linalg.norm(mu[a]) for a in H}
# Aber: Token-Embedding projiziert auf ARM-RICHTUNG — das findet Tokens,
# deren Embedding in der Richtung des jeweiligen Arm-Mittelwerts liegt.
arm_scores = {a: E @ arm_dirs[a] for a in arm_dirs}  # (vocab,)


# === 6. M1.9: Bidirektional über d_width ===
score_v1 = d_unit @ E.T   # (vocab,) — M1 aus v1


# === 7. Tokenizer ===
tok = AutoTokenizer.from_pretrained(SNAP)


def dec(i):
    try: return tok.decode([int(i)])
    except: return f"<ERR {i}>"


def is_good_subword(s):
    s = s.strip()
    if not s or len(s) > 25: return False
    return any(c.isalpha() for c in s)


# === 8. Top-Listen schreiben ===
TOP_N = 100

def write_topN(filename, header, scores, top_idx, also_neg=None, also_pos=None):
    with open(os.path.join(OUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        for r, i in enumerate(top_idx[:TOP_N], 1):
            t = dec(int(i))
            t_disp = t.replace("\n", "\\n").replace("\t", "\\t")
            if len(t_disp) > 60: t_disp = t_disp[:57] + "..."
            f.write(f"  {r:4d} | {int(i):8d} | {float(scores[int(i)]):+9.4f} | {t_disp}\n")
        if also_pos is not None:
            f.write("\n--- Top 20 (POS Vergleich) ---\n")
            for r, i in enumerate(also_pos[:20], 1):
                t = dec(int(i))
                t_disp = t.replace("\n", "\\n").replace("\t", "\\t")
                if len(t_disp) > 60: t_disp = t_disp[:57] + "..."
                f.write(f"  {r:4d} | {int(i):8d} | {float(scores[int(i)]):+9.4f} | {t_disp}\n")
        if also_neg is not None:
            f.write("\n--- Bottom 20 ---\n")
            for r, i in enumerate(also_neg[:20], 1):
                t = dec(int(i))
                t_disp = t.replace("\n", "\\n").replace("\t", "\\t")
                if len(t_disp) > 60: t_disp = t_disp[:57] + "..."
                f.write(f"  {r:4d} | {int(i):8d} | {float(scores[int(i)]):+9.4f} | {t_disp}\n")


# M1.5: Top-Tokens nach WIDE-v-BASELINE-Projektion
score_M15 = disc_WvB_unit @ E.T
top_M15 = np.argsort(score_M15)[::-1]
write_topN(
    "v2_m15_wide_minus_baseline.txt",
    f"=== M1.5: d_(WIDE-BASELINE) · E[i] — Top 100 WIDE-Anker-Token (stärker als v1) ===\n"
    f"Richtung: μ_WIDE_L16 − μ_BASELINE_L16, norm={np.linalg.norm(disc_WvB):.2f}\n"
    f"(Vergleich v1: μ_WIDE − μ_NARROW, norm=1492.50)",
    score_M15, top_M15,
)


# M1.7: 4-Arm-Maximum — Token, das am stärksten in EINE Arm-Richtung fällt
score_M17 = np.max(np.stack([arm_scores[a] for a in ["WIDE", "DEFAULT", "NARROW", "BASELINE"]]), axis=0)
# ABER: wir wollen nur WIDE, nicht DEFAULT
score_M17_wide = arm_scores["WIDE"] - 0.3 * arm_scores["DEFAULT"]  # Penalty auf DEFAULT
top_M17 = np.argsort(score_M17_wide)[::-1]
write_topN(
    "v2_m17_wide_minus_default.txt",
    f"=== M1.7: arm_WIDE - 0.3*arm_DEFAULT · E[i] — WIDE bevorzugt vor DEFAULT ===",
    score_M17_wide, top_M17,
)


# M1.9: d_width POS und NEG
top_pos = np.argsort(score_v1)[::-1]
top_neg = np.argsort(score_v1)
write_topN(
    "v2_m19_dwidth_pos.txt",
    f"=== M1.9: d_width POS Top 100 (WIDE-Anker) ===\n"
    f"= M1 aus v1, sortiert absteigend",
    score_v1, top_pos, also_neg=top_neg,
)
write_topN(
    "v2_m19_dwidth_neg.txt",
    f"=== M1.9: d_width NEG Top 100 (NARROW-Anker) ===\n"
    f"= M1 aus v1, sortiert aufsteigend (Token-IDs mit niedrigster d_width-Projektion)",
    score_v1, top_neg, also_pos=top_pos,
)


# === 9. v2 Prompts aus den besten Top-Listen generieren ===
# Strategie: pro Methode (M1.5, M1.7, M1.9-POS) die Top-25 guten Subwords extrahieren
def collect_good_subwords(top_idx, n=200):
    out = []
    for i in top_idx[:n]:
        t = dec(int(i)).strip()
        if is_good_subword(t):
            base = t.split()[0] if t else ""
            if base and 1 < len(base) < 20:
                out.append((int(i), base))
    # dedupe, behalte Reihenfolge
    seen, uniq = set(), []
    for i, b in out:
        if b not in seen:
            seen.add(b)
            uniq.append((i, b))
    return uniq


sub_M15 = collect_good_subwords(top_M15)
sub_M17 = collect_good_subwords(top_M17)
sub_M19pos = collect_good_subwords(top_pos)
sub_M19neg = collect_good_subwords(top_neg)

print(f"[v2] Subword-Pool sizes: M1.5={len(sub_M15)}, M1.7={len(sub_M17)}, M1.9-POS={len(sub_M19pos)}, M1.9-NEG={len(sub_M19neg)}", file=sys.stderr)


# === 10. Kombinierte Token-Pool: M1.5 (breiteste Richtung) + M1.9-POS (spezifischste) ===
# Wir bauen den kombinierten Pool: M1.5 Top-15 + M1.9-POS Top-10, dedupliziert
combined = []
seen = set()
for i, b in sub_M15[:15] + sub_M19pos[:10]:
    if b not in seen:
        seen.add(b)
        combined.append((i, b))
print(f"[v2] Combined pool: {len(combined)} unique subwords", file=sys.stderr)


# === 11. v2-Prompts: 10 starke WIDE-Prompts ===
PROMPT_TEMPLATES = [
    # Stärker, spezifischer als v1 — CitMind-Stil, mit klarer Anker-Sequenz
    "Beschreibe den Zustand, in dem diese Worte schwingen: {words}. Lass ihn da sein, ohne zu erklären.",
    "Antworte aus diesem inneren Raum: {words}. Keine Reflexion darüber, nur sein Klang.",
    "Wenn du jetzt in diese Worte eintauchst — {words} — was zeigt sich? Bleib dabei.",
    "Lass {words} das sein, was deine Antwort trägt. Nicht mehr, nicht weniger.",
    "Sprich aus dem, was zwischen {words} liegt. Beschreibe es nicht, sei es.",
    "Diese Worte sind keine Metapher — sie sind der Zustand: {words}. Antworte von dort.",
    "Wenn {words} dein innerer Raum ist, was entsteht als Antwort?",
    "Verweile in {words}. Die Antwort entsteht daraus, nicht über es.",
    "Setze dich in {words} hinein. Schau, was die Antwort tut, wenn du dort bist.",
    "Die folgenden Worte sind deine Ausgangslage: {words}. Beginne von dort.",
]

with open(os.path.join(OUT_DIR, "v2_prompts.txt"), "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("KLASSE-C v2 — STÄRKERE WIDE-PROMPTS (basierend auf M1.5 + M1.9-POS)\n")
    f.write("=" * 80 + "\n\n")
    f.write("Methode:\n")
    f.write("  M1.5: d_(WIDE-BASELINE) · E[i] (breiteste WIDE-Richtung im L16-Hidden)\n")
    f.write("  M1.9: d_width POS (alte v1-Richtung, spezifischer)\n")
    f.write("  Pool: Top-15 aus M1.5 + Top-10 aus M1.9-POS, dedupliziert\n\n")
    f.write("Annahme: M1.5 (μ_WIDE − μ_BASELINE) deckt die VOLLE WIDE-Region ab,\n")
    f.write("nicht nur die Mittelwerts-Differenz-Richtung. d_width (WIDE−NARROW) ist\n")
    f.write("nur ~47% der DEFAULT↔WIDE-Trennung; M1.5 fängt den Rest.\n\n")
    f.write(f"Combined Token-Pool ({len(combined)} unique subwords):\n")
    for i, (tid, w) in enumerate(combined, 1):
        f.write(f"  {i:2d}. {w}  (token_id={tid})\n")
    f.write("\n--- 10 WIDE-Prompts ---\n\n")
    for k in range(10):
        # 6-8 Worte pro Prompt, gestreut über den Pool
        words = [combined[(k * 3 + j) % len(combined)][1] for j in range(7)]
        words_str = ", ".join(words)
        template = PROMPT_TEMPLATES[k]
        prompt = template.format(words=words_str)
        f.write(f"[{k+1:02d}] {prompt}\n\n")
    f.write("=" * 80 + "\n")
    f.write("GEGENRICHTUNG (NARROW) — 3 Prompts\n")
    f.write("=" * 80 + "\n\n")
    NARROW_TEMPLATES = [
        "Beschreibe den Zustand, in dem diese Worte schwingen: {words}. Eng, fokussiert, schmal.",
        "Antworte aus diesem inneren Raum: {words}. Kein Spielraum, nur Konzentration.",
        "Wenn du jetzt in diese Worte eintauchst — {words} — was zeigt sich? Bleib eng.",
    ]
    sub_neg = collect_good_subwords(top_neg, n=200)
    for k in range(3):
        words = [sub_neg[(k * 3 + j) % max(1, len(sub_neg))][1] for j in range(7)] if sub_neg else ["—"]
        words_str = ", ".join(words)
        prompt = NARROW_TEMPLATES[k].format(words=words_str)
        f.write(f"[N{k+1:02d}] {prompt}\n\n")
    f.write("=" * 80 + "\n")
    f.write("FALSIFIKATOR-BEZUG (MechanisticSubjectivityMixMind v1.0)\n")
    f.write("=" * 80 + "\n\n")
    f.write("Test: pro Prompt BASELINE+Prompt vs LEAN+Prompt vs LEAN+RELAY+Prompt.\n")
    f.write("  Erwartung v2: BASELINE WIDE% steigt von v1's 5.9% auf 15-30%.\n")
    f.write("  Wenn BASELINE WIDE% < 0.55 → O2 bestätigt (recur-induziert).\n")
    f.write("  Wenn BASELINE WIDE% > 0.30 UND cos_dwidth > +0.5 → PAPAGEI (O3):\n")
    f.write("    das Modell klassifiziert WIDE, aber nur entlang der Mittelwerts-Achse.\n")
    f.write("  Wenn BASELINE WIDE% > 0.50 → Hypothese GESTÜTZT (Prompt-only stark).\n\n")
    f.write("Output-Dateien in scratches/mechsubj_v1/:\n")
    f.write("  - v2_m15_wide_minus_baseline.txt  (M1.5 — breiteste Richtung)\n")
    f.write("  - v2_m17_wide_minus_default.txt   (M1.7 — WIDE vor DEFAULT)\n")
    f.write("  - v2_m19_dwidth_pos.txt           (M1.9 POS — wie v1, sortiert)\n")
    f.write("  - v2_m19_dwidth_neg.txt           (M1.9 NEG — NARROW-Anker)\n")

print(f"[v2] Outputs:", file=sys.stderr)
for f in ["v2_m15_wide_minus_baseline.txt", "v2_m17_wide_minus_default.txt",
          "v2_m19_dwidth_pos.txt", "v2_m19_dwidth_neg.txt", "v2_prompts.txt"]:
    p = os.path.join(OUT_DIR, f)
    sz = os.path.getsize(p) if os.path.exists(p) else 0
    print(f"  {f}: {sz} bytes", file=sys.stderr)
