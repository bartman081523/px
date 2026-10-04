"""reclean_pools.py — Entfernt Mehrfach-Tokens und baut Suite neu.

Bidirektionale Tokens ('KATA', 'DARK', 'SILENT') werden aus dem WIDE-Pool
entfernt (Modell hat sie beiden Seiten zugeordnet — kein klarer WIDE-Anker).
In NARROW verbleiben sie, weil sie dort primär zugeordnet waren (besonders
ENG / KONZENTRIERT / DICHT decken sich mit seite15 NARROW_VOCAB).
"""
import json
import os

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
MODEL = "gemma-3-1b-it"

BIDIR = {"KATA", "DARK", "SILENT"}

with open(os.path.join(OUT_DIR, f"wide_narrow_tokens_{MODEL}.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

wide_clean = [t for t in data["wide_tokens"] if t["token"] not in BIDIR]
data["wide_tokens"] = wide_clean

# Mark bidir in narrow for transparency
for t in data["narrow_tokens"]:
    if t["token"] in BIDIR:
        t["note"] = "model assigned this token to BOTH WIDE and NARROW — kept in NARROW"

with open(os.path.join(OUT_DIR, f"wide_narrow_tokens_{MODEL}.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"cleaned WIDE pool: {len(wide_clean)} tokens (removed {len(BIDIR)} bidir)")
print(f"NARROW pool: {len(data['narrow_tokens'])} tokens (kept, marked where bidir)")

# Rebuild suite
import sys
sys.path.insert(0, OUT_DIR)
from token_flow_v1 import build_prompt_suite  # type: ignore

suite = build_prompt_suite(data["wide_tokens"], data["narrow_tokens"], data["few_shots"])
suite_path = os.path.join(OUT_DIR, f"wide_narrow_prompts_{MODEL}.txt")
with open(suite_path, "w", encoding="utf-8") as f:
    f.write(suite)
print(f"rebuilt: {suite_path} ({len(suite.splitlines())} lines, {len(suite)} bytes)")
