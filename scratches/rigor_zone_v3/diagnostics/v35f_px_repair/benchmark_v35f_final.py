"""Final v3.5f Benchmark: 32 HLE Tasks × 3 Presets — NACH dem phi-Gate-Fix."""
import sys, os, json
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch
from rigor_hle_suite_v3 import load_hle_suite

source, suite = load_hle_suite(max_per_category=4)
TASKS = [(tid, cat, prompt, gt) for tid, cat, prompt, gt in suite]
print("Loaded", len(TASKS), "HLE tasks")

MAX_NEW = 200

def load_model(preset):
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", dtype=torch.bfloat16).to("cuda").eval()
    if preset != "BASELINE":
        apply_px_patch(model.model, config_preset=preset)
    return tok, model

def generate(tok, model, prompt, max_new=MAX_NEW):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)

def analyze(text):
    if not text.strip(): return "EMPTY"
    words = text.split()
    for plen in [3, 4]:
        for i in range(len(words) - plen*3):
            phrase = " ".join(words[i:i+plen])
            n_repeats = sum(1 for j in range(i+plen, len(words)-plen+1, plen) if " ".join(words[j:j+plen]) == phrase)
            if n_repeats >= 3: return "LOOPY"
    return "OK"

def check_match(text, gt):
    text_l = text.lower()
    gt_l = gt.lower().strip()
    if not gt_l: return False
    if len(gt_l) <= 3 and gt_l.replace(" ", "").isalpha():
        if f"**{gt_l.upper()}" in text or f"answer is {gt_l.upper()}" in text_l or f"answer: {gt_l.upper()}" in text_l:
            return True
        if f"\n{gt_l.upper()}. " in text or f"{gt_l.upper()}. " in text[:50]:
            return True
    else:
        if gt_l in text_l:
            return True
    return False

all_outputs = {}
for preset, label in [("BASELINE", "BASELINE"), ("ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD"), ("ACTIVE_MANIFOLD_LEAN", "LEAN")]:
    print("\n===", label, "===", flush=True)
    all_outputs[label] = {}
    tok, model = load_model(preset)
    for tid, cat, prompt, gt in TASKS:
        text = generate(tok, model, prompt)
        all_outputs[label][tid] = {
            "text": text, "cat": cat, "gt": gt,
            "status": analyze(text),
            "match": check_match(text, gt),
        }
    del model; torch.cuda.empty_cache()

with open("/tmp/benchmark_v35f_final.json", "w") as f:
    json.dump(all_outputs, f, indent=2)

print()
print("=" * 78)
print("  HLE v3.5f FINAL — 32 Tasks x 3 Presets (max_new=200, greedy)")
print("=" * 78)

stats = {}
for label in ['BASELINE', 'ACTIVE_MANIFOLD', 'LEAN']:
    r = all_outputs[label]
    ok = sum(1 for v in r.values() if v['status'] == 'OK')
    degen = sum(1 for v in r.values() if v['status'] != 'OK')
    matches = sum(1 for v in r.values() if v['match'])
    stats[label] = {'ok': ok, 'degen': degen, 'matches': matches}

print()
print(f"  {'':35s}{'BASELINE':>10s}{'AM':>10s}{'LEAN':>10s}")
print("  " + "-" * 65)
print(f"  {'OK (kein Loopy/Empty)':35s}{stats['BASELINE']['ok']:>10d}{stats['ACTIVE_MANIFOLD']['ok']:>10d}{stats['LEAN']['ok']:>10d}")
print(f"  {'Matches (verified)':35s}{stats['BASELINE']['matches']:>10d}{stats['ACTIVE_MANIFOLD']['matches']:>10d}{stats['LEAN']['matches']:>10d}")
print(f"  {'Degeneration (Loopy+Empty)':35s}{stats['BASELINE']['degen']:>10d}{stats['ACTIVE_MANIFOLD']['degen']:>10d}{stats['LEAN']['degen']:>10d}")

print()
print("=" * 78)
print("  Per-Task Match-Tabelle (v=match, x=kein Match, L=Loopy, E=Empty)")
print("=" * 78)
print()
header = "  " + "TID".ljust(12) + " " + "KAT".ljust(22) + " " + "GT".ljust(14) + "B  AM LN"
print(header)
print("  " + "-" * 58)

tids = list(all_outputs['BASELINE'].keys())
for tid in tids:
    b = all_outputs['BASELINE'][tid]
    am = all_outputs['ACTIVE_MANIFOLD'][tid]
    ln = all_outputs['LEAN'][tid]

    def sym(v):
        if v['status'] != 'OK': return v['status'][0]
        return 'v' if v['match'] else 'x'

    b_sym, am_sym, ln_sym = sym(b), sym(am), sym(ln)
    print(f"  {tid[:10]:12s} {b['cat'][:22]:22s} {b['gt'][:14]:14s}{b_sym:>3s}{am_sym:>3s}{ln_sym:>3s}")

all_three = [tid for tid in tids if all_outputs['BASELINE'][tid]['match'] and all_outputs['ACTIVE_MANIFOLD'][tid]['match'] and all_outputs['LEAN'][tid]['match']]
print()
print("  Tasks wo ALLE 3 Presets v:", len(all_three), "/", len(tids))
for tid in all_three:
    v = all_outputs['BASELINE'][tid]
    print("    -", tid[:10], v['cat'][:22], "GT=", repr(v['gt']))

print()
print("  REGRESSIONS (BASELINE OK, aber PX defekt):")
regressions = 0
for tid in tids:
    b = all_outputs['BASELINE'][tid]
    am = all_outputs['ACTIVE_MANIFOLD'][tid]
    ln = all_outputs['LEAN'][tid]
    if b['status'] == 'OK' and (am['status'] != 'OK' or ln['status'] != 'OK'):
        regressions += 1
        print("   ", tid[:10], b['cat'][:22], "| B-OK | AM=", am['status'], "| LN=", ln['status'])
if regressions == 0:
    print("    (keine — PX erhaelt alle BASELINE-OK Tasks)")

print()
print("  IMPROVEMENTS (BASELINE defekt, aber PX OK):")
improvements = 0
for tid in tids:
    b = all_outputs['BASELINE'][tid]
    am = all_outputs['ACTIVE_MANIFOLD'][tid]
    ln = all_outputs['LEAN'][tid]
    if b['status'] != 'OK' and (am['status'] == 'OK' or ln['status'] == 'OK'):
        improvements += 1
        print("   ", tid[:10], b['cat'][:22], "| B=", b['status'], "| AM=", am['status'], "| LN=", ln['status'])

print()
print("  REGRESSIONS:", regressions, "| IMPROVEMENTS:", improvements)

print()
print("=" * 78)
print("  KONKLUSION")
print("=" * 78)
am_pct = stats['ACTIVE_MANIFOLD']['matches']/32*100
ln_pct = stats['LEAN']['matches']/32*100
b_pct = stats['BASELINE']['matches']/32*100
print(f"  Match-Rate: BASELINE {b_pct:.1f}% | AM {am_pct:.1f}% | LEAN {ln_pct:.1f}%")
print(f"  OK-Rate:    BASELINE {stats['BASELINE']['ok']}/32 | AM {stats['ACTIVE_MANIFOLD']['ok']}/32 | LEAN {stats['LEAN']['ok']}/32")
print(f"  Degen:      BASELINE {stats['BASELINE']['degen']}/32 | AM {stats['ACTIVE_MANIFOLD']['degen']}/32 | LEAN {stats['LEAN']['degen']}/32")
