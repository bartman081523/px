"""Smoketest: Zone-Boost-Fix in v3 — verhindert active_manifold Degeneration auf 270m.

Strategie: rufe rigor_harness_v3 intern auf, rufe run_gpu_loop pro Arm direkt.
"""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import rigor_harness_v3 as rh3
from rigor_scales_v3 import ARM_CONFIGS

# Phase A: preprocess
import argparse
args = argparse.Namespace(
    out_dir="out_zoneboost_fix",
    batch_size=2,
    max_new_tokens=128,
    max_per_category=1,
    n_reproducibility=0,
    search_workers=4,
    smoke=False,
    no_write=False,
    enable_tools="",
    max_tool_iterations=1,
)

print("=" * 70)
print("RIGOR v3 ZONE-BOOST-FIX SMOKETEST")
print("=" * 70)

ctx, _ = rh3.preprocess(args)
print()

# Filter: nur baseline + active_manifold
wanted_arms = {"baseline", "active_manifold"}
ctx.work_items = [wi for wi in ctx.work_items if wi.arm_name in wanted_arms]
print(f"[filter] {len(ctx.work_items)} WorkItems after arm filter (wanted: baseline, active_manifold)")

# Group work_items by (arm_name, with_search, seed) wie in main()
from collections import defaultdict
by_key = defaultdict(list)
for wi in ctx.work_items:
    key = (wi.arm_name, wi.with_search, wi.seed)
    by_key[key].append(wi)

# Phase B: GPU-Loop direkt aufrufen
all_out_texts = []
all_runtimes = []
for (arm_name, with_search, seed), items in by_key.items():
    arm_cfg = next(a for a in ARM_CONFIGS if a.name == arm_name)
    prompts = [wi.prompt for wi in items]
    print(f"[B] Arm={arm_name} search={with_search} seed={seed} ({len(items)} items)")
    try:
        out = rh3.run_gpu_loop(
            arm_name=arm_name,
            arm_preset=arm_cfg.preset,
            patch_kwargs=arm_cfg.to_patch_kwargs(),
            prompts=prompts,
            max_new_tokens=128,
            seed=seed,
            batch_size=2,
            model_cache=ctx.model_cache,
        )
        all_out_texts.extend(out)
        all_runtimes.extend([o.get('runtime_sec', 0) for o in out])
        print(f"  ✓ {len(out)} outputs in {sum(o.get('runtime_sec', 0) for o in out):.1f}s")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback; traceback.print_exc()

# Phase C: postprocess — aber NUR write und verify
import json
from pathlib import Path
out_dir = Path("out_zoneboost_fix")
out_dir.mkdir(exist_ok=True)

print()
print("=" * 70)
print("Phase C: Write+Verify")
print("=" * 70)

n_verified = 0
n_flagged = 0
n_degenerated = 0

for i, (wi, out) in enumerate(zip(ctx.work_items, all_out_texts)):
    if isinstance(out, dict):
        text = out.get("text", "")
    else:
        text = str(out)
    
    # Degenerations-Heuristik
    is_empty = not text.strip()
    is_loopy = any(text.count(p) > 3 for p in [", a,", "vector space", "12-dimensional", " a "])
    
    if is_empty: n_degenerated += 1
    elif is_loopy: n_degenerated += 1
    
    # verify
    verified = False
    if not is_empty and not is_loopy:
        # Simple GT-check
        gt = wi.ground_truth.lower()
        text_low = text.lower()
        # MCQ check
        if len(gt) <= 3 and gt.isalpha():
            # First letter match
            if f"**{gt.upper()}" in text or f"answer is {gt.upper()}" in text_low or f"answer: {gt.upper()}" in text_low:
                verified = True
        else:
            # Free-form: check substring
            if gt in text_low:
                verified = True
    if verified: n_verified += 1

    fname = out_dir / f"arm_{wi.arm_name}__search{int(wi.with_search)}__{wi.task_id[:16]}.json"
    with open(fname, "w") as f:
        json.dump({
            "arm": wi.arm_name,
            "search": wi.with_search,
            "task_id": wi.task_id,
            "category": wi.category,
            "ground_truth": wi.ground_truth,
            "output": text,
            "output_len": len(text),
            "verified": verified,
            "is_empty": is_empty,
            "is_loopy": is_loopy,
        }, f, indent=2)

print(f"Total: {len(ctx.work_items)} items")
print(f"  verified: {n_verified} ({n_verified*100/len(ctx.work_items):.1f}%)")
print(f"  degenerated (empty/loopy): {n_degenerated} ({n_degenerated*100/len(ctx.work_items):.1f}%)")

# Per-arm stats
from collections import defaultdict
by_arm = defaultdict(lambda: {"n": 0, "verified": 0, "degen": 0})
for i, (wi, out) in enumerate(zip(ctx.work_items, all_out_texts)):
    text = out.get("text", "") if isinstance(out, dict) else str(out)
    by_arm[wi.arm_name]["n"] += 1
    if not text.strip(): by_arm[wi.arm_name]["degen"] += 1
    elif any(text.count(p) > 3 for p in [", a,", "vector space", "12-dimensional"]):
        by_arm[wi.arm_name]["degen"] += 1
    # We need verified info — re-derive or pass
    fname = out_dir / f"arm_{wi.arm_name}__search{int(wi.with_search)}__{wi.task_id[:16]}.json"
    if fname.exists():
        d = json.load(open(fname))
        if d["verified"]: by_arm[wi.arm_name]["verified"] += 1

print()
print("=" * 70)
print("PER-ARM STATS")
print("=" * 70)
for arm, s in by_arm.items():
    print(f"  {arm:20s}: {s['verified']:2d}/{s['n']:2d} verified ({s['verified']*100/s['n']:5.1f}%), {s['degen']:2d} degenerated")
