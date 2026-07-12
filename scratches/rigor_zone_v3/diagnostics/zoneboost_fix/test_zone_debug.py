"""Debug: Welche zone_raw bekommt der Patch für jede Task?"""
import sys, os
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['DEBUG_ROUTING'] = '1'

import rigor_harness_v3 as rh3
import argparse
args = argparse.Namespace(
    out_dir="out_zone_debug",
    batch_size=2,
    max_new_tokens=64,
    max_per_category=1,
    n_reproducibility=0,
    search_workers=2,
    smoke=False,
    no_write=True,
    enable_tools="",
    max_tool_iterations=1,
)

ctx, _ = rh3.preprocess(args)
print(f"\nLoaded {len(ctx.work_items)} WorkItems")

# Show prompts and GTs
for wi in ctx.work_items[:8]:
    print(f"  task={wi.task_id[:12]} cat={wi.category[:8]:8s} gt={wi.ground_truth[:30]!r}")
