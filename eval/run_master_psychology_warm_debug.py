import sys
import os
import builtins
sys.path.insert(0, ".")

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

from eval.run_master_psychology_warm import run_warm_eval
from eval.master_psychology_prompts import get_master_prompt_collection

# We monkey-patch get_master_prompt_collection to skip to 45
original_get = get_master_prompt_collection
def mock_get():
    return original_get()[45:]
import eval.run_master_psychology_warm
eval.run_master_psychology_warm.get_master_prompt_collection = mock_get

run_warm_eval("270M", "ACTIVE_MANIFOLD", "eval/results/SR64_FINAL_DEBUG")
