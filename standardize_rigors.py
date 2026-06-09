import os
import re

def standardize_patch(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    # Fix relative imports
    # 2026-06-09: routed to isolated baseline (gemma3 must remain
    # byte-identical to pre-gemma4 implementation)
    content = re.sub(r'from \.px_modules import', 'from px_patches.gemma3_270m_px_baseline.px_modules import', content)
    content = re.sub(r'from \.auto_tune import', 'from px_patches.gemma3_270m_px_baseline.auto_tune import', content)
    content = re.sub(r'from \.persona_engine import', 'from px_patches.gemma3_270m_px_baseline.persona_engine import', content)

    # Some older patches might use from . import px_modules
    content = re.sub(r'from \. import px_modules', 'import px_patches.gemma3_270m_px_baseline.px_modules as px_modules', content)
    
    with open(file_path, "w") as f:
        f.write(content)

variants_dir = "px_patches/rigor_modules"
for vf in os.listdir(variants_dir):
    if vf.endswith(".py"):
        standardize_patch(os.path.join(variants_dir, vf))
        print(f"Standardized {vf}")
