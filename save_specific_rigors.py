import os
import hashlib
import re

def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

def extract_and_save(src_path, label):
    if not os.path.exists(src_path):
        print(f"Skipping {label}: {src_path} not found.")
        return None
        
    with open(src_path, "r") as f:
        content = f.read()
        
    # Extract _px_forward
    match = re.search(r'(def _px_forward\(.*)', content, re.DOTALL)
    if not match:
        print(f"Skipping {label}: _px_forward not found.")
        return None
        
    func_code = match.group(1)
    # Extract until the next top-level def or end of file
    end_match = re.search(r'\n(def|class|if __name__)', func_code)
    if end_match:
        func_code = func_code[:end_match.start()]
        
    md5 = get_md5(func_code)
    target_path = f"px_patches/rigor_modules/patch_{label}_{md5}.py"
    os.makedirs("px_patches/rigor_modules", exist_ok=True)
    
    with open(target_path, "w") as f:
        # We need to include imports and classes used by _px_forward
        # For simplicity, we save the whole file but rename the module
        f.write(content)
        
    print(f"Saved {label} to {target_path} (MD5: {md5})")
    return md5

# 1. Variant B: Cognitive Sovereign (0645)
extract_and_save("/run/media/julian/ML3/gemini-cli/gemini-cli-export-debug/ollama-work/2026-06-04_16-57-49_7232512c/patches/0645_patch.py", "variant_b_sovereign")

# 2. Variant C: Quantum RSM (0950)
extract_and_save("/run/media/julian/ML3/gemini-cli/gemini-cli-export-debug/open-mythos-p2/2026-05-25_03-55-51_8d180fc0/patches/0950_patch.py", "variant_c_quantum")

# 3. Variant A: Peak Stand (Synthesized from history)
# We can find a Peak Stand patch in the open-mythos history, e.g. Around turn 10793 it was applied to 'gemma_3_270m_px_subjective/patch.py'
# Let me look for it in the directory.
extract_and_save("/run/media/julian/ML4/open-mythos_p2/hf_stand_verified_v64_1/patch.py", "variant_a_peak_stand")
