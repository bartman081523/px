import os
import hashlib
import re

def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

def extract_px_forward(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        content = f.read()
    match = re.search(r'def _px_forward\(.*?\):.*?(?=\n\w|\Z)', content, re.DOTALL)
    if match:
        return match.group(0)
    return None

def save_variant(file_path, label):
    code = extract_px_forward(file_path)
    if not code:
        print(f"[-] Could not find _px_forward in {file_path}")
        return
    md5 = get_md5(code)
    target_name = f"patch_rigor_{label}_{md5}.py"
    target_path = os.path.join("px_patches/rigor_modules", target_name)
    os.makedirs("px_patches/rigor_modules", exist_ok=True)
    
    # We copy the whole file for runtime stability, but rename it
    with open(file_path, "r") as f:
        full_content = f.read()
    with open(target_path, "w") as f:
        f.write(full_content)
    print(f"[+] Saved {label} to {target_path} (MD5: {md5})")

# Extract peak variants
save_variant("/run/media/julian/ML4/open-mythos_p2/gemma_3_270m_px_peak/patch.py", "peak_rigor")
save_variant("/run/media/julian/ML4/ollama-work/dmt_space_upload/patch.py", "peak_subjective")

# Extract historical variants
hist_dir_1 = "/run/media/julian/ML3/gemini-cli/gemini-cli-export-debug/ollama-work/2026-06-04_16-57-49_7232512c/patches"
hist_dir_2 = "/run/media/julian/ML3/gemini-cli/gemini-cli-export-debug/open-mythos-p2/2026-05-25_03-55-51_8d180fc0/patches"

# Select promising ones based on previous analysis
hist_files = [
    (os.path.join(hist_dir_1, "0645_patch.py"), "hist_0645"),
    (os.path.join(hist_dir_2, "0950_patch.py"), "hist_0950"),
    (os.path.join(hist_dir_2, "0990_patch.py"), "hist_0990"),
]

for fp, lbl in hist_files:
    save_variant(fp, lbl)
