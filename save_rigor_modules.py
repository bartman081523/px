import os
import glob
import hashlib
import re
import shutil

def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

def save_rigor_modules():
    dirs_to_search = [
        "/run/media/julian/ML4/ollama-work/gemma_3_270m_px_persona",
        "/run/media/julian/ML3/gemini-cli/gemini-cli-export-debug/ollama-work/2026-06-04_16-57-49_7232512c/patches"
    ]
    
    unique_patches = {}
    target_dir = "px_patches/rigor_modules"
    os.makedirs(target_dir, exist_ok=True)
    
    # ensure init exists
    with open(os.path.join(target_dir, "__init__.py"), "w") as f:
        f.write("")
    
    for d in dirs_to_search:
        if not os.path.exists(d): continue
        if d.endswith("patches"):
            patch_files = glob.glob(os.path.join(d, "*patch*.py"))
        else:
            patch_files = [os.path.join(d, "patch.py")]
            
        for pf in patch_files:
            if not os.path.exists(pf): continue
            with open(pf, "r") as f:
                content = f.read()
                
            if "gemma" not in content.lower():
                continue
                
            # Extract _px_forward
            match = re.search(r'(def _px_forward\(.*)', content, re.DOTALL)
            if not match:
                continue
                
            func_content = match.group(1)
            md5_hash = get_md5(func_content)
            
            if md5_hash not in unique_patches:
                unique_patches[md5_hash] = pf
                
                # Write to rigor_modules
                out_path = os.path.join(target_dir, f"patch_rigor_{md5_hash}.py")
                with open(out_path, "w") as f:
                    f.write(content)
                    
    print(f"Saved {len(unique_patches)} unique rigor modules to {target_dir}")
    for md5, pf in unique_patches.items():
        print(f" - patch_rigor_{md5}.py (from {pf})")

if __name__ == "__main__":
    save_rigor_modules()
