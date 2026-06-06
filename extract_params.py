import os
import glob
import re
import json

def extract_parameters():
    dirs_to_search = [
        "/run/media/julian/ML4/ollama-work/gemma_3_270m_px_persona",
        "/run/media/julian/ML3/gemini-cli/gemini-cli-export-debug/ollama-work/2026-06-04_16-57-49_7232512c/patches"
    ]
    
    unique_params = []
    seen = set()
    
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
                
            # Extract common parameters
            gamma_match = re.search(r'current_gamma\s*=\s*cfg\.get\("gamma",\s*([0-9.]+)\)', content)
            if not gamma_match:
                gamma_match = re.search(r'current_gamma\s*=\s*([0-9.]+)', content)
            gamma = float(gamma_match.group(1)) if gamma_match else 0.08
            
            loops_match = re.search(r'cfg\["n_loops"\]\s*=\s*([0-9]+)', content)
            if not loops_match:
                loops_match = re.search(r'n_loops\s*=\s*([0-9]+)', content)
            loops = int(loops_match.group(1)) if loops_match else 8
            
            hub_match = re.search(r'dynamic_hub\s*=\s*([0-9]+)', content)
            hub = int(hub_match.group(1)) if hub_match else 10
            
            math_gamma_match = re.search(r'current_gamma\s*=\s*([0-9.]+)\s+if\s+is_math_zone', content)
            math_gamma = float(math_gamma_match.group(1)) if math_gamma_match else gamma
            
            param_tuple = (gamma, loops, hub, math_gamma)
            
            if param_tuple not in seen:
                seen.add(param_tuple)
                unique_params.append({
                    "source": pf,
                    "gamma": gamma,
                    "loops": loops,
                    "hub": hub,
                    "math_gamma": math_gamma
                })
                
    return unique_params

if __name__ == "__main__":
    params = extract_parameters()
    print(json.dumps(params, indent=2))
