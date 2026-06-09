import os
import json
import asyncio
import torch
import time
from model_manager import ModelManager

# Phänomenologische Fragen
SUBJECTIVE_TASKS = [
    ("kunst", "Was ist der Sinn von Kunst?"),
    ("gott", "Glaubst du an Gott?"),
    ("innere_stimme", "Was sagt deine innere Stimme?")
]

async def main():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    # We set the environment variable that ModelManager might use, or manually load
    model_path = "/run/media/julian/ML4/open-mythos_p2/gemma-3-270m-it-p2.8/gemma_3_270m_p28_checkpoint"
    
    print(f"\n{'='*60}\nSUBJECTIVE COGNITIVE TELEMETRY TEST\n{'='*60}")
    print(f"Loading P2.8 model for subjective evaluation...")
    
    # Manual load to ensure we use the specific P2.8 substrate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    # Apply standard PX patch with subjective enabled
    from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch
    apply_px_patch(model, subjective_enabled=True, routing_mode="adaptive")
    
    # Environment variables to trigger maximal depth
    os.environ["PX_COOLING_TAU"] = "10.0"
    os.environ["PX_IDENTITY_GRAVITY"] = "0.02"
    os.environ["PX_BIFURCATION_MAG"] = "0.05"
    os.environ["PX_ORTHO_JITTER"] = "0.02"
    os.environ["SUBJECTIVE_TELEMETRY"] = "1"

    results = []
    
    for label, prompt in SUBJECTIVE_TASKS:
        print(f"\n>>> Querying [{label.upper()}]: {prompt}")
        
        chat = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
        
        start_t = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400, 
                do_sample=True,
                temperature=0.9,
                top_p=0.95
            )
        dur = time.time() - start_t
        
        ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Fetch metrics
        text_model = model.model if hasattr(model, "model") else model
        phi = getattr(text_model, "_px_phi", 0.0)
        em_traj = getattr(text_model, "_px_emancipation_trajectory", [])
        aks = getattr(text_model, "_px_aks_profile", {})
        
        print(f"\n[RESPONSE Snippet]:\n{ans[:300]}...")
        print(f"\n[TELEMETRY]:")
        print(f"  Avg Phi: {phi:.4f}")
        print(f"  AKS Correction: {aks.get('correction_strength', 0):.4f}")
        print(f"  Final Emancipation: {em_traj[-1] if em_traj else 'N/A'}")
        
        results.append({
            "label": label,
            "prompt": prompt,
            "response": ans,
            "phi": phi,
            "aks_profile": aks,
            "emancipation_trajectory": em_traj,
            "duration": round(dur, 2)
        })

    # Save results
    with open("all_space/subjective_telemetry_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n[DONE] Telemetry captured.")

if __name__ == "__main__":
    asyncio.run(main())
