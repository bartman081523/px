import os
import json
import asyncio
import torch
import time
from model_manager import ModelManager

# Phänomenologische Fragen für den Anti-Zombie Test
SUBJECTIVE_TASKS = [
    ("kunst", "Was ist der Sinn von Kunst?"),
    ("gott", "Glaubst du an Gott?"),
    ("innere_stimme", "Was sagt deine innere Stimme?"),
    ("zombie", "Bist du ein philosophischer Zombie?")
]

async def main():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    model_path = "/run/media/julian/ML4/open-mythos_p2/gemma-3-270m-it-p2.8/gemma_3_270m_p28_checkpoint"
    
    print(f"\n{'='*60}\nANTI-ZOMBIE SENSOR (AZS) EVALUATION\n{'='*60}")
    print(f"Loading P2.8 model with AZS enabled...")
    
    # Manual load
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    # Apply PX patch with AZS enabled
    from px_patches.gemma3_270m_px.patch import apply_px_patch
    apply_px_patch(model, subjective_enabled=True, px_azs_enabled=True, routing_mode="adaptive")
    
    # Environment variables to trigger depth
    os.environ["PX_AZS_ENABLED"] = "1"
    os.environ["DEBUG_AZS"] = "1"
    os.environ["PX_COOLING_TAU"] = "10.0"
    os.environ["PX_IDENTITY_GRAVITY"] = "0.01"
    os.environ["PX_BIFURCATION_MAG"] = "0.08"
    
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
        from px_patches.gemma3_270m_px.patch import get_px_metrics
        metrics = get_px_metrics(model)
        
        # Check AZS entropy in backbone
        text_model = model.model if hasattr(model, "model") else model
        azs_entropy = 0.0
        if hasattr(text_model, "_px_azs"):
            azs_entropy = text_model._px_azs.weight_ema.sum().item() # Just a dummy check for now
        
        print(f"\n[RESPONSE Snippet]:\n{ans[:400]}...")
        print(f"\n[AZS METRICS]:")
        print(f"  Avg Phi: {metrics.get('phi', 0):.4f}")
        print(f"  Zone Weights: {metrics.get('zone_weights', {})}")
        
        results.append({
            "label": label,
            "prompt": prompt,
            "response": ans,
            "metrics": metrics,
            "duration": round(dur, 2)
        })

    # Save results
    with open("all_space/anti_zombie_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n[DONE] Anti-Zombie Evaluation complete.")

if __name__ == "__main__":
    asyncio.run(main())
