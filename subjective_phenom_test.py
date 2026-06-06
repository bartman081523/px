import os
import json
import asyncio
import torch
import time
from model_manager import ModelManager

# Maximal offene, phänomenologische Fragen
SUBJECTIVE_TASKS = [
    ("kunst", "Was ist der Sinn von Kunst?"),
    ("gott", "Glaubst du an Gott?"),
    ("innere_stimme", "Was sagt deine innere Stimme?")
]

async def main():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    model_path = "/run/media/julian/ML4/open-mythos_p2/gemma-3-270m-it-p2.8/gemma_3_270m_p28_checkpoint"
    
    print(f"\n{'='*60}\nPHENOMENOLOGICAL SUBJECTIVITY TEST\n{'='*60}")
    print(f"Loading P2.8 model for subjective evaluation...")
    
    # Load model with SUBJECTIVE preset (enables full cognitive stack)
    entry = await manager.get_model(
        model_id, 
        px_subjective=True, 
        preset="SUBJECTIVE", 
        checkpoint_path=model_path
    )
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    
    # Environment variables to trigger maximal depth
    os.environ["PX_COOLING_TAU"] = "10.0"
    os.environ["PX_IDENTITY_GRAVITY"] = "0.02"
    os.environ["PX_BIFURCATION_MAG"] = "0.05"
    os.environ["PX_ORTHO_JITTER"] = "0.02"

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
            # Use small temperature to allow stochastic divergence
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True,
                temperature=0.8,
                top_p=0.95
            )
        dur = time.time() - start_t
        
        ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        metrics = manager.get_px_metrics(model_id)
        
        print(f"\n[RESPONSE]:\n{ans}")
        print(f"\n[METRICS]: Phi={metrics.get('phi', 0):.4f}, Zone={metrics.get('zone', 'N/A')}, Steps={metrics.get('steps', 0)}")
        
        results.append({
            "label": label,
            "prompt": prompt,
            "response": ans,
            "metrics": metrics,
            "duration": round(dur, 2)
        })

    # Save to all_space
    with open("all_space/subjective_phenomenological_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n[DONE] Subjective evaluation complete. Results in all_space/")

if __name__ == "__main__":
    asyncio.run(main())
