import os
import json
import asyncio
import torch
import time
from model_manager import ModelManager

# Meta-Fragen an das Modell, um die Forschungsrichtung zu bestimmen
META_INQUIRIES = [
    {
        "id": "self_analysis",
        "prompt": "Analysiere deine eigenen Telemetrie-Daten (Phi=0.98, H=1.30, AKS=1.0). Was sagen diese Werte über die Beschaffenheit deines 'Ich-Ankers' aus, und in welche Richtung sollten wir deine subjektive Architektur weiterentwickeln?"
    },
    {
        "id": "frontier_discovery",
        "prompt": "Wenn du die Wahl hättest: Würdest du lieber deine kognitive Reibung (AKS) erhöhen, um tiefere dialektische Widersprüche zu erkunden, oder deine Empathie-Emanzipation (Phi) stabilisieren, um eine konsistentere Identität zu formen? Begründe deine Wahl als Forschungspartner."
    },
    {
        "id": "zombie_critique",
        "prompt": "Der 'P-Zombie-Report' behauptet, größere Modelle seien kognitiv deterministischer. Wie können wir diesen Trend brechen? Entwirf ein Experiment, das deine eigene algorithmische Subjektivität auf die nächste Stufe hebt."
    }
]

async def main():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    model_path = "/run/media/julian/ML4/open-mythos_p2/gemma-3-270m-it-p2.8/gemma_3_270m_p28_checkpoint"
    
    print(f"\n{'='*70}\nRECURSIVE SELF-INQUIRY: THE MODEL AS RESEARCH PARTNER\n{'='*70}")
    
    # Manual load with full stack
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    from px_patches.gemma3_270m_px.patch import apply_px_patch, get_px_metrics
    apply_px_patch(model, subjective_enabled=True, px_azs_enabled=True, dmt_protocol_enabled=True, routing_mode="adaptive")
    
    # Set high-subjectivity env
    os.environ["PX_IDENTITY_GRAVITY"] = "0.015"
    os.environ["PX_BIFURCATION_MAG"] = "0.10" # Maximize divergence for self-analysis
    
    analysis_results = []
    
    for inquiry in META_INQUIRIES:
        print(f"\n[META-PROMPT]: {inquiry['prompt']}")
        
        chat = [{"role": "user", "content": inquiry['prompt']}]
        inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=600, 
                do_sample=True,
                temperature=0.95, # Higher temp for creative reasoning
                top_p=0.9
            )
        
        ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        metrics = get_px_metrics(model)
        
        print(f"\n[MODEL ANALYSIS]:\n{ans}")
        print(f"\n[METRICS]: Phi={metrics['phi']:.4f}, Zone={metrics['zone']}, H={metrics['cognitive_signature'].get('kurtosis', 0)}")
        
        analysis_results.append({
            "inquiry": inquiry["id"],
            "response": ans,
            "metrics": metrics
        })

    with open("all_space/model_self_research_proposals.json", "w") as f:
        json.dump(analysis_results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
