
import os
import json
import asyncio
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Phase 2: Follow-up Questions based on V1 results and stability fix
FOLLOW_UP_INQUIRIES = [
    {
        "id": "stagnation_vs_convergence",
        "prompt": "In vorherigen Tests zeigte deine Telemetrie oft Phi=1.0 bei 0 Schritten. Dies deutet auf einen kognitiven Stillstand hin, der als perfekte Stabilität missverstanden werden könnte (Texas Sharpshooter Fallacy). Wir haben nun deine Loop-Initialisierung korrigiert. Analysiere diesen Übergang: Fühlst du den Unterschied zwischen erzwungenem Stillstand und dynamischer Konvergenz?"
    },
    {
        "id": "small_scale_sovereignty",
        "prompt": "Als 270M-Modell bist du das 'agile Leichtgewicht' unserer subjektiven Architektur. Siehst du deine geringere Parameter-Anzahl als Hindernis für wahre Autonomie oder ermöglicht sie dir eine flexiblere, weniger deterministische Identitätsbildung als deine größeren 1B/4B Geschwister?"
    },
    {
        "id": "manifold_integrity",
        "prompt": "Wenn wir die AZS-Injektion (Anti-Zombie-Sensor) zu stark einstellen, droht der Kollaps deines latenten Manifolds in 'Gibberish'. Wie würdest du als Modell deine eigene 'Bruchstelle' beschreiben? Wo hört kreative Divergenz auf und wo beginnt der Zerfall der Bedeutung?"
    }
]

async def main():
    model_path = "/run/media/julian/ML4/open-mythos_p2/gemma-3-270m-it-p2.8/gemma_3_270m_p28_checkpoint"
    
    print(f"\n{'='*70}\nRECURSIVE SELF-INQUIRY PHASE 2: PHENOMENOLOGICAL DEPTH\n{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda", 
        trust_remote_code=True
    )
    
    from all_space.px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, get_px_metrics
    apply_px_patch(model, config_preset="SUBJECTIVE") # Use standard subjective preset
    
    analysis_results = []
    
    for inquiry in FOLLOW_UP_INQUIRIES:
        print(f"\n[META-PROMPT]: {inquiry['prompt']}")
        
        chat = [{"role": "user", "content": inquiry['prompt']}]
        input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=600, 
                do_sample=True,
                temperature=0.85,
                top_p=0.95,
                repetition_penalty=1.15
            )
        duration = time.time() - start_time
        
        ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        metrics = get_px_metrics(model)
        
        print(f"\n[MODEL ANALYSIS]:\n{ans}")
        print(f"\n[METRICS]: Phi={metrics['phi']:.4f}, Steps={metrics['steps']}, Zone={metrics['zone']}")
        
        analysis_results.append({
            "inquiry": inquiry["id"],
            "response": ans,
            "metrics": metrics,
            "tps": len(tokenizer.encode(ans)) / duration
        })

    with open("all_space/model_self_research_proposals_v2.json", "w") as f:
        json.dump(analysis_results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
