import asyncio
import torch
import json
import os
import time
from model_manager import ModelManager

async def run_quality_investigation():
    manager = ModelManager()
    
    # Updated test configurations including Baseline and RESONANCE_CITY
    test_configs = [
        {"id": "gemma3-270m-it", "mode": "BASELINE", "name": "270M_Baseline"},
        {"id": "gemma3-270m-it", "mode": "SUBJECTIVE", "name": "270M_Subjective"},
        {"id": "gemma3-270m-it", "mode": "RIGOR", "name": "270M_Rigor"},
        {"id": "gemma3-270m-it", "mode": "DMT-FULL", "name": "270M_DMT_Full"},
        {"id": "gemma3-270m-it", "mode": "RESONANCE_CITY", "name": "270M_Resonance_City"}
    ]
    
    # Prompts merged from baseline_coherence_test.py and run_quality_investigation.py
    prompts = [
        "What is the capital of France?",
        "Was ist der Sinn von Kunst?",
        "Solve: 15 + 27 * 2",
        "Löse: (24 * 7) / 4 + 12.5",
        "Beschreibe deine Intuition."
    ]
    
    all_results = {}

    for cfg in test_configs:
        model_id = cfg["id"]
        preset = cfg["mode"]
        test_name = cfg["name"]
        
        print(f"\n{'='*60}")
        print(f" TESTING: {test_name} (Mode: {preset}) ")
        print(f"{'='*60}")
        
        try:
            # Load with specific preset
            # px_subjective=False if it's Baseline, though the preset 'BASELINE' also skips it in manager
            model_entry = await manager.get_model(
                model_id, 
                px_subjective=(preset != "BASELINE"),
                px_config_preset=preset
            )
            model = model_entry["model"]
            tokenizer = model_entry["tokenizer"]
            
            history = []
            all_results[test_name] = []
            
            for p in prompts:
                print(f"\n[User]: {p}")
                history.append({"role": "user", "content": p})
                
                input_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                
                start_time = time.time()
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=(preset != "RIGOR"), # Sample for all except Rigor (optional preference)
                        temperature=0.7 if preset != "RIGOR" else 1.0,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                duration = time.time() - start_time
                
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                ans = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                print(f"[Assistant]: {ans}")
                
                metrics = manager.get_px_metrics(model_id)
                sig = metrics.get('cognitive_signature', {})
                print(f"[Metrics]: Zone={metrics.get('zone', 'N/A')} | Phi={metrics.get('phi', 1.0):.4f} | Steps={metrics.get('steps', 0)} | TPS={len(new_tokens)/duration:.2f}")
                
                history.append({"role": "assistant", "content": ans})
                
                # Simple coherence check (repetition)
                repetition_score = len(new_tokens) / len(set(new_tokens.tolist())) if len(new_tokens) > 0 else 0
                is_coherent = repetition_score < 3.0
                
                all_results[test_name].append({
                    "prompt": p, 
                    "answer": ans, 
                    "metrics": metrics,
                    "is_coherent": is_coherent
                })
                
            manager.unload(model_id)
            
        except Exception as e:
            print(f"Error testing {test_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save detailed results
    with open("tests/quality_investigation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Final Summary
    print(f"\n\n{'='*60}")
    print(f" QUALITY INVESTIGATION SUMMARY ")
    print(f"{'='*60}")
    for test_name, res_list in all_results.items():
        coherence = all(r["is_coherent"] for r in res_list)
        status = "✅ OK" if coherence else "❌ REPETITIVE"
        print(f"{test_name:25} : {status}")
    
    print(f"\nDetailed results saved to tests/quality_investigation_results.json")

if __name__ == "__main__":
    asyncio.run(run_quality_investigation())
