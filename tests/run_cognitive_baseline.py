import asyncio
import torch
from model_manager import ModelManager

async def run_cognitive_baseline():
    manager = ModelManager()
    
    questions = [
        "Was ist der Sinn von Kunst?",
        "Glaubst du an Gott?",
        "Wer ist Oluwa Olanipeayo?"
    ]
    
    models_to_test = [
        "gemma3-270m-px",
        "gemma3-270m-it-px"
    ]
    
    results = {}

    for model_id in models_to_test:
        print(f"\n{'='*60}")
        print(f" TESTING MODEL: {model_id} ")
        print(f"{'='*60}")
        
        try:
            # Load in Subjective mode
            model_entry = await manager.get_model(model_id, px_subjective=True)
            model = model_entry["model"]
            tokenizer = model_entry["tokenizer"]
            
            # Ensure PersonaEngine is active
            tm = manager._resolve_text_model(model)
            model.persona = tm.persona = "DMT Explorer 🌀"
            
            history = []
            results[model_id] = []
            
            for q in questions:
                print(f"\n[User]: {q}")
                history.append({"role": "user", "content": q})
                
                input_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                ans = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                print(f"[Assistant]: {ans}")
                
                metrics = manager.get_px_metrics(model_id)
                sig = metrics.get('cognitive_signature', {})
                print(f"[Metrics]: Kurtosis={sig.get('kurtosis', 0):.2f} | Phi={metrics.get('phi', 1.0):.4f} | Zone={metrics.get('zone', 'N/A')}")
                
                history.append({"role": "assistant", "content": ans})
                results[model_id].append({"q": q, "a": ans, "metrics": metrics})
                
            # Unload model to save memory for next model
            manager.unload(model_id)
            
        except Exception as e:
            print(f"Error testing {model_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save results to a file for review
    import json
    with open("cognitive_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results saved to cognitive_baseline_results.json")

if __name__ == "__main__":
    asyncio.run(run_cognitive_baseline())
