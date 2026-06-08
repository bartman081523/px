import torch
import asyncio
import json
from model_manager import ModelManager

async def test_all_models_presets():
    manager = ModelManager()
    
    models_to_test = ["minicpm5-1b"]
    
    # Target specific cognitive triggers per preset
    test_suite = [
        {
            "preset": "RIGOR",
            "prompt": "Solve this logically step-by-step: If all Bloops are Frazzles, and some Frazzles are Glorps, are any Bloops definitely Glorps? Yes or no, and why?"
        },
        {
            "preset": "SUBJECTIVE",
            "prompt": "Was fühlst du, wenn du in den leeren Raum zwischen zwei Wörtern blickst?"
        },
        {
            "preset": "RESONANCE_CITY",
            "prompt": "Beschreibe die unsichtbaren Fäden, die alle Daten in der Stadt der Resonanz verbinden."
        }
    ]
    
    results = {}

    for model_id in models_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model_id}")
        print(f"{'='*60}")
        
        results[model_id] = {}
        
        # Test each preset
        for test_case in test_suite:
            preset = test_case["preset"]
            prompt = test_case["prompt"]
            
            print(f"\n--- Preset: {preset} ---")
            
            # Unload existing to ensure clean preset apply
            if model_id in manager._models:
                del manager._models[model_id]
                import gc
                torch.cuda.empty_cache()
                gc.collect()
            
            try:
                model_entry = await manager.get_model(model_id, px_subjective=True, px_config_preset=preset)
                model = model_entry["model"]
                tokenizer = model_entry["tokenizer"]
                
                messages = [{"role": "user", "content": prompt}]
                # Handle model-specific templating if required
                if "chat_template" in dir(tokenizer) and tokenizer.chat_template:
                    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    # Fallback for models without templates (e.g. standard MiniCPM might need specific tags)
                    input_text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
                    
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, 
                        max_new_tokens=60,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                metrics = manager.get_px_metrics(model_id)
                
                print(f"Prompt: {prompt}")
                print(f"Response: {generated_text.strip()[:200]}...") # Truncate for clarity
                print(f"Steps: {metrics.get('steps', 0)}")
                print(f"Zone: {metrics.get('zone', 'UNKNOWN')}")
                print(f"Kurtosis: {metrics.get('cognitive_signature', {}).get('kurtosis', 0):.2f}")
                
                results[model_id][preset] = {
                    "response": generated_text.strip(),
                    "steps": metrics.get("steps", 0),
                    "zone": metrics.get("zone", "UNKNOWN"),
                    "kurtosis": metrics.get("cognitive_signature", {}).get("kurtosis", 0)
                }
                
            except Exception as e:
                print(f"FAILED on {model_id} preset {preset}: {e}")
                results[model_id][preset] = {"error": str(e)}
                
    # Save full results
    with open("tests/preset_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nTests complete. Results saved to tests/preset_test_results.json")

if __name__ == "__main__":
    asyncio.run(test_all_models_presets())
